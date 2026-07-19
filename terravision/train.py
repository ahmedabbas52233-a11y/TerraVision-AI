"""
TerraVision AI · terravision/train.py
Training pipeline for both V1 (seq_len=1) and V2 (6-month time-series) models.

Rewritten to match core.py v3.0.0: the model now predicts a RAW output that
compute_yield() converts to t/ha via CROP_PARAMS[crop]["ndvi_scale"] plus NDVI-based
modifiers — there is no fixed YIELD_MAX. Training therefore optimizes the raw output
against a synthetic target consistent with that scaling, so a trained checkpoint is
directly usable by api.py's inference path without further conversion.

Usage
─────
  python -m terravision.train                       # train V2 (default)
  python -m terravision.train --model v1             # train V1 (legacy)
  python -m terravision.train --eval-only            # evaluate existing checkpoint
  python -m terravision.train --model v2 --epochs 300 --n 8000 --lr 5e-4

Author  : Ahmad Abbas Hussain
Contact : ahmedabbas52233@gmail.com
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from terravision.core import (
    CROP_PARAMS,
    TerraVisionTransformer,
    TerraVisionTransformerV2,
    mc_dropout_confidence,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("terravision.train")

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"


# ─────────────────────────────────────────────────────────────────────────────
# DATASET GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
#
# Ground truth is synthetic (see README "Known Issues"): NDVI/temperature/moisture
# are sampled per-crop from CROP_PARAMS priors, and the RAW target the model learns
# to predict is defined so that compute_yield(raw, ndvi, crop) reproduces a
# plausible yield curve. This keeps the trained checkpoint consistent with the
# actual inference path in core.py, unlike the previous version of this file.


def _raw_target_from_yield(target_yield: float, ndvi: float, crop: str) -> float:
    """
    Invert compute_yield()'s NDVI modifier so we can generate a RAW training
    target from a desired yield curve. This keeps train.py's target consistent
    with exactly what compute_yield() will do to the model's raw output at
    inference time, rather than training against an unrelated scale.
    """
    scale = CROP_PARAMS[crop]["ndvi_scale"]

    if ndvi < 0.05:
        modifier = 0.15
    elif ndvi < 0.10:
        modifier = 0.15 + (ndvi - 0.05) / 0.05 * 0.25
    elif ndvi < 0.20:
        modifier = 0.40 + (ndvi - 0.10) / 0.10 * 0.35
    elif ndvi < 0.30:
        modifier = 0.75 + (ndvi - 0.20) / 0.10 * 0.15
    elif ndvi < 0.60:
        modifier = 0.90 + (ndvi - 0.30) / 0.30 * 0.10
    else:
        modifier = 1.00 + (ndvi - 0.60) / 0.40 * 0.15

    denom = max(scale * modifier, 1e-6)
    return target_yield / denom


def generate_v1_dataset(
    n_samples: int = 4_000, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-observation dataset for V1 model: (N, 3) → (N, 1) raw target."""
    rng = np.random.default_rng(seed)
    n_per = n_samples // len(CROP_PARAMS)
    Xs, raws = [], []

    for crop, cfg in CROP_PARAMS.items():
        ndvi = rng.normal(cfg["ndvi_prior"], 0.15, n_per).clip(-0.1, 1.0)
        temp_K = rng.normal(cfg["temp_K"], 4.0, n_per).clip(263.0, 318.0)
        moisture = rng.normal(cfg["moisture"], 0.012, n_per).clip(0.01, 0.12)

        thermal = np.exp(-((temp_K - cfg["temp_K"]) ** 2) / (2 * 8.0**2))
        base_yield = (
            cfg["ndvi_scale"] * np.clip(ndvi, 0, 1) * 0.6 * thermal
            + rng.normal(0, 0.15, n_per)
        ).clip(0.1, None)

        raw = np.array(
            [_raw_target_from_yield(y, n, crop) for y, n in zip(base_yield, ndvi, strict=True)],
            dtype=np.float32,
        )

        Xs.append(np.column_stack([ndvi, temp_K, moisture]))
        raws.append(raw)

    X = np.vstack(Xs).astype(np.float32)
    y = np.concatenate(raws).astype(np.float32)
    idx = rng.permutation(len(X))
    return torch.from_numpy(X[idx]), torch.from_numpy(y[idx].reshape(-1, 1))


def generate_v2_dataset(
    n_samples: int = 4_000,
    seq_len: int = 6,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    6-month time-series dataset for V2 model: (N, seq_len, 3) → (N, 1) raw target.
    Raw target is derived from the sequence's peak NDVI, using the same inversion
    as the V1 generator so both checkpoints train consistently with core.py.
    """
    rng = np.random.default_rng(seed)
    n_per = n_samples // len(CROP_PARAMS)
    Xs, raws = [], []

    for crop, cfg in CROP_PARAMS.items():
        n = n_per
        peak_ndvi = rng.normal(cfg["ndvi_prior"], 0.15, n).clip(-0.1, 1.0)

        sequences = []
        for j in range(n):
            t = np.linspace(0, np.pi, seq_len)
            envelope = np.sin(t) * peak_ndvi[j]
            noise = rng.normal(0, 0.03, seq_len)
            ndvi_seq = np.clip(envelope + noise, -0.1, 1.0)

            temp_K = rng.normal(cfg["temp_K"], 4.0, seq_len).clip(263, 318)
            moisture = rng.normal(cfg["moisture"], 0.012, seq_len).clip(0.01, 0.12)
            sequences.append(np.stack([ndvi_seq, temp_K, moisture], axis=1))

        X_crop = np.stack(sequences, axis=0).astype(np.float32)

        base_yield = (
            cfg["ndvi_scale"] * np.clip(peak_ndvi, 0, 1) * 0.6
            + rng.normal(0, 0.15, n)
        ).clip(0.1, None)

        raw_crop = np.array(
            [
                _raw_target_from_yield(y, n_val, crop)
                for y, n_val in zip(base_yield, peak_ndvi, strict=True)
            ],
            dtype=np.float32,
        )

        Xs.append(X_crop)
        raws.append(raw_crop)

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(raws)
    idx = rng.permutation(len(X))
    return torch.from_numpy(X[idx]), torch.from_numpy(y[idx].reshape(-1, 1))


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


# ─────────────────────────────────────────────────────────────────────────────
# GENERIC TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    ckpt_path: Path,
    epochs: int = 200,
    lr: float = 1e-3,
    batch: int = 128,
    patience: int = 20,
    seed: int = 42,
    label: str = "model",
) -> dict:
    torch.manual_seed(seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    n_total = len(X)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    ds = TensorDataset(X, y)
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed)
    )

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_ep = 0
    pat_cnt = 0
    t0 = time.time()

    log.info(
        "Training %s for up to %d epochs | train=%d val=%d test=%d",
        label, epochs, n_train, n_val, n_test,
    )

    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_losses.append(loss.item())
        sch.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_dl:
                val_losses.append(loss_fn(model(xb), yb).item())

        t_mse = float(np.mean(train_losses))
        v_mse = float(np.mean(val_losses))

        if ep % 25 == 0 or ep == 1:
            log.info(
                "  Epoch %3d  train=%.4f  val=%.4f  lr=%.2e",
                ep, t_mse, v_mse, opt.param_groups[0]["lr"],
            )

        if v_mse < best_val - 1e-5:
            best_val = v_mse
            best_ep = ep
            pat_cnt = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                log.info(
                    "  Early stop at epoch %d (best val=%.4f at ep %d)",
                    ep, best_val, best_ep,
                )
                break

    elapsed = time.time() - t0
    log.info(
        "Training done in %.1f s — best val_MSE=%.4f at epoch %d",
        elapsed, best_val, best_ep,
    )

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            preds.append(model(xb).squeeze(1).numpy())
            trues.append(yb.squeeze(1).numpy())
    metrics = compute_metrics(np.concatenate(trues), np.concatenate(preds))
    log.info(
        "Test (raw output) → MSE=%.4f  RMSE=%.4f  MAE=%.4f  R²=%.4f",
        metrics["mse"], metrics["rmse"], metrics["mae"], metrics["r2"],
    )

    X_test = torch.stack([X[i] for i in test_ds.indices])
    conf = mc_dropout_confidence(model, X_test[:1], n_passes=30)
    log.info(
        "MC Dropout (30 passes) → std=±%.3f  95%% CI ±%.3f  confidence=%.1f%%",
        conf["std_yield"],
        (conf["ci_95_upper"] - conf["ci_95_lower"]) / 2,
        conf["confidence_pct"],
    )

    stats = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "model_version": label,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "best_epoch": best_ep,
        "best_val_mse": round(best_val, 6),
        "test_metrics_raw_output": {k: round(v, 4) for k, v in metrics.items()},
        "mc_dropout": {
            "n_passes": 30,
            "std_yield": conf["std_yield"],
            "confidence_pct": conf["confidence_pct"],
            "ci_95_lower": conf["ci_95_lower"],
            "ci_95_upper": conf["ci_95_upper"],
        },
        "hyperparameters": {"lr": lr, "batch": batch, "patience": patience, "seed": seed},
        "note": (
            "Trained on synthetic data (see README Known Issues). Metrics are on the "
            "model's RAW output, before compute_yield()'s NDVI-based scaling to t/ha — "
            "not a real-world yield accuracy figure."
        ),
    }
    stats_path = MODEL_DIR / "training_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    log.info("Checkpoint → %s", ckpt_path)
    log.info("Stats      → %s", stats_path)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINTS
# ─────────────────────────────────────────────────────────────────────────────
def train(
    model_version: str = "v2",
    n_samples: int = 4_000,
    epochs: int = 200,
    lr: float = 1e-3,
    batch: int = 128,
    patience: int = 20,
    seed: int = 42,
) -> dict:
    if model_version == "v2":
        log.info("Generating V2 time-series dataset (6-month phenological sequences)…")
        X, y = generate_v2_dataset(n_samples, seed=seed)
        model = TerraVisionTransformerV2()
        ckpt = MODEL_DIR / "terravision_v2.pth"
        label = "TerraVisionTransformerV2"
    else:
        log.info("Generating V1 single-observation dataset…")
        X, y = generate_v1_dataset(n_samples, seed=seed)
        model = TerraVisionTransformer()
        ckpt = MODEL_DIR / "terravision_v1.pth"
        label = "TerraVisionTransformer"

    return train_model(model, X, y, ckpt, epochs, lr, batch, patience, seed, label)


def eval_only(model_version: str = "v2") -> None:
    from terravision.core import load_model

    m = load_model()
    if m is None:
        log.error("No checkpoint found.")
        return
    log.info("Loaded %s", type(m).__name__)
    n = 2_000
    if model_version == "v2":
        X, y = generate_v2_dataset(n, seed=99)
    else:
        X, y = generate_v1_dataset(n, seed=99)
    m.eval()
    with torch.no_grad():
        pred = m(X).squeeze(1).numpy()
    metrics = compute_metrics(y.numpy().ravel(), pred)
    log.info(
        "Eval (raw output) → MSE=%.4f RMSE=%.4f MAE=%.4f R²=%.4f",
        metrics["mse"], metrics["rmse"], metrics["mae"], metrics["r2"],
    )
    conf = mc_dropout_confidence(m, X[:64], n_passes=30)
    log.info(
        "MC Dropout → std=±%.3f  confidence=%.1f %%",
        conf["std_yield"], conf["confidence_pct"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train TerraVision AI Transformer")
    ap.add_argument("--model", default="v2", choices=["v1", "v2"])
    ap.add_argument("--n", type=int, default=4_000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    if args.eval_only:
        eval_only(args.model)
    else:
        train(
            args.model, args.n, args.epochs, args.lr, args.batch, args.patience, args.seed
        )


if __name__ == "__main__":
    main()
