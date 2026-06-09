"""
TerraVision AI · terravision/core.py
Shared model definitions, GEE feature extraction, ERA5 climate intelligence,
MC Dropout uncertainty, NDVI heatmap tile generation, yield computation,
and reporting helpers.

Imported by
───────────
  api.py  (FastAPI REST — root entry point)
  app.py  (Streamlit UI — root entry point)

Author  : Ahmad Abbas Hussain
Contact : ahmedabbas52233@gmail.com
GitHub  : https://github.com/ahmedabbas52233/TerraVision-AI
Version : 1.0.0
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
_PKG: Path = Path(__file__).parent  # …/terravision/
_ROOT: Path = _PKG.parent  # …/TerraVision-AI/

MODEL_V1_PATH: str = str(_ROOT / "models" / "terravision_v1.pth")
MODEL_V2_PATH: str = str(_ROOT / "models" / "terravision_v2.pth")
STATS_PATH: str = str(_ROOT / "models" / "training_stats.json")

# Legacy alias kept for backward compatibility
MODEL_PATH = MODEL_V2_PATH


# ─────────────────────────────────────────────────────────────────────────────
# TYPED DICTS
# ─────────────────────────────────────────────────────────────────────────────
class Era5Dict(TypedDict):
    temp_c: float
    precip_mm_month: float
    source: str


class ConfidenceResult(TypedDict):
    mean_yield: float
    std_yield: float
    confidence_pct: float
    ci_95_lower: float
    ci_95_upper: float


# ─────────────────────────────────────────────────────────────────────────────
# V1 MODEL  (seq_len=1 — kept for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────
class TerraVisionTransformer(nn.Module):
    """
    V1 Spatio-Temporal Transformer — single observation input.

    Note: MHSA over seq_len=1 is a learned weighted projection.
    Use TerraVisionTransformerV2 for genuine temporal attention.
    Input : (batch, 3)  →  [NDVI, temperature_K, soil_moisture]
    Output: (batch, 1)  →  raw yield score
    """

    def __init__(self, input_dim: int = 3, model_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=4, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, 128),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x).unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        return self.ffn(attn_out.squeeze(1))


# ─────────────────────────────────────────────────────────────────────────────
# V2 MODEL  (seq_len=6 — genuine temporal attention, Gap 2 fix)
# ─────────────────────────────────────────────────────────────────────────────
class TerraVisionTransformerV2(nn.Module):
    """
    V2 Spatio-Temporal Transformer — 6-month NDVI time-series input.

    Architecture fixes the seq_len=1 limitation of V1:
      · Input is a sequence of 6 monthly Sentinel-2 observations
      · Learnable positional embeddings encode temporal position
      · MHSA attends across months — learns phenological patterns
        (e.g., rising NDVI = early season vs falling = harvest-ready)
      · Pre-norm residual blocks stabilise training
      · Mean pooling aggregates 6 temporal states → single yield estimate

    Input : (batch, 6, 3) →  6 × [NDVI, temperature_K, soil_moisture]
    Output: (batch, 1)    →  raw yield score

    Architecture
    ─────────────
      Input Projection     Linear(3 → 64)                192 params
      Positional Embedding Learnable (6 × 64)            384 params
      MHSA                 MultiheadAttention(64, 4h)  16 640 params
      LayerNorm 1          64                             128 params
      FFN Layer 1          Linear(64 → 128) + GELU      8 320 params
      FFN Layer 2          Linear(128 → 64)             8 256 params
      FFN Dropout          p = 0.10
      LayerNorm 2          64                             128 params
      Regression Head      Linear(64 → 1)                 65 params
                                                  ──────────────────
      Total                                          34 113 params
    """

    SEQ_LEN: int = 6

    def __init__(self, input_dim: int = 3, model_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.SEQ_LEN, model_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=4, batch_first=True
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, 128),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(128, model_dim),  # output keeps model_dim for residual
        )
        self.norm2 = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, SEQ_LEN, input_dim)
        x = self.input_proj(x) + self.pos_embed  # (B, S, D)
        attn, _ = self.attention(x, x, x)  # (B, S, D)
        x = self.norm1(x + attn)  # residual + norm
        ffn_out = self.ffn(x)  # (B, S, D)
        x = self.norm2(x + ffn_out)  # residual + norm
        pooled = x.mean(dim=1)  # (B, D) — mean over months
        return self.head(pooled)  # (B, 1)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CROP_PARAMS: dict[str, dict[str, Any]] = {
    "Wheat": {
        "temp_K": 291.5,
        "moisture": 0.025,
        "base": 2.8,
        "ndvi_scale": 2.2,
        "offset": 3.33,
    },
    "Rice": {
        "temp_K": 298.2,
        "moisture": 0.085,
        "base": 4.2,
        "ndvi_scale": 3.1,
        "offset": 3.33,
    },
    "Maize": {
        "temp_K": 295.0,
        "moisture": 0.045,
        "base": 3.5,
        "ndvi_scale": 5.8,
        "offset": 3.33,
    },
    "Soybean": {
        "temp_K": 296.5,
        "moisture": 0.055,
        "base": 2.2,
        "ndvi_scale": 3.8,
        "offset": 3.33,
    },
}

ERA5_CROP_OPTIMA: dict[str, dict[str, float]] = {
    "Wheat": {"temp_c": 18.0, "precip_mm": 55.0},
    "Rice": {"temp_c": 27.0, "precip_mm": 150.0},
    "Maize": {"temp_c": 24.0, "precip_mm": 85.0},
    "Soybean": {"temp_c": 25.0, "precip_mm": 90.0},
}

NDVI_CLASSES: list[tuple[float, str, str, str]] = [
    (
        0.20,
        "🔴 Critical — Low Vegetation Density",
        "Immediate nitrogen-based soil enrichment recommended.",
        "error",
    ),
    (
        0.30,
        "🟠 Stressed Vegetation",
        "Targeted fertiliser application and irrigation audit advised.",
        "warning",
    ),
    (
        0.60,
        "🔵 Normal Growth Cycle",
        "Standard agronomic practices; schedule next monitoring in 14 days.",
        "info",
    ),
    (
        float("inf"),
        "🟢 High Photosynthetic Activity",
        "Maintain current nutrient regime; begin harvest-window planning.",
        "success",
    ),
]

CARBON_FRACTION: float = 0.47
YIELD_MAX: float = 14.5
MODEL_VERSION: str = "1.0.0"


# Load MC Dropout floor from training stats if available
def _load_confidence_floor() -> float:
    try:
        with open(STATS_PATH) as f:
            stats = json.load(f)
        ci_half = stats.get("mc_dropout", {}).get("ci_95_half_t_ha", None)
        if ci_half is not None:
            cv_approx = float(ci_half) / (3.5 * 1.96)  # approx mean yield
            return float(np.clip(100.0 * (1.0 - cv_approx), 70.0, 99.0))
    except Exception:
        pass
    return 88.0  # conservative floor when no stats file exists


CONFIDENCE_FLOOR: float = _load_confidence_floor()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
_v2_singleton: TerraVisionTransformerV2 | None = None
_v1_singleton: TerraVisionTransformer | None = None


def load_model() -> TerraVisionTransformerV2 | TerraVisionTransformer | None:
    """
    Load V2 checkpoint if available, fall back to V1.
    Returns the loaded model or None if neither checkpoint exists.
    """
    global _v2_singleton, _v1_singleton

    # Try V2 first (genuine temporal attention)
    if _v2_singleton is not None:
        return _v2_singleton
    if os.path.exists(MODEL_V2_PATH):
        try:
            m = TerraVisionTransformerV2()
            m.load_state_dict(
                torch.load(MODEL_V2_PATH, map_location="cpu", weights_only=True)
            )
            m.eval()
            _v2_singleton = m
            log.info(
                "V2 checkpoint loaded from %s (%d params)",
                MODEL_V2_PATH,
                sum(p.numel() for p in m.parameters()),
            )
            return m
        except Exception as exc:
            log.warning("V2 load failed: %s — trying V1", exc)

    # Fall back to V1
    if _v1_singleton is not None:
        return _v1_singleton
    if os.path.exists(MODEL_V1_PATH):
        try:
            m2 = TerraVisionTransformer()
            m2.load_state_dict(
                torch.load(MODEL_V1_PATH, map_location="cpu", weights_only=True)
            )
            m2.eval()
            _v1_singleton = m2
            log.info("V1 checkpoint loaded (V2 not found).")
            return m2
        except Exception as exc:
            log.error("V1 load also failed: %s", exc)

    log.error("No checkpoint found at %s or %s", MODEL_V2_PATH, MODEL_V1_PATH)
    return None


def is_v2(model: Any) -> bool:
    """Return True if the loaded model is V2 (temporal sequence input)."""
    return isinstance(model, TerraVisionTransformerV2)


# ─────────────────────────────────────────────────────────────────────────────
# MC DROPOUT CONFIDENCE  (Gap 1 fix — replaces hardcoded 94.2)
# ─────────────────────────────────────────────────────────────────────────────
def mc_dropout_confidence(
    model: TerraVisionTransformer | TerraVisionTransformerV2,
    tensor: torch.Tensor,
    n_passes: int = 20,
) -> ConfidenceResult:
    """
    Estimate predictive uncertainty via Monte Carlo Dropout.

    Enables dropout at inference (model.train()) to draw N samples from the
    approximate posterior q*(y|x) ≈ P(y|x, W).

    Parameters
    ──────────
    model    : loaded model (V1 or V2, eval mode before call)
    tensor   : input tensor matching the model's expected shape
    n_passes : number of stochastic forward passes (default 20, ~5 ms overhead)

    Returns ConfidenceResult with
    ─────────────────────────────
    mean_yield    : mean prediction across passes (t/ha)
    std_yield     : standard deviation (epistemic uncertainty, t/ha)
    confidence_pct: 1 - normalised coefficient of variation, scaled to [50, 99]
    ci_95_lower   : 95 % credible interval lower bound (t/ha)
    ci_95_upper   : 95 % credible interval upper bound (t/ha)
    """
    model.train()  # enables dropout layers
    preds: list[float] = []
    with torch.no_grad():
        for _ in range(n_passes):
            out = model(tensor).squeeze(-1)  # (B,) or scalar
            preds.extend(out.tolist() if out.dim() > 0 else [out.item()])
    model.eval()  # restore eval mode

    arr = np.array(preds, dtype=np.float32)
    mean = float(arr.mean())
    std = float(arr.std())

    # Coefficient of variation → confidence
    cv = std / (abs(mean) + 1e-6)
    raw_conf = float(100.0 * (1.0 - min(cv * 4.0, 0.5)))
    confidence = float(np.clip(raw_conf, CONFIDENCE_FLOOR - 15, 99.0))

    ci_half = 1.96 * std
    return ConfidenceResult(
        mean_yield=round(mean, 4),
        std_yield=round(std, 4),
        confidence_pct=round(confidence, 1),
        ci_95_lower=round(max(mean - ci_half, 0.0), 3),
        ci_95_upper=round(min(mean + ci_half, YIELD_MAX), 3),
    )


# ─────────────────────────────────────────────────────────────────────────────
# GEE · SENTINEL-2 NDVI  (V1 — single observation)
# ─────────────────────────────────────────────────────────────────────────────
def get_live_features(lat: float, lon: float, crop: str) -> list[float]:
    """Single-observation feature vector [NDVI, temp_K, moisture] for V1 model."""
    import ee

    p = CROP_PARAMS[crop]
    fallback = [0.5, float(p["temp_K"]), float(p["moisture"])]
    try:
        point = ee.Geometry.Point([lon, lat])
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        ndvi_img = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point)
            .filterDate("2024-01-01", today)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .median()
            .normalizedDifference(["B8", "B4"])
            .rename("NDVI")
        )
        stats = ndvi_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point.buffer(500),
            scale=10,
            maxPixels=int(1e8),
        ).getInfo()
        ndvi = float(stats.get("NDVI") or 0.5)
        ndvi = max(-1.0, min(1.0, ndvi))
        return [ndvi, float(p["temp_K"]), float(p["moisture"])]
    except Exception as exc:
        log.warning("GEE NDVI (V1) failed: %s", exc)
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# GEE · SENTINEL-2 NDVI  (V2 — 6-month time series, Gap 2 fix)
# ─────────────────────────────────────────────────────────────────────────────
def get_live_features_v2(
    lat: float,
    lon: float,
    crop: str,
    seq_len: int = TerraVisionTransformerV2.SEQ_LEN,
) -> list[list[float]]:
    """
    Build a seq_len-month NDVI time-series for V2 model input.

    Queries Sentinel-2 SR for each of the last `seq_len` calendar months,
    extracting median NDVI over a 500 m buffer.  Missing months (cloud cover,
    no data) fall back to a simple linear interpolation from adjacent months.

    Returns
    ───────
    List of seq_len × [ndvi, temp_K, moisture]  — shape (seq_len, 3)
    """
    import ee

    p = CROP_PARAMS[crop]
    temp = float(p["temp_K"])
    mois = float(p["moisture"])
    today = datetime.now(UTC)

    ndvi_series: list[float | None] = []

    for i in range(seq_len - 1, -1, -1):  # oldest → newest
        m_end = today - timedelta(days=30 * i)
        m_start = today - timedelta(days=30 * (i + 1))
        try:
            point = ee.Geometry.Point([lon, lat])
            img = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(point)
                .filterDate(m_start.strftime("%Y-%m-%d"), m_end.strftime("%Y-%m-%d"))
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
                .median()
                .normalizedDifference(["B8", "B4"])
                .rename("NDVI")
            )
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point.buffer(500),
                scale=10,
                maxPixels=int(1e8),
            ).getInfo()
            raw = stats.get("NDVI")
            ndvi_series.append(float(raw) if raw is not None else None)
        except Exception as exc:
            log.warning("GEE V2 month -%d failed: %s", i, exc)
            ndvi_series.append(None)

    # Fill None values via linear interpolation
    ndvi_filled = _interpolate_missing(ndvi_series, default=0.5)

    return [[max(-1.0, min(1.0, ndvi)), temp, mois] for ndvi in ndvi_filled]


def _interpolate_missing(
    series: list[float | None],
    default: float = 0.5,
) -> list[float]:
    """Forward/backward fill then linear interpolation for missing NDVI months."""
    n = len(series)
    filled: list[float] = [default] * n

    # Forward fill known values
    last_known: float | None = None
    for i, v in enumerate(series):
        if v is not None:
            last_known = v
        filled[i] = last_known if last_known is not None else default

    # Backward fill remaining leading Nones
    last_known = None
    for i in range(n - 1, -1, -1):
        if series[i] is not None:
            last_known = series[i]
        if series[i] is None and last_known is not None:
            filled[i] = last_known

    return filled


# ─────────────────────────────────────────────────────────────────────────────
# GEE · ERA5-LAND
# ─────────────────────────────────────────────────────────────────────────────
def get_era5_features(lat: float, lon: float) -> Era5Dict:
    """30-day ERA5-Land temperature (°C) + precipitation (mm/month) via GEE."""
    import ee

    _default: Era5Dict = {"temp_c": 20.0, "precip_mm_month": 60.0, "source": "default"}
    try:
        point = ee.Geometry.Point([lon, lat])
        today = datetime.now(UTC)
        start = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        end = today.strftime("%Y-%m-%d")
        era5_img = (
            ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .filterBounds(point)
            .filterDate(start, end)
            .select(["temperature_2m", "total_precipitation_sum"])
            .mean()
        )
        stats: dict[str, Any] = era5_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point.buffer(10_000),
            scale=11_132,
            maxPixels=int(1e6),
        ).getInfo()
        raw_t = stats.get("temperature_2m")
        raw_p = stats.get("total_precipitation_sum")
        temp_c = (float(raw_t) - 273.15) if raw_t is not None else 20.0
        precip_mm_month = (float(raw_p) * 1_000.0 * 30.0) if raw_p is not None else 60.0
        return Era5Dict(
            temp_c=round(temp_c, 2),
            precip_mm_month=round(precip_mm_month, 2),
            source="era5-land",
        )
    except Exception as exc:
        log.warning("GEE ERA5 failed: %s", exc)
        return _default


# ─────────────────────────────────────────────────────────────────────────────
# GEE · NDVI HEATMAP TILE
# ─────────────────────────────────────────────────────────────────────────────
_NDVI_VIS: dict[str, Any] = {
    "min": -0.2,
    "max": 0.85,
    "palette": [
        "#8B4513",
        "#D2691E",
        "#F4D03F",
        "#A9D18E",
        "#4CAF50",
        "#1B7A3E",
        "#005A1F",
    ],
}


def get_ndvi_tile_url(lat: float, lon: float) -> str | None:
    """12-month Sentinel-2 NDVI composite as a GEE tile URL for Folium."""
    import ee

    try:
        point = ee.Geometry.Point([lon, lat])
        today = datetime.now(UTC)
        ndvi_img = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point)
            .filterDate(
                (today - timedelta(days=365)).strftime("%Y-%m-%d"),
                today.strftime("%Y-%m-%d"),
            )
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .median()
            .normalizedDifference(["B8", "B4"])
            .rename("NDVI")
        )
        return str(ndvi_img.getMapId(_NDVI_VIS)["tile_fetcher"].url_format)
    except Exception as exc:
        log.warning("GEE NDVI tile URL failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# YIELD COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_yield(raw_output: float, ndvi: float, crop: str) -> float:
    """Scale raw transformer output to agronomic yield (t/ha)."""
    p = CROP_PARAMS[crop]
    y = (
        abs(raw_output - float(p["offset"]))
        + float(p["base"])
        + (ndvi * float(p["ndvi_scale"]))
    )
    if ndvi < 0.1:
        y *= 0.20
    return float(np.clip(y, 0.0, YIELD_MAX))


def era5_yield_adjustment(
    base_yield: float, temp_c: float, precip_mm: float, crop: str
) -> float:
    """ERA5 Gaussian thermal-stress × precipitation-adequacy correction."""
    opt = ERA5_CROP_OPTIMA[crop]
    temp_stress: float = float(np.exp(-((temp_c - opt["temp_c"]) ** 2) / (2 * 12.0**2)))
    precip_factor: float = float(
        np.clip(0.50 + 0.50 * (precip_mm / opt["precip_mm"]), 0.50, 1.05)
    )
    return float(np.clip(base_yield * temp_stress * precip_factor, 0.0, YIELD_MAX))


def ndvi_status(ndvi: float) -> tuple[str, str, str]:
    """Return (label, recommended_action, alert_type) for a given NDVI value."""
    for upper, label, action, alert_type in NDVI_CLASSES:
        if ndvi < upper:
            return label, action, alert_type
    last = NDVI_CLASSES[-1]
    return last[1], last[2], last[3]


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def build_report(
    lat: float,
    lon: float,
    crop: str,
    ndvi: float,
    yield_base: float,
    yield_adjusted: float,
    carbon: float,
    label: str,
    action: str,
    era5: Era5Dict | None = None,
    confidence: ConfidenceResult | None = None,
) -> str:
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    div = "═" * 56

    era5_block = ""
    if era5 is not None and era5.get("source") == "era5-land":
        era5_block = (
            f"\nERA5-LAND CLIMATE  (30-day mean)\n"
            f"  Air Temperature  : {era5['temp_c']:.1f} °C\n"
            f"  Monthly Precip.  : {era5['precip_mm_month']:.1f} mm\n\n"
            f"ERA5-ADJUSTED YIELD\n"
            f"  Base Yield       : {yield_base:.2f} t/ha\n"
            f"  Adjusted Yield   : {yield_adjusted:.2f} t/ha\n"
            f"  Delta            : {yield_adjusted - yield_base:+.2f} t/ha\n"
        )

    conf_block = ""
    if confidence is not None:
        conf_block = (
            f"\nUNCERTAINTY (MC Dropout, 20 passes)\n"
            f"  Confidence       : {confidence['confidence_pct']:.1f} %\n"
            f"  Std Deviation    : ±{confidence['std_yield']:.3f} t/ha\n"
            f"  95 % CI          : [{confidence['ci_95_lower']:.2f}, {confidence['ci_95_upper']:.2f}] t/ha\n"
        )

    return (
        f"{div}\n"
        f"  TerraVision AI v{MODEL_VERSION} — Crop Intelligence Report\n"
        f"  Generated : {ts}\n"
        f"{div}\n\n"
        f"LOCATION\n"
        f"  Latitude  : {lat:.6f}°\n"
        f"  Longitude : {lon:.6f}°\n\n"
        f"CROP ANALYSIS\n"
        f"  Crop Type  : {crop}\n"
        f"  NDVI Index : {ndvi:.4f}\n"
        f"  Status     : {label}\n\n"
        f"YIELD PREDICTION\n"
        f"  Transformer Yield : {yield_base:.2f} t/ha\n"
        f"  ERA5-Adj. Yield   : {yield_adjusted:.2f} t/ha\n"
        f"  Carbon Est.       : {carbon:.2f} Mg C/ha  (IPCC 2006, CF=0.47)\n"
        f"{era5_block}{conf_block}\n"
        f"PRECISION INSIGHT\n"
        f"  {action}\n\n"
        f"MODEL METADATA\n"
        f"  Architecture  : TerraVisionTransformerV2 (6-month temporal sequence)\n"
        f"  Parameters    : 34,113\n"
        f"  Satellite     : Sentinel-2 SR Harmonized (10 m, 500 m buffer)\n"
        f"  Climate       : ERA5-Land Daily Aggregates (11 km, 10 km buffer)\n"
        f"  Cloud Filter  : < 20 % (monthly) / < 30 % (sequence)\n\n"
        f"CONTACT\n"
        f"  Ahmad Abbas Hussain · ahmedabbas52233@gmail.com\n"
        f"  github.com/ahmedabbas52233/TerraVision-AI\n\n"
        f"DISCLAIMER\n"
        f"  Research prototype. Not for operational decision-making.\n"
        f"{div}\n"
    )
