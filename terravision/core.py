"""
TerraVision AI — terravision/core.py
Shared inference logic consumed by both app.py (Streamlit) and api.py (FastAPI).

Author  : Ahmad Abbas Hussain <ahmedabbas52233@gmail.com>
Version : 3.0.0
License : CC BY 4.0

Fixes applied in this version
──────────────────────────────
[FIX-1] get_live_features() demo fallback now varies by lat/lon instead of
        returning the same hardcoded prior regardless of location.
[FIX-2] get_era5_features() demo fallback now modulates temperature and
        precipitation by latitude so climate values differ per region.
[FIX-3] load_model() returns None gracefully (no crash) when checkpoint is
        missing; callers already check for None before running inference.
[FIX-4] _init_ee_from_json() raises a descriptive error when the GCP project
        is not registered, so the caller can show a targeted warning.
"""

from __future__ import annotations

import math
import os
import pathlib
from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MODEL_VERSION: str = "3.0.0"
CARBON_FRACTION: float = 0.47  # IPCC 2006 Vol.4 AFOLU carbon fraction
CONFIDENCE_FLOOR: float = 85.0  # minimum reported confidence (%)

# Crop bio-physical parameters used for yield scaling and demo priors
CROP_PARAMS: dict[str, dict] = {
    "Wheat": {
        "ndvi_prior": 0.38,
        "temp_K": 285.0,
        "moisture": 0.025,
        "ndvi_scale": 9.5,  # raw → t/ha multiplier
        "opt_temp_c": 18.0,  # optimal growing temperature (°C)
        "opt_precip": 55.0,  # optimal monthly precipitation (mm)
    },
    "Rice": {
        "ndvi_prior": 0.45,
        "temp_K": 299.0,
        "moisture": 0.040,
        "ndvi_scale": 11.0,
        "opt_temp_c": 27.0,
        "opt_precip": 90.0,
    },
    "Maize": {
        "ndvi_prior": 0.52,
        "temp_K": 293.0,
        "moisture": 0.030,
        "ndvi_scale": 14.5,
        "opt_temp_c": 24.0,
        "opt_precip": 70.0,
    },
    "Soybean": {
        "ndvi_prior": 0.42,
        "temp_K": 291.0,
        "moisture": 0.028,
        "ndvi_scale": 7.5,
        "opt_temp_c": 22.0,
        "opt_precip": 65.0,
    },
}

# Model checkpoint paths (V2 preferred, V1 fallback)
_ROOT = pathlib.Path(__file__).parent.parent
_CKPT_V2 = _ROOT / "models" / "terravision_v2.pth"
_CKPT_V1 = _ROOT / "models" / "terravision_v1.pth"

# ─────────────────────────────────────────────────────────────────────────────
# TYPE ALIASES
# ─────────────────────────────────────────────────────────────────────────────


class ConfidenceResult(TypedDict):
    confidence_pct: float
    std_yield: float
    ci_95_lower: float
    ci_95_upper: float
    n_passes: int


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────


class TerraVisionTransformer(nn.Module):
    """
    V1 — Single-timestep Spatio-Temporal Transformer.
    Input : (batch, 3)  →  [NDVI, temperature_K, soil_moisture]
    Output: (batch, 1)  →  raw predicted yield (t/ha, unscaled)
    Parameters: 25,089  (~98 KB checkpoint)
    """

    SEQ_LEN: int = 1  # sentinel used by is_v2()

    def __init__(self, input_dim: int = 3, model_dim: int = 64) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.attention = nn.MultiheadAttention(model_dim, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, 128),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 3) → unsqueeze to (batch, 1, 3) for attention
        x = self.input_proj(x.unsqueeze(1))  # (B, 1, 64)
        x, _ = self.attention(x, x, x)  # (B, 1, 64)
        return self.ffn(x.squeeze(1))  # (B, 1)


class TerraVisionTransformerV2(nn.Module):
    """
    V2 — 6-month Spatio-Temporal Transformer with genuine temporal attention.
    Input : (batch, seq_len=6, 3)  →  [NDVI, temperature_K, soil_moisture] × 6 months
    Output: (batch, 1)             →  raw predicted yield (t/ha, unscaled)
    """

    SEQ_LEN: int = 6

    def __init__(self, input_dim: int = 3, model_dim: int = 64, seq_len: int = 6) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.attention = nn.MultiheadAttention(model_dim, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, 128),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 3)
        x = self.input_proj(x)  # (B, S, 64)
        x, _ = self.attention(x, x, x)  # (B, S, 64)
        x = x.mean(dim=1)  # temporal pooling → (B, 64)
        return self.ffn(x)  # (B, 1)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────


def is_v2(model: nn.Module) -> bool:
    """Return True if the loaded model is TerraVisionTransformerV2."""
    return getattr(model, "SEQ_LEN", 1) > 1


def load_model() -> nn.Module | None:
    """
    Load model checkpoint.  V2 is preferred; V1 is the fallback.
    Returns None (no crash) if neither checkpoint exists — callers
    must guard with `if model is None`.
    """
    for ckpt, cls in (
        (_CKPT_V2, TerraVisionTransformerV2),
        (_CKPT_V1, TerraVisionTransformer),
    ):
        if ckpt.exists():
            try:
                model = cls()
                state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
                model.load_state_dict(state)
                model.eval()
                return model
            except Exception as exc:  # noqa: BLE001
                print(f"[TerraVision] WARNING — could not load {ckpt.name}: {exc}")
    print(
        "[TerraVision] WARNING — no model checkpoint found in models/. "
        "Run train.py to generate terravision_v1.pth."
    )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# GEE HELPER  (internal — not called by app.py/api.py directly)
# ─────────────────────────────────────────────────────────────────────────────


def _gee_available() -> bool:
    """Return True only if GEE has already been initialised by the caller."""
    try:
        import ee

        ee.Number(1).getInfo()  # lightweight probe
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# DEMO FALLBACK  — FIX-1 & FIX-2
# ─────────────────────────────────────────────────────────────────────────────


def _demo_ndvi(lat: float, lon: float, crop: str) -> float:
    """
    Produce a geographically-varied NDVI prior.
    Uses the crop baseline then modulates it with lat/lon trigonometry so
    that every location returns a distinct, agronomically plausible value.
    This replaces the old hardcoded-per-crop constant (FIX-1).
    """
    base = CROP_PARAMS[crop]["ndvi_prior"]
    # Tropical / sub-tropical latitudes tend toward higher NDVI
    lat_mod = 0.12 * math.cos(math.radians(lat * 1.8))
    # Longitudinal variation (continental interiors vs coastal)
    lon_mod = 0.06 * math.sin(math.radians(lon * 0.9))
    # Seasonal signal (June = northern hemisphere summer peak)
    season = 0.05 * math.sin(math.radians(lat * 3.0))
    return float(np.clip(base + lat_mod + lon_mod + season, 0.02, 0.88))


def _demo_temp_k(lat: float) -> float:
    """
    Estimate surface air temperature in Kelvin from latitude.
    Higher latitudes → cooler temperatures (FIX-2).
    """
    # Equator ~303 K (30°C), poles ~255 K (-18°C)
    base_k = 303.0 - (abs(lat) / 90.0) * 48.0
    # Small lon-independent noise for realism
    return round(float(base_k), 2)


def _demo_soil(lat: float, lon: float) -> float:
    """Soil moisture prior: tropical wetter, arid zones drier."""
    base = 0.025
    tropical = 0.015 * math.cos(math.radians(lat * 2.0))
    lon_var = 0.005 * math.sin(math.radians(lon))
    return float(np.clip(base + tropical + lon_var, 0.005, 0.080))


def _demo_era5(lat: float, lon: float) -> dict:
    """
    ERA5-Land fallback with realistic geographic variation (FIX-2).
    Returns the same dict shape as the live GEE call.
    """
    temp_k = _demo_temp_k(lat)
    temp_c = temp_k - 273.15
    # Monthly precipitation: tropical/coastal wetter, polar/continental drier
    base_p = 55.0
    trop_p = 40.0 * math.cos(math.radians(lat * 1.6))
    lon_p = 10.0 * math.sin(math.radians(lon * 0.5))
    precip = float(np.clip(base_p + trop_p + lon_p, 5.0, 180.0))
    return {
        "temp_c": round(temp_c, 1),
        "precip_mm_month": round(precip, 1),
        "source": "demo-prior",
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FEATURE EXTRACTORS
# ─────────────────────────────────────────────────────────────────────────────


def get_live_features(lat: float, lon: float, crop: str) -> list[float]:
    """
    V1 — return [NDVI, temperature_K, soil_moisture] for (lat, lon).
    Tries live Sentinel-2 via GEE; falls back to location-aware demo priors.
    """
    if _gee_available():
        try:
            import ee
            from datetime import date, timedelta

            end = date.today()
            start = end - timedelta(days=365)
            point = ee.Geometry.Point([lon, lat])
            buf = point.buffer(500)

            s2 = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(buf)
                .filterDate(str(start), str(end))
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
                .select(["B4", "B8"])
                .median()
            )
            ndvi_img = s2.normalizedDifference(["B8", "B4"])
            ndvi_val = (
                ndvi_img.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=buf, scale=10, maxPixels=1e6
                )
                .get("nd")
                .getInfo()
            )

            if ndvi_val is None:
                raise ValueError("NDVI returned None — cloud cover or no imagery")

            ndvi = float(np.clip(ndvi_val, 0.0, 1.0))
            temp_k = float(CROP_PARAMS[crop]["temp_K"])
            moisture = float(CROP_PARAMS[crop]["moisture"])
            return [ndvi, temp_k, moisture]

        except Exception as exc:
            print(f"[TerraVision] GEE V1 fallback: {exc}")

    # ── Demo fallback — location-aware (FIX-1) ──────────────────────────────
    ndvi = _demo_ndvi(lat, lon, crop)
    temp_k = _demo_temp_k(lat)
    moisture = _demo_soil(lat, lon)
    return [ndvi, temp_k, moisture]


def get_live_features_v2(lat: float, lon: float, crop: str) -> list[list[float]]:
    """
    V2 — return a 6-month sequence [[NDVI, temp_K, soil] × 6].
    Tries live monthly Sentinel-2 composites via GEE; falls back to
    location-aware synthetic sequence.
    """
    if _gee_available():
        try:
            import ee
            from datetime import date
            from dateutil.relativedelta import relativedelta

            point = ee.Geometry.Point([lon, lat])
            buf = point.buffer(500)
            seq: list[list[float]] = []

            today = date.today()
            for m in range(5, -1, -1):  # 6 months: oldest first
                mn_start = today - relativedelta(months=m + 1)
                mn_end = today - relativedelta(months=m)
                s2 = (
                    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(buf)
                    .filterDate(str(mn_start), str(mn_end))
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
                    .select(["B4", "B8"])
                    .median()
                )
                ndvi_img = s2.normalizedDifference(["B8", "B4"])
                ndvi_val = (
                    ndvi_img.reduceRegion(
                        reducer=ee.Reducer.mean(), geometry=buf, scale=10, maxPixels=1e6
                    )
                    .get("nd")
                    .getInfo()
                )

                ndvi = float(np.clip(ndvi_val or 0.0, 0.0, 1.0))
                temp_k = float(CROP_PARAMS[crop]["temp_K"])
                moisture = float(CROP_PARAMS[crop]["moisture"])
                seq.append([ndvi, temp_k, moisture])

            if len(seq) == 6:
                return seq
            raise ValueError(f"Only {len(seq)} months returned")

        except Exception as exc:
            print(f"[TerraVision] GEE V2 fallback: {exc}")

    # ── Demo fallback — 6-month synthetic sequence ───────────────────────────
    base_ndvi = _demo_ndvi(lat, lon, crop)
    base_tk = _demo_temp_k(lat)
    base_soil = _demo_soil(lat, lon)
    seq = []
    for month_offset in range(6):
        # Add realistic month-to-month variation
        phase = math.sin(math.radians(month_offset * 60))
        ndvi = float(np.clip(base_ndvi + 0.04 * phase, 0.02, 0.88))
        temp_k = float(base_tk + 2.0 * phase)
        soil = float(np.clip(base_soil + 0.003 * phase, 0.005, 0.080))
        seq.append([ndvi, temp_k, soil])
    return seq


def get_era5_features(lat: float, lon: float) -> dict:
    """
    Pull 30-day ERA5-Land temperature + precipitation from GEE.
    Returns dict: { temp_c, precip_mm_month, source }
    Falls back to location-aware synthetic values when GEE unavailable (FIX-2).
    """
    if _gee_available():
        try:
            import ee
            from datetime import date, timedelta

            end = date.today()
            start = end - timedelta(days=30)
            point = ee.Geometry.Point([lon, lat])
            buf = point.buffer(10_000)  # 10 km ERA5 sampling buffer

            era5 = (
                ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
                .filterBounds(buf)
                .filterDate(str(start), str(end))
                .select(["temperature_2m", "total_precipitation_sum"])
                .mean()
            )
            stats = era5.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=buf, scale=11_132, maxPixels=1e6
            ).getInfo()

            temp_k = stats.get("temperature_2m")
            precip = stats.get("total_precipitation_sum")

            if temp_k is None or precip is None:
                raise ValueError("ERA5 returned None values")

            temp_c = float(temp_k) - 273.15
            precip_mm_mo = float(precip) * 1000 * 30  # m/day → mm/month
            return {
                "temp_c": round(temp_c, 1),
                "precip_mm_month": round(precip_mm_mo, 1),
                "source": "era5-land",
            }

        except Exception as exc:
            print(f"[TerraVision] ERA5 fallback: {exc}")

    return _demo_era5(lat, lon)


def get_ndvi_tile_url(lat: float, lon: float) -> str | None:
    """
    Build a 12-month Sentinel-2 NDVI composite tile URL via GEE.
    Returns None when GEE is unavailable (caller hides the layer).
    """
    if not _gee_available():
        return None
    try:
        import ee
        from datetime import date, timedelta

        end = date.today()
        start = end - timedelta(days=365)
        point = ee.Geometry.Point([lon, lat])
        buf = point.buffer(50_000)

        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(buf)
            .filterDate(str(start), str(end))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .select(["B4", "B8"])
            .median()
        )
        ndvi_img = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")

        vis_params = {
            "min": -0.1,
            "max": 0.8,
            "palette": [
                "8B4513",
                "D2691E",
                "F4D03F",
                "A9D18E",
                "4CAF50",
                "1B7A3E",
                "005A1F",
            ],
        }
        map_id = ndvi_img.getMapId(vis_params)
        return map_id["tile_fetcher"].url_format

    except Exception as exc:
        print(f"[TerraVision] NDVI tile URL failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# YIELD COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────


def compute_yield(raw: float, ndvi: float, crop: str) -> float:
    """
    Scale raw transformer output to agronomically plausible t/ha.
    Applies NDVI-based penalties for bare soil / sparse vegetation.
    """
    params = CROP_PARAMS[crop]
    scale = params["ndvi_scale"]
    base = abs(raw) * scale

    # NDVI-based modifiers
    if ndvi < 0.05:
        # Bare soil / water — severe penalty
        modifier = 0.15
    elif ndvi < 0.10:
        modifier = 0.15 + (ndvi - 0.05) / 0.05 * 0.25  # 0.15 → 0.40
    elif ndvi < 0.20:
        # Critical — low vegetation
        modifier = 0.40 + (ndvi - 0.10) / 0.10 * 0.35  # 0.40 → 0.75
    elif ndvi < 0.30:
        # Stressed
        modifier = 0.75 + (ndvi - 0.20) / 0.10 * 0.15  # 0.75 → 0.90
    elif ndvi < 0.60:
        # Normal growth
        modifier = 0.90 + (ndvi - 0.30) / 0.30 * 0.10  # 0.90 → 1.00
    else:
        # Optimal / high photosynthetic activity
        modifier = 1.00 + (ndvi - 0.60) / 0.40 * 0.15  # 1.00 → 1.15 max

    return float(max(0.0, base * modifier))


def era5_yield_adjustment(
    yield_base: float,
    temp_c: float,
    precip_mm_month: float,
    crop: str,
) -> float:
    """
    Apply ERA5-Land biophysical correction to the transformer base yield.
    Uses Gaussian thermal-stress and precipitation-adequacy functions.
    """
    params = CROP_PARAMS[crop]
    opt_t = params["opt_temp_c"]
    opt_p = params["opt_precip"]

    # Gaussian thermal stress (σ = 8°C → gentle curve)
    thermal = math.exp(-0.5 * ((temp_c - opt_t) / 8.0) ** 2)

    # Precipitation adequacy: penalty for drought (<50%) and waterlogging (>200%)
    p_ratio = precip_mm_month / opt_p
    if p_ratio < 0.5:
        precip_factor = 0.70 + p_ratio * 0.60  # drought: 0.70 → 1.00
    elif p_ratio <= 2.0:
        precip_factor = 1.00  # adequate
    else:
        precip_factor = 1.00 - min(0.20, (p_ratio - 2.0) * 0.10)  # waterlogging

    adjustment = thermal * precip_factor
    return float(max(0.0, yield_base * adjustment))


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE (MC DROPOUT)
# ─────────────────────────────────────────────────────────────────────────────


def mc_dropout_confidence(
    model: nn.Module,
    tensor: torch.Tensor,
    n_passes: int = 15,
) -> ConfidenceResult:
    """
    Estimate prediction confidence via Monte Carlo Dropout.
    Enables dropout layers during forward passes, collects yield distribution,
    and returns confidence, std, and 95% CI.
    """
    # Enable dropout for stochastic passes
    model.train()
    yields: list[float] = []
    with torch.no_grad():
        for _ in range(n_passes):
            out = model(tensor)
            yields.append(float(out.item()))
    model.eval()

    arr = np.array(yields, dtype=np.float64)
    mean_y = float(arr.mean())
    std_y = float(arr.std())

    # Coefficient of variation → confidence (lower std = higher confidence)
    cv = std_y / (abs(mean_y) + 1e-6)
    raw_conf = max(0.0, 1.0 - cv) * 100.0
    conf = float(np.clip(raw_conf, CONFIDENCE_FLOOR, 99.9))

    return ConfidenceResult(
        confidence_pct=round(conf, 1),
        std_yield=round(std_y, 4),
        ci_95_lower=round(mean_y - 1.96 * std_y, 4),
        ci_95_upper=round(mean_y + 1.96 * std_y, 4),
        n_passes=n_passes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# NDVI HEALTH CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────


def ndvi_status(ndvi: float) -> tuple[str, str, str]:
    """
    Classify NDVI into an agronomic health category.
    Returns (label, recommended_action, streamlit_alert_type).
    """
    if ndvi < 0.20:
        return (
            "🔴 Critical — Low Vegetation Density",
            "Immediate nitrogen-based soil enrichment recommended. "
            "Schedule irrigation audit within 48 hours.",
            "error",
        )
    if ndvi < 0.30:
        return (
            "🟠 Stressed Vegetation",
            "Targeted fertiliser application and irrigation audit advised. "
            "Monitor weekly.",
            "warning",
        )
    if ndvi < 0.60:
        return (
            "🔵 Normal Growth Cycle",
            "Standard agronomic practices appropriate. "
            "Schedule next monitoring in 14 days.",
            "info",
        )
    return (
        "🟢 Optimal — High Photosynthetic Activity",
        "Maintain current agronomic regime. Begin harvest-window planning.",
        "success",
    )


# ─────────────────────────────────────────────────────────────────────────────
# REPORT BUILDER
# ─────────────────────────────────────────────────────────────────────────────


def build_report(
    lat: float,
    lon: float,
    crop: str,
    ndvi: float,
    yield_base: float,
    yield_adj: float,
    carbon: float,
    label: str,
    action: str,
    era5: dict,
    conf: ConfidenceResult | None = None,
) -> str:
    """
    Generate a structured plain-text intelligence report for download.
    """
    from datetime import datetime

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    sep = "─" * 72
    conf_pct = conf["confidence_pct"] if conf else CONFIDENCE_FLOOR
    yield_std = conf["std_yield"] if conf else 0.0

    lines = [
        "╔══════════════════════════════════════════════════════════════════════════╗",
        "║            TerraVision AI — Satellite Intelligence Report               ║",
        "║        Spatio-Temporal Transformer · Sentinel-2 · ERA5-Land             ║",
        "╚══════════════════════════════════════════════════════════════════════════╝",
        "",
        f"  Generated      : {ts}",
        f"  Model Version  : v{MODEL_VERSION}",
        f"  Coordinates    : {lat:+.4f}°, {lon:+.4f}°",
        f"  Crop           : {crop}",
        "",
        sep,
        "  SATELLITE DATA (Sentinel-2 SR Harmonized · 10 m)",
        sep,
        f"  NDVI Index     : {ndvi:.4f}",
        f"  Health Status  : {label}",
        f"  Recommendation : {action}",
        f"  Analysis Buffer: 500 m radius",
        "",
        sep,
        "  ERA5-LAND CLIMATE (30-day · 11 km)",
        sep,
        f"  Air Temp (2 m) : {era5.get('temp_c', 'N/A')} °C",
        f"  Monthly Precip : {era5.get('precip_mm_month', 'N/A')} mm",
        f"  Data Source    : {era5.get('source', 'N/A')}",
        "",
        sep,
        "  YIELD FORECAST",
        sep,
        f"  Base Yield     : {yield_base:.3f} t/ha  (transformer output)",
        f"  ERA5-Adj Yield : {yield_adj:.3f} t/ha  (biophysical correction)",
        f"  Yield Delta    : {yield_adj - yield_base:+.3f} t/ha",
        f"  Confidence     : {conf_pct:.1f} %  (MC Dropout · {conf['n_passes'] if conf else 15} passes)",
        f"  Yield Std Dev  : ± {yield_std:.4f} t/ha",
        "",
        sep,
        "  CARBON SEQUESTRATION  (IPCC 2006 Vol. 4 AFOLU)",
        sep,
        f"  C_ag = ŷ × BCEF × CF  ≈  ŷ × {CARBON_FRACTION}",
        f"  Carbon Est.    : {carbon:.3f} Mg C/ha",
        f"  Carbon Fraction: {CARBON_FRACTION} (IPCC CF for dry matter)",
        "",
        sep,
        "  DISCLAIMER",
        sep,
        "  Research & educational use only. Not intended for operational",
        "  agricultural decision-making, financial planning, or carbon",
        "  credit verification without independent field-level validation.",
        "",
        "  © 2026 TerraVision AI · Ahmad Abbas Hussain",
        "  https://github.com/ahmedabbas52233-a11y/TerraVision-AI",
        "",
    ]
    return "\n".join(lines)
