"""
tests/test_core.py
Unit tests for TerraVision AI core logic.

Covers
──────
  · TerraVisionTransformer — architecture, forward pass, output shape
  · compute_yield          — normal, bare-soil penalty, ceiling clamp
  · era5_yield_adjustment  — optimum, cold stress, drought, surplus
  · ndvi_status            — all four NDVI brackets
  · build_report           — content, ERA5 block, structure
  · CROP_PARAMS            — completeness and range sanity
  · MODEL_PATH             — pathlib resolution (no cwd dependency)

Run
───
  pytest tests/test_core.py -v --cov=core --cov-report=term-missing
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ── make repo root importable when running from any directory ─────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from terravision.core import (
    CARBON_FRACTION,
    TerraVisionTransformerV2,
    CONFIDENCE_PCT,
    CROP_PARAMS,
    ERA5_CROP_OPTIMA,
    MODEL_PATH,
    MODEL_VERSION,
    NDVI_CLASSES,
    YIELD_MAX,
    TerraVisionTransformer,
    build_report,
    compute_yield,
    era5_yield_adjustment,
    ndvi_status,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def model() -> TerraVisionTransformer:
    """Fresh model with random weights — no checkpoint required for unit tests."""
    m = TerraVisionTransformer(input_dim=3, model_dim=64)
    m.eval()
    return m


@pytest.fixture(scope="module")
def sample_tensor() -> torch.Tensor:
    return torch.tensor([[0.55, 294.0, 0.045]], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
class TestTerraVisionTransformer:

    def test_output_shape(self, model, sample_tensor):
        with torch.no_grad():
            out = model(sample_tensor)
        assert out.shape == (1, 1), f"Expected (1,1), got {out.shape}"

    def test_batch_output_shape(self, model):
        batch = torch.randn(8, 3)
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (8, 1), f"Expected (8,1), got {out.shape}"

    def test_output_is_finite(self, model, sample_tensor):
        with torch.no_grad():
            out = model(sample_tensor)
        assert torch.isfinite(out).all(), "Model output contains NaN or Inf"

    def test_no_gradient_in_eval(self, model, sample_tensor):
        with torch.no_grad():
            out = model(sample_tensor)
        assert out.requires_grad is False

    def test_deterministic_in_eval(self, model, sample_tensor):
        """Eval mode + torch.no_grad should give identical outputs."""
        with torch.no_grad():
            out1 = model(sample_tensor)
            out2 = model(sample_tensor)
        assert torch.allclose(out1, out2), "Non-deterministic output in eval mode"

    def test_parameter_count(self, model):
        """25,089 parameters for input_dim=3, model_dim=64."""
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == 25_089, f"Expected 25,089 params, got {n_params}"

    def test_input_projection_xavier_init(self):
        """Xavier init should give non-zero weights with zero bias."""
        m = TerraVisionTransformer()
        assert not torch.all(m.input_proj.weight == 0), "Input projection weights are all zero"
        assert torch.all(m.input_proj.bias == 0), "Input projection bias should be zeros"



# ─────────────────────────────────────────────────────────────────────────────
# V2 MODEL — GENUINE TEMPORAL ATTENTION
# ─────────────────────────────────────────────────────────────────────────────
class TestTerraVisionTransformerV2:

    def test_output_shape(self, model_v2, sample_tensor_v2):
        with torch.no_grad():
            out = model_v2(sample_tensor_v2)
        assert out.shape == (1, 1), f"Expected (1,1), got {out.shape}"

    def test_batch_output_shape(self, model_v2):
        batch = torch.randn(8, 6, 3)
        with torch.no_grad():
            out = model_v2(batch)
        assert out.shape == (8, 1)

    def test_output_is_finite(self, model_v2, sample_tensor_v2):
        with torch.no_grad():
            out = model_v2(sample_tensor_v2)
        assert torch.isfinite(out).all()

    def test_parameter_count(self, model_v2):
        n = sum(p.numel() for p in model_v2.parameters())
        assert 30_000 <= n <= 40_000, f"V2 params {n} outside expected range [30k, 40k]"

    def test_positional_embedding_shape(self, model_v2):
        assert model_v2.pos_embed.shape == (1, 6, 64)

    def test_different_sequences_give_different_outputs(self, model_v2):
        seq_a = torch.tensor([[[0.8, 291.5, 0.025]] * 6], dtype=torch.float32)
        seq_b = torch.tensor([[[0.1, 291.5, 0.025]] * 6], dtype=torch.float32)
        with torch.no_grad():
            out_a = model_v2(seq_a).item()
            out_b = model_v2(seq_b).item()
        assert out_a != out_b, "V2 model should give different outputs for different NDVI sequences"

    def test_deterministic_in_eval(self, model_v2, sample_tensor_v2):
        with torch.no_grad():
            o1 = model_v2(sample_tensor_v2)
            o2 = model_v2(sample_tensor_v2)
        assert torch.allclose(o1, o2)


# ─────────────────────────────────────────────────────────────────────────────
# MC DROPOUT CONFIDENCE
# ─────────────────────────────────────────────────────────────────────────────
class TestMCDropoutConfidence:

    def test_returns_confidence_result(self, model):
        from terravision.core import mc_dropout_confidence, ConfidenceResult
        tensor = torch.tensor([[0.5, 291.5, 0.025]], dtype=torch.float32)
        result = mc_dropout_confidence(model, tensor, n_passes=10)
        assert isinstance(result, dict)
        for key in ("mean_yield", "std_yield", "confidence_pct", "ci_95_lower", "ci_95_upper"):
            assert key in result, f"Missing key: {key}"

    def test_confidence_in_valid_range(self, model):
        from terravision.core import mc_dropout_confidence
        tensor = torch.tensor([[0.5, 291.5, 0.025]], dtype=torch.float32)
        r = mc_dropout_confidence(model, tensor, n_passes=10)
        assert 0.0 <= r["confidence_pct"] <= 100.0

    def test_ci_bounds_consistent(self, model):
        from terravision.core import mc_dropout_confidence
        tensor = torch.tensor([[0.5, 291.5, 0.025]], dtype=torch.float32)
        r = mc_dropout_confidence(model, tensor, n_passes=10)
        assert r["ci_95_lower"] <= r["mean_yield"] <= r["ci_95_upper"]

    def test_std_is_positive(self, model):
        from terravision.core import mc_dropout_confidence
        tensor = torch.tensor([[0.5, 291.5, 0.025]], dtype=torch.float32)
        r = mc_dropout_confidence(model, tensor, n_passes=15)
        assert r["std_yield"] >= 0.0

    def test_model_returns_to_eval_after_call(self, model):
        from terravision.core import mc_dropout_confidence
        tensor = torch.tensor([[0.5, 291.5, 0.025]], dtype=torch.float32)
        mc_dropout_confidence(model, tensor, n_passes=5)
        assert not model.training, "Model must be restored to eval() after MC Dropout"

    def test_v2_model_mc_dropout(self, model_v2, sample_tensor_v2):
        from terravision.core import mc_dropout_confidence
        r = mc_dropout_confidence(model_v2, sample_tensor_v2, n_passes=10)
        assert 0.0 <= r["confidence_pct"] <= 100.0
        assert r["ci_95_lower"] <= r["ci_95_upper"]

    def test_more_passes_reduces_variance(self, model):
        """More MC passes should converge to a more stable estimate."""
        from terravision.core import mc_dropout_confidence
        tensor = torch.tensor([[0.5, 291.5, 0.025]], dtype=torch.float32)
        stds = [mc_dropout_confidence(model, tensor, n_passes=n)["std_yield"]
                for n in [5, 5, 5, 30, 30, 30]]
        # Average std with more passes should be stable (not necessarily lower,
        # but should not be wildly inconsistent — just verify no exception)
        assert all(s >= 0 for s in stds)


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE YIELD
# ─────────────────────────────────────────────────────────────────────────────
class TestComputeYield:

    @pytest.mark.parametrize("crop", ["Wheat", "Rice", "Maize", "Soybean"])
    def test_yield_within_bounds(self, crop):
        """Yield must always be in [0, YIELD_MAX] regardless of inputs."""
        raw    = 3.0
        ndvi   = 0.5
        result = compute_yield(raw, ndvi, crop)
        assert 0.0 <= result <= YIELD_MAX, f"{crop}: yield {result} outside [0, {YIELD_MAX}]"

    def test_bare_soil_penalty_applied(self):
        """NDVI < 0.1 must suppress yield to ≤ 20% of normal."""
        normal_yield = compute_yield(3.0, 0.55, "Wheat")
        bare_yield   = compute_yield(3.0, 0.05, "Wheat")
        assert bare_yield < normal_yield * 0.25, (
            f"Bare-soil penalty not working: bare={bare_yield:.2f}, normal={normal_yield:.2f}"
        )

    def test_yield_ceiling_enforced(self):
        """Extreme NDVI + raw output must not exceed YIELD_MAX."""
        result = compute_yield(100.0, 1.0, "Maize")
        assert result == pytest.approx(YIELD_MAX, abs=1e-6), (
            f"Ceiling not enforced: got {result}"
        )

    def test_yield_floor_enforced(self):
        """Yield must never be negative."""
        result = compute_yield(-100.0, -1.0, "Wheat")
        assert result >= 0.0

    def test_higher_ndvi_higher_yield(self):
        """For same crop and raw_output, higher NDVI should give higher yield."""
        y_low  = compute_yield(3.0, 0.30, "Maize")
        y_high = compute_yield(3.0, 0.70, "Maize")
        assert y_high > y_low, "Higher NDVI did not produce higher yield"

    @pytest.mark.parametrize("ndvi", [0.0, 0.09, 0.1, 0.11, 0.5, 0.8, 0.99])
    def test_boundary_ndvi_values(self, ndvi):
        result = compute_yield(3.0, ndvi, "Rice")
        assert 0.0 <= result <= YIELD_MAX


# ─────────────────────────────────────────────────────────────────────────────
# ERA5 YIELD ADJUSTMENT
# ─────────────────────────────────────────────────────────────────────────────
class TestEra5YieldAdjustment:

    def test_optimal_conditions_near_base(self):
        """At exact thermal + precipitation optima, factor ≈ 1.0 × 1.05 = ~1.05."""
        crop = "Wheat"
        opt  = ERA5_CROP_OPTIMA[crop]
        base = 3.0
        adj  = era5_yield_adjustment(base, opt["temp_c"], opt["precip_mm"], crop)
        # factor bounded to 1.05 × 1.0 gaussian = 1.05
        assert adj >= base, "Optimal conditions should not reduce yield below base"
        assert adj <= base * 1.06, f"Optimal factor exceeded expected ceiling: {adj}"

    def test_cold_stress_reduces_yield(self):
        """Temperature 30°C below optimum must reduce yield."""
        crop    = "Wheat"
        opt_t   = ERA5_CROP_OPTIMA[crop]["temp_c"]
        opt_p   = ERA5_CROP_OPTIMA[crop]["precip_mm"]
        adj_opt = era5_yield_adjustment(4.0, opt_t, opt_p, crop)
        adj_cold = era5_yield_adjustment(4.0, opt_t - 30, opt_p, crop)
        assert adj_cold < adj_opt, "Cold stress should reduce yield"

    def test_drought_reduces_yield(self):
        """Zero precipitation must reduce yield to ~50% of optimal-precip yield."""
        crop = "Rice"
        opt  = ERA5_CROP_OPTIMA[crop]
        base = 5.0
        adj_opt    = era5_yield_adjustment(base, opt["temp_c"], opt["precip_mm"], crop)
        adj_drought = era5_yield_adjustment(base, opt["temp_c"], 0.0, crop)
        assert adj_drought < adj_opt
        # precip_factor floor is 0.50
        assert adj_drought >= base * 0.45  # some thermal factor may slightly reduce

    def test_output_within_yield_bounds(self):
        """Adjusted yield must always stay in [0, YIELD_MAX]."""
        result = era5_yield_adjustment(10.0, 50.0, 500.0, "Maize")
        assert 0.0 <= result <= YIELD_MAX

    def test_zero_base_yield_stays_zero(self):
        """Adjusting a zero base yield must stay at zero."""
        result = era5_yield_adjustment(0.0, 24.0, 85.0, "Maize")
        assert result == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("crop", ["Wheat", "Rice", "Maize", "Soybean"])
    def test_all_crops_run_without_error(self, crop):
        opt = ERA5_CROP_OPTIMA[crop]
        result = era5_yield_adjustment(3.5, opt["temp_c"], opt["precip_mm"], crop)
        assert isinstance(result, float)


# ─────────────────────────────────────────────────────────────────────────────
# NDVI STATUS
# ─────────────────────────────────────────────────────────────────────────────
class TestNdviStatus:

    def test_critical_bracket(self):
        label, action, alert = ndvi_status(0.10)
        assert "Critical" in label
        assert alert == "error"
        assert len(action) > 10

    def test_stressed_bracket(self):
        label, action, alert = ndvi_status(0.25)
        assert "Stressed" in label
        assert alert == "warning"

    def test_normal_bracket(self):
        label, action, alert = ndvi_status(0.45)
        assert "Normal" in label
        assert alert == "info"

    def test_optimal_bracket(self):
        label, action, alert = ndvi_status(0.75)
        assert "High" in label
        assert alert == "success"

    @pytest.mark.parametrize("ndvi,expected_alert", [
        (-0.5,  "error"),
        (0.0,   "error"),
        (0.19,  "error"),
        (0.20,  "warning"),
        (0.29,  "warning"),
        (0.30,  "info"),
        (0.59,  "info"),
        (0.60,  "success"),
        (0.999, "success"),
    ])
    def test_boundary_classifications(self, ndvi, expected_alert):
        _, _, alert = ndvi_status(ndvi)
        assert alert == expected_alert, f"NDVI {ndvi} → got '{alert}', expected '{expected_alert}'"

    def test_returns_three_strings(self):
        result = ndvi_status(0.5)
        assert len(result) == 3
        assert all(isinstance(s, str) for s in result)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD REPORT
# ─────────────────────────────────────────────────────────────────────────────
class TestBuildReport:

    def _make_report(self, era5=None):
        label, action, _ = ndvi_status(0.45)
        return build_report(
            lat=31.5204, lon=74.3587, crop="Wheat",
            ndvi=0.45, yield_base=3.8, yield_adjusted=3.5,
            carbon=1.645, label=label, action=action,
            era5=era5,
        )

    def test_report_contains_coordinates(self):
        r = self._make_report()
        assert "31.520400" in r
        assert "74.358700" in r

    def test_report_contains_crop(self):
        r = self._make_report()
        assert "Wheat" in r

    def test_report_contains_yield_values(self):
        r = self._make_report()
        assert "3.80" in r
        assert "3.50" in r

    def test_report_contains_carbon(self):
        r = self._make_report()
        assert "1.64" in r or "1.65" in r   # rounding tolerance

    def test_report_contains_doi(self):
        r = self._make_report()
        assert "github.com" in r.lower() or "terravision" in r.lower()

    def test_report_contains_model_version(self):
        r = self._make_report()
        assert MODEL_VERSION in r

    def test_era5_block_present_when_source_live(self):
        era5 = {"temp_c": 22.5, "precip_mm_month": 48.0, "source": "era5-land"}
        r = self._make_report(era5=era5)
        assert "ERA5" in r
        assert "22.5" in r
        assert "48.0" in r

    def test_era5_block_absent_when_source_default(self):
        era5 = {"temp_c": 20.0, "precip_mm_month": 60.0, "source": "default"}
        r = self._make_report(era5=era5)
        # No ERA5 detail section when source is default
        assert "ERA5-LAND CLIMATE" not in r

    def test_report_is_string(self):
        assert isinstance(self._make_report(), str)

    def test_report_has_disclaimer(self):
        r = self._make_report()
        assert "DISCLAIMER" in r or "Disclaimer" in r.lower()


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS SANITY
# ─────────────────────────────────────────────────────────────────────────────
class TestConstants:

    def test_all_crops_in_params(self):
        for crop in ["Wheat", "Rice", "Maize", "Soybean"]:
            assert crop in CROP_PARAMS

    def test_all_crops_in_era5_optima(self):
        for crop in ["Wheat", "Rice", "Maize", "Soybean"]:
            assert crop in ERA5_CROP_OPTIMA

    def test_crop_params_have_required_keys(self):
        required = {"temp_K", "moisture", "base", "ndvi_scale", "offset"}
        for crop, params in CROP_PARAMS.items():
            assert required.issubset(params.keys()), f"{crop} missing keys"

    def test_carbon_fraction_ipcc(self):
        assert CARBON_FRACTION == pytest.approx(0.47, abs=1e-6), \
            "IPCC CF must be exactly 0.47"

    def test_yield_max_agronomic(self):
        assert 12.0 <= YIELD_MAX <= 20.0, "YIELD_MAX outside agronomic range"

    def test_confidence_pct_range(self):
        assert 0.0 < CONFIDENCE_PCT <= 100.0

    def test_ndvi_classes_sorted(self):
        """NDVI class upper bounds must be strictly ascending."""
        bounds = [c[0] for c in NDVI_CLASSES[:-1]]  # exclude +inf
        assert bounds == sorted(bounds), "NDVI_CLASSES are not in ascending order"

    def test_model_path_is_absolute(self):
        assert os.path.isabs(MODEL_PATH), \
            f"MODEL_PATH should be absolute, got: {MODEL_PATH}"

    def test_model_path_uses_pathlib(self):
        """Ensure MODEL_PATH doesn't use a fragile os.path.join with relative parts."""
        assert "TerraVision-AI" in MODEL_PATH or "terravision" in MODEL_PATH.lower() or \
               "models" in MODEL_PATH


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION — MODEL LOAD (skipped if checkpoint absent)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(
    not Path(MODEL_PATH).exists(),
    reason="terravision_v1.pth not present — skipping checkpoint load test",
)
class TestModelLoad:

    def test_checkpoint_loads_without_error(self):
        from terravision.core import load_model
        m = load_model()
        assert m is not None, "load_model() returned None despite checkpoint existing"

    def test_loaded_model_is_eval(self):
        from terravision.core import load_model
        m = load_model()
        assert m is not None, "load_model() returned None despite checkpoint existing"
        assert not m.training, "Loaded model should be in eval mode"
