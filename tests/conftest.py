"""
tests/conftest.py
Shared pytest fixtures for TerraVision AI test suite.
Available to all test files without explicit import.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

# Ensure project root is on PYTHONPATH regardless of how pytest is invoked
sys.path.insert(0, str(Path(__file__).parent.parent))

# Inject test credentials before any module imports
os.environ.setdefault("TERRAVISION_API_KEY", "pytest-key-xyz789")
os.environ.setdefault("TERRAVISION_ENV", "development")


# ─────────────────────────────────────────────────────────────────────────────
# SHARED FIXTURES
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def fresh_model():
    """A freshly initialised TerraVisionTransformer with random weights."""
    from terravision.core import TerraVisionTransformer

    m = TerraVisionTransformer(input_dim=3, model_dim=64)
    m.eval()
    return m


@pytest.fixture(scope="session")
def sample_features() -> list[float]:
    """Representative Sentinel-2 + crop-prior feature vector."""
    return [0.55, 291.5, 0.025]


@pytest.fixture(scope="session")
def sample_era5() -> dict:
    return {"temp_c": 22.5, "precip_mm_month": 58.0, "source": "era5-land"}


@pytest.fixture(scope="session")
def sample_tensor(sample_features) -> torch.Tensor:
    return torch.tensor([sample_features], dtype=torch.float32)


@pytest.fixture(scope="session")
def all_crops() -> list[str]:
    return ["Wheat", "Rice", "Maize", "Soybean"]


# ─────────────────────────────────────────────────────────────────────────────
# API CLIENT FIXTURES
# ─────────────────────────────────────────────────────────────────────────────
_MOCK_FEATURES = [0.55, 291.5, 0.025]
_MOCK_FEATURES_V2 = [[0.55, 291.5, 0.025]] * 6  # V2 format (6×3)
_MOCK_ERA5 = {"temp_c": 20.5, "precip_mm_month": 62.3, "source": "era5-land"}
_MOCK_TILE_URL = "https://earthengine.googleapis.com/v1alpha/projects/earthengine-legacy/maps/mock-id/tiles/{z}/{x}/{y}"


@pytest.fixture(scope="module")
def api_client():
    """
    FastAPI TestClient with GEE and model fully mocked.
    Safe to use in CI without any GEE credentials.

    CRITICAL: We must patch _IS_V2 and import api INSIDE the patch context
    so that api.py evaluates _IS_V2 based on our mocked V2 model.
    """
    from fastapi.testclient import TestClient

    from terravision.core import TerraVisionTransformerV2

    fake_model = TerraVisionTransformerV2()
    fake_model.eval()

    # Patch at module level BEFORE importing api
    with (
        patch("api._MODEL", fake_model),
        patch("api._MODEL_READY", True),
        patch("api._GEE_READY", True),
        patch("api._IS_V2", True),  # Force V2 code path
        patch("api.get_live_features_v2", return_value=_MOCK_FEATURES_V2),
        patch("api.get_era5_features", return_value=_MOCK_ERA5),
        patch("api.get_ndvi_tile_url", return_value=_MOCK_TILE_URL),
    ):
        # Import api AFTER all patches are active
        import api

        with TestClient(api.app, raise_server_exceptions=True) as client:
            yield client


@pytest.fixture(scope="module")
def client(api_client):
    """Alias for api_client — matches test naming convention in test_api.py."""
    yield api_client


@pytest.fixture
def mock_inference():
    """
    No-op fixture for backward compatibility.
    All mocking is already done in the api_client/client fixture.
    """
    yield
