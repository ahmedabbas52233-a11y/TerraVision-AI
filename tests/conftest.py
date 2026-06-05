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
# API CLIENT FIXTURE
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def api_client():
    """
    FastAPI TestClient with GEE and model fully mocked.
    Safe to use in CI without any GEE credentials.
    """
    from fastapi.testclient import TestClient
    from terravision.core import TerraVisionTransformer

    fake_model = TerraVisionTransformer()
    fake_model.eval()

    with (
        patch("api._MODEL",       fake_model),
        patch("api._MODEL_READY", True),
        patch("api._GEE_READY",   True),
    ):
        from api import app
        with TestClient(app, raise_server_exceptions=True) as client:
            yield client
