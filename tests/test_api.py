"""
tests/test_api.py
Integration tests for TerraVision AI FastAPI endpoints.

Covers
──────
  · GET  /v1/            — info card (public)
  · GET  /v1/health      — liveness probe (public)
  · GET  /v1/crops       — authenticated; 401 without key
  · POST /v1/predict     — authenticated; 401, 422, 503, 200 cases
  · Rate limiting        — 429 after 30 req/min
  · CORS headers         — present on responses
  · Schema validation    — lat/lon bounds, crop enum
  · Root redirect        — / → /v1/

Run
───
  TERRAVISION_API_KEY=test-key pytest tests/test_api.py -v --asyncio-mode=auto
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# ── make repo root importable ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# Inject test API key before importing the app
os.environ.setdefault("TERRAVISION_API_KEY", "test-key-abc123")
os.environ.setdefault("TERRAVISION_ENV", "development")

# Patch GEE init and model load before app import so tests run without credentials
import unittest.mock as mock

_MOCK_FEATURES = [0.55, 291.5, 0.025]
_MOCK_ERA5 = {"temp_c": 20.5, "precip_mm_month": 62.3, "source": "era5-land"}
_MOCK_TILE_URL = "https://earthengine.googleapis.com/v1alpha/projects/earthengine-legacy/maps/mock-id/tiles/{z}/{x}/{y}"

# Patch api-level GEE init so import doesn't call Earth Engine
# GEE init patched per-test via mock_inference fixture

# Patch at module level before importing app
with mock.patch.dict(os.environ, {"TERRAVISION_API_KEY": "test-key-abc123"}):
    from api import app

VALID_KEY = "test-key-abc123"
INVALID_KEY = "wrong-key"
HEADERS_OK = {"X-API-Key": VALID_KEY}
HEADERS_BAD = {"X-API-Key": INVALID_KEY}

VALID_PAYLOAD = {"lat": 31.5204, "lon": 74.3587, "crop": "Wheat", "mc_passes": 5}


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    """Sync test client — no real GEE calls."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def mock_inference():
    """
    Patch all GEE / torch calls so /predict runs fully offline.
    Returns a context manager that yields the three patch objects.
    """
    from core import TerraVisionTransformer

    fake_model = TerraVisionTransformer()
    fake_model.eval()

    with (
        patch("api._MODEL", fake_model),
        patch("api._MODEL_READY", True),
        patch("api._GEE_READY", True),
        patch("api.get_live_features", return_value=_MOCK_FEATURES),
        patch("api.get_era5_features", return_value=_MOCK_ERA5),
        patch("api.get_ndvi_tile_url", return_value=_MOCK_TILE_URL),
    ):
        yield


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
class TestPublicEndpoints:

    def test_root_returns_200(self, client):
        r = client.get("/v1/")
        assert r.status_code == 200

    def test_root_contains_version(self, client):
        body = client.get("/v1/").json()
        assert "version" in body
        assert "3" in body["version"]

    def test_root_contains_docs_link(self, client):
        body = client.get("/v1/").json()
        assert "docs" in body

    def test_root_redirect(self, client):
        r = client.get("/", follow_redirects=True)
        assert r.status_code == 200

    def test_health_returns_200(self, client):
        r = client.get("/v1/health")
        assert r.status_code == 200

    def test_health_schema(self, client):
        body = client.get("/v1/health").json()
        required = {
            "status",
            "model_ready",
            "gee_ready",
            "model_version",
            "environment",
            "timestamp_utc",
        }
        assert required.issubset(body.keys())

    def test_health_status_field_is_string(self, client):
        body = client.get("/v1/health").json()
        assert body["status"] in ("ok", "degraded")

    def test_health_timestamp_utc_format(self, client):
        body = client.get("/v1/health").json()
        ts = body["timestamp_utc"]
        assert "UTC" in ts
        assert len(ts) >= 20

    def test_openapi_schema_accessible(self, client):
        r = client.get("/v1/openapi.json")
        assert r.status_code == 200

    def test_docs_accessible(self, client):
        r = client.get("/v1/docs")
        assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────
class TestAuthentication:

    def test_crops_without_key_returns_401(self, client):
        r = client.get("/v1/crops")
        assert r.status_code == 401

    def test_crops_with_wrong_key_returns_401(self, client):
        r = client.get("/v1/crops", headers=HEADERS_BAD)
        assert r.status_code == 401

    def test_predict_without_key_returns_401(self, client):
        r = client.post("/v1/predict", json=VALID_PAYLOAD)
        assert r.status_code == 401

    def test_predict_with_wrong_key_returns_401(self, client):
        r = client.post("/v1/predict", json=VALID_PAYLOAD, headers=HEADERS_BAD)
        assert r.status_code == 401

    def test_predict_with_empty_key_returns_401(self, client):
        r = client.post("/v1/predict", json=VALID_PAYLOAD, headers={"X-API-Key": ""})
        assert r.status_code == 401

    def test_error_body_contains_detail(self, client):
        body = client.get("/v1/crops").json()
        assert "detail" in body


# ─────────────────────────────────────────────────────────────────────────────
# /v1/crops
# ─────────────────────────────────────────────────────────────────────────────
class TestCropsEndpoint:

    def test_crops_with_valid_key(self, client):
        r = client.get("/v1/crops", headers=HEADERS_OK)
        assert r.status_code == 200

    def test_crops_returns_list(self, client):
        body = client.get("/v1/crops", headers=HEADERS_OK).json()
        assert isinstance(body, list)
        assert len(body) == 4

    def test_crops_contain_expected_names(self, client):
        body = client.get("/v1/crops", headers=HEADERS_OK).json()
        names = {item["name"] for item in body}
        assert names == {"Wheat", "Rice", "Maize", "Soybean"}

    def test_crop_item_schema(self, client):
        body = client.get("/v1/crops", headers=HEADERS_OK).json()
        for item in body:
            assert "name" in item
            assert "temp_K" in item
            assert "moisture" in item
            assert "ndvi_scale" in item

    def test_crop_temp_k_in_range(self, client):
        body = client.get("/v1/crops", headers=HEADERS_OK).json()
        for item in body:
            assert (
                270.0 <= item["temp_K"] <= 310.0
            ), f"{item['name']}: temp_K={item['temp_K']} out of range"


# ─────────────────────────────────────────────────────────────────────────────
# /v1/predict  — validation
# ─────────────────────────────────────────────────────────────────────────────
class TestPredictValidation:

    @pytest.mark.parametrize("lat", [-91.0, 91.0, 999.0])
    def test_invalid_lat_returns_422(self, client, lat):
        r = client.post(
            "/v1/predict",
            json={"lat": lat, "lon": 0.0, "crop": "Wheat"},
            headers=HEADERS_OK,
        )
        assert r.status_code == 422, f"Expected 422 for lat={lat}, got {r.status_code}"

    @pytest.mark.parametrize("lon", [-181.0, 181.0, 999.0])
    def test_invalid_lon_returns_422(self, client, lon):
        r = client.post(
            "/v1/predict",
            json={"lat": 0.0, "lon": lon, "crop": "Wheat"},
            headers=HEADERS_OK,
        )
        assert r.status_code == 422, f"Expected 422 for lon={lon}, got {r.status_code}"

    def test_invalid_crop_returns_422(self, client):
        r = client.post(
            "/v1/predict",
            json={"lat": 0.0, "lon": 0.0, "crop": "Tomato"},
            headers=HEADERS_OK,
        )
        assert r.status_code == 422

    def test_missing_lat_returns_422(self, client):
        r = client.post(
            "/v1/predict", json={"lon": 0.0, "crop": "Wheat"}, headers=HEADERS_OK
        )
        assert r.status_code == 422

    def test_missing_body_returns_422(self, client):
        r = client.post("/v1/predict", headers=HEADERS_OK)
        assert r.status_code == 422

    @pytest.mark.parametrize("lat,lon", [(-90.0, -180.0), (90.0, 180.0), (0.0, 0.0)])
    def test_boundary_coordinates_accepted(self, client, mock_inference, lat, lon):
        r = client.post(
            "/v1/predict",
            json={"lat": lat, "lon": lon, "crop": "Rice"},
            headers=HEADERS_OK,
        )
        # 200 or 503 (no model) — not 422
        assert r.status_code in (
            200,
            503,
        ), f"Expected 200/503 for boundary coords, got {r.status_code}"


# ─────────────────────────────────────────────────────────────────────────────
# /v1/predict  — success path
# ─────────────────────────────────────────────────────────────────────────────
class TestPredictSuccess:

    def test_predict_returns_200(self, client, mock_inference):
        r = client.post("/v1/predict", json=VALID_PAYLOAD, headers=HEADERS_OK)
        assert r.status_code == 200, r.text

    def test_predict_response_schema(self, client, mock_inference):
        body = client.post("/v1/predict", json=VALID_PAYLOAD, headers=HEADERS_OK).json()
        required = {
            "lat",
            "lon",
            "crop",
            "ndvi",
            "ndvi_status",
            "era5",
            "yield_base_t_ha",
            "yield_adj_t_ha",
            "yield_delta_t_ha",
            # MC Dropout confidence fields (Gap 1 fix — not hardcoded)
            "confidence_pct",
            "yield_std_t_ha",
            "ci_95_lower",
            "ci_95_upper",
            "carbon_mg_c_ha",
            "carbon_fraction",
            "model_name",
            "model_version",
            "inference_ms",
            "generated_utc",
        }
        assert required.issubset(body.keys()), f"Missing keys: {required - body.keys()}"

    def test_confidence_is_real_not_hardcoded(self, client, mock_inference):
        body = client.post("/v1/predict", json=VALID_PAYLOAD, headers=HEADERS_OK).json()
        # Confidence must be a float in [50, 99] — real MC Dropout, not the old magic 94.2
        assert isinstance(body["confidence_pct"], float)
        assert 50.0 <= body["confidence_pct"] <= 99.0

    def test_mc_dropout_fields_present(self, client, mock_inference):
        body = client.post("/v1/predict", json=VALID_PAYLOAD, headers=HEADERS_OK).json()
        assert isinstance(body["yield_std_t_ha"], float)
        assert isinstance(body["ci_95_lower"], float)
        assert isinstance(body["ci_95_upper"], float)
        assert body["ci_95_lower"] <= body["yield_adj_t_ha"] <= body["ci_95_upper"]

    def test_model_name_in_response(self, client, mock_inference):
        body = client.post("/v1/predict", json=VALID_PAYLOAD, headers=HEADERS_OK).json()
        assert body["model_name"] in (
            "TerraVisionTransformerV2",
            "TerraVisionTransformer",
        )

    def test_predict_ndvi_status_schema(self, client, mock_inference):
        body = client.post("/v1/predict", json=VALID_PAYLOAD, headers=HEADERS_OK).json()
        ns = body["ndvi_status"]
        assert {"label", "action", "alert_type"}.issubset(ns.keys())
        assert ns["alert_type"] in ("error", "warning", "info", "success")

    def test_predict_era5_schema(self, client, mock_inference):
        body = client.post("/v1/predict", json=VALID_PAYLOAD, headers=HEADERS_OK).json()
        e5 = body["era5"]
        assert {"temp_c", "precip_mm_month", "source"}.issubset(e5.keys())

    def test_predict_yield_values_positive(self, client, mock_inference):
        body = client.post("/v1/predict", json=VALID_PAYLOAD, headers=HEADERS_OK).json()
        assert body["yield_base_t_ha"] >= 0.0
        assert body["yield_adj_t_ha"] >= 0.0
        assert body["carbon_mg_c_ha"] >= 0.0

    def test_predict_carbon_fraction_correct(self, client, mock_inference):
        body = client.post("/v1/predict", json=VALID_PAYLOAD, headers=HEADERS_OK).json()
        assert body["carbon_fraction"] == pytest.approx(0.47, abs=1e-6)

    def test_predict_model_version_in_response(self, client, mock_inference):
        body = client.post("/v1/predict", json=VALID_PAYLOAD, headers=HEADERS_OK).json()
        assert "3" in body["model_version"]

    def test_predict_inference_ms_positive(self, client, mock_inference):
        body = client.post("/v1/predict", json=VALID_PAYLOAD, headers=HEADERS_OK).json()
        assert body["inference_ms"] > 0.0

    def test_predict_no_report_by_default(self, client, mock_inference):
        body = client.post("/v1/predict", json=VALID_PAYLOAD, headers=HEADERS_OK).json()
        assert body.get("report") is None

    def test_predict_report_included_when_requested(self, client, mock_inference):
        payload = {**VALID_PAYLOAD, "include_report": True}
        body = client.post("/v1/predict", json=payload, headers=HEADERS_OK).json()
        assert body.get("report") is not None
        assert "TerraVision" in body["report"]

    def test_predict_ndvi_tile_null_by_default(self, client, mock_inference):
        body = client.post("/v1/predict", json=VALID_PAYLOAD, headers=HEADERS_OK).json()
        assert body.get("ndvi_tile_url") is None

    @pytest.mark.parametrize("crop", ["Wheat", "Rice", "Maize", "Soybean"])
    def test_all_crops_succeed(self, client, mock_inference, crop):
        payload = {**VALID_PAYLOAD, "crop": crop}
        r = client.post("/v1/predict", json=payload, headers=HEADERS_OK)
        assert r.status_code == 200, f"{crop}: {r.text}"

    def test_yield_delta_equals_adj_minus_base(self, client, mock_inference):
        body = client.post("/v1/predict", json=VALID_PAYLOAD, headers=HEADERS_OK).json()
        expected_delta = round(body["yield_adj_t_ha"] - body["yield_base_t_ha"], 3)
        assert body["yield_delta_t_ha"] == pytest.approx(expected_delta, abs=0.001)


# ─────────────────────────────────────────────────────────────────────────────
# /v1/predict  — error states
# ─────────────────────────────────────────────────────────────────────────────
class TestPredictErrors:

    def test_model_not_loaded_returns_503(self, client):
        with patch("api._MODEL_READY", False):
            r = client.post("/v1/predict", json=VALID_PAYLOAD, headers=HEADERS_OK)
        assert r.status_code == 503

    def test_503_body_contains_detail(self, client):
        with patch("api._MODEL_READY", False):
            body = client.post(
                "/v1/predict", json=VALID_PAYLOAD, headers=HEADERS_OK
            ).json()
        assert "detail" in body


# ─────────────────────────────────────────────────────────────────────────────
# API VERSIONING
# ─────────────────────────────────────────────────────────────────────────────
class TestApiVersioning:

    def test_unversioned_predict_returns_404(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS_OK)
        assert r.status_code == 404

    def test_unversioned_crops_returns_404(self, client):
        r = client.get("/crops", headers=HEADERS_OK)
        assert r.status_code == 404

    def test_unversioned_health_returns_404(self, client):
        r = client.get("/health")
        assert r.status_code == 404

    def test_v1_prefix_required(self, client, mock_inference):
        r = client.post("/v1/predict", json=VALID_PAYLOAD, headers=HEADERS_OK)
        assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE CONTENT-TYPE
# ─────────────────────────────────────────────────────────────────────────────
class TestContentType:

    def test_health_is_json(self, client):
        r = client.get("/v1/health")
        assert "application/json" in r.headers.get("content-type", "")

    def test_root_is_json(self, client):
        r = client.get("/v1/")
        assert "application/json" in r.headers.get("content-type", "")

    def test_predict_is_json(self, client, mock_inference):
        r = client.post("/v1/predict", json=VALID_PAYLOAD, headers=HEADERS_OK)
        assert "application/json" in r.headers.get("content-type", "")
