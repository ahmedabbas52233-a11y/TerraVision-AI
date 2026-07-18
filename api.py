from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
from datetime import UTC, datetime
from typing import Literal, cast

import torch
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request
from starlette.types import ExceptionHandler

from terravision.core import (
    CARBON_FRACTION,
    CROP_PARAMS,
    MODEL_VERSION,
    ConfidenceResult,
    build_report,
    compute_yield,
    era5_yield_adjustment,
    get_era5_features,
    get_live_features,
    get_live_features_v2,
    get_ndvi_tile_url,
    is_v2,
    load_model,
    mc_dropout_confidence,
    ndvi_status,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("terravision.api")

# ── Earth Engine init ─────────────────────────────────────────────────────────


def _normalise_secret(val: object) -> str | None:
    """
    Coerce whatever the env / secret store passes in into a plain JSON string.

    os.getenv() → str (fine as-is)
    Streamlit secrets → AttrDict (must be re-serialised)
    Returns None when val is falsy.
    """
    if not val:
        return None
    if isinstance(val, (str, bytes, bytearray)):
        return val if isinstance(val, str) else val.decode()
    try:
        return json.dumps(dict(val))  # type: ignore[call-overload]
    except Exception:
        return str(val)


def _init_ee() -> bool:
    """
    Initialise GEE for the API context.
    Priority:
      1. GCP_SERVICE_ACCOUNT_JSON env var  (Railway / Docker)
      2. Application Default Credentials   (local dev)
    Handles AttrDict from Streamlit secrets transparently.
    """
    import ee

    # ── 1. Service account env var ────────────────────────────────────────────
    raw_sa = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    sa = _normalise_secret(raw_sa)
    if sa:
        try:
            info = json.loads(sa)
            creds = ee.ServiceAccountCredentials(info["client_email"], key_data=sa)
            ee.Initialize(creds)
            log.info("GEE initialised via service account: %s", info.get("client_email"))
            return True
        except json.JSONDecodeError:
            log.error(
                "GCP_SERVICE_ACCOUNT_JSON is not valid JSON — "
                "paste the full key file content as a single-line string."
            )
            return False
        except Exception as exc:
            err = str(exc)
            if "not registered" in err or "Earth Engine" in err:
                log.error(
                    "GEE project not registered for Earth Engine. "
                    "Visit https://console.cloud.google.com/earth-engine/configuration "
                    "and register the project, then re-deploy. Details: %s",
                    exc,
                )
            else:
                log.warning("GEE service-account init failed (demo mode): %s", exc)
            return False

    # ── 2. Application Default Credentials (local dev fallback) ───────────────
    try:
        ee.Initialize()
        log.info("GEE initialised via Application Default Credentials.")
        return True
    except Exception as exc:
        log.warning("GEE ADC init failed (demo mode): %s", exc)
        return False


_GEE_READY: bool = _init_ee()
_MODEL = load_model()
_MODEL_READY: bool = _MODEL is not None
_IS_V2: bool = _MODEL is not None and is_v2(_MODEL)

if _MODEL_READY:
    _n = sum(p.numel() for p in _MODEL.parameters())  # type: ignore[union-attr]
    log.info("Model loaded — %s  %d params", type(_MODEL).__name__, _n)
else:
    log.error(
        "No model checkpoint found in models/. "
        "Run `python train.py` to generate terravision_v1.pth, "
        "then restart the API server."
    )

# ── Rate limiter ──────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ── Config ────────────────────────────────────────────────────────────────────

_API_KEY: str = os.getenv("TERRAVISION_API_KEY", "dev-insecure-key")
_ENV: str = os.getenv("TERRAVISION_ENV", "development")

# SECURITY: fail closed — refuse to start with the known-public default key
# outside local development.
if _ENV != "development" and _API_KEY == "dev-insecure-key":
    raise RuntimeError(
        "TERRAVISION_API_KEY must be set to a real secret when TERRAVISION_ENV "
        "is not 'development'. Refusing to start with the public default key."
    )

_CORS_ORIGINS: list[str] = (
    ["*"]
    if _ENV == "development"
    else os.getenv("CORS_ORIGINS", "https://terravision-ai.streamlit.app").split(",")
)

# ── API key dependency ────────────────────────────────────────────────────────

_hdr = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(k: str | None = Security(_hdr)) -> str:
    # SECURITY: constant-time comparison avoids leaking timing information
    # about how many leading characters of the key match.
    if not k or not hmac.compare_digest(k, _API_KEY):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )
    return k


# ── Noop coroutine ────────────────────────────────────────────────────────────


async def _noop_str() -> str | None:
    return None


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="TerraVision AI API",
    description=(
        "Satellite-native crop yield intelligence.\n\n"
        "**V2 model**: 6-month Sentinel-2 time-series → genuine temporal attention.\n\n"
        "**Confidence**: live MC Dropout (20 passes) — not hardcoded.\n\n"
        "**Auth**: X-API-Key header on /v1/predict and /v1/crops.\n\n"
        "**Rate limit**: 30 req/min per IP."
    ),
    version=MODEL_VERSION,
    contact={
        "name": "Ahmad Abbas Hussain",
        "email": "ahmedabbas52233@gmail.com",
        "url": "https://github.com/ahmedabbas52233-a11y/TerraVision-AI",
    },
    license_info={
        "name": "CC BY 4.0",
        "url": "https://creativecommons.org/licenses/by/4.0/",
    },
    docs_url="/v1/docs",
    redoc_url="/v1/redoc",
    openapi_url="/v1/openapi.json",
)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded, cast(ExceptionHandler, _rate_limit_exceeded_handler)
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
    allow_credentials=False,
)

# ── Schemas ───────────────────────────────────────────────────────────────────

CropType = Literal["Wheat", "Rice", "Maize", "Soybean"]


class PredictRequest(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    crop: CropType = Field("Wheat")
    include_ndvi_tile: bool = Field(False)
    include_report: bool = Field(False)
    mc_passes: int = Field(
        20, ge=5, le=100, description="MC Dropout passes for confidence (5-100)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"lat": 31.5204, "lon": 74.3587, "crop": "Wheat", "mc_passes": 20}
            ]
        }
    }


class Era5Response(BaseModel):
    temp_c: float
    precip_mm_month: float
    source: str


class NdviStatus(BaseModel):
    label: str
    action: str
    alert_type: str


class PredictResponse(BaseModel):
    lat: float
    lon: float
    crop: str
    # Satellite
    ndvi: float
    ndvi_status: NdviStatus
    ndvi_tile_url: str | None = None
    # Climate
    era5: Era5Response
    # Yield
    yield_base_t_ha: float
    yield_adj_t_ha: float
    yield_delta_t_ha: float
    # Uncertainty — real MC Dropout
    confidence_pct: float
    yield_std_t_ha: float
    ci_95_lower: float
    ci_95_upper: float
    # Carbon
    carbon_mg_c_ha: float
    carbon_fraction: float
    # Meta
    model_name: str
    model_version: str
    gee_mode: str  # "live" | "demo"
    inference_ms: float
    generated_utc: str
    report: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    gee_ready: bool
    model_name: str
    model_version: str
    environment: str
    timestamp_utc: str


class CropInfo(BaseModel):
    name: str
    temp_K: float
    moisture: float
    ndvi_scale: float


# ── Router ────────────────────────────────────────────────────────────────────

v1 = APIRouter(prefix="/v1")


@v1.get("/", tags=["Meta"])
async def root() -> dict[str, str]:
    return {
        "name": "TerraVision AI API",
        "version": MODEL_VERSION,
        "docs": "/v1/docs",
        "health": "/v1/health",
        "github": "https://github.com/ahmedabbas52233-a11y/TerraVision-AI",
    }


@v1.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health() -> HealthResponse:
    mn = type(_MODEL).__name__ if _MODEL else "none"
    return HealthResponse(
        status="ok" if (_MODEL_READY and _GEE_READY) else "degraded",
        model_ready=_MODEL_READY,
        gee_ready=_GEE_READY,
        model_name=mn,
        model_version=MODEL_VERSION,
        environment=_ENV,
        timestamp_utc=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )


@v1.get(
    "/crops",
    response_model=list[CropInfo],
    tags=["Reference"],
    dependencies=[Depends(require_api_key)],
)
async def crops() -> list[CropInfo]:
    return [
        CropInfo(
            name=n,
            temp_K=float(p["temp_K"]),
            moisture=float(p["moisture"]),
            ndvi_scale=float(p["ndvi_scale"]),
        )
        for n, p in CROP_PARAMS.items()
    ]


@v1.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Inference"],
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("30/minute")
async def predict(request: Request, req: PredictRequest) -> PredictResponse:
    if not _MODEL_READY or _MODEL is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model not loaded. "
                "Commit models/terravision_v1.pth and redeploy, "
                "or run `python train.py` and restart."
            ),
        )

    t0 = time.perf_counter()
    log.info(
        "Inference lat=%.4f lon=%.4f crop=%s v2=%s gee=%s",
        req.lat,
        req.lon,
        req.crop,
        _IS_V2,
        _GEE_READY,
    )

    # ── Parallel GEE calls ────────────────────────────────────────────────────
    if _IS_V2:
        feat_coro = asyncio.to_thread(get_live_features_v2, req.lat, req.lon, req.crop)
    else:
        feat_coro = asyncio.to_thread(get_live_features, req.lat, req.lon, req.crop)

    features, era5, ndvi_tile_url = await asyncio.gather(
        feat_coro,
        asyncio.to_thread(get_era5_features, req.lat, req.lon),
        (
            asyncio.to_thread(get_ndvi_tile_url, req.lat, req.lon)
            if req.include_ndvi_tile
            else _noop_str()
        ),
    )

    # ── Inference + MC Dropout confidence ─────────────────────────────────────
    def _infer() -> tuple[float, float, float, ConfidenceResult]:
        if _IS_V2:
            tensor = torch.tensor([features], dtype=torch.float32)
        else:
            tensor = torch.tensor([features], dtype=torch.float32)

        with torch.no_grad():
            raw: float = _MODEL(tensor).item()  # type: ignore[union-attr]

        if _IS_V2:
            ndvi_val = float(cast(list[list[float]], features)[0][0])
        else:
            ndvi_val = float(cast(list[float], features)[0])

        ybase = compute_yield(raw, ndvi_val, req.crop)
        yadj = era5_yield_adjustment(
            ybase, era5["temp_c"], era5["precip_mm_month"], req.crop
        )
        conf = mc_dropout_confidence(_MODEL, tensor, n_passes=req.mc_passes)  # type: ignore
        return ndvi_val, ybase, yadj, conf

    ndvi, yield_base, yield_adj, conf = await asyncio.to_thread(_infer)

    carbon = yield_adj * CARBON_FRACTION
    lbl, act, alrt = ndvi_status(ndvi)
    ms = (time.perf_counter() - t0) * 1_000
    gee_mode = "live" if _GEE_READY else "demo"

    log.info(
        "Done — NDVI=%.4f yield=%.2f conf=%.1f%% mode=%s in %.0f ms",
        ndvi,
        yield_adj,
        conf["confidence_pct"],
        gee_mode,
        ms,
    )

    report_txt: str | None = None
    if req.include_report:
        report_txt = build_report(
            req.lat,
            req.lon,
            req.crop,
            ndvi,
            yield_base,
            yield_adj,
            carbon,
            lbl,
            act,
            era5,
            conf,
        )

    return PredictResponse(
        lat=req.lat,
        lon=req.lon,
        crop=req.crop,
        ndvi=round(ndvi, 4),
        ndvi_status=NdviStatus(label=lbl, action=act, alert_type=alrt),
        ndvi_tile_url=ndvi_tile_url,
        era5=Era5Response(
            temp_c=era5["temp_c"],
            precip_mm_month=era5["precip_mm_month"],
            source=era5["source"],
        ),
        yield_base_t_ha=round(yield_base, 3),
        yield_adj_t_ha=round(yield_adj, 3),
        yield_delta_t_ha=round(yield_adj - yield_base, 3),
        confidence_pct=conf["confidence_pct"],
        yield_std_t_ha=conf["std_yield"],
        ci_95_lower=conf["ci_95_lower"],
        ci_95_upper=conf["ci_95_upper"],
        carbon_mg_c_ha=round(carbon, 3),
        carbon_fraction=CARBON_FRACTION,
        model_name=type(_MODEL).__name__,  # type: ignore[union-attr]
        model_version=MODEL_VERSION,
        gee_mode=gee_mode,
        inference_ms=round(ms, 1),
        generated_utc=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        report=report_txt,
    )


app.include_router(v1)


@app.get("/", include_in_schema=False)
async def _root() -> RedirectResponse:
    return RedirectResponse(url="/v1/")
