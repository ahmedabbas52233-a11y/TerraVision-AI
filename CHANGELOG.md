# Changelog

All notable changes to TerraVision AI are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-06-01

### Initial Public Release

- Spatio-Temporal Transformer (4-head MHSA, 25,089 parameters) for crop yield regression

- Live Sentinel-2 SR NDVI extraction via Google Earth Engine (10 m, 500 m buffer, <20 % cloud)

- ERA5-Land weather intelligence — 30-day temperature + precipitation with Gaussian thermal-stress
  and precipitation-adequacy yield correction

- FastAPI REST API with versioned routes (`/v1/predict`, `/v1/health`, `/v1/crops`),
  X-API-Key authentication, slowapi rate limiting (30 req/min), and OpenAPI docs at `/v1/docs`

- React SPA frontend (`frontend/index.html`) — Leaflet satellite map, live API calls,
  ERA5 climate panel, region presets, and report download; no build step required

- Streamlit web app with 3D glassmorphism UI, animated star-field, NDVI heatmap overlay,
  and downloadable intelligence reports

- NDVI heatmap TileLayer — 12-month Sentinel-2 composite with 7-stop colour palette and toggle

- `train.py` — full training pipeline with Adam, CosineAnnealingLR, early stopping (patience=20),
  and Monte Carlo Dropout uncertainty quantification (50 passes)

- 80 pytest tests (42 unit + 38 integration) with 80 % coverage enforced in CI

- X-API-Key header authentication on all prediction endpoints

- slowapi rate limiting — 30 requests/minute per IP

- CORS restricted to configurable allow-list in production; wildcard only in development

- All FastAPI routes are `async def`; GEE calls run via `asyncio.to_thread()`

- Multi-stage Dockerfile with non-root user (`terravision`) and `HEALTHCHECK`

- `docker-compose.yml` orchestrating FastAPI API + Streamlit app with health dependency

- `terravision/` Python package — all business logic separated from entry points

- `pyproject.toml` — unified project metadata, Black, Ruff, isort, pytest configuration

- `SECURITY.md` — vulnerability disclosure policy

- `LICENSE` — CC BY 4.0

- Demo Mode — app runs gracefully without GEE credentials using crop-specific priors

### Supported Crops

- Wheat, Rice, Maize, Soybean

### Architecture

- Data tier: Sentinel-2 SR Harmonized (10 m) + ERA5-Land Daily Aggregates (11 km) via GEE

- Inference tier: Spatio-Temporal Transformer, GELU activation, Xavier init, Dropout(0.10)

- API tier: FastAPI v1 — authenticated, rate-limited, versioned, async, CORS-hardened

- Frontend tier: React 18 SPA + Leaflet.js (no build step)

- Deploy tier: Streamlit Cloud + Railway + Vercel + GitHub Pages + Hugging Face Spaces

---

## [Unreleased]

### Planned

- ERA5 precipitation and temperature fed as additional dimensions into the transformer
  (currently applied as a post-hoc correction layer)

- Plotly 12-month NDVI time-series chart for trend analysis

- CSV batch-prediction mode — accept a file of lat/lon pairs, export a results spreadsheet

- GeoJSON field-boundary upload — aggregate NDVI inside a user-defined polygon

- FastAPI `/v1/predict/batch` endpoint for programmatic batch processing

- Confidence interval bands in yield output using Monte Carlo Dropout at inference time

- PostgreSQL table for inference history with `/v1/history` endpoint

- Multi-language UI — Arabic, Spanish, French
