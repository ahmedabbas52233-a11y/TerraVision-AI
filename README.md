# 🛰️ TerraVision AI

<div align="center">

**A satellite-native crop intelligence pipeline: a Spatio-Temporal Transformer for yield
forecasting, live Sentinel-2/ERA5 ingestion via Google Earth Engine, IPCC-aligned carbon
estimation, a production FastAPI backend, and an LLM tool-calling agent layer on top.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Google Earth Engine](https://img.shields.io/badge/Google_Earth_Engine-Sentinel--2_+_ERA5-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://earthengine.google.com/)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-F55036?style=for-the-badge)](https://groq.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey?style=for-the-badge)](https://creativecommons.org/licenses/by/4.0/)

*Developed by [Ahmad Abbas Hussain](mailto:ahmedabbas52233@gmail.com)*

</div>

---

## What this project actually is

TerraVision AI is a research-grade **prototype**, not a validated production system. It
demonstrates an end-to-end ML engineering pipeline:

1. A small Spatio-Temporal Transformer (~25K params) trained on **synthetic** NDVI/climate/yield
   data with a hand-specified generative formula (see [Known Issues](#-known-issues--honest-limitations)) — this
   is a proof-of-architecture, not a model validated against real-world yield ground truth.
2. A live satellite/climate data pipeline via Google Earth Engine (Sentinel-2, ERA5-Land),
   with a transparent, labeled fallback to demo data when GEE access is unavailable.
3. A production-shaped FastAPI backend: rate limiting, API-key auth, async I/O, Pydantic v2
   schemas, auto-generated OpenAPI docs, real MC Dropout uncertainty quantification.
4. **New:** an LLM tool-calling agent (Groq / Llama 3.3 70B) that lets you ask natural-language
   questions about a field, which calls the real pipeline as a tool and reasons over the actual
   returned numbers — with guardrails against fabrication and an automated eval suite.

Every claim below is described at the level of confidence it actually deserves. Where something
is a placeholder, a fallback, or a known gap, it's labeled as such rather than glossed over.

---

## 📋 Table of Contents

- [System Architecture](#-system-architecture)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [REST API](#-rest-api)
- [🤖 LLM Agent Layer](#-llm-agent-layer)
- [Carbon Sequestration Module](#-carbon-sequestration-module)
- [NDVI Health Classification](#-ndvi-health-classification)
- [Known Issues / Honest Limitations](#-known-issues--honest-limitations)
- [Security Notes](#-security-notes)
- [Version History](#-version-history)
- [Contributing](#-contributing)
- [Author](#-author)
- [License](#-license)
- [Disclaimer](#-disclaimer)

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  TerraVision AI — Multi-Tier Architecture                │
├──────────────────┬──────────────────────┬────────────────────────────────┤
│     TIER 1       │      TIER 2          │          TIER 3                │
│  Data Acquisition│  Inference Engine    │  Agent / Deployment            │
├──────────────────┼──────────────────────┼────────────────────────────────┤
│ Google Earth     │ ST-Transformer       │ FastAPI REST backend           │
│ Engine (GEE)     │ (TerraVision)        │ (auth, rate limit, async)      │
│ • Sentinel-2 SR  │ • Input Projection   │                                │
│   Harmonized     │ • MHSA (4 heads)     │ Groq / Llama 3.3 70B agent      │
│ • ERA5-Land      │ • GELU FFN + Dropout │ • predict_crop_yield tool       │
│ • Demo fallback  │ • MC Dropout conf.   │ • guardrails + eval suite       │
│   when GEE unset │ • Mock fallback if   │                                │
│                  │   no checkpoint      │ Streamlit web app (legacy UI)   │
└──────────────────┴──────────────────────┴────────────────────────────────┘
```

Both the real inference path and the GEE data path have **transparent fallback modes** —
`model_mode: "trained" | "mock"` and `gee_mode: "live" | "demo"` are returned on every API
response, so nothing is silently faked.

---

## 🧠 Model Architecture

```python
class TerraVisionTransformer(torch.nn.Module):
    """
    Input : (batch, 3) → [NDVI, temperature_K, soil_moisture]
    Output: (batch, 1) → raw model output, converted to yield (t/ha) by compute_yield()
    """
    def __init__(self, input_dim=3, model_dim=64):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, model_dim)
        self.attention  = torch.nn.MultiheadAttention(model_dim, num_heads=4, batch_first=True)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(model_dim, 128),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.10),
            torch.nn.Linear(128, 1),
        )
```

| Layer | Dimensions | Parameters |
|---|---|---|
| Input Projection (Linear) | 3 → 64 | 256 |
| Multi-Head Self-Attention | 64, 4 heads (d_k = 16) | 16,384 |
| FFN Layer 1 (Linear + GELU) | 64 → 128 | 8,320 |
| FFN Layer 2 (Linear) | 128 → 1 | 129 |
| **Total Trainable Parameters** | — | **25,089** |

A V2 variant (`TerraVisionTransformerV2`) extends this to a genuine 6-month time-series input
with temporal attention, rather than a single observation.

---

## ⚙️ Installation

```bash
# 1. Clone
git clone https://github.com/ahmedabbas52233-a11y/TerraVision-AI.git
cd TerraVision-AI

# 2. Install dependencies (use a virtualenv — see note below)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# edit .env: set TERRAVISION_API_KEY, GROQ_API_KEY (for the agent), etc.

# 4. (Optional) Authenticate Earth Engine — without this, the API runs in
#    transparent demo mode using labeled placeholder satellite data
earthengine authenticate

# 5. Run the API
python -m uvicorn api:app --reload
# → http://127.0.0.1:8000/v1/docs  (interactive OpenAPI docs)

# 6. (Optional) Run the legacy Streamlit UI
streamlit run app.py
```

**Windows note:** use `127.0.0.1` rather than `localhost` when configuring `TERRAVISION_BASE_URL`
— Windows can resolve `localhost` to IPv6 first, causing spurious connection failures against an
IPv4-only server.

---

## 🔌 REST API

Full OpenAPI docs auto-generated at **`/v1/docs`**.

```bash
curl http://127.0.0.1:8000/v1/health

curl -s -X POST http://127.0.0.1:8000/v1/predict \
  -H "X-API-Key: your-key-here" -H "Content-Type: application/json" \
  -d '{"lat":31.5204,"lon":74.3587,"crop":"Wheat","include_report":true}'
```

**Every response includes explicit mode flags** — no silent fallbacks:

```json
{
  "lat": 31.5204, "lon": 74.3587, "crop": "Wheat",
  "ndvi": 0.4821,
  "yield_adj_t_ha": 3.54,
  "confidence_pct": 85.0,
  "model_name": "TerraVisionTransformer",
  "model_mode": "mock",
  "gee_mode": "demo",
  "inference_ms": 427.0
}
```

---

## 🤖 LLM Agent Layer

An agentic layer on top of `/v1/predict`: ask natural-language questions about crop yield, and
the agent calls the real pipeline as a tool, then reasons over the actual returned data —
never fabricating NDVI, yield, or climate numbers.

**Stack:** Groq (Llama 3.3 70B) · raw OpenAI-compatible function-calling · Python

**Flow:** user question → LLM decides whether/how to call `predict_crop_yield` → real FastAPI
backend runs (satellite + climate + yield + uncertainty) → LLM synthesizes a grounded answer,
citing the tool's actual returned numbers.

### Guardrails
- Refuses unsupported crop types rather than guessing a substitute
- Asks for missing location/crop info instead of inventing coordinates
- Validates lat/lon/crop client-side before hitting the real API
- Won't retry an identical failing call twice
- Transparently surfaces `model_mode`/`gee_mode` caveats to the end user
- Distinguishes general knowledge questions (answered directly, no tool call) from
  data questions (require a tool call)

### Evaluation
`agent/eval.py` runs 5 automated test cases: unsupported input, missing information, invalid
parameters, non-data questions, and multi-location reasoning.

**Known limitation:** the invalid-coordinate case is intermittently flaky across repeated runs
(observed 4–5 / 5 passing). LLM-generated tool arguments don't always relay the user's raw input
verbatim — the model occasionally "corrects" malformed values before they reach client-side
validation, rather than passing them through as-is. A more robust fix would pre-parse numeric
fields from the user's raw text via regex, bypassing model-generated arguments for
validation-critical parameters. Documented here as a known class of tool-calling reliability
issue, not silently patched over.

### Run it
```bash
# Terminal 1
python -m uvicorn api:app --reload

# Terminal 2
python -m agent.loop "What's the wheat yield outlook near Faisalabad, 31.4, 73.1?"
python -m agent.eval   # full automated test suite → agent/eval_results.json
```
Requires `GROQ_API_KEY` in `.env` (free tier at console.groq.com).

---

## ☁️ Carbon Sequestration Module

**IPCC 2006 Volume 4 (AFOLU) formula:**
```
C_ag = ŷ × BCEF × CF ≈ ŷ × 0.47   (Mg C/ha)
```
| Symbol | Meaning | Value |
|---|---|---|
| `ŷ` | Predicted crop yield | t/ha |
| `BCEF` | Biomass Conversion and Extension Factor | ≈ 1.0 |
| `CF` | IPCC carbon fraction of dry matter | 0.47 |

This is a standard, published conversion factor applied to the model's yield output — the
formula itself is real and correctly cited; the yield feeding into it is currently
synthetic-data-trained (see Known Issues).

---

## 🌿 NDVI Health Classification

| NDVI Range | Classification | Recommended Action |
|---|---|---|
| NDVI < 0.20 | 🔴 Critical — Low Density | Immediate nitrogen-based soil enrichment |
| 0.20 ≤ NDVI < 0.30 | 🟠 Stressed Vegetation | Targeted fertiliser + irrigation audit |
| 0.30 ≤ NDVI < 0.60 | 🔵 Normal Growth Cycle | Standard practices; monitor in 14 days |
| NDVI ≥ 0.60 | 🟢 Optimal — High Activity | Maintain regime; begin harvest planning |

---

## ⚠️ Known Issues / Honest Limitations

- **Training data is synthetic.** `train.py` generates NDVI/temperature/moisture/yield via a
  hand-specified Gaussian formula, not real agricultural ground truth. Any confidence/accuracy
  figures are therefore properties of the synthetic generative process, not validated real-world
  performance. Treat this as a proof-of-architecture, not a benchmarked yield predictor.
- **`train.py` is currently out of sync with `core.py`.** A prior refactor renamed model layers
  and removed the `YIELD_MAX` constant / changed the `ConfidenceResult` shape without updating
  the training script. Committed checkpoints fail to load with shape/key mismatches. The API
  gracefully falls back to a clearly-labeled deterministic mock inference (`model_mode: "mock"`)
  rather than serving stale or silently-wrong predictions. **Fixing `train.py` and retraining
  against the current architecture is the top open task on this project.**
- **GEE project is not currently registered for Earth Engine access** (free-tier limitation);
  NDVI/satellite data falls back to labeled demo values (`gee_mode: "demo"`).
- **Agent eval flakiness** — see [LLM Agent Layer](#-llm-agent-layer) above.

---

## 🔒 Security Notes

- API key comparison uses constant-time comparison (`hmac.compare_digest`) to avoid timing
  side-channels.
- The insecure default API key (`dev-insecure-key`) is rejected outside `TERRAVISION_ENV=development`
  — the app refuses to start with the default key set in a non-development environment.
- CORS is locked to explicit origins outside development mode.
- `.env` is git-ignored; use `.env.example` as a template and never commit real secrets.
- Docker deployments should use `--env-file` rather than passing secrets as inline `-e` flags
  (inline flags leak into shell history and process listings).

---

## 📦 Experimental Setup

| Parameter | Value |
| --- | --- |
| Framework | PyTorch 2.x |
| Optimiser | Adam (lr = 1e-3, weight_decay = 1e-4) |
| Loss | MSE |
| Training data | Synthetic (see Known Issues) |
| Satellite source | Sentinel-2 SR Harmonized (GEE), demo fallback available |
| Agent LLM | Groq — Llama 3.3 70B |

---

## 🆕 Version History

**v3.0.0** — ERA5-Land climate correction, FastAPI REST wrapper, NDVI heatmap tile layer,
`core.py` refactor (shared logic for `app.py`/`api.py`). *(Introduced the train/core drift
documented above.)*
**v3.1.0 (this update)** — LLM tool-calling agent layer with guardrails and automated eval;
mock-inference fallback for graceful degradation; security hardening (constant-time key
comparison, fail-closed default key); README rewritten for accuracy.
**v2.0** — GELU activation, dropout, Xavier init, Soybean support, `weights_only=True` security
fix (CVE-2024-40715).

---

## 🤝 Contributing

**Top priority:** fix `train.py` to match the current `core.py` architecture and retrain against
it (see Known Issues). **Other good first issues:** real dataset integration (replacing synthetic
training data), CSV batch-processing mode, time-series NDVI charting, regex-based argument
pre-validation for the agent layer.

---

## 👤 Author

**Ahmad Abbas Hussain** — AI & Full-Stack Developer
[Email](mailto:ahmedabbas52233@gmail.com) · [LinkedIn](https://www.linkedin.com/in/ahmad-abbas-hussain-7000151a3/) · [GitHub](https://github.com/ahmedabbas52233-a11y)

---

## 📜 License

Licensed under **Creative Commons Attribution 4.0 International**. Free to share and adapt with
appropriate credit. Commercial enquiries: [ahmedabbas52233@gmail.com](mailto:ahmedabbas52233@gmail.com)

---

## ⚠️ Disclaimer

> **RESEARCH & EDUCATIONAL USE ONLY.** TerraVision AI is a research prototype trained on
> synthetic data. Yield predictions and carbon estimates are **not validated against real-world
> ground truth** and are not intended for operational agricultural decision-making, financial
> planning, insurance, or policy formulation. Software provided "AS IS" without warranty of any
> kind.
>
> — *Ahmad Abbas Hussain, 2026*
