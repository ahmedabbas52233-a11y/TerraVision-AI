# Contributing to TerraVision AI

Thank you for considering a contribution!
Every improvement — however small — helps the global agricultural-AI community.

---

## Before You Start

1. **Search existing issues** — your idea or bug may already be tracked.
2. **Open an issue first** for any non-trivial change so we can align before you invest time coding.
3. **Fork → Branch → PR** — never commit directly to `main`.

---

## Development Setup

```bash
git clone https://github.com/ahmedabbas52233/TerraVision-AI.git
cd TerraVision-AI
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your GEE service account JSON
streamlit run app.py
```

---

## Code Style

| Concern | Rule |
| --- | --- |
| Formatter | `black` (line length 90) |
| Linter | `ruff` |
| Types | Type-annotated function signatures required |
| Docstrings | Google style |
| Imports | `isort` — stdlib → third-party → local |

Run before pushing:

```bash
black --line-length 90 terravision/ api.py app.py train.py
ruff check terravision/ api.py app.py train.py
pytest tests/ --cov=terravision --cov=api
```

---

## Pull Request Checklist

- [ ] Branch name follows `feat/short-description` or `fix/short-description`
- [ ] Code passes `black` and `ruff` without warnings
- [ ] New functions have type annotations and docstrings
- [ ] `requirements.txt` updated if new dependencies are added
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Screenshots added to PR description for UI changes

---

## Good First Issues

Label: `good first issue`

- **ERA5 precipitation feature** — add `temperature_C` and `precipitation_mm` from ERA5
  as actual model input dimensions (currently post-hoc correction)

- **Plotly NDVI time-series** — chart 12-month NDVI trend for a coordinate

- **CSV batch mode** — accept a CSV of lat/lon pairs and export results

- **Multilingual UI** — Arabic, Spanish, or French sidebar translations

- **Unit tests for train.py** — `pytest` tests for `generate_dataset`, `compute_metrics`,
  `mc_dropout_predict`

---

## High-Impact Contributions

Label: `enhancement`

- **GeoJSON field boundaries** — let users upload a field polygon and aggregate NDVI inside

- **FastAPI batch endpoint** — `/v1/predict/batch` accepting a list of coordinates

- **Historical yield trend** — multi-year Sentinel-2 composites with animated Plotly chart

- **Confidence intervals** — expose MC Dropout std at inference time in API response

- **PostgreSQL inference log** — `/v1/history` endpoint backed by async SQLAlchemy

---
## Security
Please do not open public issues for security problems. Email ahmedabbas52233@gmail.com directly.

---
## License

By contributing, you agree your work is released under the same **CC BY 4.0** licence as this project.

---

Questions? Email [ahmedabbas52233@gmail.com](mailto:ahmedabbas52233@gmail.com)