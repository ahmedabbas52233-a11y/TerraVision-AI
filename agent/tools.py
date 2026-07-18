"""TerraVision Agent — tool definitions.

Wraps the real TerraVision AI /v1/predict endpoint (api.py) as an LLM-callable tool.
"""

from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("TERRAVISION_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("TERRAVISION_API_KEY", "dev-insecure-key")

CROP_TYPES = ["Wheat", "Rice", "Maize", "Soybean"]

# ---- Tool schema (what the LLM sees) ----

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "predict_crop_yield",
            "description": (
                "Get satellite-based crop yield prediction for a field location. "
                "Returns NDVI (vegetation health), climate data (temperature/precipitation), "
                "predicted yield with a 95% confidence interval, and carbon estimate. "
                "Use this whenever the user asks about crop health, yield outlook, or field "
                "conditions for a specific location and crop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "Latitude, -90 to 90"},
                    "lon": {"type": "number", "description": "Longitude, -180 to 180"},
                    "crop": {
                        "type": "string",
                        "enum": CROP_TYPES,
                        "description": "Crop type to predict yield for",
                    },
                    "include_report": {
                        "type": "boolean",
                        "description": "Whether to include a full human-readable report string",
                        "default": False,
                    },
                },
                "required": ["lat", "lon", "crop"],
            },
        },
    }
]


def predict_crop_yield(
    lat: float, lon: float, crop: str, include_report: bool = False
) -> dict:
    """Call the real TerraVision /v1/predict endpoint and return its JSON result."""
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/predict",
            headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
            json={
                "lat": lat,
                "lon": lon,
                "crop": crop,
                "include_report": include_report,
                "mc_passes": 20,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()

        # Surface demo/mock flags so the agent (and eval) can tell fake from real data
        notes = []
        if data.get("gee_mode") == "demo":
            notes.append(
                "GEE running in demo mode (free-tier limits) — NDVI is mock data."
            )
        if data.get("model_mode") == "mock":
            notes.append(
                "Model checkpoint not loaded — yield prediction is a mock "
                "placeholder, not real inference."
            )
        if notes:
            data["_note"] = " ".join(notes)

        return data

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        body = e.response.text if e.response is not None else str(e)
        return {"error": f"API error: {status_code} — {body}"}

    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}


TOOL_FUNCTIONS = {
    "predict_crop_yield": predict_crop_yield,
}
