"""TerraVision Agent — tool definitions.

Two tools:
  - geocode_location: resolves a place name to real coordinates (OpenStreetMap Nominatim,
    free, no API key). Used so the agent never guesses coordinates from its own memory.
  - predict_crop_yield: wraps the real TerraVision AI /v1/predict endpoint (api.py).
"""

from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("TERRAVISION_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("TERRAVISION_API_KEY", "dev-insecure-key")

CROP_TYPES = ["Wheat", "Rice", "Maize", "Soybean"]

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Nominatim's usage policy requires a descriptive User-Agent and caps free use at
# ~1 request/second — fine for a demo/portfolio agent, not for production scale.
_NOMINATIM_HEADERS = {"User-Agent": "TerraVisionAgent/1.0 (portfolio project)"}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "geocode_location",
            "description": (
                "Resolve a place name (city, town, village, region — anywhere in the "
                "world) to real latitude/longitude coordinates using OpenStreetMap. "
                "ALWAYS call this first when the user names a place instead of giving "
                "coordinates directly — never guess coordinates from memory, even for "
                "well-known cities, since place names can be ambiguous (e.g. multiple "
                "cities share the same name in different countries)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "place_name": {
                        "type": "string",
                        "description": (
                            "The place name as the user wrote it. Include country if the "
                            "user gave one, to disambiguate (e.g. 'Multan, Pakistan')."
                        ),
                    }
                },
                "required": ["place_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_crop_yield",
            "description": (
                "Get satellite-based crop yield prediction for a field location. "
                "Returns NDVI (vegetation health), climate data (temperature/precipitation), "
                "predicted yield with a 95% confidence interval, and carbon estimate. "
                "Requires real lat/lon — call geocode_location first if you only have a "
                "place name."
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
    },
]


def geocode_location(place_name: str) -> dict:
    """Resolve a place name to coordinates via OpenStreetMap Nominatim (free, no key)."""
    if not place_name or not place_name.strip():
        return {"error": "place_name cannot be empty."}
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": place_name, "format": "json", "limit": 1},
            headers=_NOMINATIM_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json()
        if not results:
            return {
                "error": (
                    f"Could not find coordinates for {place_name!r}. "
                    "Try a more specific name (add a country or region)."
                )
            }
        top = results[0]
        return {
            "lat": float(top["lat"]),
            "lon": float(top["lon"]),
            "display_name": top.get("display_name", place_name),
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Geocoding request failed: {e}"}


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
    "geocode_location": geocode_location,
    "predict_crop_yield": predict_crop_yield,
}
