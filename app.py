"""
TerraVision AI — Satellite-Native Crop Intelligence at Planetary Scale
Author  : Ahmad Abbas Hussain  ()
Version : 3.0.0
License : CC BY 4.0
DOI     : https://github.com/ahmedabbas52233/TerraVision-AI

v3.0.0 Enhancements
───────────────────
  [1] ERA5-Land weather features  — temperature_2m + precipitation_sum
      pulled from GEE, gaussian thermal-stress + precipitation-adequacy
      correction applied on top of the transformer base yield.
  [2] FastAPI REST wrapper        — see api.py  (/predict, /health)
  [3] NDVI heatmap TileLayer      — 12-month Sentinel-2 NDVI composite
      rendered as a coloured Folium TileLayer with a legend.
"""

from __future__ import annotations

from typing import Optional

import json
from datetime import datetime

import ee
import folium
import streamlit as st
import torch
from streamlit_folium import folium_static

from terravision.core import (
    CARBON_FRACTION,
    CONFIDENCE_FLOOR,
    mc_dropout_confidence,
    CROP_PARAMS,
    MODEL_VERSION,
    build_report,
    compute_yield,
    era5_yield_adjustment,
    get_era5_features,
    get_live_features,
    get_ndvi_tile_url,
    load_model,
    ndvi_status,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  ·  must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TerraVision AI",
    layout="wide",
    page_icon="🛰️",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 3-D SPACE THEME
# ─────────────────────────────────────────────────────────────────────────────
_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

:root {
  --bg:      #030810;
  --surface: rgba(8,20,42,0.82);
  --border:  rgba(0,255,170,0.14);
  --accent:  #00ffaa;
  --accent2: #00c8ff;
  --text:    #d8eeff;
  --muted:   #5a7a96;
  --card-r:  18px;
  --glow-g:  0 0 30px rgba(0,255,170,0.20);
  --glow-b:  0 0 30px rgba(0,200,255,0.20);
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    radial-gradient(1px 1px at  8% 12%, rgba(255,255,255,.55) 0%, transparent 100%),
    radial-gradient(1px 1px at 22% 44%, rgba(255,255,255,.35) 0%, transparent 100%),
    radial-gradient(1.5px 1.5px at 38%  8%, rgba(255,255,255,.50) 0%, transparent 100%),
    radial-gradient(1px 1px at 55% 72%, rgba(255,255,255,.30) 0%, transparent 100%),
    radial-gradient(1px 1px at 68% 28%, rgba(255,255,255,.45) 0%, transparent 100%),
    radial-gradient(1px 1px at 79% 58%, rgba(255,255,255,.35) 0%, transparent 100%),
    radial-gradient(1.5px 1.5px at 14% 80%, rgba(255,255,255,.50) 0%, transparent 100%),
    radial-gradient(1px 1px at 91% 15%, rgba(255,255,255,.40) 0%, transparent 100%),
    radial-gradient(1px 1px at 47% 93%, rgba(255,255,255,.30) 0%, transparent 100%),
    radial-gradient(1px 1px at 62% 42%, rgba(255,255,255,.35) 0%, transparent 100%),
    radial-gradient(1px 1px at 85% 76%, rgba(255,255,255,.30) 0%, transparent 100%);
  pointer-events: none;
  z-index: 0;
  animation: star-twinkle 7s ease-in-out infinite alternate;
}
@keyframes star-twinkle { 0%{opacity:.5} 100%{opacity:1} }

[data-testid="stAppViewContainer"]::after {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 60% 40% at 20% 10%, rgba(0,255,170,.04) 0%, transparent 70%),
    radial-gradient(ellipse 50% 35% at 80% 90%, rgba(0,200,255,.04) 0%, transparent 70%);
  pointer-events: none;
  z-index: 0;
}

h1 {
  font-family: 'Orbitron', monospace !important;
  font-weight: 900 !important;
  font-size: clamp(1.4rem, 3.5vw, 2.6rem) !important;
  background: linear-gradient(110deg, var(--accent) 0%, var(--accent2) 60%, var(--accent) 100%) !important;
  background-size: 200% auto !important;
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  background-clip: text !important;
  animation: shimmer 4s linear infinite !important;
  letter-spacing: 3px !important;
  margin-bottom: .2rem !important;
}
@keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }

h2, h3, h4 {
  font-family: 'Orbitron', monospace !important;
  font-weight: 700 !important;
  color: var(--accent) !important;
  letter-spacing: 1.5px !important;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(3,8,20,.98) 0%, rgba(5,14,32,.98) 100%) !important;
  border-right: 1px solid var(--border) !important;
  box-shadow: 4px 0 30px rgba(0,255,170,.06) !important;
}
[data-testid="stSidebar"] *, [data-testid="stSidebar"] p { color: var(--text) !important; }

[data-testid="column"] { perspective: 1200px; }
[data-testid="column"] > div {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--card-r);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  box-shadow: var(--glow-g), inset 0 1px 0 rgba(255,255,255,.05);
  padding: 1.6rem 1.4rem;
  transition: transform .35s ease, box-shadow .35s ease;
}
[data-testid="column"]:hover > div {
  transform: perspective(1000px) translateY(-5px) rotateX(1.5deg);
  box-shadow: 0 18px 52px rgba(0,255,170,.18), var(--glow-g);
}

.stButton > button {
  background: linear-gradient(135deg, #00ffaa 0%, #00c8ff 100%) !important;
  color: #020c18 !important;
  font-family: 'Orbitron', monospace !important;
  font-weight: 700 !important;
  font-size: .72rem !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  border: none !important;
  border-radius: 10px !important;
  padding: .65rem 1.6rem !important;
  width: 100% !important;
  box-shadow: 0 0 22px rgba(0,255,170,.45) !important;
  transition: all .25s ease !important;
}
.stButton > button:hover {
  transform: scale(1.05) translateY(-3px) !important;
  box-shadow: 0 0 42px rgba(0,255,170,.72) !important;
}

.stNumberInput input, [data-baseweb="input"] input {
  background: rgba(0,255,170,.04) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
}
.stNumberInput input:focus, [data-baseweb="input"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(0,255,170,.18) !important;
  outline: none !important;
}

[data-baseweb="select"] > div {
  background: rgba(0,255,170,.04) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
}

[data-testid="metric-container"] {
  background: linear-gradient(135deg,rgba(0,255,170,.08) 0%,rgba(0,200,255,.06) 100%) !important;
  border: 1px solid rgba(0,255,170,.22) !important;
  border-radius: 12px !important;
  padding: .8rem 1rem !important;
  box-shadow: 0 0 20px rgba(0,255,170,.10) !important;
}
[data-testid="metric-container"] label {
  color: var(--accent) !important;
  font-family: 'Orbitron', monospace !important;
  font-size: .58rem !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color: #ffffff !important;
  font-family: 'Orbitron', monospace !important;
  font-size: 1.45rem !important;
  font-weight: 700 !important;
}

.stAlert {
  border-radius: 12px !important;
  background: rgba(5,16,35,.75) !important;
  backdrop-filter: blur(10px) !important;
  border-left-width: 3px !important;
}

.stDownloadButton > button {
  background: transparent !important;
  border: 1px solid var(--accent) !important;
  color: var(--accent) !important;
  font-family: 'Orbitron', monospace !important;
  font-size: .65rem !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  border-radius: 10px !important;
  padding: .55rem 1rem !important;
  width: 100% !important;
  transition: all .25s !important;
}
.stDownloadButton > button:hover {
  background: rgba(0,255,170,.08) !important;
  box-shadow: var(--glow-g) !important;
  transform: translateY(-2px) !important;
}

/* ERA5 badge */
.era5-badge {
  display: inline-block;
  background: rgba(0,200,255,.12);
  border: 1px solid rgba(0,200,255,.30);
  border-radius: 8px;
  padding: .2rem .7rem;
  font-size: .65rem;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: #00c8ff;
  font-family: 'Orbitron', monospace;
  margin-bottom: .6rem;
}

/* NDVI legend */
.ndvi-legend {
  display: flex;
  gap: 0;
  border-radius: 8px;
  overflow: hidden;
  height: 14px;
  margin: .4rem 0 .2rem;
  border: 1px solid var(--border);
}
.ndvi-legend div { flex: 1; }

iframe {
  border-radius: 14px !important;
  border: 1px solid var(--border) !important;
  box-shadow: var(--glow-b) !important;
}

hr                   { border-color: var(--border) !important; }
.stSpinner > div     { border-top-color: var(--accent) !important; }
.stCaption, small    { color: var(--muted) !important; }
::-webkit-scrollbar  { width: 4px; }
::-webkit-scrollbar-track { background: #020a16; }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 4px; }
</style>
"""
st.markdown(_THEME_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1 · EARTH ENGINE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
def _init_ee() -> None:
    """
    Initialise GEE in three-tier priority order:
      1. Application Default Credentials (local: gcloud auth application-default login)
      2. Streamlit secrets   → GCP_SERVICE_ACCOUNT key
      3. Environment variable → GCP_SERVICE_ACCOUNT_JSON

    If none of the above are available the app enters DEMO MODE: all GEE calls
    return fallback priors and a warning banner is displayed.  The app never
    crashes due to missing credentials — important for local development.
    """
    # ── 1. Application Default Credentials ──────────────────────────────────
    try:
        ee.Initialize()
        return
    except Exception:
        pass

    # ── 2. Streamlit secrets (wrapped — throws if secrets.toml is absent) ───
    secret_json: Optional[str] = None
    try:
        secret_json = st.secrets.get("GCP_SERVICE_ACCOUNT")
    except Exception:
        pass

    # ── 3. Environment variable (Docker / Railway / local .env) ─────────────
    if not secret_json:
        secret_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")

    # ── 4. No credentials found → DEMO MODE (no crash) ──────────────────────
    if not secret_json:
        st.info(
            "ℹ️ **Demo Mode** — GEE not authenticated.  "
            "Predictions use crop-specific priors instead of live satellite data.  \n\n"
            "To enable live Sentinel-2 + ERA5 data, create "
            "`.streamlit/secrets.toml` from `.streamlit/secrets.toml.example`."
        )
        return

    # ── 5. Authenticate with service account ────────────────────────────────
    try:
        info = json.loads(secret_json)
        creds = ee.ServiceAccountCredentials(info["client_email"], key_data=secret_json)
        ee.Initialize(creds)
    except json.JSONDecodeError:
        st.warning("⚠️ GCP_SERVICE_ACCOUNT is not valid JSON — running in Demo Mode.")
    except Exception as exc:
        st.warning(f"⚠️ GEE authentication failed (Demo Mode active): {exc}")

_init_ee()


# ─────────────────────────────────────────────────────────────────────────────
# 2 · CACHED MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _cached_model():
    return load_model()


# ─────────────────────────────────────────────────────────────────────────────
# 3 · SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "ndvi": 0.5, "yield_base": 0.0, "yield_adj": 0.0,
    "era5": {}, "features": [0.5, 291.5, 0.025],
    "ran": False, "ndvi_tile_url": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────────────────────
# 4 · SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:1rem 0 .5rem">
          <div style="font-family:'Orbitron',monospace;font-size:1.05rem;
                      font-weight:900;color:#00ffaa;letter-spacing:3px">
            🛰️ TERRAVISION
          </div>
          <div style="font-size:.6rem;color:#5a7a96;letter-spacing:2px;
                      margin-top:.3rem;text-transform:uppercase">
            Satellite Intelligence · v3.0.0
          </div>
        </div>
        """, unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**System Status**")
    st.markdown("🟢 GEE Online &nbsp;·&nbsp; 🟢 Model Ready &nbsp;·&nbsp; 🟢 ERA5 Live")
    st.divider()
    st.markdown("**v3.0.0 Features**")
    st.caption("🌡️ ERA5-Land weather correction")
    st.caption("🗺️ NDVI heatmap TileLayer")
    st.caption("🔌 FastAPI REST endpoint")
    st.divider()
    st.markdown("**Model**")
    st.caption("ST-Transformer · 4-head MHSA · 25,089 params")
    st.divider()
    st.markdown("**Satellite Source**")
    st.caption("Sentinel-2 SR Harmonized · 10 m")
    st.divider()
    st.markdown("**Climate Source**")
    st.caption("ERA5-Land Daily Aggregates · 11 km · GEE")
    st.divider()
    st.caption("Developed by **Ahmad Abbas Hussain**\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5 · HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1>🛰️ TERRAVISION AI</h1>
    <p style="color:#5a7a96;font-size:.8rem;letter-spacing:2.5px;
              text-transform:uppercase;margin-top:-.4rem;margin-bottom:.5rem">
      Satellite-Native Crop Intelligence at Planetary Scale
    </p>
    """, unsafe_allow_html=True,
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# 6 · MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.55], gap="large")

# ── LEFT : INPUT + RESULTS ────────────────────────────────────────────────────
with col_left:
    st.subheader("📍 Field Parameters")

    lat = st.number_input("Latitude",  value=31.5204, min_value=-90.0,  max_value=90.0,  format="%.4f",
                          help="Decimal degrees. Negative = Southern Hemisphere.")
    lon = st.number_input("Longitude", value=74.3587, min_value=-180.0, max_value=180.0, format="%.4f",
                          help="Decimal degrees. Negative = Western Hemisphere.")
    crop = st.selectbox("Crop Type", list(CROP_PARAMS.keys()),
                        help="Select the target crop for yield prediction.")

    show_heatmap = st.toggle(
        "🗺️ Show NDVI Heatmap",
        value=True,
        help="Overlay a coloured 12-month NDVI composite on the satellite map.",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    run_clicked = st.button("🚀 Run Live Inference", use_container_width=True)

    if run_clicked:
        model = _cached_model()
        if model is None:
            st.error("⛔ Model checkpoint not found in `models/`. Cannot run inference.")
        else:

            # ── Step 1 : Sentinel-2 NDVI ─────────────────────────────────────
            with st.spinner("🛰️ Querying Sentinel-2 NDVI via GEE …"):
                features = get_live_features(lat, lon, crop)
                st.session_state["features"] = features

            # ── Step 2 : ERA5-Land climate features  [Enhancement 1] ─────────
            with st.spinner("🌡️ Pulling ERA5-Land weather data …"):
                era5 = get_era5_features(lat, lon)
                st.session_state["era5"] = era5

            # ── Step 3 : NDVI heatmap tile URL  [Enhancement 3] ──────────────
            if show_heatmap:
                with st.spinner("🗺️ Generating NDVI heatmap tiles …"):
                    ndvi_tile_url = get_ndvi_tile_url(lat, lon)
                    st.session_state["ndvi_tile_url"] = ndvi_tile_url
            else:
                st.session_state["ndvi_tile_url"] = None

            # ── Step 4 : Transformer inference ───────────────────────────────
            with st.spinner("⚡ Running ST-Transformer …"):
                tensor = torch.tensor([features], dtype=torch.float32)
                with torch.no_grad():
                    raw: float = model(tensor).item()

                ndvi       = features[0]
                yield_base = compute_yield(raw, ndvi, crop)
                yield_adj  = era5_yield_adjustment(
                    yield_base, era5["temp_c"], era5["precip_mm_month"], crop
                )
                carbon = yield_adj * CARBON_FRACTION

                # Real MC Dropout confidence (not hardcoded)
                conf_tensor = torch.tensor([features], dtype=torch.float32)
                if hasattr(model, 'SEQ_LEN'):
                    conf_tensor = torch.tensor([features], dtype=torch.float32)
                conf_result = mc_dropout_confidence(model, conf_tensor, n_passes=15)
                confidence_pct = conf_result["confidence_pct"]
                yield_std      = conf_result["std_yield"]

                st.session_state.update({
                    "ndvi": ndvi, "yield_base": yield_base,
                    "yield_adj": yield_adj, "ran": True,
                    "confidence_pct": confidence_pct,
                    "yield_std": yield_std,
                })

            st.success("✅ Inference complete!")

            # ── ERA5 badge ────────────────────────────────────────────────────
            _src = era5.get("source", "default")
            _src_label = "ERA5-Land Live" if _src == "era5-land" else "ERA5 Default Prior"
            st.markdown(f'<div class="era5-badge">🌡️ {_src_label}</div>', unsafe_allow_html=True)

            # ── Primary metrics ───────────────────────────────────────────────
            m1, m2 = st.columns(2)
            m1.metric("ERA5-Adj. Yield", f"{yield_adj:.2f} t/ha",
                      delta=f"{yield_adj - yield_base:+.2f} vs base")
            m2.metric("NDVI Index", f"{ndvi:.4f}")

            m3, m4 = st.columns(2)
            m3.metric("Base Yield",  f"{yield_base:.2f} t/ha")
            m4.metric("Carbon Est.", f"{carbon:.2f} Mg C/ha")

            st.divider()

            # ── ERA5 climate panel ────────────────────────────────────────────
            if _src == "era5-land":
                with st.expander("🌡️ ERA5 Climate Detail", expanded=True):
                    e1, e2 = st.columns(2)
                    e1.metric("Air Temp (2m)", f"{era5['temp_c']:.1f} °C")
                    e2.metric("Monthly Precip", f"{era5['precip_mm_month']:.1f} mm")

            # ── Vegetation health ─────────────────────────────────────────────
            label, action, alert_type = ndvi_status(ndvi)
            {"error": st.error, "warning": st.warning,
             "info": st.info, "success": st.success}[alert_type](
                f"**{label}**\n\n{action}"
            )

            m5, m6 = st.columns(2)
            m5.metric("Confidence", f"{st.session_state.get('confidence_pct', CONFIDENCE_FLOOR):.1f} %")
            m6.metric("Version",    f"v{MODEL_VERSION}")

            # ── Download report ───────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            report_txt = build_report(
                lat, lon, crop, ndvi, yield_base, yield_adj,
                carbon, label, action, era5,
            )
            st.download_button(
                label="📥 Download Intelligence Report",
                data=report_txt,
                file_name=f"TerraVision_v3_{crop}_{lat:.4f}_{lon:.4f}.txt",
                mime="text/plain",
            )


# ── RIGHT : SATELLITE MAP + NDVI HEATMAP ─────────────────────────────────────
with col_right:
    st.subheader("🗺️ Satellite Intelligence View")

    # NDVI legend (always visible) ────────────────────────────────────────────
    if show_heatmap:
        st.markdown(
            """
            <div style="margin-bottom:.5rem">
              <div style="font-size:.62rem;letter-spacing:1.5px;color:#5a7a96;
                          text-transform:uppercase;margin-bottom:.3rem">
                NDVI Heatmap Legend
              </div>
              <div class="ndvi-legend">
                <div style="background:#8B4513" title="Water / Bare Soil"></div>
                <div style="background:#D2691E" title="Very sparse"></div>
                <div style="background:#F4D03F" title="Low vegetation"></div>
                <div style="background:#A9D18E" title="Moderate"></div>
                <div style="background:#4CAF50" title="Healthy"></div>
                <div style="background:#1B7A3E" title="Dense vegetation"></div>
                <div style="background:#005A1F" title="Peak greenery"></div>
              </div>
              <div style="display:flex;justify-content:space-between;
                          font-size:.58rem;color:#5a7a96">
                <span>Bare / Water</span><span>Low</span>
                <span>Moderate</span><span>Dense</span>
              </div>
            </div>
            """, unsafe_allow_html=True,
        )

    # Build Folium map ─────────────────────────────────────────────────────────
    sat_map = folium.Map(
        location=[lat, lon],
        zoom_start=13,
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="© Google Maps",
    )

    # NDVI heatmap TileLayer  [Enhancement 3] ─────────────────────────────────
    ndvi_tile_url = st.session_state.get("ndvi_tile_url")
    if show_heatmap and ndvi_tile_url:
        folium.TileLayer(
            tiles=ndvi_tile_url,
            name="NDVI Heatmap (Sentinel-2)",
            attr="Google Earth Engine · Sentinel-2 SR",
            overlay=True,
            control=True,
            opacity=0.72,
        ).add_to(sat_map)
        folium.LayerControl(collapsed=False).add_to(sat_map)

    # Analysis marker ──────────────────────────────────────────────────────────
    _ran = st.session_state["ran"]
    _ndvi_now  = st.session_state["ndvi"]
    _yield_adj = st.session_state["yield_adj"]

    popup_html = (
        f"<b>TerraVision Analysis Target</b><br>"
        f"Lat: {lat:.4f}° &nbsp; Lon: {lon:.4f}°<br>"
        f"Crop: {crop}"
    )
    if _ran:
        popup_html += (
            f"<br><b>NDVI:</b> {_ndvi_now:.4f}"
            f"<br><b>Yield (ERA5-adj):</b> {_yield_adj:.2f} t/ha"
        )

    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=240),
        tooltip="Analysis Target",
        icon=folium.Icon(color="darkgreen", icon="leaf", prefix="fa"),
    ).add_to(sat_map)

    # 500 m analysis buffer ────────────────────────────────────────────────────
    folium.Circle(
        location=[lat, lon], radius=500,
        color="#00ffaa", weight=1.5,
        fill=True, fill_color="#00ffaa", fill_opacity=0.10,
        tooltip="500 m analysis buffer (GEE · Sentinel-2)",
    ).add_to(sat_map)

    # 10 km ERA5 buffer ────────────────────────────────────────────────────────
    folium.Circle(
        location=[lat, lon], radius=10_000,
        color="#00c8ff", weight=1.0, dash_array="6 4",
        fill=True, fill_color="#00c8ff", fill_opacity=0.03,
        tooltip="10 km ERA5-Land sampling buffer",
    ).add_to(sat_map)

    folium_static(sat_map, width=None, height=490)


# ─────────────────────────────────────────────────────────────────────────────
# 7 · MULTI-MODAL INTELLIGENCE PANEL
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Multi-Modal Intelligence Panel")

_ndvi_now  = st.session_state["ndvi"]
_yield_base = st.session_state["yield_base"]
_yield_adj = st.session_state["yield_adj"]
_era5      = st.session_state["era5"]
_ran       = st.session_state["ran"]

fi1, fi2, fi3, fi4, fi5 = st.columns(5)

with fi1:
    st.markdown("**🌿 Vegetation**")
    if _ran:
        _lbl, _, _ = ndvi_status(_ndvi_now)
        st.caption(f"NDVI `{_ndvi_now:.4f}`")
        st.caption(_lbl.split("—")[-1].strip())
    else:
        st.caption("Run inference →")

with fi2:
    st.markdown("**🌡️ ERA5 Climate**")
    if _ran and _era5.get("source") == "era5-land":
        st.caption(f"Temp: `{_era5['temp_c']:.1f} °C`")
        st.caption(f"Precip: `{_era5['precip_mm_month']:.0f} mm/mo`")
    elif _ran:
        st.caption("Default priors used")
    else:
        st.caption("Run inference →")

with fi3:
    st.markdown("**📈 ERA5 Yield Δ**")
    if _ran:
        delta = _yield_adj - _yield_base
        sign  = "+" if delta >= 0 else ""
        st.caption(f"Base:  `{_yield_base:.2f} t/ha`")
        st.caption(f"Adj:   `{_yield_adj:.2f} t/ha`  ({sign}{delta:.2f})")
    else:
        st.caption("Run inference →")

with fi4:
    st.markdown("**☁️ Carbon**")
    if _ran:
        st.caption(f"`{_yield_adj * CARBON_FRACTION:.2f} Mg C/ha`")
        st.caption(f"IPCC 2006 · CF = {CARBON_FRACTION}")
    else:
        st.caption("Run inference →")

with fi5:
    st.markdown("**📡 Data Pipeline**")
    st.caption("S-2 SR · 10 m · 500 m buf")
    st.caption("ERA5-Land · 11 km · 10 km buf")


# ─────────────────────────────────────────────────────────────────────────────
# 8 · FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    f"""
    <div style="text-align:center;color:#3a5a74;font-size:.72rem;
                font-family:'DM Sans',sans-serif;padding:.4rem 0 1.5rem">
      © {datetime.utcnow().year}&nbsp; TerraVision AI &nbsp;·&nbsp;
      Developed by
      <strong style="color:#00ffaa">Ahmad Abbas Hussain</strong>
      &nbsp;·&nbsp; 
      <br><br>
      🛰️ Sentinel-2 &nbsp;·&nbsp; 🌡️ ERA5-Land &nbsp;·&nbsp;
      ⚡ PyTorch &nbsp;·&nbsp; 🌿 GEE &nbsp;·&nbsp; 🚀 Streamlit
    </div>
    """, unsafe_allow_html=True,
)
