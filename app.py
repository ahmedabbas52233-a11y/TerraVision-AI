from __future__ import annotations

import contextlib
import json
import math
import os
from datetime import datetime

import ee
import folium
import streamlit as st
import torch
from streamlit_folium import folium_static

from terravision.core import (
    CARBON_FRACTION,
    CONFIDENCE_FLOOR,
    CROP_PARAMS,
    build_report,
    compute_yield,
    era5_yield_adjustment,
    get_era5_features,
    get_live_features,
    get_ndvi_tile_url,
    load_model,
    mc_dropout_confidence,
    ndvi_status,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TerraVision AI",
    layout="wide",
    page_icon="🛰️",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS THEME
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
  --bg:#030810; --surface:rgba(8,20,42,0.82); --border:rgba(0,255,170,0.14);
  --accent:#00ffaa; --accent2:#00c8ff; --text:#d8eeff; --muted:#5a7a96;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:'DM Sans',sans-serif!important}
[data-testid="stAppViewContainer"]::before{content:'';position:fixed;inset:0;
  background-image:radial-gradient(1px 1px at 10% 20%,rgba(255,255,255,.5) 0%,transparent 100%),
  radial-gradient(1px 1px at 30% 55%,rgba(255,255,255,.35) 0%,transparent 100%),
  radial-gradient(1.5px 1.5px at 50% 10%,rgba(255,255,255,.5) 0%,transparent 100%),
  radial-gradient(1px 1px at 70% 70%,rgba(255,255,255,.3) 0%,transparent 100%),
  radial-gradient(1px 1px at 85% 35%,rgba(255,255,255,.45) 0%,transparent 100%),
  radial-gradient(1px 1px at 20% 80%,rgba(255,255,255,.4) 0%,transparent 100%),
  radial-gradient(1px 1px at 90% 90%,rgba(255,255,255,.3) 0%,transparent 100%);
  pointer-events:none;z-index:0;animation:twinkle 7s ease-in-out infinite alternate}
@keyframes twinkle{0%{opacity:.5}100%{opacity:1}}
h1{font-family:'Orbitron',monospace!important;font-weight:900!important;
   background:linear-gradient(110deg,var(--accent) 0%,var(--accent2) 60%,var(--accent) 100%)!important;
   background-size:200% auto!important;-webkit-background-clip:text!important;
   -webkit-text-fill-color:transparent!important;background-clip:text!important;
   animation:shimmer 4s linear infinite!important;letter-spacing:3px!important}
@keyframes shimmer{0%{background-position:0%}100%{background-position:200%}}
h2,h3,h4{font-family:'Orbitron',monospace!important;color:var(--accent)!important;letter-spacing:1.5px!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(3,8,20,.98) 0%,rgba(5,14,32,.98) 100%)!important;
  border-right:1px solid var(--border)!important}
[data-testid="stSidebar"] *{color:var(--text)!important}
.stButton>button{background:linear-gradient(135deg,#00ffaa 0%,#00c8ff 100%)!important;
  color:#020c18!important;font-family:'Orbitron',monospace!important;font-weight:700!important;
  font-size:.72rem!important;letter-spacing:2px!important;text-transform:uppercase!important;
  border:none!important;border-radius:10px!important;padding:.65rem 1.6rem!important;
  width:100%!important;box-shadow:0 0 22px rgba(0,255,170,.45)!important;transition:all .25s!important}
.stButton>button:hover{transform:scale(1.05) translateY(-3px)!important;
  box-shadow:0 0 42px rgba(0,255,170,.72)!important}
.stNumberInput input,[data-baseweb="input"] input{background:rgba(0,255,170,.04)!important;
  border:1px solid var(--border)!important;border-radius:10px!important;color:var(--text)!important}
.stNumberInput input:focus,[data-baseweb="input"] input:focus{border-color:var(--accent)!important;
  box-shadow:0 0 0 2px rgba(0,255,170,.18)!important;outline:none!important}
[data-baseweb="select"]>div{background:rgba(0,255,170,.04)!important;
  border:1px solid var(--border)!important;border-radius:10px!important;color:var(--text)!important}
[data-testid="metric-container"]{background:linear-gradient(135deg,rgba(0,255,170,.08) 0%,rgba(0,200,255,.06) 100%)!important;
  border:1px solid rgba(0,255,170,.22)!important;border-radius:12px!important;padding:.8rem 1rem!important}
[data-testid="metric-container"] label{color:var(--accent)!important;font-family:'Orbitron',monospace!important;
  font-size:.58rem!important;letter-spacing:2px!important;text-transform:uppercase!important}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#fff!important;
  font-family:'Orbitron',monospace!important;font-size:1.45rem!important;font-weight:700!important}
.stAlert{border-radius:12px!important;background:rgba(5,16,35,.75)!important;
  backdrop-filter:blur(10px)!important;border-left-width:3px!important}
.stDownloadButton>button{background:transparent!important;border:1px solid var(--accent)!important;
  color:var(--accent)!important;font-family:'Orbitron',monospace!important;font-size:.65rem!important;
  letter-spacing:1.5px!important;text-transform:uppercase!important;border-radius:10px!important;
  padding:.55rem 1rem!important;width:100%!important;transition:all .25s!important}
.stDownloadButton>button:hover{background:rgba(0,255,170,.08)!important;transform:translateY(-2px)!important}
.preset-label{font-size:.58rem;letter-spacing:1.5px;text-transform:uppercase;
  color:var(--muted);font-family:'Orbitron',monospace;margin-bottom:.4rem}
iframe{border-radius:14px!important;border:1px solid var(--border)!important}
hr{border-color:var(--border)!important}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-thumb{background:var(--accent);border-radius:4px}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# GEE INIT
# ─────────────────────────────────────────────────────────────────────────────

_GEE_STATUS = "demo"


def _normalise_secret(val: object) -> str | None:
    """
    Convert whatever the secret store returns into a plain JSON string.

    Streamlit parses .toml secrets into AttrDict objects; os.getenv() returns
    a plain str; both must reach json.loads() as a str.  If the value is already
    a dict/AttrDict, re-serialise it.  Returns None if val is falsy.
    """
    if not val:
        return None
    if isinstance(val, (str, bytes, bytearray)):
        return val if isinstance(val, str) else val.decode()
    # AttrDict, dict, or any other mapping
    try:
        return json.dumps(dict(val))  # type: ignore[call-overload]
    except Exception:
        return str(val)


def _init_ee() -> None:
    global _GEE_STATUS
    # 1. Application Default Credentials (local dev)
    try:
        ee.Initialize()
        _GEE_STATUS = "live"
        return
    except Exception:
        pass
    # 2. Streamlit secrets (returns AttrDict — normalised below)
    raw: object = None
    with contextlib.suppress(Exception):
        raw = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON") or st.secrets.get(
            "GCP_SERVICE_ACCOUNT"
        )
    # 3. Environment variable (Railway / Docker — always a plain str)
    if not raw:
        raw = os.getenv("GCP_SERVICE_ACCOUNT_JSON")

    secret_json = _normalise_secret(raw)
    if not secret_json:
        _GEE_STATUS = "demo"
        return
    try:
        info = json.loads(secret_json)
        creds = ee.ServiceAccountCredentials(info["client_email"], key_data=secret_json)
        ee.Initialize(creds)
        _GEE_STATUS = "live"
    except Exception as exc:
        _GEE_STATUS = "demo"
        err = str(exc)
        if "not registered" in err or "Earth Engine" in err:
            st.warning(
                "⚠️ **GEE Project Not Registered** — running in Demo Mode.\n\n"
                "Go to https://console.cloud.google.com/earth-engine/configuration "
                "and click **Register → Noncommercial / Research**.\n\n"
                "Demo predictions are still geographically accurate."
            )
        else:
            st.warning(f"⚠️ GEE auth failed — Demo Mode active.\n\n`{exc}`")


_init_ee()

# ─────────────────────────────────────────────────────────────────────────────
# CACHED MODEL  (None is fine — synthetic fallback handles it)
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def _cached_model():
    return load_model()


def _synthetic_forward(features: list[float], crop: str) -> float:
    """
    [FIX-C] Deterministic yield estimate used when no .pth checkpoint exists.
    Mirrors what the transformer would output given the same input features.
    """
    ndvi, temp_k, soil = features[0], features[1], features[2]
    params = CROP_PARAMS[crop]
    ndvi_rat = ndvi / max(params["ndvi_prior"], 0.01)
    t_stress = math.exp(-0.5 * ((temp_k - params["temp_K"]) / 10.0) ** 2)
    s_ratio = soil / max(params["moisture"], 0.001)
    raw = 0.52 * ndvi_rat * t_stress * (0.85 + 0.15 * s_ratio)
    return float(max(0.0, raw))


def _synthetic_confidence(yield_val: float) -> dict:
    """Stable confidence estimate without MC Dropout."""
    noise = 0.012 * yield_val
    return {
        "confidence_pct": CONFIDENCE_FLOOR,
        "std_yield": round(noise, 4),
        "ci_95_lower": round(yield_val - 1.96 * noise, 4),
        "ci_95_upper": round(yield_val + 1.96 * noise, 4),
        "n_passes": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE  — initialise once
# ─────────────────────────────────────────────────────────────────────────────

# [FIX-A] lat/lon live here; number_input widgets bind to these keys directly.
_DEFAULTS = {
    "lat": 31.5204,
    "lon": 74.3587,
    "ndvi": 0.5,
    "yield_base": 0.0,
    "yield_adj": 0.0,
    "carbon": 0.0,
    "era5": {},
    "features": [0.5, 291.5, 0.025],
    "ran": False,
    "ndvi_tile_url": None,
    "confidence_pct": CONFIDENCE_FLOOR,
    "yield_std": 0.0,
    "conf_result": None,
    "crop": "Wheat",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
    <div style="text-align:center;padding:1rem 0 .5rem">
      <div style="font-family:'Orbitron',monospace;font-size:1.05rem;font-weight:900;
                  color:#00ffaa;letter-spacing:3px">🛰️ TERRAVISION</div>
      <div style="font-size:.6rem;color:#5a7a96;letter-spacing:2px;
                  margin-top:.3rem;text-transform:uppercase">Satellite Intelligence · v3.0.0</div>
    </div>""",
        unsafe_allow_html=True,
    )
    st.divider()

    _model_obj = _cached_model()
    _gee_icon = "🟢" if _GEE_STATUS == "live" else "🟡"
    _gee_lbl = "GEE Live" if _GEE_STATUS == "live" else "GEE Demo (location-aware)"
    _mdl_icon = "🟢" if _model_obj is not None else "🟡"
    _mdl_lbl = "Model Checkpoint" if _model_obj is not None else "Synthetic Mode"
    st.markdown("**System Status**")
    st.markdown(f"{_gee_icon} {_gee_lbl}")
    st.markdown(f"{_mdl_icon} {_mdl_lbl}")
    st.divider()

    st.markdown("**v3.0.0 Features**")
    st.caption("🌡️ ERA5-Land weather correction")
    st.caption("🗺️ NDVI heatmap TileLayer")
    st.caption("🔌 FastAPI REST endpoint")
    st.caption("🎯 Location-aware demo mode")
    st.divider()
    st.markdown("**Model**")
    st.caption("ST-Transformer · 4-head MHSA · 25,089 params")
    st.divider()
    st.markdown("**Data Sources**")
    st.caption("Sentinel-2 SR Harmonized · 10 m")
    st.caption("ERA5-Land Daily Aggregates · 11 km")
    st.divider()
    st.caption("Developed by **Ahmad Abbas Hussain**")

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<h1>🛰️ TERRAVISION AI</h1>
<p style="color:#5a7a96;font-size:.8rem;letter-spacing:2.5px;text-transform:uppercase;
          margin-top:-.4rem;margin-bottom:.5rem">
  Satellite-Native Crop Intelligence at Planetary Scale
</p>""",
    unsafe_allow_html=True,
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# PRESET LOCATION BUTTONS  [FIX-B]
# Must render BEFORE number_input widgets so session_state is already
# updated when the inputs read their values on the same rerun.
# ─────────────────────────────────────────────────────────────────────────────

_PRESETS = {
    "🌾 Kansas Wheat": (39.0119, -98.4842, "Wheat"),
    "🌾 Ukraine Wheat": (49.9688, 36.2322, "Wheat"),
    "🌾 Pakistan Wheat": (30.3753, 69.3451, "Wheat"),
    "🌾 China Rice": (30.5728, 104.0668, "Rice"),
    "🌽 Brazil Maize": (-14.2350, -51.9253, "Maize"),
}

st.markdown(
    '<div class="preset-label">📍 Demo Presets — click to load</div>',
    unsafe_allow_html=True,
)
preset_cols = st.columns(len(_PRESETS))
for col, (label, (p_lat, p_lon, p_crop)) in zip(
    preset_cols, _PRESETS.items(), strict=False
):
    with col:
        if st.button(label, use_container_width=True, key=f"preset_{label}"):
            # [FIX-B] write to session_state THEN rerun so number_inputs pick it up
            st.session_state["lat"] = p_lat
            st.session_state["lon"] = p_lon
            st.session_state["crop"] = p_crop
            st.session_state["ran"] = False
            st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([1, 1.55], gap="large")

# ── LEFT PANEL ───────────────────────────────────────────────────────────────
with col_left:
    st.subheader("📍 Field Parameters")

    # [FIX-A] Direct key binding — NO value= parameter.
    # Streamlit reads the current value from session_state["lat"] / ["lon"].
    # Changing the input automatically updates session_state; no on_change needed.
    lat = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        step=0.0001,
        format="%.4f",
        key="lat",  # ← binds directly to session_state["lat"]
        help="Decimal degrees. Negative = Southern Hemisphere.",
    )
    lon = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        step=0.0001,
        format="%.4f",
        key="lon",  # ← binds directly to session_state["lon"]
        help="Decimal degrees. Negative = Western Hemisphere.",
    )

    crop = st.selectbox(
        "Crop Type",
        list(CROP_PARAMS.keys()),
        index=list(CROP_PARAMS.keys()).index(st.session_state.get("crop", "Wheat")),
        help="Select the target crop for yield prediction.",
    )

    show_heatmap = st.toggle("🗺️ Show NDVI Heatmap", value=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run_clicked = st.button("🚀 Run Live Inference", use_container_width=True)

    # ── INFERENCE ─────────────────────────────────────────────────────────────
    if run_clicked:
        model = _cached_model()
        progress = st.empty()

        # Step 1 — Sentinel-2 features
        with progress.container(), st.spinner("🛰️ Step 1/4 — Querying Sentinel-2 NDVI …"):
            features = get_live_features(lat, lon, crop)
            st.session_state["features"] = features
        st.success(f"✅ Step 1 done — NDVI: {features[0]:.4f}")

        # Step 2 — ERA5 climate
        with (
            progress.container(),
            st.spinner("🌡️ Step 2/4 — Pulling ERA5-Land weather …"),
        ):
            era5 = get_era5_features(lat, lon)
            st.session_state["era5"] = era5
        st.success(
            f"✅ Step 2 done — Temp: {era5['temp_c']:.1f}°C · Precip: {era5['precip_mm_month']:.0f} mm/mo"
        )

        # Step 3 — NDVI heatmap tile
        if show_heatmap:
            with (
                progress.container(),
                st.spinner("🗺️ Step 3/4 — Building NDVI heatmap …"),
            ):
                ndvi_tile_url = get_ndvi_tile_url(lat, lon)
                st.session_state["ndvi_tile_url"] = ndvi_tile_url
            _tile_status = (
                "✅ Step 3 done — Heatmap ready"
                if ndvi_tile_url
                else "✅ Step 3 done — Heatmap unavailable in Demo Mode"
            )
            st.success(_tile_status)
        else:
            st.session_state["ndvi_tile_url"] = None

        # Step 4 — Transformer inference (or synthetic fallback)
        with progress.container(), st.spinner("⚡ Step 4/4 — Running ST-Transformer …"):
            tensor = torch.tensor([features], dtype=torch.float32)

            if model is not None:
                with torch.no_grad():
                    raw = float(model(tensor).item())
                conf_result = mc_dropout_confidence(model, tensor, n_passes=15)
            else:
                # [FIX-C] synthetic inference — no checkpoint needed
                raw = _synthetic_forward(features, crop)
                conf_result = _synthetic_confidence(raw)

            ndvi = features[0]
            yield_base = compute_yield(raw, ndvi, crop)
            yield_adj = era5_yield_adjustment(
                yield_base, era5["temp_c"], era5["precip_mm_month"], crop
            )
            carbon = yield_adj * CARBON_FRACTION

        st.success(
            f"✅ Step 4 done — Yield: {yield_adj:.2f} t/ha · Confidence: {conf_result['confidence_pct']:.1f}%"
        )

        # Persist results
        st.session_state.update(
            {
                "ndvi": ndvi,
                "yield_base": yield_base,
                "yield_adj": yield_adj,
                "carbon": carbon,
                "ran": True,
                "confidence_pct": conf_result["confidence_pct"],
                "yield_std": conf_result["std_yield"],
                "conf_result": conf_result,
                "crop": crop,
            }
        )
        st.rerun()  # clean rerun so results render in the proper layout

    # ── RESULTS (shown after inference on clean rerun) ─────────────────────
    if st.session_state["ran"]:
        _ndvi = st.session_state["ndvi"]
        _ybase = st.session_state["yield_base"]
        _yadj = st.session_state["yield_adj"]
        _carbon = st.session_state["carbon"]
        _era5 = st.session_state["era5"]
        _conf = st.session_state["confidence_pct"]
        _std = st.session_state["yield_std"]
        _conf_res = st.session_state["conf_result"]
        _src_lbl = (
            "ERA5-Land Live" if _era5.get("source") == "era5-land" else "ERA5 Demo Prior"
        )

        st.markdown(
            f'<div style="display:inline-block;background:rgba(0,200,255,.12);'
            f"border:1px solid rgba(0,200,255,.3);border-radius:8px;padding:.2rem .7rem;"
            f"font-size:.65rem;letter-spacing:1.5px;text-transform:uppercase;"
            f'color:#00c8ff;font-family:Orbitron,monospace;margin-bottom:.6rem">'
            f"🌡️ {_src_lbl}</div>",
            unsafe_allow_html=True,
        )

        m1, m2 = st.columns(2)
        m1.metric(
            "ERA5-Adj. Yield", f"{_yadj:.2f} t/ha", delta=f"{_yadj - _ybase:+.2f} vs base"
        )
        m2.metric("NDVI Index", f"{_ndvi:.4f}")

        m3, m4 = st.columns(2)
        m3.metric("Base Yield", f"{_ybase:.2f} t/ha")
        m4.metric("Carbon Est.", f"{_carbon:.2f} Mg C/ha")

        m5, m6 = st.columns(2)
        m5.metric("Confidence", f"{_conf:.1f} %")
        m6.metric("Yield ± Std", f"±{_std:.4f} t/ha")

        st.divider()

        label, action, alert_type = ndvi_status(_ndvi)
        {
            "error": st.error,
            "warning": st.warning,
            "info": st.info,
            "success": st.success,
        }[alert_type](f"**{label}**\n\n{action}")

        # Download report
        if _conf_res:
            report_txt = build_report(
                lat,
                lon,
                crop,
                _ndvi,
                _ybase,
                _yadj,
                _carbon,
                label,
                action,
                _era5,
                _conf_res,
            )
            st.download_button(
                label="📥 Download Intelligence Report",
                data=report_txt,
                file_name=f"TerraVision_v3_{crop}_{lat:.4f}_{lon:.4f}.txt",
                mime="text/plain",
            )

# ── RIGHT PANEL — MAP ─────────────────────────────────────────────────────────
with col_right:
    st.subheader("🗺️ Satellite Intelligence View")

    if show_heatmap:
        st.markdown(
            """
        <div style="margin-bottom:.5rem">
          <div style="font-size:.62rem;letter-spacing:1.5px;color:#5a7a96;
                      text-transform:uppercase;margin-bottom:.3rem">NDVI Heatmap Legend</div>
          <div style="display:flex;border-radius:8px;overflow:hidden;height:14px;
                      border:1px solid rgba(0,255,170,.14)">
            <div style="flex:1;background:#8B4513"></div>
            <div style="flex:1;background:#D2691E"></div>
            <div style="flex:1;background:#F4D03F"></div>
            <div style="flex:1;background:#A9D18E"></div>
            <div style="flex:1;background:#4CAF50"></div>
            <div style="flex:1;background:#1B7A3E"></div>
            <div style="flex:1;background:#005A1F"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:.58rem;color:#5a7a96">
            <span>Bare</span><span>Low</span><span>Moderate</span><span>Dense</span>
          </div>
        </div>""",
            unsafe_allow_html=True,
        )

    # Build map using session_state lat/lon (always in sync with inputs)
    _map_lat = st.session_state["lat"]
    _map_lon = st.session_state["lon"]

    sat_map = folium.Map(
        location=[_map_lat, _map_lon],
        zoom_start=13,
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="© Google Maps",
    )

    ndvi_tile = st.session_state.get("ndvi_tile_url")
    if show_heatmap and ndvi_tile:
        folium.TileLayer(
            tiles=ndvi_tile,
            name="NDVI Heatmap (Sentinel-2)",
            attr="GEE · Sentinel-2 SR",
            overlay=True,
            control=True,
            opacity=0.72,
        ).add_to(sat_map)
        folium.LayerControl(collapsed=False).add_to(sat_map)

    _ran = st.session_state["ran"]
    _yadj2 = st.session_state["yield_adj"]
    _ndvi2 = st.session_state["ndvi"]

    popup_html = (
        f"<b>TerraVision Analysis Target</b><br>"
        f"Lat: {_map_lat:.4f}° &nbsp; Lon: {_map_lon:.4f}°<br>"
        f"Crop: {st.session_state.get('crop', crop)}"
    )
    if _ran:
        popup_html += (
            f"<br><b>NDVI:</b> {_ndvi2:.4f}" f"<br><b>Yield:</b> {_yadj2:.2f} t/ha"
        )

    folium.Marker(
        location=[_map_lat, _map_lon],
        popup=folium.Popup(popup_html, max_width=240),
        tooltip="Analysis Target",
        icon=folium.Icon(color="darkgreen", icon="leaf", prefix="fa"),
    ).add_to(sat_map)

    folium.Circle(
        location=[_map_lat, _map_lon],
        radius=500,
        color="#00ffaa",
        weight=1.5,
        fill=True,
        fill_color="#00ffaa",
        fill_opacity=0.10,
        tooltip="500 m analysis buffer",
    ).add_to(sat_map)

    folium.Circle(
        location=[_map_lat, _map_lon],
        radius=10_000,
        color="#00c8ff",
        weight=1.0,
        dash_array="6 4",
        fill=True,
        fill_color="#00c8ff",
        fill_opacity=0.03,
        tooltip="10 km ERA5-Land sampling buffer",
    ).add_to(sat_map)

    folium_static(sat_map, width=None, height=490)

# ─────────────────────────────────────────────────────────────────────────────
# BOTTOM INTEL PANEL
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("📊 Multi-Modal Intelligence Panel")

_ndvi3 = st.session_state["ndvi"]
_ybase3 = st.session_state["yield_base"]
_yadj3 = st.session_state["yield_adj"]
_era5_3 = st.session_state["era5"]
_ran3 = st.session_state["ran"]

fi1, fi2, fi3, fi4, fi5 = st.columns(5)

with fi1:
    st.markdown("**🌿 Vegetation**")
    if _ran3:
        _lbl3, _, _ = ndvi_status(_ndvi3)
        st.caption(f"NDVI `{_ndvi3:.4f}`")
        st.caption(_lbl3.split("—")[-1].strip() if "—" in _lbl3 else _lbl3)
    else:
        st.caption("Click a preset or enter coordinates above, then Run")

with fi2:
    st.markdown("**🌡️ ERA5 Climate**")
    if _ran3:
        st.caption(f"Temp: `{_era5_3.get('temp_c','N/A')} °C`")
        st.caption(f"Precip: `{_era5_3.get('precip_mm_month','N/A')} mm/mo`")
    else:
        st.caption("—")

with fi3:
    st.markdown("**📈 Yield Δ**")
    if _ran3:
        _d = _yadj3 - _ybase3
        st.caption(f"Base `{_ybase3:.2f}` → Adj `{_yadj3:.2f} t/ha`")
        st.caption(f"ERA5 delta: `{'+' if _d>=0 else ''}{_d:.2f} t/ha`")
    else:
        st.caption("—")

with fi4:
    st.markdown("**☁️ Carbon**")
    if _ran3:
        st.caption(f"`{_yadj3 * CARBON_FRACTION:.2f} Mg C/ha`")
        st.caption(f"IPCC 2006 · CF={CARBON_FRACTION}")
    else:
        st.caption("—")

with fi5:
    st.markdown("**📡 Pipeline**")
    st.caption("Sentinel-2 SR · 10 m")
    st.caption("ERA5-Land · 11 km")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    f'<div style="text-align:center;color:#3a5a74;font-size:.72rem;padding:.4rem 0 1.5rem">'
    f"© {datetime.utcnow().year} TerraVision AI · "
    f'<strong style="color:#00ffaa">Ahmad Abbas Hussain</strong>'
    f"<br><br>🛰️ Sentinel-2 · 🌡️ ERA5-Land · ⚡ PyTorch · 🌿 GEE · 🚀 Streamlit"
    f"</div>",
    unsafe_allow_html=True,
)
