"""
app.py — FloatAi Ocean Intelligence Platform
----------------------------------------------
Light-themed multi-page Streamlit website.
Pages: Home · Explorer · Live Map · Databases · About

Run:  cd FloatAi && streamlit run app.py
"""

from __future__ import annotations
import logging, sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.query_engine import EngineResult, QueryEngine
from nlu import llm_parser

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("FloatAi.app")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FloatAi — Ocean Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
C = dict(
    navy="#0a2342", blue="#1565c0", sky="#1e88e5", teal="#00838f",
    light_blue="#e3f2fd", seafoam="#e0f7fa", white="#ffffff",
    bg="#f8fafd", card="#ffffff", border="#d0e4f7",
    text="#1a2b3c", muted="#5c7a99",
    green="#1b7a34", amber="#b35c00", red="#b71c1c",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ═══════════ BASE ═══════════ */
html, body, [data-testid="stAppViewContainer"] {{
    background: {C['bg']} !important;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    color: {C['text']};
}}
.block-container {{
    padding: 0 !important;
    max-width: 100% !important;
}}
section[data-testid="stSidebar"] {{ display: none !important; }}
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display: none !important; }}
footer {{ display: none !important; }}
button[data-testid="baseButton-header"] {{ display: none !important; }}

/* ═══════════ NAVBAR ═══════════ */
.fb-nav {{
    background: {C['navy']};
    display: flex;
    align-items: center;
    height: 58px;
    padding: 0 2.5rem;
    box-shadow: 0 2px 16px rgba(10,35,66,0.18);
    position: sticky;
    top: 0;
    z-index: 9999;
}}
.fb-nav-brand {{
    font-size: 1.25rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: 0.5px;
    margin-right: 3rem;
    white-space: nowrap;
}}
.fb-nav-brand span {{ color: {C['sky']}; }}

/* ═══════════ HERO ═══════════ */
.fb-hero {{
    background: linear-gradient(135deg, {C['navy']} 0%, #0d3b6e 45%, {C['teal']} 100%);
    padding: 5rem 2rem 6rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}}
.fb-hero::after {{
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 64px;
    background: {C['bg']};
    clip-path: ellipse(60% 100% at 50% 100%);
}}
.fb-hero-emoji {{
    font-size: 4rem;
    display: block;
    margin-bottom: 1rem;
    animation: bob 3.5s ease-in-out infinite;
}}
@keyframes bob {{
    0%,100% {{ transform: translateY(0px); }}
    50%      {{ transform: translateY(-12px); }}
}}
.fb-hero h1 {{
    font-size: 3.2rem;
    font-weight: 900;
    color: #fff;
    margin: 0 0 1rem;
    letter-spacing: -1.5px;
    line-height: 1.1;
}}
.fb-hero p {{
    font-size: 1.15rem;
    color: rgba(255,255,255,0.82);
    max-width: 650px;
    margin: 0 auto 2rem;
    line-height: 1.75;
}}
.fb-hero-search {{
    max-width: 620px;
    margin: 0 auto;
    display: flex;
    gap: 0.5rem;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 50px;
    padding: 0.4rem 0.4rem 0.4rem 1.2rem;
    backdrop-filter: blur(8px);
}}

/* ═══════════ SECTION ═══════════ */
.fb-section {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 2rem;
}}
.fb-section-title {{
    font-size: 1.8rem;
    font-weight: 800;
    color: {C['navy']};
    margin: 3rem 0 0.4rem;
    letter-spacing: -0.5px;
}}
.fb-section-sub {{
    color: {C['muted']};
    font-size: 1rem;
    margin-bottom: 1.8rem;
}}

/* ═══════════ STAT STRIP ═══════════ */
.fb-stats {{
    display: flex;
    gap: 0;
    background: {C['white']};
    border: 1px solid {C['border']};
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 2px 16px rgba(21,101,192,0.07);
    margin: 2.5rem 0;
}}
.fb-stat {{
    flex: 1;
    padding: 1.6rem 1rem;
    text-align: center;
    border-right: 1px solid {C['border']};
    transition: background 0.2s;
}}
.fb-stat:last-child {{ border-right: none; }}
.fb-stat:hover {{ background: {C['light_blue']}; }}
.fb-stat-num {{
    font-size: 2rem;
    font-weight: 900;
    color: {C['blue']};
    line-height: 1;
    margin-bottom: 0.35rem;
}}
.fb-stat-lbl {{
    font-size: 0.75rem;
    font-weight: 700;
    color: {C['muted']};
    text-transform: uppercase;
    letter-spacing: 0.8px;
}}

/* ═══════════ FEATURE CARDS ═══════════ */
.fb-features {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.2rem;
    margin: 0 0 3rem;
}}
@media (max-width: 900px) {{
    .fb-features {{ grid-template-columns: repeat(2,1fr); }}
}}
.fb-feat {{
    background: {C['white']};
    border: 1px solid {C['border']};
    border-radius: 14px;
    padding: 1.6rem;
    box-shadow: 0 2px 10px rgba(21,101,192,0.05);
    transition: transform 0.18s, box-shadow 0.18s;
}}
.fb-feat:hover {{
    transform: translateY(-5px);
    box-shadow: 0 10px 28px rgba(21,101,192,0.13);
}}
.fb-feat-icon {{ font-size: 2rem; margin-bottom: 0.8rem; }}
.fb-feat-title {{
    font-size: 1rem;
    font-weight: 700;
    color: {C['navy']};
    margin-bottom: 0.5rem;
}}
.fb-feat-desc {{ font-size: 0.875rem; color: {C['muted']}; line-height: 1.65; }}

/* ═══════════ DB GRID ═══════════ */
.fb-db-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 3rem;
}}
@media (max-width: 1100px) {{
    .fb-db-grid {{ grid-template-columns: repeat(2,1fr); }}
}}
.fb-db {{
    background: {C['white']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 1.2rem;
    box-shadow: 0 1px 6px rgba(21,101,192,0.06);
    transition: transform 0.15s, box-shadow 0.15s;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}}
.fb-db:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 22px rgba(21,101,192,0.12);
    border-color: {C['sky']};
}}
.fb-db-head {{ display: flex; align-items: center; gap: 0.7rem; }}
.fb-db-icon {{ font-size: 1.6rem; flex-shrink: 0; }}
.fb-db-name {{ font-size: 0.88rem; font-weight: 700; color: {C['navy']}; }}
.fb-db-org  {{ font-size: 0.72rem; color: {C['muted']}; }}
.fb-db-tags {{ display: flex; flex-wrap: wrap; gap: 0.3rem; }}
.fb-tag {{
    background: {C['seafoam']}; color: {C['teal']};
    border-radius: 6px; padding: 2px 8px;
    font-size: 0.7rem; font-weight: 600;
}}
.fb-live {{ color: {C['green']}; font-size: 0.72rem; font-weight: 700; }}
.fb-opt  {{ color: {C['amber']}; font-size: 0.72rem; font-weight: 700; }}

/* ═══════════ EXPLORER ═══════════ */
.fb-expl-banner {{
    background: linear-gradient(90deg, {C['navy']} 0%, {C['blue']} 100%);
    padding: 2.2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.8rem;
    color: white;
}}
.fb-expl-banner h2 {{
    margin: 0 0 0.4rem;
    font-size: 1.7rem;
    font-weight: 800;
}}
.fb-expl-banner p {{
    margin: 0; opacity: 0.82; font-size: 0.95rem;
}}

/* ═══════════ RESULT ═══════════ */
.fb-result {{
    background: {C['white']};
    border: 1px solid {C['border']};
    border-left: 5px solid {C['blue']};
    border-radius: 0 14px 14px 0;
    padding: 1.4rem 1.8rem;
    margin: 1rem 0;
    box-shadow: 0 2px 12px rgba(21,101,192,0.07);
}}
.fb-result-title {{
    font-size: 0.75rem;
    font-weight: 700;
    color: {C['muted']};
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.6rem;
}}
.fb-answer {{
    font-size: 1.05rem;
    color: {C['text']};
    line-height: 1.7;
}}
.fb-answer b {{ color: {C['blue']}; }}
.fb-question-tag {{
    display: inline-block;
    background: {C['light_blue']};
    color: {C['blue']};
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
}}

/* ═══════════ CONFIDENCE ═══════════ */
.fb-conf-high {{ display:inline-block; background:#e8f5e9; color:{C['green']};
    border-radius:20px; padding:3px 14px; font-size:0.82rem; font-weight:700; }}
.fb-conf-mod  {{ display:inline-block; background:#fff3e0; color:{C['amber']};
    border-radius:20px; padding:3px 14px; font-size:0.82rem; font-weight:700; }}
.fb-conf-low  {{ display:inline-block; background:#ffebee; color:{C['red']};
    border-radius:20px; padding:3px 14px; font-size:0.82rem; font-weight:700; }}

/* ═══════════ INFO CARD ═══════════ */
.fb-info-card {{
    background: {C['white']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 1px 6px rgba(21,101,192,0.05);
}}
.fb-info-card h4 {{
    font-size: 0.85rem;
    font-weight: 700;
    color: {C['muted']};
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 0 0 0.6rem;
}}

/* ═══════════ HEATWAVE BADGE ═══════════ */
.fb-hw-event {{
    background: linear-gradient(135deg, #fff3e0, #ffe0b2);
    border: 1px solid #ffb74d;
    border-left: 4px solid {C['amber']};
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
}}
.fb-hw-event b {{ color: {C['amber']}; }}
.fb-no-hw {{
    background: #e8f5e9;
    border: 1px solid #a5d6a7;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: {C['green']};
    font-weight: 600;
    font-size: 0.88rem;
}}

/* ═══════════ DIVIDER ═══════════ */
.fb-divider {{ height:1px; background:{C['border']}; margin:2rem 0; border:none; }}

/* ═══════════ ABOUT ═══════════ */
.fb-about-card {{
    background: {C['white']};
    border: 1px solid {C['border']};
    border-radius: 14px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 10px rgba(21,101,192,0.05);
}}
.fb-about-card h3 {{
    font-size: 1rem;
    font-weight: 700;
    color: {C['navy']};
    margin: 0 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.fb-timeline {{
    border-left: 3px solid {C['blue']};
    padding-left: 1.2rem;
    margin: 0.5rem 0;
}}
.fb-timeline-step {{
    margin-bottom: 1rem;
    font-size: 0.9rem;
    line-height: 1.6;
    color: {C['text']};
}}
.fb-timeline-step strong {{ color: {C['navy']}; }}

/* ═══════════ FOOTER ═══════════ */
.fb-footer {{
    background: {C['navy']};
    color: rgba(255,255,255,0.5);
    text-align: center;
    padding: 2.5rem 1rem;
    font-size: 0.82rem;
    margin-top: 5rem;
}}
.fb-footer a {{ color: {C['sky']}; text-decoration: none; }}
.fb-footer strong {{ color: rgba(255,255,255,0.8); }}

/* ═══════════ STREAMLIT OVERRIDES ═══════════ */
.stButton > button {{
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.18s !important;
}}
.stButton > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(21,101,192,0.2) !important;
}}
[data-testid="stForm"] {{
    background: {C['white']};
    border: 1px solid {C['border']};
    border-radius: 14px;
    padding: 1.5rem !important;
    box-shadow: 0 2px 12px rgba(21,101,192,0.06);
}}
.stSelectbox > div > div, .stTextInput > div > div > input,
.stTextArea > div > div > textarea {{
    border-color: {C['border']} !important;
    border-radius: 8px !important;
    background: {C['bg']} !important;
}}
.stMetric {{ background: {C['white']}; border: 1px solid {C['border']};
            border-radius: 10px; padding: 0.8rem 1rem !important; }}
.stMetricValue {{ color: {C['blue']} !important; font-weight: 800 !important; }}
.stAlert {{ border-radius: 10px !important; }}
.stExpander {{ border: 1px solid {C['border']} !important;
               border-radius: 10px !important; background: {C['white']} !important; }}
h1,h2,h3 {{ color: {C['navy']} !important; }}
.stRadio label {{ font-weight: 600 !important; color: {C['navy']} !important; }}
.stCheckbox label {{ font-weight: 500 !important; }}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
_SS_DEFAULTS = {
    "page":             "Home",
    "engine":           None,
    "use_synthetic":    False,
    "last_result":      None,
    "history":          [],
    "map_load":         False,
    "map_region_loaded":"Arabian Sea",
    "map_days_loaded":  30,
    "current_result":   None,
    "current_question": "",
    "extra_snap":       None,
    "extra_requested":  False,
    "extra_fetching":   False,
    "_render_counter":  0,
    "ose_result":       None,
    "ose_no_data":      False,
    "ose_error":        None,
    # FAE integration (optional)
    "fae_result":       None,
    "fae_error":        None,
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

def _go(page: str):
    # Clear extra-DB state when navigating away from Explorer
    if page != "Explorer":
        st.session_state["extra_requested"] = False
    st.session_state.page = page
    st.rerun()

def _get_engine() -> QueryEngine:
    synth = st.session_state.use_synthetic
    if (st.session_state.engine is None or
            getattr(st.session_state.engine, "use_synthetic", None) != synth):
        st.session_state.engine = QueryEngine(use_synthetic=synth)
    return st.session_state.engine

# ── Helpers ───────────────────────────────────────────────────────────────────
def _conf_badge(label: str) -> str:
    cls = {"High": "fb-conf-high", "Moderate": "fb-conf-mod", "Low": "fb-conf-low"}
    icon = {"High": "🟢", "Moderate": "🟡", "Low": "🔴"}
    return f'<span class="{cls.get(label,"fb-conf-low")}">{icon.get(label,"⚪")} {label}</span>'

def _pill(label: str, color: str, bg: str) -> str:
    return (f'<span style="display:inline-block;background:{bg};color:{color};'
            f'border-radius:20px;padding:2px 12px;font-size:0.78rem;font-weight:700;">'
            f'{label}</span>')

# ─────────────────────────────────────────────────────────────────────────────
# NAVBAR
# ─────────────────────────────────────────────────────────────────────────────
def _navbar():
    st.markdown(
        f'<div class="fb-nav"><span class="fb-nav-brand">🌊 Float<span>Bot</span></span></div>',
        unsafe_allow_html=True,
    )
    # Streamlit nav buttons in a centered row
    g = st.columns([1.0, 1, 1, 1, 1, 1, 1, 1, 1.0])
    labels = ["🏠 Home", "🔍 Explorer", "🗺️ Map", "🛢️ Databases", "🌡️ Stress", "🎣 Advisory", "ℹ️ About"]
    pages  = ["Home", "Explorer", "Map", "Databases", "Stress", "Advisory", "About"]
    cur = st.session_state.page
    for col, lbl, pg in zip(g[1:8], labels, pages):
        with col:
            kwargs = {"type": "primary"} if pg == cur else {}
            if st.button(lbl, key=f"nav_{pg}", use_container_width=True, **kwargs):
                _go(pg)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE — HOME
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    # Hero
    st.markdown(f"""
    <div class="fb-hero">
      <span class="fb-hero-emoji">🌊</span>
      <h1>Ocean Intelligence Platform</h1>
      <p>Ask plain-English questions about ocean temperature, salinity, currents,
      marine heatwaves, and coral bleaching — powered by 8 real scientific databases,
      quality-controlled and fully explainable.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    # Quick search
    with st.container():
        _, mid, _ = st.columns([1, 3, 1])
        with mid:
            with st.form("home_search", clear_on_submit=True):
                q = st.text_input("", placeholder="e.g. Was there a marine heatwave in the Arabian Sea in July 2023?",
                                  label_visibility="collapsed")
                if st.form_submit_button("🔍  Search Ocean Data", use_container_width=True, type="primary"):
                    if q.strip():
                        st.session_state["prefill_question"] = q.strip()
                        _go("Explorer")

    # Stats strip
    st.markdown("""
    <div class="fb-stats">
      <div class="fb-stat"><div class="fb-stat-num">8</div><div class="fb-stat-lbl">Connected Databases</div></div>
      <div class="fb-stat"><div class="fb-stat-num">3,900+</div><div class="fb-stat-lbl">Active Argo Floats</div></div>
      <div class="fb-stat"><div class="fb-stat-num">1981</div><div class="fb-stat-lbl">SST Archive Since</div></div>
      <div class="fb-stat"><div class="fb-stat-num">0.25°</div><div class="fb-stat-lbl">SST Resolution</div></div>
      <div class="fb-stat"><div class="fb-stat-num">15″</div><div class="fb-stat-lbl">Depth Resolution</div></div>
      <div class="fb-stat"><div class="fb-stat-num">100%</div><div class="fb-stat-lbl">QC-Verified</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Feature grid
    st.markdown('<div class="fb-section-title">⚡ What FloatAi can do</div>', unsafe_allow_html=True)
    st.markdown('<div class="fb-section-sub">Every answer is quality-controlled, sourced, and explainable.</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="fb-features">
      <div class="fb-feat">
        <div class="fb-feat-icon">🌡️</div>
        <div class="fb-feat-title">Temperature &amp; Salinity</div>
        <div class="fb-feat-desc">Query real Argo float profiles with full QC flag checking — confidence scores, exclusion explanations, and data provenance included.</div>
      </div>
      <div class="fb-feat">
        <div class="fb-feat-icon">🔥</div>
        <div class="fb-feat-title">Marine Heatwave Detection</div>
        <div class="fb-feat-desc">Hobday et al. (2016) algorithm — 11-day climatology window, 90th-percentile threshold, ≥5 consecutive days — applied to NOAA OISST data.</div>
      </div>
      <div class="fb-feat">
        <div class="fb-feat-icon">🪸</div>
        <div class="fb-feat-title">Coral Bleaching Alerts</div>
        <div class="fb-feat-desc">NOAA Coral Reef Watch Degree Heating Weeks (DHW) — Watch, Alert Level 1, and Alert Level 2 classification for any tropical region.</div>
      </div>
      <div class="fb-feat">
        <div class="fb-feat-icon">🌊</div>
        <div class="fb-feat-title">Ocean Depth</div>
        <div class="fb-feat-desc">GEBCO 2022 bathymetry at 15-arc-second resolution — from continental shelf to the deepest ocean trenches.</div>
      </div>
      <div class="fb-feat">
        <div class="fb-feat-icon">🌀</div>
        <div class="fb-feat-title">Surface Currents</div>
        <div class="fb-feat-desc">NOAA HYCOM model + GDP surface drifter observations — current speed, direction, and real-time in-situ tracking.</div>
      </div>
      <div class="fb-feat">
        <div class="fb-feat-icon">🤖</div>
        <div class="fb-feat-title">Plain-English Queries</div>
        <div class="fb-feat-desc">Gemini LLM or built-in rule-based parser (no API key needed) — ask in natural language, get structured, cited answers.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Connected databases
    st.markdown('<div class="fb-section-title">🛢️ Real databases, live data</div>', unsafe_allow_html=True)
    st.markdown('<div class="fb-section-sub">No simulated data — every answer comes from these publicly accessible scientific sources.</div>', unsafe_allow_html=True)
    dbs = [
        ("🌐","Argo GDAC","Ifremer / UCSD",["T/S Profiles","QC Flags","2000–present"],"live"),
        ("🛰️","NOAA OISST v2.1","NOAA NCEI",["Daily SST","1981–present","0.25°"],"live"),
        ("🛟","NOAA GDP Drifters","NOAA AOML",["Surface SST","Currents","Real-time"],"live"),
        ("🌀","NOAA HYCOM Model","NOAA NCEI",["Currents","T/S Model","SSH"],"live"),
        ("📚","World Ocean DB","NOAA NCEI",["Historical T/S","1772–present"],"live"),
        ("🪸","Coral Reef Watch","NOAA CoRTAD",["DHW","Bleaching Alerts"],"live"),
        ("🌊","GEBCO 2022","GEBCO / BODC",["Bathymetry","15″ grid"],"live"),
        ("🇪🇺","CMEMS Copernicus","EU Copernicus",["Physics Model","Chl-a"],"opt"),
    ]
    html = '<div class="fb-db-grid">'
    for icon, name, org, tags, status in dbs:
        tag_html = "".join(f'<span class="fb-tag">{t}</span>' for t in tags)
        st_html = f'<span class="fb-live">● Live</span>' if status=="live" else f'<span class="fb-opt">● Optional</span>'
        html += f"""
        <div class="fb-db">
          <div class="fb-db-head">
            <span class="fb-db-icon">{icon}</span>
            <div><div class="fb-db-name">{name}</div>
                 <div class="fb-db-org">{org} &nbsp;{st_html}</div></div>
          </div>
          <div class="fb-db-tags">{tag_html}</div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    # CTA row
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        if st.button("🔍 Open Explorer", use_container_width=True, type="primary"):
            _go("Explorer")
    with c2:
        if st.button("🗺️ Live Float Map", use_container_width=True):
            _go("Map")
    with c3:
        if st.button("🌡️ Stress Engine", use_container_width=True):
            _go("Stress")
    with c4:
        if st.button("🎣 Fisher Advisory", use_container_width=True):
            _go("Advisory")
    with c5:
        if st.button("🛢️ All Databases", use_container_width=True):
            _go("Databases")
    with c6:
        if st.button("ℹ️ About", use_container_width=True):
            _go("About")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE — EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
def page_explorer():
    st.markdown(f"""
    <div class="fb-expl-banner">
      <h2>🔍 Ocean Data Explorer</h2>
      <p>Choose a structured form or type a plain-English question to query real ocean databases.</p>
    </div>
    """, unsafe_allow_html=True)

    # Status bar
    s1, s2, s3 = st.columns([2, 2, 4])
    with s1:
        use_synth = st.toggle("🧪 Offline / synthetic mode",
                              value=st.session_state.use_synthetic)
        st.session_state.use_synthetic = use_synth
    with s2:
        if llm_parser.is_available():
            prov = llm_parser.active_provider() or "llm"
            st.markdown(_pill(f"🤖 LLM: {prov}", "#1b5e20", "#e8f5e9"), unsafe_allow_html=True)
        else:
            st.markdown(_pill("📐 Rule-based parser", "#b35c00", "#fff8e1"), unsafe_allow_html=True)
        if use_synth:
            st.markdown(_pill("🧪 Synthetic data", "#880e4f", "#fce4ec"), unsafe_allow_html=True)
    with s3:
        st.caption("💡 Set GOOGLE_API_KEY for AI-powered parsing. Rule-based works without any key.")

    st.markdown("<hr style='border-color:#d0e4f7;margin:1rem 0'>", unsafe_allow_html=True)

    mode = st.radio("", ["🗂️ Structured form", "💬 Plain-English question"],
                    horizontal=True, label_visibility="collapsed",
                    key="explorer_mode")

    # ── Plain-English form ──────────────────────────────────────────────
    if mode == "💬 Plain-English question":
        prefill = st.session_state.pop("prefill_question", "")

        with st.form("nlq_form", clear_on_submit=True):
            st.markdown("**Ask anything about the ocean:**")
            typed = st.text_area("", value=prefill, height=85,
                placeholder="e.g. What was the salinity in the Red Sea in Feb 2023, and was anything unusual?",
                label_visibility="collapsed", key="nlq_text")

            st.markdown("**Quick examples — click to run instantly:**")
            ec = st.columns(3)
            examples = [
                "Temperature in Arabian Sea, Aug 2023?",
                "Heatwave in Coral Sea, Jan 2022?",
                "Salinity in Red Sea, Feb 2023?",
                "Ocean depth in Bay of Bengal?",
                "Coral bleaching in Red Sea 2022?",
                "Salinity and heatwave in Arabian Sea, Jul 2023?",
            ]
            clicked_example = None
            for i, ex in enumerate(examples):
                with ec[i % 3]:
                    if st.form_submit_button(f"📌 {ex}", use_container_width=True):
                        clicked_example = ex

            run_clicked = st.form_submit_button(
                "🚀  Run Query", type="primary", use_container_width=True)

        # Determine which question to run
        if clicked_example:
            _run_and_render(clicked_example)
        elif run_clicked and typed and typed.strip():
            _run_and_render(typed.strip())

    # ── Structured form ─────────────────────────────────────────────────
    else:
        st.markdown("**Build your query:**")
        with st.form("struct_form"):
            fc1, fc2 = st.columns(2)
            with fc1:
                st.markdown("**📍 Ocean Region**")
                region = st.selectbox("", [
                    "Arabian Sea", "Bay of Bengal", "Indian Ocean",
                    "Gulf of Mexico", "Caribbean Sea", "North Atlantic", "South Atlantic",
                    "North Pacific", "South Pacific", "Tropical Pacific",
                    "Mediterranean Sea", "Red Sea", "Persian Gulf", "Gulf of Aden",
                    "Coral Sea", "Tasman Sea", "South China Sea", "Southern Ocean",
                ], label_visibility="collapsed")
            with fc2:
                st.markdown("**📅 Date Range** — *keep to 1–3 months for fastest results*")
                dd1, dd2 = st.columns(2)
                with dd1:
                    sd = st.date_input("From", value=pd.Timestamp("2023-08-01"),
                                       min_value=pd.Timestamp("1981-01-01").date(),
                                       label_visibility="collapsed")
                with dd2:
                    ed = st.date_input("To", value=pd.Timestamp("2023-08-31"),
                                       min_value=pd.Timestamp("1981-01-01").date(),
                                       label_visibility="collapsed")

            st.markdown("**🔬 What to query:**")
            pc = st.columns(5)
            with pc[0]: wt = st.checkbox("🌡️ Temperature", value=True)
            with pc[1]: ws = st.checkbox("🧂 Salinity",    value=True)
            with pc[2]: wh = st.checkbox("🔥 Heatwave",    value=True)
            with pc[3]: wc = st.checkbox("🪸 Coral",       value=False)
            with pc[4]: wd = st.checkbox("🌊 Depth",       value=False)

            depth_range = st.select_slider(
                "Depth range (dbar)", label_visibility="visible",
                options=[0, 5, 10, 25, 50, 100, 200, 500, 1000, 2000],
                value=(0, 10),
            )

            struct_run = st.form_submit_button(
                "🚀  Run Query", type="primary", use_container_width=True)

        if struct_run:
            params = " and ".join(
                [p for p, f in [("temperature", wt), ("salinity", ws)] if f]
            ) or "temperature"
            dp  = f" at {depth_range[0]}–{depth_range[1]} dbar" if depth_range != (0, 10) else ""
            hw  = " Was there a marine heatwave?" if wh else ""
            co  = " Was there coral bleaching?" if wc else ""
            de  = " How deep is the ocean here?" if wd else ""
            q   = f"What was the mean {params} in the {region} from {sd} to {ed}{dp}?{hw}{co}{de}"
            _run_and_render(q)

    # ── Render current result + extra DB panel (always, survives all rerenders) ──
    _render_current_result()

    # ── History ──────────────────────────────────────────────────────────
    if st.session_state.history:
        st.markdown("<hr style='border-color:#d0e4f7;margin:2rem 0'>", unsafe_allow_html=True)
        st.markdown("### 📜 Recent Queries")
        hcol1, hcol2 = st.columns([4, 1])
        with hcol2:
            if st.button("🗑️ Clear history", use_container_width=True):
                st.session_state.history = []
                st.rerun()
        for q, r in reversed(st.session_state.history[-5:]):
            with st.expander(f"🔍 {q[:80]}{'…' if len(q) > 80 else ''}", expanded=False):
                if r.error:
                    st.error(r.error[:300])
                else:
                    import re as _re
                    ans = r.to_answer_text()
                    ans_html = _re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>',
                                       ans.replace("\n\n", "<br><br>").replace("\n", "<br>"))
                    st.markdown(
                        f'<div class="fb-result">'
                        f'<div class="fb-question-tag">🔍 {q}</div>'
                        f'<div class="fb-result-title">📊 Result</div>'
                        f'<div class="fb-answer">{ans_html}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    for qc, lbl in [(r.qc_result, "🌡️ Temperature"),
                                    (r.qc_result_salinity, "🧂 Salinity")]:
                        if qc and qc.value is not None:
                            st.caption(
                                f"{lbl}: {qc.value:.2f} {qc.unit} — "
                                f"{qc.confidence_label} confidence "
                                f"({qc.n_good}/{qc.n_total} readings passed QC, "
                                f"{qc.depth_note})"
                            )


# ─────────────────────────────────────────────────────────────────────────────
# QUERY RUN + RESULT RENDER
# ─────────────────────────────────────────────────────────────────────────────
def _run_and_render(question: str):
    engine = _get_engine()

    # Parse first so we can show what was understood BEFORE the fetch
    parsed_preview = engine._parse(question)
    region = parsed_preview.get("region")
    intent = parsed_preview.get("intent", "?")
    param  = parsed_preview.get("parameter", "?")
    sd_p   = parsed_preview.get("start_date", "?")
    ed_p   = parsed_preview.get("end_date", "?")

    intent_icon = {"profile_query": "📊", "heatwave_query": "🔥",
                   "compound_query": "📊+🔥"}.get(intent, "❓")
    r_str = (f"lat {region['lat_min']:.0f}°–{region['lat_max']:.0f}°, "
             f"lon {region['lon_min']:.0f}°–{region['lon_max']:.0f}°") if region else "⚠️ NO REGION DETECTED"

    st.markdown(
        f"<div style='background:#e8f4fd;border:1px solid #bee3f8;border-radius:10px;"
        f"padding:10px 16px;margin-bottom:1rem;font-size:0.88rem;color:#1a365d'>"
        f"🧠 <b>Query understood as:</b> &nbsp;"
        f"{intent_icon} <code>{intent}</code> &nbsp;|&nbsp; "
        f"📦 <code>{param}</code> &nbsp;|&nbsp; "
        f"📅 <code>{sd_p} → {ed_p}</code> &nbsp;|&nbsp; "
        f"📍 <code>{r_str}</code>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if not region:
        st.error("Could not identify a region. Try naming an ocean "
                 "(e.g. 'Arabian Sea', 'Gulf of Mexico').")
        return

    try:
        n_months = max(1, round((pd.Timestamp(ed_p) - pd.Timestamp(sd_p)).days / 30)) \
                   if sd_p != "?" and ed_p != "?" else 1
    except Exception:
        n_months = 1
    est = f"~{n_months * 8}–{n_months * 15} s" if n_months <= 3 else f"~{n_months // 2} min"
    synth_note = " (instant — synthetic)" if st.session_state.use_synthetic else f" ({est})"

    with st.spinner(f"🌊 Querying {param} · {r_str}{synth_note}…"):
        try:
            result = engine.answer(question)
        except Exception as exc:
            st.error(f"**Query failed:** {exc}\n\n"
                     "💡 Try **Offline mode** or a **shorter date range**.")
            logger.exception("Query engine error")
            return

    # Store everything in session_state so it survives button-click rerenders
    st.session_state["current_result"]  = result
    st.session_state["current_question"] = question
    st.session_state["extra_snap"]       = None   # reset extra snapshot
    st.session_state["extra_requested"]  = False
    st.session_state["last_parsed"]      = result.parsed_query
    st.session_state.history.append((question, result))

    st.rerun()   # rerun so _render_current_result() picks it up cleanly


def _render_current_result():
    """
    Renders the stored result on every page rerender — survives button clicks.

    State machine for extra databases:
      extra_requested=False, extra_snap=None  → show button
      extra_requested=True,  extra_snap=None  → fetch then show
      extra_requested=False, extra_snap=set   → show results (already fetched)
    """
    result: Optional[EngineResult] = st.session_state.get("current_result")
    if result is None:
        return

    # ── Parse info banner ────────────────────────────────────────────────────
    pq     = result.parsed_query
    region = pq.get("region", {})
    intent = pq.get("intent", "?")
    param  = pq.get("parameter", "?")
    sd_p   = pq.get("start_date", "?")
    ed_p   = pq.get("end_date", "?")
    intent_icon = {"profile_query":"📊","heatwave_query":"🔥","compound_query":"📊+🔥"}.get(intent,"❓")
    r_str = (f"lat {region.get('lat_min',0):.0f}°–{region.get('lat_max',0):.0f}°, "
             f"lon {region.get('lon_min',0):.0f}°–{region.get('lon_max',0):.0f}°") if region else "?"

    st.markdown(
        f"<div style='background:#e8f4fd;border:1px solid #bee3f8;border-radius:10px;"
        f"padding:10px 16px;margin-bottom:1rem;font-size:0.88rem;color:#1a365d'>"
        f"🧠 <b>Query:</b> &nbsp;{intent_icon} <code>{intent}</code> &nbsp;|&nbsp; "
        f"📦 <code>{param}</code> &nbsp;|&nbsp; 📅 <code>{sd_p} → {ed_p}</code>"
        f" &nbsp;|&nbsp; 📍 <code>{r_str}</code></div>",
        unsafe_allow_html=True,
    )

    # ── Primary result (Argo QC + heatwave) ──────────────────────────────────
    _render_result(result, show_question=True)

    # ── Fast-DB panel (GEBCO + drifters + CRW — fetched automatically) ───────
    fast_snap = getattr(result, "ocean_snapshot", None)
    if fast_snap is not None:
        _render_fast_db_panel(fast_snap)

    st.markdown("<hr style='border-color:#d0e4f7;margin:1.5rem 0'>", unsafe_allow_html=True)

    if result.error or not pq.get("region"):
        return

    # ── Slow DB state machine ────────────────────────────────────────────────
    extra_snap      = st.session_state.get("extra_snap")
    extra_requested = st.session_state.get("extra_requested", False)

    if extra_snap is not None:
        # Already fetched — just render
        _render_snapshot_panel(extra_snap)

    elif extra_requested:
        # Fetch now (spinner shown while waiting)
        engine = _get_engine()
        with st.spinner("🌐 Fetching HYCOM · WOD · CMEMS — this takes 1–3 minutes…"):
            try:
                snap = engine.fetch_snapshot(pq)
                st.session_state["extra_snap"]      = snap
                st.session_state["extra_requested"] = False
            except Exception as exc:
                logger.exception("Extra DB fetch failed")
                st.session_state["extra_requested"] = False
                st.session_state["extra_snap"]      = None
                snap = None
                st.warning(f"Extra database fetch failed: {exc}")
        if snap is not None:
            _render_snapshot_panel(snap)

    else:
        # Show the trigger button
        col_b, col_t = st.columns([3, 5])
        with col_b:
            if st.button(
                "🌐 Load HYCOM · WOD · CMEMS",
                key="extra_db_btn",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state["extra_requested"] = True
                st.rerun()
        with col_t:
            st.caption(
                "Optional: loads NOAA HYCOM model currents, "
                "World Ocean Database historical profiles, "
                "and Copernicus CMEMS reanalysis. Takes 1–3 min."
            )


def _render_fast_db_panel(snap) -> None:
    """Render the fast-DB results (GEBCO depth, drifter SST, coral bleaching)."""
    if snap is None:
        return

    items = []
    # GEBCO
    if snap.gebco_depth_m and snap.gebco_depth_m > 0:
        d = snap.gebco_depth_m
        if   d < 200:   zone = "Continental Shelf"
        elif d < 1000:  zone = "Continental Slope"
        elif d < 3500:  zone = "Continental Rise"
        elif d < 6000:  zone = "Abyssal Plain"
        else:            zone = "Hadal Zone"
        items.append(("🌊 Ocean Depth", f"{d:.0f} m", zone, "GEBCO 2022"))

    # Drifter SST — filter fill values before computing mean
    if snap.drifter_df is not None and not snap.drifter_df.empty:
        if "SST" in snap.drifter_df.columns and snap.drifter_df["SST"].notna().any():
            sst_valid = snap.drifter_df["SST"][
                (snap.drifter_df["SST"] > -10) & (snap.drifter_df["SST"] < 50)
            ]
            if not sst_valid.empty:
                mean_sst = float(sst_valid.mean())
                n_d = snap.drifter_df["DRIFTER_ID"].nunique() \
                      if "DRIFTER_ID" in snap.drifter_df.columns else "?"
                items.append(("🛟 Surface SST", f"{mean_sst:.2f}°C",
                              f"from {n_d} drifters", "NOAA GDP"))

    # Coral bleaching
    if snap.coral_df is not None and not snap.coral_df.empty:
        if "DHW" in snap.coral_df.columns and snap.coral_df["DHW"].notna().any():
            max_dhw   = float(snap.coral_df["DHW"].max())
            max_alert = int(snap.coral_df.get("ALERT_LEVEL",
                            snap.coral_df.assign(ALERT_LEVEL=0)["ALERT_LEVEL"]).max())
            alert_labels = {0:"No Stress ✅", 1:"Watch ⚠️", 2:"Warning ⚠️",
                            3:"Alert Level 1 🚨", 4:"Alert Level 2 🚨"}
            items.append(("🪸 Coral Bleaching",
                          f"DHW {max_dhw:.1f}°C-wks",
                          alert_labels.get(max_alert, "—"),
                          "NOAA Coral Reef Watch"))

    if not items and not snap.source_report:
        return

    st.markdown("#### 🌐 External Database Results")
    if items:
        cols = st.columns(len(items))
        for i, (label, value, delta, source) in enumerate(items):
            with cols[i]:
                st.metric(label, value, delta)
                st.caption(source)

    # Source report
    if snap.source_report:
        statuses = list(snap.source_report.values())
        n_ok  = sum(1 for s in statuses if s.startswith("✅"))
        n_err = sum(1 for s in statuses if s.startswith("❌"))
        with st.expander(
            f"📋 Database status ({n_ok}/{len(statuses)} returned data)",
            expanded=False,
        ):
            for db, status in snap.source_report.items():
                icon = "✅" if status.startswith("✅") else ("❌" if status.startswith("❌") else "⚠️")
                st.markdown(f"{icon} **{db}**: {status.lstrip('✅❌⚠️').strip()}")


def _render_result(result: EngineResult, show_question: bool = False):
    """Render primary Argo result — QC panels, charts, heatwave. Used by history."""
    # Unique ID for all widget keys
    if "_render_counter" not in st.session_state:
        st.session_state["_render_counter"] = 0
    st.session_state["_render_counter"] += 1
    uid = st.session_state["_render_counter"]

    if result.error:
        st.error(result.error)
        return

    for w in result.warnings:
        # Only suppress low-level HTTP/server noise — show heatwave failures clearly
        if any(kw in w.lower() for kw in
               ("404", "http", "chunk", "server", "erddap")):
            logger.info("Suppressed low-level warning: %s", w[:120])
        # Argo no-data warnings get a blue info box, not orange warning
        elif "no argo float data" in w.lower() or "no floats were active" in w.lower():
            st.info(w, icon="🌊")
        elif "heatwave check failed" in w.lower():
            st.warning(f"⚠️ Heatwave data unavailable: SST could not be fetched for this region/period. "
                       f"Temperature and salinity results above are still valid.", icon="🌡️")
        else:
            st.warning(w)

    # Main answer — with prominent parameter + region label
    answer_html = result.to_answer_text().replace("\n\n", "<br><br>").replace("\n", "<br>")
    import re
    answer_html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', answer_html)

    # Build a clear label showing exactly what was queried
    pq = result.parsed_query
    params_label = []
    if result.qc_result:          params_label.append(f"🌡️ {result.qc_result.parameter.title()}")
    if result.qc_result_salinity: params_label.append(f"🧂 {result.qc_result_salinity.parameter.title()}")
    if result.heatwave_result:    params_label.append("🔥 Heatwave")
    region = pq.get("region", {})
    region_label = (f"lat {region.get('lat_min',0):.0f}°–{region.get('lat_max',0):.0f}°, "
                    f"lon {region.get('lon_min',0):.0f}°–{region.get('lon_max',0):.0f}°") if region else "?"

    # Build heatwave HTML block to embed inside the answer card
    hw = result.heatwave_result
    hw_html = ""
    if hw is not None:
        if hw.heatwave_detected:
            events_html = ""
            for ev in hw.overlapping_events:
                d = ev.to_dict()
                events_html += (
                    f'<div class="fb-hw-event" style="margin-top:0.6rem">'
                    f'📅 <b>{d["start_date"]}</b> → <b>{d["end_date"]}</b> &nbsp;|&nbsp; '
                    f'⏱ <b>{d["duration_days"]} days</b> &nbsp;|&nbsp; '
                    f'🌡 Peak <b>{d["peak_sst"]:.1f}°C</b> on {d["peak_date"]} &nbsp;|&nbsp; '
                    f'📈 <b>{d["mean_anomaly"]:+.2f}°C</b> above climatology'
                    f'</div>'
                )
            hw_html = (
                f'<div style="margin-top:1rem;padding-top:0.8rem;border-top:1px solid #d0e4f7">'
                f'<div style="font-size:0.75rem;font-weight:700;color:#5c7a99;text-transform:uppercase;'
                f'letter-spacing:0.8px;margin-bottom:0.4rem">🔥 Marine Heatwave Analysis</div>'
                f'<div class="fb-answer"><b style="color:#b35c00">⚠️ Marine heatwave detected</b> '
                f'during the query period.</div>'
                f'{events_html}'
                f'<div style="font-size:0.75rem;color:#5c7a99;margin-top:0.5rem">'
                f'Method: Hobday et al. (2016) · {hw.climatology_years} yr(s) SST climatology · '
                f'90th-percentile threshold · ≥5 consecutive days</div>'
                f'</div>'
            )
        else:
            hw_html = (
                f'<div style="margin-top:1rem;padding-top:0.8rem;border-top:1px solid #d0e4f7">'
                f'<div style="font-size:0.75rem;font-weight:700;color:#5c7a99;text-transform:uppercase;'
                f'letter-spacing:0.8px;margin-bottom:0.4rem">🔥 Marine Heatwave Analysis</div>'
                f'<div class="fb-no-hw">✅ No marine heatwave detected during this period. '
                f'SST remained below the 90th-percentile climatological threshold throughout.</div>'
                f'<div style="font-size:0.75rem;color:#5c7a99;margin-top:0.5rem">'
                f'Method: Hobday et al. (2016) · {hw.climatology_years} yr(s) SST climatology · '
                f'90th-percentile threshold · ≥5 consecutive days</div>'
                f'</div>'
            )

    # Build coral bleaching HTML block
    coral_html = ""
    snap = getattr(result, "ocean_snapshot", None)
    if snap is not None and snap.coral_df is not None and not snap.coral_df.empty:
        cdf = snap.coral_df
        if "DHW" in cdf.columns and cdf["DHW"].notna().any():
            max_dhw   = float(cdf["DHW"].max())
            mean_dhw  = float(cdf["DHW"].mean())
            max_alert = int(cdf["ALERT_LEVEL"].max()) if "ALERT_LEVEL" in cdf.columns else 0
            pct_stress = float((cdf["DHW"] >= 4).mean() * 100)
            alert_labels = {
                0: "No Stress",
                1: "🟡 Bleaching Watch (DHW ≥ 4)",
                2: "🟠 Bleaching Warning",
                3: "🔴 Alert Level 1 — Mass Bleaching Likely (DHW ≥ 8)",
                4: "🚨 Alert Level 2 — Mass Bleaching &amp; Mortality (DHW ≥ 12)",
            }
            alert_name = alert_labels.get(max_alert, "Unknown")
            if max_alert == 0:
                coral_body = (
                    f'<div class="fb-no-hw">✅ No significant coral bleaching thermal stress. '
                    f'Max DHW = {max_dhw:.1f}°C-weeks (threshold: 4°C-weeks for Bleaching Watch).</div>'
                )
            else:
                coral_body = (
                    f'<div class="fb-hw-event" style="margin-top:0.4rem">'
                    f'🪸 Alert: <b>{alert_name}</b><br>'
                    f'Max DHW: <b>{max_dhw:.1f}°C-weeks</b> &nbsp;|&nbsp; '
                    f'Mean DHW: {mean_dhw:.1f}°C-weeks &nbsp;|&nbsp; '
                    f'Area under stress: <b>{pct_stress:.0f}%</b> of region'
                    f'</div>'
                )
            coral_html = (
                f'<div style="margin-top:1rem;padding-top:0.8rem;border-top:1px solid #d0e4f7">'
                f'<div style="font-size:0.75rem;font-weight:700;color:#5c7a99;text-transform:uppercase;'
                f'letter-spacing:0.8px;margin-bottom:0.4rem">🪸 Coral Bleaching Analysis</div>'
                f'{coral_body}'
                f'<div style="font-size:0.75rem;color:#5c7a99;margin-top:0.5rem">'
                f'Source: NOAA Coral Reef Watch · Degree Heating Weeks (DHW) · 5 km resolution</div>'
                f'</div>'
            )

    st.markdown(
        f'<div class="fb-result">'
        f'<div class="fb-question-tag">🔍 {result.question}</div>'
        f'<div class="fb-result-title">'
        f'📊 Result &nbsp;|&nbsp; '
        f'<span style="color:{C["muted"]};font-weight:600">'
        f'{" · ".join(params_label) if params_label else "Profile"}'
        f'</span>'
        f'&nbsp;|&nbsp; <span style="color:{C["muted"]}">'
        f'📍 {pq.get("start_date","?")} → {pq.get("end_date","?")}'
        f' &nbsp;({region_label})</span>'
        f'</div>'
        f'<div class="fb-answer">{answer_html}</div>'
        f'{hw_html}'
        f'{coral_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_qc, col_chart = st.columns([1.3, 2.7])
    with col_qc:
        _render_qc_panel(result, uid)
    with col_chart:
        _render_charts(result, uid)


def _render_qc_panel(result: EngineResult, uid: int = 0):
    for qc, lbl in [(result.qc_result, "🌡️ Temperature QC"),
                    (result.qc_result_salinity, "🧂 Salinity QC")]:
        if qc is None:
            continue
        short = lbl.split()[1]  # "Temperature" or "Salinity"
        st.markdown(f"**{lbl}**")
        st.markdown(_conf_badge(qc.confidence_label) +
                    f" <span style='font-size:0.82rem;color:{C['muted']}'>"
                    f" {qc.confidence_score:.0f}% pass rate</span>",
                    unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m1.metric("Value", f"{qc.value:.2f} {qc.unit}" if qc.value is not None else "—")
        m2.metric("Readings", f"{qc.n_good}/{qc.n_total}")

        if qc.n_delayed:
            st.caption(f"🔬 {qc.n_delayed} delayed-mode readings used")

        if qc.exclusions:
            by_test: dict = {}
            for exc in qc.exclusions:
                by_test.setdefault(exc.failed_test, []).append(exc)
            # Note: st.expander does not support key= in Streamlit 1.x
            # Uniqueness is handled by the uid suffix in the label
            with st.expander(f"ℹ️ {qc.n_bad} readings excluded [{uid}]",
                             expanded=False):
                for test, recs in by_test.items():
                    st.markdown(f"**{test.replace('_',' ').title()}** ({len(recs)})")
                    st.markdown(f"*{recs[0].explanation}*")
                    for r in recs[:3]:
                        st.markdown(
                            f"&nbsp;&nbsp;• Float **{r.platform}** · "
                            f"cycle {r.cycle} · {r.pres:.0f} dbar · "
                            f"val {r.value:.3f} · flag {r.qc_flag}"
                        )
                    if len(recs) > 3:
                        st.caption(f"… and {len(recs)-3} more")

        if qc.provenance:
            with st.expander(f"📋 Sample readings used [{uid}]", expanded=False):
                rows = [{"Float": p.platform, "Cycle": p.cycle,
                         "Depth": f"{p.pres:.0f}m", "Value": f"{p.value:.3f}",
                         "Mode": p.data_mode} for p in qc.provenance]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


def _render_charts(result: EngineResult, uid: int = 0):
    charts_shown = False

    # QC pie chart
    for qc, lbl in [(result.qc_result, "Temperature"),
                    (result.qc_result_salinity, "Salinity")]:
        if qc and qc.n_total > 0:
            fig = go.Figure(go.Pie(
                labels=["Good (QC 1–2)", "Excluded (QC 3–4)"],
                values=[qc.n_good, qc.n_bad],
                hole=0.55,
                marker_colors=[C['blue'], C['red']],
                textinfo="percent",
                textfont_size=11,
                showlegend=True,
            ))
            fig.update_layout(
                title=dict(text=f"{lbl} QC", font_size=13, x=0.5),
                height=220, margin=dict(l=5, r=5, t=35, b=5),
                paper_bgcolor=C['white'], plot_bgcolor=C['white'],
                legend=dict(orientation="h", y=-0.15, font_size=10),
            )
            st.plotly_chart(fig, use_container_width=True,
                            key=f"pie_{lbl}_{uid}")
            charts_shown = True

    # SST time-series with heatwave bands
    snap = getattr(result, "ocean_snapshot", None)
    sst_df = snap.sst_ts if snap and snap.sst_ts is not None and not snap.sst_ts.empty else None

    if sst_df is None and result.heatwave_result and st.session_state.use_synthetic:
        from data.sst_loader import make_synthetic_sst
        region = result.parsed_query.get("region", {})
        lat_c = (region.get("lat_min", 0) + region.get("lat_max", 20)) / 2
        lon_c = (region.get("lon_min", 60) + region.get("lon_max", 78)) / 2
        sst_df = make_synthetic_sst(lat_center=lat_c, lon_center=lon_c,
                                    start_date="2018-01-01", end_date="2023-12-31")

    hw = result.heatwave_result
    if sst_df is not None and not sst_df.empty and "sst" in sst_df.columns:
        sst_plot = sst_df.copy()
        sst_plot["date"] = pd.to_datetime(sst_plot["date"])
        cutoff = sst_plot["date"].max() - pd.DateOffset(years=2)
        sst_plot = sst_plot[sst_plot["date"] >= cutoff]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=sst_plot["date"], y=sst_plot["sst"],
            mode="lines", name="SST (°C)",
            line=dict(color=C['sky'], width=1.5),
            fill="tozeroy", fillcolor="rgba(30,136,229,0.07)",
        ))
        if hw and hw.overlapping_events:
            for ev in hw.overlapping_events:
                fig2.add_vrect(x0=ev.start_date, x1=ev.end_date,
                               fillcolor="rgba(183,28,28,0.12)", line_width=0,
                               annotation_text="🔥", annotation_position="top left",
                               annotation_font_size=12)
        fig2.update_layout(
            title=dict(text="SST Time Series", font_size=13, x=0.5),
            height=220, margin=dict(l=5, r=5, t=35, b=5),
            paper_bgcolor=C['white'], plot_bgcolor=C['bg'],
            xaxis_title="", yaxis_title="°C",
            yaxis=dict(gridcolor=C['border']),
            xaxis=dict(gridcolor=C['border']),
            showlegend=False, font_size=11,
        )
        st.plotly_chart(fig2, use_container_width=True,
                        key=f"sst_ts_{uid}")
        charts_shown = True

    if not charts_shown:
        st.markdown(
            f'<div class="fb-info-card" style="text-align:center;padding:2rem;color:{C["muted"]}">'
            '📊<br><b>Charts appear here after a query</b><br>'
            '<small>QC breakdown pie + SST time series</small>'
            '</div>',
            unsafe_allow_html=True,
        )


def _render_hw_panel(result: EngineResult):
    hw = result.heatwave_result
    if hw is None:
        return

    st.markdown("**🔥 Marine Heatwave**")
    if hw.heatwave_detected:
        for ev in hw.overlapping_events:
            d = ev.to_dict()
            st.markdown(
                f'<div class="fb-hw-event">'
                f'📅 <b>{d["start_date"]}</b> → <b>{d["end_date"]}</b><br>'
                f'⏱ {d["duration_days"]} days &nbsp;|&nbsp; '
                f'🌡 Peak <b>{d["peak_sst"]:.1f}°C</b><br>'
                f'📈 Anomaly: <b>{d["mean_anomaly"]:+.2f}°C</b> above climatology'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f'<div class="fb-no-hw">✅ No heatwave detected in this period.</div>',
            unsafe_allow_html=True,
        )
    st.caption(f"Method: Hobday et al. (2016) · {hw.climatology_years} yrs climatology")


def _render_snapshot_panel(snap):
    """Rich render of the full OceanDB snapshot from 'Load extra databases'."""
    st.markdown(
        f"<div style='background:linear-gradient(90deg,{C['navy']},{C['blue']});"
        f"padding:1.2rem 1.8rem;border-radius:14px;color:white;margin-bottom:1.2rem'>"
        f"<div style='font-size:1.15rem;font-weight:800;margin-bottom:0.3rem'>"
        f"🌐 External Ocean Databases — Full Report</div>"
        f"<div style='opacity:0.8;font-size:0.83rem'>"
        f"GEBCO Bathymetry &nbsp;·&nbsp; NOAA GDP Drifters &nbsp;·&nbsp; "
        f"NOAA Coral Reef Watch &nbsp;·&nbsp; NOAA HYCOM Model &nbsp;·&nbsp; "
        f"NOAA World Ocean Database &nbsp;·&nbsp; Copernicus CMEMS"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # ── Build metrics ──────────────────────────────────────────────────────
    metrics = []

    if snap.gebco_depth_m and snap.gebco_depth_m > 0:
        d = snap.gebco_depth_m
        zones = [(200,"Continental Shelf"),(1000,"Slope"),(3500,"Rise"),
                 (6000,"Abyssal Plain"),(1e9,"Hadal Zone")]
        zone = next(z for thr,z in zones if d < thr)
        metrics.append(("🌊 Ocean Depth", f"{d:.0f} m", zone, "GEBCO etopo180"))

    if snap.drifter_df is not None and not snap.drifter_df.empty:
        if "SST" in snap.drifter_df.columns and snap.drifter_df["SST"].notna().any():
            # Filter out fill values (AOML uses -999 as missing)
            sst_valid = snap.drifter_df["SST"][(snap.drifter_df["SST"] > -10) & (snap.drifter_df["SST"] < 50)]
            if not sst_valid.empty:
                n_d = snap.drifter_df["DRIFTER_ID"].nunique() \
                      if "DRIFTER_ID" in snap.drifter_df.columns else len(snap.drifter_df)
                metrics.append(("🛟 Drifter SST", f"{sst_valid.mean():.2f}°C",
                                f"{n_d} buoys", "NOAA GDP"))
        if "U_CURR" in snap.drifter_df.columns:
            # Filter out fill values before computing speed
            df_c = snap.drifter_df[
                (snap.drifter_df["U_CURR"].notna()) &
                (snap.drifter_df["V_CURR"].notna()) &
                (snap.drifter_df["U_CURR"].abs() < 10) &   # physical max ~3 m/s
                (snap.drifter_df["V_CURR"].abs() < 10)
            ].copy()
            if not df_c.empty:
                spd = float(np.sqrt(df_c["U_CURR"]**2+df_c["V_CURR"]**2).mean())
                if spd < 5:  # sanity check
                    metrics.append(("🌀 Surface Current", f"{spd:.2f} m/s",
                                    "from drifters", "NOAA GDP"))

    if snap.coral_df is not None and not snap.coral_df.empty:
        if "DHW" in snap.coral_df.columns and snap.coral_df["DHW"].notna().any():
            max_dhw   = float(snap.coral_df["DHW"].max())
            max_alert = int(snap.coral_df["ALERT_LEVEL"].max()) \
                        if "ALERT_LEVEL" in snap.coral_df.columns else 0
            lbl = {0:"No Stress ✅",1:"Watch ⚠️",2:"Warning ⚠️",
                   3:"Alert Lv1 🚨",4:"Alert Lv2 🚨"}
            metrics.append(("🪸 Coral Stress",f"DHW {max_dhw:.1f}°C·wks",
                            lbl.get(max_alert,"—"),"NOAA CRW"))

    if snap.hycom_df is not None and not snap.hycom_df.empty:
        if "TEMP" in snap.hycom_df.columns and snap.hycom_df["TEMP"].notna().any():
            metrics.append(("🌊 HYCOM Temp",
                            f"{snap.hycom_df['TEMP'].dropna().mean():.2f}°C",
                            "model surface","NOAA HYCOM"))
        if "U_CURR" in snap.hycom_df.columns:
            hdf = snap.hycom_df.dropna(subset=["U_CURR","V_CURR"])
            if not hdf.empty:
                spd = float(np.sqrt(hdf["U_CURR"]**2+hdf["V_CURR"]**2).mean())
                metrics.append(("🌀 Model Current",f"{spd:.2f} m/s",
                                "HYCOM surface","NOAA HYCOM"))

    if snap.wod_df is not None and not snap.wod_df.empty:
        metrics.append(("📚 WOD Profiles",
                        f"{len(snap.wod_df):,}",
                        "historical casts","NOAA WOD"))

    if snap.cmems_df is not None and not snap.cmems_df.empty:
        if "TEMP" in snap.cmems_df.columns and snap.cmems_df["TEMP"].notna().any():
            metrics.append(("🇪🇺 CMEMS Temp",
                            f"{snap.cmems_df['TEMP'].dropna().mean():.2f}°C",
                            "reanalysis","Copernicus"))

    # Render metrics in rows of 4
    if metrics:
        for row_start in range(0, len(metrics), 4):
            row = metrics[row_start:row_start+4]
            cols = st.columns(len(row))
            for i,(label,value,delta,source) in enumerate(row):
                with cols[i]:
                    st.metric(label, value, delta)
                    st.caption(f"📡 {source}")
    else:
        st.info(
            "No data returned from external databases for this region/period.\n\n"
            "This can happen if the region has limited coverage in HYCOM, WOD, or CMEMS.",
            icon="ℹ️"
        )

    # ── WOD depth profile chart ─────────────────────────────────────────────
    if snap.wod_df is not None and not snap.wod_df.empty and "TEMP" in snap.wod_df.columns:
        with st.expander("📚 World Ocean Database — historical T/S profiles", expanded=True):
            wod = snap.wod_df.dropna(subset=["PRES","TEMP"]).copy()
            if len(wod) > 3:
                import plotly.express as _px
                fig = _px.scatter(
                    wod, x="TEMP", y="PRES",
                    color="DATA_MODE" if "DATA_MODE" in wod.columns else None,
                    labels={"TEMP":"Temperature (°C)","PRES":"Depth (dbar)"},
                    title=f"WOD historical T profiles — {len(wod):,} observations",
                    height=320, color_discrete_map={"D":"#1565c0","R":"#e65100"},
                )
                fig.update_yaxes(autorange="reversed",
                                 title="Depth (dbar)")
                fig.update_layout(
                    margin=dict(l=5,r=5,t=35,b=5),
                    paper_bgcolor=C['white'], plot_bgcolor=C['bg'],
                    legend_title_text="Mode",
                )
                st.plotly_chart(fig, use_container_width=True, key=f"wod_{id(snap)}")
            st.caption(f"NOAA World Ocean Database — {len(wod):,} historical CTD/XBT casts")

    # ── HYCOM current chart ─────────────────────────────────────────────────
    if snap.hycom_df is not None and not snap.hycom_df.empty:
        hdf = snap.hycom_df
        if "U_CURR" in hdf.columns and "V_CURR" in hdf.columns:
            hdf2 = hdf.dropna(subset=["U_CURR","V_CURR"]).copy()
            if not hdf2.empty and "TIME" in hdf2.columns:
                with st.expander("🌀 NOAA HYCOM — model surface currents", expanded=False):
                    hdf2["speed"] = np.sqrt(hdf2["U_CURR"]**2 + hdf2["V_CURR"]**2)
                    hdf2["TIME"]  = pd.to_datetime(hdf2["TIME"])
                    daily = hdf2.groupby(hdf2["TIME"].dt.normalize())["speed"].mean().reset_index()
                    import plotly.graph_objects as _go
                    fig2 = _go.Figure(_go.Scatter(
                        x=daily["TIME"], y=daily["speed"],
                        mode="lines+markers",
                        line=dict(color=C['teal'], width=2),
                        name="Speed (m/s)",
                    ))
                    fig2.update_layout(
                        title="HYCOM surface current speed",
                        height=260, margin=dict(l=5,r=5,t=35,b=5),
                        paper_bgcolor=C['white'], plot_bgcolor=C['bg'],
                        yaxis_title="m/s", xaxis_title="",
                    )
                    st.plotly_chart(fig2, use_container_width=True, key=f"hycom_{id(snap)}")
                    st.caption("NOAA HYCOM Hybrid Coordinate Ocean Model — 1/12° resolution")

    # ── Coral bleaching alert ───────────────────────────────────────────────
    if snap.coral_df is not None and not snap.coral_df.empty:
        if "DHW" in snap.coral_df.columns and snap.coral_df["DHW"].notna().any():
            max_alert = int(snap.coral_df.get("ALERT_LEVEL",
                            snap.coral_df.assign(ALERT_LEVEL=0)["ALERT_LEVEL"]).max())
            if max_alert >= 1:
                max_dhw = float(snap.coral_df["DHW"].max())
                st.warning(
                    f"🪸 **Coral bleaching thermal stress detected** — "
                    f"max DHW = {max_dhw:.1f}°C-weeks (NOAA Coral Reef Watch). "
                    f"DHW ≥ 8 causes mass bleaching in most coral species.",
                    icon="🪸"
                )

    # ── Source report ───────────────────────────────────────────────────────
    if snap.source_report:
        statuses = list(snap.source_report.values())
        n_ok = sum(1 for s in statuses if "✅" in s)
        with st.expander(
            f"📋 Database status — {n_ok}/{len(statuses)} sources returned data",
            expanded=False,
        ):
            rows = []
            for db_name, status in snap.source_report.items():
                icon = "✅" if "✅" in status else ("❌" if "❌" in status else "⚠️")
                clean = status.replace("✅","").replace("❌","").replace("⚠️","").strip()
                rows.append({"Database": db_name, "Status": f"{icon} {clean}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE — MAP
# ─────────────────────────────────────────────────────────────────────────────
_IFREMER_TABLEDAP = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv"

_REGION_BOXES: dict[str, list] = {
    "Global":            [(-180, 180, -75,  75)],
    "Arabian Sea":       [(  55,  78,   8,  26)],
    "Bay of Bengal":     [(  80, 100,   5,  23)],
    "Indian Ocean":      [(  30, 120, -40,  30)],
    "Gulf of Mexico":    [( -98, -80,  18,  31)],
    "Caribbean Sea":     [( -85, -60,  10,  25)],
    "North Atlantic":    [( -80,   0,  25,  65)],
    "South Atlantic":    [( -60,  20, -60,  -5)],
    "Mediterranean Sea": [(  -6,  37,  30,  47)],
    "Red Sea":           [(  32,  44,  12,  30)],
    "Pacific Ocean":     [( 120, 180, -55,  65), (-180, -120, -55, 65)],
    "Coral Sea":         [( 142, 162, -26, -10)],
    "Southern Ocean":    [(-180, 180, -70, -45)],
}

def _fetch_float_positions(lon0,lon1,lat0,lat1,days=30):
    import requests, io as _io
    end   = (pd.Timestamp.now()-pd.Timedelta(days=5)).strftime("%Y-%m-%dT00:00:00Z")
    start = (pd.Timestamp.now()-pd.Timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
    url = (f"{_IFREMER_TABLEDAP}?platform_number,longitude,latitude,time"
           f"&longitude>={lon0:.2f}&longitude<={lon1:.2f}"
           f"&latitude>={lat0:.2f}&latitude<={lat1:.2f}"
           f"&time>={start}&time<={end}"
           '&orderByMax("platform_number,time")')
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    if len(lines) < 3:
        return pd.DataFrame(columns=["wmo","lon","lat","last_seen","hover"])
    df = pd.read_csv(_io.StringIO("\n".join([lines[0]] + lines[2:])))
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"platform_number":"wmo","longitude":"lon",
                             "latitude":"lat","time":"last_seen"})
    df["last_seen"] = pd.to_datetime(df["last_seen"], errors="coerce", utc=True).dt.tz_localize(None)
    df["hover"] = "Float " + df["wmo"].astype(str) + " · " + df["last_seen"].dt.strftime("%Y-%m-%d").astype(str)
    return df


def page_map():
    st.markdown(f"""
    <div style="background:linear-gradient(90deg,{C['navy']},{C['blue']});
                padding:1.8rem 2.5rem;border-radius:14px;margin-bottom:1.5rem;color:#fff">
      <h2 style="margin:0 0 0.3rem;font-size:1.6rem;font-weight:800;color:#fff">
        🗺️ Live Argo Float Positions</h2>
      <p style="margin:0;opacity:0.82;font-size:0.95rem">
        Most recent position of every active Argo float — fetched live from the Ifremer GDAC.</p>
    </div>
    """, unsafe_allow_html=True)

    regions = sorted(_REGION_BOXES.keys())
    regions = ["Global"] + [r for r in regions if r != "Global"]

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        region = st.selectbox("Region", regions, label_visibility="collapsed")
    with c2:
        days = st.selectbox("Time window", [7, 14, 30, 60], index=2,
                            format_func=lambda d: f"Last {d} days",
                            label_visibility="collapsed")
    with c3:
        clicked = st.button("🔄 Load / Refresh", type="primary", use_container_width=True)

    if clicked:
        st.session_state["map_load"] = True
        st.session_state["map_region_loaded"] = region
        st.session_state["map_days_loaded"] = days

    if not st.session_state.get("map_load"):
        st.info("🌐 Click **Load / Refresh** to fetch live Argo float positions from the Ifremer GDAC.", icon="🗺️")
        st.markdown("<br>", unsafe_allow_html=True)
        sc = st.columns(4)
        sc[0].metric("Global Argo floats", "~3,900", "active")
        sc[1].metric("Measurement depth", "0–2,000 m", "per profile")
        sc[2].metric("Cycle frequency", "~10 days", "per float")
        sc[3].metric("Parameters", "T, S, Pressure", "+ QC flags")
        return

    active_region = st.session_state.get("map_region_loaded", region)
    active_days   = st.session_state.get("map_days_loaded", days)
    boxes = _REGION_BOXES.get(active_region, [(-180, 180, -75, 75)])
    lon0, lon1, lat0, lat1 = boxes[0]
    lat_c, lon_c = (lat0+lat1)/2, (lon0+lon1)/2
    lat_span = abs(lat1-lat0)
    zoom = 1 if lat_span > 60 else (2 if lat_span > 30 else 4)

    with st.spinner(f"📡 Fetching Argo float positions for {active_region}…"):
        try:
            if st.session_state.use_synthetic:
                from data.argo_loader import make_synthetic_profiles
                df_raw = make_synthetic_profiles(lat_center=lat_c, lon_center=lon_c, n_floats=40)
                df = df_raw.drop_duplicates("PLATFORM_NUMBER").rename(
                    columns={"LATITUDE":"lat","LONGITUDE":"lon","PLATFORM_NUMBER":"wmo"})
                df["hover"] = "Float " + df["wmo"].astype(str) + " (synthetic)"
            else:
                frames = []
                for b in boxes:
                    try:
                        sub = _fetch_float_positions(*b, days=active_days)
                        if not sub.empty:
                            frames.append(sub)
                    except Exception as e:
                        logger.warning("Sub-box failed: %s", e)
                if not frames:
                    st.warning(f"No Argo floats found in **{active_region}** in the last {active_days} days.")
                    return
                df = pd.concat(frames).drop_duplicates("wmo")

            if df.empty:
                st.warning("No positions returned.")
                return

            fig = go.Figure(go.Scattermap(
                lat=df["lat"], lon=df["lon"],
                mode="markers",
                marker=dict(size=8, color=C['sky'], opacity=0.85),
                text=df["hover"],
                hovertemplate="%{text}<extra></extra>",
            ))
            fig.update_layout(
                map=dict(style="open-street-map",
                         center=dict(lat=lat_c, lon=lon_c), zoom=zoom),
                height=520, margin=dict(l=0, r=0, t=0, b=0),
                hoverlabel=dict(bgcolor=C['navy'], font_color="#fff", font_size=12),
            )
            st.plotly_chart(fig, use_container_width=True)

            mc = st.columns(4)
            mc[0].metric("Floats shown", len(df))
            mc[1].metric("Region", active_region)
            mc[2].metric("Active in last", f"{active_days} days")
            mc[3].metric("Source", "Argo GDAC")
            st.caption("Ifremer ERDDAP tabledap · Most recent position per float · "
                       + pd.Timestamp.now().strftime("%Y-%m-%d"))

        except Exception as exc:
            st.error(f"Map load failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE — DATABASES
# ─────────────────────────────────────────────────────────────────────────────
def page_databases():
    st.markdown(f"""
    <div style="background:linear-gradient(90deg,{C['navy']},{C['blue']});
                padding:1.8rem 2.5rem;border-radius:14px;margin-bottom:1.5rem;color:#fff">
      <h2 style="margin:0 0 0.3rem;font-size:1.6rem;font-weight:800;color:#fff">
        🛢️ Connected Ocean Databases</h2>
      <p style="margin:0;opacity:0.82;font-size:0.95rem">
        8 real-time scientific databases — all publicly accessible, no account required
        except CMEMS full catalogue.</p>
    </div>
    """, unsafe_allow_html=True)

    databases = [
        {"icon":"🌐","name":"Argo Global Data Assembly Centre (GDAC)",
         "org":"Ifremer (France) / UCSD (USA)","status":"live",
         "vars":["In-situ Temperature","Salinity","Pressure/Depth","QC Flags","Float ID & Cycle","Data Mode R/D"],
         "desc":"The world's largest ocean observation network — ~4,000 autonomous floats measuring temperature and salinity from surface to 2,000 m. FloatAi uses argopy in expert mode to preserve all QC flags.",
         "access":"argopy → Ifremer ERDDAP (no auth)","url":"https://argo.ucsd.edu",
         "cite":"Argo (2000). Argo float data. SEANOE. doi:10.17882/42182"},
        {"icon":"🛰️","name":"NOAA OISST v2.1 — Daily Sea Surface Temperature",
         "org":"NOAA National Centers for Environmental Information","status":"live",
         "vars":["Daily SST (°C)","0.25°×0.25° grid","1981–present","Satellite + buoy blended"],
         "desc":"Primary SST product for marine heatwave detection. Combines AVHRR satellite with ship/buoy observations via optimal interpolation. Used by FloatAi's Hobday 2016 heatwave algorithm.",
         "access":"Direct HTTPS → CoastWatch ERDDAP netCDF4 (no auth)","url":"https://www.ncei.noaa.gov/products/optimum-interpolation-sst",
         "cite":"Huang et al. (2020). doi:10.25921/RE9P-PT57"},
        {"icon":"🛟","name":"NOAA Global Drifter Program (GDP)",
         "org":"NOAA Atlantic Oceanographic & Meteorological Laboratory","status":"live",
         "vars":["Surface SST (°C)","Eastward current (m/s)","Northward current (m/s)","Drifter ID"],
         "desc":"~1,300 freely drifting surface buoys measuring SST and tracking ocean surface currents. Provides independent in-situ SST for cross-validation. Coverage: 1979–present.",
         "access":"AOML ERDDAP tabledap CSV (no auth)","url":"https://www.aoml.noaa.gov/phod/gdp/",
         "cite":"Lumpkin & Speer (2007). Global ocean meridional overturning. JPO."},
        {"icon":"🌀","name":"NOAA HYCOM — Hybrid Coordinate Ocean Model",
         "org":"US Navy / NOAA NCEI","status":"live",
         "vars":["Sea surface temperature","Salinity","U/V currents (m/s)","Sea surface height (m)"],
         "desc":"High-resolution (1/12° ≈ 8 km) global ocean model. Surface current vectors, model temperature/salinity, and SSH anomaly as a proxy for mesoscale eddies. Coverage: 1994–present.",
         "access":"NOAA NCEI ERDDAP griddap netCDF4 (no auth)","url":"https://www.hycom.org",
         "cite":"Chassignet et al. (2007). The HYCOM ocean model. JPO."},
        {"icon":"📚","name":"NOAA World Ocean Database (WOD)",
         "org":"NOAA National Centers for Environmental Information","status":"live",
         "vars":["Temperature (°C)","Salinity (PSU)","WOD QC flags","Cast ID","Depth/pressure"],
         "desc":"World's largest public ocean profile collection — 18+ million T/S casts from 1772 to present, covering CTD, XBT, Argo, and moorings. WOD profiles feed FloatAi's QC module directly.",
         "access":"CoastWatch / NCEI ERDDAP tabledap CSV (no auth)","url":"https://www.ncei.noaa.gov/products/world-ocean-database",
         "cite":"Boyer et al. (2018). World Ocean Database 2018. NOAA Atlas."},
        {"icon":"🪸","name":"NOAA Coral Reef Watch (CRW)",
         "org":"NOAA / CoRTAD","status":"live",
         "vars":["Degree Heating Weeks (°C-weeks)","SST (°C)","SST Anomaly (°C)","Bleaching Alert 0–4"],
         "desc":"5 km satellite thermal stress monitoring for coral reefs globally. DHW ≥ 4 = Watch; ≥ 8 = Alert Level 1; ≥ 12 = Alert Level 2. Coverage: tropical oceans, 1985–present.",
         "access":"CoastWatch ERDDAP griddap netCDF4 (no auth)","url":"https://coralreefwatch.noaa.gov",
         "cite":"Liu et al. (2014). NOAA Coral Reef Watch. Remote Sensing."},
        {"icon":"🌊","name":"GEBCO 2022 — General Bathymetric Chart of the Oceans",
         "org":"GEBCO / British Oceanographic Data Centre","status":"live",
         "vars":["Ocean depth (m)","Elevation (m)","15-arc-second resolution"],
         "desc":"Most comprehensive public ocean depth dataset — 15″ resolution (~450 m) from multibeam sonar, gravity models, and historical soundings. Answers 'how deep is the ocean here?'",
         "access":"CoastWatch ERDDAP griddap netCDF4 (no auth)","url":"https://www.gebco.net",
         "cite":"GEBCO Compilation Group (2022). GEBCO 2022 Grid. doi:10.5285/e0f0bb80"},
        {"icon":"🇪🇺","name":"Copernicus Marine Service (CMEMS)",
         "org":"EU Copernicus / Mercator Ocean","status":"opt",
         "vars":["Temperature (°C)","Salinity (PSU)","U/V currents","Sea level","Chlorophyll-a"],
         "desc":"European-quality global ocean reanalysis (1/12°, 1993–present). Accessed via IFREMER public mirror (no auth for reanalysis) or full toolbox (free account). Set COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD for full access.",
         "access":"IFREMER ERDDAP mirror (no auth) OR copernicusmarine toolbox (free)","url":"https://marine.copernicus.eu",
         "cite":"Copernicus Marine Service (2023). Global Ocean Physics Reanalysis. doi:10.48670/moi-00021"},
    ]

    for db in databases:
        with st.container():
            c_icon, c_main = st.columns([0.5, 9.5])
            with c_icon:
                st.markdown(f"<div style='font-size:2.5rem;padding-top:0.3rem'>{db['icon']}</div>",
                            unsafe_allow_html=True)
            with c_main:
                st.markdown(
                    f"<div class='fb-about-card'>"
                    f"<h3 style='display:flex;justify-content:space-between;align-items:center'>"
                    f"<span>{db['name']}</span>"
                    f"<span style='font-size:0.75rem;font-weight:600;"
                    f"color:{'#1b7a34' if db['status']=='live' else '#b35c00'}'>"
                    f"{'● Live' if db['status']=='live' else '● Optional (free account)'}</span>"
                    f"</h3>"
                    f"<div style='font-size:0.82rem;color:{C['muted']};margin-bottom:0.6rem'>{db['org']}</div>"
                    f"<p style='font-size:0.88rem;color:{C['text']};margin-bottom:0.7rem;line-height:1.6'>{db['desc']}</p>"
                    f"<div style='display:flex;flex-wrap:wrap;gap:0.3rem;margin-bottom:0.7rem'>"
                    + "".join(f'<span class="fb-tag">{v}</span>' for v in db["vars"]) +
                    f"</div>"
                    f"<div style='font-size:0.78rem;color:{C['muted']}'>"
                    f"<b>Access:</b> {db['access']} &nbsp;|&nbsp; "
                    f"<b>Citation:</b> {db['cite']} &nbsp;|&nbsp; "
                    f"<a href='{db['url']}' target='_blank' style='color:{C['sky']}'>{db['url']}</a>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
        st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
def page_about():
    st.markdown(f"""
    <div style="background:linear-gradient(90deg,{C['navy']},{C['blue']});
                padding:1.8rem 2.5rem;border-radius:14px;margin-bottom:1.5rem;color:#fff">
      <h2 style="margin:0 0 0.3rem;font-size:1.6rem;font-weight:800;color:#fff">ℹ️ About FloatAi</h2>
      <p style="margin:0;opacity:0.82;font-size:0.95rem">
        Open-source ocean intelligence platform — quality-controlled, explainable, and grounded in peer-reviewed science.</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(f"""
        <div class="fb-about-card">
          <h3>🎯 What is FloatAi?</h3>
          <p style="font-size:0.9rem;color:{C['text']};line-height:1.7;margin:0">
            FloatAi is an ocean science assistant that answers plain-English questions
            about the ocean using real data from 8 public scientific databases.
            Unlike tools that silently use raw sensor values, FloatAi checks every
            reading against its official Argo QC flag and explains <em>why</em> any data
            was excluded — naming the specific failed test in plain language.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="fb-about-card">
          <h3>⚙️ Three core innovations</h3>
          <div class="fb-timeline">
            <div class="fb-timeline-step">
              <strong>1. Explainable QC Reasoning</strong><br>
              Every Argo reading is checked against its official QC flag. Excluded data is
              explained by test name (Spike, Gradient, Range, Density Inversion) in plain language
              — not silently dropped.
            </div>
            <div class="fb-timeline-step">
              <strong>2. Cross-Dataset Reasoning</strong><br>
              Compound questions ("was this unusual?") are answered by comparing Argo readings
              against NOAA OISST climatology using the published Hobday et al. (2016) marine
              heatwave algorithm.
            </div>
            <div class="fb-timeline-step">
              <strong>3. Benchmarked Trust</strong><br>
              Accuracy is measured against a naive (QC-blind) baseline using narrow
              single-profile queries. Real accuracy numbers, not unverified claims.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown(f"""
        <div class="fb-about-card">
          <h3>🔬 Scientific methods</h3>
          <ul style="font-size:0.88rem;line-height:1.9;color:{C['text']};padding-left:1.2rem">
            <li><b>Marine Heatwave Detection:</b> Hobday et al. (2016) — 11-day climatology,
              90th-percentile threshold, ≥5 consecutive days.<br>
              <a href="https://doi.org/10.1016/j.pocean.2015.12.014" target="_blank"
                 style="color:{C['sky']}">doi:10.1016/j.pocean.2015.12.014</a></li>
            <li><b>Argo QC Tests:</b> Argo Data Management (2022). Quality control manual.
              <a href="https://doi.org/10.13155/33951" target="_blank"
                 style="color:{C['sky']}">doi:10.13155/33951</a></li>
            <li><b>SST (OISST):</b> Huang et al. (2020).
              <a href="https://doi.org/10.25921/RE9P-PT57" target="_blank"
                 style="color:{C['sky']}">doi:10.25921/RE9P-PT57</a></li>
            <li><b>Bathymetry:</b> GEBCO 2022 Grid.
              <a href="https://doi.org/10.5285/e0f0bb80-ab44-2739-e053-6c86abc0289c" target="_blank"
                 style="color:{C['sky']}">doi:10.5285/…</a></li>
            <li><b>Coral bleaching:</b> Liu et al. (2014). NOAA Coral Reef Watch.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="fb-about-card">
          <h3>🏗️ Architecture</h3>
          <div style="font-size:0.83rem;line-height:2.0;font-family:monospace;color:{C['text']}">
            <b style="color:{C['blue']}">nlu/</b> llm_parser.py · rule_parser.py<br>
            <b style="color:{C['blue']}">core/</b> query_engine.py · qc_module.py · heatwave_module.py<br>
            <b style="color:{C['blue']}">data/</b> argo_loader · sst_loader · hycom_loader<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;drifter_loader · wod_loader · coral_watch_loader<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;gebco_loader · cmems_loader · ocean_db<br>
            <b style="color:{C['blue']}">eval/</b> benchmark.py<br>
            <b style="color:{C['blue']}">app.py</b> — this website (Streamlit)
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Setup + config
    st.markdown("<hr style='border-color:#d0e4f7;margin:2rem 0'>", unsafe_allow_html=True)
    st.markdown("### 🚀 Setup & Configuration")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.code("""# Install
pip install -r requirements.txt

# Optional: LLM parsing (smarter queries)
$env:GOOGLE_API_KEY = "your-key-here"

# Run the website
cd FloatAi
streamlit run app.py

# Run benchmark (offline)
python -m eval.benchmark --synthetic""", language="bash")
    with sc2:
        st.code("""# Use the Python API directly
from core.query_engine import QueryEngine

engine = QueryEngine(use_synthetic=False)
result = engine.answer(
    "What was the temperature in the "
    "Arabian Sea in August 2023, "
    "and was there a heatwave?"
)
print(result.to_answer_text())""", language="python")

    st.markdown("### 🔧 Current status")
    s1, s2, s3 = st.columns(3)
    with s1:
        if llm_parser.is_available():
            st.success(f"✅ LLM parser: **{llm_parser.active_provider()}** active")
        else:
            st.info("📐 Rule-based parser active\n\nSet GOOGLE_API_KEY for LLM mode")
    with s2:
        import os
        if os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME"):
            st.success("✅ CMEMS: full access")
        else:
            st.info("ℹ️ CMEMS: public mirror\n\nSet credentials for full catalogue")
    with s3:
        st.success("✅ Argo, OISST, HYCOM, WOD,\nCoral Reef Watch, GEBCO, GDP\n— no auth needed")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
def _footer():
    st.markdown(f"""
    <div class="fb-footer">
      <strong>FloatAi — Ocean Intelligence Platform</strong><br>
      Data sources: Argo GDAC · NOAA OISST · NOAA GDP Drifters · NOAA HYCOM ·
      NOAA World Ocean Database · NOAA Coral Reef Watch · GEBCO 2022 · Copernicus CMEMS<br>
      Heatwave method: <a href="https://doi.org/10.1016/j.pocean.2015.12.014">Hobday et al. (2016)</a> ·
      SST: <a href="https://doi.org/10.25921/RE9P-PT57">Huang et al. (2020)</a><br>
      Built with Streamlit · Python · argopy · plotly
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patch EngineResult
# ─────────────────────────────────────────────────────────────────────────────
def _use_synthetic_hint(self):
    return st.session_state.get("use_synthetic", False)
EngineResult.use_synthetic_hint = _use_synthetic_hint


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# PAGE — OCEAN STRESS ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def page_stress_engine():
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{C['navy']} 0%,#1a3a5c 50%,{C['teal']} 100%);
                padding:2.2rem 2.5rem;border-radius:16px;margin-bottom:1.8rem;color:white">
      <div style="font-size:1.7rem;font-weight:900;margin-bottom:0.4rem">
        🌡️ Ocean Stress Engine
      </div>
      <div style="opacity:0.82;font-size:0.95rem">
        Compute an explainable <b>thermohaline stress score</b> (0–100) for any ocean
        region and time window — with per-variable driver breakdown, data quality
        metrics, and confidence level. Powered by Argo float data + WOA23 climatology.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Check OSE is available ────────────────────────────────────────────
    try:
        from ocean_stress_engine.engine import compute_stress as _ose_compute
        from ocean_stress_engine.inputs import list_regions as _ose_list_regions
        _ose_available = True
    except Exception as _e:
        _ose_available = False
        _ose_err = str(_e)

    if not _ose_available:
        st.error(
            f"⚠️ Ocean Stress Engine package not found.\n\n"
            f"Make sure `ocean_stress_engine/` is inside the `FloatAi/` directory.\n\n"
            f"Detail: `{_ose_err}`"
        )
        return

    # ── Input form ────────────────────────────────────────────────────────
    st.markdown("**Configure your stress query:**")

    with st.form("ose_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📍 Ocean Region**")
            region_names = [r.title() for r in _ose_list_regions()]
            region_choice = st.selectbox(
                "", options=["(custom box)"] + region_names,
                label_visibility="collapsed", key="ose_region"
            )

        with col2:
            st.markdown("**📅 Time Window**")
            tw_col1, tw_col2 = st.columns(2)
            with tw_col1:
                sd = st.date_input("From", value=pd.Timestamp("2023-07-01").date(),
                                   min_value=pd.Timestamp("1981-01-01").date(),
                                   label_visibility="collapsed", key="ose_sd")
            with tw_col2:
                ed = st.date_input("To", value=pd.Timestamp("2023-07-31").date(),
                                   min_value=pd.Timestamp("1981-01-01").date(),
                                   label_visibility="collapsed", key="ose_ed")

        # Custom box (shown only when selected)
        if region_choice == "(custom box)":
            st.markdown("**Custom bounding box:**")
            bb1, bb2, bb3, bb4 = st.columns(4)
            with bb1: lat_min = st.number_input("Lat min", value=8.0,  min_value=-90.0, max_value=90.0)
            with bb2: lat_max = st.number_input("Lat max", value=26.0, min_value=-90.0, max_value=90.0)
            with bb3: lon_min = st.number_input("Lon min", value=55.0, min_value=-180.0, max_value=360.0)
            with bb4: lon_max = st.number_input("Lon max", value=78.0, min_value=-180.0, max_value=360.0)
        else:
            lat_min = lat_max = lon_min = lon_max = None

        opt1, opt2, opt3 = st.columns(3)
        with opt1:
            st.markdown("**🔬 Depth Layer**")
            depth_layer = st.selectbox(
                "", ["surface", "mid", "deep"],
                label_visibility="collapsed", key="ose_depth"
            )
        with opt2:
            st.markdown("**🌿 Ecosystem Type**")
            ecosystem = st.selectbox(
                "", ["general", "coral_reef", "fisheries"],
                label_visibility="collapsed", key="ose_eco"
            )
        with opt3:
            st.markdown("**📐 Baseline Period**")
            baseline = st.selectbox(
                "", ["2005-2020", "2005-2015", "2010-2020"],
                label_visibility="collapsed", key="ose_base"
            )

        use_synth = st.toggle("🧪 Offline / synthetic mode (no network)",
                              value=False, key="ose_synth")

        submitted = st.form_submit_button(
            "🌡️  Compute Stress Score", type="primary", use_container_width=True
        )

    # ── Run OSE ──────────────────────────────────────────────────────────
    if submitted:
        if region_choice == "(custom box)":
            region_arg = {
                "lat_min": lat_min, "lat_max": lat_max,
                "lon_min": lon_min, "lon_max": lon_max,
            }
        else:
            region_arg = region_choice.lower()

        time_window = f"{sd.isoformat()}/{ed.isoformat()}"

        with st.spinner("🌡️ Computing thermohaline stress score… (fetching Argo + climatology)"):
            try:
                result = _ose_compute(
                    region=region_arg,
                    time_window=time_window,
                    depth_layer=depth_layer,
                    ecosystem_type=ecosystem,
                    baseline_period=baseline,
                    use_synthetic=use_synth,
                )
                st.session_state["ose_result"]   = result
                st.session_state["ose_no_data"]  = False
                st.session_state["ose_error"]    = None

            except RuntimeError as exc:
                err = str(exc)
                # ── No Argo data in region → data gap, not a bug ─────────
                if "No Argo float data found" in err or "No Argo floats were active" in err:
                    st.session_state["ose_no_data"] = True
                    st.session_state["ose_error"]   = err
                    # Auto-retry with synthetic data so the user still sees
                    # a demo score with a clear caveat
                    try:
                        result_synth = _ose_compute(
                            region=region_arg,
                            time_window=time_window,
                            depth_layer=depth_layer,
                            ecosystem_type=ecosystem,
                            baseline_period=baseline,
                            use_synthetic=True,
                        )
                        st.session_state["ose_result"] = result_synth
                    except Exception:
                        st.session_state["ose_result"] = None
                else:
                    st.session_state["ose_no_data"] = False
                    st.session_state["ose_error"]   = err
                    st.session_state["ose_result"]  = None

            except ValueError as exc:
                st.session_state["ose_result"]  = None
                st.session_state["ose_no_data"] = False
                st.session_state["ose_error"]   = f"Input error: {exc}"

            except Exception as exc:
                st.session_state["ose_result"]  = None
                st.session_state["ose_no_data"] = False
                st.session_state["ose_error"]   = str(exc)

    # ── Status banners ────────────────────────────────────────────────────
    ose_no_data = st.session_state.get("ose_no_data", False)
    ose_error   = st.session_state.get("ose_error")

    if ose_no_data and ose_error:
        st.info(
            f"🌊 **No Argo float data in this region** — this is a real oceanographic "
            f"data gap, not an error.\n\n"
            f"The Argo network does not cover semi-enclosed seas (Andaman Sea, Red Sea, "
            f"Persian Gulf, Mediterranean) equally. "
            f"**Showing a synthetic demo score below** so you can explore the interface. "
            f"Switch to a well-covered region (Arabian Sea, Coral Sea, Indian Ocean) "
            f"for real results.",
            icon="ℹ️",
        )
    elif ose_error and not ose_no_data:
        st.error(f"❌ {ose_error}")

    # ── Render result (survives rerenders) ────────────────────────────────
    result = st.session_state.get("ose_result")
    if result is None:
        return

    st.markdown("---")

    # Show synthetic caveat banner if data gap triggered auto-fallback
    if st.session_state.get("ose_no_data"):
        st.warning(
            "⚠️ **Synthetic demo data** — no real Argo floats in this region. "
            "Score below is for interface demonstration only. "
            "Try **Arabian Sea**, **Coral Sea**, or **Bay of Bengal** for real results.",
            icon="🧪",
        )

    # ── Score headline ────────────────────────────────────────────────────
    cat_color = {
        "LOW":      C["green"],
        "MODERATE": C["amber"],
        "HIGH":     "#c8580a",
        "SEVERE":   C["red"],
    }.get(result.category, C["muted"])
    cat_bg = {
        "LOW": "#e8f5e9", "MODERATE": "#fff3e0",
        "HIGH": "#fce4d6", "SEVERE": "#ffebee",
    }.get(result.category, "#f5f5f5")
    conf_color = {
        "HIGH": C["green"], "MEDIUM": C["amber"], "LOW": C["red"]
    }.get(result.confidence, C["muted"])

    # Three-column headline
    h1, h2, h3 = st.columns([2, 1.2, 1])
    with h1:
        st.markdown(
            f'<div style="background:{C["white"]};border:1px solid {C["border"]};'
            f'border-radius:14px;padding:1.4rem 1.6rem;">'
            f'<div style="font-size:0.72rem;font-weight:700;color:{C["muted"]};'
            f'text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px">'
            f'🌡️ Thermohaline Stress Score</div>'
            f'<div style="display:flex;align-items:center;gap:1rem">'
            f'<div style="font-size:3.5rem;font-weight:900;color:{cat_color};line-height:1">'
            f'{result.stress_score:.1f}</div>'
            f'<div>'
            f'<div style="font-size:0.75rem;color:{C["muted"]}">/ 100</div>'
            f'<div style="background:{cat_bg};color:{cat_color};border-radius:20px;'
            f'padding:2px 14px;font-size:0.82rem;font-weight:700;margin-top:4px">'
            f'{result.category}</div>'
            f'</div></div>'
            f'<div style="font-size:0.8rem;color:{C["muted"]};margin-top:0.5rem">'
            f'📍 {result.region_name} &nbsp;|&nbsp; '
            f'📅 {result.period_start} → {result.period_end} &nbsp;|&nbsp; '
            f'{result.depth_layer_lbl}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            f'<div style="background:{C["white"]};border:1px solid {C["border"]};'
            f'border-radius:14px;padding:1.2rem 1.4rem;height:100%">'
            f'<div style="font-size:0.72rem;font-weight:700;color:{C["muted"]};'
            f'text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.6rem">'
            f'Confidence</div>'
            f'<div style="font-size:1.8rem;font-weight:900;color:{conf_color}">'
            f'{result.confidence}</div>'
            f'<div style="font-size:0.73rem;color:{C["muted"]};margin-top:0.3rem">'
            f'{result.confidence_score:.0f}% factors HIGH</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with h3:
        st.markdown(
            f'<div style="background:{C["white"]};border:1px solid {C["border"]};'
            f'border-radius:14px;padding:1.2rem 1.4rem;height:100%">'
            f'<div style="font-size:0.72rem;font-weight:700;color:{C["muted"]};'
            f'text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.6rem">'
            f'Data</div>'
            f'<div style="font-size:0.88rem;color:{C["text"]}">'
            f'<b>{result.n_floats}</b> floats<br>'
            f'<b>{result.n_profiles}</b> profiles<br>'
            f'<b>{result.n_readings_total}</b> readings</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Variable driver bars ──────────────────────────────────────────────
    left, right = st.columns([1.6, 1])

    with left:
        st.markdown(f"**📊 Variable Drivers**")
        for d in result.drivers:
            direction = "▲" if d.anomaly > 0 else "▼"
            bar_pct   = int(d.sub_score)
            st.markdown(
                f'<div style="margin-bottom:0.9rem">'
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:0.82rem;margin-bottom:3px">'
                f'<span style="font-weight:700;color:{C["navy"]}">'
                f'{d.variable.title()}</span>'
                f'<span style="color:{C["muted"]}">'
                f'obs {d.observed:.2f}{d.unit} '
                f'{direction} {abs(d.anomaly):.2f} from baseline '
                f'(z={d.anomaly_zscore:+.2f}σ) &nbsp; w={d.weight:.0%}'
                f'</span>'
                f'<span style="font-weight:800;color:{cat_color}">{d.sub_score:.0f}</span>'
                f'</div>'
                f'<div style="background:#e8f0fe;border-radius:6px;height:10px">'
                f'<div style="background:{cat_color};width:{bar_pct}%;'
                f'height:10px;border-radius:6px"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        if result.variables_missing:
            st.markdown(
                f'<div style="font-size:0.76rem;color:{C["muted"]};'
                f'background:#f8fafd;border:1px solid {C["border"]};'
                f'border-radius:8px;padding:0.5rem 0.8rem;margin-top:0.4rem">'
                f'⚠️ <b>Missing variables (weight redistributed):</b> '
                f'{", ".join(result.variables_missing)}'
                f'</div>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("**📋 Plain-English Summary**")
        # Clean ** markers → display as-is (already readable)
        import re as _re
        summary_clean = _re.sub(r'\*\*(.*?)\*\*', r'\1', result.summary_text())
        for para in summary_clean.split("\n\n"):
            st.caption(para.strip())

    # ── Data quality + caveats ────────────────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    dq_col, cav_col = st.columns(2)

    with dq_col:
        excl_pct = (
            round(result.n_readings_excluded / result.n_readings_total * 100, 1)
            if result.n_readings_total > 0 else 0.0
        )
        st.markdown("**🔬 Data Quality**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Floats",    result.n_floats)
        m2.metric("Profiles",  result.n_profiles)
        m3.metric("QC Excl.", f"{excl_pct:.0f}%")
        if result.exclusion_reasons:
            reasons_str = " · ".join(f"{k}: {v}" for k, v in result.exclusion_reasons.items())
            st.caption(f"Exclusion reasons: {reasons_str}")
        st.caption(f"Baseline: {result.baseline_source} · {result.baseline_period}")

    with cav_col:
        all_notes = result.caveats + result.confidence_reasons
        st.markdown("**⚠️ Caveats & Confidence**")
        if all_notes:
            for note in all_notes:
                st.caption(f"• {note}")
        else:
            st.caption("No caveats — data quality is sufficient.")

    # ── Full JSON expander ────────────────────────────────────────────────
    with st.expander("📄 Full JSON output", expanded=False):
        st.code(result.to_json(), language="json")

    # ── HTML card expander ────────────────────────────────────────────────
    with st.expander("🎨 Dashboard card (HTML)", expanded=False):
        st.markdown(result.dashboard_card(), unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────────────────────
# FAE SUMMARY HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _render_fae_summary(result, lang: str, conf_color: str):
    """Render a plain-English final advisory paragraph summarising all
    available data into one actionable recommendation for the fisher."""
    # Gather available indicator values
    sst_val   = result.indicators.get("sst")
    curr_val  = result.indicators.get("currents")
    chl_val   = result.indicators.get("productivity")
    front_val = result.indicators.get("thermal_front")
    bathy_val = result.indicators.get("bathymetry")

    parts = []

    # SST summary
    if sst_val and sst_val.status in ("ok", "stale"):
        v = sst_val.value or {}
        sst_c  = v.get("sst_c")
        anom   = v.get("anomaly_c")
        hw     = v.get("heatwave", False)
        if sst_c is not None:
            anom_str = ""
            if anom is not None:
                direction = "above" if anom >= 0 else "below"
                anom_str  = f", which is {abs(anom):.1f}°C {direction} the seasonal average"
            sst_part = f"Sea surface temperature near **{result.location_name}** is **{sst_c:.1f}°C**{anom_str}."
            if hw:
                sst_part += " A **marine heatwave** is currently active in this area."
            parts.append(sst_part)

    # Currents summary
    if curr_val and curr_val.status == "ok":
        v   = curr_val.value or {}
        spd = v.get("speed_ms")
        dir_c = v.get("direction_cardinal", "")
        if spd is not None:
            strength = "light" if spd < 0.3 else ("moderate" if spd < 0.6 else "strong")
            parts.append(
                f"Surface currents are **{strength}** ({spd:.2f} m/s toward {dir_c}) "
                f"based on model forecast data."
            )

    # Front summary
    if front_val and front_val.status == "ok":
        v = front_val.value or {}
        if v.get("detected"):
            dist = v.get("dist_km", 0)
            strength = v.get("strength", "")
            parts.append(
                f"A **{strength} temperature front** is detected approximately "
                f"{dist:.0f} km from this location, which may influence local fishing conditions."
            )

    # Productivity summary
    if chl_val and chl_val.status == "ok":
        v   = chl_val.value or {}
        cat = v.get("category", "")
        if cat and cat != "no_data":
            cat_desc = {"HIGH": "high — suggesting good nutrient availability",
                        "MODERATE": "moderate",
                        "LOW": "low — suggesting reduced nutrient availability"}.get(cat, cat)
            parts.append(f"Ocean productivity is **{cat_desc}** based on satellite data.")

    # Bathymetry context
    if bathy_val and bathy_val.status == "ok":
        v = bathy_val.value or {}
        depth = v.get("depth_m")
        if depth is not None:
            zone = "shallow coastal shelf" if depth < 200 else "deeper offshore waters"
            parts.append(
                f"The seabed near this location is approximately **{depth:.0f} m** deep "
                f"({zone})."
            )

    # If no data at all
    if not parts:
        return

    # Compose the summary
    summary_text = " ".join(parts)

    # Confidence note
    conf_note = {
        "HIGH":   "Data confidence is **HIGH** — conditions are well-measured.",
        "MEDIUM": "Data confidence is **MEDIUM** — some indicators could not be retrieved.",
        "LOW":    "Data confidence is **LOW** — use this advisory with caution.",
    }.get(result.confidence.label, "")

    # Disclaimer reminder
    disclaimer = (
        "_This advisory describes ocean conditions only. "
        "It does not predict fish locations or declare any trip safe._"
    )

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#f0f7ff,#e8f5e9);'
        f'border:1px solid #b3d9c4;border-left:5px solid #2e8b57;'
        f'border-radius:0 14px 14px 0;padding:1.2rem 1.6rem;'
        f'margin-top:0.5rem">'
        f'<div style="font-size:0.75rem;font-weight:700;color:#1b5c3a;'
        f'text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.5rem">'
        f'🎣 Fisher Advisory Summary</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(summary_text)
    if conf_note:
        st.markdown(conf_note)
    st.markdown(disclaimer)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE — FISHER ADVISORY ENGINE (FAE integration — optional)
# Wrapped in try/except so a missing FAE package never breaks FloatAi.
# ─────────────────────────────────────────────────────────────────────────────
def page_fisher_advisory():
    try:
        from fisher_advisory_engine.inputs import list_ports as _fae_list_ports
        _fae_available = True
    except Exception as _e:
        _fae_available = False
        _fae_err = str(_e)

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{C['navy']} 0%,#1a5c3a 50%,#2e8b57 100%);
                padding:2.2rem 2.5rem;border-radius:16px;margin-bottom:1.8rem;color:white">
      <div style="font-size:1.7rem;font-weight:900;margin-bottom:0.4rem">
        🎣 Fisher Advisory Engine
      </div>
      <div style="opacity:0.82;font-size:0.95rem">
        Plain-language sea-condition advisories for coastal fishers in Tamil Nadu.
        Describes ocean conditions — never predicts fish locations or declares trips safe.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not _fae_available:
        st.error(
            f"⚠️ Fisher Advisory Engine package not found.\n\n"
            f"Make sure `fisher_advisory_engine/` is inside the `FloatAi/` directory.\n\n"
            f"Detail: `{_fae_err}`"
        )
        return

    # ── Input form ────────────────────────────────────────────────────────
    st.markdown("**Configure your advisory:**")

    with st.form("fae_form"):
        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown("**📍 Landing Centre**")
            port_list = ["(lat/lon point)"] + [
                p.get("name", "") for p in _fae_list_ports()
                if p.get("name", "").strip() and not p["name"].startswith("#")
            ]
            port_choice = st.selectbox(
                "", options=port_list,
                label_visibility="collapsed", key="fae_port"
            )
        with fc2:
            st.markdown("**📅 Day**")
            day_choice = st.selectbox(
                "", ["today", "tomorrow", "last 7 days"],
                label_visibility="collapsed", key="fae_day"
            )

        if port_choice == "(lat/lon point)":
            lc1, lc2 = st.columns(2)
            with lc1:
                fae_lat = st.number_input("Latitude",  value=10.77, min_value=-90.0, max_value=90.0)
            with lc2:
                fae_lon = st.number_input("Longitude", value=79.84, min_value=-180.0, max_value=180.0)
        else:
            fae_lat = fae_lon = None

        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            trip_type = st.selectbox("Trip type", ["nearshore", "offshore"],
                                     label_visibility="visible", key="fae_trip")
        with tc2:
            _lang_display = {
                "English":           "en",
                "Tamil (தமிழ்)":     "ta",
                "Telugu (తెలుగు)":   "te",
                "Malayalam (മലയാളം)":"ml",
                "Hindi (हिन्दी)":    "hi",
            }
            fae_lang_label = st.selectbox("Language",
                                          list(_lang_display.keys()),
                                          label_visibility="visible",
                                          key="fae_lang_label")
            fae_lang = _lang_display[fae_lang_label]
        with tc3:
            fae_synth = st.toggle("🧪 Offline / synthetic", value=False, key="fae_synth")

        submitted = st.form_submit_button(
            "🎣  Get Advisory", type="primary", use_container_width=True
        )

    if submitted:
        location_arg = (
            {"lat": fae_lat, "lon": fae_lon}
            if port_choice == "(lat/lon point)"
            else port_choice
        )
        with st.spinner("🎣 Fetching ocean conditions…"):
            try:
                from fisher_advisory_engine.engine import compute_advisory as _fae_compute
                result = _fae_compute(
                    location=location_arg,
                    day=day_choice,
                    trip_type=trip_type,
                    language=fae_lang,
                    use_synthetic=fae_synth,
                )
                st.session_state["fae_result"] = result
                st.session_state["fae_error"]  = None
            except Exception as exc:
                st.session_state["fae_result"] = None
                st.session_state["fae_error"]  = str(exc)

    # ── Error ─────────────────────────────────────────────────────────────
    fae_err = st.session_state.get("fae_error")
    if fae_err:
        st.error(f"❌ {fae_err}")

    # ── Result ────────────────────────────────────────────────────────────
    result = st.session_state.get("fae_result")
    if result is None:
        return

    st.markdown("---")

    # Unverified warning
    if not result.location_verified:
        st.warning(
            f"⚠️ **{result.location_name}** has unverified coordinates. "
            "All values below are computed at a placeholder location and "
            "are not valid until port coordinates are confirmed in ports.csv.",
            icon="⚠️",
        )

    # Short message card
    # fae_lang is resolved from the selectbox label → code mapping
    _lang_map = {
        "English":            "en",
        "Tamil (தமிழ்)":      "ta",
        "Telugu (తెలుగు)":    "te",
        "Malayalam (മലയാളം)": "ml",
        "Hindi (हिन्दी)":     "hi",
    }
    _selected_label = st.session_state.get("fae_lang_label", "English")
    lang = _lang_map.get(_selected_label, "en")
    conf_color = {"HIGH": C["green"], "MEDIUM": C["amber"], "LOW": C["red"]}.get(
        result.confidence.label, C["muted"]
    )
    msg = result.short_message(lang)
    st.markdown(
        f'<div style="background:#fff;border:1px solid #d0e4f7;border-radius:14px;'
        f'padding:1.4rem 1.6rem;font-family:\'Segoe UI\',sans-serif;'
        f'box-shadow:0 2px 12px rgba(21,101,192,0.07)">'
        f'<div style="font-size:0.72rem;font-weight:700;color:#5c7a99;'
        f'text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px">'
        f'🎣 {result.location_name} &nbsp;·&nbsp; {result.day_label} &nbsp;·&nbsp; {result.trip_type}</div>'
        f'<div style="font-size:0.88rem;white-space:pre-wrap;line-height:1.7">{msg}</div>'
        f'<div style="margin-top:0.6rem;font-size:0.80rem;font-weight:700;color:{conf_color}">'
        f'Confidence: {result.confidence.label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if result.confidence.reasons:
        with st.expander("ℹ️ Confidence details", expanded=False):
            for r in result.confidence.reasons:
                st.caption(f"• {r}")

    # ── Final fisher advisory summary ──────────────────────────────────────
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    _render_fae_summary(result, lang, conf_color)

    with st.expander("📄 Full JSON", expanded=False):
        st.code(result.to_json(), language="json")


def main():
    _navbar()
    page = st.session_state.page
    st.markdown("<div style='padding:0 2rem'>", unsafe_allow_html=True)
    if   page == "Home":      page_home()
    elif page == "Explorer":  page_explorer()
    elif page == "Map":       page_map()
    elif page == "Databases": page_databases()
    elif page == "Stress":    page_stress_engine()
    elif page == "Advisory":
        try:
            page_fisher_advisory()
        except Exception as _fae_page_exc:
            st.error(f"Fisher Advisory page error: {_fae_page_exc}")
    elif page == "About":     page_about()
    st.markdown("</div>", unsafe_allow_html=True)
    _footer()

if __name__ == "__main__":
    main()
