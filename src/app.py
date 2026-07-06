"""
Glanerbeek Dashboard -- Live Sensor Integration
================================================
A Streamlit dashboard that fetches live soil-moisture data from
MajiSys (ITC, University of Twente) sensor stations and renders
ecological risk alerts for the Glanerbeek food forest.

Domain context  (sandy-loam baseline, from context.md S3):
  - 5-color ecological risk framework:
      1 = Green  (Optimal)              VWC >= 17 %
      2 = Yellow (Microbial Stress)     VWC 13-17 %  triggers 48 h labor window
      3 = Orange (Severe Stress)        VWC 9-13 %
      4 = Red    (Plant Wilting)        VWC 5-9 %   (permanent wilting point)
      5 = Black  (Ecosystem Cessation)  VWC < 5 %
  - Worst-case conservative aggregation: minimum VWC from top 40 cm.
  - Spatial Priority Score (1-5) ranks forest plots by urgency.
  - VWC = Volumetric Water Content (%) from soil-moisture sensors.

Run:  streamlit run src/app.py
"""

from __future__ import annotations

import datetime as _dt
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

# -- Ensure MajiSysUtil is importable from the same directory -----------
_src_dir = str(Path(__file__).resolve().parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import MajiSysUtil as msu  # noqa: E402

# -- Monkey-patch requests.get with a 30 s timeout so MajiSysUtil
#    never blocks the dashboard indefinitely. ---------------------------
import requests as _requests  # noqa: E402

_original_get = _requests.get


def _get_with_timeout(*args, **kwargs):
    kwargs.setdefault("timeout", 30)
    return _original_get(*args, **kwargs)


_requests.get = _get_with_timeout

# ----------------------------------------------
# 0. PAGE CONFIG
# ----------------------------------------------
st.set_page_config(
    page_title="Glanerbeek Dashboard",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------
# 1. CONSTANTS
# ----------------------------------------------

RISK_PALETTE: dict[int, dict] = {
    1: {"label": "Optimal",              "color": "#22c55e", "bg": "#052e16", "text_color": "#bbf7d0"},
    2: {"label": "Microbial Stress",     "color": "#eab308", "bg": "#422006", "text_color": "#fef08a"},
    3: {"label": "Severe Stress",        "color": "#f97316", "bg": "#431407", "text_color": "#fed7aa"},
    4: {"label": "Plant Wilting",        "color": "#ef4444", "bg": "#450a0a", "text_color": "#fecaca"},
    5: {"label": "Ecosystem Cessation",  "color": "#1f2937", "bg": "#030712", "text_color": "#9ca3af"},
}

# VWC thresholds -- Source: context.md S3, Sandy Loam Baseline
#   Yellow (Microbial Stress) triggers at 17 % VWC  (-60 kPa)
#   Red    (Plant Wilting)    triggers at  9 % VWC  (permanent wilting point)
VWC_THRESHOLDS: list[tuple[float, int]] = [
    (17.0, 1),   # >= 17 % -> Green  (Optimal)
    (13.0, 2),   # >= 13 % -> Yellow (Microbial Stress) -- 48 h alert trigger
    (9.0,  3),   # >=  9 % -> Orange (Severe Stress)
    (5.0,  4),   # >=  5 % -> Red    (Plant Wilting)
    (0.0,  5),   # <   5 % -> Black  (Ecosystem Cessation)
]

# Logger-to-plot mapping (sequential, as specified by user)
PLOT_DEFINITIONS: list[dict] = [
    {"id": "GP-01", "name": "Field 1 - Station 1 (ATMOS)", "area_ha": 2.4,
     "logger_id": "z6-21176", "logger_name": "F1_1_ATMOS_SMST1", "field": 1},
    {"id": "GP-02", "name": "Field 1 - Station 2",        "area_ha": 1.8,
     "logger_id": "z6-21178", "logger_name": "F1_2_SMST2",       "field": 1},
    {"id": "GP-03", "name": "Field 1 - Station 3",        "area_ha": 3.1,
     "logger_id": "z6-25928", "logger_name": "F1_3_SMST3",       "field": 1},
    {"id": "GP-04", "name": "Field 2 - Station 1 (ATMOS)", "area_ha": 2.0,
     "logger_id": "z6-21177", "logger_name": "F2_1_ATMOS_SMST1", "field": 2},
    {"id": "GP-05", "name": "Field 2 - Station 2",        "area_ha": 1.5,
     "logger_id": "z6-21179", "logger_name": "F2_2_SMST2",       "field": 2},
    {"id": "GP-06", "name": "Field 2 - Station 3 (WP)",   "area_ha": 2.7,
     "logger_id": "z6-30173", "logger_name": "F2_3_WP_SMST",     "field": 2},
]

MAX_DEPTH_CM = 40.0  # Worst-case conservative: only top 40 cm sensors


# ----------------------------------------------
# 2. LIVE DATA FETCHING
# ----------------------------------------------

def _get_cache_time_key() -> str:
    """Round current time to nearest 15 min for cache-key stability."""
    now = _dt.datetime.now()
    minutes = (now.minute // 15) * 15
    rounded = now.replace(minute=minutes, second=0, microsecond=0)
    return rounded.strftime("%Y-%m-%d %H:%M")


def _extract_worst_case_vwc(df: pd.DataFrame) -> float | None:
    """
    Worst-case conservative aggregation.

    From a soil-moisture DataFrame with columns like 'Soil moisture 5cm',
    find the **minimum** (driest) VWC across all top-40 cm sensors from
    the most recent timestamp.

    Returns VWC as a percentage (already multiplied by 100).
    """
    if df is None or df.empty:
        return None

    # Filter columns to sensors <= 40 cm depth
    top_cols: list[str] = []
    for col in df.columns:
        try:
            depth_str = col.split()[-1]                    # e.g. "5cm"
            depth_val = float(depth_str.replace("cm", ""))
            if depth_val <= MAX_DEPTH_CM:
                top_cols.append(col)
        except (ValueError, IndexError):
            continue

    if not top_cols:
        return None

    # Latest timestamp, worst (minimum) VWC across qualifying depths
    latest_row = df[top_cols].iloc[-1]
    min_vwc_fraction = latest_row.min()

    if pd.isna(min_vwc_fraction):
        return None

    return round(float(min_vwc_fraction) * 100.0, 2)


@st.cache_data(ttl=900, show_spinner="Fetching live sensor data from MajiSys...")
def fetch_live_data(time_key: str) -> tuple[dict[str, float], bool]:
    """
    Fetch the latest 24 h of VWC readings from all Glanerbeek stations.

    Returns
    -------
    (data_dict, success)
        data_dict maps logger_id (str) -> worst-case VWC % (float).
        success is False when the server could not be reached.
    """
    try:
        loggers = [
            (p["logger_id"], p["logger_name"]) for p in PLOT_DEFINITIONS
        ]
        maxdate = _dt.datetime.now()
        mindate = maxdate - _dt.timedelta(hours=24)

        # 1. Download raw CSV from each logger
        raw = msu.downloadTimeseries(loggers, mindate, maxdate)

        # 2. Group by parameter (only Soil moisture needed)
        grouped = msu.groupByParameter(raw, ["Soil moisture"], loggers)

        # 3. Convert to Pandas DataFrames
        pandas_data = msu.toPandas(grouped)

        # 4. Extract worst-case VWC per logger
        results: dict[str, float] = {}
        if "Soil moisture" in pandas_data:
            for logger_tuple, _unit, df in pandas_data["Soil moisture"]:
                logger_id = logger_tuple[0]
                vwc_pct = _extract_worst_case_vwc(df)
                if vwc_pct is not None:
                    results[logger_id] = vwc_pct

        return results, True

    except Exception:
        return {}, False


@st.cache_data(ttl=900, show_spinner="Fetching audit data...")
def fetch_raw_audit_data(time_key: str) -> tuple[dict, dict, dict, bool]:
    """
    Fetch pipeline data at each stage for the Data Pipeline Audit tab.

    Returns
    -------
    (raw_csvs, cleaned_dfs, filtered_results, success)
        raw_csvs:          logger_id (str) -> raw CSV text
        cleaned_dfs:       logger_id (str) -> full cleaned Pandas DataFrame
        filtered_results:  logger_id (str) -> {top_40cm_df, min_vwc_pct, columns_info}
        success:           False when the server could not be reached.
    """
    try:
        loggers = [
            (p["logger_id"], p["logger_name"]) for p in PLOT_DEFINITIONS
        ]
        maxdate = _dt.datetime.now()
        mindate = maxdate - _dt.timedelta(hours=24)

        # Stage 1: Raw CSV from each logger
        raw = msu.downloadTimeseries(loggers, mindate, maxdate)
        raw_csvs: dict[str, str] = {}
        for logger_tuple in loggers:
            if logger_tuple in raw:
                raw_csvs[logger_tuple[0]] = raw[logger_tuple]

        # Stage 2: Group & convert to Pandas DataFrames
        grouped = msu.groupByParameter(raw, ["Soil moisture"], loggers)
        pandas_data = msu.toPandas(grouped)

        cleaned_dfs: dict[str, pd.DataFrame] = {}
        filtered_results: dict[str, dict] = {}

        if "Soil moisture" in pandas_data:
            for logger_tuple, unit, df in pandas_data["Soil moisture"]:
                logger_id = logger_tuple[0]
                if df is not None and not df.empty:
                    cleaned_dfs[logger_id] = df

                    # Stage 3: 40 cm depth filter
                    top_cols: list[str] = []
                    all_cols_info: list[dict] = []
                    for col in df.columns:
                        try:
                            depth_str = col.split()[-1]
                            depth_val = float(depth_str.replace("cm", ""))
                            included = depth_val <= MAX_DEPTH_CM
                            all_cols_info.append({
                                "column": col,
                                "depth_cm": depth_val,
                                "included": included,
                            })
                            if included:
                                top_cols.append(col)
                        except (ValueError, IndexError):
                            continue

                    if top_cols:
                        top_df = df[top_cols]
                        latest_row = top_df.iloc[-1]
                        min_frac = latest_row.min()
                        min_pct = (
                            round(float(min_frac) * 100.0, 2)
                            if not pd.isna(min_frac)
                            else None
                        )
                        filtered_results[logger_id] = {
                            "top_40cm_df": top_df,
                            "min_vwc_pct": min_pct,
                            "columns_info": all_cols_info,
                        }

        return raw_csvs, cleaned_dfs, filtered_results, True

    except Exception:
        return {}, {}, {}, False


# ----------------------------------------------
# 3. DOMAIN HELPERS
# ----------------------------------------------

def vwc_to_risk(vwc: float) -> int:
    """Convert a VWC percentage to a 1-5 risk level."""
    for threshold, level in VWC_THRESHOLDS:
        if vwc >= threshold:
            return level
    return 5


@dataclass
class ForestPlot:
    """Forest plot with sensor data and computed risk state."""

    id: str
    name: str
    area_ha: float
    logger_id: str
    logger_name: str
    field: int
    vwc: float = 30.0
    spatial_priority: int = 1
    risk_level: int = 1
    yellow_entry_time: _dt.datetime | None = None

    def update(self, vwc: float, now: _dt.datetime) -> None:
        self.vwc = vwc
        new_risk = vwc_to_risk(vwc)

        # Track when a zone first enters Yellow (level 2) for the 48 h window
        if new_risk == 2 and self.risk_level != 2:
            self.yellow_entry_time = now
        elif new_risk != 2:
            self.yellow_entry_time = None

        self.risk_level = new_risk
        # Spatial priority mirrors risk for the prototype;
        # in production this would incorporate additional spatial analytics.
        self.spatial_priority = new_risk


def overall_risk(plots: list[ForestPlot]) -> int:
    """Farm-wide risk = maximum risk across all plots."""
    if not plots:
        return 1
    return max(p.risk_level for p in plots)


def remaining_48h(
    entry: _dt.datetime | None, now: _dt.datetime
) -> _dt.timedelta | None:
    """Time left in the 48-hour labor coordination window."""
    if entry is None:
        return None
    deadline = entry + _dt.timedelta(hours=48)
    remaining = deadline - now
    if remaining.total_seconds() <= 0:
        return _dt.timedelta(0)
    return remaining


# ----------------------------------------------
# 4. SESSION STATE
# ----------------------------------------------

def _init_state() -> None:
    """Initialise session-state on first run."""
    if "plots" not in st.session_state:
        st.session_state.plots = [
            ForestPlot(
                id=p["id"],
                name=p["name"],
                area_ha=p["area_ha"],
                logger_id=p["logger_id"],
                logger_name=p["logger_name"],
                field=p["field"],
            )
            for p in PLOT_DEFINITIONS
        ]
    if "last_live_data" not in st.session_state:
        st.session_state.last_live_data = {}
    if "connection_ok" not in st.session_state:
        st.session_state.connection_ok = True


_init_state()


# ----------------------------------------------
# 5. INJECT CUSTOM CSS
# ----------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Google Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ---- global page bg ---- */
    .stApp { background: #0b0f19; }

    /* ---- risk banner ---- */
    .risk-banner {
        border-radius: 16px;
        padding: 1.6rem 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 4px 24px rgba(0,0,0,.4);
        transition: background 0.5s ease;
        flex-wrap: wrap;
    }
    .risk-banner h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .risk-banner .subtitle {
        font-size: 0.95rem;
        opacity: 0.8;
        margin-top: 2px;
    }

    /* ---- action center ---- */
    .action-center {
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255,255,255,0.06);
        background: rgba(255,255,255,0.02);
    }
    .action-center .section-title {
        margin-bottom: 0.8rem;
    }
    .action-clear {
        color: #6b7280;
        font-size: 0.9rem;
        padding: 0.8rem 1rem;
        text-align: center;
        border: 1px dashed rgba(255,255,255,0.08);
        border-radius: 12px;
    }

    /* ---- countdown card ---- */
    .countdown-card {
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        background: linear-gradient(135deg, #422006 0%, #1c1917 100%);
        border: 1px solid rgba(234,179,8,0.25);
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 12px rgba(0,0,0,.3);
    }
    .countdown-card .zone-name {
        font-weight: 700;
        font-size: 1.05rem;
        color: #fef08a;
    }
    .countdown-card .timer {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        color: #eab308;
        font-variant-numeric: tabular-nums;
    }
    .countdown-card .timer-label {
        font-size: 0.8rem;
        color: #a3a3a3;
        margin-top: 2px;
    }
    .countdown-card .progress-bg {
        height: 6px;
        border-radius: 3px;
        background: rgba(255,255,255,0.08);
        margin-top: 0.6rem;
        overflow: hidden;
    }
    .countdown-card .progress-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #eab308, #f59e0b);
        transition: width 0.4s ease;
    }

    /* ---- data table styling ---- */
    .table-scroll-wrapper {
        width: 100%;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }
    .plot-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 6px;
        min-width: 600px;
    }
    .plot-table th {
        text-align: left;
        padding: 0.6rem 1rem;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #9ca3af;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        white-space: nowrap;
    }
    .plot-table td {
        padding: 0.75rem 1rem;
        font-size: 0.92rem;
        color: #e5e7eb;
    }
    .plot-table tr.data-row {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        transition: background 0.25s ease;
    }
    .plot-table tr.data-row:hover {
        background: rgba(255,255,255,0.07);
    }
    .plot-table tr.data-row td:first-child { border-radius: 10px 0 0 10px; }
    .plot-table tr.data-row td:last-child  { border-radius: 0 10px 10px 0; }

    .priority-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px; height: 32px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .risk-dot {
        width: 12px; height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
        box-shadow: 0 0 6px currentColor;
    }

    /* ---- section headings ---- */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #d1d5db;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ---- sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #111827 !important;
    }
    section[data-testid="stSidebar"] .stSlider label {
        font-size: 0.85rem !important;
    }

    /* ---- no-yellow message ---- */
    .no-yellow {
        color: #6b7280;
        font-size: 0.9rem;
        padding: 1rem;
        text-align: center;
        border: 1px dashed rgba(255,255,255,0.08);
        border-radius: 12px;
    }

    /* ---- live data cards in sidebar ---- */
    .live-card {
        background: rgba(255,255,255,0.04);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid;
    }
    .live-card .plot-label {
        font-weight: 600;
        font-size: 0.9rem;
        color: #e5e7eb;
    }
    .live-card .plot-meta {
        font-size: 0.78rem;
        color: #9ca3af;
        margin-top: 2px;
    }
    .live-card .vwc-value {
        font-size: 1.1rem;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
    }

    /* ---- mode badge ---- */
    .mode-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .mode-live {
        background: rgba(239,68,68,0.15);
        color: #ef4444;
        border: 1px solid rgba(239,68,68,0.3);
    }
    .mode-sim {
        background: rgba(34,197,94,0.15);
        color: #22c55e;
        border: 1px solid rgba(34,197,94,0.3);
    }

    /* ---- audit tab ---- */
    .audit-method-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .audit-method-card h4 {
        color: #a5b4fc;
        margin: 0 0 0.8rem 0;
        font-size: 1.05rem;
    }
    .audit-method-card .step {
        display: flex;
        gap: 0.8rem;
        margin-bottom: 0.8rem;
        align-items: flex-start;
    }
    .audit-method-card .step-num {
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        width: 28px; height: 28px;
        min-width: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .audit-method-card .step-text {
        color: #d1d5db;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .audit-method-card .step-text strong {
        color: #e0e7ff;
    }
    .audit-result-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .audit-result-card .logger-name {
        font-weight: 600;
        color: #e5e7eb;
        font-size: 0.95rem;
    }
    .audit-result-card .vwc-result {
        font-size: 1.3rem;
        font-weight: 800;
        font-variant-numeric: tabular-nums;
    }

    /* ---- countdown grid (desktop: side-by-side, mobile: stack) ---- */
    .countdown-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 0.8rem;
    }
    .countdown-grid .countdown-card {
        margin-bottom: 0;
    }

    /* ===== MOBILE RESPONSIVENESS ===== */
    @media (max-width: 768px) {
        .risk-banner {
            flex-direction: column;
            text-align: center;
            padding: 1.2rem 1rem;
            gap: 0.6rem;
        }
        .risk-banner h1 {
            font-size: 1.15rem;
        }
        .risk-banner .subtitle {
            font-size: 0.82rem;
        }
        .risk-banner div[style*="text-align:right"] {
            text-align: center !important;
        }
        .countdown-card .timer {
            font-size: 1.5rem;
        }
        .countdown-card .zone-name {
            font-size: 0.92rem;
        }
        .plot-table th {
            padding: 0.4rem 0.6rem;
            font-size: 0.68rem;
        }
        .plot-table td {
            padding: 0.5rem 0.6rem;
            font-size: 0.82rem;
        }
        .priority-badge {
            width: 26px; height: 26px;
            font-size: 0.82rem;
        }
        .section-title {
            font-size: 0.95rem;
        }
        .audit-method-card {
            padding: 1rem 1.1rem;
        }
        .audit-method-card .step-text {
            font-size: 0.82rem;
        }
        .live-card {
            padding: 0.5rem 0.7rem;
        }
        .live-card .plot-label {
            font-size: 0.82rem;
        }
        .live-card .vwc-value {
            font-size: 0.95rem;
        }
        .countdown-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------
# 6. SIDEBAR -- MODE TOGGLE + CONTROLS
# ----------------------------------------------
with st.sidebar:
    st.markdown("## Dashboard Controls")

    live_mode = st.toggle("Live Data Mode", value=False, key="live_mode")

    if live_mode:
        st.markdown(
            '<span class="mode-badge mode-live">LIVE — MajiSys Sensors</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Fetching last 24 h from ITC Twente sensor network.  "
            "Worst-case VWC from top 40 cm depths."
        )
    else:
        st.markdown(
            '<span class="mode-badge mode-sim">SIMULATION — Manual</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Drag sliders to simulate VWC readings and test "
            "ecological thresholds."
        )

    st.divider()

    now = _dt.datetime.now()

    # -- LIVE MODE ---------------------------------
    if live_mode:
        time_key = _get_cache_time_key()
        live_data, success = fetch_live_data(time_key)

        # Fallback logic: persist last-known-good data in session state
        if success and live_data:
            st.session_state.last_live_data = live_data
            st.session_state.connection_ok = True
        elif not success:
            st.session_state.connection_ok = False
            # Fall back to last-known values if available
            if st.session_state.last_live_data:
                live_data = st.session_state.last_live_data
            else:
                live_data = {}

        # Update plots from live readings
        for plot in st.session_state.plots:
            if plot.logger_id in live_data:
                plot.update(live_data[plot.logger_id], now)

        # Render live reading cards in sidebar
        for plot in st.session_state.plots:
            ri = RISK_PALETTE[plot.risk_level]
            has_data = plot.logger_id in live_data
            vwc_display = f"{plot.vwc:.1f}%" if has_data else "-- no data"
            st.markdown(
                f"""
                <div class="live-card" style="border-color:{ri['color']};">
                    <div style="display:flex;justify-content:space-between;
                                align-items:center;">
                        <div>
                            <div class="plot-label">
                                <span class="risk-dot"
                                      style="color:{ri['color']};
                                             background:{ri['color']};"></span>
                                {plot.id} — {plot.name}
                            </div>
                            <div class="plot-meta">
                                Logger: {plot.logger_id} | Field {plot.field}
                            </div>
                        </div>
                        <div class="vwc-value" style="color:{ri['color']};">
                            {vwc_display}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -- MANUAL SIMULATION MODE --------------------
    else:
        for plot in st.session_state.plots:
            ri = RISK_PALETTE[plot.risk_level]
            label = f"**{plot.id}** — {plot.name}"
            new_vwc = st.slider(
                label,
                min_value=0.0,
                max_value=60.0,
                value=plot.vwc,
                step=0.5,
                format="%.1f %%",
                key=f"vwc_{plot.id}",
            )
            plot.update(new_vwc, now)

    st.divider()
    st.caption(
        "VWC thresholds:  >=17% Optimal  |  >=13% Microbial Stress  |  "
        ">=9% Severe  |  >=5% Wilting  |  <5% Cessation"
    )


# ----------------------------------------------
# 7. CONNECTION WARNING (live mode only)
# ----------------------------------------------
if live_mode and not st.session_state.connection_ok:
    st.warning(
        "**Connection Lost** — Unable to reach the MajiSys sensor "
        "server at `majisysdemo.itc.utwente.nl`. Displaying last known "
        "values. Data will refresh automatically when the connection "
        "is restored.",
    )


# ----------------------------------------------
# 8. GLOBAL RISK BANNER
# ----------------------------------------------
farm_risk = overall_risk(st.session_state.plots)
ri = RISK_PALETTE[farm_risk]

mode_indicator = (
    '<span class="mode-badge mode-live" style="margin-left:8px;">LIVE</span>'
    if live_mode
    else '<span class="mode-badge mode-sim" style="margin-left:8px;">SIM</span>'
)

st.markdown(
    f"""
    <div class="risk-banner"
         style="background:{ri['bg']}; border:1px solid {ri['color']}33;">
        <div>
            <h1 style="color:{ri['text_color']};">
                Glanerbeek Forest Dashboard {mode_indicator}
            </h1>
            <div class="subtitle" style="color:{ri['text_color']};">
                Farm-wide ecological status&ensp;&middot;&ensp;
                {len(st.session_state.plots)} monitored plots
            </div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:1.4rem;font-weight:800;color:{ri['color']};">
                Level {farm_risk}
            </div>
            <div style="font-size:0.85rem;font-weight:600;color:{ri['color']};">
                {ri['label']}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------
# 9. LAYOUT -- TABBED
# ----------------------------------------------
tab_dashboard, tab_audit = st.tabs(["Live Dashboard", "Data Pipeline Audit"])

with tab_dashboard:

    # ============================================
    # 9a. ACTION CENTER -- 48-HOUR COUNTDOWN
    #     (Full-width block at the very top)
    # ============================================
    st.markdown(
        '<div class="section-title">48 h Labor Coordination Window</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Zones in Yellow (Microbial Stress, VWC 13-17%) trigger a "
        "48-hour window for coordinated field intervention."
    )

    yellow_plots = [
        p for p in st.session_state.plots if p.risk_level == 2
    ]

    if not yellow_plots:
        st.markdown(
            '<div class="no-yellow">'
            "All Clear — No zones are currently in Yellow state.</div>",
            unsafe_allow_html=True,
        )
    else:
        cards_html = '<div class="countdown-grid">'
        for p in yellow_plots:
            rem = remaining_48h(p.yellow_entry_time, now)
            if rem is None:
                continue

            total_secs = rem.total_seconds()
            hours = int(total_secs // 3600)
            minutes = int((total_secs % 3600) // 60)
            seconds = int(total_secs % 60)
            pct = min(100.0, (total_secs / (48 * 3600)) * 100)

            # Urgency colour shift as time runs out
            if pct > 50:
                bar_gradient = (
                    "linear-gradient(90deg, #22c55e, #eab308)"
                )
            elif pct > 20:
                bar_gradient = (
                    "linear-gradient(90deg, #eab308, #f97316)"
                )
            else:
                bar_gradient = (
                    "linear-gradient(90deg, #f97316, #ef4444)"
                )

            cards_html += f"""
            <div class="countdown-card">
                <div class="zone-name">
                    {p.id} — {p.name}
                </div>
                <div class="timer">
                    {hours:02d}h {minutes:02d}m {seconds:02d}s
                </div>
                <div class="timer-label">
                    remaining of 48-hour coordination window
                </div>
                <div class="progress-bg">
                    <div class="progress-fill"
                         style="width:{pct:.1f}%;
                                background:{bar_gradient};"></div>
                </div>
            </div>
            """
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

    st.divider()

    # ============================================
    # 9b. SPATIAL PRIORITY TABLE
    #     (Full-width block below the action center)
    # ============================================
    st.markdown(
        '<div class="section-title">Spatial Priority Ranking</div>',
        unsafe_allow_html=True,
    )

    # Sort plots by spatial_priority descending (highest urgency first)
    sorted_plots = sorted(
        st.session_state.plots,
        key=lambda p: p.spatial_priority,
        reverse=True,
    )

    rows_html = ""
    for p in sorted_plots:
        pi = RISK_PALETTE[p.risk_level]
        source_label = (
            f"{p.logger_id}" if live_mode else "Manual"
        )
        rows_html += f"""
        <tr class="data-row">
            <td style="font-weight:600;">{p.id}</td>
            <td>{p.name}</td>
            <td>{p.area_ha:.1f} ha</td>
            <td>
                <span class="risk-dot"
                      style="color:{pi['color']};
                             background:{pi['color']};"></span>
                {pi['label']}
            </td>
            <td style="text-align:center;">
                <span class="priority-badge"
                      style="background:{pi['color']}22;
                             color:{pi['color']};">
                    {p.spatial_priority}
                </span>
            </td>
            <td style="font-variant-numeric:tabular-nums;color:#93c5fd;">
                {p.vwc:.1f}%
            </td>
            <td style="font-size:0.78rem;color:#6b7280;">
                {source_label}
            </td>
        </tr>
        """

    st.markdown(
        f"""
        <div class="table-scroll-wrapper">
        <table class="plot-table">
            <thead>
                <tr>
                    <th>Plot ID</th>
                    <th>Name</th>
                    <th>Area</th>
                    <th>Risk State</th>
                    <th style="text-align:center;">Priority</th>
                    <th>VWC</th>
                    <th>Source</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------
    # 10. FOOTER
    # ------------------------------------------
    st.divider()
    mode_str = "Live (MajiSys)" if live_mode else "Manual Simulation"
    st.caption(
        f"Glanerbeek Dashboard  |  Prototype v0.2  |  Mode: {mode_str}  |  "
        f"Last refresh: {now.strftime('%Y-%m-%d %H:%M:%S')}  |  "
        f"Thresholds: 17% Yellow  |  9% Red (Sandy Loam Baseline)"
    )

with tab_audit:
    # ------------------------------------------
    # AUDIT TAB -- Data Pipeline Transparency
    # ------------------------------------------
    st.markdown(
        '<div class="section-title">Data Pipeline Audit</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Academic transparency view — inspect each stage of the "
        "sensor data pipeline from raw API response to final VWC values."
    )

    # -- Methodology explainer (full-width, top of audit tab) --
    st.markdown(
        """
        <div class="audit-method-card">
            <h4>3-Step Pipeline Methodology</h4>
            <div class="step">
                <div class="step-num">1</div>
                <div class="step-text">
                    <strong>Fetch ZENTRA API Payload</strong><br>
                    HTTP GET requests to the MajiSys server at
                    <code>majisysdemo.itc.utwente.nl</code> retrieve
                    raw CSV data for each ZENTRA Z6 datalogger.
                    Each logger reports soil-moisture readings at
                    multiple depth intervals (5 cm, 10 cm, 20 cm, etc.).
                </div>
            </div>
            <div class="step">
                <div class="step-num">2</div>
                <div class="step-text">
                    <strong>Clean Missing Data via Pandas</strong><br>
                    Raw CSV rows are parsed and converted into Pandas
                    DataFrames. Sensors with &lt;90 % data completeness
                    are discarded. An <em>inner join</em> aligns
                    timestamps across depth columns, ensuring only
                    complete observations are retained.
                </div>
            </div>
            <div class="step">
                <div class="step-num">3</div>
                <div class="step-text">
                    <strong>Worst-Case Conservative 40 cm Filter</strong><br>
                    Only sensors at depths &le; 40 cm (the root zone most
                    relevant to food-forest ecology) are retained.
                    From the most recent timestamp, the <em>minimum</em>
                    VWC fraction is selected &mdash; the driest sensor
                    reading &mdash; and converted to a percentage
                    (m&sup3;/m&sup3; &times; 100). This conservative
                    approach ensures the dashboard never underestimates
                    drought stress.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="audit-method-card" style="border-color: rgba(234,179,8,0.2);
                    background: linear-gradient(135deg, #422006 0%, #1c1917 100%);">
            <h4 style="color:#fef08a;">Technical Parameters</h4>
            <div class="step-text" style="color:#d1d5db;">
                <strong style="color:#fef08a;">API Endpoint:</strong>
                <code>http://majisysdemo.itc.utwente.nl/florapulse/get7days.py</code><br><br>
                <strong style="color:#fef08a;">Time Window:</strong> Last 24 hours<br><br>
                <strong style="color:#fef08a;">Max Depth Filter:</strong> &le; 40 cm<br><br>
                <strong style="color:#fef08a;">Aggregation:</strong> min() across qualifying sensors<br><br>
                <strong style="color:#fef08a;">Data Freshness:</strong> Cached for 15 min (TTL = 900 s)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # -- Data views (full-width, below methodology) --
    audit_active = st.toggle(
        "Load Pipeline Data from MajiSys API",
        key="audit_active",
    )

    if audit_active:
        time_key = _get_cache_time_key()
        raw_csvs, cleaned_dfs, filtered_results, audit_ok = (
            fetch_raw_audit_data(time_key)
        )

        if not audit_ok:
            st.error(
                "Could not reach the MajiSys API. "
                "Check your network connection.",
            )
        else:
            # -- View 1: Raw API Data ----------------
            st.markdown(
                '<div class="section-title">'
                'View 1 — Raw API Response</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Unprocessed CSV text returned by each ZENTRA "
                "datalogger. Showing the first 30 lines per logger."
            )

            for logger_id, csv_text in raw_csvs.items():
                station_name = next(
                    (p["name"] for p in PLOT_DEFINITIONS
                     if p["logger_id"] == logger_id),
                    logger_id,
                )
                csv_lines = csv_text.split("\n")
                preview = "\n".join(csv_lines[:30])
                with st.expander(
                    f"{logger_id} — {station_name}  "
                    f"({len(csv_lines)} lines total)"
                ):
                    st.code(preview, language="csv")

            st.divider()

            # -- View 2: Processed / Filtered Data ---
            st.markdown(
                '<div class="section-title">'
                'View 2 — Processed Data</div>',
                unsafe_allow_html=True,
            )

            show_filter = st.toggle(
                "Run Worst-Case 40 cm Filter",
                key="show_filter",
            )

            if show_filter:
                for logger_id in cleaned_dfs:
                    station_name = next(
                        (p["name"] for p in PLOT_DEFINITIONS
                         if p["logger_id"] == logger_id),
                        logger_id,
                    )
                    df_full = cleaned_dfs[logger_id]
                    fr = filtered_results.get(logger_id)

                    with st.expander(
                        f"{logger_id} — {station_name}",
                        expanded=True,
                    ):
                        st.markdown(
                            "**Cleaned DataFrame** (all depths)"
                        )
                        st.dataframe(
                            df_full.tail(10),
                            use_container_width=True,
                        )

                        if fr:
                            cols_info = fr["columns_info"]
                            inc = [
                                c for c in cols_info
                                if c["included"]
                            ]
                            exc = [
                                c for c in cols_info
                                if not c["included"]
                            ]

                            st.markdown(
                                "**Included sensors** (depth <= 40 cm)"
                            )
                            for c in inc:
                                st.markdown(
                                    f"- `{c['column']}` "
                                    f"— {c['depth_cm']:.0f} cm"
                                )

                            if exc:
                                st.markdown(
                                    "**Excluded sensors** (depth > 40 cm)"
                                )
                                for c in exc:
                                    st.markdown(
                                        f"- `{c['column']}` "
                                        f"— {c['depth_cm']:.0f} cm"
                                    )

                            st.markdown(
                                "**Filtered DataFrame** "
                                "(top 40 cm only)"
                            )
                            top_df = fr["top_40cm_df"]
                            st.dataframe(
                                top_df.tail(10),
                                use_container_width=True,
                            )

                            vwc = fr["min_vwc_pct"]
                            if vwc is not None:
                                risk = vwc_to_risk(vwc)
                                ri_card = RISK_PALETTE[risk]
                                st.markdown(
                                    f"""
                                    <div class="audit-result-card"
                                         style="border-color:
                                         {ri_card['color']}33;">
                                        <div class="logger-name">
                                            <span class="risk-dot"
                                                  style="color:{ri_card['color']};
                                                         background:{ri_card['color']};"></span>
                                            Worst-Case Result
                                        </div>
                                        <div class="vwc-result"
                                             style="color:
                                             {ri_card['color']};">
                                            {vwc:.2f} % VWC
                                        </div>
                                        <div style="font-size:0.8rem;
                                             color:#9ca3af;
                                             margin-top:4px;">
                                            Risk Level {risk}
                                            &mdash; {ri_card['label']}
                                            &ensp;&middot;&ensp;
                                            min() of {len(inc)}
                                            sensor(s) at most recent
                                            timestamp
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
            else:
                st.caption(
                    "Toggle the filter above to see how the raw "
                    "data is transformed into the worst-case VWC "
                    "values shown on the Live Dashboard."
                )
    else:
        st.info(
            "Toggle **Load Pipeline Data** above to fetch raw "
            "sensor data and inspect the processing pipeline.",
        )
