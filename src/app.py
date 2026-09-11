"""
Glanerbeek Dashboard V3 — Live Sensor Decision-Support
=======================================================
A Streamlit dashboard that fetches live soil-moisture data from
MajiSys (ITC, University of Twente) sensor stations and renders
ecological risk alerts for the Glanerbeek food forest.

Thesis thresholds (sandy-loam baseline):
  - 3-level classification:
      Green  (Optimal)              VWC > 17 %
      Yellow (Irrigation Trigger)   9 % < VWC <= 17 %   (plan irrigation)
      Red    (Critical)             VWC <= 9 %           (permanent wilting point)
  - Worst-case conservative aggregation: minimum VWC from top 40 cm.
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
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------
# SIDEBAR SETTINGS
# ----------------------------------------------
st.sidebar.markdown("### Appearance")
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

# ----------------------------------------------
# 1. CONSTANTS — 3-level risk (thesis-aligned)
# ----------------------------------------------

RISK_PALETTE: dict[int, dict] = {
    1: {"label": "Optimal",              "color": "#16a34a", "bg": "#f0fdf4", "text_color": "#14532d",
        "emoji": "🟢", "action": "No action required"},
    2: {"label": "Irrigation Trigger",   "color": "#ca8a04", "bg": "#fefce8", "text_color": "#713f12",
        "emoji": "🟡", "action": "Plan irrigation within 48 hours"},
    3: {"label": "Critical",             "color": "#dc2626", "bg": "#fef2f2", "text_color": "#7f1d1d",
        "emoji": "🔴", "action": "Immediate intervention required"},
}

# VWC thresholds — 3 levels matching thesis (Sandy Loam Baseline)
#   17 % = Irrigation Trigger (50% depletion of plant-available water)
#    9 % = Permanent Wilting Point (critical biological failure)
VWC_THRESHOLDS: list[tuple[float, int]] = [
    (17.0, 1),   # > 17 % -> Green  (Optimal)
    (9.0,  2),   # > 9 %  -> Yellow (Irrigation Trigger)
    (0.0,  3),   # <= 9 % -> Red    (Critical / PWP)
]

# Logger-to-plot mapping
# 5 soil-moisture stations confirmed by WUNDER / Jessica / Yijian.
# GP-06 (z6-30173, F2_3_WP_SMST) EXCLUDED — it is a Water-Potential
# (Matric Potential) station using TEROS 21 sensors, NOT a VWC station.
PLOT_DEFINITIONS: list[dict] = [
    {"id": "GP-01", "name": "F1-1 (ATMOS + Soil Moisture)", "area_ha": 2.4,
     "logger_id": "z6-21176", "logger_name": "F1_1_ATMOS_SMST1", "field": 1},
    {"id": "GP-02", "name": "F1-2 (Soil Moisture)",         "area_ha": 1.8,
     "logger_id": "z6-21178", "logger_name": "F1_2_SMST2",       "field": 1},
    {"id": "GP-03", "name": "F1-3 (Soil Moisture)",         "area_ha": 3.1,
     "logger_id": "z6-25928", "logger_name": "F1_3_SMST3",       "field": 1},
    {"id": "GP-04", "name": "F2-1 (ATMOS + Soil Moisture)", "area_ha": 2.0,
     "logger_id": "z6-21177", "logger_name": "F2_1_ATMOS_SMST1", "field": 2},
    {"id": "GP-05", "name": "F2-2 (Soil Moisture)",         "area_ha": 1.5,
     "logger_id": "z6-21179", "logger_name": "F2_2_SMST2",       "field": 2},
]

MAX_DEPTH_CM = 40.0  # Worst-case conservative: only top 40 cm sensors

# Depth colors for chart lines (consistent across all charts)
DEPTH_COLORS = {
    "5 cm": "#60a5fa",    # blue
    "10 cm": "#34d399",   # green
    "20 cm": "#fbbf24",   # amber
    "40 cm": "#f87171",   # red
    "80 cm": "#a78bfa",   # purple
}


# ----------------------------------------------
# 2. LIVE DATA FETCHING
# ----------------------------------------------

def _get_cache_time_key() -> str:
    """Round current time to nearest 15 min for cache-key stability."""
    now = _dt.datetime.now()
    minutes = (now.minute // 15) * 15
    rounded = now.replace(minute=minutes, second=0, microsecond=0)
    return rounded.strftime("%Y-%m-%d %H:%M")


def _get_historical_cache_key() -> str:
    """Round current time to nearest hour for historical data cache-key."""
    now = _dt.datetime.now()
    rounded = now.replace(minute=0, second=0, microsecond=0)
    return rounded.strftime("%Y-%m-%d %H:00")


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


def _extract_per_depth_vwc(df: pd.DataFrame) -> dict[str, float]:
    """
    Extract the latest VWC reading per depth from a sensor DataFrame.

    Returns a dict mapping depth label (e.g. '5 cm') -> VWC percentage.
    """
    if df is None or df.empty:
        return {}

    result = {}
    for col in df.columns:
        try:
            depth_str = col.split()[-1]
            depth_val = float(depth_str.replace("cm", ""))
            label = f"{depth_val:.0f} cm"
            latest_val = df[col].iloc[-1]
            if not pd.isna(latest_val):
                result[label] = round(float(latest_val) * 100.0, 2)
        except (ValueError, IndexError):
            continue
    return result


@st.cache_data(ttl=900, show_spinner="Fetching live sensor data…")
def fetch_live_data(time_key: str) -> tuple[dict[str, float], dict[str, dict[str, float]], bool]:
    """
    Fetch the latest 24 h of VWC readings from all Glanerbeek stations.

    Returns
    -------
    (worst_case_dict, per_depth_dict, success)
        worst_case_dict maps logger_id -> worst-case VWC %.
        per_depth_dict maps logger_id -> {depth_label: VWC %}.
        success is False when the server could not be reached.
    """
    try:
        loggers = [
            (p["logger_id"], p["logger_name"]) for p in PLOT_DEFINITIONS
        ]
        maxdate = _dt.datetime.now()
        mindate = maxdate - _dt.timedelta(hours=24)

        raw = msu.downloadTimeseries(loggers, mindate, maxdate)
        grouped = msu.groupByParameter(raw, ["Soil moisture"], loggers)
        pandas_data = msu.toPandas(grouped)

        worst_results: dict[str, float] = {}
        depth_results: dict[str, dict[str, float]] = {}

        if "Soil moisture" in pandas_data:
            for logger_tuple, _unit, df in pandas_data["Soil moisture"]:
                logger_id = logger_tuple[0]
                vwc_pct = _extract_worst_case_vwc(df)
                if vwc_pct is not None:
                    worst_results[logger_id] = vwc_pct
                depth_results[logger_id] = _extract_per_depth_vwc(df)

        return worst_results, depth_results, True

    except Exception:
        return {}, {}, False


@st.cache_data(ttl=3600, show_spinner="Fetching historical sensor data…")
def fetch_historical_data(
    time_key: str, logger_id: str, logger_name: str, days: int = 7,
) -> tuple[pd.DataFrame | None, bool]:
    """
    Fetch *days* days of soil-moisture data for a single logger.

    Returns the full DataFrame (ALL depth columns, including > 40 cm)
    for historical charting only.  This data is **never** used for alerts.
    """
    try:
        logger_tuple = (logger_id, logger_name)
        maxdate = _dt.datetime.now()
        mindate = maxdate - _dt.timedelta(days=days)

        raw = msu.downloadTimeseries([logger_tuple], mindate, maxdate)
        grouped = msu.groupByParameter(raw, ["Soil moisture"], [logger_tuple])
        pandas_data = msu.toPandas(grouped)

        if "Soil moisture" in pandas_data:
            for lt, _unit, df in pandas_data["Soil moisture"]:
                if lt[0] == logger_id and df is not None and not df.empty:
                    return df, True

        return None, True  # no data but no error

    except Exception:
        return None, False


def _parse_depth_columns(df: pd.DataFrame) -> list[tuple[float, str]]:
    """
    Parse and sort depth columns from a soil-moisture DataFrame.
    Returns a list of ``(depth_cm, column_name)`` tuples, sorted shallowest-first.
    """
    depth_cols: list[tuple[float, str]] = []
    for col in df.columns:
        try:
            depth_str = col.split()[-1]
            depth_val = float(depth_str.replace("cm", ""))
            depth_cols.append((depth_val, col))
        except (ValueError, IndexError):
            continue
    depth_cols.sort(key=lambda x: x[0])
    return depth_cols


# Lookback options for the historical timeframe toggle
_LOOKBACK_OPTIONS: dict[str, int] = {
    "7 Days": 7,
    "14 Days": 14,
    "30 Days": 30,
}


def _render_historical_chart(plot: ForestPlot) -> None:
    """Render a VWC trend chart with colored background threshold bands."""

    # ---- Timeframe toggle ----
    lookback_label = st.radio(
        "Lookback period",
        options=list(_LOOKBACK_OPTIONS.keys()),
        index=2,  # default to 30 days
        horizontal=True,
        key=f"lookback_{plot.id}",
    )
    lookback_days = _LOOKBACK_OPTIONS[lookback_label]

    hist_key = _get_historical_cache_key()
    df, success = fetch_historical_data(
        hist_key, plot.logger_id, plot.logger_name, days=lookback_days,
    )

    if not success:
        st.error("Could not fetch historical data from MajiSys.")
        return
    if df is None or df.empty:
        st.info(
            f"No historical data returned for the last {lookback_days} days."
        )
        return

    depth_cols = _parse_depth_columns(df)
    if not depth_cols:
        st.info("No depth sensors found in the data.")
        return

    # ---- Depth selector (multiselect) ----
    depth_labels = [f"{d:.0f} cm" for d, _ in depth_cols]
    # Default: show top 40cm sensors only
    defaults = [lbl for d, _ in depth_cols for lbl in [f"{d:.0f} cm"] if d <= MAX_DEPTH_CM]
    if not defaults:
        defaults = [depth_labels[0]]

    selected = st.multiselect(
        "Select sensor depths to view",
        options=depth_labels,
        default=defaults,
        key=f"depth_sel_{plot.id}",
    )

    if not selected:
        st.info("Select at least one sensor depth to view the chart.")
        return

    # ---- Build chart DataFrame from selected depths ----
    depth_map = {f"{d:.0f} cm": col for d, col in depth_cols}
    chart_data: dict[str, pd.Series] = {}
    for label in selected:
        chart_data[label] = df[depth_map[label]] * 100.0

    chart_df = pd.DataFrame(chart_data, index=df.index)
    chart_df.index = pd.to_datetime(chart_df.index)
    chart_df.index.name = "Timestamp"
    chart_df = chart_df.dropna(how="all")

    # Resample to daily averages (temporal aggregation)
    chart_df = chart_df.resample('1D').mean()
    chart_df = chart_df.sort_index()
    # Format dates without year (e.g. "01 Jun")
    chart_df.index = chart_df.index.strftime('%d %b')
    chart_df = chart_df.dropna(how="all")

    if chart_df.empty:
        st.info("No valid data for the selected depths and timeframe.")
        return

    # ---- Render with Plotly for gradient background bands ----
    try:
        import plotly.graph_objects as go

        fig = go.Figure()

        # Get y-axis range
        y_min = max(0, chart_df.min().min() - 2)
        y_max = min(60, chart_df.max().max() + 3)

        # Add colored background bands (threshold zones)
        # Red zone: 0 to 9%
        fig.add_hrect(
            y0=0, y1=9,
            fillcolor="rgba(239, 68, 68, 0.12)",
            line_width=0,
            annotation_text="Critical (<9%)",
            annotation_position="bottom left",
            annotation_font=dict(size=10, color="rgba(239, 68, 68, 0.5)"),
        )
        # Yellow zone: 9% to 17%
        fig.add_hrect(
            y0=9, y1=17,
            fillcolor="rgba(234, 179, 8, 0.10)",
            line_width=0,
            annotation_text="Irrigation Trigger (9–17%)",
            annotation_position="bottom left",
            annotation_font=dict(size=10, color="rgba(234, 179, 8, 0.5)"),
        )
        # Green zone: above 17%
        fig.add_hrect(
            y0=17, y1=y_max + 5,
            fillcolor="rgba(34, 197, 94, 0.08)",
            line_width=0,
            annotation_text="Optimal (>17%)",
            annotation_position="bottom left",
            annotation_font=dict(size=10, color="rgba(34, 197, 94, 0.4)"),
        )

        # Add data lines for each depth
        for col in chart_df.columns:
            color = DEPTH_COLORS.get(col, "#94a3b8")
            fig.add_trace(go.Scatter(
                x=chart_df.index,
                y=chart_df[col],
                mode='lines+markers',
                name=col,
                line=dict(color=color, width=2.5),
                marker=dict(size=4, color=color),
                hovertemplate=f'{col}<br>%{{x}}<br>VWC: %{{y:.1f}}%<extra></extra>',
            ))

        fig.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(11,15,25,1)' if dark_mode else '#ffffff',
            font=dict(family="Inter", color="#cbd5e1" if dark_mode else "#334155"),
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.06)' if dark_mode else '#e2e8f0',
                tickfont=dict(size=10, color="#cbd5e1" if dark_mode else "#64748b"),
                nticks=5,
                tickangle=0,
                fixedrange=True,
            ),
            yaxis=dict(
                title="VWC (%)",
                title_font=dict(size=11, color="#cbd5e1" if dark_mode else "#475569"),
                gridcolor='rgba(255,255,255,0.06)' if dark_mode else '#e2e8f0',
                range=[y_min, y_max],
                tickfont=dict(size=10, color="#cbd5e1" if dark_mode else "#64748b"),
                fixedrange=True,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                font=dict(size=11, color="#f1f5f9" if dark_mode else "#1e293b"),
            ),
            hovermode="x unified",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"chart_{plot.id}",
            config={"displayModeBar": False, "responsive": True, "scrollZoom": False},
        )

    except ImportError:
        # Fallback to Streamlit native chart if plotly not installed
        st.line_chart(chart_df, height=360)

    st.caption(
        "Colored bands: 🟢 >17% Optimal  |  🟡 9–17% Irrigation Trigger  |  "
        "🔴 <9% Critical (PWP).  "
        "Dashboard alerts use only the top 40 cm (worst-case conservative); "
        "deeper sensors are shown for visual context only."
    )


# ----------------------------------------------
# 3. DOMAIN HELPERS — 3-level risk
# ----------------------------------------------

def vwc_to_risk(vwc: float) -> int:
    """Convert a VWC percentage to a 1-3 risk level."""
    for threshold, level in VWC_THRESHOLDS:
        if vwc >= threshold:
            return level
    return 3


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
    per_depth_vwc: dict = None  # depth_label -> VWC %

    def __post_init__(self):
        if self.per_depth_vwc is None:
            self.per_depth_vwc = {}

    def update(self, vwc: float, per_depth: dict[str, float] | None = None) -> None:
        self.vwc = vwc
        self.risk_level = vwc_to_risk(vwc)
        self.spatial_priority = self.risk_level
        if per_depth is not None:
            self.per_depth_vwc = per_depth


def overall_risk(plots: list[ForestPlot]) -> int:
    """Farm-wide risk = maximum risk across all plots."""
    if not plots:
        return 1
    return max(p.risk_level for p in plots)


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
    if "last_depth_data" not in st.session_state:
        st.session_state.last_depth_data = {}
    if "connection_ok" not in st.session_state:
        st.session_state.connection_ok = True


_init_state()


# ----------------------------------------------
# 5. INJECT CUSTOM CSS
# ----------------------------------------------
light_css = """
    /* ---- global page bg (WHITE per Jessica's feedback) ---- */
    .stApp { background: #f8fafc; }

    /* ---- risk banner ---- */
    .risk-banner {
        border-radius: 14px;
        padding: 1rem 1.4rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.8rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 12px rgba(0,0,0,.10);
        transition: background 0.4s ease;
    }
    .risk-banner-left h1 {
        margin: 0;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.25;
    }
    .risk-banner-left .subtitle {
        font-size: 0.85rem;
        opacity: 0.85;
        margin-top: 3px;
    }
    .risk-banner-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 9999px;
        flex-shrink: 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    }
    .risk-banner-badge .badge-emoji {
        font-size: 1.25rem;
        line-height: 1;
    }
    .risk-banner-badge .badge-text {
        font-weight: 800;
        font-size: 0.92rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }

    /* ---- zone cards ---- */
    .zone-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1rem 1.3rem;
        margin-bottom: 0.6rem;
        border-left: 4px solid;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        transition: background 0.25s ease, transform 0.15s ease;
    }
    .zone-card:hover {
        background: #f1f5f9;
    }
    .zone-card .zone-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }
    .zone-card .zone-name {
        font-weight: 700;
        font-size: 1rem;
        color: #1e293b;
    }
    .zone-card .zone-meta {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 2px;
    }
    .zone-card .vwc-big {
        font-size: 1.45rem;
        font-weight: 800;
        line-height: 1;
        margin-top: 0.3rem;
        font-variant-numeric: tabular-nums;
    }
    .zone-card .zone-action {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.5rem;
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 6px;
    }

    /* ---- typography tweaks ---- */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ---- priority table ---- */
    .table-scroll-wrapper {
        overflow-x: auto;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .plot-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 4px;
    }
    .plot-table th {
        text-align: left;
        padding: 0.5rem 0.8rem;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        border-bottom: 1px solid #e2e8f0;
        white-space: nowrap;
    }
    .plot-table td {
        padding: 0.65rem 0.8rem;
        font-size: 0.88rem;
        color: #334155;
    }
    .plot-table tr.data-row {
        background: #ffffff;
        border-radius: 10px;
        transition: background 0.25s ease;
    }
    .plot-table tr.data-row:hover {
        background: #f1f5f9;
    }
    .plot-table tr.data-row td:first-child { border-radius: 10px 0 0 10px; }
    .plot-table tr.data-row td:last-child  { border-radius: 0 10px 10px 0; }

    .risk-dot {
        display: inline-block;
        width: 10px; height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .priority-badge {
        display: inline-flex;
        align-items: center; justify-content: center;
        width: 28px; height: 28px;
        border-radius: 50%;
        font-weight: 800;
        font-size: 0.85rem;
    }
    .mode-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: rgba(220,38,38,0.10);
        color: #dc2626;
        border: 1px solid rgba(220,38,38,0.25);
    }
    .mode-badge::before {
        content: '';
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #dc2626;
        animation: pulse-dot 1.5s infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    /* ---- sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #f1f5f9 !important;
    }

    /* High contrast text for outdoor sunlight */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #64748b !important;
        font-size: 0.82rem !important;
    }
    [data-testid="stWidgetLabel"] p,
    .stRadio label,
    .stMultiSelect label,
    div[data-testid="stRadio"] p {
        color: #334155 !important;
        font-weight: 500 !important;
    }
    .stRadio label span, .stRadio div[role="radiogroup"] label div {
        color: #475569 !important;
    }

    [data-testid="stExpander"] {
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
        background: #ffffff !important;
        overflow: hidden !important;
    }
    [data-testid="stExpander"] summary {
        color: #1e293b !important;
        background: #f8fafc !important;
        font-weight: 600 !important;
    }
    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:focus,
    [data-testid="stExpander"] summary:active {
        background: #f1f5f9 !important;
        color: #0f172a !important;
    }
    details[data-testid="stExpander"][open] summary {
        background: #f1f5f9 !important;
        border-bottom: 1px solid #e2e8f0 !important;
        color: #2563eb !important;
    }
    [data-testid="stExpanderDetails"] {
        background: #ffffff !important;
        padding: 0.75rem 0.6rem !important;
    }
"""

dark_css = """
    /* ---- global page bg ---- */
    .stApp { background: #0b0f19; }

    /* ---- risk banner ---- */
    .risk-banner {
        border-radius: 14px;
        padding: 1rem 1.4rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.8rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 20px rgba(0,0,0,.35);
        transition: background 0.4s ease;
    }
    .risk-banner-left h1 {
        margin: 0;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.25;
    }
    .risk-banner-left .subtitle {
        font-size: 0.85rem;
        opacity: 0.85;
        margin-top: 3px;
        color: #cbd5e1 !important;
    }
    .risk-banner-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 9999px;
        flex-shrink: 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    }
    .risk-banner-badge .badge-emoji {
        font-size: 1.25rem;
        line-height: 1;
    }
    .risk-banner-badge .badge-text {
        font-weight: 800;
        font-size: 0.92rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }

    /* ---- zone cards ---- */
    .zone-card {
        background: rgba(255,255,255,0.03);
        border-radius: 14px;
        padding: 1rem 1.3rem;
        margin-bottom: 0.6rem;
        border-left: 4px solid;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        transition: background 0.25s ease, transform 0.15s ease;
    }
    .zone-card:hover {
        background: rgba(255,255,255,0.06);
    }
    .zone-card .zone-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }
    .zone-card .zone-name {
        font-weight: 700;
        font-size: 1rem;
        color: #e5e7eb;
    }
    .zone-card .zone-meta {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 2px;
    }
    .zone-card .vwc-big {
        font-size: 1.45rem;
        font-weight: 800;
        line-height: 1;
        margin-top: 0.3rem;
        font-variant-numeric: tabular-nums;
    }
    .zone-card .zone-action {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.5rem;
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 6px;
    }

    /* ---- typography tweaks ---- */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ---- priority table ---- */
    .table-scroll-wrapper {
        overflow-x: auto;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .plot-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 4px;
    }
    .plot-table th {
        text-align: left;
        padding: 0.5rem 0.8rem;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #9ca3af;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        white-space: nowrap;
    }
    .plot-table td {
        padding: 0.65rem 0.8rem;
        font-size: 0.88rem;
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

    .risk-dot {
        display: inline-block;
        width: 10px; height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .priority-badge {
        display: inline-flex;
        align-items: center justify-content: center;
        width: 28px; height: 28px;
        border-radius: 50%;
        font-weight: 800;
        font-size: 0.85rem;
    }
    .mode-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: rgba(239,68,68,0.18);
        color: #ef4444;
        border: 1px solid rgba(239,68,68,0.35);
    }
    .mode-badge::before {
        content: '';
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #ef4444;
        animation: pulse-dot 1.5s infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    /* ---- sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #111827 !important;
    }

    /* High contrast text for outdoor sunlight */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #94a3b8 !important;
        font-size: 0.82rem !important;
    }
    [data-testid="stWidgetLabel"] p,
    .stRadio label,
    .stMultiSelect label,
    div[data-testid="stRadio"] p {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }
    .stRadio label span, .stRadio div[role="radiogroup"] label div {
        color: #cbd5e1 !important;
    }

    [data-testid="stExpander"] {
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        background: rgba(255,255,255,0.02) !important;
        overflow: hidden !important;
    }
    [data-testid="stExpander"] summary {
        color: #f1f5f9 !important;
        background: rgba(255,255,255,0.03) !important;
        font-weight: 600 !important;
    }
    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:focus,
    [data-testid="stExpander"] summary:active {
        background: rgba(255,255,255,0.07) !important;
        color: #ffffff !important;
    }
    details[data-testid="stExpander"][open] summary {
        background: rgba(255,255,255,0.05) !important;
        border-bottom: 1px solid rgba(255,255,255,0.08) !important;
        color: #60a5fa !important;
    }
    [data-testid="stExpanderDetails"] {
        background: rgba(11, 15, 25, 0.6) !important;
        padding: 0.75rem 0.6rem !important;
    }
"""

st.markdown(
    f"""
    <style>
    /* ---- Google Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

    {dark_css if dark_mode else light_css}

    /* ===== MOBILE-SPECIFIC OPTIMIZATIONS (Smartphones & Field Devices) ===== */
    @media (max-width: 768px) {{
        /* Eliminate massive top empty space on mobile */
        .block-container {{
            padding-top: 1.0rem !important;
            padding-bottom: 3.0rem !important;
            padding-left: 0.6rem !important;
            padding-right: 0.6rem !important;
            max-width: 100% !important;
        }}

        header[data-testid="stHeader"] {{
            height: 2.0rem !important;
            background: transparent !important;
        }}

        /* Compact, space-efficient risk banner */
        .risk-banner {{
            padding: 0.7rem 0.9rem !important;
            border-radius: 12px !important;
            gap: 0.5rem !important;
        }}
        .risk-banner-left h1 {{
            font-size: 1.02rem !important;
        }}
        .risk-banner-left .subtitle {{
            font-size: 0.72rem !important;
            line-height: 1.3 !important;
            color: #64748b !important;
        }}
        .risk-banner-badge {{
            padding: 5px 10px !important;
            gap: 5px !important;
        }}
        .risk-banner-badge .badge-emoji {{
            font-size: 1.05rem !important;
        }}
        .risk-banner-badge .badge-text {{
            font-size: 0.78rem !important;
        }}

        /* Ergonomic touch targets for navigation tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px !important;
            background: #f1f5f9 !important;
            padding: 4px !important;
            border-radius: 12px !important;
            margin-bottom: 0.5rem !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            padding: 10px 14px !important;
            font-size: 0.90rem !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            min-height: 44px !important;
            flex: 1 !important;
            justify-content: center !important;
        }}

        /* Zone card touch ergonomics */
        .zone-card {{
            padding: 0.8rem 0.95rem !important;
            border-radius: 12px !important;
            margin-bottom: 0.5rem !important;
        }}
        .zone-card .vwc-big {{ font-size: 1.25rem !important; }}
        .zone-card .zone-name {{ font-size: 0.92rem !important; }}
        .zone-card .zone-action {{ 
            font-size: 0.78rem !important; 
            padding: 0.3rem 0.6rem !important;
        }}

        /* Expander headers touch friendly */
        [data-testid="stExpander"] summary {{
            padding: 0.65rem 0.8rem !important;
            font-size: 0.88rem !important;
            min-height: 44px !important;
        }}

        .plot-table th {{
            padding: 0.35rem 0.5rem !important;
            font-size: 0.65rem !important;
        }}
        .plot-table td {{
            padding: 0.45rem 0.5rem !important;
            font-size: 0.78rem !important;
        }}
        .priority-badge {{
            width: 26px !important; height: 26px !important;
            font-size: 0.82rem !important;
        }}

        /* Stack Streamlit columns on mobile */
        [data-testid="column"] {{
            width: 100% !important;
            flex: 100% !important;
            min-width: 100% !important;
        }}

        /* Allow vertical page scrolling over Plotly charts on mobile */
        .js-plotly-plot .plotly .drag,
        .js-plotly-plot .plotly .nsewdrag,
        .js-plotly-plot .plotly .main-svg {{
            touch-action: pan-y pinch-zoom !important;
        }}
    }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------
# 6. DATA FETCH (always live)
# ----------------------------------------------
now = _dt.datetime.now()
time_key = _get_cache_time_key()
live_data, depth_data, success = fetch_live_data(time_key)

# Fallback logic: persist last-known-good data in session state
if success and live_data:
    st.session_state.last_live_data = live_data
    st.session_state.last_depth_data = depth_data
    st.session_state.connection_ok = True
elif not success:
    st.session_state.connection_ok = False
    if st.session_state.last_live_data:
        live_data = st.session_state.last_live_data
        depth_data = st.session_state.last_depth_data
    else:
        live_data = {}
        depth_data = {}

# Update plots from live readings
for plot in st.session_state.plots:
    if plot.logger_id in live_data:
        plot.update(
            live_data[plot.logger_id],
            per_depth=depth_data.get(plot.logger_id, {}),
        )


# ----------------------------------------------
# 7. CONNECTION WARNING
# ----------------------------------------------
if not st.session_state.connection_ok:
    st.warning(
        "**Connection Lost** — Unable to reach the MajiSys sensor "
        "server. Displaying last known values. Data will refresh "
        "automatically when the connection is restored.",
    )


# ----------------------------------------------
# 8. GLOBAL RISK BANNER
# ----------------------------------------------
farm_risk = overall_risk(st.session_state.plots)
ri = RISK_PALETTE[farm_risk]

st.markdown(
    f"""
    <div class="risk-banner"
         style="background:{ri['bg']}; border:1px solid {ri['color']}33;">
        <div class="risk-banner-left">
            <h1 style="color:{ri['text_color']};">
                🌳 Glanerbeek Forest
                <span class="mode-badge" style="margin-left:6px; vertical-align:middle;">LIVE</span>
            </h1>
            <div class="subtitle" style="color:{ri['text_color']};">
                Farm-wide status&ensp;&middot;&ensp;{len(st.session_state.plots)} zones&ensp;&middot;&ensp;Updated {now.strftime('%H:%M')}
            </div>
        </div>
        <div class="risk-banner-badge" style="background:{ri['color']}22; border:1.5px solid {ri['color']}; color:{ri['color']};">
            <span class="badge-emoji">{ri['emoji']}</span>
            <span class="badge-text">{ri['label']}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------
# 9. LAYOUT -- TABBED (Dashboard + Farm Map)
# ----------------------------------------------
tab_map, tab_dashboard = st.tabs(["🗺️ Farm Map", "📊 Zone Monitor"])

# ============================================
# 10. FARM MAP TAB
# ============================================
with tab_map:
    st.markdown(
        '<div class="section-title">🗺️ Farm Sensor Map</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Spatial overview of all monitoring zones. Each station shows concentric "
        "depth rings colored by moisture status. Hover for exact VWC readings."
    )

    # ── Sensor pixel positions on the satellite image (1200 x 1315 px) ──
    # Mapped from WUNDER PDF page 6 station labels to pixel coordinates.
    # GP-06 EXCLUDED (Water Potential station, not a VWC station).
    SENSOR_POSITIONS = {
        "GP-01": {"x": 327, "y": 1060, "label": "GP-01", "sub": "F1-1 · ATMOS + VWC", "field": 1},
        "GP-02": {"x": 160, "y": 695,  "label": "GP-02", "sub": "F1-2 · Soil Moisture", "field": 1},
        "GP-03": {"x": 324, "y": 811,  "label": "GP-03", "sub": "F1-3 · Soil Moisture", "field": 1},
        "GP-04": {"x": 513, "y": 568,  "label": "GP-04", "sub": "F2-1 · ATMOS + VWC", "field": 2},
        "GP-05": {"x": 428, "y": 231,  "label": "GP-05", "sub": "F2-2 · Soil Moisture", "field": 2},
    }

    # Image dimensions for coordinate mapping
    _IMG_W, _IMG_H = 1200, 1315

    DEPTH_ORDER = ["40 cm", "20 cm", "10 cm", "5 cm"]  # outer to inner
    RING_SIZES = [70, 52, 36, 22]  # marker sizes for concentric rings (larger for mobile tap targets)

    def _vwc_to_color(vwc: float) -> str:
        """Map VWC percentage to a color on a continuous gradient."""
        if vwc >= 25:
            return "rgb(34, 197, 94)"    # bright green
        elif vwc >= 17:
            r = int(34 + (234 - 34) * (25 - vwc) / 8)
            g = int(197 + (179 - 197) * (25 - vwc) / 8)
            b = int(94 + (8 - 94) * (25 - vwc) / 8)
            return f"rgb({r}, {g}, {b})"
        elif vwc >= 9:
            r = int(234 + (239 - 234) * (17 - vwc) / 8)
            g = int(179 + (68 - 179) * (17 - vwc) / 8)
            b = int(8 + (68 - 8) * (17 - vwc) / 8)
            return f"rgb({r}, {g}, {b})"
        else:
            return "rgb(239, 68, 68)"    # bright red

    try:
        import plotly.graph_objects as go
        import base64

        # ── Load satellite image as base64 for Plotly background ──
        _sat_path = Path(__file__).resolve().parent / "satellite_map.png"
        with open(_sat_path, "rb") as _f:
            _sat_b64 = base64.b64encode(_f.read()).decode()

        fig = go.Figure()

        # ── Draw Concentric Rings for Each Sensor Station ──
        for plot in st.session_state.plots:
            pos = SENSOR_POSITIONS.get(plot.id)
            if not pos:
                continue

            # Draw rings from outer (deepest) to inner (shallowest)
            for depth_label, ring_size in zip(DEPTH_ORDER, RING_SIZES):
                vwc_val = plot.per_depth_vwc.get(depth_label)
                if vwc_val is not None:
                    color = _vwc_to_color(vwc_val)
                    hover_text = (
                        f"<b>{plot.id} — {plot.name}</b><br>"
                        f"Depth: {depth_label}<br>"
                        f"VWC: {vwc_val:.1f}%<br>"
                        f"Status: {RISK_PALETTE[vwc_to_risk(vwc_val)]['label']}"
                    )
                else:
                    color = "rgba(107, 114, 128, 0.4)"
                    hover_text = (
                        f"<b>{plot.id} — {plot.name}</b><br>"
                        f"Depth: {depth_label}<br>"
                        f"No data"
                    )

                fig.add_trace(go.Scatter(
                    x=[pos["x"]],
                    y=[_IMG_H - pos["y"]],  # flip Y for image coords
                    mode='markers',
                    marker=dict(
                        size=ring_size,
                        color=color,
                        line=dict(width=1.5, color="rgba(255,255,255,0.65)"),
                    ),
                    hovertemplate=hover_text + "<extra></extra>",
                    showlegend=False,
                ))

            fig.add_annotation(
                x=pos["x"], y=_IMG_H - pos["y"] + 36,  # 36px above the center
                text=f"<b>{plot.id}</b>",
                showarrow=False,
                font=dict(size=13, color="#f8fafc" if dark_mode else "#0f172a"),
            )

        # ── Color scale legend (horizontal at bottom) ──
        legend_x_vals = [120, 270, 420, 570, 720, 870]
        legend_vwc_vals = [5, 9, 13, 17, 21, 25]
        for lx, lv in zip(legend_x_vals, legend_vwc_vals):
            fig.add_trace(go.Scatter(
                x=[lx], y=[-22],
                mode='markers+text',
                marker=dict(size=13, color=_vwc_to_color(lv)),
                text=[f"{lv}%"],
                textposition="bottom center",
                textfont=dict(size=9, color="#9ca3af" if dark_mode else "#64748b"),
                showlegend=False,
                hoverinfo='skip',
            ))

        fig.add_annotation(
            x=50, y=-22,
            text="<b>VWC:</b>",
            showarrow=False,
            font=dict(size=10, color="#9ca3af" if dark_mode else "#475569"),
        )

        # Depth ring legend text
        fig.add_annotation(
            x=_IMG_W / 2, y=-95,
            text="Concentric ring depths: outer = 40 cm → inner = 5 cm",
            showarrow=False,
            font=dict(size=9.5, color="#6b7280" if dark_mode else "#64748b"),
        )

        fig.update_layout(
            height=680,
            margin=dict(l=5, r=5, t=5, b=115),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter"),
            dragmode=False,  # fix image in place (no 1-finger pan/zoom box)
            xaxis=dict(
                range=[0, _IMG_W],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                fixedrange=False,  # allow pinch-to-zoom on mobile
            ),
            yaxis=dict(
                range=[-120, _IMG_H],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="x",
                fixedrange=False,  # allow pinch-to-zoom on mobile
            ),
            images=[dict(
                source=f"data:image/png;base64,{_sat_b64}",
                xref="x", yref="y",
                x=0, y=_IMG_H,
                sizex=_IMG_W, sizey=_IMG_H,
                sizing="stretch",
                opacity=1.0,
                layer="below",
            )],
            hovermode="closest",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="farm_map",
            config={"displayModeBar": False, "responsive": True, "scrollZoom": False, "doubleClick": False},
        )

    except ImportError:
        st.error("Plotly is required for the Farm Map. Install with: `pip install plotly`")

    st.caption(
        "Satellite imagery from WUNDER project (ITC, University of Twente). "
        "Each concentric ring represents a sensor depth "
        "(outer ring = 40 cm, inner = 5 cm). Colors indicate VWC severity on a "
        "continuous gradient from 🟢 optimal to 🔴 critical."
    )








with tab_dashboard:

    # ============================================
    # 9a. ZONE CARDS WITH LIVE DATA
    # ============================================
    st.markdown(
        '<div class="section-title">📡 Live Zone Status</div>',
        unsafe_allow_html=True,
    )

    for plot in st.session_state.plots:
        ri = RISK_PALETTE[plot.risk_level]
        has_data = plot.logger_id in live_data
        vwc_display = f"{plot.vwc:.1f}%" if has_data else "—"

        # Action text
        action_bg = f"{ri['color']}15"
        action_text = ri['action']

        st.markdown(
            f"""
            <div class="zone-card" style="border-color:{ri['color']};">
                <div class="zone-header">
                    <div>
                        <div class="zone-name">
                            <span class="risk-dot"
                                  style="color:{ri['color']};
                                         background:{ri['color']};"></span>
                            {plot.id} — {plot.name}
                        </div>
                        <div class="zone-meta">
                            Field {plot.field}&ensp;&middot;&ensp;{plot.area_ha} ha&ensp;&middot;&ensp;Logger: {plot.logger_id}
                        </div>
                    </div>
                    <div class="vwc-big" style="color:{ri['color']};">
                        {vwc_display}
                    </div>
                </div>
                <div class="zone-action"
                     style="background:{action_bg}; color:{ri['color']};">
                    {ri['emoji']} {action_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Historical trend chart (expandable)
        with st.expander(f"📈 Historical Trend — {plot.id}"):
            _render_historical_chart(plot)

    # ============================================
    # 9b. SPATIAL PRIORITY TABLE
    # ============================================
    st.divider()

    st.markdown(
        '<div class="section-title">🎯 Spatial Priority Ranking</div>',
        unsafe_allow_html=True,
    )

    # Sort plots by priority descending (highest urgency first)
    sorted_plots = sorted(
        st.session_state.plots,
        key=lambda p: p.spatial_priority,
        reverse=True,
    )

    rows_html = ""
    for p in sorted_plots:
        pi = RISK_PALETTE[p.risk_level]
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
            <td style="font-variant-numeric:tabular-nums;color:{'#93c5fd' if dark_mode else '#2563eb'};">
                {p.vwc:.1f}%
            </td>
        </tr>
        """

    st.markdown(
        f"""
        <div class="table-scroll-wrapper">
        <table class="plot-table">
            <thead>
                <tr>
                    <th>Zone</th>
                    <th>Name</th>
                    <th>Area</th>
                    <th>Status</th>
                    <th style="text-align:center;">Priority</th>
                    <th>VWC</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- FOOTER ----
    st.divider()
    st.caption(
        f"Glanerbeek Dashboard V3  |  Live (MajiSys)  |  "
        f"Updated: {now.strftime('%d %b %H:%M')}  |  "
        f"Thresholds: 17% Irrigation Trigger  |  9% Critical (Sandy Loam)"
    )



