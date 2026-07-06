"""Quick standalone script to print fetch_live_data results."""
import datetime as _dt
import sys
from pathlib import Path

import pandas as pd

# Ensure MajiSysUtil is importable
_src_dir = str(Path(__file__).resolve().parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import MajiSysUtil as msu

# ── Constants (copied from app.py) ──
PLOT_DEFINITIONS = [
    {"id": "GP-01", "name": "Field 1 - Station 1 (ATMOS)", "logger_id": "z6-21176", "logger_name": "F1_1_ATMOS_SMST1", "field": 1},
    {"id": "GP-02", "name": "Field 1 - Station 2",         "logger_id": "z6-21178", "logger_name": "F1_2_SMST2",       "field": 1},
    {"id": "GP-03", "name": "Field 1 - Station 3",         "logger_id": "z6-25928", "logger_name": "F1_3_SMST3",       "field": 1},
    {"id": "GP-04", "name": "Field 2 - Station 1 (ATMOS)", "logger_id": "z6-21177", "logger_name": "F2_1_ATMOS_SMST1", "field": 2},
    {"id": "GP-05", "name": "Field 2 - Station 2",         "logger_id": "z6-21179", "logger_name": "F2_2_SMST2",       "field": 2},
    {"id": "GP-06", "name": "Field 2 - Station 3 (WP)",    "logger_id": "z6-30173", "logger_name": "F2_3_WP_SMST",     "field": 2},
]

MAX_DEPTH_CM = 40.0


def _extract_worst_case_vwc(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    top_cols = []
    for col in df.columns:
        try:
            depth_str = col.split()[-1]
            depth_val = float(depth_str.replace("cm", ""))
            if depth_val <= MAX_DEPTH_CM:
                top_cols.append(col)
        except (ValueError, IndexError):
            continue
    if not top_cols:
        return None
    latest_row = df[top_cols].iloc[-1]
    min_vwc_fraction = latest_row.min()
    if pd.isna(min_vwc_fraction):
        return None
    return round(float(min_vwc_fraction) * 100.0, 2)


def fetch_live_data():
    loggers = [(p["logger_id"], p["logger_name"]) for p in PLOT_DEFINITIONS]
    maxdate = _dt.datetime.now()
    mindate = maxdate - _dt.timedelta(hours=24)

    print(f"Time window: {mindate} -> {maxdate}\n")
    print("Downloading timeseries from MajiSys API …")
    raw = msu.downloadTimeseries(loggers, mindate, maxdate)

    print("Grouping by parameter (Soil moisture) …")
    grouped = msu.groupByParameter(raw, ["Soil moisture"], loggers)

    print("Converting to Pandas DataFrames …\n")
    pandas_data = msu.toPandas(grouped)

    results = {}
    if "Soil moisture" in pandas_data:
        for logger_tuple, _unit, df in pandas_data["Soil moisture"]:
            logger_id = logger_tuple[0]
            vwc_pct = _extract_worst_case_vwc(df)
            if vwc_pct is not None:
                results[logger_id] = vwc_pct

    return results, True


if __name__ == "__main__":
    try:
        data_dict, success = fetch_live_data()
        print("=" * 50)
        print(f"Success: {success}")
        print(f"Number of loggers with data: {len(data_dict)}")
        print("=" * 50)
        for logger_id, vwc in data_dict.items():
            # Find friendly name
            name = next(
                (p["name"] for p in PLOT_DEFINITIONS if p["logger_id"] == logger_id),
                logger_id,
            )
            print(f"  {logger_id}  ({name}):  VWC = {vwc:.2f} %")
        if not data_dict:
            print("  (no data returned)")
    except Exception as e:
        print(f"ERROR: {e}")
