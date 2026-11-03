"""
data/wod_loader.py
------------------
NOAA World Ocean Database (WOD) loader.

Source: NOAA CoastWatch ERDDAP — no account required.
  Base URL : https://coastwatch.pfeg.noaa.gov/erddap/tabledap
  Dataset  : WOD_PFL_*  (WOD CTD and profiling float observations)

WOD is the world's largest publicly available collection of ocean profile data,
covering temperature and salinity measurements from 1772 to present from
multiple instrument types (CTD, XBT, Argo, moorings, etc.).

This loader fetches WOD temperature/salinity profiles for a region/date range
and returns them in the same standard schema as argo_loader so they can be
fed directly into qc_module.apply_qc_filter().

Output schema (same as argo_loader)
------------------------------------
PLATFORM_NUMBER  : str   – WOD cast number (as string)
CYCLE_NUMBER     : int   – 0 (WOD doesn't have cycles; placeholder for schema compat)
LATITUDE         : float
LONGITUDE        : float
TIME             : datetime64[ns]
PRES             : float – pressure in dbar (converted from depth if needed)
TEMP             : float – temperature °C
PSAL             : float – salinity PSU
TEMP_QC          : int   – WOD quality flag mapped to Argo scale (1=good, 4=bad)
PSAL_QC          : int
DATA_MODE        : str   – 'D' (WOD data is delayed-mode by definition)
SOURCE           : str   – 'wod'
"""

import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ERDDAP_BASE    = "https://coastwatch.pfeg.noaa.gov/erddap/tabledap"
_WOD_DATASET    = "wod_ctd"          # WOD CTD observations
_REQUEST_TIMEOUT = 60
_MAX_RETRIES     = 3

# WOD quality flag → Argo QC flag mapping
# WOD: 0=accepted, 1=failed range, 2=failed gradient, 3=failed spike,
#      4=climatology, 6=failed density inversion, 9=missing
_WOD_TO_ARGO_QC = {
    0: 1,   # good
    1: 4,   # bad (range)
    2: 3,   # probably bad (gradient)
    3: 4,   # bad (spike)
    4: 2,   # probably good (climatology check passed)
    6: 4,   # bad (density inversion)
    9: 9,   # missing
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_wod_profiles(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch WOD CTD profiles for a bounding box and date range.

    Returns pd.DataFrame in the standard Argo-compatible schema + SOURCE='wod'.
    """
    start_date, end_date = _clamp_dates(start_date, end_date)

    # WOD tabledap: use CSV format (simpler than nc for tabular data)
    url = (
        f"{_ERDDAP_BASE}/{_WOD_DATASET}.csv"
        f"?cast,time,latitude,longitude,depth,Temperature,Salinity,"
        f"Temperature_WODflag,Salinity_WODflag"
        f"&latitude>={lat_min:.4f}&latitude<={lat_max:.4f}"
        f"&longitude>={lon_min:.4f}&longitude<={lon_max:.4f}"
        f"&time>={start_date}T00:00:00Z"
        f"&time<={end_date}T23:59:59Z"
    )
    logger.debug("WOD URL: %s", url[:200])

    last_exc: Optional[Exception] = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            import requests
            resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
            if resp.status_code == 404:
                # Dataset may not exist on this ERDDAP; try alternate
                return _fetch_wod_alternate(
                    lat_min, lat_max, lon_min, lon_max, start_date, end_date
                )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"WOD ERDDAP HTTP {resp.status_code}: {resp.text[:200]}"
                )
            df = _parse_csv_response(resp.text)
            logger.info("WOD: %d rows fetched", len(df))
            return df
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                time.sleep(2 * attempt)

    raise RuntimeError(
        f"WOD fetch failed after {_MAX_RETRIES} attempts.\n"
        f"Last error: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Alternate WOD access via NOAA NCEI ERDDAP
# ---------------------------------------------------------------------------

def _fetch_wod_alternate(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    start_date: str, end_date: str,
) -> pd.DataFrame:
    """
    Fallback: query NOAA NCEI ERDDAP for WOD data.
    Dataset: https://www.ncei.noaa.gov/erddap/tabledap/wod_ctd_2018.html
    """
    import requests

    ncei_base = "https://www.ncei.noaa.gov/erddap/tabledap"
    datasets_to_try = ["wod_ctd", "wod_ctd_2018", "wod_OSD"]

    for dataset_id in datasets_to_try:
        url = (
            f"{ncei_base}/{dataset_id}.csv"
            f"?cast,time,latitude,longitude,depth,Temperature,Salinity,"
            f"Temperature_WODflag,Salinity_WODflag"
            f"&latitude>={lat_min:.4f}&latitude<={lat_max:.4f}"
            f"&longitude>={lon_min:.4f}&longitude<={lon_max:.4f}"
            f"&time>={start_date}T00:00:00Z"
            f"&time<={end_date}T23:59:59Z"
        )
        try:
            resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
            if resp.status_code == 200:
                df = _parse_csv_response(resp.text)
                logger.info("WOD alt (%s): %d rows fetched", dataset_id, len(df))
                return df
        except Exception as exc:
            logger.debug("WOD alt dataset %s failed: %s", dataset_id, exc)

    # Return empty DataFrame rather than crashing — WOD is supplementary
    logger.warning(
        "All WOD sources failed for region lat=[%s,%s] lon=[%s,%s] "
        "dates=[%s,%s]. Returning empty DataFrame.",
        lat_min, lat_max, lon_min, lon_max, start_date, end_date,
    )
    return _empty_df()


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def _parse_csv_response(csv_text: str) -> pd.DataFrame:
    """Parse ERDDAP CSV response into the standard schema."""
    import io

    if not csv_text or len(csv_text) < 50:
        return _empty_df()

    lines = csv_text.strip().splitlines()
    if len(lines) < 3:  # header + units + at least one data row
        return _empty_df()

    # ERDDAP CSV: row 0 = column names, row 1 = units, row 2+ = data
    header   = lines[0].split(",")
    data_rows = lines[2:]  # skip units row

    if not data_rows:
        return _empty_df()

    raw = pd.read_csv(io.StringIO("\n".join([lines[0]] + data_rows)))

    # Normalise column names
    col_map = {
        "cast":              "PLATFORM_NUMBER",
        "time":              "TIME",
        "latitude":          "LATITUDE",
        "longitude":         "LONGITUDE",
        "depth":             "PRES",      # depth ≈ pressure in dbar for shallow water
        "temperature":       "TEMP",
        "salinity":          "PSAL",
        "temperature_wodflag": "TEMP_QC_RAW",
        "salinity_wodflag":    "PSAL_QC_RAW",
    }
    raw.columns = [c.strip().lower() for c in raw.columns]
    raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

    # Fill required columns
    if "PLATFORM_NUMBER" not in raw.columns:
        raw["PLATFORM_NUMBER"] = "wod_unknown"
    if "CYCLE_NUMBER" not in raw.columns:
        raw["CYCLE_NUMBER"] = 0

    raw["PLATFORM_NUMBER"] = raw["PLATFORM_NUMBER"].astype(str).str.strip()
    raw["CYCLE_NUMBER"]    = 0
    raw["DATA_MODE"]       = "D"  # WOD is always delayed-mode
    raw["SOURCE"]          = "wod"

    # Map WOD QC flags to Argo-compatible scale
    for raw_col, out_col in [("TEMP_QC_RAW", "TEMP_QC"), ("PSAL_QC_RAW", "PSAL_QC")]:
        if raw_col in raw.columns:
            raw[out_col] = (
                pd.to_numeric(raw[raw_col], errors="coerce")
                  .fillna(9)
                  .astype(int)
                  .map(lambda x: _WOD_TO_ARGO_QC.get(x, 9))
            )
        else:
            raw[out_col] = 1  # assume good if flag absent

    # Time parse
    raw["TIME"] = pd.to_datetime(raw.get("TIME"), errors="coerce", utc=True)
    if raw["TIME"].dt.tz is not None:
        raw["TIME"] = raw["TIME"].dt.tz_localize(None)

    # Numeric coercions
    for col in ("LATITUDE", "LONGITUDE", "PRES", "TEMP", "PSAL"):
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # Return only standard schema columns that exist
    std_cols = ["PLATFORM_NUMBER", "CYCLE_NUMBER", "LATITUDE", "LONGITUDE",
                "TIME", "PRES", "TEMP", "PSAL", "TEMP_QC", "PSAL_QC",
                "DATA_MODE", "SOURCE"]
    return raw[[c for c in std_cols if c in raw.columns]].reset_index(drop=True)


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "PLATFORM_NUMBER", "CYCLE_NUMBER", "LATITUDE", "LONGITUDE",
        "TIME", "PRES", "TEMP", "PSAL", "TEMP_QC", "PSAL_QC",
        "DATA_MODE", "SOURCE",
    ])


def _clamp_dates(start_date: str, end_date: str) -> Tuple[str, str]:
    """WOD archive goes back to 1772; cap to yesterday."""
    yesterday = pd.Timestamp(datetime.now(timezone.utc).date()) - pd.Timedelta(days=1)
    ts = pd.Timestamp(start_date)
    te = min(pd.Timestamp(end_date), yesterday)
    if ts > te:
        te = yesterday
        ts = te - pd.Timedelta(days=30)
    return ts.strftime("%Y-%m-%d"), te.strftime("%Y-%m-%d")
