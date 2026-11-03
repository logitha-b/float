"""
data/drifter_loader.py
----------------------
NOAA Global Drifter Program (GDP) surface drifter loader.

Source: NOAA AOML ERDDAP — no account required, fully public.
  Base URL : https://erddap.aoml.noaa.gov/gdp/erddap/tabledap
  Datasets :
    drifters_6hourly  — 6-hourly quality-controlled interpolated drifter data
    drifters_hourly   — hourly drifter observations

The GDP network tracks ~1,000+ freely drifting buoys at the ocean surface,
measuring sea-surface temperature and providing surface current velocities
(from drifter displacement). The dataset covers 1979–present.

Useful for:
  • Real-time surface temperature from in-situ buoys (independent of Argo)
  • Surface current patterns (Lagrangian tracking)
  • Cross-validation of HYCOM model currents

Output schema
-------------
DataFrame columns returned by fetch_drifter_sst():
    TIME        : datetime64[ns]
    LATITUDE    : float
    LONGITUDE   : float   (–180 to 180)
    SST         : float   °C  (sea-surface temperature from drifter)
    U_CURR      : float   m/s (eastward velocity, derived from displacement)
    V_CURR      : float   m/s (northward velocity)
    DRIFTER_ID  : str     unique drifter ID
    SOURCE      : str     'gdp_drifter'
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_AOML_BASE      = "https://erddap.aoml.noaa.gov/gdp/erddap/tabledap"
# Correct verified dataset IDs on AOML ERDDAP (June 2024)
_DATASETS       = ["drifter_6hour_qc", "drifter_hourly_qc"]
_REQUEST_TIMEOUT = 25
_MAX_RETRIES     = 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_drifter_sst(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch GDP surface drifter observations for a bounding box and date range.

    Returns pd.DataFrame with columns: TIME, LATITUDE, LONGITUDE, SST,
    U_CURR, V_CURR, DRIFTER_ID, SOURCE
    """
    start_date, end_date = _clamp_dates(start_date, end_date)

    last_exc: Optional[Exception] = None
    for dataset_id in _DATASETS:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                df = _fetch_csv(
                    dataset_id, lat_min, lat_max, lon_min, lon_max,
                    start_date, end_date,
                )
                if df is not None and not df.empty:
                    logger.info(
                        "GDP Drifter [%s..%s] %d rows from %s",
                        start_date, end_date, len(df), dataset_id,
                    )
                    return df
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(2 * attempt)
                logger.debug("Drifter attempt %d/%d failed (%s): %s",
                             attempt, _MAX_RETRIES, dataset_id, exc)

    logger.warning(
        "All GDP drifter sources failed for [%s..%s]. "
        "Returning empty DataFrame. Last error: %s",
        start_date, end_date, last_exc,
    )
    return _empty_df()


def fetch_drifter_timeseries(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    radius_deg: float = 2.0,
) -> pd.DataFrame:
    """
    Return daily-mean drifter SST near a point, as a time series.

    Returns pd.DataFrame with columns: date, sst (compatible with SST schema)
    """
    df = fetch_drifter_sst(
        lat_min=lat - radius_deg, lat_max=lat + radius_deg,
        lon_min=lon - radius_deg, lon_max=lon + radius_deg,
        start_date=start_date, end_date=end_date,
    )
    if df.empty or "SST" not in df.columns:
        return pd.DataFrame(columns=["date", "sst"])

    df["date"] = pd.to_datetime(df["TIME"]).dt.normalize()
    ts = df.groupby("date")["SST"].mean().reset_index().rename(columns={"SST": "sst"})
    return ts.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal fetch
# ---------------------------------------------------------------------------

def _fetch_csv(
    dataset_id: str,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    start_date: str, end_date: str,
) -> Optional[pd.DataFrame]:
    import requests, io

    url = (
        f"{_AOML_BASE}/{dataset_id}.csv"
        f"?ID,time,latitude,longitude,sst,ve,vn"
        f"&latitude>={lat_min:.4f}&latitude<={lat_max:.4f}"
        f"&longitude>={lon_min:.4f}&longitude<={lon_max:.4f}"
        f"&time>={start_date}T00:00:00Z"
        f"&time<={end_date}T23:59:59Z"
        f'&orderBy("time")'
    )
    logger.debug("GDP Drifter URL: %s", url[:200])

    resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
    if resp.status_code == 404:
        raise RuntimeError(f"Dataset {dataset_id} not found on AOML ERDDAP (404)")
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} from {dataset_id}")

    text = resp.text
    lines = text.strip().splitlines()
    if len(lines) < 3:
        return _empty_df()

    # ERDDAP CSV: row 0 = headers, row 1 = units, row 2+ = data
    raw = pd.read_csv(io.StringIO("\n".join([lines[0]] + lines[2:])))
    raw.columns = [c.strip().lower() for c in raw.columns]

    rename = {
        "id":         "DRIFTER_ID",
        "ID":         "DRIFTER_ID",
        "time":       "TIME",
        "latitude":   "LATITUDE",
        "longitude":  "LONGITUDE",
        "sst":        "SST",
        "ve":         "U_CURR",
        "vn":         "V_CURR",
    }
    raw = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns})

    for col in ("LATITUDE", "LONGITUDE", "SST", "U_CURR", "V_CURR"):
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # Replace AOML fill values (-999, -999999) with NaN
    for col in ("SST", "U_CURR", "V_CURR"):
        if col in raw.columns:
            raw.loc[raw[col] < -100, col] = float("nan")
            raw.loc[raw[col] > 100 if col != "SST" else raw[col] > 50, col] = float("nan")

    raw["TIME"]   = pd.to_datetime(raw.get("TIME"), errors="coerce", utc=True)
    if hasattr(raw["TIME"].dt, "tz") and raw["TIME"].dt.tz is not None:
        raw["TIME"] = raw["TIME"].dt.tz_localize(None)

    raw["SOURCE"] = "gdp_drifter"
    if "DRIFTER_ID" not in raw.columns:
        raw["DRIFTER_ID"] = "unknown"
    raw["DRIFTER_ID"] = raw["DRIFTER_ID"].astype(str)

    return raw.dropna(subset=["LATITUDE", "LONGITUDE", "TIME"]).reset_index(drop=True)


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "TIME", "LATITUDE", "LONGITUDE", "SST", "U_CURR", "V_CURR",
        "DRIFTER_ID", "SOURCE",
    ])


def _clamp_dates(start_date: str, end_date: str) -> Tuple[str, str]:
    """GDP archive starts 1979; cap to yesterday."""
    gdp_start = pd.Timestamp("1979-01-01")
    yesterday  = pd.Timestamp(datetime.now(timezone.utc).date()) - pd.Timedelta(days=1)
    ts = max(pd.Timestamp(start_date), gdp_start)
    te = min(pd.Timestamp(end_date),   yesterday)
    if ts > te:
        te = yesterday
        ts = te - pd.Timedelta(days=30)
    return ts.strftime("%Y-%m-%d"), te.strftime("%Y-%m-%d")
