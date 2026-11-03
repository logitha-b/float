"""
data/hycom_loader.py
--------------------
HYCOM (Hybrid Coordinate Ocean Model) data fetcher.

Source: NOAA NCEI ERDDAP — no account required, fully public.
  Base URL : https://www.ncei.noaa.gov/erddap/griddap
  Datasets :
    Hycom_sfc_3d  — Global surface + 3D aggregation (temperature, salinity,
                    u/v currents, sea-surface height)

HYCOM provides ocean model output (not direct observations) at 1/12° (~8 km)
horizontal resolution. Useful for:
  • Current speed and direction
  • Model-predicted temperature/salinity when Argo coverage is sparse
  • Sea-surface height anomaly (SSHA) as a proxy for mesoscale eddies

Output schema
-------------
DataFrame columns returned by fetch_hycom_surface():
    TIME      : datetime64[ns]
    LATITUDE  : float
    LONGITUDE : float   (–180 to 180)
    TEMP      : float   °C  (water_temp)
    PSAL      : float   PSU (salinity)
    U_CURR    : float   m/s (eastward current)
    V_CURR    : float   m/s (northward current)
    SSH       : float   m   (sea-surface height above geoid)
    SOURCE    : str     'hycom_model'

All missing variables are filled with NaN so the caller can select what it needs.
"""

import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ERDDAP configuration
# ---------------------------------------------------------------------------
_NCEI_BASE = "https://www.ncei.noaa.gov/erddap/griddap"

# Dataset preference order: global surface+3D first, then regional fallbacks
_HYCOM_DATASETS = [
    ("Hycom_sfc_3d",    ["water_temp", "salinity", "water_u", "water_v", "surf_el"]),
    ("HYCOM_reg1_latest3d", ["water_temp", "salinity", "water_u", "water_v", "surf_el"]),
    ("HYCOM_reg6_latest2d", ["water_temp", "salinity", "water_u", "water_v", "surf_el"]),
]

GRID_RES       = 1 / 12   # ~0.083°
_REQUEST_TIMEOUT = 60     # seconds
_MAX_RETRIES     = 3
_RETRY_BACKOFF   = 2.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_hycom_surface(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch HYCOM surface ocean model fields for a bounding box and date range.

    Parameters
    ----------
    lat_min, lat_max : float  Latitude bounds (–90 to 90).
    lon_min, lon_max : float  Longitude bounds (–180 to 180).
    start_date       : str    ISO date string, e.g. '2023-06-01'.
    end_date         : str    ISO date string, e.g. '2023-08-31'.

    Returns
    -------
    pd.DataFrame with standard schema columns + SOURCE='hycom_model'.
    """
    import requests

    start_date, end_date = _clamp_dates(start_date, end_date)

    # HYCOM uses –180/180 longitude convention (unlike OISST)
    lat_min_g = _snap(lat_min, GRID_RES)
    lat_max_g = _snap(lat_max, GRID_RES)
    if lat_min_g >= lat_max_g:
        lat_max_g = lat_min_g + GRID_RES

    lon_min_g = _snap(lon_min, GRID_RES)
    lon_max_g = _snap(lon_max, GRID_RES)
    if lon_min_g >= lon_max_g:
        lon_max_g = lon_min_g + GRID_RES

    last_exc: Optional[Exception] = None
    for dataset_id, variables in _HYCOM_DATASETS:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                df = _download_hycom_nc(
                    dataset_id, variables,
                    lat_min_g, lat_max_g, lon_min_g, lon_max_g,
                    start_date, end_date,
                )
                logger.info(
                    "HYCOM [%s..%s] %d rows from %s",
                    start_date, end_date, len(df), dataset_id,
                )
                df["SOURCE"] = "hycom_model"
                return df
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF * attempt)

    raise RuntimeError(
        f"HYCOM fetch failed for {start_date}–{end_date}.\n"
        f"Last error: {last_exc}\n"
        "Check internet connection; NCEI ERDDAP may be temporarily unavailable."
    )


def fetch_hycom_current_speed(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    radius_deg: float = 1.0,
) -> pd.DataFrame:
    """
    Return daily-mean current speed (m/s) and direction (degrees) at a point.

    Returns pd.DataFrame with columns: TIME, speed_ms, direction_deg
    """
    df = fetch_hycom_surface(
        lat_min=lat - radius_deg, lat_max=lat + radius_deg,
        lon_min=lon - radius_deg, lon_max=lon + radius_deg,
        start_date=start_date, end_date=end_date,
    )
    if "U_CURR" not in df.columns or "V_CURR" not in df.columns:
        raise RuntimeError("HYCOM response did not include current components.")

    df = df.dropna(subset=["U_CURR", "V_CURR"])
    df["speed_ms"]      = np.sqrt(df["U_CURR"]**2 + df["V_CURR"]**2)
    df["direction_deg"] = (np.degrees(np.arctan2(df["U_CURR"], df["V_CURR"])) + 360) % 360

    result = (df.groupby("TIME")[["speed_ms", "direction_deg"]]
                .mean()
                .reset_index()
                .rename(columns={"TIME": "TIME"}))
    return result.sort_values("TIME").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal download + parse
# ---------------------------------------------------------------------------

def _download_hycom_nc(
    dataset_id: str,
    variables: List[str],
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    start_date: str, end_date: str,
) -> pd.DataFrame:
    import requests
    try:
        import netCDF4 as nc4  # type: ignore
    except ImportError as exc:
        raise RuntimeError("netCDF4 not installed: pip install netCDF4") from exc

    # Build variable selector string (only request depth level 0 = surface)
    var_str = ",".join(
        f"{v}[({start_date}T00:00:00Z):1:({end_date}T00:00:00Z)]"
        f"[(0.0):1:(0.0)]"
        f"[({lat_min:.4f}):1:({lat_max:.4f})]"
        f"[({lon_min:.4f}):1:({lon_max:.4f})]"
        for v in variables
    )
    url = f"{_NCEI_BASE}/{dataset_id}.nc?{var_str}"
    logger.debug("HYCOM URL: %s", url[:200])

    resp = requests.get(url, timeout=_REQUEST_TIMEOUT, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(
            f"HTTP {resp.status_code} from HYCOM dataset '{dataset_id}': "
            f"{resp.text[:200]}"
        )

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc", prefix="hycom_")
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)

        if os.path.getsize(tmp_path) < 200:
            raise RuntimeError("HYCOM response too small — likely an error page.")

        ds = nc4.Dataset(tmp_path)
        try:
            rows = _parse_hycom_dataset(ds, variables)
        finally:
            ds.close()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not rows:
        return pd.DataFrame(columns=["TIME", "LATITUDE", "LONGITUDE",
                                      "TEMP", "PSAL", "U_CURR", "V_CURR", "SSH"])
    df = pd.DataFrame(rows)
    df["TIME"] = pd.to_datetime(df["TIME"])
    return df


def _parse_hycom_dataset(ds, variables: List[str]) -> list:
    import netCDF4 as nc4  # type: ignore

    # Time
    time_var = ds.variables["time"]
    times_nc = nc4.num2date(
        time_var[:], units=time_var.units,
        calendar=getattr(time_var, "calendar", "standard"),
    )
    times_pd = pd.to_datetime([t.isoformat() for t in times_nc]).normalize()

    lats = ds.variables["lat"][:]
    lons = ds.variables["lon"][:]

    # Variable name → output column mapping
    var_col_map = {
        "water_temp": "TEMP",
        "salinity":   "PSAL",
        "water_u":    "U_CURR",
        "water_v":    "V_CURR",
        "surf_el":    "SSH",
    }

    # Extract arrays, dropping depth dimension if present
    arrays = {}
    for vname in variables:
        if vname not in ds.variables:
            continue
        arr = ds.variables[vname][:]
        if arr.ndim == 4:
            arr = arr[:, 0, :, :]  # drop depth
        arrays[vname] = arr

    rows = []
    for ti, t in enumerate(times_pd):
        for lati, lat_val in enumerate(lats):
            for loni, lon_val in enumerate(lons):
                row = {"TIME": t, "LATITUDE": float(lat_val), "LONGITUDE": float(lon_val)}
                for vname, col in var_col_map.items():
                    if vname in arrays:
                        cell = arrays[vname][ti, lati, loni]
                        row[col] = float(cell) if not np.ma.is_masked(cell) else np.nan
                    else:
                        row[col] = np.nan
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _clamp_dates(start_date: str, end_date: str) -> Tuple[str, str]:
    """HYCOM archive starts ~2000; cap to yesterday."""
    hycom_start = pd.Timestamp("1994-01-01")
    yesterday   = pd.Timestamp(datetime.now(timezone.utc).date()) - pd.Timedelta(days=1)
    ts = max(pd.Timestamp(start_date), hycom_start)
    te = min(pd.Timestamp(end_date),   yesterday)
    if ts > te:
        te = yesterday
        ts = te - pd.Timedelta(days=7)
    return ts.strftime("%Y-%m-%d"), te.strftime("%Y-%m-%d")


def _snap(value: float, grid: float) -> float:
    return round(round(value / grid) * grid, 6)
