"""
data/coral_watch_loader.py
--------------------------
NOAA Coral Reef Watch (CRW) bleaching alert loader.

Source: NOAA CoastWatch ERDDAP — no account required.
  Dataset : NOAA_DHW   (Degree Heating Weeks + bleaching alert area)
  Base URL: https://coastwatch.pfeg.noaa.gov/erddap/griddap

Coral Reef Watch monitors thermal stress on coral reefs globally. Key products:
  • Degree Heating Weeks (DHW): accumulated heat stress (°C-weeks)
    - DHW ≥ 4 → Watch; DHW ≥ 8 → Alert Level 1; DHW ≥ 12 → Alert Level 2
  • Sea Surface Temperature (SST) from satellite (5 km resolution)
  • SST Anomaly vs. Maximum Monthly Mean (MMM) climatology

Output schema
-------------
fetch_coral_bleaching_alert() → pd.DataFrame with columns:
    TIME         : datetime64[ns]
    LATITUDE     : float
    LONGITUDE    : float   (–180 to 180)
    DHW          : float   Degree Heating Weeks (°C-weeks)
    SST          : float   °C (satellite SST)
    SST_ANOMALY  : float   °C above MMM
    ALERT_LEVEL  : int     0=no stress, 1=watch, 2=warning, 3=alert1, 4=alert2
    SOURCE       : str     'noaa_crw'

bleaching_alert_summary() → plain-language string describing current risk.
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

_COASTWATCH_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
# NOAA_DHW on CoastWatch — confirmed working, has CRW_DHW only (no CRW_SST)
_CRW_DATASET     = "NOAA_DHW"
_REQUEST_TIMEOUT = 25
_MAX_RETRIES     = 2

# Degree Heating Week → alert level mapping
def _dhw_to_alert(dhw: float) -> int:
    if dhw >= 12:  return 4  # Bleaching Alert Level 2
    if dhw >= 8:   return 3  # Bleaching Alert Level 1
    if dhw >= 4:   return 1  # Bleaching Watch
    if dhw >= 0:   return 0  # No stress
    return 0

_ALERT_LABELS = {
    0: "No Stress",
    1: "Bleaching Watch (DHW ≥ 4)",
    2: "Bleaching Warning",
    3: "Bleaching Alert Level 1 (DHW ≥ 8)",
    4: "Bleaching Alert Level 2 (DHW ≥ 12)",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_coral_bleaching_alert(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch NOAA CRW bleaching stress indicators.

    The CRW ERDDAP times out on large regions — automatically caps to a
    4°×4° centred sub-box if the input box is larger.
    """
    start_date, end_date = _clamp_dates(start_date, end_date)

    # Cap to 4°×4° to avoid ERDDAP timeouts on large regions
    lat_c = (lat_min + lat_max) / 2
    lon_c = (lon_min + lon_max) / 2
    if abs(lat_max - lat_min) > 4 or abs(lon_max - lon_min) > 4:
        lat_min, lat_max = lat_c - 2, lat_c + 2
        lon_min, lon_max = lon_c - 2, lon_c + 2

    # CRW uses 0–360 longitude convention
    lon_min_360 = lon_min % 360
    lon_max_360 = lon_max % 360

    datasets = [
        (_COASTWATCH_BASE, _CRW_DATASET, ["CRW_DHW"]),
    ]

    last_exc: Optional[Exception] = None
    for base_url, dataset_id, variables in datasets:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                df = _download_crw_nc(
                    base_url, dataset_id, variables,
                    lat_min, lat_max, lon_min_360, lon_max_360,
                    start_date, end_date,
                )
                if df is not None and not df.empty:
                    logger.info(
                        "CRW [%s..%s] %d rows from %s",
                        start_date, end_date, len(df), dataset_id,
                    )
                    return df
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(2 * attempt)

    logger.warning(
        "CRW fetch failed for [%s..%s]: %s. Returning empty.", start_date, end_date, last_exc
    )
    return _empty_df()


def bleaching_alert_summary(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    start_date: str, end_date: str,
) -> str:
    """
    Return a plain-language summary of coral bleaching risk for a region/period.
    """
    df = fetch_coral_bleaching_alert(
        lat_min, lat_max, lon_min, lon_max, start_date, end_date
    )
    if df.empty:
        return (
            "Coral bleaching stress data is not available for this region or period. "
            "The NOAA Coral Reef Watch product covers tropical coral reef areas (±30° latitude)."
        )

    max_dhw    = float(df["DHW"].max())
    mean_dhw   = float(df["DHW"].mean())
    max_alert  = int(df["ALERT_LEVEL"].max())
    alert_name = _ALERT_LABELS.get(max_alert, "Unknown")
    pct_stress = float((df["DHW"] >= 4).mean() * 100)

    lines = [
        f"**NOAA Coral Reef Watch — Bleaching Stress Summary**",
        f"Period: {start_date} to {end_date}",
        f"",
        f"• Maximum Degree Heating Weeks: **{max_dhw:.1f} °C-weeks**",
        f"• Mean DHW across region: {mean_dhw:.1f} °C-weeks",
        f"• Highest alert level reached: **{alert_name}**",
        f"• Area under thermal stress (DHW ≥ 4): {pct_stress:.0f}% of region",
    ]

    if max_alert >= 3:
        lines.append(
            "\n⚠️ Significant coral bleaching is likely in this region. "
            "DHW ≥ 8 causes mass bleaching and mortality in most coral species."
        )
    elif max_alert >= 1:
        lines.append(
            "\n⚠️ Elevated thermal stress detected. Monitor closely — "
            "bleaching may occur if conditions persist."
        )
    else:
        lines.append("\n✅ No significant bleaching thermal stress detected.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal download
# ---------------------------------------------------------------------------

def _download_crw_nc(
    base_url: str,
    dataset_id: str, variables: list,
    lat_min: float, lat_max: float,
    lon_min_360: float, lon_max_360: float,
    start_date: str, end_date: str,
) -> pd.DataFrame:
    import requests
    try:
        import netCDF4 as nc4  # type: ignore
    except ImportError as exc:
        raise RuntimeError("netCDF4 not installed: pip install netCDF4") from exc

    var_str = ",".join(
        f"{v}[({start_date}T12:00:00Z):1:({end_date}T12:00:00Z)]"
        f"[({lat_min:.4f}):1:({lat_max:.4f})]"
        f"[({lon_min_360:.4f}):1:({lon_max_360:.4f})]"
        for v in variables
    )
    url = f"{base_url}/{dataset_id}.nc?{var_str}"
    logger.debug("CRW URL: %s", url[:200])

    resp = requests.get(url, timeout=_REQUEST_TIMEOUT, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} from {dataset_id}")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc", prefix="crw_")
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)

        if os.path.getsize(tmp_path) < 200:
            raise RuntimeError("CRW response too small.")

        ds = nc4.Dataset(tmp_path)
        try:
            rows = _parse_crw_dataset(ds, variables)
        finally:
            ds.close()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not rows:
        return _empty_df()

    df = pd.DataFrame(rows)
    df["TIME"]        = pd.to_datetime(df["TIME"])
    df["ALERT_LEVEL"] = df["DHW"].apply(_dhw_to_alert)
    df["SOURCE"]      = "noaa_crw"
    return df


def _parse_crw_dataset(ds, variables: list) -> list:
    import netCDF4 as nc4  # type: ignore

    time_var = ds.variables["time"]
    times_nc = nc4.num2date(
        time_var[:], units=time_var.units,
        calendar=getattr(time_var, "calendar", "standard"),
    )
    times_pd = pd.to_datetime([t.isoformat() for t in times_nc]).normalize()

    lats = ds.variables["latitude"][:] if "latitude" in ds.variables else ds.variables["lat"][:]
    lons = ds.variables["longitude"][:] if "longitude" in ds.variables else ds.variables["lon"][:]

    var_col = {
        "CRW_DHW":        "DHW",
        "CRW_SST":        "SST",
        "CRW_SSTANOMALY": "SST_ANOMALY",
    }
    # Only load arrays that are actually present in this dataset
    arrays = {}
    for v in variables:
        if v in ds.variables:
            arr = ds.variables[v][:]
            if arr.ndim == 3:
                arrays[v] = arr

    rows = []
    for ti, t in enumerate(times_pd):
        for lati, lat_val in enumerate(lats):
            for loni, lon_val in enumerate(lons):
                row: dict = {"TIME": t, "LATITUDE": float(lat_val)}
                lon180 = float(lon_val) - 360 if float(lon_val) > 180 else float(lon_val)
                row["LONGITUDE"] = lon180

                # Fill all expected columns (nan if not in this dataset)
                for vname, col in var_col.items():
                    if vname in arrays:
                        cell = arrays[vname][ti, lati, loni]
                        row[col] = float(cell) if not np.ma.is_masked(cell) else np.nan
                    else:
                        row[col] = np.nan

                # Only keep rows where we have actual DHW data
                if not np.isnan(row.get("DHW", np.nan)):
                    rows.append(row)
    return rows


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "TIME", "LATITUDE", "LONGITUDE", "DHW", "SST",
        "SST_ANOMALY", "ALERT_LEVEL", "SOURCE",
    ])


def _clamp_dates(start_date: str, end_date: str) -> Tuple[str, str]:
    """CRW archive starts 1985; cap to yesterday."""
    crw_start = pd.Timestamp("1985-01-01")
    yesterday  = pd.Timestamp(datetime.now(timezone.utc).date()) - pd.Timedelta(days=1)
    ts = max(pd.Timestamp(start_date), crw_start)
    te = min(pd.Timestamp(end_date),   yesterday)
    if ts > te:
        te = yesterday
        ts = te - pd.Timedelta(days=30)
    return ts.strftime("%Y-%m-%d"), te.strftime("%Y-%m-%d")
