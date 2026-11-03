"""
data/cmems_loader.py
--------------------
Copernicus Marine Environment Monitoring Service (CMEMS) data loader.

CMEMS provides European-quality ocean analysis and forecast products.
Since April 2024, CMEMS moved from MOTU/OPeNDAP to the new Copernicus
Marine Data Store. This loader uses the PUBLIC ERDDAP mirror at
MyOcean / IFREMER for no-auth access, plus the copernicusmarine Python
toolbox (optional, requires free registration).

Two access modes:
  1. No-auth ERDDAP mirror (MyOcean/IFREMER) — works without any account
     for several reanalysis and satellite products.
  2. copernicusmarine toolbox — requires free CMEMS account. If credentials
     are set via env vars COPERNICUSMARINE_SERVICE_USERNAME /
     COPERNICUSMARINE_SERVICE_PASSWORD, this mode is used automatically
     for the full product catalogue.

Products accessed:
  • GLOBAL_MULTIYEAR_PHY_001_030 — Global ocean physics reanalysis
    (temperature, salinity, currents, sea level: 1993–present, 1/12°)
  • OCEANCOLOUR_GLO_BGC_L4_MY_009_104 — Global chlorophyll-a (4 km, satellite)

Output schema
-------------
fetch_cmems_physics() → pd.DataFrame:
    TIME, LATITUDE, LONGITUDE, TEMP, PSAL, U_CURR, V_CURR, SSH, SOURCE

fetch_cmems_chlorophyll() → pd.DataFrame:
    TIME, LATITUDE, LONGITUDE, CHLOR_A (mg/m³), SOURCE
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# IFREMER public ERDDAP mirror (no auth needed for reanalysis products)
_IFREMER_BASE = "https://erddap.ifremer.fr/erddap/griddap"

# Dataset IDs on the IFREMER ERDDAP mirror
_CMEMS_PHYS_DATASETS = [
    ("cmems_mod_glo_phy_my_0.083deg_P1D-m", ["thetao", "so", "uo", "vo", "zos"]),
    ("global-reanalysis-phy-001-030-daily",  ["thetao", "so", "uo", "vo", "zos"]),
]

_CMEMS_CHL_DATASETS = [
    ("cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D", ["CHL"]),
]

_REQUEST_TIMEOUT = 60
_MAX_RETRIES     = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_cmems_physics(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    start_date: str, end_date: str,
    depth: float = 0.5,
) -> pd.DataFrame:
    """
    Fetch CMEMS global ocean physics (temp, salinity, currents, SSH).

    Tries the IFREMER ERDDAP mirror first (no auth). If available and
    COPERNICUSMARINE credentials are set, uses the toolbox as fallback.

    Returns pd.DataFrame with standard schema + SOURCE='cmems_physics'.
    """
    start_date, end_date = _clamp_dates(start_date, end_date)

    # Try IFREMER ERDDAP mirror first
    for dataset_id, variables in _CMEMS_PHYS_DATASETS:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                df = _download_cmems_nc(
                    _IFREMER_BASE, dataset_id, variables,
                    lat_min, lat_max, lon_min, lon_max,
                    start_date, end_date, depth,
                )
                if df is not None and not df.empty:
                    logger.info("CMEMS physics: %d rows from %s", len(df), dataset_id)
                    df["SOURCE"] = "cmems_physics"
                    return df
            except Exception as exc:
                if attempt < _MAX_RETRIES:
                    time.sleep(2 * attempt)
                logger.debug("CMEMS phys attempt %d failed (%s): %s",
                             attempt, dataset_id, exc)

    # Try copernicusmarine toolbox if credentials available
    if _cmems_credentials_available():
        try:
            return _fetch_via_toolbox(
                "cmems_mod_glo_phy_my_0.083deg_P1D-m",
                ["thetao", "so", "uo", "vo", "zos"],
                lat_min, lat_max, lon_min, lon_max,
                start_date, end_date, depth,
            )
        except Exception as exc:
            logger.warning("CMEMS toolbox fetch failed: %s", exc)

    logger.warning("All CMEMS physics sources failed. Returning empty DataFrame.")
    return _empty_physics_df()


def fetch_cmems_chlorophyll(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    start_date: str, end_date: str,
) -> pd.DataFrame:
    """
    Fetch CMEMS satellite chlorophyll-a for a region (mg/m³).

    Returns pd.DataFrame with columns: TIME, LATITUDE, LONGITUDE, CHLOR_A, SOURCE
    """
    start_date, end_date = _clamp_dates(start_date, end_date)

    for dataset_id, variables in _CMEMS_CHL_DATASETS:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                df = _download_cmems_nc(
                    _IFREMER_BASE, dataset_id, variables,
                    lat_min, lat_max, lon_min, lon_max,
                    start_date, end_date, depth=None,
                )
                if df is not None and not df.empty:
                    logger.info("CMEMS Chl-a: %d rows from %s", len(df), dataset_id)
                    df["SOURCE"] = "cmems_chlorophyll"
                    return df
            except Exception as exc:
                if attempt < _MAX_RETRIES:
                    time.sleep(2 * attempt)
                logger.debug("CMEMS Chl attempt %d failed (%s): %s",
                             attempt, dataset_id, exc)

    logger.warning("CMEMS chlorophyll unavailable. Returning empty.")
    return pd.DataFrame(columns=["TIME", "LATITUDE", "LONGITUDE", "CHLOR_A", "SOURCE"])


# ---------------------------------------------------------------------------
# copernicusmarine toolbox path (requires free account)
# ---------------------------------------------------------------------------

def _cmems_credentials_available() -> bool:
    return bool(
        os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME") and
        os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")
    )


def _fetch_via_toolbox(
    dataset_id: str, variables: list,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    start_date: str, end_date: str,
    depth: Optional[float],
) -> pd.DataFrame:
    """Fetch using copernicusmarine Python toolbox (requires free registration)."""
    try:
        import copernicusmarine as cm  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "copernicusmarine toolbox not installed. "
            "Run: pip install copernicusmarine\n"
            "Then register free at: https://data.marine.copernicus.eu/"
        ) from exc

    username = os.environ["COPERNICUSMARINE_SERVICE_USERNAME"]
    password = os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"]

    ds = cm.open_dataset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_latitude=lat_min,
        maximum_latitude=lat_max,
        minimum_longitude=lon_min,
        maximum_longitude=lon_max,
        start_datetime=f"{start_date}T00:00:00",
        end_datetime=f"{end_date}T23:59:59",
        minimum_depth=depth or 0.0,
        maximum_depth=depth or 1.0,
        username=username,
        password=password,
    )
    df = ds.to_dataframe().reset_index()
    return _normalise_cmems_df(df)


# ---------------------------------------------------------------------------
# Internal ERDDAP download
# ---------------------------------------------------------------------------

def _download_cmems_nc(
    base_url: str, dataset_id: str, variables: list,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    start_date: str, end_date: str,
    depth: Optional[float],
) -> Optional[pd.DataFrame]:
    import requests, tempfile
    try:
        import netCDF4 as nc4  # type: ignore
    except ImportError as exc:
        raise RuntimeError("netCDF4 not installed") from exc

    depth_selector = f"[({depth or 0.5}):1:({depth or 0.5})]" if depth is not None else ""

    var_str = ",".join(
        f"{v}[({start_date}T00:00:00Z):1:({end_date}T23:59:59Z)]"
        f"{depth_selector}"
        f"[({lat_min:.4f}):1:({lat_max:.4f})]"
        f"[({lon_min:.4f}):1:({lon_max:.4f})]"
        for v in variables
    )
    url = f"{base_url}/{dataset_id}.nc?{var_str}"
    logger.debug("CMEMS URL: %s", url[:200])

    resp = requests.get(url, timeout=_REQUEST_TIMEOUT, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} from CMEMS dataset {dataset_id}")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc", prefix="cmems_")
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)

        if os.path.getsize(tmp_path) < 200:
            raise RuntimeError("CMEMS response too small.")

        ds = nc4.Dataset(tmp_path)
        try:
            rows = _parse_cmems_dataset(ds, variables)
        finally:
            ds.close()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["TIME"] = pd.to_datetime(df["TIME"])
    return df


def _parse_cmems_dataset(ds, variables: list) -> list:
    import netCDF4 as nc4  # type: ignore

    time_var = ds.variables["time"]
    times_nc = nc4.num2date(
        time_var[:], units=time_var.units,
        calendar=getattr(time_var, "calendar", "standard"),
    )
    times_pd = pd.to_datetime([t.isoformat() for t in times_nc]).normalize()

    lats = ds.variables.get("lat") or ds.variables.get("latitude")
    lons = ds.variables.get("lon") or ds.variables.get("longitude")
    if lats is None or lons is None:
        return []
    lats, lons = lats[:], lons[:]

    var_col_map = {
        "thetao": "TEMP", "so": "PSAL", "uo": "U_CURR",
        "vo": "V_CURR", "zos": "SSH", "CHL": "CHLOR_A",
    }

    arrays = {}
    for v in variables:
        if v in ds.variables:
            arr = ds.variables[v][:]
            if arr.ndim == 4:
                arr = arr[:, 0, :, :]
            arrays[v] = arr

    rows = []
    for ti, t in enumerate(times_pd):
        for lati, lat_val in enumerate(lats):
            for loni, lon_val in enumerate(lons):
                row = {"TIME": t, "LATITUDE": float(lat_val), "LONGITUDE": float(lon_val)}
                for vname, col in var_col_map.items():
                    if vname in arrays:
                        cell = arrays[vname][ti, lati, loni]
                        row[col] = float(cell) if not np.ma.is_masked(cell) else np.nan
                rows.append(row)
    return rows


def _normalise_cmems_df(df: pd.DataFrame) -> pd.DataFrame:
    rename = {"thetao": "TEMP", "so": "PSAL", "uo": "U_CURR", "vo": "V_CURR",
              "zos": "SSH", "CHL": "CHLOR_A",
              "latitude": "LATITUDE", "longitude": "LONGITUDE", "time": "TIME"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["SOURCE"] = "cmems_physics"
    return df


def _empty_physics_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "TIME", "LATITUDE", "LONGITUDE", "TEMP", "PSAL",
        "U_CURR", "V_CURR", "SSH", "SOURCE",
    ])


def _clamp_dates(start_date: str, end_date: str) -> Tuple[str, str]:
    """CMEMS reanalysis starts 1993-01-01."""
    cmems_start = pd.Timestamp("1993-01-01")
    yesterday   = pd.Timestamp(datetime.now(timezone.utc).date()) - pd.Timedelta(days=1)
    ts = max(pd.Timestamp(start_date), cmems_start)
    te = min(pd.Timestamp(end_date),   yesterday)
    if ts > te:
        te = yesterday
        ts = te - pd.Timedelta(days=30)
    return ts.strftime("%Y-%m-%d"), te.strftime("%Y-%m-%d")
