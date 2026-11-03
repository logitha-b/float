"""
data/gebco_loader.py
--------------------
GEBCO (General Bathymetric Chart of the Oceans) bathymetry loader.

Source: GEBCO REST API (no account required) and NOAA CoastWatch ERDDAP.
  REST API  : https://www.gebco.net/data_and_products/gebco_web_services/web_map_service/
  ERDDAP    : https://coastwatch.pfeg.noaa.gov/erddap/griddap/GEBCO_2022

GEBCO provides the world's most comprehensive ocean depth (bathymetry) data
at 15-arc-second resolution (~450 m). Useful for:
  • Answering "how deep is the ocean at [location]?"
  • Contextualising float profile depths
  • Identifying seamounts, trenches, ridges

Output schema
-------------
fetch_gebco_depth() → pd.DataFrame with columns:
    LATITUDE   : float
    LONGITUDE  : float
    DEPTH_M    : float  – depth in metres (positive = below sea surface)
    SOURCE     : str    – 'gebco_2022'

fetch_gebco_point_depth() → float: depth in metres at nearest grid point
"""

import logging
import os
import tempfile
import time
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_COASTWATCH_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
# etopo180 = ETOPO1 1 arc-minute relief model — confirmed working on CoastWatch
# GEBCO_2022 returns 404 on CoastWatch as of 2026
_GEBCO_DATASET   = "etopo180"        # 1 arc-minute (~1.8 km) global relief
_GEBCO_ALT       = "etopo360"        # fallback
_REQUEST_TIMEOUT = 20
_MAX_RETRIES     = 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=128)
def fetch_gebco_depth(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> pd.DataFrame:
    """
    Fetch bathymetry (ocean depth) for a bounding box.

    Parameters
    ----------
    lat_min, lat_max : float  Latitude bounds.
    lon_min, lon_max : float  Longitude bounds (–180 to 180).

    Returns
    -------
    pd.DataFrame with columns: LATITUDE, LONGITUDE, DEPTH_M, SOURCE
    Positive DEPTH_M = below sea surface (depths are negated from elevation).
    """
    datasets = [(_COASTWATCH_BASE, _GEBCO_DATASET, "altitude"),
                (_COASTWATCH_BASE, _GEBCO_ALT,     "altitude")]

    last_exc: Optional[Exception] = None
    for base_url, dataset_id, var_name in datasets:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                df = _download_gebco_nc(
                    base_url, dataset_id, var_name,
                    lat_min, lat_max, lon_min, lon_max,
                )
                logger.info(
                    "GEBCO: %d grid cells from %s", len(df), dataset_id
                )
                return df
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(2 * attempt)
                logger.debug("GEBCO attempt %d failed (%s): %s", attempt, dataset_id, exc)

    raise RuntimeError(
        f"GEBCO bathymetry fetch failed.\nLast error: {last_exc}\n"
        "Check internet connection."
    )


def fetch_gebco_point_depth(lat: float, lon: float) -> Optional[float]:
    """
    Return the ocean depth (m) at the nearest GEBCO grid point to (lat, lon).

    Returns None if the location appears to be on land (elevation > 0).
    Returns a positive float for ocean depth (e.g., 3840.0 for 3840 m deep).
    """
    try:
        # Request a tiny 0.5°×0.5° box around the point
        df = fetch_gebco_depth(
            lat_min=lat - 0.25, lat_max=lat + 0.25,
            lon_min=lon - 0.25, lon_max=lon + 0.25,
        )
        if df.empty:
            return None
        # Find nearest point
        df["dist"] = np.sqrt((df["LATITUDE"] - lat)**2 + (df["LONGITUDE"] - lon)**2)
        nearest = df.loc[df["dist"].idxmin()]
        depth = float(nearest["DEPTH_M"])
        return depth if depth > 0 else None  # None = land
    except Exception as exc:
        logger.warning("GEBCO point depth failed at (%.2f, %.2f): %s", lat, lon, exc)
        return None


def describe_bathymetry(lat: float, lon: float) -> str:
    """
    Return a plain-language description of the ocean depth at a point.
    """
    depth = fetch_gebco_point_depth(lat, lon)
    if depth is None:
        return (
            f"The location ({lat:.2f}°, {lon:.2f}°) appears to be on land or "
            "the bathymetry data is unavailable for this point."
        )
    if depth < 200:
        zone = "continental shelf (shallow coastal water)"
    elif depth < 1000:
        zone = "upper continental slope"
    elif depth < 3500:
        zone = "continental rise / abyssal transition"
    elif depth < 6000:
        zone = "abyssal plain (deep open ocean)"
    else:
        zone = "hadal zone (ocean trench)"

    return (
        f"The ocean depth at ({lat:.2f}°, {lon:.2f}°) is approximately "
        f"**{depth:.0f} m** ({zone}), based on GEBCO 2022 bathymetry."
    )


# ---------------------------------------------------------------------------
# Internal download
# ---------------------------------------------------------------------------

def _download_gebco_nc(
    base_url: str, dataset_id: str, var_name: str,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
) -> pd.DataFrame:
    import requests
    try:
        import netCDF4 as nc4  # type: ignore
    except ImportError as exc:
        raise RuntimeError("netCDF4 not installed: pip install netCDF4") from exc

    url = (
        f"{base_url}/{dataset_id}.nc"
        f"?{var_name}[({lat_min:.4f}):1:({lat_max:.4f})]"
        f"[({lon_min:.4f}):1:({lon_max:.4f})]"
    )
    logger.debug("GEBCO URL: %s", url)

    resp = requests.get(url, timeout=_REQUEST_TIMEOUT, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} from {dataset_id}")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc", prefix="gebco_")
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)

        if os.path.getsize(tmp_path) < 100:
            raise RuntimeError("GEBCO response too small.")

        ds = nc4.Dataset(tmp_path)
        try:
            # etopo180 uses 'latitude'/'longitude'; older datasets use 'lat'/'lon'
            lat_key = "latitude" if "latitude" in ds.variables else "lat"
            lon_key = "longitude" if "longitude" in ds.variables else "lon"
            lats = ds.variables[lat_key][:]
            lons = ds.variables[lon_key][:]
            elev = ds.variables[var_name][:]  # shape: (lat, lon)

            rows = []
            for lati, lat_val in enumerate(lats):
                for loni, lon_val in enumerate(lons):
                    cell = elev[lati, loni]
                    if np.ma.is_masked(cell):
                        continue
                    elev_m = float(cell)
                    # Convert elevation to depth: negative elevation = below sea level
                    depth_m = -elev_m if elev_m < 0 else 0.0
                    rows.append({
                        "LATITUDE":  float(lat_val),
                        "LONGITUDE": float(lon_val),
                        "DEPTH_M":   round(depth_m, 1),
                        "SOURCE":    dataset_id.lower(),
                    })
        finally:
            ds.close()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["LATITUDE", "LONGITUDE", "DEPTH_M", "SOURCE"]
    )
