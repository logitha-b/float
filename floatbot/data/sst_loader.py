"""
data/sst_loader.py
------------------
Real daily SST data fetcher — NOAA OISST v2.1 via ERDDAP (CoastWatch/OceanWatch).

Design decisions (learned from debugging):
  - Chunks are 1 MONTH maximum. The CoastWatch ERDDAP times out on
    requests spanning >~3 months for a 2°×2° box. Monthly chunks are
    small enough to complete in under 30 s each.
  - Downloads use requests (plain HTTPS) + temp-file netCDF4 parsing.
    pydap appends ".dds" and produces a 404; netCDF4 memory= fails on
    Windows. Temp-file approach works on all platforms.
  - Retry with exponential back-off: each chunk is attempted up to
    MAX_RETRIES times before trying the next server/product.
  - Two ERDDAP servers (CoastWatch, OceanWatch) × two products
    (final, NRT) = 4 attempts per chunk before raising.

Output schema (identical to synthetic loader):
    date : datetime64[ns]   (daily, UTC-normalised)
    sst  : float            (°C, spatially averaged when using timeseries API)
    lat  : float            (only in the full fetch_sst() output)
    lon  : float            (only in the full fetch_sst() output)

Citation: Huang, B., et al. (2020). NOAA 0.25-degree Daily Optimum
Interpolation SST (OISST), Version 2.1. NOAA NCEI.
https://doi.org/10.25921/RE9P-PT57
"""

import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ERDDAP server list — tried in order per chunk
# ---------------------------------------------------------------------------
_ERDDAP_SERVERS: List[Tuple[str, str, str]] = [
    # (base_url,  final_dataset_id,   nrt_dataset_id)
    # CoastWatch is the most reliable — confirmed working
    (
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap",
        "ncdcOisst21Agg",
        "ncdcOisst21NrtAgg",
    ),
    # OceanWatch as secondary fallback
    (
        "https://oceanwatch.pfeg.noaa.gov/erddap/griddap",
        "ncdcOisst21Agg",
        "ncdcOisst21NrtAgg",
    ),
]

GRID_RES    = 0.25   # OISST grid spacing in degrees
OISST_START = "1981-09-01"

# Keep every individual ERDDAP request small to avoid timeouts.
# 1 month of daily SST over a 2°×2° box = ~30 time steps × ~64 cells ≈ fast.
_MONTHS_PER_CHUNK = 1

# HTTP settings
_REQUEST_TIMEOUT = 15    # seconds — fail fast so dead servers don't block for minutes
_MAX_RETRIES     = 2     # attempts per (server, product) — 2 × 15s = 30s max per chunk
_RETRY_BACKOFF   = 2.0   # seconds between retries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_sst(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch daily SST for a lat/lon bounding box over a date range.

    Parameters
    ----------
    lat_min, lat_max : float  Latitude bounds (–90 to 90).
    lon_min, lon_max : float  Longitude bounds (–180 to 180).
    start_date       : str    ISO date, e.g. '2023-01-01'.
    end_date         : str    ISO date, e.g. '2023-12-31'.

    Returns
    -------
    pd.DataFrame with columns: date, lat, lon, sst
    """
    start_date, end_date = _clamp_dates(start_date, end_date)

    # Snap to OISST 0.25° grid; ensure at least one grid cell per dimension
    lat_min_g = _snap(lat_min, GRID_RES)
    lat_max_g = _snap(lat_max, GRID_RES)
    if lat_min_g >= lat_max_g:
        lat_max_g = lat_min_g + GRID_RES

    # OISST uses 0–360 longitude convention
    lon_min_g = _snap(_to_360(lon_min), GRID_RES)
    lon_max_g = _snap(_to_360(lon_max), GRID_RES)
    if lon_min_g >= lon_max_g:
        lon_max_g = lon_min_g + GRID_RES

    frames: List[pd.DataFrame] = []
    chunks = list(_monthly_chunks(start_date, end_date, _MONTHS_PER_CHUNK))
    failed_chunks = 0
    logger.info(
        "SST fetch: %d monthly chunks, lat=[%.2f,%.2f] lon360=[%.2f,%.2f]",
        len(chunks), lat_min_g, lat_max_g, lon_min_g, lon_max_g,
    )

    for i, (chunk_start, chunk_end) in enumerate(chunks):
        try:
            df_chunk = _fetch_chunk_with_fallback(
                lat_min_g, lat_max_g, lon_min_g, lon_max_g, chunk_start, chunk_end
            )
            if df_chunk is not None and not df_chunk.empty:
                frames.append(df_chunk)
        except Exception as exc:
            failed_chunks += 1
            logger.warning(
                "SST chunk [%s..%s] skipped (all servers failed): %s",
                chunk_start, chunk_end, str(exc)[:100],
            )
            # Continue — skip this month and use whatever we have
        if (i + 1) % 12 == 0:
            logger.info("  SST progress: %d/%d chunks done (%d failed)",
                        i + 1, len(chunks), failed_chunks)

    if not frames:
        raise RuntimeError(
            f"SST fetch returned no data for region "
            f"lat=[{lat_min:.2f},{lat_max:.2f}] lon=[{lon_min:.2f},{lon_max:.2f}] "
            f"dates=[{start_date},{end_date}].\n"
            "All monthly chunks failed. Check your internet connection."
        )

    if failed_chunks > 0:
        logger.info(
            "SST fetch: %d/%d chunks succeeded, %d skipped (used %d days of data)",
            len(chunks) - failed_chunks, len(chunks), failed_chunks,
            sum(len(f) for f in frames),
        )

    df = pd.concat(frames, ignore_index=True)
    df = (df.sort_values("date")
            .drop_duplicates(subset=["date", "lat", "lon"])
            .reset_index(drop=True))
    return df


def fetch_sst_timeseries(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    radius_deg: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch daily-mean SST for a point ± radius_deg, spatially averaged.

    Returns pd.DataFrame with columns: date, sst
    """
    df = fetch_sst(
        lat_min=lat - radius_deg,
        lat_max=lat + radius_deg,
        lon_min=lon - radius_deg,
        lon_max=lon + radius_deg,
        start_date=start_date,
        end_date=end_date,
    )
    ts = df.groupby("date")["sst"].mean().reset_index()
    return ts.sort_values("date").reset_index(drop=True)


def make_synthetic_sst(
    lat_center: float = 20.0,
    lon_center: float = -60.0,
    start_date: str = "2010-01-01",
    end_date: str = "2023-12-31",
    heatwave_year: int = 2023,
    heatwave_month: int = 7,
    heatwave_duration_days: int = 18,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic daily SST time series (same schema as fetch_sst_timeseries).
    Injects a synthetic marine heatwave for offline testing.
    Returns pd.DataFrame with columns: date, sst
    """
    rng = np.random.default_rng(seed)
    dates    = pd.date_range(start=start_date, end=end_date, freq="D")
    doy      = np.asarray(dates.day_of_year, dtype=float)
    seasonal = 26.0 + 4.0 * np.sin(2 * np.pi * (doy - 80) / 365)
    noise    = rng.normal(0, 0.3, len(dates))
    sst      = np.array(seasonal + noise, dtype=float)

    hw_start = pd.Timestamp(f"{heatwave_year}-{heatwave_month:02d}-01")
    hw_end   = hw_start + pd.Timedelta(days=heatwave_duration_days)
    mask     = np.asarray((dates >= hw_start) & (dates < hw_end))
    sst[mask] += 3.5

    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "sst":  np.round(sst, 3),
    })


# ---------------------------------------------------------------------------
# Internal: fallback chain over servers × products
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _fetch_chunk_with_fallback(
    lat_min: float, lat_max: float,
    lon_min_360: float, lon_max_360: float,
    start_date: str, end_date: str,
) -> Optional[pd.DataFrame]:
    """
    Try CoastWatch-final → CoastWatch-NRT → OceanWatch-final → OceanWatch-NRT.
    Each (server, product) pair is retried up to _MAX_RETRIES times with
    exponential back-off before moving to the next pair.
    """
    last_exc: Optional[Exception] = None

    # NRT dataset only covers approximately the last 60 days.
    # Skip it entirely for historical chunks to avoid 404 errors.
    chunk_end_ts = pd.Timestamp(end_date)
    nrt_cutoff   = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=60)
    use_nrt      = chunk_end_ts >= nrt_cutoff

    for base_url, final_id, nrt_id in _ERDDAP_SERVERS:
        datasets_to_try = (final_id, nrt_id) if use_nrt else (final_id,)
        for dataset_id in datasets_to_try:
            delay = _RETRY_BACKOFF
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    df = _download_nc(
                        base_url, dataset_id,
                        lat_min, lat_max, lon_min_360, lon_max_360,
                        start_date, end_date,
                    )
                    server_host = base_url.split("//")[1].split("/")[0]
                    logger.debug(
                        "SST [%s..%s] %d rows from %s / %s",
                        start_date, end_date, len(df), server_host, dataset_id,
                    )
                    return df
                except Exception as exc:
                    last_exc = exc
                    if attempt < _MAX_RETRIES:
                        logger.debug(
                            "SST attempt %d/%d failed (%s/%s): %s — retrying in %.1fs",
                            attempt, _MAX_RETRIES, base_url, dataset_id, exc, delay,
                        )
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.debug(
                            "SST giving up on %s/%s after %d attempts: %s",
                            base_url, dataset_id, _MAX_RETRIES, exc,
                        )

    # All options exhausted — raise with a clear message
    raise RuntimeError(
        f"All SST servers/products failed for chunk {start_date}–{end_date}.\n"
        f"Last error: {last_exc}\n"
        "On Windows, install pip-system-certs if you see SSL certificate errors."
    )


# ---------------------------------------------------------------------------
# Internal: download and parse a single .nc chunk
# ---------------------------------------------------------------------------

def _download_nc(
    base_url: str,
    dataset_id: str,
    lat_min: float,
    lat_max: float,
    lon_min_360: float,
    lon_max_360: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Build the ERDDAP griddap .nc URL, stream-download it, write to a temp
    file, then parse with netCDF4.

    Using a temp file (rather than memory=) is required on Windows because
    netCDF4's in-memory dataset API is not supported on all Windows builds.
    Streaming avoids buffering the entire response before writing.
    """
    try:
        import requests as _req
    except ImportError as exc:
        raise RuntimeError("requests not installed: pip install requests") from exc
    try:
        import netCDF4 as nc4  # type: ignore
    except ImportError as exc:
        raise RuntimeError("netCDF4 not installed: pip install netCDF4") from exc

    url = (
        f"{base_url}/{dataset_id}.nc"
        f"?sst[({start_date}T00:00:00Z):1:({end_date}T00:00:00Z)]"
        f"[(0.0):1:(0.0)]"
        f"[({lat_min:.4f}):1:({lat_max:.4f})]"
        f"[({lon_min_360:.4f}):1:({lon_max_360:.4f})]"
    )
    logger.debug("OISST URL: %s", url)

    # Stream download — avoids loading the whole response into RAM at once
    resp = _req.get(url, timeout=_REQUEST_TIMEOUT, stream=True)
    if resp.status_code != 200:
        # Read a bit of the body for diagnostics, then raise
        snippet = resp.text[:300].strip()
        raise RuntimeError(
            f"HTTP {resp.status_code} from {dataset_id} "
            f"({start_date}→{end_date}): {snippet}"
        )

    # Write streamed content to a named temp file
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc", prefix="oisst_")
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)

        file_size = os.path.getsize(tmp_path)
        if file_size < 200:
            with open(tmp_path, "r", errors="replace") as fh:
                content = fh.read(300)
            raise RuntimeError(
                f"ERDDAP returned only {file_size} bytes "
                f"(likely an error page): {content}"
            )

        ds = nc4.Dataset(tmp_path)
        try:
            rows = _parse_nc_dataset(ds)
        finally:
            ds.close()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not rows:
        return pd.DataFrame(columns=["date", "lat", "lon", "sst"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _parse_nc_dataset(ds) -> list:
    """
    Extract (date, lat, lon, sst) rows from an open netCDF4 Dataset.
    Handles both 3D (time, lat, lon) and 4D (time, zlev, lat, lon) arrays.
    """
    import netCDF4 as nc4  # type: ignore

    time_var = ds.variables["time"]
    times_nc = nc4.num2date(
        time_var[:],
        units=time_var.units,
        calendar=getattr(time_var, "calendar", "standard"),
    )
    times_pd = pd.to_datetime([t.isoformat() for t in times_nc]).normalize()

    lats    = ds.variables["latitude"][:] if "latitude" in ds.variables else ds.variables["lat"][:]
    lons    = ds.variables["longitude"][:] if "longitude" in ds.variables else ds.variables["lon"][:]
    sst_raw = ds.variables["sst"][:]

    # Drop the zlev (altitude) dimension if present
    if sst_raw.ndim == 4:
        sst_raw = sst_raw[:, 0, :, :]

    rows = []
    for ti, t in enumerate(times_pd):
        for lati, lat_val in enumerate(lats):
            for loni, lon_val in enumerate(lons):
                cell = sst_raw[ti, lati, loni]
                if np.ma.is_masked(cell):
                    continue
                val = float(cell)
                if np.isnan(val):
                    continue
                # Convert 0–360 back to –180/180
                lon180 = float(lon_val) - 360 if float(lon_val) > 180 else float(lon_val)
                rows.append({
                    "date": t,
                    "lat":  float(lat_val),
                    "lon":  lon180,
                    "sst":  round(val, 4),
                })
    return rows


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _clamp_dates(start_date: str, end_date: str) -> Tuple[str, str]:
    """Clamp start/end to the OISST archive window: 1981-09-01 … yesterday."""
    archive_start = pd.Timestamp(OISST_START)
    yesterday     = pd.Timestamp(datetime.now(timezone.utc).date()) - pd.Timedelta(days=1)

    ts = max(pd.Timestamp(start_date), archive_start)
    te = min(pd.Timestamp(end_date),   yesterday)

    if ts > te:
        logger.warning(
            "SST date range [%s, %s] outside archive bounds; defaulting to last 30 days.",
            start_date, end_date,
        )
        te = yesterday
        ts = te - pd.Timedelta(days=30)

    return ts.strftime("%Y-%m-%d"), te.strftime("%Y-%m-%d")


def _monthly_chunks(
    start_date: str, end_date: str, months_per_chunk: int = 1
) -> Iterator[Tuple[str, str]]:
    """
    Split a date range into windows of at most months_per_chunk months.
    Yields (chunk_start_str, chunk_end_str) pairs as ISO dates.
    """
    ts  = pd.Timestamp(start_date)
    te  = pd.Timestamp(end_date)
    cur = ts
    while cur <= te:
        chunk_end = min(
            cur + pd.DateOffset(months=months_per_chunk) - pd.Timedelta(days=1),
            te,
        )
        yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cur = chunk_end + pd.Timedelta(days=1)


def _to_360(lon: float) -> float:
    """Convert –180/180 longitude to 0/360 (OISST convention)."""
    return lon % 360


def _snap(value: float, grid: float) -> float:
    """Snap a coordinate to the nearest OISST grid point."""
    return round(round(value / grid) * grid, 4)
