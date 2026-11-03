"""
data/argo_loader.py
-------------------
Real Argo float data fetcher via argopy (Argo GDAC, Ifremer ERDDAP).

Large requests are split into 1-month chunks to avoid ERDDAP timeouts.
A single month of data over a moderate region (e.g. Arabian Sea) fetches
in ~10–20 seconds; a full year over a large region can take >2 minutes
and reliably times out.

Required output columns
-----------------------
PLATFORM_NUMBER  : str   – float WMO number
CYCLE_NUMBER     : int   – dive cycle number
LATITUDE         : float
LONGITUDE        : float
TIME             : datetime64[ns]
PRES             : float – pressure / depth (dbar)
TEMP             : float – in-situ temperature (°C)
PSAL             : float – practical salinity (PSU)
TEMP_QC          : int   – Argo QC flag (1=good … 4=bad, 9=missing)
PSAL_QC          : int
DATA_MODE        : str   – 'R' real-time, 'D' delayed-mode
"""

import logging
import warnings
from datetime import datetime, timezone
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Custom exception for "no data in region" — distinct from server/network errors
class ArgoNoDataError(RuntimeError):
    """Raised when the GDAC confirms there are no floats in the requested region/period."""
    pass

# Argo GDAC processing lag — cap queries to avoid empty fetches
_ARGO_MAX_LAG_DAYS = 20

# argopy option key for timeout
_ARGOPY_OPTIONS = {"src": "erddap", "api_timeout": 90}

# Split date range into chunks of this many months to avoid ERDDAP timeouts.
# 1 month is safe for any region; increase if you know the region is sparse.
_MONTHS_PER_CHUNK = 1

# If a region spans more than this many degrees of lon × lat, automatically
# reduce the depth range to surface-only to cut data volume.
_LARGE_REGION_THRESHOLD_DEG2 = 400   # e.g. 20° × 20° = 400


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def fetch_argo_profiles(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    start_date: str,
    end_date: str,
    depth_min: float = 0.0,
    depth_max: float = 2000.0,
) -> pd.DataFrame:
    """
    Fetch Argo profiles from the GDAC for a space-time bounding box.

    Splits the date range into 1-month chunks automatically to prevent
    ERDDAP request timeouts on large queries.

    Returns pd.DataFrame with the standard schema.
    Raises RuntimeError with actionable message on failure.
    """
    try:
        import argopy
        from argopy import DataFetcher as ArgoDataFetcher
    except ImportError as exc:
        raise RuntimeError(
            "argopy is not installed. Run: pip install 'argopy>=1.4' 'erddapy<3'"
        ) from exc

    # Cap end_date to GDAC lag window
    end_date = _cap_end_date(end_date)
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        raise RuntimeError(
            f"The requested period ends too recently for the Argo GDAC archive "
            f"(~{_ARGO_MAX_LAG_DAYS}-day processing lag). "
            f"The latest available date is approximately {end_date}."
        )

    # For large regions, cap depth to surface to dramatically reduce data volume
    region_area = abs(lon_max - lon_min) * abs(lat_max - lat_min)
    if region_area > _LARGE_REGION_THRESHOLD_DEG2 and depth_max > 200:
        logger.info(
            "Large region (%.0f deg²) — capping depth to 0–200 dbar to reduce data volume",
            region_area,
        )
        depth_max = 200.0

    chunks = list(_monthly_chunks(start_date, end_date, _MONTHS_PER_CHUNK))
    logger.info(
        "Fetching Argo in %d monthly chunk(s): lon=[%.1f,%.1f] lat=[%.1f,%.1f] "
        "depth=[%.0f,%.0f]",
        len(chunks), lon_min, lon_max, lat_min, lat_max, depth_min, depth_max,
    )

    frames: List[pd.DataFrame] = []
    last_exc: Optional[Exception] = None

    for chunk_start, chunk_end in chunks:
        logger.info("  Argo chunk [%s .. %s]", chunk_start, chunk_end)
        fetched = False
        for attempt in range(1, 3):   # up to 2 attempts per chunk
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                try:
                    argopy.set_options(**_ARGOPY_OPTIONS)
                    fetcher = ArgoDataFetcher(mode="expert").region(
                        [lon_min, lon_max, lat_min, lat_max,
                         depth_min, depth_max, chunk_start, chunk_end]
                    )
                    ds = fetcher.to_xarray()
                    df_chunk = _xarray_to_dataframe(ds)
                    if not df_chunk.empty:
                        frames.append(df_chunk)
                        logger.info("    → %d rows", len(df_chunk))
                    else:
                        logger.info("    → 0 rows (empty result)")
                    fetched = True
                    last_exc = None   # clear on success
                    break
                except FileNotFoundError as exc:
                    # argopy raises FileNotFoundError for genuinely empty results
                    last_exc = exc
                    logger.info(
                        "    No Argo profiles in chunk [%s..%s] (empty region)",
                        chunk_start, chunk_end,
                    )
                    fetched = True   # "fetched" but empty — don't retry
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < 2:
                        logger.warning(
                            "    Argo chunk [%s..%s] attempt %d failed: %s — retrying",
                            chunk_start, chunk_end, attempt, str(exc)[:80],
                        )
                        import time as _time
                        _time.sleep(3)
                    else:
                        logger.warning(
                            "    Argo chunk [%s..%s] failed after %d attempts: %s",
                            chunk_start, chunk_end, attempt, str(exc)[:120],
                        )

    if not frames:
        # Distinguish between "genuinely no data" and "server error"
        is_empty = (last_exc is None or isinstance(last_exc, FileNotFoundError))
        if is_empty:
            raise ArgoNoDataError(
                f"No Argo floats were active in this region "
                f"(lon {lon_min:.0f}°–{lon_max:.0f}°, lat {lat_min:.0f}°–{lat_max:.0f}°) "
                f"during {start_date} to {end_date}.\n"
                "This is a real data gap — the Argo network does not cover all ocean areas equally.\n"
                "Semi-enclosed seas (Red Sea, Persian Gulf, Mediterranean) and polar regions "
                "have very sparse or no Argo coverage."
            )
        else:
            hint = (
                "Could not retrieve Argo profile data: "
                "No Argo data found for this region and time period. "
                "Possible reasons:\n"
                "  • No floats were active in this region during the selected dates.\n"
                "  • The Argo GDAC ERDDAP is temporarily overloaded — try again in a minute.\n"
                "  • Try a shorter date range (e.g. one month instead of a full year).\n"
            )
            if last_exc:
                hint += f"Last error: {str(last_exc)[:200]}"
            raise RuntimeError(hint)

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(
        subset=["PLATFORM_NUMBER", "CYCLE_NUMBER", "PRES"]
    ).reset_index(drop=True)
    logger.info("Argo fetch complete: %d total rows from %d chunk(s)", len(result), len(chunks))
    return result


def fetch_argo_float(
    platform_number: str,
    cycle_numbers: Optional[list] = None,
) -> pd.DataFrame:
    """Fetch profiles for a specific float by WMO number."""
    try:
        import argopy
        from argopy import DataFetcher as ArgoDataFetcher
    except ImportError as exc:
        raise RuntimeError(
            "argopy is not installed. Run: pip install 'argopy>=1.4' 'erddapy<3'"
        ) from exc

    logger.info("Fetching float %s cycles=%s", platform_number, cycle_numbers)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            argopy.set_options(**_ARGOPY_OPTIONS)
            if cycle_numbers:
                fetcher = ArgoDataFetcher(mode="expert").profile(
                    int(platform_number), cycle_numbers
                )
            else:
                fetcher = ArgoDataFetcher(mode="expert").float(int(platform_number))
            ds = fetcher.to_xarray()
        except Exception as exc:
            raise RuntimeError(
                f"Argo GDAC fetch failed for float {platform_number}: {exc}"
            ) from exc
    return _xarray_to_dataframe(ds)


# ---------------------------------------------------------------------------
# Synthetic data generator — identical schema, for offline testing
# ---------------------------------------------------------------------------

def make_synthetic_profiles(
    n_floats: int = 3,
    cycles_per_float: int = 24,   # 24 × 10 days = ~240 days from Jan 2023 → Aug 2023
    lat_center: float = 20.0,
    lon_center: float = -60.0,
    bad_fraction: float = 0.12,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic Argo profiles (sequential ~10-day cycles).
    Default covers Jan–Aug 2023 so the form's default date range works.
    bad_fraction controls what share get a bad QC flag.
    """
    rng = np.random.default_rng(seed)
    rows = []
    base_time = pd.Timestamp("2023-01-01")

    for float_idx in range(n_floats):
        platform = str(1901000 + float_idx)
        lat = lat_center + rng.uniform(-2, 2)
        lon = lon_center + rng.uniform(-2, 2)

        for cycle in range(1, cycles_per_float + 1):
            cycle_time = base_time + pd.Timedelta(days=(float_idx * 3 + cycle * 10))
            depths = np.arange(5, 2005, 50, dtype=float)
            for pres in depths:
                temp = 25.0 - 0.01 * pres + rng.normal(0, 0.3)
                psal = 35.0 + 0.002 * pres + rng.normal(0, 0.05)
                is_bad = rng.random() < bad_fraction
                qc   = int(rng.choice([3, 4])) if is_bad else int(rng.choice([1, 2], p=[0.7, 0.3]))
                mode = rng.choice(["R", "D"], p=[0.4, 0.6])
                rows.append({
                    "PLATFORM_NUMBER": platform, "CYCLE_NUMBER": cycle,
                    "LATITUDE": lat + rng.normal(0, 0.05),
                    "LONGITUDE": lon + rng.normal(0, 0.05),
                    "TIME": cycle_time, "PRES": pres,
                    "TEMP": round(temp, 4), "PSAL": round(psal, 4),
                    "TEMP_QC": qc, "PSAL_QC": qc, "DATA_MODE": mode,
                })

    df = pd.DataFrame(rows)
    df["TIME"] = pd.to_datetime(df["TIME"])
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _monthly_chunks(
    start_date: str, end_date: str, months: int = 1
) -> Iterator[Tuple[str, str]]:
    """Yield (chunk_start, chunk_end) ISO strings in windows of `months` months."""
    ts = pd.Timestamp(start_date)
    te = pd.Timestamp(end_date)
    cur = ts
    while cur <= te:
        chunk_end = min(
            cur + pd.DateOffset(months=months) - pd.Timedelta(days=1), te
        )
        yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cur = chunk_end + pd.Timedelta(days=1)


def _cap_end_date(end_date: str) -> str:
    """Clamp end_date to (today – GDAC lag)."""
    today = datetime.now(timezone.utc).date()
    cap = pd.Timestamp(today) - pd.Timedelta(days=_ARGO_MAX_LAG_DAYS)
    if pd.Timestamp(end_date) > cap:
        capped = cap.strftime("%Y-%m-%d")
        logger.info("end_date capped from %s to %s (GDAC lag)", end_date, capped)
        return capped
    return end_date


_COL_ALIASES = {
    "PLATFORM_NUMBER": "PLATFORM_NUMBER", "CYCLE_NUMBER": "CYCLE_NUMBER",
    "LATITUDE": "LATITUDE", "LONGITUDE": "LONGITUDE", "TIME": "TIME",
    "PRES": "PRES", "TEMP": "TEMP", "PSAL": "PSAL",
    "TEMP_QC": "TEMP_QC", "PSAL_QC": "PSAL_QC", "DATA_MODE": "DATA_MODE",
    "platform_number": "PLATFORM_NUMBER", "cycle_number": "CYCLE_NUMBER",
    "latitude": "LATITUDE", "longitude": "LONGITUDE", "time": "TIME",
    "pres": "PRES", "temp": "TEMP", "psal": "PSAL",
    "temp_qc": "TEMP_QC", "psal_qc": "PSAL_QC", "data_mode": "DATA_MODE",
}

_REQUIRED_DEFAULTS = {
    "PLATFORM_NUMBER": "", "CYCLE_NUMBER": 0,
    "LATITUDE": np.nan, "LONGITUDE": np.nan, "TIME": pd.NaT,
    "PRES": np.nan, "TEMP": np.nan, "PSAL": np.nan,
    "TEMP_QC": 9, "PSAL_QC": 9, "DATA_MODE": "R",
}


def _xarray_to_dataframe(ds) -> pd.DataFrame:
    """Convert an argopy xarray Dataset to the standard FloatAi DataFrame."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        df = ds.to_dataframe().reset_index()

    rename = {c: _COL_ALIASES[c] for c in df.columns if c in _COL_ALIASES}
    df = df.rename(columns=rename)
    df.columns = [_COL_ALIASES.get(c, c.upper()) for c in df.columns]

    for col, default in _REQUIRED_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    df["PLATFORM_NUMBER"] = df["PLATFORM_NUMBER"].astype(str).str.strip()
    df["CYCLE_NUMBER"] = pd.to_numeric(df["CYCLE_NUMBER"], errors="coerce").fillna(0).astype(int)
    df["TEMP_QC"] = pd.to_numeric(df["TEMP_QC"], errors="coerce").fillna(9).astype(int)
    df["PSAL_QC"] = pd.to_numeric(df["PSAL_QC"], errors="coerce").fillna(9).astype(int)
    df["TIME"] = pd.to_datetime(df["TIME"], errors="coerce", utc=True).dt.tz_localize(None)
    df["DATA_MODE"] = df["DATA_MODE"].astype(str).str.strip().str.upper()
    df["DATA_MODE"] = df["DATA_MODE"].where(df["DATA_MODE"].isin(["R", "D", "A"]), "R")

    return df[list(_REQUIRED_DEFAULTS.keys())].copy()
