"""
ocean_stress_engine/climatology.py
------------------------------------
Climatological baseline fetcher for the Ocean Stress Engine.

Three data sources (never hard-codes values):
  Tier 1 — WOA23 (temperature + salinity mean)
    NOAA AOML CWCGOM ERDDAP, 0.25-degree monthly grid.
    Citation: Garcia et al. (2024), NOAA Atlas NESDIS 89/90.
              doi:10.25923/bg7n-vp49 (temperature)
              doi:10.25923/70qt-9574 (salinity)

  Tier 2 — Historical Argo fallback (temperature + salinity)
    Same region, same calendar month, 2005-2020 via fetch_argo_profiles().

  SST — NOAA OISST v2.1 historical baseline
    Fetches the same calendar month over the baseline years using the
    existing fetch_sst() from data/sst_loader.py.  Used as the primary
    skin-SST variable that drives coral bleaching detection.
    Citation: Huang et al. (2020), doi:10.25921/RE9P-PT57

Caching
    All results cached in ose_cache/ with ose_ prefix, TTL 30 days.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import date
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class BaselineStats:
    """Climatological statistics for one region/month/depth."""
    temperature_mean:  Optional[float]   # °C  (Argo subsurface, 0-50m)
    temperature_std:   Optional[float]   # °C
    salinity_mean:     Optional[float]   # PSU
    salinity_std:      Optional[float]   # PSU
    sst_mean:          Optional[float]   # °C  (OISST skin surface)
    sst_std:           Optional[float]   # °C
    source:            str               # "woa23" | "argo_historical" | "none"
    calendar_month:    int
    n_profiles:        int
    n_readings:        int
    caveat:            Optional[str]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_baseline(
    region,
    calendar_month: int,
    depth_min: float,
    depth_max: float,
    baseline_start: date,
    baseline_end: date,
    use_synthetic: bool = False,
) -> BaselineStats:
    """
    Return climatological mean + std for temperature, salinity, and SST.

    Always tries to populate sst_mean/sst_std from OISST regardless of
    whether WOA23 or Argo historical is used for temp/sal.
    Results are cached on disk with ose_ prefix.
    """
    cfg = _load_config()

    if use_synthetic:
        return _synthetic_baseline(calendar_month)

    cache_dir = _cache_dir(cfg)
    cache_key = (
        f"ose_clim_{region.lat_min}_{region.lat_max}_"
        f"{region.lon_min}_{region.lon_max}_{depth_min}_{depth_max}_"
        f"m{calendar_month:02d}_{baseline_start.year}-{baseline_end.year}"
    )
    ttl_days = cfg.get("climatology", {}).get("cache", {}).get("ttl_days", 30)

    cached = _cache_load(cache_dir, cache_key, ttl_days)
    if cached is not None:
        logger.info("OSE climatology cache hit: %s", cache_key)
        return BaselineStats(**cached)

    # ── Tier 1: WOA23 (temp + sal) ────────────────────────────────────────
    baseline = None
    try:
        baseline = _fetch_woa23(region, calendar_month, depth_min, depth_max, cfg)
    except Exception as exc:
        logger.warning("WOA23 failed (%s) — falling back to Argo historical.", exc)

    # ── Tier 2: Argo historical fallback ──────────────────────────────────
    if baseline is None:
        try:
            baseline = _fetch_argo_historical(
                region, calendar_month, depth_min, depth_max,
                baseline_start, baseline_end, cfg,
            )
        except Exception as exc2:
            logger.error("Argo historical baseline failed: %s", exc2)
            return BaselineStats(
                temperature_mean=None, temperature_std=None,
                salinity_mean=None,    salinity_std=None,
                sst_mean=None,         sst_std=None,
                source="none", calendar_month=calendar_month,
                n_profiles=0, n_readings=0,
                caveat="Both WOA23 and Argo historical baseline fetches failed.",
            )

    # ── OISST skin-SST historical baseline (always attempted) ────────────
    sst_mean, sst_std = None, None
    try:
        sst_mean, sst_std = _fetch_oisst_historical_baseline(
            region, calendar_month, baseline_start, baseline_end
        )
        logger.info(
            "OISST historical baseline month=%d: SST=%.2f±%.2f°C",
            calendar_month, sst_mean or 0, sst_std or 0,
        )
    except Exception as exc:
        logger.warning("OISST historical SST baseline failed (non-fatal): %s", exc)

    baseline.sst_mean = sst_mean
    baseline.sst_std  = sst_std

    _cache_save(cache_dir, cache_key, asdict(baseline))
    return baseline


# ---------------------------------------------------------------------------
# Tier 1: WOA23 ERDDAP
# ---------------------------------------------------------------------------

def _fetch_woa23(
    region, calendar_month: int,
    depth_min: float, depth_max: float,
    cfg: dict,
) -> BaselineStats:
    """
    Fetch WOA23 monthly climatology (temperature + salinity analyzed mean).

    Server:   https://cwcgom.aoml.noaa.gov/erddap/griddap
    Datasets: WOA_TEMP_d48d_916b_3478 (t_an), WOA_SAL_873e_726a_1b4d (s_an)
    Time:     12 monthly steps as epoch seconds; index 0=Jan, 6=Jul, etc.
    Std:      estimated from spatial variability across grid cells.
    """
    import requests
    import tempfile
    import netCDF4 as nc4  # type: ignore

    woa_cfg  = cfg.get("climatology", {}).get("woa23", {})
    base_url = woa_cfg.get("base_url", "https://cwcgom.aoml.noaa.gov/erddap/griddap")
    timeout  = int(woa_cfg.get("timeout_s", 25))
    temp_id  = woa_cfg.get("temperature_dataset", "WOA_TEMP_d48d_916b_3478")
    sal_id   = woa_cfg.get("salinity_dataset",    "WOA_SAL_873e_726a_1b4d")
    temp_var = woa_cfg.get("temperature_var", "t_an")
    sal_var  = woa_cfg.get("salinity_var",    "s_an")

    lat_min, lat_max = region.lat_min, region.lat_max
    lon_min, lon_max = region.lon_min, region.lon_max

    # Get time axis once to find epoch value for this month
    time_url = f"{base_url}/{temp_id}.nc?time"
    resp_t   = requests.get(time_url, timeout=timeout)
    if resp_t.status_code != 200:
        raise RuntimeError(f"WOA23 time axis: HTTP {resp_t.status_code}")

    tmp_t = tempfile.mktemp(suffix=".nc")
    try:
        with open(tmp_t, "wb") as fh:
            fh.write(resp_t.content)
        ds_t = nc4.Dataset(tmp_t)
        time_vals = list(ds_t.variables["time"][:])
        ds_t.close()
    finally:
        try: os.unlink(tmp_t)
        except OSError: pass

    month_idx = calendar_month - 1
    if month_idx >= len(time_vals):
        raise RuntimeError(f"WOA23: month index {month_idx} out of range ({len(time_vals)} steps).")
    epoch_val = float(time_vals[month_idx])

    t_mean = t_std = s_mean = s_std = None
    for dataset_id, var_name, which in [
        (temp_id, temp_var, "temp"),
        (sal_id,  sal_var,  "sal"),
    ]:
        url = (
            f"{base_url}/{dataset_id}.nc"
            f"?{var_name}[({epoch_val:.1f}):1:({epoch_val:.1f})]"
            f"[({depth_min:.2f}):1:({depth_max:.2f})]"
            f"[({lat_min:.2f}):1:({lat_max:.2f})]"
            f"[({lon_min:.2f}):1:({lon_max:.2f})]"
        )
        logger.debug("WOA23 URL: %s", url)
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"WOA23 HTTP {resp.status_code} for {dataset_id}")

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc", prefix="ose_woa_")
        try:
            with os.fdopen(tmp_fd, "wb") as fh:
                fh.write(resp.content)
            ds = nc4.Dataset(tmp_path)
            try:
                arr   = np.array(ds.variables[var_name][:]).flatten()
                arr   = np.where(np.abs(arr) > 1e30, np.nan, arr)
                valid = arr[~np.isnan(arr)]
                if len(valid) == 0:
                    raise RuntimeError(f"WOA23 {dataset_id}: all values masked.")
                mean_val = float(np.nanmean(valid))
                std_val  = float(np.nanstd(valid, ddof=1)) if len(valid) > 1 else None
            finally:
                ds.close()
        finally:
            try: os.unlink(tmp_path)
            except OSError: pass

        if which == "temp":
            t_mean, t_std = mean_val, std_val
        else:
            s_mean, s_std = mean_val, std_val

    logger.info(
        "WOA23 T=%.2f±%.2f°C  S=%.2f±%.2fPSU  month=%d",
        t_mean or 0, t_std or 0, s_mean or 0, s_std or 0, calendar_month,
    )
    return BaselineStats(
        temperature_mean=t_mean, temperature_std=t_std,
        salinity_mean=s_mean,    salinity_std=s_std,
        sst_mean=None,           sst_std=None,
        source="woa23", calendar_month=calendar_month,
        n_profiles=0, n_readings=0,
        caveat="WOA23 std estimated from spatial variability across grid cells.",
    )


# ---------------------------------------------------------------------------
# Tier 2: Historical Argo baseline
# ---------------------------------------------------------------------------

def _fetch_argo_historical(
    region, calendar_month: int,
    depth_min: float, depth_max: float,
    baseline_start: date, baseline_end: date,
    cfg: dict,
) -> BaselineStats:
    """Build mean+std from historical Argo profiles, same month, 2005-2020."""
    from data.argo_loader import fetch_argo_profiles, ArgoNoDataError
    from core.qc_module import GOOD_FLAGS

    frames_temp, frames_sal = [], []
    year = baseline_start.year
    while year <= baseline_end.year:
        import calendar as _cal
        start_str = f"{year}-{calendar_month:02d}-01"
        last_day  = _cal.monthrange(year, calendar_month)[1]
        end_str   = f"{year}-{calendar_month:02d}-{last_day:02d}"
        if date.fromisoformat(end_str) < baseline_start:
            year += 1; continue
        if date.fromisoformat(start_str) > baseline_end:
            break
        try:
            df = fetch_argo_profiles(
                lon_min=region.lon_min, lon_max=region.lon_max,
                lat_min=region.lat_min, lat_max=region.lat_max,
                start_date=start_str, end_date=end_str,
                depth_min=depth_min, depth_max=depth_max,
            )
            if df is not None and not df.empty:
                dm = (df["PRES"] >= depth_min) & (df["PRES"] <= depth_max)
                df_d = df[dm].copy()
                gt = df_d[df_d["TEMP_QC"].isin(GOOD_FLAGS)]["TEMP"].dropna()
                gs = df_d[df_d["PSAL_QC"].isin(GOOD_FLAGS)]["PSAL"].dropna()
                if not gt.empty: frames_temp.append(gt)
                if not gs.empty: frames_sal.append(gs)
        except ArgoNoDataError:
            pass
        except Exception as exc:
            logger.warning("Argo historical year %d: %s", year, exc)
        year += 1

    temp_all = pd.concat(frames_temp) if frames_temp else pd.Series(dtype=float)
    sal_all  = pd.concat(frames_sal)  if frames_sal  else pd.Series(dtype=float)

    if temp_all.empty and sal_all.empty:
        raise RuntimeError(
            f"No historical Argo data for {region.name} month={calendar_month} "
            f"depth={depth_min}–{depth_max}."
        )

    def _safe_std(s: pd.Series) -> Optional[float]:
        return float(s.std(ddof=1)) if len(s) >= 2 else None

    caveat = None
    if len(temp_all) < 10 or len(sal_all) < 10:
        caveat = (f"Sparse Argo baseline (T:{len(temp_all)} S:{len(sal_all)} readings).")

    return BaselineStats(
        temperature_mean=float(temp_all.mean()) if not temp_all.empty else None,
        temperature_std =_safe_std(temp_all),
        salinity_mean   =float(sal_all.mean())  if not sal_all.empty  else None,
        salinity_std    =_safe_std(sal_all),
        sst_mean=None, sst_std=None,
        source="argo_historical", calendar_month=calendar_month,
        n_profiles=len(frames_temp), n_readings=len(temp_all)+len(sal_all),
        caveat=caveat,
    )


# ---------------------------------------------------------------------------
# OISST skin-SST historical baseline
# ---------------------------------------------------------------------------

def _fetch_oisst_historical_baseline(
    region,
    calendar_month: int,
    baseline_start: date,
    baseline_end: date,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute the historical mean and std of NOAA OISST for the given calendar
    month over the baseline period, spatially averaged over the region centre
    (2°×2° box to keep requests small).

    Uses the existing data/sst_loader.fetch_sst() — read-only.
    One monthly request per year → ~15 years = 15 small ERDDAP requests.

    Returns (mean_sst, std_sst) in °C, or (None, None) on failure.
    Citation: Huang et al. (2020), doi:10.25921/RE9P-PT57
    """
    from data.sst_loader import fetch_sst
    import calendar as _cal

    # Use a 2°×2° centre box to keep each request small
    lat_c   = (region.lat_min + region.lat_max) / 2
    lon_c   = (region.lon_min + region.lon_max) / 2
    lat_min = round(lat_c - 1.0, 2)
    lat_max = round(lat_c + 1.0, 2)
    lon_min = round(lon_c - 1.0, 2)
    lon_max = round(lon_c + 1.0, 2)

    monthly_means = []
    year = baseline_start.year
    while year <= baseline_end.year:
        start_str = f"{year}-{calendar_month:02d}-01"
        last_day  = _cal.monthrange(year, calendar_month)[1]
        end_str   = f"{year}-{calendar_month:02d}-{last_day:02d}"

        if date.fromisoformat(end_str) < baseline_start:
            year += 1; continue
        if date.fromisoformat(start_str) > baseline_end:
            break

        try:
            df = fetch_sst(
                lat_min=lat_min, lat_max=lat_max,
                lon_min=lon_min, lon_max=lon_max,
                start_date=start_str, end_date=end_str,
            )
            if df is not None and not df.empty and "sst" in df.columns:
                valid = df["sst"].dropna()
                if not valid.empty:
                    monthly_means.append(float(valid.mean()))
        except Exception as exc:
            logger.debug("OISST baseline year %d month %d: %s", year, calendar_month, exc)
        year += 1

    if not monthly_means:
        return None, None

    arr = np.array(monthly_means)
    mean_sst = float(np.mean(arr))
    std_sst  = float(np.std(arr, ddof=1)) if len(arr) >= 2 else None
    return mean_sst, std_sst


# ---------------------------------------------------------------------------
# Synthetic baseline (tests — no network)
# ---------------------------------------------------------------------------

def _synthetic_baseline(calendar_month: int) -> BaselineStats:
    seasonal_offset = 2.0 * np.sin(2 * np.pi * (calendar_month - 3) / 12)
    mean_t = round(25.0 + seasonal_offset, 2)
    return BaselineStats(
        temperature_mean=mean_t,  temperature_std=1.5,
        salinity_mean=35.0,       salinity_std=0.5,
        sst_mean=round(mean_t + 0.3, 2),  sst_std=0.8,
        source="synthetic", calendar_month=calendar_month,
        n_profiles=0, n_readings=0,
        caveat="Synthetic baseline — for testing only.",
    )


# ---------------------------------------------------------------------------
# Cache helpers (ose_ prefix on all files)
# ---------------------------------------------------------------------------

def _cache_dir(cfg: dict) -> str:
    rel  = cfg.get("climatology", {}).get("cache", {}).get("directory", "ose_cache")
    base = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base, rel)
    os.makedirs(path, exist_ok=True)
    return path


def _cache_path(cache_dir: str, key: str) -> str:
    safe = key.replace("/", "_").replace("\\", "_").replace(":", "_")
    return os.path.join(cache_dir, f"{safe}.json")


def _cache_load(cache_dir: str, key: str, ttl_days: int) -> Optional[dict]:
    path = _cache_path(cache_dir, key)
    if not os.path.exists(path):
        return None
    if (time.time() - os.path.getmtime(path)) / 86400 > ttl_days:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _cache_save(cache_dir: str, key: str, data: dict) -> None:
    path = _cache_path(cache_dir, key)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
    except Exception as exc:
        logger.warning("OSE cache write failed: %s", exc)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

_CFG_CACHE: Optional[dict] = None

def _load_config() -> dict:
    global _CFG_CACHE
    if _CFG_CACHE is not None:
        return _CFG_CACHE
    import yaml  # type: ignore
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        _CFG_CACHE = yaml.safe_load(fh)
    return _CFG_CACHE
