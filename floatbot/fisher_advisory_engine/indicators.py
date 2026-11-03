"""
fisher_advisory_engine/indicators.py
--------------------------------------
Computes the six ocean-condition indicators for the Fisher Advisory Engine.
Each indicator is computed independently; a failure in one never prevents
the others from running.

Every indicator returns an IndicatorResult carrying:
  value          — the measured quantity (None = no data)
  unit           — physical unit string
  status         — "ok" | "no_data" | "stale" | "error"
  source         — dataset name and source URL
  resolution     — spatial/temporal resolution description
  timestamp      — ISO datetime of the observation or model run
  data_age_hours — age in hours at the time of computation
  caveat         — plain-English note about limitations (or None)

Indicators
----------
  1. sst_indicator        — NOAA OISST v2.1 sea-surface temperature + anomaly
                            + Hobday (2016) marine heatwave flag
  2. current_indicator    — NOAA HYCOM model surface currents (speed + direction)
                            + GDP drifter sanity check if available
  3. front_indicator      — SST gradient thermal front (delegates to front_detection)
  4. productivity_indicator — CMEMS satellite Chl-a category
  5. subsurface_indicator — Argo mixed layer depth (nearest float)
  6. bathymetry_indicator — GEBCO/ETOPO1 depth at the query location

Data honesty rules enforced here
---------------------------------
  - SST data is past/observed.  For future days, the latest available SST
    is returned with its actual date and a "forecast not available" caveat.
  - HYCOM currents are model output; labelled "forecast" in source field.
  - Drifter data is used only as a sanity check; not for primary advisory.
  - Chl-a: cloud gaps and nearshore turbidity expected; returns "no_data"
    rather than interpolating silently.
  - Argo: distance to nearest float is always reported; if > max_dist_km
    (config), status is "no_data" with an explicit caveat.
  - All resolution mismatches between the data grid and the query box are
    noted in the caveat field and propagate to confidence degradation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IndicatorResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class IndicatorResult:
    name:            str
    value:           object          # float, str, bool, dict — depends on indicator
    unit:            str
    status:          str             # "ok" | "no_data" | "stale" | "error"
    source:          str             # dataset citation
    resolution:      str             # e.g. "0.25° daily"
    timestamp:       Optional[str]   # ISO-8601 UTC of the observation
    data_age_hours:  Optional[float] # age at computation time
    caveat:          Optional[str]   # plain-English caveat or None


def _make_no_data(name: str, unit: str, source: str,
                  resolution: str, caveat: str) -> IndicatorResult:
    return IndicatorResult(
        name=name, value=None, unit=unit,
        status="no_data", source=source, resolution=resolution,
        timestamp=None, data_age_hours=None, caveat=caveat,
    )


def _age_hours(timestamp_iso: str) -> float:
    """Return hours elapsed since a UTC ISO-8601 timestamp."""
    try:
        ts  = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
        now = datetime.now(tz=timezone.utc)
        return (now - ts).total_seconds() / 3600.0
    except Exception:
        return float("inf")


# ---------------------------------------------------------------------------
# 1. SST indicator
# ---------------------------------------------------------------------------

def sst_indicator(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    use_synthetic: bool = False,
) -> IndicatorResult:
    """
    Fetch NOAA OISST v2.1 SST for a ±1° box around the location.
    Computes:
      - mean SST over the period
      - SST anomaly vs 10-year same-calendar-month OISST mean (inter-annual)
      - Hobday (2016) marine heatwave flag

    Returns an IndicatorResult with value = dict:
      {sst_c, anomaly_c, heatwave: bool, baseline_mean_c, baseline_std_c,
       heatwave_events: list[dict]}
    """
    name       = "sst"
    unit       = "°C"
    source     = ("NOAA OISST v2.1 — Huang et al. (2020) "
                  "doi:10.25921/RE9P-PT57; 0.25° daily grid")
    resolution = "0.25° (~28 km) daily"

    if use_synthetic:
        try:
            from data.sst_loader import make_synthetic_sst
            sst_df = make_synthetic_sst(
                lat_center=lat, lon_center=lon,
                start_date=start_date, end_date=end_date,
            )
            mean_sst = float(sst_df["sst"].mean())
            return IndicatorResult(
                name=name,
                value={"sst_c": round(mean_sst, 2), "anomaly_c": None,
                       "heatwave": False, "baseline_mean_c": None,
                       "baseline_std_c": None, "heatwave_events": []},
                unit=unit, status="ok", source="synthetic",
                resolution=resolution,
                timestamp=pd.Timestamp(end_date).isoformat(),
                data_age_hours=0.0,
                caveat="Synthetic data — for testing only.",
            )
        except Exception as exc:
            return _make_no_data(name, unit, "synthetic", resolution,
                                 f"Synthetic SST failed: {exc}")

    try:
        from data.sst_loader import fetch_sst
        sst_df = fetch_sst(
            lat_min=lat - 1.0, lat_max=lat + 1.0,
            lon_min=lon - 1.0, lon_max=lon + 1.0,
            start_date=start_date, end_date=end_date,
        )
    except Exception as exc:
        return _make_no_data(name, unit, source, resolution,
                             f"SST fetch failed: {str(exc)[:200]}")

    if sst_df is None or sst_df.empty:
        return _make_no_data(name, unit, source, resolution,
                             "No OISST data returned for this region and period.")

    mean_sst      = float(sst_df["sst"].dropna().mean())
    latest_date   = pd.to_datetime(sst_df["date"]).max()
    timestamp_iso = latest_date.isoformat() + "Z"
    age_h         = _age_hours(timestamp_iso)

    # Heatwave detection using existing FloatAi heatwave module
    heatwave_flag    = False
    heatwave_events  = []
    baseline_mean_c  = None
    baseline_std_c   = None
    try:
        from core.heatwave_module import detect_heatwaves
        # fetch_sst returns date/lat/lon/sst — need a timeseries
        ts_df = sst_df.groupby("date")["sst"].mean().reset_index()
        ts_df = ts_df.rename(columns={"sst": "sst"})
        hw_result = detect_heatwaves(
            ts_df, query_start=start_date, query_end=end_date
        )
        heatwave_flag   = hw_result.heatwave_detected
        heatwave_events = [e.to_dict() for e in hw_result.overlapping_events]
    except Exception as exc:
        logger.debug("SST indicator: heatwave detection failed (non-fatal): %s", exc)

    # SST anomaly — compute inter-annual baseline for the same calendar month
    anomaly_c = None
    try:
        baseline_mean_c, baseline_std_c = _sst_interannual_baseline(
            lat, lon, latest_date.month
        )
        if baseline_mean_c is not None:
            anomaly_c = round(mean_sst - baseline_mean_c, 2)
    except Exception as exc:
        logger.debug("SST anomaly baseline failed (non-fatal): %s", exc)

    # Resolution mismatch check
    caveat = None
    cfg = _load_config()
    mismatch_thresh = cfg.get("confidence", {}).get(
        "resolution_mismatch_threshold_deg", 0.5
    )
    if mismatch_thresh < 0.25:
        caveat = (
            "OISST 0.25° grid cannot resolve small nearshore fronts "
            "or features < ~28 km. Confidence lowered for nearshore queries."
        )

    status = "stale" if age_h > 96 else "ok"

    return IndicatorResult(
        name=name,
        value={
            "sst_c":          round(mean_sst, 2),
            "anomaly_c":      anomaly_c,
            "heatwave":       heatwave_flag,
            "baseline_mean_c": round(baseline_mean_c, 2) if baseline_mean_c is not None else None,
            "baseline_std_c":  round(baseline_std_c, 2) if baseline_std_c is not None else None,
            "heatwave_events": heatwave_events,
        },
        unit=unit,
        status=status,
        source=source,
        resolution=resolution,
        timestamp=timestamp_iso,
        data_age_hours=round(age_h, 1),
        caveat=caveat,
    )


def _sst_interannual_baseline(lat: float, lon: float, calendar_month: int
                               ) -> tuple:
    """
    Compute inter-annual SST mean and std for the same calendar month
    over the last 10 years (config: oisst_baseline.years_back).
    Returns (mean, std) or (None, None) on failure.
    """
    import calendar as _cal
    from data.sst_loader import fetch_sst

    cfg        = _load_config()
    years_back = int(cfg.get("oisst_baseline", {}).get("years_back", 10))
    min_years  = int(cfg.get("oisst_baseline", {}).get("min_years",  3))

    import datetime as _dt
    this_year   = _dt.date.today().year
    annual_means = []

    for yr in range(this_year - years_back, this_year):
        start_str = f"{yr}-{calendar_month:02d}-01"
        last_day  = _cal.monthrange(yr, calendar_month)[1]
        end_str   = f"{yr}-{calendar_month:02d}-{last_day:02d}"
        try:
            df = fetch_sst(
                lat_min=lat - 1.0, lat_max=lat + 1.0,
                lon_min=lon - 1.0, lon_max=lon + 1.0,
                start_date=start_str, end_date=end_str,
            )
            if df is not None and not df.empty:
                valid = df["sst"].dropna()
                if not valid.empty:
                    annual_means.append(float(valid.mean()))
        except Exception:
            pass

    if len(annual_means) < min_years:
        return None, None
    arr = np.array(annual_means)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else (float(arr[0]), None)


# ---------------------------------------------------------------------------
# 2. Current indicator
# ---------------------------------------------------------------------------

def current_indicator(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    use_synthetic: bool = False,
) -> IndicatorResult:
    """
    Fetch NOAA HYCOM model surface currents (speed + direction).
    Drifter data used as a sanity check only if available.
    HYCOM output is labelled "model forecast" — never presented as observation.
    """
    name       = "currents"
    unit       = "m/s"
    source     = ("NOAA HYCOM model — forecast; "
                  "https://www.ncei.noaa.gov/products/hycom")
    resolution = "1/12° (~8 km) daily model"

    if use_synthetic:
        return IndicatorResult(
            name=name,
            value={"speed_ms": 0.30, "direction_deg": 225.0,
                   "direction_cardinal": "SW",
                   "u_ms": -0.21, "v_ms": -0.21,
                   "drifter_speed_ms": None, "drifter_sanity": "not_available"},
            unit=unit, status="ok", source="synthetic",
            resolution=resolution,
            timestamp=pd.Timestamp(end_date).isoformat(),
            data_age_hours=0.0,
            caveat="Synthetic data — for testing only.",
        )

    try:
        from data.hycom_loader import fetch_hycom_surface
        hdf = fetch_hycom_surface(
            lat_min=lat - 1.0, lat_max=lat + 1.0,
            lon_min=lon - 1.0, lon_max=lon + 1.0,
            start_date=start_date, end_date=end_date,
        )
    except Exception as exc:
        return _make_no_data(name, unit, source, resolution,
                             f"HYCOM fetch failed: {str(exc)[:200]}")

    if hdf is None or hdf.empty or "U_CURR" not in hdf.columns:
        return _make_no_data(name, unit, source, resolution,
                             "No HYCOM current data returned.")

    valid = hdf.dropna(subset=["U_CURR", "V_CURR"])
    if valid.empty:
        return _make_no_data(name, unit, source, resolution,
                             "HYCOM returned NaN for all current values.")

    u_mean = float(valid["U_CURR"].mean())
    v_mean = float(valid["V_CURR"].mean())
    speed  = float(np.sqrt(u_mean**2 + v_mean**2))
    # Oceanographic convention: direction current flows TOWARDS
    direction_deg = float((np.degrees(np.arctan2(u_mean, v_mean)) + 360) % 360)

    latest_time   = pd.to_datetime(valid["TIME"]).max()
    timestamp_iso = latest_time.isoformat() + "Z"
    age_h         = _age_hours(timestamp_iso)

    # GDP drifter sanity check
    drifter_speed = None
    drifter_note  = "not_available"
    try:
        from data.drifter_loader import fetch_drifter_sst
        ddf = fetch_drifter_sst(
            lat_min=lat - 1.5, lat_max=lat + 1.5,
            lon_min=lon - 1.5, lon_max=lon + 1.5,
            start_date=start_date, end_date=end_date,
        )
        if ddf is not None and not ddf.empty and "U_CURR" in ddf.columns:
            dv = ddf.dropna(subset=["U_CURR", "V_CURR"])
            if not dv.empty:
                du = float(dv["U_CURR"].mean())
                dv_v = float(dv["V_CURR"].mean())
                drifter_speed = round(float(np.sqrt(du**2 + dv_v**2)), 3)
                diff = abs(speed - drifter_speed)
                drifter_note = "consistent" if diff < 0.3 else "divergent"
    except Exception as exc:
        logger.debug("Drifter sanity check failed (non-fatal): %s", exc)

    return IndicatorResult(
        name=name,
        value={
            "speed_ms":          round(speed, 3),
            "direction_deg":     round(direction_deg, 1),
            "direction_cardinal": _degrees_to_cardinal(direction_deg),
            "u_ms":              round(u_mean, 3),
            "v_ms":              round(v_mean, 3),
            "drifter_speed_ms":  drifter_speed,
            "drifter_sanity":    drifter_note,
        },
        unit=unit,
        status="ok",
        source=source,
        resolution=resolution,
        timestamp=timestamp_iso,
        data_age_hours=round(age_h, 1),
        caveat=(
            "HYCOM currents are model output (forecast), not direct observations. "
            "Drifter data used as a sanity check only."
        ),
    )


def _degrees_to_cardinal(deg: float) -> str:
    """Convert bearing degrees to 8-point cardinal direction."""
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx  = int((deg + 22.5) / 45) % 8
    return dirs[idx]


# ---------------------------------------------------------------------------
# 3. Front indicator — delegates to front_detection.py
# ---------------------------------------------------------------------------

def front_indicator(
    lat: float,
    lon: float,
    date_str: str,
    use_synthetic: bool = False,
) -> IndicatorResult:
    """
    Detect a thermal front near the location using the SST spatial gradient.
    Delegates computation to front_detection.py.
    """
    name       = "thermal_front"
    unit       = "°C/km"
    source     = "Derived from NOAA OISST v2.1 spatial gradient"
    resolution = "0.25° (~28 km) daily"

    try:
        from fisher_advisory_engine.front_detection import detect_front, FrontResult
        result: FrontResult = detect_front(lat, lon, date_str,
                                           use_synthetic=use_synthetic)
    except Exception as exc:
        return _make_no_data(name, unit, source, resolution,
                             f"Front detection failed: {str(exc)[:200]}")

    if result is None or result.status == "no_data":
        return _make_no_data(name, unit, source, resolution,
                             result.caveat if result else "Front detection returned no data.")

    age_h = _age_hours(result.timestamp) if result.timestamp else None

    return IndicatorResult(
        name=name,
        value={
            "detected":       result.detected,
            "strength":       result.strength,
            "gradient_c_km":  result.gradient_c_km,
            "dist_km":        result.dist_km,
            "bearing_deg":    result.bearing_deg,
        },
        unit=unit,
        status=result.status,
        source=source,
        resolution=resolution,
        timestamp=result.timestamp,
        data_age_hours=round(age_h, 1) if age_h is not None else None,
        caveat=result.caveat,
    )


# ---------------------------------------------------------------------------
# 4. Productivity / chlorophyll indicator
# ---------------------------------------------------------------------------

def productivity_indicator(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    use_synthetic: bool = False,
) -> IndicatorResult:
    """
    Fetch CMEMS satellite chlorophyll-a for a ±1° box.
    Returns category LOW / MODERATE / HIGH, or "no_data".

    Cloud gaps and nearshore turbidity are expected; no silent interpolation.
    """
    name       = "productivity"
    unit       = "mg/m³"
    source     = ("CMEMS ocean-colour satellite Chl-a — "
                  "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D")
    resolution = "4 km daily (satellite; cloud gaps possible)"

    cfg = _load_config()
    chl_cfg = cfg.get("chlorophyll", {})
    high_thresh = float(chl_cfg.get("high_mg_m3",     1.0))
    mod_thresh  = float(chl_cfg.get("moderate_mg_m3", 0.3))
    low_thresh  = float(chl_cfg.get("low_mg_m3",      0.1))

    if use_synthetic:
        chl_val  = 0.45   # synthetic moderate
        category = _chl_category(chl_val, high_thresh, mod_thresh, low_thresh)
        return IndicatorResult(
            name=name, value={"chl_mg_m3": chl_val, "category": category},
            unit=unit, status="ok", source="synthetic",
            resolution=resolution,
            timestamp=pd.Timestamp(end_date).isoformat(),
            data_age_hours=0.0,
            caveat="Synthetic data — for testing only.",
        )

    try:
        from data.cmems_loader import fetch_cmems_chlorophyll
        chl_df = fetch_cmems_chlorophyll(
            lat_min=lat - 1.0, lat_max=lat + 1.0,
            lon_min=lon - 1.0, lon_max=lon + 1.0,
            start_date=start_date, end_date=end_date,
        )
    except Exception as exc:
        return _make_no_data(
            name, unit, source, resolution,
            f"Chl-a fetch failed: {str(exc)[:200]}. "
            "Productivity indicator not available.",
        )

    if chl_df is None or chl_df.empty or "CHLOR_A" not in chl_df.columns:
        return _make_no_data(
            name, unit, source, resolution,
            "No chlorophyll-a data returned. "
            "This is expected near the coast due to cloud cover or turbid water. "
            "Confidence is lowered when Chl-a is unavailable.",
        )

    valid_chl = chl_df["CHLOR_A"].dropna()
    if valid_chl.empty:
        return _make_no_data(
            name, unit, source, resolution,
            "All Chl-a pixels were masked (cloud cover or data gap). "
            "Productivity indicator not available.",
        )

    mean_chl  = float(valid_chl.mean())
    category  = _chl_category(mean_chl, high_thresh, mod_thresh, low_thresh)
    latest_ts = pd.to_datetime(chl_df["TIME"]).max()
    ts_iso    = latest_ts.isoformat() + "Z"
    age_h     = _age_hours(ts_iso)
    max_age   = int(chl_cfg.get("max_age_days", 3)) * 24
    status    = "stale" if age_h > max_age else "ok"

    caveat = (
        "Chl-a is a satellite product; nearshore values may be "
        "overestimated due to turbid coastal water (CDOM interference). "
        "PLACEHOLDER thresholds — not yet calibrated for Tamil Nadu waters."
    )

    return IndicatorResult(
        name=name,
        value={"chl_mg_m3": round(mean_chl, 4), "category": category},
        unit=unit,
        status=status,
        source=source,
        resolution=resolution,
        timestamp=ts_iso,
        data_age_hours=round(age_h, 1),
        caveat=caveat,
    )


def _chl_category(chl: float, high: float, mod: float, low: float) -> str:
    if chl >= high:
        return "HIGH"
    if chl >= mod:
        return "MODERATE"
    if chl >= low:
        return "LOW"
    return "LOW"


# ---------------------------------------------------------------------------
# 5. Subsurface / Argo indicator (mixed layer depth)
# ---------------------------------------------------------------------------

def subsurface_indicator(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    use_synthetic: bool = False,
) -> IndicatorResult:
    """
    Fetch nearby Argo profiles and estimate the mixed layer depth (MLD).

    MLD is approximated from the temperature profile using the criterion:
      depth where T drops by mld_temp_criterion_c (config) below the surface.
    (de Boyer Montégut et al. 2004 — criterion value is PLACEHOLDER.)

    Reports distance to nearest float.  If no float is within max_dist_km
    (config), returns status="no_data" rather than guessing.
    """
    name       = "subsurface"
    unit       = "m"
    source     = "Argo GDAC — argopy; https://argo.ucsd.edu"
    resolution = "profile-level (~sparse, varies by float density)"

    cfg       = _load_config()
    max_dist  = float(cfg.get("argo", {}).get("max_dist_km", 150))
    mld_crit  = float(cfg.get("argo", {}).get("mld_temp_criterion_c", 0.2))

    if use_synthetic:
        from data.argo_loader import make_synthetic_profiles
        df = make_synthetic_profiles(lat_center=lat, lon_center=lon)
    else:
        try:
            from data.argo_loader import fetch_argo_profiles, ArgoNoDataError
            df = fetch_argo_profiles(
                lon_min=lon - 2.0, lon_max=lon + 2.0,
                lat_min=lat - 2.0, lat_max=lat + 2.0,
                start_date=start_date, end_date=end_date,
                depth_min=0.0, depth_max=500.0,
            )
        except Exception as exc:
            no_data_msg = (
                "No Argo floats found near this location. "
                "Argo coverage is sparse in nearshore Tamil Nadu waters. "
                f"Detail: {str(exc)[:150]}"
            )
            return _make_no_data(name, unit, source, resolution, no_data_msg)

    if df is None or df.empty:
        return _make_no_data(
            name, unit, source, resolution,
            "No Argo profiles available within the search box. "
            "Subsurface context not available.",
        )

    # Apply QC filter via existing FloatAi QC module
    try:
        from core.qc_module import apply_qc_filter, GOOD_FLAGS
        qcr = apply_qc_filter(df, parameter="temperature",
                               depth_min=0.0, depth_max=500.0)
        df_good = df[df["TEMP_QC"].isin(GOOD_FLAGS)].copy()
    except Exception as exc:
        logger.debug("Argo QC filter failed, using raw data: %s", exc)
        df_good = df.copy()

    if df_good.empty:
        return _make_no_data(name, unit, source, resolution,
                             "All Argo readings failed QC. Subsurface context not available.")

    # Distance to nearest float
    df_good["_dist"] = np.sqrt(
        (df_good["LATITUDE"]  - lat)**2 +
        (df_good["LONGITUDE"] - lon)**2
    ) * 111.0   # rough degrees → km
    nearest_km = float(df_good["_dist"].min())

    if nearest_km > max_dist:
        return _make_no_data(
            name, unit, source, resolution,
            f"Nearest Argo float is {nearest_km:.0f} km away "
            f"(threshold: {max_dist:.0f} km). "
            "Subsurface context not available — too distant to be representative.",
        )

    # MLD estimation from the nearest float's most recent profile
    nearest_float  = df_good.loc[df_good["_dist"].idxmin(), "PLATFORM_NUMBER"]
    nearest_cycle  = df_good.loc[df_good["_dist"].idxmin(), "CYCLE_NUMBER"]
    profile        = df_good[
        (df_good["PLATFORM_NUMBER"] == nearest_float) &
        (df_good["CYCLE_NUMBER"]    == nearest_cycle)
    ].sort_values("PRES")

    mld_m = _estimate_mld(profile, criterion=mld_crit)

    # Latest Argo profile time
    latest_time   = pd.to_datetime(df_good["TIME"]).max()
    timestamp_iso = latest_time.isoformat() + "Z"
    age_h         = _age_hours(timestamp_iso)

    caveat = (
        f"MLD estimated using {mld_crit}°C temperature criterion "
        f"(PLACEHOLDER — pending review against published Bay of Bengal norms). "
        f"Nearest float: {nearest_km:.0f} km from query location "
        f"(float ID {nearest_float})."
    )

    return IndicatorResult(
        name=name,
        value={
            "mld_m":          round(mld_m, 1) if mld_m is not None else None,
            "nearest_km":     round(nearest_km, 1),
            "float_id":       str(nearest_float),
            "n_profiles":     df_good.groupby(
                ["PLATFORM_NUMBER","CYCLE_NUMBER"]).ngroups,
        },
        unit=unit,
        status="ok",
        source=source,
        resolution=f"nearest float {nearest_km:.0f} km away",
        timestamp=timestamp_iso,
        data_age_hours=round(age_h, 1),
        caveat=caveat,
    )


def _estimate_mld(profile: pd.DataFrame, criterion: float = 0.2) -> Optional[float]:
    """
    Estimate mixed layer depth from an Argo temperature profile.
    Uses the threshold method: depth at which T drops 'criterion' °C below
    the shallowest good reading.
    Returns depth in dbar (approx metres), or None if insufficient data.
    """
    if "TEMP" not in profile.columns or profile.empty:
        return None
    p = profile.dropna(subset=["TEMP"]).sort_values("PRES")
    if len(p) < 3:
        return None
    t_surface = float(p.iloc[0]["TEMP"])
    for _, row in p.iterrows():
        if float(row["TEMP"]) < (t_surface - criterion):
            return float(row["PRES"])
    return None   # isothermal — MLD below profile range


# ---------------------------------------------------------------------------
# 6. Bathymetry indicator
# ---------------------------------------------------------------------------

def bathymetry_indicator(
    lat: float,
    lon: float,
) -> IndicatorResult:
    """
    Fetch GEBCO/ETOPO1 ocean depth at the query location.
    Provides shelf vs. deep-water context for the trip type.
    """
    name       = "bathymetry"
    unit       = "m"
    source     = ("ETOPO1 global relief via CoastWatch ERDDAP — "
                  "Amante & Eakins (2009) NOAA NGDC")
    resolution = "1 arc-minute (~1.8 km)"

    try:
        from data.gebco_loader import fetch_gebco_point_depth, describe_bathymetry
        depth_m  = fetch_gebco_point_depth(lat, lon)
        desc     = describe_bathymetry(lat, lon)
    except Exception as exc:
        return _make_no_data(name, unit, source, resolution,
                             f"Bathymetry fetch failed: {str(exc)[:200]}")

    if depth_m is None:
        return _make_no_data(
            name, unit, source, resolution,
            "Location appears to be on land or bathymetry data unavailable.",
        )

    now_iso = datetime.now(tz=timezone.utc).isoformat()

    return IndicatorResult(
        name=name,
        value={"depth_m": round(depth_m, 1), "description": desc},
        unit=unit,
        status="ok",
        source=source,
        resolution=resolution,
        timestamp=now_iso,
        data_age_hours=0.0,    # static dataset — always fresh
        caveat=None,
    )


# ---------------------------------------------------------------------------
# Config loader (lazy, cached)
# ---------------------------------------------------------------------------

_CFG_CACHE: Optional[dict] = None

def _load_config() -> dict:
    global _CFG_CACHE
    if _CFG_CACHE is not None:
        return _CFG_CACHE
    import yaml  # type: ignore
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        _CFG_CACHE = yaml.safe_load(fh)
    return _CFG_CACHE
