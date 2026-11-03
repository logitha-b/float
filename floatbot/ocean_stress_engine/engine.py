"""
ocean_stress_engine/engine.py
-------------------------------
Main orchestrator for the Ocean Stress Engine.

compute_stress() is the single public entry point. It:
  1. Validates inputs (inputs.py)
  2. Fetches Argo profiles via existing fetch_argo_profiles() — read-only
  3. Applies QC via existing apply_qc_filter() — read-only, imported not modified
  4. Fetches climatological baseline (climatology.py)
  5. Computes scores (scoring.py)
  6. Computes confidence (confidence.py)
  7. Builds and returns a StressResult (output.py)

The engine never writes to any Argo database. All new writes go to ose_cache/.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ocean_stress_engine.inputs      import validate_inputs
from ocean_stress_engine.climatology import get_baseline
from ocean_stress_engine.scoring     import (
    compute_scores,
    load_weights_and_breakpoints,
)
from ocean_stress_engine.confidence  import (
    compute_confidence,
    nearest_float_distance,
)
from ocean_stress_engine.output      import (
    EvidenceRecord,
    StressResult,
    build_result,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_stress(
    region,
    time_window:       str,
    depth_layer:       Optional[str] = None,
    ecosystem_type:    str           = "general",
    baseline_period:   Optional[str] = None,
    use_synthetic:     bool          = False,
) -> StressResult:
    """
    Compute the thermohaline stress score for a region and time window.

    Parameters
    ----------
    region          : str or dict   Named region or lat/lon bbox dict.
    time_window     : str           "YYYY-MM", "YYYY-MM-DD/YYYY-MM-DD",
                                    "last N days", or "YYYY".
    depth_layer     : str or None   "surface" (default), "mid", or "deep".
    ecosystem_type  : str           "coral_reef", "fisheries", or "general".
    baseline_period : str or None   "YYYY-YYYY". Default: "2005-2020".
    use_synthetic   : bool          Skip live network calls (for tests).

    Returns
    -------
    StressResult    Full result with .to_json(), .summary_text(),
                    .dashboard_card() methods.

    Raises
    ------
    ValueError      For invalid inputs (clear message).
    RuntimeError    If Argo data cannot be fetched and use_synthetic=False.
    """
    # ── 1. Validate inputs ────────────────────────────────────────────────
    inputs = validate_inputs(
        region=region,
        time_window=time_window,
        depth_layer=depth_layer,
        ecosystem_type=ecosystem_type,
        baseline_period=baseline_period,
    )

    # ── 2. Fetch Argo profiles ────────────────────────────────────────────
    df_raw = _fetch_profiles(inputs, use_synthetic)

    # ── 3. Apply QC and extract per-variable observed values ──────────────
    observed, qc_stats, evidence = _apply_qc_and_observe(df_raw, inputs, use_synthetic)

    # ── 4. Fetch climatological baseline ─────────────────────────────────
    mid_date  = inputs.time_window.start + (
        (inputs.time_window.end - inputs.time_window.start) / 2
    )
    cal_month = mid_date.month
    baseline  = get_baseline(
        region=inputs.region,
        calendar_month=cal_month,
        depth_min=inputs.depth_layer.min_dbar,
        depth_max=inputs.depth_layer.max_dbar,
        baseline_start=inputs.baseline_start,
        baseline_end=inputs.baseline_end,
        use_synthetic=use_synthetic,
    )

    # ── 4b. Replace SST baseline std with Hobday daily-climatology std ────
    # The monthly-mean OISST baseline has tiny inter-annual std (±0.4°C)
    # which makes even large anomalies appear small.
    # Instead: fetch full 1-year daily SST ending at query start, compute
    # the daily climatology std for the query month using Hobday window.
    # This gives a physically meaningful spread (~0.8–1.5°C typical).
    if not use_synthetic and observed.get("sst") is not None:
        try:
            sst_mean_hob, sst_std_hob = _fetch_hobday_sst_stats(inputs)
            if sst_mean_hob is not None and sst_std_hob is not None:
                baseline.sst_mean = sst_mean_hob
                baseline.sst_std  = sst_std_hob
                logger.info(
                    "Hobday SST baseline month=%d: mean=%.2f std=%.2f°C",
                    cal_month, sst_mean_hob, sst_std_hob,
                )
        except Exception as exc:
            logger.warning("Hobday SST baseline failed (using monthly fallback): %s", exc)

    # ── 5. Load weights + breakpoints from config ─────────────────────────
    raw_weights, breakpoints = load_weights_and_breakpoints(inputs.ecosystem_type)

    # ── 6. Compute scores ─────────────────────────────────────────────────
    scoring = compute_scores(
        observed=observed,
        baseline=baseline,
        raw_weights=raw_weights,
        breakpoints=breakpoints,
    )

    # ── 7. Compute confidence ─────────────────────────────────────────────
    lat_c, lon_c = inputs.region.centre
    dist_deg     = nearest_float_distance(
        lat_c, lon_c,
        qc_stats["float_lats"], qc_stats["float_lons"],
    )
    confidence = compute_confidence(
        n_profiles=qc_stats["n_profiles"],
        n_readings_total=qc_stats["n_readings_total"],
        n_readings_excluded=qc_stats["n_readings_excluded"],
        nearest_float_dist_deg=dist_deg,
        variables_missing=scoring.variables_missing,
    )

    # ── 8. Build result ───────────────────────────────────────────────────
    result = build_result(
        inputs=inputs,
        scoring=scoring,
        confidence=confidence,
        n_floats=qc_stats["n_floats"],
        n_profiles=qc_stats["n_profiles"],
        n_readings_total=qc_stats["n_readings_total"],
        n_readings_excluded=qc_stats["n_readings_excluded"],
        exclusion_reasons=qc_stats["exclusion_reasons"],
        evidence=evidence,
        baseline_source=baseline.source,
    )

    logger.info(
        "OSE stress=%s category=%s confidence=%s floats=%d region=%s period=%s/%s",
        result.stress_score, result.category, result.confidence,
        result.n_floats, result.region_name,
        result.period_start, result.period_end,
    )
    return result


# ---------------------------------------------------------------------------
# Internal: fetch profiles
# ---------------------------------------------------------------------------

def _fetch_profiles(inputs, use_synthetic: bool) -> pd.DataFrame:
    """
    Fetch Argo profiles for the query region and time window.
    Read-only: calls existing fetch_argo_profiles() or make_synthetic_profiles().
    """
    if use_synthetic:
        from data.argo_loader import make_synthetic_profiles
        df = make_synthetic_profiles(
            lat_center=inputs.region.centre[0],
            lon_center=inputs.region.centre[1],
        )
        df["TIME"] = pd.to_datetime(df["TIME"])
        mask = (
            (df["TIME"].dt.date >= inputs.time_window.start) &
            (df["TIME"].dt.date <= inputs.time_window.end)
        )
        return df[mask].reset_index(drop=True)

    from data.argo_loader import fetch_argo_profiles, ArgoNoDataError
    try:
        df = fetch_argo_profiles(
            lon_min=inputs.region.lon_min,
            lon_max=inputs.region.lon_max,
            lat_min=inputs.region.lat_min,
            lat_max=inputs.region.lat_max,
            start_date=inputs.time_window.start_iso,
            end_date=inputs.time_window.end_iso,
            depth_min=inputs.depth_layer.min_dbar,
            depth_max=inputs.depth_layer.max_dbar,
        )
        return df
    except ArgoNoDataError as exc:
        raise RuntimeError(
            f"No Argo float data found for {inputs.region.name} "
            f"({inputs.time_window.start_iso} – {inputs.time_window.end_iso}). "
            f"Detail: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Internal: apply QC + compute observed means
# ---------------------------------------------------------------------------

def _apply_qc_and_observe(
    df: pd.DataFrame,
    inputs,
    use_synthetic: bool = False,
) -> tuple[Dict[str, Optional[float]], dict, List[EvidenceRecord]]:
    """
    Apply QC filtering to the Argo DataFrame and extract observed means.

    Uses existing apply_qc_filter() — imported, not modified.

    Returns
    -------
    observed : dict variable → mean value (None if no good data)
    qc_stats : dict with float/profile/reading counts and exclusion breakdown
    evidence : list[EvidenceRecord] sample of used profiles
    """
    from core.qc_module import apply_qc_filter, GOOD_FLAGS, BAD_FLAGS

    observed:          Dict[str, Optional[float]] = {}
    exclusion_reasons: Counter = Counter()
    float_lats:  List[float] = []
    float_lons:  List[float] = []

    # Depth filter
    depth_mask = (
        (df["PRES"] >= inputs.depth_layer.min_dbar) &
        (df["PRES"] <= inputs.depth_layer.max_dbar)
    )
    df_depth = df[depth_mask].copy()
    n_readings_total = len(df_depth)

    # Temperature QC
    temp_result = apply_qc_filter(
        df_depth,
        parameter="temperature",
        statistic="mean",
        depth_min=inputs.depth_layer.min_dbar,
        depth_max=inputs.depth_layer.max_dbar,
    )
    observed["temperature"] = temp_result.value

    # Salinity QC
    sal_result = apply_qc_filter(
        df_depth,
        parameter="salinity",
        statistic="mean",
        depth_min=inputs.depth_layer.min_dbar,
        depth_max=inputs.depth_layer.max_dbar,
    )
    observed["salinity"] = sal_result.value

    # Oxygen and pH: not in standard Argo v1 — mark as missing
    observed["oxygen"] = None
    observed["ph"]     = None

    # NOAA OISST skin SST — primary bleaching driver (0-5m)
    # Use the full region box (capped at 10°×10° for speed) for both
    # observed and baseline so they represent the same area.
    observed["sst"] = None
    if not use_synthetic:
        try:
            from data.sst_loader import fetch_sst as _fetch_sst_grid
            # Cap region to 10°×10° centred on region to keep requests fast
            lat_c  = (inputs.region.lat_min + inputs.region.lat_max) / 2
            lon_c  = (inputs.region.lon_min + inputs.region.lon_max) / 2
            half   = min(5.0,
                         abs(inputs.region.lat_max - inputs.region.lat_min) / 2,
                         abs(inputs.region.lon_max - inputs.region.lon_min) / 2)
            sst_lat_min = round(max(lat_c - half, -90), 2)
            sst_lat_max = round(min(lat_c + half,  90), 2)
            sst_lon_min = round(lon_c - half, 2)
            sst_lon_max = round(lon_c + half, 2)
            sst_df = _fetch_sst_grid(
                lat_min=sst_lat_min, lat_max=sst_lat_max,
                lon_min=sst_lon_min, lon_max=sst_lon_max,
                start_date=inputs.time_window.start_iso,
                end_date=inputs.time_window.end_iso,
            )
            if sst_df is not None and not sst_df.empty and "sst" in sst_df.columns:
                valid_sst = sst_df["sst"].dropna()
                if not valid_sst.empty:
                    observed["sst"] = round(float(valid_sst.mean()), 4)
                    logger.info(
                        "OISST observed SST: %.2f°C (n=%d values, box=%.1f-%.1f,%.1f-%.1f)",
                        observed["sst"], len(valid_sst),
                        sst_lat_min, sst_lat_max, sst_lon_min, sst_lon_max,
                    )
        except Exception as exc:
            logger.warning("OISST observed SST fetch failed (non-fatal): %s", exc)
    else:
        # Synthetic SST: Argo temp + small warm bias (skin effect)
        if observed.get("temperature") is not None:
            observed["sst"] = round(observed["temperature"] + 0.5, 4)

    # Aggregate exclusion reasons from both QC results
    for qcr in (temp_result, sal_result):
        for exc_rec in qcr.exclusions:
            exclusion_reasons[exc_rec.failed_test] += 1

    n_readings_excluded = temp_result.n_bad + sal_result.n_bad

    # Float / profile counts from raw (pre-QC) depth-filtered data
    n_floats   = df_depth["PLATFORM_NUMBER"].nunique() if not df_depth.empty else 0
    n_profiles = (
        df_depth.groupby(["PLATFORM_NUMBER", "CYCLE_NUMBER"]).ngroups
        if not df_depth.empty else 0
    )

    # Spatial coverage: collect lat/lon of QC-good readings
    good_mask = df_depth["TEMP_QC"].isin(GOOD_FLAGS) | df_depth["PSAL_QC"].isin(GOOD_FLAGS)
    df_good   = df_depth[good_mask]
    float_lats = df_good["LATITUDE"].dropna().tolist()
    float_lons = df_good["LONGITUDE"].dropna().tolist()

    # Evidence: up to 20 sample profiles (first unique float+cycle combos)
    evidence: List[EvidenceRecord] = []
    if not df_good.empty:
        sample = (
            df_good
            .drop_duplicates(subset=["PLATFORM_NUMBER", "CYCLE_NUMBER"])
            .head(20)
        )
        for _, row in sample.iterrows():
            evidence.append(EvidenceRecord(
                float_id=str(row["PLATFORM_NUMBER"]),
                cycle=int(row["CYCLE_NUMBER"]),
                depth_m=float(row["PRES"]),
                lat=float(row["LATITUDE"]),
                lon=float(row["LONGITUDE"]),
            ))

    qc_stats = {
        "n_floats":           n_floats,
        "n_profiles":         n_profiles,
        "n_readings_total":   n_readings_total,
        "n_readings_excluded": n_readings_excluded,
        "exclusion_reasons":  dict(exclusion_reasons),
        "float_lats":         float_lats,
        "float_lons":         float_lons,
    }
    return observed, qc_stats, evidence


# ---------------------------------------------------------------------------
# Internal: inter-annual SST climatology stats
# ---------------------------------------------------------------------------

def _fetch_hobday_sst_stats(inputs) -> tuple:
    """
    Compute SST climatological mean and inter-annual std.

    Fetches the SAME calendar month over each baseline year (2005-2015 by
    default), computes the annual mean for that month, then returns
    (grand_mean, inter_annual_std).

    Inter-annual std (year-to-year spread ~0.3-0.8 degC) is the correct
    denominator for detecting anomalous years like 2016 El Nino.

    Uses a 4x4 degree box centred on the region for speed.
    Returns (mean_sst, std_sst) or (None, None) on failure.
    """
    from data.sst_loader import fetch_sst
    import calendar as _cal
    import numpy as np

    lat_c = (inputs.region.lat_min + inputs.region.lat_max) / 2
    lon_c = (inputs.region.lon_min + inputs.region.lon_max) / 2
    half  = min(5.0,
                abs(inputs.region.lat_max - inputs.region.lat_min) / 2,
                abs(inputs.region.lon_max - inputs.region.lon_min) / 2)
    lat_min = round(max(lat_c - half, -90), 2)
    lat_max = round(min(lat_c + half,  90), 2)
    lon_min = round(lon_c - half, 2)
    lon_max = round(lon_c + half, 2)

    mid_date   = inputs.time_window.start + (
        (inputs.time_window.end - inputs.time_window.start) / 2
    )
    target_mon = mid_date.month

    annual_means = []
    year = inputs.baseline_start.year
    while year <= min(inputs.baseline_end.year,
                      inputs.time_window.start.year - 1):
        start_str = f"{year}-{target_mon:02d}-01"
        last_day  = _cal.monthrange(year, target_mon)[1]
        end_str   = f"{year}-{target_mon:02d}-{last_day:02d}"
        try:
            df = fetch_sst(
                lat_min=lat_min, lat_max=lat_max,
                lon_min=lon_min, lon_max=lon_max,
                start_date=start_str, end_date=end_str,
            )
            if df is not None and not df.empty and "sst" in df.columns:
                valid = df["sst"].dropna()
                if not valid.empty:
                    annual_means.append(float(valid.mean()))
        except Exception as exc:
            logger.debug("SST inter-annual year %d: %s", year, exc)
        year += 1

    if len(annual_means) < 3:
        return None, None

    arr      = np.array(annual_means)
    mean_sst = float(np.mean(arr))
    std_sst  = float(np.std(arr, ddof=1))
    logger.info(
        "Inter-annual SST baseline month=%d: mean=%.2f std=%.2f degC "
        "(%d years, lat=%.1f-%.1f lon=%.1f-%.1f)",
        target_mon, mean_sst, std_sst, len(arr),
        lat_min, lat_max, lon_min, lon_max,
    )
    return mean_sst, std_sst



