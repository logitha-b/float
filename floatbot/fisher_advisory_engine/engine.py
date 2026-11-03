"""
fisher_advisory_engine/engine.py
----------------------------------
Main orchestrator for the Fisher Advisory Engine.

compute_advisory() is the single public entry point.  It:

  1. Validates inputs                    (inputs.py)
  2. Computes all 6 indicators in parallel using ThreadPoolExecutor
     — each indicator failure is isolated and never blocks the others
     (indicators.py, front_detection.py)
  3. Builds the flat context dict for the rules engine
  4. Evaluates rules                     (rules_engine.py)
  5. Computes confidence                 (confidence.py)
  6. Assembles and returns AdvisoryResult (output.py)

The engine is read-only with respect to all ocean databases.  Any new
writes (cache) go to fae_cache/ with fae_ prefix.

Data-honesty enforcement
------------------------
  - If the day is in the future, SST is flagged as "latest observed" with
    its actual date.  The result carries is_future_day=True so the i18n
    renderer can label fields appropriately.
  - HYCOM currents are always labelled "model forecast" in the source field.
  - Unverified port coordinates produce a LOW confidence result with an
    explicit caveat that all values are meaningless until coordinates are
    confirmed.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_advisory(
    location,
    day: str = "today",
    trip_type: str = "nearshore",
    boat_type=None,
    language=None,
    use_synthetic: bool = False,
) -> "AdvisoryResult":
    """
    Compute a fisher condition advisory for a location and day.

    Parameters
    ----------
    location     : str or dict  Port name or {"lat": float, "lon": float}.
    day          : str          "today", "tomorrow", "last 7 days", "YYYY-MM-DD".
    trip_type    : str          "nearshore" | "offshore".
    boat_type    : str or None  "traditional" | "mechanized" | None.
    language     : str | list | None
    use_synthetic: bool         Skip live network calls (for tests).

    Returns
    -------
    AdvisoryResult

    Raises
    ------
    ValueError   For invalid inputs (clear message).
    """
    from fisher_advisory_engine.inputs     import validate_inputs
    from fisher_advisory_engine.rules_engine import evaluate_rules
    from fisher_advisory_engine.confidence  import compute_confidence
    from fisher_advisory_engine.output      import build_result

    # ── 1. Validate inputs ─────────────────────────────────────────────────
    inputs = validate_inputs(
        location=location,
        day=day,
        trip_type=trip_type,
        boat_type=boat_type,
        language=language,
    )

    lat       = inputs.location.lat
    lon       = inputs.location.lon
    start_iso = inputs.day.start_iso
    end_iso   = inputs.day.end_iso

    # For future days we still fetch the latest available observed data
    # (SST, HYCOM) and label it accordingly in the output.
    fetch_end = end_iso   # callers may pass "tomorrow"; loaders cap to yesterday

    # ── 2. Compute indicators in parallel ──────────────────────────────────
    indicators = _compute_indicators_parallel(
        lat, lon, start_iso, fetch_end, use_synthetic
    )

    # ── 3. Build rules context dict ────────────────────────────────────────
    context = _build_rules_context(indicators)

    # ── 4. Evaluate rules ──────────────────────────────────────────────────
    rules_triggered, rules_disabled = evaluate_rules(
        context, trip_type=inputs.trip_type
    )

    # ── 5. Nearest Argo distance ───────────────────────────────────────────
    nearest_argo_km = _extract_nearest_argo_km(indicators)

    # ── 6. Confidence ──────────────────────────────────────────────────────
    confidence = compute_confidence(indicators, inputs, nearest_argo_km)

    # ── 7. Aggregate caveats ───────────────────────────────────────────────
    caveats: List[str] = []
    if inputs.day.is_future:
        caveats.append(
            f"Day requested ({inputs.day.label}) is in the future. "
            "SST is the latest available observed data (not a forecast). "
            "HYCOM currents are a model forecast. "
            "Sea conditions may differ from what is shown here."
        )
    if not inputs.location.verified:
        caveats.append(
            f"⚠ Port '{inputs.location.name}' has unverified coordinates. "
            "All indicator values are computed at placeholder (0°, 0°) and "
            "are NOT valid until coordinates are confirmed in ports.csv."
        )

    # ── 8. Assemble result ─────────────────────────────────────────────────
    result = build_result(
        inputs=inputs,
        indicators=indicators,
        rules_triggered=rules_triggered,
        rules_disabled=rules_disabled,
        confidence=confidence,
        nearest_argo_km=nearest_argo_km,
        languages_served=inputs.languages,
        caveats=caveats,
    )

    logger.info(
        "FAE advisory: location=%s trip=%s day=%s confidence=%s "
        "indicators_ok=%d/%d rules_triggered=%d",
        inputs.location.name, inputs.trip_type, inputs.day.label,
        confidence.label,
        sum(1 for i in indicators.values()
            if i is not None and getattr(i, "status", "no_data") in ("ok","stale")),
        len(indicators),
        len(rules_triggered),
    )
    return result


# ---------------------------------------------------------------------------
# Internal: parallel indicator computation
# ---------------------------------------------------------------------------

def _compute_indicators_parallel(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    use_synthetic: bool,
) -> Dict[str, Any]:
    """
    Run all 6 indicator functions in parallel.
    A failure in one indicator is caught and stored as a no_data result;
    it never prevents the others from running.
    """
    from fisher_advisory_engine.indicators import (
        sst_indicator,
        current_indicator,
        front_indicator,
        productivity_indicator,
        subsurface_indicator,
        bathymetry_indicator,
        _make_no_data,
    )

    # Front detection uses the mid-point date
    import pandas as pd
    try:
        mid_date = (
            pd.Timestamp(start_date) +
            (pd.Timestamp(end_date) - pd.Timestamp(start_date)) / 2
        ).strftime("%Y-%m-%d")
    except Exception:
        mid_date = start_date

    tasks = {
        "sst":          lambda: sst_indicator(lat, lon, start_date, end_date, use_synthetic),
        "currents":     lambda: current_indicator(lat, lon, start_date, end_date, use_synthetic),
        "thermal_front":lambda: front_indicator(lat, lon, mid_date, use_synthetic),
        "productivity": lambda: productivity_indicator(lat, lon, start_date, end_date, use_synthetic),
        "subsurface":   lambda: subsurface_indicator(lat, lon, start_date, end_date, use_synthetic),
        "bathymetry":   lambda: bathymetry_indicator(lat, lon),
    }

    results: Dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                logger.warning(
                    "Indicator '%s' raised an exception (treated as no_data): %s",
                    name, exc,
                )
                results[name] = _make_no_data(
                    name=name, unit="", source="",
                    resolution="",
                    caveat=f"Indicator computation error: {str(exc)[:200]}",
                )

    return results


# ---------------------------------------------------------------------------
# Internal: build rules context
# ---------------------------------------------------------------------------

def _build_rules_context(
    indicators: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Flatten indicator results into the flat context dict that the rules
    engine expects.  Missing or no_data indicators produce None values.
    """
    def _get(name: str, key: str, default=None):
        ind = indicators.get(name)
        if ind is None or getattr(ind, "status", "no_data") == "no_data":
            return default
        val = getattr(ind, "value", None)
        if isinstance(val, dict):
            return val.get(key, default)
        return default

    return {
        # SST
        "sst_c":          _get("sst", "sst_c"),
        "sst_anomaly_c":  _get("sst", "anomaly_c"),
        "heatwave":       _get("sst", "heatwave"),
        # Currents
        "current_speed_ms":  _get("currents", "speed_ms"),
        "current_dir_deg":   _get("currents", "direction_deg"),
        # Thermal front
        "front_detected":    _get("thermal_front", "detected"),
        "front_strength":    _get("thermal_front", "strength"),
        "front_dist_km":     _get("thermal_front", "dist_km"),
        "front_bearing_deg": _get("thermal_front", "bearing_deg"),
        # Productivity
        "chl_category":      _get("productivity", "category", "no_data"),
        # Subsurface
        "mld_m":             _get("subsurface", "mld_m"),
        # Bathymetry
        "depth_m":           _get("bathymetry", "depth_m"),
    }


# ---------------------------------------------------------------------------
# Internal: nearest Argo distance
# ---------------------------------------------------------------------------

def _extract_nearest_argo_km(indicators: Dict[str, Any]) -> Optional[float]:
    """Extract nearest Argo float distance from the subsurface indicator."""
    sub = indicators.get("subsurface")
    if sub is None:
        return None
    val = getattr(sub, "value", None)
    if isinstance(val, dict):
        return val.get("nearest_km")
    return None
