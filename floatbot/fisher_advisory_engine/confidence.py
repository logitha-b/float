"""
fisher_advisory_engine/confidence.py
--------------------------------------
Computes the FAE confidence level (HIGH / MEDIUM / LOW) from data quality
and spatial coverage metrics.

Degradation factors (worst-case aggregation)
---------------------------------------------
  1. Indicators available  — fewer available indicators → lower confidence
  2. SST data age          — stale SST lowers confidence
  3. Current data age      — stale or model-only currents lower confidence
  4. Chlorophyll gap       — Chl-a unavailable ("no_data") lowers confidence
  5. Argo distance         — nearest float > threshold lowers confidence
  6. SST resolution mismatch — 0.25° grid vs nearshore query lowers confidence
  7. Unverified coordinates — port coords not confirmed lowers confidence

Every factor that degrades confidence is recorded in a reasons list, which
is included verbatim in the output so the user can see exactly why confidence
is what it is.

All numeric thresholds are read from config.yaml (labeled PLACEHOLDER).

Public API
----------
    from fisher_advisory_engine.confidence import compute_confidence
    result = compute_confidence(indicators, inputs, nearest_argo_km)
    result.label    # "HIGH" | "MEDIUM" | "LOW"
    result.score    # 0-100 numeric proxy
    result.reasons  # list[str] — plain-English degradation reasons
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class FaeConfidenceResult:
    label:   str           # "HIGH" | "MEDIUM" | "LOW"
    score:   float         # 0-100 numeric proxy
    reasons: List[str]     # plain-English degradation reasons (empty = no degradation)


# ---------------------------------------------------------------------------
# Factor labels
# ---------------------------------------------------------------------------

_HIGH   = "HIGH"
_MEDIUM = "MEDIUM"
_LOW    = "LOW"

_RANK = {_HIGH: 2, _MEDIUM: 1, _LOW: 0}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_confidence(
    indicators: Dict[str, object],  # name → IndicatorResult (or None)
    inputs,                          # ValidatedInputs
    nearest_argo_km: Optional[float],
) -> FaeConfidenceResult:
    """
    Compute FAE confidence from indicator availability and data quality.

    Parameters
    ----------
    indicators      : dict  name → IndicatorResult (or None if not computed)
    inputs          : ValidatedInputs
    nearest_argo_km : float or None  distance to nearest Argo float

    Returns
    -------
    FaeConfidenceResult
    """
    cfg          = _load_config()
    conf_cfg     = cfg.get("confidence", {})
    reasons:     List[str] = []
    factors:     List[str] = []

    # ── Factor 1: Indicators available ──────────────────────────────────
    n_ok = sum(
        1 for ind in indicators.values()
        if ind is not None and getattr(ind, "status", "no_data") in ("ok", "stale")
    )
    n_total = len(indicators)

    min_high   = int(conf_cfg.get("min_indicators_high",   4))
    min_medium = int(conf_cfg.get("min_indicators_medium", 2))

    if n_ok >= min_high:
        factors.append(_HIGH)
    elif n_ok >= min_medium:
        factors.append(_MEDIUM)
        reasons.append(
            f"Only {n_ok}/{n_total} indicators have data "
            f"(≥{min_high} needed for HIGH confidence)."
        )
    else:
        factors.append(_LOW)
        reasons.append(
            f"Only {n_ok}/{n_total} indicators have data "
            f"(≥{min_medium} needed for MEDIUM confidence). "
            "Results are highly uncertain."
        )

    # ── Factor 2: SST data age ───────────────────────────────────────────
    sst_ind = indicators.get("sst")
    if sst_ind is not None and getattr(sst_ind, "data_age_hours", None) is not None:
        age_sst = float(sst_ind.data_age_hours)
        max_high   = float(conf_cfg.get("max_sst_age_hours_high",   48))
        max_medium = float(conf_cfg.get("max_sst_age_hours_medium", 96))

        if age_sst <= max_high:
            factors.append(_HIGH)
        elif age_sst <= max_medium:
            factors.append(_MEDIUM)
            reasons.append(
                f"SST data is {age_sst:.0f} h old "
                f"(≤{max_high:.0f} h needed for HIGH confidence)."
            )
        else:
            factors.append(_LOW)
            reasons.append(
                f"SST data is {age_sst:.0f} h old — considered stale "
                f"(threshold: {max_medium:.0f} h). "
                "Conditions may have changed."
            )

    # ── Factor 3: Current data age ───────────────────────────────────────
    curr_ind = indicators.get("currents")
    if curr_ind is not None and getattr(curr_ind, "data_age_hours", None) is not None:
        age_curr = float(curr_ind.data_age_hours)
        max_curr = float(conf_cfg.get("max_curr_age_hours_high", 24))

        if age_curr <= max_curr:
            factors.append(_HIGH)
        else:
            factors.append(_MEDIUM)
            reasons.append(
                f"HYCOM current data is {age_curr:.0f} h old "
                f"(model forecast lag; ≤{max_curr:.0f} h for HIGH confidence). "
                "Currents are model output, not observations."
            )

    # ── Factor 4: Chlorophyll gap ────────────────────────────────────────
    chl_ind = indicators.get("productivity")
    if chl_ind is None or getattr(chl_ind, "status", "no_data") == "no_data":
        factors.append(_MEDIUM)
        reasons.append(
            "Chlorophyll-a data unavailable (cloud cover or coastal turbidity). "
            "Productivity indicator not contributing to advisory."
        )

    # ── Factor 5: Argo distance ──────────────────────────────────────────
    if nearest_argo_km is not None:
        max_argo = float(conf_cfg.get("argo_dist_degrade_km", 150))
        if nearest_argo_km <= max_argo:
            factors.append(_HIGH)
        else:
            factors.append(_MEDIUM)
            reasons.append(
                f"Nearest Argo float is {nearest_argo_km:.0f} km away "
                f"(threshold: {max_argo:.0f} km). "
                "Subsurface context is not representative of this location."
            )
    else:
        # No Argo data at all
        factors.append(_MEDIUM)
        reasons.append(
            "No Argo float data found near this location. "
            "Subsurface context is not available."
        )

    # ── Factor 6: Resolution mismatch (nearshore) ────────────────────────
    trip_type = inputs.trip_type
    if trip_type == "nearshore":
        mismatch_thresh = float(
            conf_cfg.get("resolution_mismatch_threshold_deg", 0.5)
        )
        # OISST is 0.25°; for nearshore queries within a small area this
        # is a known limitation.
        factors.append(_MEDIUM)
        reasons.append(
            "OISST 0.25° (~28 km) grid resolution is coarse relative to "
            "nearshore query area. Small-scale coastal features (fronts, "
            "plumes) may not be resolved. "
            "PLACEHOLDER: resolution mismatch threshold not yet calibrated."
        )

    # ── Factor 7: Unverified port coordinates ────────────────────────────
    if not inputs.location.verified:
        factors.append(_LOW)
        reasons.append(
            f"Port '{inputs.location.name}' has unverified coordinates "
            "(verified=false in ports.csv). "
            "All indicator values are computed at the placeholder (0, 0) "
            "location and are MEANINGLESS until coordinates are confirmed."
        )

    # ── Worst-case aggregation ───────────────────────────────────────────
    if not factors:
        worst = _HIGH
    else:
        worst = min(factors, key=lambda f: _RANK.get(f, 0))

    # Numeric score: fraction of HIGH factors × 100
    n_high = sum(1 for f in factors if f == _HIGH)
    score  = round(n_high / len(factors) * 100, 1) if factors else 100.0

    return FaeConfidenceResult(label=worst, score=score, reasons=reasons)


# ---------------------------------------------------------------------------
# Config loader
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
