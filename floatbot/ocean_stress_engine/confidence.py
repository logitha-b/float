"""
ocean_stress_engine/confidence.py
-----------------------------------
Confidence level computation for the Ocean Stress Engine.

Confidence is degraded by three independent factors:
  1. Profile count   — fewer floats → less representativeness
  2. QC exclusion fraction — high fraction of bad readings → unreliable obs
  3. Spatial coverage — distance of nearest float from region centre

The worst-case factor determines the final label: HIGH / MEDIUM / LOW.
Thresholds are read from config.yaml (never hard-coded here).

Public API
----------
    from ocean_stress_engine.confidence import compute_confidence
    conf = compute_confidence(
        n_profiles=8,
        n_readings_total=1200,
        n_readings_excluded=180,
        nearest_float_dist_deg=2.5,
        variables_missing=["oxygen","ph"],
    )
    conf.label          # "MEDIUM"
    conf.reasons        # list of plain-text explanations
    conf.score          # 0-100 numeric (for future use)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceResult:
    label:   str          # "HIGH" | "MEDIUM" | "LOW"
    score:   float        # 0-100 numeric proxy
    reasons: List[str]    # plain-language explanations for each degradation


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_confidence(
    n_profiles:              int,
    n_readings_total:        int,
    n_readings_excluded:     int,
    nearest_float_dist_deg:  float,
    variables_missing:       Optional[List[str]] = None,
) -> ConfidenceResult:
    """
    Compute a confidence label from observation quality metrics.

    Parameters
    ----------
    n_profiles             : int    Number of distinct Argo float profiles used.
    n_readings_total       : int    Total QC-depth-filtered readings before exclusion.
    n_readings_excluded    : int    Readings removed by QC filters.
    nearest_float_dist_deg : float  Great-circle distance in degrees from the
                                    nearest float position to the region centre.
    variables_missing      : list   Variables absent from the score.
    """
    cfg    = _load_config().get("confidence", {})
    reasons: List[str] = []
    factors: List[str] = []  # "HIGH" / "MEDIUM" / "LOW" per factor

    # ── Factor 1: Profile count ────────────────────────────────────────────
    min_high   = int(cfg.get("min_profiles_high",   5))
    min_medium = int(cfg.get("min_profiles_medium", 2))

    if n_profiles >= min_high:
        factors.append("HIGH")
    elif n_profiles >= min_medium:
        factors.append("MEDIUM")
        reasons.append(
            f"Profile count is low ({n_profiles} profiles; "
            f"≥{min_high} needed for HIGH confidence)."
        )
    else:
        factors.append("LOW")
        reasons.append(
            f"Very few profiles ({n_profiles}; "
            f"≥{min_medium} needed for MEDIUM confidence). "
            "Results may not represent the full region."
        )

    # ── Factor 2: QC exclusion fraction ───────────────────────────────────
    max_excl_high   = float(cfg.get("max_exclusion_fraction_high",   0.15))
    max_excl_medium = float(cfg.get("max_exclusion_fraction_medium", 0.40))

    excl_frac = (n_readings_excluded / n_readings_total
                 if n_readings_total > 0 else 0.0)

    if excl_frac <= max_excl_high:
        factors.append("HIGH")
    elif excl_frac <= max_excl_medium:
        factors.append("MEDIUM")
        reasons.append(
            f"QC exclusion fraction is elevated "
            f"({excl_frac:.0%} of readings failed QC; "
            f"≤{max_excl_high:.0%} needed for HIGH confidence)."
        )
    else:
        factors.append("LOW")
        reasons.append(
            f"High QC exclusion fraction ({excl_frac:.0%} of readings failed QC). "
            "Observed values may not be representative."
        )

    # ── Factor 3: Spatial coverage (nearest float distance) ───────────────
    max_dist_high   = float(cfg.get("max_distance_high_deg",   3.0))
    max_dist_medium = float(cfg.get("max_distance_medium_deg", 7.0))

    if nearest_float_dist_deg <= max_dist_high:
        factors.append("HIGH")
    elif nearest_float_dist_deg <= max_dist_medium:
        factors.append("MEDIUM")
        reasons.append(
            f"Nearest float is {nearest_float_dist_deg:.1f}° from region centre "
            f"(≤{max_dist_high:.1f}° needed for HIGH confidence)."
        )
    else:
        factors.append("LOW")
        reasons.append(
            f"Nearest float is {nearest_float_dist_deg:.1f}° from region centre — "
            "sparse spatial coverage. Results may not reflect conditions "
            "throughout the region."
        )

    # ── Factor 4: Missing variables ────────────────────────────────────────
    mv = variables_missing or []
    if mv:
        reasons.append(
            f"Missing variables: {', '.join(mv)}. "
            "Weights re-normalised; score reflects available variables only."
        )
        # Missing variables degrade confidence by one step (cap at MEDIUM if HIGH)
        factors.append("MEDIUM")

    # ── Worst-case aggregation ─────────────────────────────────────────────
    rank = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
    worst = min(factors, key=lambda f: rank[f])

    # Numeric score: proportion of factors that are HIGH * 100
    n_high = sum(1 for f in factors if f == "HIGH")
    score  = round(n_high / len(factors) * 100, 1) if factors else 0.0

    return ConfidenceResult(label=worst, score=score, reasons=reasons)


def nearest_float_distance(
    lat_centre: float,
    lon_centre: float,
    float_lats,     # array-like of float latitudes
    float_lons,     # array-like of float longitudes
) -> float:
    """
    Return the distance in degrees (Euclidean approximation) from the
    region centre to the nearest float.  Returns a large value if no floats.
    """
    import numpy as np
    lats = np.asarray(float_lats, dtype=float)
    lons = np.asarray(float_lons, dtype=float)
    if len(lats) == 0:
        return 999.0
    dists = np.sqrt((lats - lat_centre) ** 2 + (lons - lon_centre) ** 2)
    return float(np.min(dists))


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
