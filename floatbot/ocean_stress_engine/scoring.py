"""
ocean_stress_engine/scoring.py
--------------------------------
Scoring pipeline for the Ocean Stress Engine.

Pipeline (per call to compute_scores()):
  1. For each available variable, compute the z-score anomaly:
         z = (observed - baseline_mean) / baseline_std
     Stress is directional: both warm AND cold anomalies drive stress,
     so we use |z| for the sub-score mapping (the sign is preserved in
     the driver record for explainability).

  2. Map |z| to a 0-100 sub-score via a piecewise-linear function whose
     breakpoints are read from config.yaml (never hard-coded here).

  3. Re-normalise weights across available variables only (missing-data
     rule: if a variable has no observed value OR no baseline std, its
     weight is redistributed proportionally to the remaining variables).

  4. Compute the composite stress score:
         stress_score = sum(sub_score_i * normalised_weight_i)

  5. Assign a category label from the config thresholds.

Public API
----------
    from ocean_stress_engine.scoring import compute_scores, VariableDriver
    result = compute_scores(
        observed={"temperature": 29.8, "salinity": 36.5},
        baseline=baseline_stats,           # BaselineStats from climatology.py
        raw_weights={"temperature": 0.7, "salinity": 0.3},
        breakpoints=[[0,0],[1,25],[2,50],[3,75],[4,100]],
    )
    result.stress_score    # 0-100 float
    result.category        # "LOW"|"MODERATE"|"HIGH"|"SEVERE"
    result.drivers         # list[VariableDriver]
    result.variables_used
    result.variables_missing
    result.caveats
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Variables the OSE knows about. Oxygen and pH are optional (BGC-Argo).
_ALL_VARIABLES = ("temperature", "salinity", "sst", "oxygen", "ph")

# Mapping from variable name to BaselineStats attribute names
_BASELINE_ATTRS: Dict[str, Tuple[str, str]] = {
    "temperature": ("temperature_mean", "temperature_std"),
    "salinity":    ("salinity_mean",    "salinity_std"),
    "sst":         ("sst_mean",         "sst_std"),
    # oxygen and ph: future BGC-Argo extension — not in BaselineStats v1
}

_UNITS = {
    "temperature": "°C",
    "salinity":    "PSU",
    "sst":         "°C",
    "oxygen":      "µmol/kg",
    "ph":          "(dimensionless)",
}


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass
class VariableDriver:
    variable:       str
    sub_score:      float           # 0-100
    observed:       float           # measured value
    baseline_mean:  float           # climatological mean
    baseline_std:   float           # climatological std
    anomaly:        float           # observed - baseline_mean  (signed)
    anomaly_zscore: float           # (observed - baseline_mean) / std  (signed)
    weight:         float           # normalised weight used in composite
    unit:           str


@dataclass
class ScoringResult:
    stress_score:      float                 # 0-100 composite
    category:          str                   # LOW / MODERATE / HIGH / SEVERE
    drivers:           List[VariableDriver]
    variables_used:    List[str]
    variables_missing: List[str]
    caveats:           List[str]             # weight redistribution notes etc.
    raw_weights:       Dict[str, float]      # from config (before normalisation)
    effective_weights: Dict[str, float]      # after normalisation


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_scores(
    observed:   Dict[str, Optional[float]],   # variable → measured value (None = no data)
    baseline,                                  # BaselineStats
    raw_weights: Dict[str, float],
    breakpoints: List[List[float]],
) -> ScoringResult:
    """
    Compute per-variable sub-scores and the composite thermohaline stress score.

    Parameters
    ----------
    observed    : dict  variable → observed mean value (None if unavailable)
    baseline    : BaselineStats
    raw_weights : dict  variable → weight (from config, sum need not be 1.0)
    breakpoints : list  [[z_score, sub_score], …] from config, monotone increasing
    """
    caveats:           List[str] = []
    drivers:           List[VariableDriver] = []
    variables_used:    List[str] = []
    variables_missing: List[str] = []

    # ------------------------------------------------------------------
    # 1. Determine which variables are computable
    # ------------------------------------------------------------------
    eligible: Dict[str, Tuple[float, float, float]] = {}  # var → (obs, mean, std)

    for var in _ALL_VARIABLES:
        obs = observed.get(var)
        mean_attr, std_attr = _BASELINE_ATTRS.get(var, (None, None))

        if mean_attr is None:
            # BGC variable not yet supported in BaselineStats v1.
            # Always listed as missing so the caller knows it was considered.
            variables_missing.append(var)
            if raw_weights.get(var, 0) > 0:
                caveats.append(
                    f"{var}: BGC-Argo variable — baseline not available in v1. "
                    f"Weight ({raw_weights[var]:.2f}) redistributed."
                )
            continue

        baseline_mean = getattr(baseline, mean_attr, None)
        baseline_std  = getattr(baseline, std_attr,  None)

        if obs is None:
            variables_missing.append(var)
            if raw_weights.get(var, 0) > 0:
                caveats.append(
                    f"{var}: no observed value available. "
                    f"Weight ({raw_weights[var]:.2f}) redistributed."
                )
            continue

        if baseline_mean is None:
            variables_missing.append(var)
            caveats.append(
                f"{var}: baseline mean unavailable. "
                f"Weight ({raw_weights[var]:.2f}) redistributed."
            )
            continue

        if baseline_std is None or baseline_std == 0.0:
            variables_missing.append(var)
            caveats.append(
                f"{var}: baseline std is zero or unavailable "
                f"(too few historical readings). "
                f"Weight ({raw_weights[var]:.2f}) redistributed."
            )
            continue

        eligible[var] = (float(obs), float(baseline_mean), float(baseline_std))

    if not eligible:
        return ScoringResult(
            stress_score=0.0,
            category="LOW",
            drivers=[],
            variables_used=[],
            variables_missing=list(variables_missing),
            caveats=caveats + ["No eligible variables — stress score is indeterminate."],
            raw_weights=raw_weights,
            effective_weights={},
        )

    # ------------------------------------------------------------------
    # 2. Re-normalise weights across eligible variables only
    # ------------------------------------------------------------------
    effective_weights = _renormalise_weights(raw_weights, set(eligible.keys()))

    # ------------------------------------------------------------------
    # 3. Compute per-variable sub-scores and build driver records
    # ------------------------------------------------------------------
    composite = 0.0
    for var, (obs, mean, std) in eligible.items():
        anomaly = obs - mean
        z       = anomaly / std          # signed z-score
        abs_z   = abs(z)
        sub     = _zscore_to_subscore(abs_z, breakpoints)
        w       = effective_weights[var]

        composite += sub * w
        variables_used.append(var)

        drivers.append(VariableDriver(
            variable=var,
            sub_score=round(sub, 2),
            observed=round(obs, 4),
            baseline_mean=round(mean, 4),
            baseline_std=round(std, 4),
            anomaly=round(anomaly, 4),
            anomaly_zscore=round(z, 4),
            weight=round(w, 4),
            unit=_UNITS.get(var, ""),
        ))

    # Sort drivers by sub_score descending (main driver first)
    drivers.sort(key=lambda d: d.sub_score, reverse=True)

    # Add baseline source caveat if present
    if baseline.caveat:
        caveats.append(f"Baseline caveat: {baseline.caveat}")

    stress_score = round(float(np.clip(composite, 0, 100)), 2)
    category     = _score_to_category(stress_score)

    return ScoringResult(
        stress_score=stress_score,
        category=category,
        drivers=drivers,
        variables_used=variables_used,
        variables_missing=variables_missing,
        caveats=caveats,
        raw_weights=raw_weights,
        effective_weights={k: round(v, 4) for k, v in effective_weights.items()},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _renormalise_weights(
    raw_weights: Dict[str, float],
    available: set,
) -> Dict[str, float]:
    """
    Keep only weights for available variables and normalise to sum=1.0.
    If the total available weight is 0, assign equal weight to all available.
    """
    subset = {v: raw_weights.get(v, 0.0) for v in available}
    total  = sum(subset.values())

    if total == 0.0:
        # Equal fallback
        logger.warning("All raw weights for available variables are 0; using equal weights.")
        n = len(available)
        return {v: 1.0 / n for v in available}

    return {v: w / total for v, w in subset.items()}


def _zscore_to_subscore(abs_z: float, breakpoints: List[List[float]]) -> float:
    """
    Piecewise-linear interpolation from |z-score| to 0-100 sub-score.

    breakpoints: [[z0, s0], [z1, s1], ...] monotonically increasing z values.
    """
    if not breakpoints or len(breakpoints) < 2:
        raise ValueError("breakpoints must have at least 2 entries.")

    bps = sorted(breakpoints, key=lambda x: x[0])

    # Below first breakpoint
    if abs_z <= bps[0][0]:
        return float(bps[0][1])

    # Above last breakpoint — clamp to 100
    if abs_z >= bps[-1][0]:
        return 100.0

    # Find the interpolation segment
    for i in range(len(bps) - 1):
        z0, s0 = bps[i]
        z1, s1 = bps[i + 1]
        if z0 <= abs_z <= z1:
            t = (abs_z - z0) / (z1 - z0)
            return float(s0 + t * (s1 - s0))

    return 100.0   # safety fallback


def _score_to_category(score: float) -> str:
    """Map 0-100 score to LOW / MODERATE / HIGH / SEVERE using config thresholds."""
    cfg = _load_config()
    cats = cfg.get("categories", {
        "LOW":      [0,  25],
        "MODERATE": [25, 50],
        "HIGH":     [50, 75],
        "SEVERE":   [75, 100],
    })
    # Ordered from most severe so a score of exactly 75 → HIGH, not SEVERE
    order = ["SEVERE", "HIGH", "MODERATE", "LOW"]
    for label in order:
        lo, hi = cats.get(label, [0, 0])
        if score >= lo:
            return label
    return "LOW"


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


def load_weights_and_breakpoints(ecosystem_type: str) -> Tuple[Dict[str, float], List[List[float]]]:
    """
    Load ecosystem weights and anomaly→score breakpoints from config.
    Called by engine.py so the engine never touches config directly.
    """
    cfg = _load_config()
    weights = cfg.get("ecosystem_weights", {}).get(ecosystem_type, {})
    if not weights:
        raise ValueError(
            f"No weights found for ecosystem_type='{ecosystem_type}' in config.yaml."
        )
    bps_raw = cfg.get("anomaly_to_score", {}).get("breakpoints", [])
    breakpoints = [[float(z), float(s)] for z, s in bps_raw]
    return dict(weights), breakpoints
