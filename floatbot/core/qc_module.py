"""
core/qc_module.py
-----------------
Feature 1: QC-Aware Filtering & Explanation

Accepts a DataFrame with the standard Argo schema (from argo_loader) and:

1. Splits readings into good (QC flag 1 or 2) and bad (QC flag 3 or 4).
2. Within good data, prioritises delayed-mode ('D') over real-time ('R').
3. Computes the requested statistic using only good data.
4. For each excluded reading, identifies which Argo RTQC test likely failed
   and generates a plain-language explanation.
5. Computes a confidence score and label (High / Moderate / Low).
6. Returns full provenance (float ID, cycle, depth) for a sample of used readings.
7. Supports optional depth filtering (depth_min / depth_max).

Public API
----------
    result = apply_qc_filter(df, parameter="temperature",
                              depth_min=0, depth_max=10)
    # result is a QCResult dataclass
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Argo QC flag semantics
# ---------------------------------------------------------------------------
GOOD_FLAGS = {1, 2}       # 1=good, 2=probably good
BAD_FLAGS = {3, 4}         # 3=probably bad, 4=bad
MISSING_FLAG = 9

# Confidence thresholds (% of readings passing QC)
HIGH_THRESHOLD = 90.0
MODERATE_THRESHOLD = 70.0

# Physical range limits for the lightweight local RTQC re-implementation
TEMP_RANGE = (-2.5, 40.0)   # °C (global ocean range)
PSAL_RANGE = (2.0, 41.0)    # PSU (global ocean range)

# Spike / gradient thresholds (approximate Argo RTQC values)
TEMP_SPIKE_THRESHOLD = 2.0   # °C — jump vs. interpolated neighbors
PSAL_SPIKE_THRESHOLD = 0.3   # PSU
TEMP_GRADIENT_THRESHOLD = 9.0  # °C per 500 dbar
PSAL_GRADIENT_THRESHOLD = 1.5  # PSU per 500 dbar

# Plain-language test descriptions
_TEST_DESCRIPTIONS = {
    "spike": (
        "the value jumped implausibly compared to nearby depths, "
        "suggesting a sensor glitch rather than a real change"
    ),
    "gradient": (
        "the value changed too sharply between adjacent depths, "
        "which usually indicates a faulty reading"
    ),
    "range": (
        "the value fell outside the physically possible range for ocean "
        "{parameter}, indicating a sensor malfunction"
    ),
    "density_inversion": (
        "the reading would imply denser water sitting above lighter water, "
        "which is physically unstable and usually signals a sensor error"
    ),
    "unknown": (
        "this reading was flagged as suspect by the Argo GDAC quality-control "
        "system; the specific test was not recorded in the available metadata"
    ),
}

MAX_PROVENANCE_SAMPLES = 5


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class ExclusionRecord:
    platform: str
    cycle: int
    pres: float
    parameter: str
    value: float
    qc_flag: int
    failed_test: str          # 'spike', 'gradient', 'range', 'density_inversion', 'unknown'
    explanation: str          # plain-language sentence


@dataclass
class ProvenanceRecord:
    platform: str
    cycle: int
    pres: float
    value: float
    data_mode: str            # 'R' or 'D'


@dataclass
class QCResult:
    parameter: str            # 'temperature' or 'salinity'
    statistic: str            # e.g. 'mean'
    value: Optional[float]    # computed result, or None if no good data
    unit: str                 # '°C' or 'PSU'

    n_total: int
    n_good: int
    n_bad: int

    confidence_score: float   # 0–100 %
    confidence_label: str     # 'High', 'Moderate', 'Low'

    depth_min: float
    depth_max: float
    depth_note: str           # plain-language depth description for the answer

    exclusions: List[ExclusionRecord] = field(default_factory=list)
    provenance: List[ProvenanceRecord] = field(default_factory=list)

    has_delayed_mode: bool = False   # True if any used reading is DATA_MODE='D'
    n_delayed: int = 0
    n_realtime: int = 0

    def confidence_badge(self) -> str:
        """Return emoji + label for the Streamlit UI."""
        icons = {"High": "🟢", "Moderate": "🟡", "Low": "🔴"}
        return f"{icons.get(self.confidence_label, '⚪')} {self.confidence_label}"

    def to_answer_text(self) -> str:
        """Compose a human-readable answer sentence."""
        if self.value is None:
            return (
                f"No valid {self.parameter} readings were found "
                f"in the requested region and depth range ({self.depth_note}). "
                f"All {self.n_total} readings failed quality control."
            )
        mode_note = ""
        if self.has_delayed_mode:
            mode_note = (
                f" ({self.n_delayed} of {self.n_good} used readings are "
                "delayed-mode / scientifically verified)"
            )
        return (
            f"The {self.statistic} {self.parameter} was "
            f"**{self.value:.2f} {self.unit}** "
            f"({self.depth_note}), based on {self.n_good} quality-controlled "
            f"readings out of {self.n_total} total.{mode_note}"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_qc_filter(
    df: pd.DataFrame,
    parameter: str = "temperature",
    statistic: str = "mean",
    depth_min: float = 0.0,
    depth_max: float = 2000.0,
) -> QCResult:
    """
    Apply QC filtering to an Argo DataFrame and return a QCResult.

    Parameters
    ----------
    df         : pd.DataFrame   Standard Argo schema from argo_loader.
    parameter  : str            'temperature' or 'salinity'.
    statistic  : str            'mean', 'min', 'max', or 'median'.
    depth_min  : float          Lower pressure bound (dbar).  Default 0.
    depth_max  : float          Upper pressure bound (dbar).  Default 2000.
    """
    parameter = parameter.lower().strip()
    if parameter not in ("temperature", "salinity"):
        raise ValueError(f"parameter must be 'temperature' or 'salinity', got '{parameter}'")

    col = "TEMP" if parameter == "temperature" else "PSAL"
    qc_col = f"{col}_QC"
    unit = "°C" if parameter == "temperature" else "PSU"

    # --- 1. Depth filter -------------------------------------------------------
    df_depth = df[
        (df["PRES"] >= depth_min) & (df["PRES"] <= depth_max)
    ].copy()

    depth_note = _depth_note(depth_min, depth_max)
    n_total = len(df_depth)

    if n_total == 0:
        return _empty_result(parameter, statistic, unit, depth_min, depth_max, depth_note)

    # --- 2. Split good / bad ---------------------------------------------------
    good_mask = df_depth[qc_col].isin(GOOD_FLAGS)
    bad_mask = df_depth[qc_col].isin(BAD_FLAGS)

    df_good = df_depth[good_mask].copy()
    df_bad = df_depth[bad_mask].copy()

    n_good = len(df_good)
    n_bad = len(df_bad)

    # --- 3. Within good data, prioritise delayed-mode -------------------------
    if not df_good.empty:
        delayed = df_good[df_good["DATA_MODE"] == "D"]
        realtime = df_good[df_good["DATA_MODE"] != "D"]
        # If there are any delayed-mode readings, prefer those; otherwise use all good
        if not delayed.empty:
            df_used = delayed
        else:
            df_used = df_good
        n_delayed = len(delayed)
        n_realtime = len(realtime)
        has_delayed = n_delayed > 0
    else:
        df_used = df_good
        n_delayed = 0
        n_realtime = 0
        has_delayed = False

    # --- 4. Compute statistic --------------------------------------------------
    values = df_used[col].dropna()
    computed_value: Optional[float]
    if len(values) == 0:
        computed_value = None
    else:
        ops = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
        }
        computed_value = ops.get(statistic, float(values.mean()))

    # --- 5. Confidence score ---------------------------------------------------
    confidence_score = (n_good / n_total * 100) if n_total > 0 else 0.0
    confidence_label = _confidence_label(confidence_score)

    # --- 6. Explain exclusions -------------------------------------------------
    exclusions = _explain_exclusions(df_bad, df_depth, col, parameter)

    # --- 7. Provenance sample --------------------------------------------------
    provenance = _build_provenance(df_used, col, MAX_PROVENANCE_SAMPLES)

    return QCResult(
        parameter=parameter,
        statistic=statistic,
        value=computed_value,
        unit=unit,
        n_total=n_total,
        n_good=n_good,
        n_bad=n_bad,
        confidence_score=round(confidence_score, 1),
        confidence_label=confidence_label,
        depth_min=depth_min,
        depth_max=depth_max,
        depth_note=depth_note,
        exclusions=exclusions,
        provenance=provenance,
        has_delayed_mode=has_delayed,
        n_delayed=n_delayed,
        n_realtime=n_realtime,
    )


# ---------------------------------------------------------------------------
# QC test approximation
# ---------------------------------------------------------------------------

def _identify_failed_test(
    row: pd.Series,
    df_profile: pd.DataFrame,
    col: str,
    parameter: str,
) -> Tuple[str, str]:
    """
    Approximate which Argo RTQC test a bad reading likely failed.

    Returns (test_name, plain_language_explanation).
    The official flag is never overridden — this is purely for explaining it.
    """
    value = row[col]

    # Range test
    lo, hi = (TEMP_RANGE if parameter == "temperature" else PSAL_RANGE)
    if pd.isna(value) or value < lo or value > hi:
        return "range", _TEST_DESCRIPTIONS["range"].format(parameter=parameter)

    # Sort the profile by pressure for neighbor comparisons
    profile_sorted = df_profile.sort_values("PRES")
    pres_vals = profile_sorted["PRES"].values
    col_vals = profile_sorted[col].values

    idx_arr = np.where(np.isclose(pres_vals, row["PRES"]))[0]
    if len(idx_arr) == 0:
        return "unknown", _TEST_DESCRIPTIONS["unknown"]
    idx = idx_arr[0]

    # Spike test: compare against linear interpolation of neighbors
    if 0 < idx < len(pres_vals) - 1:
        v_prev = col_vals[idx - 1]
        v_next = col_vals[idx + 1]
        if not (np.isnan(v_prev) or np.isnan(v_next)):
            interp = (v_prev + v_next) / 2.0
            spike_thresh = TEMP_SPIKE_THRESHOLD if parameter == "temperature" else PSAL_SPIKE_THRESHOLD
            if abs(value - interp) > spike_thresh:
                return "spike", _TEST_DESCRIPTIONS["spike"]

    # Gradient test: change over 500 dbar
    if idx > 0:
        dp = abs(row["PRES"] - pres_vals[idx - 1])
        if dp > 0:
            grad = abs(value - col_vals[idx - 1]) / dp * 500
            grad_thresh = (TEMP_GRADIENT_THRESHOLD if parameter == "temperature"
                           else PSAL_GRADIENT_THRESHOLD)
            if grad > grad_thresh:
                return "gradient", _TEST_DESCRIPTIONS["gradient"]

    # Density inversion (temperature only — a rough proxy)
    if parameter == "temperature" and idx > 0:
        if value > col_vals[idx - 1] + 0.5:   # temp increasing with depth
            return "density_inversion", _TEST_DESCRIPTIONS["density_inversion"]

    return "unknown", _TEST_DESCRIPTIONS["unknown"]


def _explain_exclusions(
    df_bad: pd.DataFrame,
    df_full: pd.DataFrame,
    col: str,
    parameter: str,
) -> List[ExclusionRecord]:
    """Build ExclusionRecord list for all bad readings."""
    records: List[ExclusionRecord] = []
    for _, row in df_bad.iterrows():
        if pd.isna(row.get(col)):
            continue
        test, explanation = _identify_failed_test(row, df_full, col, parameter)
        records.append(ExclusionRecord(
            platform=str(row["PLATFORM_NUMBER"]),
            cycle=int(row["CYCLE_NUMBER"]),
            pres=float(row["PRES"]),
            parameter=parameter,
            value=float(row[col]),
            qc_flag=int(row[f"{col}_QC"]),
            failed_test=test,
            explanation=explanation,
        ))
    return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_provenance(
    df: pd.DataFrame,
    col: str,
    n: int = MAX_PROVENANCE_SAMPLES,
) -> List[ProvenanceRecord]:
    sample = df.head(n)
    records = []
    for _, row in sample.iterrows():
        records.append(ProvenanceRecord(
            platform=str(row["PLATFORM_NUMBER"]),
            cycle=int(row["CYCLE_NUMBER"]),
            pres=float(row["PRES"]),
            value=float(row[col]) if not pd.isna(row[col]) else float("nan"),
            data_mode=str(row.get("DATA_MODE", "R")),
        ))
    return records


def _confidence_label(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "High"
    if score >= MODERATE_THRESHOLD:
        return "Moderate"
    return "Low"


def _depth_note(depth_min: float, depth_max: float) -> str:
    if depth_min == 0 and depth_max <= 10:
        return f"surface only, 0–{depth_max:.0f} m"
    if depth_min == 0:
        return f"0–{depth_max:.0f} m depth"
    return f"{depth_min:.0f}–{depth_max:.0f} m depth"


def _empty_result(
    parameter: str, statistic: str, unit: str,
    depth_min: float, depth_max: float, depth_note: str,
) -> QCResult:
    return QCResult(
        parameter=parameter,
        statistic=statistic,
        value=None,
        unit=unit,
        n_total=0,
        n_good=0,
        n_bad=0,
        confidence_score=0.0,
        confidence_label="Low",
        depth_min=depth_min,
        depth_max=depth_max,
        depth_note=depth_note,
    )
