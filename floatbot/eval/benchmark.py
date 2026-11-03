"""
eval/benchmark.py
-----------------
Feature 4: Benchmark & Evaluation

Establishes ground truth from real Argo profiles, runs narrow single-profile
queries through both a QC-aware system and a naive (QC-blind) baseline, and
reports accuracy for each.

Ground truth methodology
------------------------
Real Argo data does not have a synthetic "true" answer key, so ground truth
is established by using readings that carry the HIGHEST quality signal:
  - DATA_MODE == 'D' (delayed-mode, scientifically verified by the GDAC)
  - TEMP_QC or PSAL_QC in {1} (best-quality flag only)

For each test case we:
  1. Identify a specific float + cycle with delayed-mode good data.
  2. Compute the "truth" value from those delay-mode QC=1 readings.
  3. Run the naive system (uses all readings, ignores QC).
  4. Run the QC-aware system (Feature 1: good flags only, D-mode preferred).
  5. Compare both answers to truth within a defined tolerance.

Tolerance: 0.2 °C for temperature, 0.05 PSU for salinity.

Usage
-----
    python -m eval.benchmark --synthetic          # offline, no network
    python -m eval.benchmark                      # real Argo data
    python -m eval.benchmark --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Add project root to path when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.qc_module import apply_qc_filter, GOOD_FLAGS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
TOLERANCE_TEMP = 0.2   # °C
TOLERANCE_PSAL = 0.05  # PSU


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """A single benchmark question."""
    case_id: str
    description: str
    platform_number: str
    cycle_number: int
    parameter: str          # 'temperature' or 'salinity'
    depth_min: float
    depth_max: float
    # Ground-truth value is computed from the data; set during run
    ground_truth: Optional[float] = None
    naive_answer: Optional[float] = None
    qcaware_answer: Optional[float] = None
    naive_correct: Optional[bool] = None
    qcaware_correct: Optional[bool] = None
    n_total: int = 0
    n_good: int = 0
    n_bad: int = 0
    notes: str = ""


# Test cases drawn from well-known Argo floats with documented delayed-mode data
BENCHMARK_CASES: List[TestCase] = [
    TestCase(
        case_id="TC01",
        description="Surface temperature, North Atlantic float 6902869, cycle 50",
        platform_number="6902869",
        cycle_number=50,
        parameter="temperature",
        depth_min=0,
        depth_max=10,
    ),
    TestCase(
        case_id="TC02",
        description="Full-depth mean temperature, float 6902869, cycle 50",
        platform_number="6902869",
        cycle_number=50,
        parameter="temperature",
        depth_min=0,
        depth_max=2000,
    ),
    TestCase(
        case_id="TC03",
        description="Surface salinity, float 6902869, cycle 50",
        platform_number="6902869",
        cycle_number=50,
        parameter="salinity",
        depth_min=0,
        depth_max=10,
    ),
    TestCase(
        case_id="TC04",
        description="Surface temperature, Pacific float 5906030, cycle 10",
        platform_number="5906030",
        cycle_number=10,
        parameter="temperature",
        depth_min=0,
        depth_max=10,
    ),
    TestCase(
        case_id="TC05",
        description="Full-depth mean salinity, float 5906030, cycle 10",
        platform_number="5906030",
        cycle_number=10,
        parameter="salinity",
        depth_min=0,
        depth_max=2000,
    ),
]

# Synthetic test cases — used in --synthetic mode
SYNTHETIC_CASES: List[TestCase] = [
    TestCase(
        case_id="SYN01",
        description="Synthetic: surface temperature, float 1901000, cycle 3",
        platform_number="1901000",
        cycle_number=3,
        parameter="temperature",
        depth_min=0,
        depth_max=10,
    ),
    TestCase(
        case_id="SYN02",
        description="Synthetic: full-depth temperature, float 1901000, cycle 3",
        platform_number="1901000",
        cycle_number=3,
        parameter="temperature",
        depth_min=0,
        depth_max=2000,
    ),
    TestCase(
        case_id="SYN03",
        description="Synthetic: surface salinity, float 1901001, cycle 5",
        platform_number="1901001",
        cycle_number=5,
        parameter="salinity",
        depth_min=0,
        depth_max=10,
    ),
    TestCase(
        case_id="SYN04",
        description="Synthetic: full-depth salinity, float 1901001, cycle 5",
        platform_number="1901001",
        cycle_number=5,
        parameter="salinity",
        depth_min=0,
        depth_max=2000,
    ),
    TestCase(
        case_id="SYN05",
        description="Synthetic: surface temperature, float 1901002, cycle 8",
        platform_number="1901002",
        cycle_number=8,
        parameter="temperature",
        depth_min=0,
        depth_max=10,
    ),
]


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    cases: List[TestCase],
    data_df: pd.DataFrame,
    synthetic: bool = False,
) -> List[TestCase]:
    """
    Run all test cases against a pre-fetched DataFrame and populate results.

    Parameters
    ----------
    cases    : list of TestCase
    data_df  : pd.DataFrame   Full Argo DataFrame (all floats/cycles needed)
    synthetic: bool           If True, ground truth strategy is adjusted for
                              synthetic data (uses QC=1 D-mode, same as real,
                              but synthetic data encodes these flags explicitly).

    Returns
    -------
    The same list of TestCase, with all result fields populated.
    """
    col_map = {"temperature": "TEMP", "salinity": "PSAL"}
    tol_map = {"temperature": TOLERANCE_TEMP, "salinity": TOLERANCE_PSAL}

    for tc in cases:
        col = col_map[tc.parameter]
        qc_col = f"{col}_QC"

        # Filter to this float + cycle
        df = data_df[
            (data_df["PLATFORM_NUMBER"].astype(str) == str(tc.platform_number)) &
            (data_df["CYCLE_NUMBER"] == tc.cycle_number)
        ].copy()

        if df.empty:
            tc.notes = f"No data found for float {tc.platform_number} cycle {tc.cycle_number}"
            logger.warning(tc.notes)
            continue

        # Apply depth filter
        df = df[(df["PRES"] >= tc.depth_min) & (df["PRES"] <= tc.depth_max)]

        if df.empty:
            tc.notes = f"No data in depth range {tc.depth_min}–{tc.depth_max} m"
            continue

        tc.n_total = len(df)

        # --- Ground truth: delayed-mode QC=1 readings -------------------------
        gt_mask = (df["DATA_MODE"] == "D") & (df[qc_col] == 1)
        gt_df = df[gt_mask]

        if gt_df.empty:
            # Fall back to any QC=1 reading if no D-mode available
            gt_df = df[df[qc_col] == 1]

        if gt_df.empty:
            tc.notes = "No QC=1 readings available to establish ground truth"
            continue

        tc.ground_truth = float(gt_df[col].dropna().mean())

        # --- Naive baseline: all readings, no QC check ------------------------
        naive_vals = df[col].dropna()
        tc.naive_answer = float(naive_vals.mean()) if len(naive_vals) > 0 else None

        # --- QC-aware system (Feature 1) ---------------------------------------
        qc_result = apply_qc_filter(
            df,
            parameter=tc.parameter,
            statistic="mean",
            depth_min=tc.depth_min,
            depth_max=tc.depth_max,
        )
        tc.qcaware_answer = qc_result.value
        tc.n_good = qc_result.n_good
        tc.n_bad = qc_result.n_bad

        # --- Correctness check ------------------------------------------------
        tol = tol_map[tc.parameter]

        if tc.naive_answer is not None and tc.ground_truth is not None:
            tc.naive_correct = abs(tc.naive_answer - tc.ground_truth) <= tol

        if tc.qcaware_answer is not None and tc.ground_truth is not None:
            tc.qcaware_correct = abs(tc.qcaware_answer - tc.ground_truth) <= tol

    return cases


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(cases: List[TestCase]) -> None:
    """Print a formatted benchmark report to stdout."""
    completed = [c for c in cases if c.ground_truth is not None]
    if not completed:
        print("\n[benchmark] No test cases completed — check data availability.\n")
        return

    naive_scores = [c.naive_correct for c in completed if c.naive_correct is not None]
    qc_scores = [c.qcaware_correct for c in completed if c.qcaware_correct is not None]

    naive_acc = sum(naive_scores) / len(naive_scores) * 100 if naive_scores else 0
    qc_acc = sum(qc_scores) / len(qc_scores) * 100 if qc_scores else 0
    delta = qc_acc - naive_acc

    print("\n" + "=" * 72)
    print("  FloatAi Benchmark Results")
    print("=" * 72)
    print(f"  {'Case':<8} {'Parameter':<12} {'Depth':>14}  {'Truth':>7}  "
          f"{'Naive':>7}  {'QC':>7}  {'Naive✓':>7}  {'QC✓':>5}  Notes")
    print("-" * 72)

    for c in cases:
        depth_str = f"{c.depth_min:.0f}–{c.depth_max:.0f} m"
        truth_s = f"{c.ground_truth:.3f}" if c.ground_truth is not None else "—"
        naive_s = f"{c.naive_answer:.3f}" if c.naive_answer is not None else "—"
        qc_s = f"{c.qcaware_answer:.3f}" if c.qcaware_answer is not None else "—"
        naive_ok = ("✓" if c.naive_correct else "✗") if c.naive_correct is not None else "—"
        qc_ok = ("✓" if c.qcaware_correct else "✗") if c.qcaware_correct is not None else "—"
        notes = c.notes[:30] if c.notes else ""
        print(f"  {c.case_id:<8} {c.parameter:<12} {depth_str:>14}  "
              f"{truth_s:>7}  {naive_s:>7}  {qc_s:>7}  {naive_ok:>7}  {qc_ok:>5}  {notes}")

    print("-" * 72)
    print(f"  Naive baseline accuracy  : {naive_acc:.1f}%  ({sum(naive_scores)}/{len(naive_scores)})")
    print(f"  QC-aware accuracy        : {qc_acc:.1f}%  ({sum(qc_scores)}/{len(qc_scores)})")
    print(f"  Improvement delta        : {delta:+.1f} percentage points")
    print("=" * 72 + "\n")


def save_csv(cases: List[TestCase], path: str) -> None:
    """Save full per-question results to a CSV."""
    rows = []
    for c in cases:
        rows.append({
            "case_id": c.case_id,
            "description": c.description,
            "platform_number": c.platform_number,
            "cycle_number": c.cycle_number,
            "parameter": c.parameter,
            "depth_min": c.depth_min,
            "depth_max": c.depth_max,
            "ground_truth": c.ground_truth,
            "naive_answer": c.naive_answer,
            "qcaware_answer": c.qcaware_answer,
            "naive_correct": c.naive_correct,
            "qcaware_correct": c.qcaware_correct,
            "n_total": c.n_total,
            "n_good": c.n_good,
            "n_bad": c.n_bad,
            "notes": c.notes,
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(description="FloatAi accuracy benchmark")
    ap.add_argument("--synthetic", action="store_true",
                    help="Use synthetic data instead of real Argo fetches")
    ap.add_argument("--output", default="eval/benchmark_results.csv",
                    help="Path for CSV output (default: eval/benchmark_results.csv)")
    args = ap.parse_args()

    if args.synthetic:
        cases = SYNTHETIC_CASES
        print("\n[benchmark] Running in SYNTHETIC mode — no network required.\n")
        from data.argo_loader import make_synthetic_profiles
        data_df = make_synthetic_profiles(n_floats=3, cycles_per_float=12, bad_fraction=0.15)
    else:
        cases = BENCHMARK_CASES
        print("\n[benchmark] Fetching real Argo data for benchmark floats …\n")
        data_df = _fetch_real_benchmark_data(cases)

    cases = run_benchmark(cases, data_df, synthetic=args.synthetic)
    print_report(cases)
    save_csv(cases, args.output)


def _fetch_real_benchmark_data(cases: List[TestCase]) -> pd.DataFrame:
    """Fetch real Argo data for all unique floats in the benchmark set."""
    from data.argo_loader import fetch_argo_float

    frames = []
    seen = set()
    for c in cases:
        if c.platform_number in seen:
            continue
        seen.add(c.platform_number)
        print(f"  Fetching float {c.platform_number} …")
        try:
            df = fetch_argo_float(c.platform_number)
            frames.append(df)
        except Exception as exc:
            logger.error("Failed to fetch float %s: %s", c.platform_number, exc)

    if not frames:
        raise RuntimeError(
            "Could not fetch any real Argo data. "
            "Run with --synthetic for offline testing."
        )
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    main()
