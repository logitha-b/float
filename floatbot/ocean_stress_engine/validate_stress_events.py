"""
ocean_stress_engine/validate_stress_events.py
-----------------------------------------------
Real-data validation script for the Ocean Stress Engine.

Tests whether the OSE produces plausible scores during known marine
heatwave / coral bleaching events, and low scores during quiet periods.

IMPORTANT SEPARATION:
  - This script uses REAL Argo data from the live GDAC.
  - Unit tests (tests/) use ONLY synthetic/mock data.
  - Run this script separately; it requires a network connection.

Known Events Tested
-------------------
  Event 1 — 2016 Global Coral Bleaching (Arabian Sea / Indian Ocean)
    The 2016 El Niño-driven event is one of the most severe coral bleaching
    events on record. SST in the Arabian Sea / Indian Ocean were anomalously
    high from early 2016 through mid-2016.
    Reference: Hughes et al. (2017), Nature 543, 373–377.
               https://doi.org/10.1038/nature21707

  Event 2 — 2020 Marine Heatwave (Coral Sea)
    A significant marine heatwave affected the Coral Sea and Great Barrier
    Reef in early 2020, causing widespread bleaching.
    Reference: GBRMPA (2020) Annual Report; Skirving et al. (2021).

  Control — 2012 Quiet Period (Arabian Sea)
    2012 was not marked by any significant pan-Indian-Ocean thermal event.
    Used as a reference "low stress" year.

Interpretation Rules (stated explicitly per requirements)
---------------------------------------------------------
  The script does NOT claim significance — it reports what the data show.
  If profile counts fall below a threshold, a SPARSE warning is printed
  and no conclusion is drawn for that event.

  A result is considered "consistent with elevated stress" if:
    stress_score >= 40  AND  confidence in ("HIGH", "MEDIUM")

  A result is considered "consistent with low stress" if:
    stress_score < 40   AND  confidence in ("HIGH", "MEDIUM")

  If confidence == "LOW" or n_profiles < 3, the result is flagged SPARSE
  and neither conclusion is drawn.

Run
---
    cd FloatAi/
    python ocean_stress_engine/validate_stress_events.py

    # Verbose (show full JSON for each event):
    python ocean_stress_engine/validate_stress_events.py --verbose

    # Save comparison table to CSV:
    python ocean_stress_engine/validate_stress_events.py --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import os
import time

# Ensure FloatAi/ is on the path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ocean_stress_engine.engine import compute_stress

# ---------------------------------------------------------------------------
# Event definitions
# ---------------------------------------------------------------------------

EVENTS = [
    # ── Elevated-stress events (expect score >= 40) ──────────────────────
    {
        "id":          "2016_arabian_sea_bleaching",
        "label":       "2016 Global Bleaching — Arabian Sea (Mar–May)",
        "region":      "arabian sea",
        "time_window": "2016-04-01/2016-05-31",
        "depth_layer": "surface",
        "ecosystem":   "coral_reef",
        "expect":      "elevated",          # "elevated" or "low"
        "reference":   "Hughes et al. (2017) Nature 543:373. doi:10.1038/nature21707",
    },
    {
        "id":          "2016_indian_ocean_bleaching",
        "label":       "2016 Global Bleaching — Indian Ocean (Feb–Apr)",
        "region":      "indian ocean",
        "time_window": "2016-02-01/2016-04-30",
        "depth_layer": "surface",
        "ecosystem":   "coral_reef",
        "expect":      "elevated",
        "reference":   "Hughes et al. (2017) Nature 543:373.",
    },
    {
        "id":          "2020_coral_sea_heatwave",
        "label":       "2020 Coral Sea Marine Heatwave (Jan–Mar)",
        "region":      "coral sea",
        "time_window": "2020-01-01/2020-03-31",
        "depth_layer": "surface",
        "ecosystem":   "coral_reef",
        "expect":      "elevated",
        "reference":   "GBRMPA Annual Report 2020; Skirving et al. (2021).",
    },
    {
        "id":          "2020_arabian_sea_heatwave",
        "label":       "2020 Arabian Sea Warming (Apr–Jun)",
        "region":      "arabian sea",
        "time_window": "2020-04-01/2020-06-30",
        "depth_layer": "surface",
        "ecosystem":   "coral_reef",
        "expect":      "elevated",
        "reference":   "Chowdary et al. (2022) Ocean Dynamics.",
    },
    # ── Low-stress control periods ────────────────────────────────────────
    {
        "id":          "2012_arabian_sea_control",
        "label":       "2012 Arabian Sea — Control (Quiet Year)",
        "region":      "arabian sea",
        "time_window": "2012-04-01/2012-06-30",
        "depth_layer": "surface",
        "ecosystem":   "coral_reef",
        "expect":      "low",
        "reference":   "No major thermal event recorded (NOAA CRW archive).",
    },
    {
        "id":          "2013_coral_sea_control",
        "label":       "2013 Coral Sea — Control (Quiet Year)",
        "region":      "coral sea",
        "time_window": "2013-01-01/2013-03-31",
        "depth_layer": "surface",
        "ecosystem":   "coral_reef",
        "expect":      "low",
        "reference":   "No major thermal event recorded (NOAA CRW archive).",
    },
]

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

ELEVATED_THRESHOLD  = 40.0   # score >= this → consistent with elevated stress
SPARSE_MIN_PROFILES = 3      # fewer profiles → flag as SPARSE, no conclusion drawn


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_validation(verbose: bool = False, csv_path: str | None = None):
    print("=" * 72)
    print("  Ocean Stress Engine — Real-Data Validation")
    print("  Known stress events: 2016 bleaching, 2020 heatwave")
    print("  Control periods:     2012 Arabian Sea, 2013 Coral Sea")
    print(f"  Elevated-stress threshold: score >= {ELEVATED_THRESHOLD}")
    print(f"  Sparse data threshold:     n_profiles < {SPARSE_MIN_PROFILES}")
    print("=" * 72)
    print()

    rows = []

    for ev in EVENTS:
        print(f"  Running: {ev['label']}")
        print(f"           {ev['region']} | {ev['time_window']} | "
              f"{ev['depth_layer']} | {ev['ecosystem']}")

        t0 = time.time()
        result = error_msg = None
        try:
            result = compute_stress(
                region=ev["region"],
                time_window=ev["time_window"],
                depth_layer=ev["depth_layer"],
                ecosystem_type=ev["ecosystem"],
                baseline_period="2005-2020",
            )
        except Exception as exc:
            error_msg = str(exc)

        elapsed = time.time() - t0

        # ── Evaluate result ────────────────────────────────────────────────
        if error_msg:
            status    = "ERROR"
            conclusion = f"fetch failed: {error_msg[:120]}"
            score_str  = "n/a"
            n_prof     = 0
            conf       = "n/a"
        else:
            score_str = f"{result.stress_score:.1f}"
            n_prof    = result.n_profiles
            conf      = result.confidence

            is_sparse = (n_prof < SPARSE_MIN_PROFILES or conf == "LOW")

            if is_sparse:
                status     = "SPARSE"
                conclusion = (
                    f"Only {n_prof} profile(s) — too few to draw a conclusion "
                    f"(need ≥ {SPARSE_MIN_PROFILES}). Confidence: {conf}."
                )
            elif ev["expect"] == "elevated":
                if result.stress_score >= ELEVATED_THRESHOLD:
                    status     = "✅ PASS"
                    conclusion = (
                        f"Score {result.stress_score:.1f} ≥ {ELEVATED_THRESHOLD} "
                        f"— consistent with elevated stress."
                    )
                else:
                    status     = "⚠️  MISS"
                    conclusion = (
                        f"Score {result.stress_score:.1f} < {ELEVATED_THRESHOLD} "
                        f"— lower than expected for a known stress event. "
                        f"Possible causes: sparse Argo coverage in this period, "
                        f"or weights need re-calibration."
                    )
            else:  # expect "low"
                if result.stress_score < ELEVATED_THRESHOLD:
                    status     = "✅ PASS"
                    conclusion = (
                        f"Score {result.stress_score:.1f} < {ELEVATED_THRESHOLD} "
                        f"— consistent with low / background stress."
                    )
                else:
                    status     = "⚠️  MISS"
                    conclusion = (
                        f"Score {result.stress_score:.1f} ≥ {ELEVATED_THRESHOLD} "
                        f"— higher than expected for a control period. "
                        f"Check baseline or consider narrowing the region."
                    )

        # ── Print result ───────────────────────────────────────────────────
        print(f"  → Score: {score_str:>6}  Profiles: {n_prof:>4}  "
              f"Confidence: {conf:<7}  {status}")
        print(f"     {conclusion}")
        if result and result.drivers:
            for d in result.drivers:
                direction = "▲" if d.anomaly > 0 else "▼"
                print(f"       {d.variable:<14} "
                      f"obs={d.observed:.2f}{d.unit} "
                      f"{direction}{abs(d.anomaly):.2f} "
                      f"(z={d.anomaly_zscore:+.2f}σ)  "
                      f"sub={d.sub_score:.1f}/100  w={d.weight:.0%}")
        if result and result.variables_missing:
            print(f"     Missing vars: {', '.join(result.variables_missing)}")
        if result and result.caveats:
            for c in result.caveats:
                print(f"     ⚠ {c}")
        print(f"     Ref: {ev['reference']}")
        print(f"     [{elapsed:.1f}s]")
        print()

        if verbose and result:
            print("     --- JSON (verbose) ---")
            import json
            d = result.to_dict()
            # Print only the key fields to keep output manageable
            brief = {k: d[k] for k in
                     ("stress_score", "category", "confidence",
                      "variables_used", "variables_missing",
                      "drivers", "data_quality", "caveats")}
            print(json.dumps(brief, indent=4, default=str))
            print()

        rows.append({
            "id":           ev["id"],
            "label":        ev["label"],
            "expect":       ev["expect"],
            "score":        score_str,
            "n_profiles":   n_prof,
            "confidence":   conf,
            "status":       status,
            "conclusion":   conclusion,
            "elapsed_s":    f"{elapsed:.1f}",
            "reference":    ev["reference"],
        })

    # ── Summary table ──────────────────────────────────────────────────────
    print("=" * 72)
    print("  SUMMARY TABLE")
    print("=" * 72)
    header = f"  {'Label':<42} {'Expect':<9} {'Score':>6}  {'Profiles':>8}  {'Status'}"
    print(header)
    print("  " + "-" * 68)
    passes = misses = sparse_count = errors = 0
    for r in rows:
        print(f"  {r['label']:<42} {r['expect']:<9} {r['score']:>6}  "
              f"{str(r['n_profiles']):>8}  {r['status']}")
        if "PASS"   in r["status"]: passes       += 1
        elif "MISS"  in r["status"]: misses       += 1
        elif "SPARSE" in r["status"]: sparse_count += 1
        elif "ERROR"  in r["status"]: errors       += 1

    total_conclusive = passes + misses
    print()
    print(f"  Results:  {passes} PASS  |  {misses} MISS  |  "
          f"{sparse_count} SPARSE (no conclusion)  |  {errors} ERROR")
    if total_conclusive > 0:
        pct = 100 * passes / total_conclusive
        print(f"  Pass rate (conclusive only): {passes}/{total_conclusive} "
              f"= {pct:.0f}%")
    print()
    print("  NOTE: Weights are PLACEHOLDERS (see config.yaml).")
    print("  MISS results indicate the weights need calibration against")
    print("  these events, not necessarily that the data are wrong.")
    print("=" * 72)

    # ── CSV output ─────────────────────────────────────────────────────────
    if csv_path:
        fieldnames = list(rows[0].keys()) if rows else []
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Results written to: {csv_path}")

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="OSE real-data validation against known stress events.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print full JSON result for each event.",
    )
    p.add_argument(
        "--csv",
        metavar="PATH",
        default=None,
        help="Save comparison table to this CSV file.",
    )
    p.add_argument(
        "--event",
        metavar="ID",
        default=None,
        help=(
            "Run only one event by its id "
            "(e.g. '2016_arabian_sea_bleaching'). "
            "Use --list to see all ids."
        ),
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List all event IDs and exit.",
    )
    args = p.parse_args()

    global EVENTS
    if args.list:
        print("Available event IDs:")
        for ev in EVENTS:
            print(f"  {ev['id']:<45} ({ev['expect']})")
        return 0

    if args.event:
        matched = [e for e in EVENTS if e["id"] == args.event]
        if not matched:
            print(f"Error: unknown event id '{args.event}'.", file=sys.stderr)
            print("Use --list to see available ids.", file=sys.stderr)
            return 1
        EVENTS = matched

    run_validation(verbose=args.verbose, csv_path=args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
