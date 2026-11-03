"""
fisher_advisory_engine/validate_incois.py
-------------------------------------------
Validation script: compare FAE front and chlorophyll indicators against
INCOIS Potential Fishing Zone (PFZ) advisory files.

IMPORTANT SEPARATION
--------------------
This script uses REAL data and external reference files.
Unit tests (tests/) use ONLY synthetic/mock data — no network.

Usage
-----
    python fisher_advisory_engine/validate_incois.py

    # With specific validation data folder:
    python fisher_advisory_engine/validate_incois.py --data-dir validation_data/

    # Verbose (show per-date comparisons):
    python fisher_advisory_engine/validate_incois.py --verbose

    # Save report to CSV:
    python fisher_advisory_engine/validate_incois.py --csv validation_report.csv

What to place in validation_data/
-----------------------------------
INCOIS PFZ advisories are available from:
  https://incois.gov.in/portal/osf/pfz.jsp

Place the following files in validation_data/:
  - incois_pfz_<YYYY-MM-DD>.csv  (one file per day)
    Required columns: date, lat, lon, pfz_active (1/0), district
    Optional columns: front_noted (1/0), chl_category (HIGH/MODERATE/LOW)

If the folder is empty or no matching files are found, the script prints
what is needed and exits cleanly.

Honesty
-------
The script explicitly states sample sizes and says plainly when there is
too little data to conclude anything.  It never fabricates agreement rates.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

_VALIDATION_DATA_DIR = Path(__file__).parent / "validation_data"
_MIN_SAMPLES_TO_CONCLUDE = 30   # fewer → "insufficient data"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python fisher_advisory_engine/validate_incois.py",
        description="Validate FAE indicators against INCOIS PFZ reference data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir", default=str(_VALIDATION_DATA_DIR),
        help=f"Folder containing incois_pfz_*.csv files. "
             f"Default: {_VALIDATION_DATA_DIR}",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-date comparisons.",
    )
    p.add_argument(
        "--csv", default=None, metavar="PATH",
        help="Save validation report to this CSV file.",
    )
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("  FAE — INCOIS PFZ Validation Script")
    print("=" * 70)

    # ── Check if validation data exists ──────────────────────────────────
    if not data_dir.exists() or not any(data_dir.glob("incois_pfz_*.csv")):
        print(f"\n  Validation data folder: {data_dir}")
        print("\n  No INCOIS PFZ files found.")
        print("\n  To run validation, place files matching incois_pfz_YYYY-MM-DD.csv")
        print("  in the validation_data/ folder.")
        print("\n  Required CSV columns per file:")
        print("    date          : ISO date (YYYY-MM-DD)")
        print("    lat           : float (decimal degrees)")
        print("    lon           : float (decimal degrees)")
        print("    pfz_active    : 1 (fishing zone active) or 0")
        print("    district      : Tamil Nadu district name")
        print("    front_noted   : 1 or 0 (optional)")
        print("    chl_category  : HIGH / MODERATE / LOW (optional)")
        print("\n  Source: https://incois.gov.in/portal/osf/pfz.jsp")
        print("\n  SKIPPING validation — no data to compare.")
        return 0

    # ── Load reference files ──────────────────────────────────────────────
    ref_rows = []
    for fpath in sorted(data_dir.glob("incois_pfz_*.csv")):
        with open(fpath, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ref_rows.append(row)

    print(f"\n  Loaded {len(ref_rows)} reference rows from {data_dir}")

    if len(ref_rows) < 3:
        print("\n  Fewer than 3 reference rows — nothing to compare.")
        print("  Add more INCOIS PFZ files to produce a meaningful report.")
        return 0

    # ── Attempt to compute FAE indicators for each reference point ────────
    print("\n  Computing FAE indicators for each reference point...")
    print("  (This may take several minutes for real data.)")

    results = []
    from fisher_advisory_engine.indicators import sst_indicator, front_indicator, productivity_indicator

    for i, row in enumerate(ref_rows):
        date_str = row.get("date", "").strip()
        try:
            lat = float(row.get("lat", 0))
            lon = float(row.get("lon", 0))
        except (ValueError, TypeError):
            continue

        pfz_ref   = int(row.get("pfz_active", 0))
        front_ref = int(row.get("front_noted", -1)) if row.get("front_noted") else None
        chl_ref   = row.get("chl_category", "").strip().upper() or None

        # Compute FAE SST indicator
        fae_sst = fae_front = fae_chl = None
        try:
            sst_res = sst_indicator(lat, lon, date_str, date_str)
            if sst_res.status == "ok" and isinstance(sst_res.value, dict):
                fae_sst = sst_res.value.get("sst_c")
        except Exception:
            pass

        try:
            fr_res = front_indicator(lat, lon, date_str)
            if fr_res.status == "ok" and isinstance(fr_res.value, dict):
                fae_front = 1 if fr_res.value.get("detected") else 0
        except Exception:
            pass

        try:
            chl_res = productivity_indicator(lat, lon, date_str, date_str)
            if chl_res.status == "ok" and isinstance(chl_res.value, dict):
                fae_chl = chl_res.value.get("category")
        except Exception:
            pass

        results.append({
            "date":       date_str,
            "lat":        lat,
            "lon":        lon,
            "district":   row.get("district", ""),
            "pfz_ref":    pfz_ref,
            "front_ref":  front_ref,
            "chl_ref":    chl_ref,
            "fae_sst":    fae_sst,
            "fae_front":  fae_front,
            "fae_chl":    fae_chl,
        })

        if args.verbose:
            print(
                f"  {date_str} ({lat:.2f},{lon:.2f}) "
                f"PFZ={pfz_ref} front_ref={front_ref} chl_ref={chl_ref} "
                f"| FAE front={fae_front} chl={fae_chl}"
            )

    # ── Agreement analysis ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    n_total = len(results)
    print(f"\n  Total comparison points: {n_total}")

    if n_total < _MIN_SAMPLES_TO_CONCLUDE:
        print(
            f"\n  ⚠ Only {n_total} comparison points available "
            f"(minimum recommended: {_MIN_SAMPLES_TO_CONCLUDE}).\n"
            "  Agreement rates below are reported but CANNOT be used to "
            "draw statistical conclusions.\n"
            "  Add more INCOIS PFZ files for a meaningful evaluation."
        )

    # Front agreement
    front_pairs = [
        r for r in results
        if r["front_ref"] is not None and r["fae_front"] is not None
    ]
    if front_pairs:
        agree = sum(1 for r in front_pairs if r["front_ref"] == r["fae_front"])
        pct   = 100 * agree / len(front_pairs)
        print(f"\n  Front detection agreement:")
        print(f"    Pairs with both ref and FAE data: {len(front_pairs)}")
        print(f"    Agreement: {agree}/{len(front_pairs)} = {pct:.0f}%")
        if len(front_pairs) < _MIN_SAMPLES_TO_CONCLUDE:
            print(f"    ⚠ Insufficient sample size — rate is indicative only.")
    else:
        print("\n  Front detection: no paired comparison data available.")

    # Chl agreement
    chl_pairs = [
        r for r in results
        if r["chl_ref"] and r["fae_chl"] and r["fae_chl"] != "no_data"
    ]
    if chl_pairs:
        agree = sum(1 for r in chl_pairs
                    if r["chl_ref"].upper() == (r["fae_chl"] or "").upper())
        pct   = 100 * agree / len(chl_pairs)
        print(f"\n  Chlorophyll category agreement:")
        print(f"    Pairs with both ref and FAE data: {len(chl_pairs)}")
        print(f"    Exact match: {agree}/{len(chl_pairs)} = {pct:.0f}%")
        if len(chl_pairs) < _MIN_SAMPLES_TO_CONCLUDE:
            print(f"    ⚠ Insufficient sample size — rate is indicative only.")
    else:
        print("\n  Chlorophyll: no paired comparison data available.")

    print(
        "\n  NOTE: FAE thresholds are PLACEHOLDERS. "
        "Agreement rates do not validate the FAE — they show whether "
        "placeholder thresholds happen to align with INCOIS reference data."
    )

    # ── CSV output ────────────────────────────────────────────────────────
    if args.csv and results:
        fields = list(results[0].keys())
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Report saved to: {args.csv}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
