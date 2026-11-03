"""
ocean_stress_engine/cli.py
----------------------------
Command-line interface for the Ocean Stress Engine.

Usage (from FloatAi/ directory):

    python -m ocean_stress_engine --help

    python -m ocean_stress_engine \\
        --region "Arabian Sea" \\
        --time_window "2023-07" \\
        --depth_layer surface \\
        --ecosystem coral_reef \\
        --output summary

    python -m ocean_stress_engine \\
        --region '{"lat_min":8,"lat_max":26,"lon_min":55,"lon_max":78}' \\
        --time_window "2023-07-01/2023-07-31" \\
        --output json

    # Offline / synthetic data (no network):
    python -m ocean_stress_engine --region "Arabian Sea" \\
        --time_window "2023-07" --synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
import os


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ocean_stress_engine",
        description=(
            "Ocean Stress Engine (OSE) v1.0  —  "
            "Thermohaline stress score (0-100) for an ocean region."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ocean_stress_engine --region "Arabian Sea" --time_window "2023-07"
  python -m ocean_stress_engine --region "Coral Sea"   --time_window "2022-01" --ecosystem coral_reef
  python -m ocean_stress_engine --region "Arabian Sea" --time_window "2023-07" --output json
  python -m ocean_stress_engine --region "Arabian Sea" --time_window "2023-07" --synthetic
  python -m ocean_stress_engine --list-regions
        """,
    )

    p.add_argument(
        "--region", "-r",
        type=str,
        default=None,
        help=(
            "Named region (e.g. 'Arabian Sea') or JSON bbox string "
            "(e.g. '{\"lat_min\":8,\"lat_max\":26,\"lon_min\":55,\"lon_max\":78}')."
        ),
    )
    p.add_argument(
        "--time_window", "-t",
        type=str,
        default=None,
        help=(
            "Time window: 'YYYY-MM', 'YYYY-MM-DD/YYYY-MM-DD', "
            "'last N days', or 'YYYY'."
        ),
    )
    p.add_argument(
        "--depth_layer", "-d",
        type=str,
        default="surface",
        choices=["surface", "mid", "deep"],
        help="Depth layer (default: surface = 0-50 m).",
    )
    p.add_argument(
        "--ecosystem", "-e",
        type=str,
        default="general",
        choices=["coral_reef", "fisheries", "general"],
        dest="ecosystem_type",
        help="Ecosystem type for weighting (default: general).",
    )
    p.add_argument(
        "--baseline", "-b",
        type=str,
        default=None,
        dest="baseline_period",
        help="Baseline period as 'YYYY-YYYY' (default: 2005-2020).",
    )
    p.add_argument(
        "--output", "-o",
        type=str,
        default="summary",
        choices=["summary", "json", "card"],
        help=(
            "Output format: "
            "'summary' (plain-English paragraph, default), "
            "'json' (full JSON), "
            "'card' (HTML dashboard card)."
        ),
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        default=False,
        help="Use synthetic data — no network calls (for testing).",
    )
    p.add_argument(
        "--list-regions",
        action="store_true",
        default=False,
        help="List all supported named regions and exit.",
    )
    return p


def main(argv=None):
    parser = _build_parser()
    args   = parser.parse_args(argv)

    # Ensure FloatAi/ is on sys.path so data/core imports resolve
    FloatAi_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if FloatAi_dir not in sys.path:
        sys.path.insert(0, FloatAi_dir)

    # --list-regions
    if args.list_regions:
        from ocean_stress_engine.inputs import list_regions
        print("Supported named regions:")
        for name in list_regions():
            print(f"  {name}")
        return 0

    # Require --region and --time_window for scoring
    if not args.region:
        parser.error("--region is required (or use --list-regions to see options).")
    if not args.time_window:
        parser.error("--time_window is required.")

    # Parse region: try JSON dict first, then treat as named string
    region = args.region.strip()
    if region.startswith("{"):
        try:
            region = json.loads(region)
        except json.JSONDecodeError as exc:
            parser.error(f"Could not parse --region as JSON: {exc}")

    from ocean_stress_engine.engine import compute_stress

    print(f"Computing thermohaline stress for: {args.region!r}  |  {args.time_window}  |  {args.depth_layer}  |  {args.ecosystem_type}", flush=True)
    if args.synthetic:
        print("  (synthetic data mode — no network calls)", flush=True)

    try:
        result = compute_stress(
            region=region,
            time_window=args.time_window,
            depth_layer=args.depth_layer,
            ecosystem_type=args.ecosystem_type,
            baseline_period=args.baseline_period,
            use_synthetic=args.synthetic,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\nUnexpected error: {exc}", file=sys.stderr)
        return 1

    print()   # blank line before output

    if args.output == "json":
        print(result.to_json())

    elif args.output == "card":
        print(result.dashboard_card())

    else:   # summary (default)
        # Print score header
        cat_emoji = {"LOW": "🟢", "MODERATE": "🟡", "HIGH": "🟠", "SEVERE": "🔴"}.get(
            result.category, "⚪"
        )
        conf_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(
            result.confidence, "⚪"
        )
        print("=" * 60)
        print(f"  Thermohaline Stress Score")
        print(f"  {result.region_name}  |  {result.period_start} – {result.period_end}")
        print("=" * 60)
        print(f"  Score:       {cat_emoji}  {result.stress_score:.1f}/100  ({result.category})")
        print(f"  Confidence:  {conf_emoji}  {result.confidence}")
        print(f"  Ecosystem:   {result.ecosystem_type}")
        print(f"  Depth:       {result.depth_layer_lbl}")
        print()
        print("  Variable Drivers:")
        for d in result.drivers:
            bar = "█" * int(d.sub_score / 5)
            direction = "▲" if d.anomaly > 0 else "▼"
            print(
                f"    {d.variable:<14} {bar:<20} {d.sub_score:5.1f}/100 "
                f"| obs={d.observed:.2f}{d.unit} {direction}{abs(d.anomaly):.2f} "
                f"(z={d.anomaly_zscore:+.2f}σ, w={d.weight:.0%})"
            )
        if result.variables_missing:
            print(f"\n  Missing vars:  {', '.join(result.variables_missing)}")
        print()
        print("  Data quality:")
        print(f"    Floats: {result.n_floats}  |  Profiles: {result.n_profiles}  "
              f"|  Readings: {result.n_readings_total} "
              f"({result.n_readings_excluded} excluded by QC)")
        print()
        print("  Summary:")
        # Indent the summary paragraph
        for line in result.summary_text().split("\n\n"):
            print(f"    {line.strip()}")
            print()
        if result.caveats:
            print("  Caveats:")
            for c in result.caveats:
                print(f"    • {c}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
