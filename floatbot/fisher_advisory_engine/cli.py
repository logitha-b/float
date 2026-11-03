"""
fisher_advisory_engine/cli.py
-------------------------------
Command-line interface for the Fisher Advisory Engine.

Usage (from FloatAi/ directory):
    python -m fisher_advisory_engine --help
    python -m fisher_advisory_engine --location Nagapattinam --day today
    python -m fisher_advisory_engine --location Nagapattinam --day today --lang ta
    python -m fisher_advisory_engine --location Nagapattinam --day today --output json
    python -m fisher_advisory_engine --list-ports
    python -m fisher_advisory_engine --list-languages
    python -m fisher_advisory_engine --rules-report
    python -m fisher_advisory_engine --synthetic   # offline/test mode

i18n review workflow:
    python -m fisher_advisory_engine.i18n export-review --lang ta --out ta_review.csv
    python -m fisher_advisory_engine.i18n import-review --lang ta --file ta_review.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m fisher_advisory_engine",
        description=(
            "Fisher Advisory Engine (FAE) v1.0 — "
            "sea-condition advisories for coastal fishers in Tamil Nadu."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m fisher_advisory_engine --location Nagapattinam --day today
  python -m fisher_advisory_engine --location Nagapattinam --day today --lang ta
  python -m fisher_advisory_engine --location Nagapattinam --trip offshore --output json
  python -m fisher_advisory_engine --location Nagapattinam --day today --synthetic
  python -m fisher_advisory_engine --list-ports
  python -m fisher_advisory_engine --rules-report
        """,
    )

    p.add_argument(
        "--location", "-l",
        default=None,
        help=(
            "Port name (e.g. 'Nagapattinam') or JSON bbox "
            "'{\"lat\":10.77,\"lon\":79.84}'."
        ),
    )
    p.add_argument(
        "--day", "-d",
        default="today",
        help="'today', 'tomorrow', 'last 7 days', or 'YYYY-MM-DD'. Default: today.",
    )
    p.add_argument(
        "--trip", "-t",
        default="nearshore",
        choices=["nearshore", "offshore"],
        dest="trip_type",
        help="Trip type. Default: nearshore.",
    )
    p.add_argument(
        "--boat",
        default=None,
        choices=["traditional", "mechanized"],
        dest="boat_type",
        help="Boat type (optional, metadata only in v1).",
    )
    p.add_argument(
        "--lang",
        default=None,
        help=(
            "Language code ('en', 'ta') or 'all_enabled'. "
            "Default: district default or 'en'."
        ),
    )
    p.add_argument(
        "--output", "-o",
        default="short",
        choices=["short", "json", "card"],
        help=(
            "'short' (default) — phone-readable message; "
            "'json' — full JSON; "
            "'card' — HTML dashboard card."
        ),
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        default=False,
        help="Use synthetic data (no network). For testing.",
    )
    p.add_argument(
        "--list-ports",
        action="store_true",
        default=False,
        help="List all registered port names and exit.",
    )
    p.add_argument(
        "--list-languages",
        action="store_true",
        default=False,
        help="List registered languages and their status, then exit.",
    )
    p.add_argument(
        "--rules-report",
        action="store_true",
        default=False,
        help="Show rules status (active/disabled) and exit.",
    )
    return p


def main(argv=None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    # Ensure FloatAi/ is on sys.path
    FloatAi_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if FloatAi_dir not in sys.path:
        sys.path.insert(0, FloatAi_dir)

    # ── --list-ports ──────────────────────────────────────────────────────
    if args.list_ports:
        from fisher_advisory_engine.inputs import list_ports
        ports = list_ports()
        print(f"Registered ports ({len(ports)} total — all unverified):")
        for p in ports:
            v = "✓" if str(p.get("verified","")).lower() == "true" else "✗ unverified"
            print(f"  {p.get('name','?'):<35} {p.get('district','?'):<25} [{v}]")
        return 0

    # ── --list-languages ──────────────────────────────────────────────────
    if args.list_languages:
        from fisher_advisory_engine.i18n.registry import get_language_config, get_mode
        lang_cfg = get_language_config()
        mode     = get_mode()
        print(f"Mode: {mode}")
        print(f"{'Code':<6} {'Name':<20} {'Enabled':<9} {'Status'}")
        print("-" * 52)
        for code, entry in lang_cfg.items():
            print(
                f"  {code:<6} {entry.get('native_name','?'):<20} "
                f"{'yes' if entry.get('enabled') else 'no':<9} "
                f"{entry.get('review_status','draft')}"
            )
        return 0

    # ── --rules-report ────────────────────────────────────────────────────
    if args.rules_report:
        from fisher_advisory_engine.rules_engine import rules_status_report
        report = rules_status_report()
        active   = [r for r in report if r["active"]]
        disabled = [r for r in report if not r["active"]]
        print(f"Rules: {len(active)} active, {len(disabled)} disabled (unsourced)\n")
        print("ACTIVE rules:")
        if active:
            for r in active:
                print(f"  [{r['severity'].upper()}] {r['id']}: {r['message_key']}")
                print(f"         Ref: {r['reference'][:80]}")
        else:
            print("  (none — all rules require citations before activation)")
        print("\nDISABLED rules (need citation):")
        for r in disabled:
            print(f"  {r['id']}: {r['message_key']}")
        return 0

    # ── Advisory query ────────────────────────────────────────────────────
    if not args.location:
        parser.error(
            "--location is required. "
            "Use --list-ports to see available ports."
        )

    # Parse location: try JSON dict, else treat as named string
    location = args.location.strip()
    if location.startswith("{"):
        try:
            location = json.loads(location)
        except json.JSONDecodeError as exc:
            parser.error(f"Cannot parse --location as JSON: {exc}")

    # Language: single code or None
    lang_arg = args.lang if args.lang else None

    print(
        f"Computing advisory: {args.location!r} | {args.day} | "
        f"{args.trip_type}{' | synthetic' if args.synthetic else ''}",
        flush=True,
    )

    from fisher_advisory_engine.engine import compute_advisory

    try:
        result = compute_advisory(
            location=location,
            day=args.day,
            trip_type=args.trip_type,
            boat_type=args.boat_type,
            language=lang_arg,
            use_synthetic=args.synthetic,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\nUnexpected error: {exc}", file=sys.stderr)
        return 1

    print()

    # ── Output ───────────────────────────────────────────────────────────
    if args.output == "json":
        print(result.to_json())

    elif args.output == "card":
        for lang in result.languages_served:
            print(f"\n--- {lang} ---")
            print(result.dashboard_card(lang))

    else:  # short (default)
        for lang in result.languages_served:
            print(f"{'─'*60}")
            print(f"  Language: {lang} | Confidence: {result.confidence.label}")
            print(f"{'─'*60}")
            print(result.short_message(lang))
            print()

        # Fallback summary
        if any(result.fallback_used.values()):
            print("  [Fallback to English used for keys:]")
            for lang, keys in result.fallback_used.items():
                if keys:
                    print(f"    {lang}: {', '.join(keys[:5])}"
                          f"{'...' if len(keys) > 5 else ''}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
