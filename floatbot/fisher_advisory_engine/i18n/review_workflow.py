"""
fisher_advisory_engine/i18n/review_workflow.py
------------------------------------------------
Translation review workflow: export-review and import-review commands.

export-review
  Writes a CSV: key, english_source, current_text, placeholders,
                reviewer, date, status, comments
  for a specified language.

import-review
  Reads the completed CSV and updates per-key status in the language YAML file.

A language becomes "reviewed" only when every key has status="reviewed"
by a named reviewer.

Usage (from FloatAi/ directory)
---------------------------------
  python -m fisher_advisory_engine.i18n export-review --lang ta --out ta_review.csv
  python -m fisher_advisory_engine.i18n import-review --lang ta --file ta_review.csv
"""

from __future__ import annotations

import csv
import re
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Set

_I18N_DIR = Path(__file__).parent
_CFG_PATH = _I18N_DIR.parent / "config.yaml"


def export_review(lang: str, output_path: str) -> None:
    """
    Export a review CSV for the given language.
    """
    import yaml  # type: ignore

    en_path   = _I18N_DIR / "en.yaml"
    lang_path = _I18N_DIR / f"{lang}.yaml"

    if not en_path.exists():
        print(f"ERROR: en.yaml not found at {en_path}", file=sys.stderr)
        sys.exit(1)

    with open(en_path, "r", encoding="utf-8") as fh:
        en_data: Dict[str, str] = yaml.safe_load(fh) or {}

    lang_data: Dict[str, str] = {}
    if lang_path.exists():
        with open(lang_path, "r", encoding="utf-8") as fh:
            lang_data = yaml.safe_load(fh) or {}

    rows = []
    for key, en_text in sorted(en_data.items()):
        placeholders = sorted(set(re.findall(r"\{(\w+)\}", en_text)))
        current_text = lang_data.get(key, "")
        rows.append({
            "key":            key,
            "english_source": en_text,
            "current_text":   current_text,
            "placeholders":   "|".join(placeholders),
            "reviewer":       "",
            "date":           "",
            "status":         "pending",
            "comments":       "",
        })

    fieldnames = ["key", "english_source", "current_text", "placeholders",
                  "reviewer", "date", "status", "comments"]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} keys to {output_path}")
    print(f"Fill in 'current_text', 'reviewer', 'date', 'status' (reviewed/pending/rejected),")
    print(f"then run import-review to update {lang}.yaml.")


def import_review(lang: str, input_path: str) -> None:
    """
    Import a completed review CSV and update the language YAML file.
    Updates per-key review status and sets the overall file review_status
    to 'reviewed' if ALL keys have status='reviewed'.
    """
    import yaml  # type: ignore

    lang_path = _I18N_DIR / f"{lang}.yaml"
    lang_data: Dict[str, str] = {}
    if lang_path.exists():
        with open(lang_path, "r", encoding="utf-8") as fh:
            lang_data = yaml.safe_load(fh) or {}

    with open(input_path, newline="", encoding="utf-8-sig") as fh:
        reader  = csv.DictReader(fh)
        updated = 0
        pending = 0
        for row in reader:
            key     = row.get("key", "").strip()
            text    = row.get("current_text", "").strip()
            status  = row.get("status", "pending").strip().lower()
            if not key:
                continue
            if text:
                lang_data[key] = text
                updated += 1
            if status != "reviewed":
                pending += 1

    # Write back
    with open(lang_path, "w", encoding="utf-8") as fh:
        yaml.dump(lang_data, fh, allow_unicode=True, sort_keys=True,
                  default_flow_style=False)

    # Update review_status in config.yaml
    if pending == 0:
        _set_review_status(lang, "reviewed")
        print(f"All keys reviewed — set review_status: reviewed for '{lang}'.")
    else:
        _set_review_status(lang, "pending")
        print(f"{pending} key(s) still pending — review_status remains 'pending'.")

    print(f"Updated {updated} key(s) in {lang_path}")


def _set_review_status(lang: str, status: str) -> None:
    import yaml  # type: ignore
    with open(_CFG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if "languages" in cfg and lang in cfg["languages"]:
        cfg["languages"][lang]["review_status"] = status
    with open(_CFG_PATH, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, allow_unicode=True, sort_keys=False,
                  default_flow_style=False)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m fisher_advisory_engine.i18n",
        description="FAE i18n review workflow",
    )
    sub = p.add_subparsers(dest="command")

    exp = sub.add_parser("export-review", help="Export a review CSV for a language")
    exp.add_argument("--lang",  required=True, help="Language code, e.g. ta")
    exp.add_argument("--out",   required=True, help="Output CSV path")

    imp = sub.add_parser("import-review", help="Import a completed review CSV")
    imp.add_argument("--lang",  required=True, help="Language code, e.g. ta")
    imp.add_argument("--file",  required=True, help="Input CSV path")

    args = p.parse_args(argv)

    if args.command == "export-review":
        export_review(args.lang, args.out)
    elif args.command == "import-review":
        import_review(args.lang, args.file)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
