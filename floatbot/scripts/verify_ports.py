"""
scripts/verify_ports.py
------------------------
Cross-checks every row in fisher_advisory_engine/ports.csv that has
coordinates against OpenStreetMap Nominatim, then writes a verification
report for human review.

IMPORTANT — this script does NOT edit ports.csv.
You review the report, fix any flagged rows manually, then set
verified=true and source="OpenStreetMap Nominatim, queried <date>"
for each row you are satisfied with.

Nominatim usage policy
----------------------
  https://operations.osmfoundation.org/policies/nominatim/
  - Must set a descriptive User-Agent identifying the application.
  - At most 1 request per second (enforced with time.sleep).
  - Do not run this script repeatedly in a loop.

Usage
-----
    python scripts/verify_ports.py
    python scripts/verify_ports.py --csv report.csv
    python scripts/verify_ports.py --threshold 10    # flag if >10 km apart
    python scripts/verify_ports.py --skip-no-coords  # skip rows without coords

Output columns
--------------
  name            : port name from ports.csv
  my_lat          : latitude supplied in ports.csv
  my_lon          : longitude supplied in ports.csv
  osm_lat         : Nominatim result latitude (blank if no match)
  osm_lon         : Nominatim result longitude (blank if no match)
  osm_display     : Nominatim display_name (first result)
  distance_km     : great-circle distance between my coords and OSM coords
  flag            : ">5km" | "no_match" | "ok"
  notes           : any additional notes
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import date
from pathlib import Path

_PORTS_CSV   = Path(__file__).parent.parent / "fisher_advisory_engine" / "ports.csv"
_NOMINATIM   = "https://nominatim.openstreetmap.org/search"
_USER_AGENT  = (
    "FloatAi-FAE/1.0 verify_ports.py "
    "(https://github.com/your-org/FloatAi; contact: your@email.com)"
)
_SLEEP_SEC   = 1.1   # Nominatim policy: max 1 req/sec — we use 1.1 to be safe


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlam       = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2)**2
    return R * 2 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Nominatim query
# ---------------------------------------------------------------------------

def query_nominatim(name: str) -> dict | None:
    """
    Query Nominatim for '<name>, Tamil Nadu, India'.
    Returns the first result dict or None if no match.
    Raises on HTTP error.
    """
    import urllib.request
    import urllib.parse

    query_str = f"{name}, Tamil Nadu, India"
    params    = urllib.parse.urlencode({
        "q":              query_str,
        "format":         "json",
        "limit":          1,
        "addressdetails": 0,
    })
    url = f"{_NOMINATIM}?{params}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": _USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return data[0] if data else None


# ---------------------------------------------------------------------------
# Port loader (skips comment lines)
# ---------------------------------------------------------------------------

def load_ports(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        non_comment = [
            line for line in fh
            if not line.strip().startswith("#") and line.strip()
        ]
    reader = csv.DictReader(non_comment)
    for row in reader:
        name = row.get("name", "").strip()
        if not name or name.startswith("#"):
            continue
        rows.append(dict(row))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python scripts/verify_ports.py",
        description=(
            "Cross-check ports.csv coordinates against OpenStreetMap Nominatim. "
            "Does NOT edit ports.csv — writes a report only."
        ),
    )
    p.add_argument(
        "--csv", default=None, metavar="PATH",
        help="Output CSV path (default: verify_ports_report_<date>.csv).",
    )
    p.add_argument(
        "--threshold", type=float, default=5.0, metavar="KM",
        help="Flag rows where distance > this many km. Default: 5.",
    )
    p.add_argument(
        "--skip-no-coords", action="store_true", default=False,
        help="Skip rows that have no coordinates in ports.csv.",
    )
    args = p.parse_args(argv)

    today    = date.today().isoformat()
    out_path = args.csv or f"verify_ports_report_{today}.csv"

    ports = load_ports(_PORTS_CSV)
    print(f"Loaded {len(ports)} ports from {_PORTS_CSV}")
    print(f"Threshold: {args.threshold} km")
    print(f"Report will be written to: {out_path}")
    print(f"Querying Nominatim (≥1 s between requests — please wait)...\n")

    fieldnames = [
        "name", "my_lat", "my_lon",
        "osm_lat", "osm_lon", "osm_display",
        "distance_km", "flag", "notes",
    ]
    report_rows = []
    n_ok = n_flagged = n_no_match = n_skipped = 0

    for i, port in enumerate(ports):
        name    = port.get("name", "").strip()
        lat_raw = port.get("lat", "").strip()
        lon_raw = port.get("lon", "").strip()

        # Skip rows without coordinates
        if not lat_raw or not lon_raw:
            if args.skip_no_coords:
                n_skipped += 1
                continue
            report_rows.append({
                "name":       name,
                "my_lat":     "",
                "my_lon":     "",
                "osm_lat":    "",
                "osm_lon":    "",
                "osm_display":"",
                "distance_km":"",
                "flag":       "no_coords_in_csv",
                "notes":      "No coordinates in ports.csv — operator needs to supply.",
            })
            n_skipped += 1
            print(f"  [{i+1:2d}/{len(ports)}] {name:<40} → SKIPPED (no coords)")
            continue

        try:
            my_lat = float(lat_raw)
            my_lon = float(lon_raw)
        except ValueError:
            report_rows.append({
                "name":       name, "my_lat": lat_raw, "my_lon": lon_raw,
                "osm_lat":    "", "osm_lon": "", "osm_display": "",
                "distance_km":"", "flag": "bad_coords",
                "notes": f"Cannot parse lat={lat_raw!r} lon={lon_raw!r} as float.",
            })
            print(f"  [{i+1:2d}/{len(ports)}] {name:<40} → BAD COORDS")
            continue

        # Nominatim query
        try:
            result = query_nominatim(name)
        except Exception as exc:
            report_rows.append({
                "name":       name, "my_lat": my_lat, "my_lon": my_lon,
                "osm_lat":    "", "osm_lon": "", "osm_display": "",
                "distance_km":"", "flag": "query_error",
                "notes":      f"Nominatim error: {exc}",
            })
            print(f"  [{i+1:2d}/{len(ports)}] {name:<40} → QUERY ERROR: {exc}")
            time.sleep(_SLEEP_SEC)
            continue

        if result is None:
            report_rows.append({
                "name":       name, "my_lat": my_lat, "my_lon": my_lon,
                "osm_lat":    "", "osm_lon": "", "osm_display": "",
                "distance_km":"", "flag": "no_match",
                "notes":      "Nominatim returned no results for this query.",
            })
            n_no_match += 1
            print(f"  [{i+1:2d}/{len(ports)}] {name:<40} → NO MATCH")
            time.sleep(_SLEEP_SEC)
            continue

        osm_lat  = float(result["lat"])
        osm_lon  = float(result["lon"])
        osm_disp = result.get("display_name", "")[:120]
        dist_km  = haversine_km(my_lat, my_lon, osm_lat, osm_lon)

        if dist_km > args.threshold:
            flag  = f">{args.threshold:.0f}km"
            n_flagged += 1
        else:
            flag  = "ok"
            n_ok += 1

        report_rows.append({
            "name":        name,
            "my_lat":      round(my_lat, 6),
            "my_lon":      round(my_lon, 6),
            "osm_lat":     round(osm_lat, 6),
            "osm_lon":     round(osm_lon, 6),
            "osm_display": osm_disp,
            "distance_km": round(dist_km, 2),
            "flag":        flag,
            "notes":       "",
        })

        status = f"dist={dist_km:.1f} km  [{flag}]"
        print(f"  [{i+1:2d}/{len(ports)}] {name:<40} → {status}")

        # Nominatim policy: ≥1 request per second
        time.sleep(_SLEEP_SEC)

    # Write report
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Total ports checked : {len(ports)}")
    print(f"  OK  (≤{args.threshold:.0f} km)        : {n_ok}")
    print(f"  Flagged (>{args.threshold:.0f} km)     : {n_flagged}")
    print(f"  No OSM match        : {n_no_match}")
    print(f"  Skipped / no coords : {n_skipped}")
    print(f"\n  Report written to   : {out_path}")
    print(f"\n  Next steps:")
    print(f"  1. Open {out_path} and review every flagged row.")
    print(f"  2. For rows where OSM disagrees, decide which coordinate is correct.")
    print(f"  3. Edit fisher_advisory_engine/ports.csv manually:")
    print(f"       - Set lat/lon to the confirmed value.")
    print(f"       - Set verified=true.")
    print( f"       - Set source=\"OpenStreetMap Nominatim, queried {today}\"")
    print(f"         (or cite your preferred authoritative source).")
    print(f"  4. Re-run this script only if you change coordinates — Nominatim")
    print(f"     policy forbids repeated automated queries for the same data.")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
