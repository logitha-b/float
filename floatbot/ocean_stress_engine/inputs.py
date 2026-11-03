"""
ocean_stress_engine/inputs.py
------------------------------
Input validation and parsing for the Ocean Stress Engine.

Handles:
  - Region: named string OR lat/lon bbox dict
  - Time window: "YYYY-MM", "YYYY-MM-DD/YYYY-MM-DD", "last N days"
  - Depth layer: "surface" | "mid" | "deep" | None
  - Ecosystem type: "coral_reef" | "fisheries" | "general"
  - Baseline period: "YYYY-YYYY" string

OSE-local region list — does NOT import nlu.rule_parser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# OSE-local named region registry
# (lon_min, lon_max, lat_min, lat_max)
# Source: standard oceanographic sub-basin definitions.
# ---------------------------------------------------------------------------
_OSE_REGIONS: Dict[str, Dict[str, float]] = {
    # Indian Ocean sub-basins
    "arabian sea":          {"lon_min": 55,  "lon_max": 78,  "lat_min":  8,  "lat_max": 26},
    "bay of bengal":        {"lon_min": 80,  "lon_max": 100, "lat_min":  5,  "lat_max": 23},
    "lakshadweep sea":      {"lon_min": 72,  "lon_max": 78,  "lat_min":  8,  "lat_max": 15},
    "andaman sea":          {"lon_min": 92,  "lon_max": 100, "lat_min":  6,  "lat_max": 18},
    "mozambique channel":   {"lon_min": 35,  "lon_max": 50,  "lat_min":-26,  "lat_max":-10},
    "indian ocean":         {"lon_min": 30,  "lon_max": 120, "lat_min":-40,  "lat_max": 30},
    # Atlantic
    "gulf of mexico":       {"lon_min":-98,  "lon_max": -80, "lat_min": 18,  "lat_max": 31},
    "caribbean sea":        {"lon_min":-85,  "lon_max": -60, "lat_min": 10,  "lat_max": 25},
    "caribbean":            {"lon_min":-85,  "lon_max": -60, "lat_min": 10,  "lat_max": 25},
    "north atlantic":       {"lon_min":-80,  "lon_max":   0, "lat_min": 25,  "lat_max": 65},
    "south atlantic":       {"lon_min":-60,  "lon_max":  20, "lat_min":-60,  "lat_max": -5},
    "tropical atlantic":    {"lon_min":-60,  "lon_max":  20, "lat_min":-20,  "lat_max": 20},
    "atlantic ocean":       {"lon_min":-80,  "lon_max":  20, "lat_min":-60,  "lat_max": 65},
    # Pacific
    "north pacific":        {"lon_min": 120, "lon_max": 240, "lat_min": 20,  "lat_max": 65},
    "south pacific":        {"lon_min": 150, "lon_max": 290, "lat_min":-60,  "lat_max":-10},
    "tropical pacific":     {"lon_min": 140, "lon_max": 280, "lat_min":-20,  "lat_max": 20},
    "pacific ocean":        {"lon_min": 120, "lon_max": 290, "lat_min":-60,  "lat_max": 65},
    "coral sea":            {"lon_min": 142, "lon_max": 162, "lat_min":-26,  "lat_max":-10},
    "tasman sea":           {"lon_min": 145, "lon_max": 170, "lat_min":-48,  "lat_max":-30},
    "south china sea":      {"lon_min": 100, "lon_max": 122, "lat_min":  0,  "lat_max": 25},
    "philippine sea":       {"lon_min": 126, "lon_max": 142, "lat_min":  5,  "lat_max": 25},
    # Mediterranean / marginal
    "mediterranean sea":    {"lon_min":  -6, "lon_max":  37, "lat_min": 30,  "lat_max": 47},
    "mediterranean":        {"lon_min":  -6, "lon_max":  37, "lat_min": 30,  "lat_max": 47},
    "red sea":              {"lon_min":  32, "lon_max":  44, "lat_min": 12,  "lat_max": 30},
    "persian gulf":         {"lon_min":  48, "lon_max":  57, "lat_min": 23,  "lat_max": 30},
    "gulf of aden":         {"lon_min":  42, "lon_max":  52, "lat_min": 10,  "lat_max": 15},
    "black sea":            {"lon_min":  28, "lon_max":  42, "lat_min": 41,  "lat_max": 47},
    "north sea":            {"lon_min":  -4, "lon_max":  10, "lat_min": 51,  "lat_max": 62},
    # Polar
    "southern ocean":       {"lon_min":-180, "lon_max": 180, "lat_min":-70,  "lat_max":-45},
    "arctic ocean":         {"lon_min":-180, "lon_max": 180, "lat_min": 70,  "lat_max": 90},
}

VALID_DEPTH_LAYERS    = ("surface", "mid", "deep")
VALID_ECOSYSTEM_TYPES = ("coral_reef", "fisheries", "general")


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass
class ParsedRegion:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    name: str          # human-readable label

    @property
    def centre(self) -> Tuple[float, float]:
        return (self.lat_min + self.lat_max) / 2, (self.lon_min + self.lon_max) / 2

    @property
    def area_deg2(self) -> float:
        return abs(self.lat_max - self.lat_min) * abs(self.lon_max - self.lon_min)


@dataclass
class ParsedTimeWindow:
    start: date
    end: date

    @property
    def start_iso(self) -> str:
        return self.start.isoformat()

    @property
    def end_iso(self) -> str:
        return self.end.isoformat()

    @property
    def calendar_months(self) -> list[Tuple[int, int]]:
        """List of (year, month) tuples covered by this window."""
        months = []
        cur = self.start.replace(day=1)
        end_month = self.end.replace(day=1)
        while cur <= end_month:
            months.append((cur.year, cur.month))
            # advance one month
            if cur.month == 12:
                cur = cur.replace(year=cur.year + 1, month=1)
            else:
                cur = cur.replace(month=cur.month + 1)
        return months


@dataclass
class ParsedDepthLayer:
    min_dbar: float
    max_dbar: float
    label: str


@dataclass
class ValidatedInputs:
    region:          ParsedRegion
    time_window:     ParsedTimeWindow
    depth_layer:     ParsedDepthLayer
    ecosystem_type:  str
    baseline_start:  date
    baseline_end:    date


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_inputs(
    region,
    time_window: str,
    depth_layer: Optional[str] = None,
    ecosystem_type: str = "general",
    baseline_period: Optional[str] = None,
) -> ValidatedInputs:
    """
    Parse and validate all OSE inputs. Raises ValueError with a clear
    message on any invalid input.

    Parameters
    ----------
    region          : str or dict   Named region or bbox dict.
    time_window     : str           "2023-07", "2023-07-01/2023-07-31",
                                    or "last N days".
    depth_layer     : str or None   "surface", "mid", "deep". Default: surface.
    ecosystem_type  : str           "coral_reef", "fisheries", "general".
    baseline_period : str or None   "2005-2020" or "YYYY-YYYY". Default: 2005-2020.
    """
    parsed_region    = _parse_region(region)
    parsed_time      = _parse_time_window(time_window)
    parsed_depth     = _parse_depth_layer(depth_layer)
    eco              = _parse_ecosystem(ecosystem_type)
    base_start, base_end = _parse_baseline(baseline_period)

    return ValidatedInputs(
        region=parsed_region,
        time_window=parsed_time,
        depth_layer=parsed_depth,
        ecosystem_type=eco,
        baseline_start=base_start,
        baseline_end=base_end,
    )


# ---------------------------------------------------------------------------
# Region parser
# ---------------------------------------------------------------------------

def _parse_region(region) -> ParsedRegion:
    if isinstance(region, dict):
        required = {"lat_min", "lat_max", "lon_min", "lon_max"}
        missing = required - set(region.keys())
        if missing:
            raise ValueError(
                f"Region dict missing keys: {missing}. "
                f"Required: lat_min, lat_max, lon_min, lon_max."
            )
        lat_min = float(region["lat_min"])
        lat_max = float(region["lat_max"])
        lon_min = float(region["lon_min"])
        lon_max = float(region["lon_max"])
        _validate_bbox(lat_min, lat_max, lon_min, lon_max)
        return ParsedRegion(
            lat_min=lat_min, lat_max=lat_max,
            lon_min=lon_min, lon_max=lon_max,
            name=f"Custom bbox (lat {lat_min}–{lat_max}, lon {lon_min}–{lon_max})",
        )

    if not isinstance(region, str) or not region.strip():
        raise ValueError(
            "region must be a non-empty string (region name) or a dict with "
            "lat_min, lat_max, lon_min, lon_max."
        )

    key = region.strip().lower()
    # Longest-match: try full key first, then progressively shorter substrings
    best_name = None
    best_len  = 0
    for name in _OSE_REGIONS:
        if name in key and len(name) > best_len:
            best_name = name
            best_len  = len(name)

    if best_name is None:
        available = ", ".join(sorted(_OSE_REGIONS.keys()))
        raise ValueError(
            f"Unknown region: '{region}'. "
            f"Available named regions: {available}. "
            f"Or pass a dict: {{lat_min:…, lat_max:…, lon_min:…, lon_max:…}}."
        )

    box = _OSE_REGIONS[best_name]
    return ParsedRegion(
        lat_min=box["lat_min"], lat_max=box["lat_max"],
        lon_min=box["lon_min"], lon_max=box["lon_max"],
        name=best_name.title(),
    )


def _validate_bbox(lat_min, lat_max, lon_min, lon_max):
    if not (-90 <= lat_min < lat_max <= 90):
        raise ValueError(
            f"Invalid latitude range: lat_min={lat_min}, lat_max={lat_max}. "
            "Requires -90 ≤ lat_min < lat_max ≤ 90."
        )
    if not (-180 <= lon_min < lon_max <= 360):
        raise ValueError(
            f"Invalid longitude range: lon_min={lon_min}, lon_max={lon_max}. "
            "Requires -180 ≤ lon_min < lon_max ≤ 360."
        )


# ---------------------------------------------------------------------------
# Time window parser
# ---------------------------------------------------------------------------

def _parse_time_window(time_window: str) -> ParsedTimeWindow:
    if not isinstance(time_window, str) or not time_window.strip():
        raise ValueError("time_window must be a non-empty string.")

    tw = time_window.strip()

    # "last N days"
    m = re.match(r"^last\s+(\d+)\s+days?$", tw, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        if n < 1:
            raise ValueError("'last N days' requires N ≥ 1.")
        end   = date.today() - timedelta(days=1)
        start = end - timedelta(days=n - 1)
        return ParsedTimeWindow(start=start, end=end)

    # "YYYY-MM" (full calendar month)
    m = re.match(r"^(\d{4})-(\d{2})$", tw)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        _validate_year_month(year, month)
        start = date(year, month, 1)
        end   = _last_day_of_month(year, month)
        return ParsedTimeWindow(start=start, end=end)

    # "YYYY-MM-DD/YYYY-MM-DD" or "YYYY-MM-DD to YYYY-MM-DD"
    m = re.match(
        r"^(\d{4}-\d{2}-\d{2})\s*(?:/|to)\s*(\d{4}-\d{2}-\d{2})$", tw, re.IGNORECASE
    )
    if m:
        try:
            start = date.fromisoformat(m.group(1))
            end   = date.fromisoformat(m.group(2))
        except ValueError as exc:
            raise ValueError(f"Invalid date in time_window: {exc}") from exc
        if start > end:
            raise ValueError(
                f"time_window start ({start}) must be before end ({end})."
            )
        return ParsedTimeWindow(start=start, end=end)

    # "YYYY" (full year)
    m = re.match(r"^(\d{4})$", tw)
    if m:
        year = int(m.group(1))
        if not (1981 <= year <= date.today().year):
            raise ValueError(f"Year {year} is out of range (1981–{date.today().year}).")
        return ParsedTimeWindow(start=date(year, 1, 1), end=date(year, 12, 31))

    raise ValueError(
        f"Cannot parse time_window: '{time_window}'. "
        "Accepted formats: 'YYYY-MM', 'YYYY-MM-DD/YYYY-MM-DD', 'last N days', 'YYYY'."
    )


def _validate_year_month(year: int, month: int):
    if not (1 <= month <= 12):
        raise ValueError(f"Month {month} is invalid (must be 1–12).")
    if not (1981 <= year <= date.today().year + 1):
        raise ValueError(
            f"Year {year} is out of Argo/OISST range (1981–{date.today().year})."
        )


def _last_day_of_month(year: int, month: int) -> date:
    import calendar
    return date(year, month, calendar.monthrange(year, month)[1])


# ---------------------------------------------------------------------------
# Depth layer parser
# ---------------------------------------------------------------------------

def _parse_depth_layer(depth_layer: Optional[str]) -> ParsedDepthLayer:
    # Load from config so defaults stay in one place
    cfg = _load_config()
    layers = cfg.get("depth_layers", {})

    if depth_layer is None:
        depth_layer = "surface"

    key = depth_layer.strip().lower()
    if key not in layers:
        raise ValueError(
            f"Unknown depth_layer: '{depth_layer}'. "
            f"Valid options: {', '.join(layers.keys())}."
        )
    spec = layers[key]
    return ParsedDepthLayer(
        min_dbar=float(spec["min_dbar"]),
        max_dbar=float(spec["max_dbar"]),
        label=spec["label"],
    )


# ---------------------------------------------------------------------------
# Ecosystem type parser
# ---------------------------------------------------------------------------

def _parse_ecosystem(ecosystem_type: str) -> str:
    eco = ecosystem_type.strip().lower()
    if eco not in VALID_ECOSYSTEM_TYPES:
        raise ValueError(
            f"Unknown ecosystem_type: '{ecosystem_type}'. "
            f"Valid options: {', '.join(VALID_ECOSYSTEM_TYPES)}."
        )
    return eco


# ---------------------------------------------------------------------------
# Baseline period parser
# ---------------------------------------------------------------------------

def _parse_baseline(baseline_period: Optional[str]) -> Tuple[date, date]:
    cfg = _load_config()
    defaults = cfg.get("climatology", {}).get("default_period", {})
    default_start = defaults.get("start", "2005-01-01")
    default_end   = defaults.get("end",   "2020-12-31")

    if baseline_period is None:
        return date.fromisoformat(default_start), date.fromisoformat(default_end)

    # "YYYY-YYYY"
    m = re.match(r"^(\d{4})-(\d{4})$", baseline_period.strip())
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if y1 >= y2:
            raise ValueError(
                f"baseline_period start year ({y1}) must be before end year ({y2})."
            )
        if y1 < 1981:
            raise ValueError("baseline_period cannot start before 1981 (Argo/OISST start).")
        return date(y1, 1, 1), date(y2, 12, 31)

    raise ValueError(
        f"Cannot parse baseline_period: '{baseline_period}'. "
        "Expected format: 'YYYY-YYYY' (e.g. '2005-2020')."
    )


# ---------------------------------------------------------------------------
# Config loader (lazy, cached at module level)
# ---------------------------------------------------------------------------

_CFG_CACHE: Optional[dict] = None

def _load_config() -> dict:
    global _CFG_CACHE
    if _CFG_CACHE is not None:
        return _CFG_CACHE
    import os
    import yaml  # type: ignore
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        _CFG_CACHE = yaml.safe_load(fh)
    return _CFG_CACHE


def list_regions() -> list[str]:
    """Return sorted list of all supported named regions."""
    return sorted(_OSE_REGIONS.keys())
