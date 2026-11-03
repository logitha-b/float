"""
nlu/rule_parser.py
------------------
Feature 3 (rule-based branch): no-API-key regex/keyword fallback parser.

Produces the same ParsedQuery dict as llm_parser so all downstream code
is parser-agnostic. Handles:

  - Named ocean regions (Gulf of Mexico, North Atlantic, Pacific, etc.)
  - Explicit lat/lon bounding boxes ("between 20N and 30N, 60W and 70W")
  - Temperature / salinity keyword detection
  - Month-year patterns ("July 2023", "in 2023", "Jan–Mar 2022")
  - Relative date phrases ("last month", "this month", "this year")
  - Heatwave / anomaly intent detection
  - Depth keywords ("surface", "deep", "0 to 10 m", etc.)
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named region lookup  (lon_min, lon_max, lat_min, lat_max)
# Longest-match wins, so more-specific names (e.g. "arabian sea") beat
# shorter ones (e.g. "indian ocean") when both appear in a query.
# ---------------------------------------------------------------------------
_NAMED_REGIONS: dict[str, dict] = {
    # Atlantic
    "gulf of mexico":          {"lon_min": -98,  "lon_max": -80,  "lat_min":  18, "lat_max":  31},
    "caribbean sea":           {"lon_min": -85,  "lon_max": -60,  "lat_min":  10, "lat_max":  25},
    "caribbean":               {"lon_min": -85,  "lon_max": -60,  "lat_min":  10, "lat_max":  25},
    "north atlantic ocean":    {"lon_min": -80,  "lon_max":   0,  "lat_min":  25, "lat_max":  65},
    "north atlantic":          {"lon_min": -80,  "lon_max":   0,  "lat_min":  25, "lat_max":  65},
    "south atlantic ocean":    {"lon_min": -60,  "lon_max":  20,  "lat_min": -60, "lat_max":  -5},
    "south atlantic":          {"lon_min": -60,  "lon_max":  20,  "lat_min": -60, "lat_max":  -5},
    "tropical atlantic":       {"lon_min": -60,  "lon_max":  20,  "lat_min": -20, "lat_max":  20},
    "atlantic ocean":          {"lon_min": -80,  "lon_max":  20,  "lat_min": -60, "lat_max":  65},
    "atlantic":                {"lon_min": -80,  "lon_max":  20,  "lat_min": -60, "lat_max":  65},
    # Pacific
    "north pacific ocean":     {"lon_min": 120,  "lon_max": 240,  "lat_min":  20, "lat_max":  65},
    "north pacific":           {"lon_min": 120,  "lon_max": 240,  "lat_min":  20, "lat_max":  65},
    "south pacific ocean":     {"lon_min": 150,  "lon_max": 290,  "lat_min": -60, "lat_max": -10},
    "south pacific":           {"lon_min": 150,  "lon_max": 290,  "lat_min": -60, "lat_max": -10},
    "tropical pacific":        {"lon_min": 140,  "lon_max": 280,  "lat_min": -20, "lat_max":  20},
    "pacific ocean":           {"lon_min": 120,  "lon_max": 290,  "lat_min": -60, "lat_max":  65},
    "pacific":                 {"lon_min": 120,  "lon_max": 290,  "lat_min": -60, "lat_max":  65},
    # Indian Ocean sub-basins (more specific → listed before "indian ocean")
    "arabian sea":             {"lon_min":  55,  "lon_max":  78,  "lat_min":   8, "lat_max":  26},
    "bay of bengal":           {"lon_min":  80,  "lon_max":  100, "lat_min":   5, "lat_max":  23},
    "lakshadweep sea":         {"lon_min":  72,  "lon_max":  78,  "lat_min":   8, "lat_max":  15},
    "andaman sea":             {"lon_min":  92,  "lon_max":  100, "lat_min":   6, "lat_max":  18},
    "mozambique channel":      {"lon_min":  35,  "lon_max":  50,  "lat_min": -26, "lat_max": -10},
    "indian ocean":            {"lon_min":  30,  "lon_max": 120,  "lat_min": -40, "lat_max":  30},
    # Mediterranean / marginal seas
    "mediterranean sea":       {"lon_min":  -6,  "lon_max":  37,  "lat_min":  30, "lat_max":  47},
    "mediterranean":           {"lon_min":  -6,  "lon_max":  37,  "lat_min":  30, "lat_max":  47},
    "black sea":               {"lon_min":  28,  "lon_max":  42,  "lat_min":  41, "lat_max":  47},
    "north sea":               {"lon_min":  -4,  "lon_max":  10,  "lat_min":  51, "lat_max":  62},
    "baltic sea":              {"lon_min":  10,  "lon_max":  30,  "lat_min":  54, "lat_max":  66},
    "norwegian sea":           {"lon_min":  -5,  "lon_max":  30,  "lat_min":  62, "lat_max":  75},
    "labrador sea":            {"lon_min": -60,  "lon_max": -45,  "lat_min":  53, "lat_max":  65},
    # Pacific marginal seas
    "coral sea":               {"lon_min": 142,  "lon_max": 162,  "lat_min": -26, "lat_max": -10},
    "tasman sea":              {"lon_min": 145,  "lon_max": 170,  "lat_min": -48, "lat_max": -30},
    "bering sea":              {"lon_min": 160,  "lon_max": 205,  "lat_min":  52, "lat_max":  66},
    "south china sea":         {"lon_min": 100,  "lon_max": 122,  "lat_min":   0, "lat_max":  25},
    "east china sea":          {"lon_min": 120,  "lon_max": 132,  "lat_min":  24, "lat_max":  34},
    "sea of japan":            {"lon_min": 128,  "lon_max": 142,  "lat_min":  32, "lat_max":  52},
    "japan sea":               {"lon_min": 128,  "lon_max": 142,  "lat_min":  32, "lat_max":  52},
    "philippine sea":          {"lon_min": 126,  "lon_max": 142,  "lat_min":   5, "lat_max":  25},
    "banda sea":               {"lon_min": 122,  "lon_max": 132,  "lat_min": -10, "lat_max":  -2},
    "java sea":                {"lon_min": 104,  "lon_max": 118,  "lat_min":  -8, "lat_max":  -4},
    # Middle East / marginal
    "red sea":                 {"lon_min":  32,  "lon_max":  44,  "lat_min":  12, "lat_max":  30},
    "persian gulf":            {"lon_min":  48,  "lon_max":  57,  "lat_min":  23, "lat_max":  30},
    "gulf of oman":            {"lon_min":  56,  "lon_max":  65,  "lat_min":  20, "lat_max":  27},
    "gulf of aden":            {"lon_min":  42,  "lon_max":  52,  "lat_min":  10, "lat_max":  15},
    # Polar
    "southern ocean":          {"lon_min": -180, "lon_max": 180,  "lat_min": -70, "lat_max": -45},
    "arctic ocean":            {"lon_min": -180, "lon_max": 180,  "lat_min":  70, "lat_max":  90},
    "arctic":                  {"lon_min": -180, "lon_max": 180,  "lat_min":  70, "lat_max":  90},
    # Equatorial shorthand — catches "equator of arabian sea" etc. via sub-string match
    "equator":                 {"lon_min":  -180,"lon_max": 180,  "lat_min":  -5, "lat_max":   5},
}

# Month name → number
_MONTH_NAMES = {
    "january": 1, "jan": 1, "february": 2, "feb": 2,
    "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# Heatwave / anomaly keywords
_HEATWAVE_KEYWORDS = re.compile(
    r"\b(heatwave|heat wave|marine heatwave|mhw|unusual|anomaly|anomalous|"
    r"warm event|warming event|extreme|abnormal|above.?normal|record.?warm)\b",
    re.IGNORECASE,
)

# Coral / bleaching keywords
_CORAL_KEYWORDS = re.compile(
    r"\b(coral|bleach|bleaching|reef|dhw|degree.?heating|crw|coral reef watch)\b",
    re.IGNORECASE,
)

# Profile / measurement keywords
_PROFILE_KEYWORDS = re.compile(
    r"\b(temperature|temp|salinity|salt|salty|psal|warm|cool|cold|fresh|saline|sst|sea surface)\b",
    re.IGNORECASE,
)

# Salinity specific
_SALINITY_KEYWORDS = re.compile(
    r"\b(salinity|salt|salty|psal|fresh|saline)\b", re.IGNORECASE
)

# Depth keywords
_DEPTH_SURFACE = re.compile(r"\bsurface\b", re.IGNORECASE)
_DEPTH_DEEP = re.compile(r"\bdeep\b", re.IGNORECASE)
_DEPTH_RANGE = re.compile(
    r"(\d+)\s*(?:m|dbar|meters?|metres?)\s*(?:to|-)\s*(\d+)\s*(?:m|dbar|meters?|metres?)",
    re.IGNORECASE,
)
_DEPTH_EXPLICIT = re.compile(
    r"(?:depth|below|deeper than|above|shallower than)\s+(\d+)\s*(?:m|dbar|meters?|metres?)?",
    re.IGNORECASE,
)

# Lat/lon explicit coordinates
_LAT_LON = re.compile(
    r"(\d+(?:\.\d+)?)\s*°?\s*([NS])[,\s]+(\d+(?:\.\d+)?)\s*°?\s*([EW])",
    re.IGNORECASE,
)
_LAT_LON_BOX = re.compile(
    r"(?:between|from)\s+"
    r"(\d+(?:\.\d+)?)\s*°?\s*([NS])\s+(?:and|to)\s+(\d+(?:\.\d+)?)\s*°?\s*([NS])"
    r".*?"
    r"(\d+(?:\.\d+)?)\s*°?\s*([EW])\s+(?:and|to)\s+(\d+(?:\.\d+)?)\s*°?\s*([EW])",
    re.IGNORECASE,
)


def parse(question: str, dataset_latest_date: Optional[str] = None) -> dict:
    """
    Parse a plain-English question using rules and return a ParsedQuery dict.

    Parameters
    ----------
    question            : str   The user's raw question.
    dataset_latest_date : str   ISO date of the most recent data point, used to
                                resolve relative date phrases like 'last month'.

    Returns
    -------
    dict matching the ParsedQuery schema.
    """
    q = question.lower()

    intent = _detect_intent(q)
    parameter = _detect_parameter(q)
    region = _detect_region(q)
    start_date, end_date = _detect_dates(q, dataset_latest_date)
    depth_min, depth_max = _detect_depth(q)

    return {
        "intent": intent,
        "parameter": parameter,
        "region": region,
        "start_date": start_date,
        "end_date": end_date,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "raw_question": question,
        "parser_used": "rule_based",
    }


# ---------------------------------------------------------------------------
# Sub-detectors
# ---------------------------------------------------------------------------

def _detect_intent(q: str) -> str:
    has_heatwave = bool(_HEATWAVE_KEYWORDS.search(q))
    has_profile  = bool(_PROFILE_KEYWORDS.search(q))
    has_coral    = bool(_CORAL_KEYWORDS.search(q))

    # Coral queries are always compound: SST heatwave analysis + Argo surface
    if has_coral:
        return "compound_query"
    if has_heatwave and has_profile:
        return "compound_query"
    if has_heatwave:
        return "heatwave_query"
    return "profile_query"


def _detect_parameter(q: str) -> Optional[str]:
    if _SALINITY_KEYWORDS.search(q):
        return "salinity"
    if _PROFILE_KEYWORDS.search(q):
        return "temperature"
    # Coral queries implicitly want SST (temperature)
    if _CORAL_KEYWORDS.search(q):
        return "temperature"
    return None


def _detect_region(q: str) -> Optional[dict]:
    # Try explicit lat/lon bounding box first
    box = _LAT_LON_BOX.search(q)
    if box:
        lat1 = _parse_lat(box.group(1), box.group(2))
        lat2 = _parse_lat(box.group(3), box.group(4))
        lon1 = _parse_lon(box.group(5), box.group(6))
        lon2 = _parse_lon(box.group(7), box.group(8))
        return {
            "lat_min": min(lat1, lat2),
            "lat_max": max(lat1, lat2),
            "lon_min": min(lon1, lon2),
            "lon_max": max(lon1, lon2),
        }

    # Try named region (longest match wins — so "arabian sea" beats "indian ocean"
    # and "north atlantic" beats "atlantic").
    # Strip common prepositions so "equator of arabian sea" → "arabian sea" matches.
    q_stripped = re.sub(r"\b(equator of|waters of|off the coast of|near the|in the)\b", "", q)
    best: Optional[dict] = None
    best_len = 0
    for name, region_box in _NAMED_REGIONS.items():
        if name in q_stripped and len(name) > best_len:
            best = region_box
            best_len = len(name)

    # If "equatorial" + a named ocean appears, intersect with ±5° equatorial band
    if best is not None:
        if re.search(r"\b(equator|equatorial|equator of)\b", q):
            clamped = dict(best)
            clamped["lat_min"] = max(best["lat_min"], -5.0)
            clamped["lat_max"] = min(best["lat_max"],  5.0)
            # Only use the clamped box if the lat band is valid (non-empty)
            if clamped["lat_min"] < clamped["lat_max"]:
                return clamped
            # Otherwise fall through and return the original region box
        return dict(best)

    # Try single lat/lon point → expand to ±2° box
    pt = _LAT_LON.search(q)
    if pt:
        lat = _parse_lat(pt.group(1), pt.group(2))
        lon = _parse_lon(pt.group(3), pt.group(4))
        return {
            "lat_min": lat - 2, "lat_max": lat + 2,
            "lon_min": lon - 2, "lon_max": lon + 2,
        }

    return None


def _detect_dates(
    q: str, dataset_latest_date: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Return (start_date, end_date) as ISO strings."""
    anchor = pd.Timestamp(dataset_latest_date) if dataset_latest_date else pd.Timestamp.now()

    # Relative phrases
    if re.search(r"\blast\s+month\b", q):
        prev = anchor - pd.offsets.MonthBegin(1)
        start = prev.replace(day=1)
        end = (anchor.replace(day=1) - pd.Timedelta(days=1))
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    if re.search(r"\bthis\s+month\b", q):
        start = anchor.replace(day=1)
        return start.strftime("%Y-%m-%d"), anchor.strftime("%Y-%m-%d")

    if re.search(r"\blast\s+year\b", q):
        y = anchor.year - 1
        return f"{y}-01-01", f"{y}-12-31"

    if re.search(r"\bthis\s+year\b", q):
        y = anchor.year
        return f"{y}-01-01", anchor.strftime("%Y-%m-%d")

    # Month + year patterns: "July 2023", "Jul 2023", "07/2023"
    m = re.search(
        r"\b(" + "|".join(_MONTH_NAMES.keys()) + r")[,\s]+(\d{4})\b", q
    )
    if m:
        month_num = _MONTH_NAMES[m.group(1).lower()]
        year = int(m.group(2))
        start, end = _month_bounds(year, month_num)
        return start, end

    # Numeric month/year: 07/2023 or 2023/07
    m = re.search(r"\b(\d{1,2})[/\-](\d{4})\b", q)
    if m:
        month_num, year = int(m.group(1)), int(m.group(2))
        if 1 <= month_num <= 12:
            start, end = _month_bounds(year, month_num)
            return start, end

    # Year only: "in 2023"
    m = re.search(r"\bin\s+(\d{4})\b|\b(\d{4})\b", q)
    if m:
        year_str = m.group(1) or m.group(2)
        if year_str and 1990 <= int(year_str) <= 2100:
            return f"{year_str}-01-01", f"{year_str}-12-31"

    # Explicit date range: "2023-01-01 to 2023-03-31"
    m = re.search(
        r"(\d{4}-\d{2}-\d{2})\s*(?:to|-)\s*(\d{4}-\d{2}-\d{2})", q
    )
    if m:
        return m.group(1), m.group(2)

    return None, None


def _detect_depth(q: str) -> Tuple[Optional[float], Optional[float]]:
    # Explicit range: "10 m to 200 m"
    m = _DEPTH_RANGE.search(q)
    if m:
        return float(m.group(1)), float(m.group(2))

    # Surface shorthand
    if _DEPTH_SURFACE.search(q):
        return 0.0, 10.0

    # Coral queries are always surface (SST relevant)
    if _CORAL_KEYWORDS.search(q):
        return 0.0, 10.0

    # Deep shorthand
    if _DEPTH_DEEP.search(q):
        return 200.0, 2000.0

    return None, None


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _parse_lat(value: str, hemi: str) -> float:
    v = float(value)
    return -v if hemi.upper() == "S" else v


def _parse_lon(value: str, hemi: str) -> float:
    v = float(value)
    return -v if hemi.upper() == "W" else v


def _month_bounds(year: int, month: int) -> Tuple[str, str]:
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}"
