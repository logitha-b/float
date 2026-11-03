"""
fisher_advisory_engine/inputs.py
----------------------------------
Input validation and parsing for the Fisher Advisory Engine.

Handles:
  - location: landing centre name (NFC-normalised, case-insensitive,
              alias-aware, "did you mean?" on near-miss) or lat/lon dict
  - day: "today" | "tomorrow" | "last 7 days" | "YYYY-MM-DD"
  - trip_type: "nearshore" | "offshore"
  - boat_type: "traditional" | "mechanized" | None
  - language: code string | list of codes | "all_enabled" | None

Port registry is loaded from ports.csv (FAE-local, no nlu imports).
NFC normalisation is applied to all string comparisons.
"""

from __future__ import annotations

import csv
import logging
import os
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PORTS_CSV = Path(__file__).parent / "ports.csv"

VALID_TRIP_TYPES  = ("nearshore", "offshore")
VALID_BOAT_TYPES  = ("traditional", "mechanized")
VALID_DAY_SPECIAL = ("today", "tomorrow", "last 7 days")


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass
class ParsedLocation:
    lat: float
    lon: float
    name: str                        # canonical English name
    district: Optional[str]          # Tamil Nadu district, if known
    verified: bool                   # True only when coordinates are confirmed
    source_row: Optional[dict] = None  # raw ports.csv row, for alias lookup


@dataclass
class ParsedDay:
    start: date
    end: date
    label: str          # human-readable, e.g. "today", "2024-06-15"
    is_future: bool     # True if end > today (only HYCOM/model data available)
    is_past: bool       # True if end < today

    @property
    def start_iso(self) -> str:
        return self.start.isoformat()

    @property
    def end_iso(self) -> str:
        return self.end.isoformat()


@dataclass
class ValidatedInputs:
    location:     ParsedLocation
    day:          ParsedDay
    trip_type:    str
    boat_type:    Optional[str]
    languages:    List[str]        # resolved list of BCP-47 codes to serve
    raw_language: object           # original caller input, for error reporting


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_inputs(
    location,
    day: str = "today",
    trip_type: str = "nearshore",
    boat_type: Optional[str] = None,
    language=None,
) -> ValidatedInputs:
    """
    Parse and validate all FAE inputs.  Raises ValueError with a clear,
    actionable message for every invalid input.

    Parameters
    ----------
    location  : str or dict  Port name or {"lat": float, "lon": float}.
    day       : str          "today", "tomorrow", "last 7 days", or "YYYY-MM-DD".
    trip_type : str          "nearshore" | "offshore".
    boat_type : str or None  "traditional" | "mechanized" | None.
    language  : str | list | None
                             BCP-47 code, list of codes, "all_enabled", or None.
    """
    parsed_location = _parse_location(location)
    parsed_day      = _parse_day(day)
    parsed_trip     = _parse_trip_type(trip_type)
    parsed_boat     = _parse_boat_type(boat_type)
    parsed_langs    = _parse_language(language, parsed_location)

    return ValidatedInputs(
        location=parsed_location,
        day=parsed_day,
        trip_type=parsed_trip,
        boat_type=parsed_boat,
        languages=parsed_langs,
        raw_language=language,
    )


# ---------------------------------------------------------------------------
# Location parser
# ---------------------------------------------------------------------------

def _parse_location(location) -> ParsedLocation:
    if isinstance(location, dict):
        required = {"lat", "lon"}
        missing  = required - set(location.keys())
        if missing:
            raise ValueError(
                f"Location dict missing keys: {missing}. "
                "Required: lat (float), lon (float)."
            )
        lat = float(location["lat"])
        lon = float(location["lon"])
        _validate_latlon(lat, lon)
        return ParsedLocation(
            lat=lat, lon=lon,
            name=f"Custom point ({lat:.4f}°N, {lon:.4f}°E)",
            district=None,
            verified=True,   # user-supplied coords are taken as-is
        )

    if not isinstance(location, str) or not location.strip():
        raise ValueError(
            "location must be a non-empty string (port name) or a dict "
            "with 'lat' and 'lon' keys."
        )

    ports  = _load_ports()
    query  = _nfc(location.strip().lower())
    match  = _find_port(query, ports)

    if match is None:
        suggestions = _did_you_mean(query, ports, n=3)
        hint = (
            f"  Did you mean: {', '.join(suggestions)}?"
            if suggestions else
            "  Use --list-ports to see all available landing centres."
        )
        raise ValueError(
            f"Unknown location: '{location}'.\n{hint}\n"
            "Or pass a dict: {\"lat\": ..., \"lon\": ...}."
        )

    # Warn when coords are unverified (they will be 0.0 / empty from CSV)
    if not match["verified"]:
        logger.warning(
            "Port '%s' has unverified coordinates (verified=false in ports.csv). "
            "Coordinates are TODO — results will be incorrect until verified.",
            match["name"],
        )

    lat_raw = match.get("lat", "")
    lon_raw = match.get("lon", "")
    try:
        lat = float(lat_raw) if lat_raw and lat_raw != "" else 0.0
        lon = float(lon_raw) if lon_raw and lon_raw != "" else 0.0
    except (ValueError, TypeError):
        lat, lon = 0.0, 0.0

    return ParsedLocation(
        lat=lat,
        lon=lon,
        name=match["name"],
        district=match.get("district") or None,
        verified=str(match.get("verified", "false")).lower() == "true",
        source_row=match,
    )


def _validate_latlon(lat: float, lon: float) -> None:
    if not (-90 <= lat <= 90):
        raise ValueError(
            f"Invalid latitude {lat}: must be between -90 and 90."
        )
    if not (-180 <= lon <= 180):
        raise ValueError(
            f"Invalid longitude {lon}: must be between -180 and 180."
        )


# ---------------------------------------------------------------------------
# Day parser
# ---------------------------------------------------------------------------

def _parse_day(day: str) -> ParsedDay:
    if not isinstance(day, str) or not day.strip():
        raise ValueError(
            "day must be a non-empty string: 'today', 'tomorrow', "
            "'last 7 days', or 'YYYY-MM-DD'."
        )

    today     = date.today()
    day_lower = day.strip().lower()

    if day_lower == "today":
        return ParsedDay(
            start=today, end=today,
            label="today",
            is_future=False,
            is_past=False,
        )

    if day_lower == "tomorrow":
        tomorrow = today + timedelta(days=1)
        return ParsedDay(
            start=tomorrow, end=tomorrow,
            label="tomorrow",
            is_future=True,
            is_past=False,
        )

    if day_lower == "last 7 days":
        start = today - timedelta(days=6)
        return ParsedDay(
            start=start, end=today,
            label="last 7 days",
            is_future=False,
            is_past=(today - timedelta(days=6)) < today,
        )

    # "YYYY-MM-DD"
    import re as _re
    m = _re.match(r"^(\d{4})-(\d{2})-(\d{2})$", day.strip())
    if m:
        try:
            d = date.fromisoformat(day.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid date '{day}': {exc}") from exc
        return ParsedDay(
            start=d, end=d,
            label=d.isoformat(),
            is_future=d > today,
            is_past=d < today,
        )

    raise ValueError(
        f"Cannot parse day: '{day}'. "
        "Accepted values: 'today', 'tomorrow', 'last 7 days', or 'YYYY-MM-DD'."
    )


# ---------------------------------------------------------------------------
# Trip type parser
# ---------------------------------------------------------------------------

def _parse_trip_type(trip_type: str) -> str:
    t = trip_type.strip().lower() if isinstance(trip_type, str) else ""
    if t not in VALID_TRIP_TYPES:
        raise ValueError(
            f"Unknown trip_type: '{trip_type}'. "
            f"Valid options: {', '.join(VALID_TRIP_TYPES)}."
        )
    return t


# ---------------------------------------------------------------------------
# Boat type parser
# ---------------------------------------------------------------------------

def _parse_boat_type(boat_type) -> Optional[str]:
    if boat_type is None:
        return None
    b = str(boat_type).strip().lower()
    if b not in VALID_BOAT_TYPES:
        raise ValueError(
            f"Unknown boat_type: '{boat_type}'. "
            f"Valid options: {', '.join(VALID_BOAT_TYPES)}, or None."
        )
    return b


# ---------------------------------------------------------------------------
# Language parser
# ---------------------------------------------------------------------------

def _parse_language(language, location: ParsedLocation) -> List[str]:
    """
    Resolve the caller's language argument to a list of enabled BCP-47 codes.

    Resolution order:
      1. If language is a specific code or list, validate each code exists
         and is enabled (in demo mode: enabled; in production: also reviewed).
      2. If "all_enabled", return all enabled languages from config.
      3. If None, look up the district default from config; if none found,
         fall back to ["en"].
    """
    cfg  = _load_config()
    mode = cfg.get("mode", "demo")
    langs_cfg = cfg.get("languages", {})

    def _is_serveable(code: str) -> bool:
        entry = langs_cfg.get(code)
        if not entry:
            return False
        if not entry.get("enabled", False):
            return False
        if mode == "production" and entry.get("review_status") != "reviewed":
            return False
        return True

    if language is None:
        # District-based default
        district = (location.district or "").lower()
        for code, entry in langs_cfg.items():
            defaults = [d.lower() for d in entry.get("default_for_districts", [])]
            if district and district in defaults and _is_serveable(code):
                return [code]
        return ["en"]

    if isinstance(language, str):
        if language.strip().lower() == "all_enabled":
            return [c for c in langs_cfg if _is_serveable(c)]
        codes = [language.strip().lower()]
    elif isinstance(language, (list, tuple)):
        codes = [str(c).strip().lower() for c in language]
    else:
        raise ValueError(
            f"language must be a string, list of strings, 'all_enabled', or None. "
            f"Got: {type(language).__name__}."
        )

    # Validate each code
    result = []
    unknown = []
    for code in codes:
        if code not in langs_cfg:
            unknown.append(code)
        elif not _is_serveable(code):
            entry = langs_cfg[code]
            if not entry.get("enabled", False):
                raise ValueError(
                    f"Language '{code}' is not enabled. "
                    f"Enable it in config.yaml to use it."
                )
            if mode == "production" and entry.get("review_status") != "reviewed":
                raise ValueError(
                    f"Language '{code}' is not yet reviewed "
                    f"(review_status={entry.get('review_status')!r}). "
                    f"In production mode only 'reviewed' languages are served. "
                    f"Switch to mode: demo in config.yaml to serve draft languages."
                )
        else:
            result.append(code)

    if unknown:
        available = ", ".join(sorted(langs_cfg.keys()))
        raise ValueError(
            f"Unknown language code(s): {unknown}. "
            f"Registered languages: {available}."
        )

    return result if result else ["en"]


# ---------------------------------------------------------------------------
# Port registry (loaded once and cached)
# ---------------------------------------------------------------------------

_PORTS_CACHE: Optional[List[dict]] = None

def _load_ports() -> List[dict]:
    global _PORTS_CACHE
    if _PORTS_CACHE is not None:
        return _PORTS_CACHE

    rows: List[dict] = []
    if not _PORTS_CSV.exists():
        logger.warning("ports.csv not found at %s", _PORTS_CSV)
        _PORTS_CACHE = []
        return _PORTS_CACHE

    with open(_PORTS_CSV, newline="", encoding="utf-8-sig") as fh:
        for line in fh:
            stripped = line.strip()
            # Skip comment lines and blank lines
            if not stripped or stripped.startswith("#"):
                continue
            break
        # Re-open for proper CSV parsing, skipping comment lines
        fh.seek(0)
        non_comment_lines = [
            line for line in fh
            if not line.strip().startswith("#") and line.strip()
        ]

    reader = csv.DictReader(non_comment_lines)
    for row in reader:
        # Skip blank or comment-looking rows
        if not row.get("name", "").strip():
            continue
        if row["name"].strip().startswith("#"):
            continue
        rows.append(dict(row))

    _PORTS_CACHE = rows
    logger.debug("FAE ports loaded: %d entries from %s", len(rows), _PORTS_CSV)
    return _PORTS_CACHE


def _find_port(query_nfc: str, ports: List[dict]) -> Optional[dict]:
    """
    Find a port by NFC-normalised, case-insensitive match against:
      - primary English name
      - transliteration_aliases (pipe-separated)
      - name_ta, name_te, name_ml, name_hi script names
    Longest match wins when multiple ports could match.
    """
    best: Optional[dict] = None
    best_len = 0

    for row in ports:
        candidates = _port_candidates(row)
        for cand in candidates:
            norm = _nfc(cand.strip().lower())
            # Exact match
            if norm == query_nfc and len(norm) > best_len:
                best = row
                best_len = len(norm)
            # Substring match (query contains port name or vice versa)
            elif norm and (norm in query_nfc or query_nfc in norm):
                if len(norm) > best_len:
                    best = row
                    best_len = len(norm)

    return best


def _port_candidates(row: dict) -> List[str]:
    """All searchable name strings for a port row."""
    names = [row.get("name", "")]
    # Script names
    for col in ("name_ta", "name_te", "name_ml", "name_hi"):
        val = row.get(col, "").strip()
        if val:
            names.append(val)
    # Pipe-separated aliases
    aliases_raw = row.get("transliteration_aliases", "")
    if aliases_raw:
        for alias in aliases_raw.split("|"):
            a = alias.strip().strip('"')
            if a:
                names.append(a)
    return [n for n in names if n]


def _did_you_mean(query: str, ports: List[dict], n: int = 3) -> List[str]:
    """
    Return up to n port names with the smallest edit distance to query.
    Uses a simple character-overlap heuristic (no network needed, no imports).
    """
    def _score(cand: str) -> float:
        """Fraction of query bigrams present in candidate."""
        q = _nfc(cand.lower())
        if not query or not q:
            return 0.0
        q_bigrams = {query[i:i+2] for i in range(len(query) - 1)} or {query}
        c_bigrams = {q[i:i+2] for i in range(len(q) - 1)} or {q}
        if not q_bigrams:
            return 0.0
        return len(q_bigrams & c_bigrams) / len(q_bigrams)

    scored = []
    for row in ports:
        for cand in _port_candidates(row):
            s = _score(cand)
            if s > 0.3:
                scored.append((s, row["name"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    # Deduplicate by port name
    seen   = set()
    result = []
    for _, name in scored:
        if name not in seen:
            seen.add(name)
            result.append(name)
        if len(result) >= n:
            break
    return result


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _nfc(s: str) -> str:
    """NFC-normalise a Unicode string."""
    return unicodedata.normalize("NFC", s)


def list_ports() -> List[dict]:
    """Return all port rows from ports.csv (for CLI --list-ports)."""
    return _load_ports()


# ---------------------------------------------------------------------------
# Config loader (lazy, cached)
# ---------------------------------------------------------------------------

_CFG_CACHE: Optional[dict] = None

def _load_config() -> dict:
    global _CFG_CACHE
    if _CFG_CACHE is not None:
        return _CFG_CACHE
    import yaml  # type: ignore
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        _CFG_CACHE = yaml.safe_load(fh)
    return _CFG_CACHE
