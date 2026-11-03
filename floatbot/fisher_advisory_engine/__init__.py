"""
fisher_advisory_engine
----------------------
Fisher Advisory Engine (FAE) — a self-contained FloatAi module that turns
real ocean-condition data into short, plain-language CONDITION advisories
for coastal fishers in Tamil Nadu, in multiple languages.

What it does
------------
Given a landing centre (or lat/lon point), a day, and a trip type, the FAE:
  1. Fetches sea-surface temperature, currents, chlorophyll, and bathymetry
     from existing FloatAi data loaders (read-only)
  2. Runs the Hobday (2016) marine heatwave check via the existing FloatAi
     heatwave module (read-only)
  3. Computes a thermal front indicator from the SST spatial gradient
  4. Retrieves Argo subsurface context (mixed layer depth) if a float is nearby
  5. Evaluates a rules engine (rules.yaml) — every advisory line traces to a
     rule id and the indicator values that triggered it
  6. Computes a confidence level (HIGH/MEDIUM/LOW) from data age, gaps, and
     spatial coverage
  7. Renders the result in one or more languages via the i18n layer

What it does NOT do
-------------------
  - It never predicts exact fish locations
  - It never declares a trip safe or unsafe
  - It never uses machine learning
  - It never presents observation data as a forecast
  - It never machine-translates at runtime

Safety line
-----------
Every output in every language ends with a fixed safety_line advising the
user to check official IMD and INCOIS weather, wave, and cyclone warnings
before sailing.  This line cannot be configured off.

Public API
----------
    from fisher_advisory_engine import compute_advisory
    result = compute_advisory(
        location="Nagapattinam",
        day="today",
        trip_type="nearshore",
        language="en",
    )
    print(result.short_message("en"))
    print(result.to_json())

Deleting this package does NOT affect FloatAi.  The only external touches
are two optional, try/except-guarded hooks:
  - core/query_engine.py  — get_fisher_advisory() appended at bottom
  - app.py                — "🎣 Advisory" page registered
"""

from __future__ import annotations

# Lazy import so that a missing dependency never breaks FloatAi at startup.
# compute_advisory and AdvisoryResult are imported only when actually called.

def compute_advisory(
    location,
    day: str = "today",
    trip_type: str = "nearshore",
    boat_type=None,
    language=None,
):
    """
    Compute a fisher condition advisory for a location and day.

    Parameters
    ----------
    location  : str or dict  Landing centre name (e.g. "Nagapattinam") or
                             {"lat": float, "lon": float}.
    day       : str          "today", "tomorrow", "last 7 days", or "YYYY-MM-DD".
    trip_type : str          "nearshore" | "offshore".
    boat_type : str or None  "traditional" | "mechanized" | None.
    language  : str or list or None
                             BCP-47 code ("en"), list of codes, or "all_enabled".
                             If None, uses the district default from config
                             (currently empty — falls back to "en").

    Returns
    -------
    AdvisoryResult  Full result with .to_json(), .short_message(lang),
                    .speech_text(lang), .dashboard_card(lang) methods.

    Raises
    ------
    ValueError   For invalid inputs (clear message).
    """
    from fisher_advisory_engine.engine import compute_advisory as _compute
    return _compute(
        location=location,
        day=day,
        trip_type=trip_type,
        boat_type=boat_type,
        language=language,
    )


def get_version() -> str:
    return "1.0.0"


__version__ = "1.0.0"
__all__ = ["compute_advisory", "get_version"]
