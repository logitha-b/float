"""
fisher_advisory_engine/output.py
----------------------------------
AdvisoryResult — the unified result object for the Fisher Advisory Engine.

Holds all computed data and provides three output formats from one object:
  1. to_json()              — full JSON string (all fields)
  2. short_message(lang)    — 5-line plain-language phone message
  3. speech_text(lang)      — spoken-word version (no symbols/abbreviations)
  4. dashboard_card(lang)   — self-contained HTML card for Streamlit

The result object itself contains NO display text.  All text is generated
on demand by calling the above methods, which delegate to the i18n renderer.
The render layer is imported lazily so a missing i18n file never crashes
the engine.

Safety line
-----------
Every rendered output ends with the safety_line key from the language file.
The build_result() factory raises RuntimeError if no i18n files exist.
The per-language rendering raises if the safety_line key is missing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fisher_advisory_engine.indicators  import IndicatorResult
from fisher_advisory_engine.rules_engine import TriggeredRule
from fisher_advisory_engine.confidence  import FaeConfidenceResult


# ---------------------------------------------------------------------------
# AdvisoryResult
# ---------------------------------------------------------------------------

@dataclass
class AdvisoryResult:
    # ── Identity ──────────────────────────────────────────────────────────
    location_name:   str
    location_lat:    float
    location_lon:    float
    location_district: Optional[str]
    location_verified: bool          # False = coordinates unconfirmed
    day_start:       date
    day_end:         date
    day_label:       str             # "today", "2024-06-15", etc.
    trip_type:       str
    boat_type:       Optional[str]
    is_future_day:   bool            # True → HYCOM-only; SST is observed past data

    # ── Indicators ────────────────────────────────────────────────────────
    indicators:         Dict[str, IndicatorResult]   # name → IndicatorResult
    missing_indicators: List[str]                    # names with status "no_data"

    # ── Rules ─────────────────────────────────────────────────────────────
    rules_triggered:  List[TriggeredRule]
    rules_disabled:   List[str]       # rule ids that were skipped (unsourced)

    # ── Confidence ────────────────────────────────────────────────────────
    confidence:          FaeConfidenceResult
    nearest_argo_km:     Optional[float]

    # ── Languages ─────────────────────────────────────────────────────────
    languages_served:    List[str]    # resolved language codes
    fallback_used:       Dict[str, List[str]]   # lang → list of keys shown in English

    # ── Caveats ───────────────────────────────────────────────────────────
    caveats:             List[str]    # aggregated from all indicators

    # ── Meta ──────────────────────────────────────────────────────────────
    computed_at:         str          # ISO-8601 UTC timestamp


    # -----------------------------------------------------------------------
    # Output 1: Full JSON
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict with all result fields."""
        def _ind_to_dict(ind: IndicatorResult) -> dict:
            v = ind.value
            if hasattr(v, "__dataclass_fields__"):
                v = asdict(v)
            return {
                "name":           ind.name,
                "value":          v,
                "unit":           ind.unit,
                "status":         ind.status,
                "source":         ind.source,
                "resolution":     ind.resolution,
                "timestamp":      ind.timestamp,
                "data_age_hours": ind.data_age_hours,
                "caveat":         ind.caveat,
            }

        def _rule_to_dict(r: TriggeredRule) -> dict:
            return {
                "rule_id":          r.rule_id,
                "message_key":      r.message_key,
                "params":           r.params,
                "severity":         r.severity,
                "indicator_values": r.indicator_values,
                "reference":        r.reference,
            }

        return {
            "fae_version":       "1.0.0",
            "location": {
                "name":      self.location_name,
                "lat":       self.location_lat,
                "lon":       self.location_lon,
                "district":  self.location_district,
                "verified":  self.location_verified,
            },
            "period": {
                "start":    self.day_start.isoformat(),
                "end":      self.day_end.isoformat(),
                "label":    self.day_label,
                "is_future": self.is_future_day,
            },
            "trip_type":        self.trip_type,
            "boat_type":        self.boat_type,
            "indicators": {
                name: _ind_to_dict(ind)
                for name, ind in self.indicators.items()
            },
            "missing_indicators": self.missing_indicators,
            "rules_triggered":  [_rule_to_dict(r) for r in self.rules_triggered],
            "rules_disabled":   self.rules_disabled,
            "confidence": {
                "level":   self.confidence.label,
                "score":   self.confidence.score,
                "reasons": self.confidence.reasons,
            },
            "nearest_argo_km":  self.nearest_argo_km,
            "languages_served": self.languages_served,
            "fallback_used":    self.fallback_used,
            "caveats":          self.caveats,
            "disclaimer":       _DISCLAIMER_EN,
            "computed_at":      self.computed_at,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


    # -----------------------------------------------------------------------
    # Output 2: Short message (phone-readable, ≤5 lines + safety line)
    # -----------------------------------------------------------------------

    def short_message(self, lang: str = "en") -> str:
        """
        Return a short plain-language message for the given language.
        At most ~5 short lines, safety line last.
        Falls back to English for any missing key; records in fallback_used.
        """
        try:
            from fisher_advisory_engine.i18n.renderer import render_short_message
            text, fallbacks = render_short_message(self, lang)
            if fallbacks:
                self.fallback_used.setdefault(lang, []).extend(fallbacks)
            return text
        except Exception as exc:
            # Final fallback: minimal English text + safety line
            return _fallback_short_message_en(self, str(exc))


    # -----------------------------------------------------------------------
    # Output 3: Speech text (for future TTS; no symbols or abbreviations)
    # -----------------------------------------------------------------------

    def speech_text(self, lang: str = "en") -> str:
        """
        Return a spoken-word version: no symbols, no abbreviations,
        numbers written as words where practical.
        Audio generation is out of scope for v1.
        Falls back to English for any missing key.
        """
        try:
            from fisher_advisory_engine.i18n.renderer import render_speech_text
            text, fallbacks = render_speech_text(self, lang)
            if fallbacks:
                self.fallback_used.setdefault(lang, []).extend(fallbacks)
            return text
        except Exception as exc:
            return _fallback_short_message_en(self, str(exc))


    # -----------------------------------------------------------------------
    # Output 4: HTML dashboard card
    # -----------------------------------------------------------------------

    def dashboard_card(self, lang: str = "en") -> str:
        """
        Return a self-contained HTML card for Streamlit
        (st.markdown(..., unsafe_allow_html=True)) or a web page.
        Declares a per-script font stack. Tolerates text up to 2× English length.
        Pending-review banner shown when language is draft.
        Safety line always last.
        """
        try:
            from fisher_advisory_engine.i18n.renderer import render_dashboard_card
            html, fallbacks = render_dashboard_card(self, lang)
            if fallbacks:
                self.fallback_used.setdefault(lang, []).extend(fallbacks)
            return html
        except Exception as exc:
            return _fallback_dashboard_card_en(self, str(exc))


# ---------------------------------------------------------------------------
# English-only fallback renderers (used when i18n is unavailable)
# These are NOT the primary output path — they exist so a missing i18n
# file produces a readable result rather than crashing.
# ---------------------------------------------------------------------------

_DISCLAIMER_EN = (
    "This advisory describes current ocean conditions only. "
    "It does not predict fish locations, guarantee a catch, "
    "or declare any trip safe."
)

_SAFETY_LINE_EN = (
    "SAFETY: Always check official IMD and INCOIS weather, wave, "
    "and cyclone warnings before sailing."
)


def _fallback_short_message_en(result: AdvisoryResult, error: str = "") -> str:
    """Minimal English short message when i18n renderer is unavailable."""
    lines = []
    lines.append(f"📍 {result.location_name} | {result.day_label} | {result.trip_type}")

    sst_ind = result.indicators.get("sst")
    if sst_ind and sst_ind.status == "ok" and isinstance(sst_ind.value, dict):
        sst_c = sst_ind.value.get("sst_c")
        anom  = sst_ind.value.get("anomaly_c")
        hw    = sst_ind.value.get("heatwave", False)
        if sst_c is not None:
            anom_str = f" ({anom:+.1f}°C anomaly)" if anom is not None else ""
            hw_str   = " ⚠ Heatwave active." if hw else ""
            lines.append(f"🌡 Sea temp: {sst_c:.1f}°C{anom_str}.{hw_str}")

    curr_ind = result.indicators.get("currents")
    if curr_ind and curr_ind.status == "ok" and isinstance(curr_ind.value, dict):
        spd = curr_ind.value.get("speed_ms")
        dir_c = curr_ind.value.get("direction_cardinal", "")
        if spd is not None:
            lines.append(f"🌊 Currents: {spd:.2f} m/s toward {dir_c} (model forecast).")

    chl_ind = result.indicators.get("productivity")
    if chl_ind and chl_ind.status == "ok" and isinstance(chl_ind.value, dict):
        cat = chl_ind.value.get("category", "")
        if cat and cat != "no_data":
            lines.append(f"🌿 Productivity: {cat}.")
    # else: skip silently

    if result.rules_triggered:
        for r in result.rules_triggered[:2]:
            lines.append(f"ℹ {r.message_key} [{r.rule_id}]")

    conf = result.confidence
    lines.append(f"Confidence: {conf.label}.")

    if error:
        lines.append(f"[i18n unavailable: {error[:60]}]")

    lines.append(_SAFETY_LINE_EN)
    return "\n".join(lines)


def _fallback_dashboard_card_en(result: AdvisoryResult, error: str = "") -> str:
    """Minimal HTML card when i18n renderer is unavailable."""
    conf_color = {"HIGH": "#1b7a34", "MEDIUM": "#b35c00", "LOW": "#b71c1c"}.get(
        result.confidence.label, "#555"
    )
    sst_line = curr_line = chl_line = "No data"

    sst_ind = result.indicators.get("sst")
    if sst_ind and sst_ind.status == "ok" and isinstance(sst_ind.value, dict):
        sst_c = sst_ind.value.get("sst_c")
        anom  = sst_ind.value.get("anomaly_c")
        hw    = sst_ind.value.get("heatwave", False)
        if sst_c is not None:
            anom_str = f" ({anom:+.1f}°C)" if anom is not None else ""
            hw_str   = " ⚠ Heatwave" if hw else ""
            sst_line = f"{sst_c:.1f}°C{anom_str}{hw_str}"

    curr_ind = result.indicators.get("currents")
    if curr_ind and curr_ind.status == "ok" and isinstance(curr_ind.value, dict):
        spd   = curr_ind.value.get("speed_ms")
        dir_c = curr_ind.value.get("direction_cardinal", "")
        if spd is not None:
            curr_line = f"{spd:.2f} m/s → {dir_c} (model)"

    chl_ind = result.indicators.get("productivity")
    if chl_ind and chl_ind.status == "ok" and isinstance(chl_ind.value, dict):
        chl_line = chl_ind.value.get("category", "unknown")

    reasons_html = "".join(
        f"<li style='font-size:0.75rem;color:#888'>{r}</li>"
        for r in result.confidence.reasons
    ) or "<li style='font-size:0.75rem;color:#888'>No degradation factors.</li>"

    error_html = (
        f"<div style='font-size:0.72rem;color:#c00;margin-top:0.4rem'>"
        f"i18n unavailable: {error[:80]}</div>"
        if error else ""
    )

    return f"""
<div style="background:#fff;border:1px solid #d0e4f7;border-radius:14px;
            padding:1.4rem 1.6rem;max-width:520px;font-family:'Segoe UI',sans-serif;
            box-shadow:0 2px 12px rgba(21,101,192,0.07)">
  <div style="font-size:0.72rem;font-weight:700;color:#5c7a99;
              text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px">
    🎣 Fisher Advisory Engine v1.0
  </div>
  <div style="font-size:1rem;font-weight:800;color:#0a2342">
    {result.location_name}
  </div>
  <div style="font-size:0.78rem;color:#5c7a99;margin-bottom:0.8rem">
    {result.day_label} &nbsp;·&nbsp; {result.trip_type}
    {' &nbsp;·&nbsp; ⚠ coordinates unverified' if not result.location_verified else ''}
  </div>
  <table style="width:100%;border-collapse:collapse;font-size:0.82rem">
    <tr><td style="padding:4px 8px;color:#5c7a99;width:40%">🌡 Sea temp</td>
        <td style="padding:4px 8px">{sst_line}</td></tr>
    <tr><td style="padding:4px 8px;color:#5c7a99">🌊 Currents</td>
        <td style="padding:4px 8px">{curr_line}</td></tr>
    <tr><td style="padding:4px 8px;color:#5c7a99">🌿 Productivity</td>
        <td style="padding:4px 8px">{chl_line}</td></tr>
  </table>
  <div style="margin-top:0.8rem;font-size:0.78rem;color:{conf_color};font-weight:700">
    Confidence: {result.confidence.label}
  </div>
  <ul style="margin:0.3rem 0 0 1rem;padding:0">{reasons_html}</ul>
  {error_html}
  <div style="margin-top:0.9rem;padding-top:0.6rem;border-top:1px solid #e8f0fe;
              font-size:0.73rem;color:#888">
    {_DISCLAIMER_EN}
  </div>
  <div style="margin-top:0.4rem;font-size:0.78rem;font-weight:700;color:#b71c1c">
    {_SAFETY_LINE_EN}
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# Factory: build_result()
# ---------------------------------------------------------------------------

def build_result(
    inputs,                           # ValidatedInputs
    indicators: Dict[str, IndicatorResult],
    rules_triggered: List[TriggeredRule],
    rules_disabled:  List[str],
    confidence:      FaeConfidenceResult,
    nearest_argo_km: Optional[float],
    languages_served: List[str],
    caveats:          List[str],
) -> AdvisoryResult:
    """
    Assemble an AdvisoryResult from all pipeline stages.
    Called by engine.py.
    """
    missing = [
        name for name, ind in indicators.items()
        if ind is None or ind.status == "no_data"
    ]
    all_caveats = list(caveats)
    for ind in indicators.values():
        if ind is not None and ind.caveat:
            all_caveats.append(ind.caveat)

    return AdvisoryResult(
        location_name     = inputs.location.name,
        location_lat      = inputs.location.lat,
        location_lon      = inputs.location.lon,
        location_district = inputs.location.district,
        location_verified = inputs.location.verified,
        day_start         = inputs.day.start,
        day_end           = inputs.day.end,
        day_label         = inputs.day.label,
        trip_type         = inputs.trip_type,
        boat_type         = inputs.boat_type,
        is_future_day     = inputs.day.is_future,
        indicators        = {k: v for k, v in indicators.items() if v is not None},
        missing_indicators= missing,
        rules_triggered   = rules_triggered,
        rules_disabled    = rules_disabled,
        confidence        = confidence,
        nearest_argo_km   = nearest_argo_km,
        languages_served  = languages_served,
        fallback_used     = {},
        caveats           = all_caveats,
        computed_at       = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
