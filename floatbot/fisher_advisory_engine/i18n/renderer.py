"""
fisher_advisory_engine/i18n/renderer.py
-----------------------------------------
Turns an AdvisoryResult into language-specific text by calling
registry.get_string(lang, key, **params).

Three render functions:
  render_short_message(result, lang)  → (str, List[str] fallback_keys)
  render_speech_text(result, lang)    → (str, List[str] fallback_keys)
  render_dashboard_card(result, lang) → (str, List[str] fallback_keys)

Rules
-----
  - Logic never produces display text — it only extracts values from result.
  - Every user-visible string comes from a key in the language file.
  - The safety_line is always last, always from the language file (never
    hardcoded here).
  - A pending-review banner is injected for draft languages in demo mode.
  - Fallback keys (shown in English) are accumulated and returned so the
    caller can record them in result.fallback_used.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# render_short_message
# ---------------------------------------------------------------------------

def render_short_message(
    result, lang: str
) -> Tuple[str, List[str]]:
    """
    Render a short phone-readable message (~5 lines + safety line).
    Returns (text, list_of_fallback_keys).
    """
    from fisher_advisory_engine.i18n.registry import get_string, get_language_config, get_mode

    fallbacks: List[str] = []

    def gs(key: str, **params) -> str:
        text, fell = get_string(lang, key, **params)
        if fell:
            fallbacks.append(key)
        return text

    lines: List[str] = []

    # ── Pending review banner (demo mode only) ────────────────────────────
    _maybe_add_banner(lines, lang, gs, fallbacks)

    # ── Location + day header ─────────────────────────────────────────────
    lines.append(gs(
        "header_short",
        location=result.location_name,
        day=result.day_label,
        trip_type=gs(f"trip_type_{result.trip_type}"),
    ))

    # ── Unverified coordinates warning ───────────────────────────────────
    if not result.location_verified:
        lines.append(gs("unverified_coords_warning"))

    # ── Future day notice ─────────────────────────────────────────────────
    if result.is_future_day:
        lines.append(gs("future_day_notice"))

    # ── SST line ──────────────────────────────────────────────────────────
    sst_ind = result.indicators.get("sst")
    if sst_ind and sst_ind.status in ("ok", "stale"):
        val = sst_ind.value or {}
        sst_c   = val.get("sst_c")
        anom    = val.get("anomaly_c")
        heatwave = val.get("heatwave", False)
        if sst_c is not None:
            anom_str = (
                gs("sst_anomaly_inline",
                   anomaly=f"{anom:+.1f}",
                   direction=gs("above" if anom >= 0 else "below"))
                if anom is not None else ""
            )
            lines.append(gs(
                "sst_line",
                sst_c=f"{sst_c:.1f}",
                anomaly_inline=anom_str,
            ))
            if heatwave:
                lines.append(gs("heatwave_active_short"))
        if sst_ind.status == "stale":
            lines.append(gs("sst_stale_notice",
                            age_h=f"{sst_ind.data_age_hours:.0f}"))
    # else: no SST data — skip silently

    # ── Currents line ─────────────────────────────────────────────────────
    curr_ind = result.indicators.get("currents")
    if curr_ind and curr_ind.status == "ok":
        val = curr_ind.value or {}
        spd   = val.get("speed_ms")
        dir_c = val.get("direction_cardinal", "")
        if spd is not None:
            lines.append(gs(
                "currents_line",
                speed=f"{spd:.2f}",
                direction=gs(f"cardinal_{dir_c.lower()}", **{}),
            ))
    # else: no current data — skip silently

    # ── Productivity line ─────────────────────────────────────────────────
    chl_ind = result.indicators.get("productivity")
    if chl_ind and chl_ind.status == "ok":
        cat = (chl_ind.value or {}).get("category", "no_data")
        if cat and cat != "no_data":
            lines.append(gs("productivity_line", category=gs(f"chl_category_{cat.lower()}")))
    # else: no Chl-a data — skip silently

    # ── Advisory rules (max 2 lines) ─────────────────────────────────────
    for rule in result.rules_triggered[:2]:
        try:
            msg = gs(rule.message_key, **rule.params)
            severity_prefix = gs(f"severity_{rule.severity}")
            lines.append(f"{severity_prefix}: {msg}")
        except Exception as exc:
            logger.debug("Rule %s render failed: %s", rule.rule_id, exc)

    # ── Confidence line ───────────────────────────────────────────────────
    lines.append(gs(
        "confidence_line",
        level=gs(f"confidence_{result.confidence.label.lower()}"),
    ))

    # ── Final advisory recommendation ────────────────────────────────────
    sst_for_final = result.indicators.get("sst")
    sst_c_final   = None
    heatwave_final = False
    anom_final     = None
    if sst_for_final and sst_for_final.status in ("ok", "stale"):
        v = sst_for_final.value or {}
        sst_c_final   = v.get("sst_c")
        heatwave_final = v.get("heatwave", False)
        anom_final     = v.get("anomaly_c")

    lines.append("")   # blank line before final advisory
    lines.append(f"── {gs('final_advisory_heading')} ──")

    if sst_c_final is None:
        lines.append(gs("final_no_sst",
                        location=result.location_name,
                        trip_type=gs(f"trip_type_{result.trip_type}")))
    elif heatwave_final:
        lines.append(gs("final_heatwave",
                        location=result.location_name,
                        sst_c=f"{sst_c_final:.1f}"))
    elif anom_final is not None and anom_final >= 1.0:
        lines.append(gs("final_warm_anomaly",
                        location=result.location_name,
                        sst_c=f"{sst_c_final:.1f}",
                        trip_type=gs(f"trip_type_{result.trip_type}")))
    elif anom_final is not None and anom_final <= -1.0:
        lines.append(gs("final_cool_anomaly",
                        location=result.location_name,
                        sst_c=f"{sst_c_final:.1f}",
                        trip_type=gs(f"trip_type_{result.trip_type}")))
    else:
        lines.append(gs("final_normal",
                        location=result.location_name,
                        trip_type=gs(f"trip_type_{result.trip_type}")))

    lines.append(gs("final_disclaimer_short"))

    # ── Safety line (always last, always from language file) ──────────────
    lines.append(gs("safety_line"))

    return "\n".join(lines), fallbacks


# ---------------------------------------------------------------------------
# render_speech_text
# ---------------------------------------------------------------------------

def render_speech_text(
    result, lang: str
) -> Tuple[str, List[str]]:
    """
    Render a spoken-word version: no symbols, no abbreviations, numbers
    written in a speakable form.  For TTS use in the future.
    """
    from fisher_advisory_engine.i18n.registry import get_string

    fallbacks: List[str] = []

    def gs(key: str, **params) -> str:
        text, fell = get_string(lang, key, **params)
        if fell:
            fallbacks.append(key)
        return text

    lines: List[str] = []

    lines.append(gs(
        "speech_header",
        location=result.location_name,
        day=result.day_label,
    ))

    sst_ind = result.indicators.get("sst")
    if sst_ind and sst_ind.status in ("ok", "stale"):
        val  = sst_ind.value or {}
        sst_c = val.get("sst_c")
        if sst_c is not None:
            lines.append(gs("speech_sst", sst_c=_speakable_number(sst_c, 1)))
        if val.get("heatwave"):
            lines.append(gs("speech_heatwave"))

    curr_ind = result.indicators.get("currents")
    if curr_ind and curr_ind.status == "ok":
        val = curr_ind.value or {}
        spd = val.get("speed_ms")
        if spd is not None:
            lines.append(gs(
                "speech_currents",
                speed=_speakable_number(spd, 2),
                direction=gs(f"cardinal_{(val.get('direction_cardinal','') or '').lower()}"),
            ))

    lines.append(gs(
        "speech_confidence",
        level=gs(f"confidence_{result.confidence.label.lower()}"),
    ))

    lines.append(gs("speech_safety_line"))

    return " ".join(lines), fallbacks


# ---------------------------------------------------------------------------
# render_dashboard_card
# ---------------------------------------------------------------------------

def render_dashboard_card(
    result, lang: str
) -> Tuple[str, List[str]]:
    """
    Render a self-contained HTML card with per-script font stack.
    Language selector, indicators table, confidence, pending-review banner,
    safety line last.
    """
    from fisher_advisory_engine.i18n.registry import (
        get_string, get_language_config, get_mode
    )

    fallbacks: List[str] = []

    def gs(key: str, **params) -> str:
        text, fell = get_string(lang, key, **params)
        if fell:
            fallbacks.append(key)
        return text

    lang_cfg   = get_language_config()
    lang_entry = lang_cfg.get(lang, {})
    font_stack = lang_entry.get("font_stack", "'Segoe UI', sans-serif")
    conf_color = {"HIGH": "#1b7a34", "MEDIUM": "#b35c00", "LOW": "#b71c1c"}.get(
        result.confidence.label, "#555"
    )

    # ── Pending review banner ─────────────────────────────────────────────
    review_status = lang_entry.get("review_status", "draft")
    mode          = get_mode()
    banner_html   = ""
    if mode == "demo" and review_status in ("draft", "pending"):
        banner_text = gs("pending_review_banner")
        banner_html = (
            f'<div style="background:#fff3e0;border:1px solid #ffb74d;'
            f'border-radius:8px;padding:6px 12px;font-size:0.75rem;'
            f'color:#b35c00;margin-bottom:0.7rem">'
            f'⚠️ {banner_text}</div>'
        )

    # ── Unverified warning ────────────────────────────────────────────────
    unverified_html = ""
    if not result.location_verified:
        unverified_html = (
            f'<div style="background:#ffebee;border:1px solid #ef9a9a;'
            f'border-radius:8px;padding:6px 12px;font-size:0.75rem;'
            f'color:#b71c1c;margin-bottom:0.7rem">'
            f'⚠️ {gs("unverified_coords_warning")}</div>'
        )

    # ── Indicator rows ────────────────────────────────────────────────────
    rows_html = ""

    # SST
    sst_ind = result.indicators.get("sst")
    if sst_ind and sst_ind.status in ("ok", "stale"):
        val  = sst_ind.value or {}
        sst_c = val.get("sst_c")
        anom  = val.get("anomaly_c")
        hw    = val.get("heatwave", False)
        if sst_c is not None:
            anom_str = f" ({anom:+.1f}°C)" if anom is not None else ""
            hw_str   = f' <span style="color:#b35c00">⚠ {gs("heatwave_active_short")}</span>' if hw else ""
            rows_html += _card_row(
                gs("indicator_sst"),
                f'{sst_c:.1f}°C{anom_str}{hw_str}',
                f'<span style="font-size:0.7rem;color:#888">{sst_ind.source[:40]}</span>'
            )
    # else: skip SST row silently

    # Currents
    curr_ind = result.indicators.get("currents")
    if curr_ind and curr_ind.status == "ok":
        val   = curr_ind.value or {}
        spd   = val.get("speed_ms")
        dir_c = val.get("direction_cardinal", "")
        if spd is not None:
            direction_label = gs(f"cardinal_{dir_c.lower()}")
            rows_html += _card_row(
                gs("indicator_currents"),
                gs("currents_line", speed=f"{spd:.2f}", direction=direction_label),
                f'<span style="font-size:0.7rem;color:#888">{gs("forecast_label")}</span>'
            )
    # else: skip currents row silently

    # Front
    front_ind = result.indicators.get("thermal_front")
    if front_ind and front_ind.status == "ok":
        val = front_ind.value or {}
        if val.get("detected"):
            front_val = gs(
                "front_detected_inline",
                strength=gs(f"front_strength_{val.get('strength','moderate')}"),
                dist_km=f"{val.get('dist_km',0):.0f}",
            )
            rows_html += _card_row(gs("indicator_front"), front_val, "")
    # else: skip front row silently

    # Productivity
    chl_ind = result.indicators.get("productivity")
    if chl_ind and chl_ind.status == "ok":
        cat       = (chl_ind.value or {}).get("category", "no_data")
        if cat and cat != "no_data":
            chl_label = gs(f"chl_category_{cat.lower()}")
            rows_html += _card_row(gs("indicator_productivity"), chl_label, "")
    # else: skip productivity row silently

    # Confidence reasons
    reasons_html = ""
    if result.confidence.reasons:
        items = "".join(
            f'<li style="font-size:0.74rem;color:#666;margin-bottom:2px">{r}</li>'
            for r in result.confidence.reasons
        )
        reasons_html = f'<ul style="margin:4px 0 0 1rem;padding:0">{items}</ul>'

    # Advisory messages
    advisory_html = ""
    if result.rules_triggered:
        advisory_html = (
            f'<div style="margin-top:0.8rem;font-size:0.82rem;'
            f'font-weight:700;color:#0a2342">{gs("what_this_means_heading")}</div>'
        )
        for rule in result.rules_triggered:
            try:
                msg = gs(rule.message_key, **rule.params)
                sev_color = "#b35c00" if rule.severity == "caution" else "#1565c0"
                advisory_html += (
                    f'<div style="font-size:0.82rem;color:{sev_color};'
                    f'margin-top:4px">• {msg}</div>'
                )
            except Exception:
                pass
    else:
        advisory_html = (
            f'<div style="margin-top:0.8rem;font-size:0.80rem;color:#888">'
            f'{gs("no_advisory_rules_active")}</div>'
        )

    safety_html = (
        f'<div style="margin-top:0.8rem;padding:8px 12px;'
        f'background:#ffebee;border-radius:8px;'
        f'font-size:0.80rem;font-weight:700;color:#b71c1c">'
        f'{gs("safety_line")}</div>'
    )

    disclaimer_html = (
        f'<div style="margin-top:0.5rem;font-size:0.70rem;color:#aaa">'
        f'{gs("disclaimer")}</div>'
    )

    html = f"""
<div style="background:#fff;border:1px solid #d0e4f7;border-radius:14px;
            padding:1.4rem 1.6rem;max-width:560px;
            font-family:{font_stack};
            box-shadow:0 2px 12px rgba(21,101,192,0.07)">

  {banner_html}
  {unverified_html}

  <div style="font-size:0.72rem;font-weight:700;color:#5c7a99;
              text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px">
    🎣 {gs("card_title")}
  </div>
  <div style="font-size:1rem;font-weight:800;color:#0a2342">
    {result.location_name}
  </div>
  <div style="font-size:0.78rem;color:#5c7a99;margin-bottom:0.9rem">
    {result.day_label} &nbsp;·&nbsp; {gs(f"trip_type_{result.trip_type}")}
  </div>

  <table style="width:100%;border-collapse:collapse;font-size:0.82rem">
    {rows_html}
  </table>

  {advisory_html}

  <div style="margin-top:0.8rem;font-size:0.80rem;color:{conf_color};font-weight:700">
    {gs("confidence_label_heading")}: {gs(f"confidence_{result.confidence.label.lower()}")}
  </div>
  {reasons_html}

  {safety_html}
  {disclaimer_html}

  <div style="font-size:0.68rem;color:#bbb;margin-top:0.5rem">
    {gs("data_age_label")}: {_data_age_summary(result)} &nbsp;·&nbsp;
    {gs("computed_at_label")}: {result.computed_at}
  </div>
</div>
"""
    return html, fallbacks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _card_row(label: str, value: str, note: str) -> str:
    return (
        f"<tr>"
        f'<td style="padding:4px 8px;color:#5c7a99;vertical-align:top;'
        f'width:38%;font-weight:600">{label}</td>'
        f'<td style="padding:4px 8px;vertical-align:top">'
        f'{value} {note}</td>'
        f"</tr>"
    )


def _maybe_add_banner(lines, lang, gs_fn, fallbacks):
    from fisher_advisory_engine.i18n.registry import get_language_config, get_mode
    lang_cfg   = get_language_config()
    entry      = lang_cfg.get(lang, {})
    if (get_mode() == "demo" and
            entry.get("review_status") in ("draft", "pending")):
        try:
            text, fell = __import__(
                "fisher_advisory_engine.i18n.registry", fromlist=["get_string"]
            ).get_string(lang, "pending_review_banner")
            if fell:
                fallbacks.append("pending_review_banner")
            lines.append(f"[{text}]")
        except Exception:
            pass


def _speakable_number(value: float, decimals: int) -> str:
    """Format a float for speech: no special symbols."""
    return f"{value:.{decimals}f}"


def _data_age_summary(result) -> str:
    ages = []
    for name, ind in result.indicators.items():
        if ind and getattr(ind, "data_age_hours", None) is not None:
            ages.append(f"{name}: {ind.data_age_hours:.0f}h")
    return ", ".join(ages[:3]) if ages else "unknown"
