"""
fisher_advisory_engine/rules_engine.py
----------------------------------------
Loads rules.yaml and evaluates each rule against a flat indicator context.

Design principles
-----------------
  - DISABLED by default: a rule is only evaluated when its `reference` field
    contains something other than "UNSOURCED - needs citation".
  - Every advisory line shown must trace back to a rule id and the indicator
    values that triggered it (stored in TriggeredRule).
  - The rules engine never produces display text — it emits message_key +
    params.  The i18n renderer turns those into language-specific sentences.
  - Rule conditions are evaluated with Python eval() in a restricted sandbox
    that contains ONLY the indicator variables.  No builtins are accessible.
  - Wording is hedged at the template level ("may", "could"); the rules
    engine itself enforces no wording rules — that is the i18n layer's job.

Public API
----------
    from fisher_advisory_engine.rules_engine import evaluate_rules, TriggeredRule
    triggered = evaluate_rules(context, trip_type="nearshore")
    # triggered : List[TriggeredRule]
    # disabled_rules : List[str]  — rule ids that were skipped

    report = rules_status_report()
    # Returns list of dicts describing every rule and whether it is active.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_UNSOURCED_SENTINEL = "UNSOURCED - needs citation"
_RULES_YAML = Path(__file__).parent / "rules.yaml"


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass
class TriggeredRule:
    rule_id:          str
    message_key:      str
    params:           Dict[str, str]   # filled placeholder values (all strings)
    severity:         str              # "info" | "caution"
    indicator_values: Dict[str, Any]   # snapshot of context at trigger time
    reference:        str              # citation that enabled this rule


# ---------------------------------------------------------------------------
# Public: evaluate rules
# ---------------------------------------------------------------------------

def evaluate_rules(
    context: Dict[str, Any],
    trip_type: str = "nearshore",
) -> tuple[List[TriggeredRule], List[str]]:
    """
    Evaluate all ACTIVE rules against the indicator context.

    Parameters
    ----------
    context   : dict  Flat mapping of indicator variable names → values.
                      Keys expected (all may be None if not available):
                        sst_c, sst_anomaly_c, heatwave,
                        current_speed_ms, current_dir_deg,
                        front_detected, front_strength,
                        front_dist_km, front_bearing_deg,
                        chl_category, mld_m, depth_m
    trip_type : str   "nearshore" | "offshore"

    Returns
    -------
    (triggered, disabled_ids)
      triggered    : list[TriggeredRule]  rules that fired
      disabled_ids : list[str]            rule ids that were skipped (unsourced)
    """
    rules        = _load_rules()
    triggered:   List[TriggeredRule] = []
    disabled_ids: List[str]          = []

    # Add trip_type to context so rules can reference it
    eval_ctx = {**context, "trip_type": trip_type}

    for rule in rules:
        rule_id   = rule.get("id", "UNKNOWN")
        reference = str(rule.get("reference", "")).strip()

        # Disabled check
        if not reference or reference == _UNSOURCED_SENTINEL:
            disabled_ids.append(rule_id)
            continue

        condition   = rule.get("condition", "False")
        message_key = rule.get("message_key", "")
        params_tmpl = rule.get("params") or {}
        severity    = rule.get("severity", "info")

        # Evaluate condition in restricted sandbox
        try:
            fired = _safe_eval(condition, eval_ctx)
        except Exception as exc:
            logger.warning(
                "Rule %s condition evaluation error (skipped): %s", rule_id, exc
            )
            continue

        if not fired:
            continue

        # Fill params: resolve "{expr:.1f}" style templates against context
        filled_params = _fill_params(params_tmpl, eval_ctx)

        triggered.append(TriggeredRule(
            rule_id=rule_id,
            message_key=message_key,
            params=filled_params,
            severity=severity,
            indicator_values={k: eval_ctx.get(k) for k in _CONTEXT_KEYS},
            reference=reference,
        ))
        logger.debug("Rule %s fired (severity=%s)", rule_id, severity)

    return triggered, disabled_ids


# ---------------------------------------------------------------------------
# Public: rules status report
# ---------------------------------------------------------------------------

def rules_status_report() -> List[Dict[str, str]]:
    """
    Return a list of dicts describing every rule and its active status.
    Used by the deliverables report and CLI --report flag.

    Each dict has: id, severity, message_key, active (bool), reference,
                   condition_preview (first 80 chars)
    """
    rules = _load_rules()
    report = []
    for rule in rules:
        reference = str(rule.get("reference", "")).strip()
        active    = bool(reference and reference != _UNSOURCED_SENTINEL)
        condition = str(rule.get("condition", ""))
        report.append({
            "id":                 rule.get("id", "UNKNOWN"),
            "severity":           rule.get("severity", "info"),
            "message_key":        rule.get("message_key", ""),
            "active":             active,
            "reference":          reference if reference else "(none)",
            "condition_preview":  condition[:80] + ("…" if len(condition) > 80 else ""),
        })
    return report


# ---------------------------------------------------------------------------
# Indicator context keys (all may be None)
# ---------------------------------------------------------------------------

_CONTEXT_KEYS = (
    "sst_c",
    "sst_anomaly_c",
    "heatwave",
    "current_speed_ms",
    "current_dir_deg",
    "front_detected",
    "front_strength",
    "front_dist_km",
    "front_bearing_deg",
    "chl_category",
    "mld_m",
    "depth_m",
    "trip_type",
)


# ---------------------------------------------------------------------------
# Internal: safe eval
# ---------------------------------------------------------------------------

def _safe_eval(condition: str, context: Dict[str, Any]) -> bool:
    """
    Evaluate a rule condition string in a restricted sandbox.

    The sandbox contains ONLY the keys in context plus a small set of
    safe built-ins (True, False, None, abs, round, min, max).
    Any attempt to access __builtins__, import, or use other names
    raises a NameError, which is caught and treated as False.
    """
    safe_builtins = {
        "__builtins__": {},  # disable all builtins
        "True":  True,
        "False": False,
        "None":  None,
        "abs":   abs,
        "round": round,
        "min":   min,
        "max":   max,
    }
    sandbox = {**safe_builtins, **context}
    try:
        result = eval(condition, sandbox)  # noqa: S307
        return bool(result)
    except Exception as exc:
        raise RuntimeError(
            f"Condition eval failed: {condition!r} → {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Internal: param filling
# ---------------------------------------------------------------------------

# Matches {expr} or {expr:.Nf} style placeholders in rule params
_PARAM_RE = re.compile(r"\{([^}]+)\}")


def _fill_params(
    params_tmpl: Dict[str, str],
    context: Dict[str, Any],
) -> Dict[str, str]:
    """
    Fill param template strings by evaluating the expression inside braces.

    Example: "{sst_anomaly_c:.1f}" → "1.2"
    If the expression raises, the placeholder is replaced with "N/A".
    """
    result: Dict[str, str] = {}
    for key, tmpl in params_tmpl.items():
        try:
            # Use Python's format_map with a lazy-eval dict
            filled = _PARAM_RE.sub(
                lambda m: _eval_param_expr(m.group(1), context),
                tmpl,
            )
            result[key] = filled
        except Exception as exc:
            logger.debug("Param fill failed for key %s: %s", key, exc)
            result[key] = "N/A"
    return result


def _eval_param_expr(expr: str, context: Dict[str, Any]) -> str:
    """
    Evaluate a single param expression like 'sst_anomaly_c:.1f'.
    Splits on ':' to separate variable name from format spec.
    """
    parts = expr.split(":", 1)
    var_name  = parts[0].strip()
    fmt_spec  = parts[1].strip() if len(parts) > 1 else ""
    try:
        val = context.get(var_name)
        if val is None:
            return "N/A"
        if fmt_spec:
            return format(val, fmt_spec)
        return str(val)
    except Exception:
        return "N/A"


# ---------------------------------------------------------------------------
# Rules loader (cached)
# ---------------------------------------------------------------------------

_RULES_CACHE: Optional[List[dict]] = None


def _load_rules() -> List[dict]:
    global _RULES_CACHE
    if _RULES_CACHE is not None:
        return _RULES_CACHE

    import yaml  # type: ignore
    if not _RULES_YAML.exists():
        logger.error("rules.yaml not found at %s", _RULES_YAML)
        _RULES_CACHE = []
        return _RULES_CACHE

    with open(_RULES_YAML, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    _RULES_CACHE = data.get("rules", []) if isinstance(data, dict) else []
    logger.debug(
        "FAE rules loaded: %d rules (%d active, %d disabled)",
        len(_RULES_CACHE),
        sum(
            1 for r in _RULES_CACHE
            if str(r.get("reference","")).strip() not in ("", _UNSOURCED_SENTINEL)
        ),
        sum(
            1 for r in _RULES_CACHE
            if str(r.get("reference","")).strip() in ("", _UNSOURCED_SENTINEL)
        ),
    )
    return _RULES_CACHE


def reload_rules() -> None:
    """Force a reload of rules.yaml (useful after editing during development)."""
    global _RULES_CACHE
    _RULES_CACHE = None
    _load_rules()
