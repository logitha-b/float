"""
fisher_advisory_engine/i18n/registry.py
-----------------------------------------
Language registry: loads all enabled language files, validates required keys,
and exposes helpers for the renderer and review workflow.

On load it:
  1. Reads the language list from config.yaml.
  2. For each enabled language (and draft languages in demo mode), loads
     the corresponding i18n/<code>.yaml file.
  3. Validates:
       a. safety_line key is present (raises if missing — non-removable).
       b. disclaimer key is present (raises if missing).
       c. Every key in the English file exists in this language file
          (missing keys are logged, not raised; they will fall back to English).
  4. Exposes get_string(lang, key, **params) which:
       - Returns the rendered template for key in lang.
       - Falls back to English if the key is missing.
       - Records the fallback so the caller can report it.
       - Raises if the English fallback is also missing.

Adding a new language
---------------------
  1. Create i18n/<code>.yaml with the same keys as en.yaml.
  2. Add an entry to config.yaml under languages:.
  3. Set enabled: true when ready.
  No code changes needed.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_I18N_DIR  = Path(__file__).parent
_CFG_PATH  = _I18N_DIR.parent / "config.yaml"

# Cache: lang_code → dict of key → template string
_LANG_CACHE:  Dict[str, Dict[str, str]] = {}
_LOADED      = False
_REQUIRED_KEYS = {"safety_line", "disclaimer"}

# Keys present in English (ground truth for parity check)
_EN_KEYS: Set[str] = set()


def get_language_config() -> Dict[str, dict]:
    """Return the languages section of config.yaml."""
    import yaml  # type: ignore
    with open(_CFG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("languages", {})


def get_mode() -> str:
    import yaml  # type: ignore
    with open(_CFG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("mode", "demo")


def load_all(force: bool = False) -> None:
    """
    Load all enabled (and, in demo mode, draft) language files.
    Called lazily on first use.  Use force=True to reload after edits.
    """
    global _LOADED, _LANG_CACHE, _EN_KEYS

    if _LOADED and not force:
        return

    import yaml  # type: ignore

    lang_cfg    = get_language_config()
    mode        = get_mode()
    _LANG_CACHE = {}
    _EN_KEYS    = set()   # reset before re-populating

    # Always load English first so we have the ground-truth key set
    _load_one("en", lang_cfg.get("en", {}), strict=True)
    _EN_KEYS = set(_LANG_CACHE.get("en", {}).keys())

    for code, entry in lang_cfg.items():
        if code == "en":
            continue
        if not entry.get("enabled", False):
            continue
        # In production, only reviewed languages are loaded
        if mode == "production" and entry.get("review_status") != "reviewed":
            logger.debug("Skipping %s in production mode (not reviewed)", code)
            continue
        _load_one(code, entry, strict=False)

    _LOADED = True
    logger.debug(
        "FAE i18n loaded: %s",
        ", ".join(f"{c} ({len(v)} keys)" for c, v in _LANG_CACHE.items()),
    )


def _load_one(code: str, entry: dict, strict: bool) -> None:
    """Load and validate a single language file."""
    import yaml  # type: ignore

    path = _I18N_DIR / f"{code}.yaml"
    if not path.exists():
        if strict:
            raise FileNotFoundError(
                f"Required language file not found: {path}. "
                "The English language file (en.yaml) is mandatory."
            )
        logger.warning("Language file missing: %s (skipping)", path)
        return

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    # Validate required keys
    for req in _REQUIRED_KEYS:
        if req not in data:
            msg = (
                f"Language file {path.name} is missing required key '{req}'. "
                f"Every language file MUST contain '{req}'. "
                "This language will not be served until the key is added."
            )
            if strict:
                raise KeyError(msg)
            logger.error(msg)
            return   # do not add this language to the cache

    # Check parity with English (log missing keys, don't raise)
    if code != "en" and _EN_KEYS:
        missing = _EN_KEYS - set(data.keys())
        if missing:
            logger.warning(
                "Language '%s': %d keys missing vs English: %s",
                code, len(missing), sorted(missing)[:10],
            )

    _LANG_CACHE[code] = {k: str(v) for k, v in data.items()}
    logger.debug("Loaded %s: %d keys from %s", code, len(data), path.name)


def get_string(
    lang: str,
    key: str,
    **params,
) -> Tuple[str, bool]:
    """
    Return (rendered_string, fell_back_to_english).

    Parameters
    ----------
    lang   : BCP-47 code
    key    : template key
    params : named values to substitute into the template

    Raises KeyError if the key is missing in both lang and English.
    """
    load_all()

    # Auto-heal: if requested lang has 0 keys it wasn't loaded yet —
    # force a reload (handles Streamlit module-cache edge cases)
    if lang != "en" and len(_LANG_CACHE.get(lang, {})) == 0:
        load_all(force=True)

    lang_data = _LANG_CACHE.get(lang, {})
    en_data   = _LANG_CACHE.get("en", {})
    fell_back = False

    if key in lang_data:
        template = lang_data[key]
    elif key in en_data:
        template = en_data[key]
        fell_back = (lang != "en")
        if fell_back:
            logger.debug("Fallback to English for key '%s' in lang '%s'", key, lang)
    else:
        raise KeyError(
            f"Key '{key}' not found in language '{lang}' or English fallback. "
            "Add this key to i18n/en.yaml first."
        )

    rendered = _render_template(template, params)
    return rendered, fell_back


def _render_template(template: str, params: dict) -> str:
    """
    Substitute {param_name} placeholders in template.
    Unknown placeholders are left unchanged with a warning.
    """
    def _replacer(m):
        name = m.group(1)
        if name in params:
            return str(params[name])
        logger.warning("Template placeholder '{%s}' has no value supplied.", name)
        return m.group(0)   # leave unchanged

    return re.sub(r"\{(\w+)\}", _replacer, template)


def list_languages() -> List[str]:
    """Return list of loaded language codes."""
    load_all()
    return list(_LANG_CACHE.keys())


def key_parity_report() -> Dict[str, List[str]]:
    """
    Return dict: lang → list of keys missing vs English.
    Used by tests and the review workflow.
    """
    load_all()
    report = {}
    en_keys = set(_LANG_CACHE.get("en", {}).keys())
    for code, data in _LANG_CACHE.items():
        if code == "en":
            continue
        missing = sorted(en_keys - set(data.keys()))
        if missing:
            report[code] = missing
    return report


def placeholder_parity_report() -> Dict[str, List[str]]:
    """
    Return dict: lang → list of keys where placeholder set differs from English.
    Uses the same placeholder extraction as the renderer.
    """
    load_all()
    report: Dict[str, List[str]] = {}
    en_data = _LANG_CACHE.get("en", {})

    def _placeholders(tmpl: str) -> Set[str]:
        return set(re.findall(r"\{(\w+)\}", tmpl))

    for code, data in _LANG_CACHE.items():
        if code == "en":
            continue
        mismatched = []
        for key, en_tmpl in en_data.items():
            lang_tmpl = data.get(key)
            if lang_tmpl is None:
                continue   # missing key — reported separately
            en_phs   = _placeholders(en_tmpl)
            lang_phs = _placeholders(lang_tmpl)
            if en_phs != lang_phs:
                mismatched.append(
                    f"{key}: EN has {sorted(en_phs)}, "
                    f"{code} has {sorted(lang_phs)}"
                )
        if mismatched:
            report[code] = mismatched
    return report
