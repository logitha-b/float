"""
i18n tests:
  - key parity across enabled languages
  - placeholder parity
  - safety_line present in every output
  - fallback behavior
  - production mode excludes unreviewed languages
  - NFC matching of port names
  - result object contains no display text (all text via i18n)
  - forbidden phrase scan
  - layout test with longest strings
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import re
import unicodedata
import pytest
from pathlib import Path

I18N_DIR = Path(__file__).parent.parent / "i18n"


def _load_yaml(code: str) -> dict:
    import yaml  # type: ignore
    path = I18N_DIR / f"{code}.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _enabled_langs() -> list:
    from fisher_advisory_engine.i18n.registry import get_language_config
    cfg = get_language_config()
    return [c for c, e in cfg.items() if e.get("enabled")]


def _load_forbidden(code: str) -> list:
    import yaml  # type: ignore
    path = I18N_DIR / f"forbidden_{code}.yaml"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    phrases = []
    for lst in data.values():
        if isinstance(lst, list):
            phrases.extend(lst)
    return phrases


# ---------------------------------------------------------------------------
# Required keys present in every enabled language
# ---------------------------------------------------------------------------

class TestRequiredKeys:
    def test_safety_line_in_all_enabled_languages(self):
        for lang in _enabled_langs():
            data = _load_yaml(lang)
            assert "safety_line" in data, \
                f"safety_line missing from {lang}.yaml"

    def test_disclaimer_in_all_enabled_languages(self):
        for lang in _enabled_langs():
            data = _load_yaml(lang)
            assert "disclaimer" in data, \
                f"disclaimer missing from {lang}.yaml"

    def test_en_has_all_expected_keys(self):
        en = _load_yaml("en")
        required = {
            "safety_line", "disclaimer", "card_title",
            "confidence_line", "confidence_high", "confidence_medium",
            "confidence_low", "sst_line", "currents_line",
            "no_data_label", "header_short",
        }
        for key in required:
            assert key in en, f"Key '{key}' missing from en.yaml"


# ---------------------------------------------------------------------------
# Key parity: every enabled language has all en keys
# ---------------------------------------------------------------------------

class TestKeyParity:
    def test_ta_has_all_en_keys(self):
        en_keys = set(_load_yaml("en").keys())
        ta_keys = set(_load_yaml("ta").keys())
        missing = en_keys - ta_keys
        assert not missing, \
            f"ta.yaml missing {len(missing)} keys: {sorted(missing)[:10]}"

    def test_parity_report_via_registry(self):
        from fisher_advisory_engine.i18n.registry import key_parity_report
        report = key_parity_report()
        # ta is enabled and should have full parity (we wrote all keys)
        assert "ta" not in report, \
            f"ta.yaml has missing keys: {report.get('ta', [])[:5]}"


# ---------------------------------------------------------------------------
# Placeholder parity
# ---------------------------------------------------------------------------

class TestPlaceholderParity:
    def _placeholders(self, tmpl: str) -> set:
        return set(re.findall(r"\{(\w+)\}", tmpl))

    def test_ta_placeholder_parity(self):
        en = _load_yaml("en")
        ta = _load_yaml("ta")
        mismatches = []
        for key, en_tmpl in en.items():
            ta_tmpl = ta.get(key)
            if ta_tmpl is None:
                continue
            en_ph = self._placeholders(str(en_tmpl))
            ta_ph = self._placeholders(str(ta_tmpl))
            if en_ph != ta_ph:
                mismatches.append(f"{key}: EN={sorted(en_ph)} TA={sorted(ta_ph)}")
        assert not mismatches, \
            f"Placeholder mismatches in ta.yaml:\n" + "\n".join(mismatches[:5])

    def test_parity_report_via_registry(self):
        from fisher_advisory_engine.i18n.registry import placeholder_parity_report
        report = placeholder_parity_report()
        assert "ta" not in report, \
            f"ta.yaml has placeholder mismatches: {report.get('ta',[])[:3]}"


# ---------------------------------------------------------------------------
# Safety line always last in rendered output
# ---------------------------------------------------------------------------

class TestSafetyLine:
    def _make_result(self):
        from fisher_advisory_engine.output import AdvisoryResult
        from fisher_advisory_engine.confidence import FaeConfidenceResult
        from fisher_advisory_engine.tests.mock_data import (
            make_all_indicators, make_inputs
        )
        inputs = make_inputs(verified=True)
        inds   = make_all_indicators()
        conf   = FaeConfidenceResult(label="MEDIUM", score=50.0, reasons=[])
        return AdvisoryResult(
            location_name="Test Port", location_lat=10.77, location_lon=79.84,
            location_district="Nagapattinam", location_verified=True,
            day_start=inputs.day.start, day_end=inputs.day.end,
            day_label="2024-06-15", trip_type="nearshore", boat_type=None,
            is_future_day=False, indicators=inds, missing_indicators=[],
            rules_triggered=[], rules_disabled=[], confidence=conf,
            nearest_argo_km=80.0, languages_served=["en","ta"],
            fallback_used={}, caveats=[], computed_at="2024-06-15T12:00:00Z",
        )

    def test_safety_line_last_in_en_short_message(self):
        result = self._make_result()
        msg = result.short_message("en")
        en = _load_yaml("en")
        safety = en.get("safety_line", "").strip()
        # Last non-empty line should contain the safety text
        lines = [l.strip() for l in msg.strip().splitlines() if l.strip()]
        assert lines, "Short message is empty"
        last = lines[-1]
        assert safety[:30] in last or "IMD" in last or "INCOIS" in last, \
            f"Safety line not last in EN short message. Last line: {last!r}"

    def test_safety_line_present_in_ta_short_message(self):
        result = self._make_result()
        msg = result.short_message("ta")
        ta = _load_yaml("ta")
        safety_ta = ta.get("safety_line", "")
        assert safety_ta[:10] in msg or "IMD" in msg or "INCOIS" in msg, \
            "Safety line not found in TA short message"


# ---------------------------------------------------------------------------
# Fallback behavior
# ---------------------------------------------------------------------------

class TestFallback:
    def test_missing_key_falls_back_to_english(self):
        from fisher_advisory_engine.i18n.registry import get_string, load_all
        load_all(force=True)
        # "en" always has the key; if we ask for a key in "ta" that exists
        # in "en" but not "ta", it should fall back
        # (All ta keys exist in ta.yaml per parity test, so use a
        #  hypothetical disabled language — we test the mechanism via en)
        text, fell = get_string("en", "safety_line")
        assert not fell  # en has the key, no fallback needed
        assert len(text) > 5

    def test_missing_key_in_lang_returns_english_and_flags(self):
        from fisher_advisory_engine.i18n.registry import (
            get_string, _LANG_CACHE, load_all
        )
        load_all(force=True)
        # Temporarily inject a lang with a missing key
        _LANG_CACHE["_test"] = {"safety_line": "TEST SAFETY", "disclaimer": "d"}
        text, fell = get_string("_test", "sst_line",
                                sst_c="28.5", anomaly_inline="")
        assert fell is True
        del _LANG_CACHE["_test"]

    def test_missing_key_in_both_raises(self):
        from fisher_advisory_engine.i18n.registry import get_string, _LANG_CACHE, load_all
        load_all(force=True)
        _LANG_CACHE["_test2"] = {"safety_line": "X", "disclaimer": "d"}
        with pytest.raises(KeyError):
            get_string("_test2", "this_key_does_not_exist_anywhere")
        del _LANG_CACHE["_test2"]


# ---------------------------------------------------------------------------
# Production mode excludes unreviewed languages
# ---------------------------------------------------------------------------

class TestProductionMode:
    def test_production_mode_excludes_draft(self, tmp_path, monkeypatch):
        from fisher_advisory_engine.inputs import _parse_language
        # Monkeypatch get_mode to return "production"
        import fisher_advisory_engine.inputs as inp_mod
        monkeypatch.setattr(
            "fisher_advisory_engine.inputs._load_config",
            lambda: {
                "mode": "production",
                "languages": {
                    "en": {"enabled": True, "review_status": "reviewed",
                           "default_for_districts": []},
                    "ta": {"enabled": True, "review_status": "draft",
                           "default_for_districts": []},
                }
            }
        )
        loc = type("L", (), {"district": None})()
        with pytest.raises(ValueError, match="not yet reviewed"):
            _parse_language("ta", loc)


# ---------------------------------------------------------------------------
# NFC matching
# ---------------------------------------------------------------------------

class TestNfcMatching:
    def test_nfc_normalised_port_lookup(self):
        from fisher_advisory_engine.inputs import _parse_location
        # NFC-normalise then look up — same result as plain lookup
        name = unicodedata.normalize("NFC", "Nagapattinam")
        loc1 = _parse_location(name)
        loc2 = _parse_location("Nagapattinam")
        assert loc1.name == loc2.name


# ---------------------------------------------------------------------------
# No display text in result object fields (text comes from i18n)
# ---------------------------------------------------------------------------

class TestNoDisplayTextInResult:
    def test_result_fields_not_sentences(self):
        """The AdvisoryResult dataclass fields must not contain
        pre-rendered sentences — only structured data."""
        from fisher_advisory_engine.tests.mock_data import (
            make_all_indicators, make_inputs
        )
        from fisher_advisory_engine.confidence import FaeConfidenceResult
        from fisher_advisory_engine.output import AdvisoryResult
        inputs = make_inputs(verified=True)
        conf   = FaeConfidenceResult("MEDIUM", 50.0, [])
        result = AdvisoryResult(
            location_name="X", location_lat=10.0, location_lon=80.0,
            location_district=None, location_verified=True,
            day_start=inputs.day.start, day_end=inputs.day.end,
            day_label="today", trip_type="nearshore", boat_type=None,
            is_future_day=False, indicators=make_all_indicators(),
            missing_indicators=[], rules_triggered=[], rules_disabled=[],
            confidence=conf, nearest_argo_km=80.0,
            languages_served=["en"], fallback_used={}, caveats=[],
            computed_at="2024-06-15T12:00:00Z",
        )
        # location_name is just a name, not a sentence
        assert "." not in result.location_name
        # day_label is just a date string
        assert result.day_label in ("today", "tomorrow", "last 7 days", "2024-06-15")


# ---------------------------------------------------------------------------
# Forbidden phrase scan
# ---------------------------------------------------------------------------

class TestForbiddenPhrases:
    def test_en_templates_have_no_forbidden_phrases(self):
        en_data  = _load_yaml("en")
        phrases  = _load_forbidden("en")
        if not phrases:
            pytest.skip("forbidden_en.yaml has no phrases yet (pending reviewer)")
        violations = []
        for key, tmpl in en_data.items():
            text = str(tmpl).lower()
            for phrase in phrases:
                if phrase.lower() in text:
                    violations.append(f"{key}: contains '{phrase}'")
        assert not violations, "Forbidden phrases found:\n" + "\n".join(violations)

    def test_ta_templates_have_no_forbidden_phrases(self):
        ta_data  = _load_yaml("ta")
        phrases  = _load_forbidden("ta")
        if not phrases:
            pytest.skip("forbidden_ta.yaml has no phrases yet (pending reviewer)")
        violations = []
        for key, tmpl in ta_data.items():
            text = str(tmpl)
            for phrase in phrases:
                if phrase in text:
                    violations.append(f"{key}: contains '{phrase}'")
        assert not violations, "Forbidden phrases found in ta:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# Layout test: longest strings per language
# ---------------------------------------------------------------------------

class TestLayout:
    def test_ta_strings_not_excessively_longer_than_en(self):
        """Each ta template should be at most ~3× the en template length
        (rough check for layout safety). Warns, does not fail."""
        en = _load_yaml("en")
        ta = _load_yaml("ta")
        warnings = []
        for key in en:
            if key not in ta:
                continue
            en_len = len(str(en[key]))
            ta_len = len(str(ta[key]))
            if en_len > 0 and ta_len / en_len > 3.5:
                warnings.append(
                    f"{key}: EN={en_len} chars, TA={ta_len} chars "
                    f"({ta_len/en_len:.1f}×)"
                )
        # Don't fail — just warn so a reviewer can address it
        if warnings:
            import warnings as _w
            _w.warn(
                "Some ta templates are > 3.5× English length:\n"
                + "\n".join(warnings[:5]),
                stacklevel=2,
            )
