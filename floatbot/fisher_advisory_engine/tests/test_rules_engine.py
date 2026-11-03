"""Tests for rules_engine.py — triggering, disabled-rule behavior, safe eval."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pytest
from fisher_advisory_engine.rules_engine import (
    evaluate_rules, rules_status_report,
    _safe_eval, _fill_params,
)
from fisher_advisory_engine.tests.mock_data import make_rules_context


class TestSafeEval:
    def test_true_condition(self):
        assert _safe_eval("sst_c is not None and sst_c >= 28.0",
                          {"sst_c": 29.0}) is True

    def test_false_condition(self):
        assert _safe_eval("sst_c is not None and sst_c >= 32.0",
                          {"sst_c": 28.0}) is False

    def test_none_value(self):
        assert _safe_eval("sst_c is not None", {"sst_c": None}) is False

    def test_no_builtins_blocked(self):
        with pytest.raises((NameError, RuntimeError)):
            _safe_eval("__import__('os').getcwd()", {})

    def test_unknown_name_is_none(self):
        # Unknown names in sandbox raise NameError → treated as False (not raised)
        # The rules engine wraps this in a try/except; here we test
        # that _safe_eval raises RuntimeError which evaluate_rules catches.
        with pytest.raises(RuntimeError):
            _safe_eval("unknown_var is not None", {})


class TestFillParams:
    def test_float_format(self):
        params = {"anomaly": "{sst_anomaly_c:.1f}"}
        filled = _fill_params(params, {"sst_anomaly_c": 1.234})
        assert filled["anomaly"] == "1.2"

    def test_none_value_gives_na(self):
        params = {"val": "{sst_c:.1f}"}
        filled = _fill_params(params, {"sst_c": None})
        assert filled["val"] == "N/A"

    def test_unknown_key_gives_na(self):
        params = {"val": "{does_not_exist}"}
        filled = _fill_params(params, {})
        assert filled["val"] == "N/A"


class TestEvaluateRules:
    def test_all_rules_disabled_by_default(self):
        ctx = make_rules_context()
        triggered, disabled = evaluate_rules(ctx)
        # All rules have UNSOURCED reference → all disabled
        assert len(triggered) == 0
        assert len(disabled) == 13

    def test_disabled_ids_are_strings(self):
        ctx = make_rules_context()
        _, disabled = evaluate_rules(ctx)
        assert all(isinstance(d, str) for d in disabled)

    def test_triggered_is_empty_list(self):
        ctx = make_rules_context(sst_anomaly_c=5.0, heatwave=True)
        triggered, _ = evaluate_rules(ctx)
        assert isinstance(triggered, list)
        assert len(triggered) == 0  # still disabled

    def test_context_with_none_values_safe(self):
        # Rules must not crash when all values are None
        ctx = {k: None for k in (
            "sst_c", "sst_anomaly_c", "heatwave",
            "current_speed_ms", "current_dir_deg",
            "front_detected", "front_strength",
            "front_dist_km", "front_bearing_deg",
            "chl_category", "mld_m", "depth_m",
        )}
        triggered, disabled = evaluate_rules(ctx)
        assert isinstance(triggered, list)
        assert isinstance(disabled, list)

    def test_rules_status_report_has_correct_count(self):
        report = rules_status_report()
        assert len(report) == 13
        assert all("id" in r and "active" in r for r in report)
        assert all(not r["active"] for r in report)  # all disabled

    def test_rules_report_has_condition_preview(self):
        report = rules_status_report()
        for r in report:
            assert "condition_preview" in r
            assert len(r["condition_preview"]) > 0
