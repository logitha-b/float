"""
ocean_stress_engine/tests/test_missing_variables.py
------------------------------------------------------
Unit tests for the missing-data rule:

  "If a variable has no data in the region, do NOT guess or fill it in.
   Re-normalise the weights across the available variables, list the
   missing ones in variables_missing, add a caveat, and lower the
   confidence accordingly."

Tests cover:
  - salinity missing  → weight goes to temperature
  - temperature missing → weight goes to salinity
  - both missing → score is indeterminate (0, no drivers)
  - baseline std=0 / None → variable excluded, weight re-distributed
  - missing variable listed in variables_missing, not variables_used
  - caveat generated for every missing variable
  - confidence degraded when variables are missing
  - compute_stress() with synthetic data, missing salinity

No network calls. Uses mock_data.py and use_synthetic=True.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from ocean_stress_engine.scoring import compute_scores
from ocean_stress_engine.confidence import compute_confidence
from ocean_stress_engine.tests.mock_data import (
    make_baseline,
    make_baseline_no_temp,
    make_baseline_zero_std,
    make_observed,
    make_weights,
    STANDARD_BREAKPOINTS,
)


# ---------------------------------------------------------------------------
# Weight re-normalisation when a variable is missing
# ---------------------------------------------------------------------------

class TestWeightRedistribution:

    def test_salinity_missing_temperature_gets_full_weight(self):
        """When salinity + SST have no observed value, temperature weight → 1.0."""
        result = compute_scores(
            observed=make_observed(temp=29.0, sal=None, sst=None),
            baseline=make_baseline(),
            raw_weights=make_weights("general"),
            breakpoints=STANDARD_BREAKPOINTS,
        )
        assert "salinity" in result.variables_missing
        assert "temperature" in result.variables_used
        temp_driver = next(d for d in result.drivers if d.variable == "temperature")
        assert temp_driver.weight == pytest.approx(1.0, abs=1e-9)

    def test_temperature_missing_salinity_gets_full_weight(self):
        """When temperature + SST are None, salinity weight → 1.0."""
        result = compute_scores(
            observed=make_observed(temp=None, sal=36.0, sst=None),
            baseline=make_baseline(),
            raw_weights=make_weights("general"),
            breakpoints=STANDARD_BREAKPOINTS,
        )
        assert "temperature" in result.variables_missing
        assert "salinity" in result.variables_used
        sal_driver = next(d for d in result.drivers if d.variable == "salinity")
        assert sal_driver.weight == pytest.approx(1.0, abs=1e-9)

    def test_coral_reef_temperature_missing_salinity_takes_all_weight(self):
        """For coral_reef, if temp + SST missing, salinity gets 1.0."""
        result = compute_scores(
            observed=make_observed(temp=None, sal=36.0, sst=None),
            baseline=make_baseline(),
            raw_weights=make_weights("coral_reef"),
            breakpoints=STANDARD_BREAKPOINTS,
        )
        sal_driver = next(d for d in result.drivers if d.variable == "salinity")
        assert sal_driver.weight == pytest.approx(1.0, abs=1e-9)

    def test_effective_weights_always_sum_to_one_with_missing(self):
        result = compute_scores(
            observed=make_observed(temp=29.0, sal=None, sst=None),
            baseline=make_baseline(),
            raw_weights=make_weights("general"),
            breakpoints=STANDARD_BREAKPOINTS,
        )
        total = sum(result.effective_weights.values())
        assert total == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Both variables missing
# ---------------------------------------------------------------------------

class TestBothVariablesMissing:

    def test_both_observed_none_score_is_zero(self):
        result = compute_scores(
            observed=make_observed(temp=None, sal=None, sst=None),
            baseline=make_baseline(),
            raw_weights=make_weights("general"),
            breakpoints=STANDARD_BREAKPOINTS,
        )
        assert result.stress_score == pytest.approx(0.0, abs=0.01)

    def test_both_none_no_drivers(self):
        result = compute_scores(
            observed=make_observed(temp=None, sal=None, sst=None),
            baseline=make_baseline(),
            raw_weights=make_weights("general"),
            breakpoints=STANDARD_BREAKPOINTS,
        )
        assert result.drivers == []

    def test_both_none_variables_used_empty(self):
        result = compute_scores(
            observed=make_observed(temp=None, sal=None, sst=None),
            baseline=make_baseline(),
            raw_weights=make_weights("general"),
            breakpoints=STANDARD_BREAKPOINTS,
        )
        assert result.variables_used == []

    def test_both_none_caveat_mentions_indeterminate(self):
        result = compute_scores(
            observed=make_observed(temp=None, sal=None, sst=None),
            baseline=make_baseline(),
            raw_weights=make_weights("general"),
            breakpoints=STANDARD_BREAKPOINTS,
        )
        assert any("indeterminate" in c.lower() for c in result.caveats)


# ---------------------------------------------------------------------------
# Baseline std = 0 or None
# ---------------------------------------------------------------------------

class TestBaselineStdMissing:

    def test_zero_std_temperature_excluded(self):
        """If temperature baseline std = 0, that variable is excluded."""
        result = compute_scores(
            observed=make_observed(temp=29.0, sal=36.0),
            baseline=make_baseline_zero_std(),
            raw_weights=make_weights("general"),
            breakpoints=STANDARD_BREAKPOINTS,
        )
        assert "temperature" in result.variables_missing
        assert "salinity" in result.variables_used

    def test_none_temperature_baseline_excluded(self):
        """If temperature baseline mean = None, variable is excluded."""
        result = compute_scores(
            observed=make_observed(temp=29.0, sal=36.0),
            baseline=make_baseline_no_temp(),
            raw_weights=make_weights("general"),
            breakpoints=STANDARD_BREAKPOINTS,
        )
        assert "temperature" in result.variables_missing

    def test_zero_std_produces_caveat(self):
        result = compute_scores(
            observed=make_observed(temp=29.0, sal=36.0),
            baseline=make_baseline_zero_std(),
            raw_weights=make_weights("general"),
            breakpoints=STANDARD_BREAKPOINTS,
        )
        temp_caveats = [c for c in result.caveats if "temperature" in c.lower()]
        assert len(temp_caveats) > 0, "Expected a caveat mentioning temperature"

    def test_zero_std_weight_redistributed_to_salinity(self):
        """If temp std=0 (excluded), remaining weight goes to sst + salinity, not sal alone."""
        result = compute_scores(
            observed=make_observed(temp=29.0, sal=36.0, sst=None),  # sst=None so only sal left
            baseline=make_baseline_zero_std(),   # temp std=0 → excluded
            raw_weights=make_weights("general"),
            breakpoints=STANDARD_BREAKPOINTS,
        )
        sal_driver = next(
            (d for d in result.drivers if d.variable == "salinity"), None
        )
        assert sal_driver is not None
        assert sal_driver.weight == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# variables_missing is always populated for BGC vars
# ---------------------------------------------------------------------------

class TestBgcAlwaysMissing:

    def test_oxygen_always_in_missing(self):
        result = compute_scores(
            observed=make_observed(temp=29.0, sal=36.0),
            baseline=make_baseline(),
            raw_weights=make_weights("fisheries"),
            breakpoints=STANDARD_BREAKPOINTS,
        )
        assert "oxygen" in result.variables_missing

    def test_ph_always_in_missing(self):
        result = compute_scores(
            observed=make_observed(temp=29.0, sal=36.0),
            baseline=make_baseline(),
            raw_weights=make_weights("fisheries"),
            breakpoints=STANDARD_BREAKPOINTS,
        )
        assert "ph" in result.variables_missing

    def test_bgc_missing_generates_caveat(self):
        result = compute_scores(
            observed=make_observed(temp=29.0, sal=36.0),
            baseline=make_baseline(),
            raw_weights={"temperature": 0.50, "salinity": 0.25,
                         "oxygen": 0.25, "ph": 0.00},
            breakpoints=STANDARD_BREAKPOINTS,
        )
        oxygen_caveats = [c for c in result.caveats if "oxygen" in c.lower()]
        assert len(oxygen_caveats) > 0


# ---------------------------------------------------------------------------
# Confidence is degraded when variables are missing
# ---------------------------------------------------------------------------

class TestConfidenceDegradedByMissingVars:

    def test_missing_vars_cannot_give_high_confidence(self):
        """Even with many profiles + low exclusion, missing vars cap at MEDIUM."""
        result = compute_confidence(
            n_profiles=20,
            n_readings_total=2000,
            n_readings_excluded=100,      # 5% exclusion — well under HIGH threshold
            nearest_float_dist_deg=1.0,   # very close
            variables_missing=["oxygen", "ph"],
        )
        # Missing variables inject a MEDIUM factor, so result can't be HIGH
        assert result.label in ("MEDIUM", "LOW")

    def test_no_missing_vars_can_achieve_high_confidence(self):
        """Without missing vars and good data, HIGH is achievable."""
        result = compute_confidence(
            n_profiles=20,
            n_readings_total=2000,
            n_readings_excluded=100,
            nearest_float_dist_deg=1.0,
            variables_missing=[],
        )
        assert result.label == "HIGH"

    def test_missing_vars_in_reasons(self):
        result = compute_confidence(
            n_profiles=10,
            n_readings_total=1000,
            n_readings_excluded=50,
            nearest_float_dist_deg=2.0,
            variables_missing=["oxygen"],
        )
        missing_reasons = [r for r in result.reasons if "oxygen" in r.lower()]
        assert len(missing_reasons) > 0

    def test_many_missing_gives_low_score(self):
        result = compute_confidence(
            n_profiles=1,
            n_readings_total=50,
            n_readings_excluded=30,       # 60% excluded → LOW factor
            nearest_float_dist_deg=10.0,  # far → LOW factor
            variables_missing=["oxygen", "ph"],
        )
        assert result.label == "LOW"


# ---------------------------------------------------------------------------
# compute_stress() end-to-end with synthetic data, missing salinity
# ---------------------------------------------------------------------------

class TestComputeStressMissingSalinity:

    def test_synthetic_missing_salinity_via_engine(self, monkeypatch):
        """
        Patch apply_qc_filter so salinity returns value=None,
        then verify the engine propagates the missing-data rule correctly.
        """
        from ocean_stress_engine import engine as eng
        from core.qc_module import QCResult

        def _mock_qc(df, parameter, statistic, depth_min, depth_max):
            if parameter == "salinity":
                return QCResult(
                    parameter="salinity", statistic="mean",
                    value=None, unit="PSU",
                    n_total=100, n_good=0, n_bad=100,
                    confidence_score=0.0, confidence_label="Low",
                    depth_min=depth_min, depth_max=depth_max,
                    depth_note="surface",
                )
            # temperature returns a normal value
            return QCResult(
                parameter="temperature", statistic="mean",
                value=29.5, unit="°C",
                n_total=100, n_good=95, n_bad=5,
                confidence_score=95.0, confidence_label="High",
                depth_min=depth_min, depth_max=depth_max,
                depth_note="surface",
            )

        monkeypatch.setattr(eng, "_apply_qc_and_observe",
                            _patched_apply_qc_and_observe_no_sal)

        from ocean_stress_engine.engine import compute_stress
        result = compute_stress(
            region="Arabian Sea",
            time_window="2023-07",
            depth_layer="surface",
            ecosystem_type="general",
            use_synthetic=True,
        )

        assert "salinity" in result.variables_missing
        assert "temperature" in result.variables_used
        # Temperature driver should have weight ~1.0
        temp_driver = next(
            (d for d in result.drivers if d.variable == "temperature"), None
        )
        assert temp_driver is not None
        assert temp_driver.weight == pytest.approx(1.0, abs=0.01)


def _patched_apply_qc_and_observe_no_sal(df, inputs, use_synthetic=False):
    """
    Replacement for engine._apply_qc_and_observe that returns
    salinity=None to simulate missing salinity data.
    """
    import numpy as np
    observed = {
        "temperature": 29.5,
        "salinity":    None,   # <-- deliberately missing
        "oxygen":      None,
        "ph":          None,
    }
    qc_stats = {
        "n_floats":            5,
        "n_profiles":          15,
        "n_readings_total":    750,
        "n_readings_excluded": 38,
        "exclusion_reasons":   {"spike": 20, "gradient": 18},
        "float_lats":          [17.0, 16.5, 18.0, 15.5, 19.0],
        "float_lons":          [65.0, 66.0, 64.5, 67.0, 63.5],
    }
    from ocean_stress_engine.output import EvidenceRecord
    evidence = [EvidenceRecord(float_id="9999001", cycle=1,
                               depth_m=5.0, lat=17.0, lon=65.0)]
    return observed, qc_stats, evidence
