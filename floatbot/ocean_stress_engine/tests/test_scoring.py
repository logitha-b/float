"""
ocean_stress_engine/tests/test_scoring.py
------------------------------------------
Unit tests for scoring.py:
  - _zscore_to_subscore interpolation
  - _renormalise_weights
  - _score_to_category boundaries
  - compute_scores end-to-end
  - Sub-score rises with anomaly magnitude
  - Composite score bounded 0-100

No network calls. No Argo data. Uses mock_data.py.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from ocean_stress_engine.scoring import (
    _zscore_to_subscore,
    _renormalise_weights,
    _score_to_category,
    compute_scores,
)
from ocean_stress_engine.tests.mock_data import (
    make_baseline,
    make_observed,
    make_weights,
    STANDARD_BREAKPOINTS,
)


# ---------------------------------------------------------------------------
# _zscore_to_subscore
# ---------------------------------------------------------------------------

class TestZscoreToSubscore:
    BPS = STANDARD_BREAKPOINTS

    def test_at_zero_returns_zero(self):
        assert _zscore_to_subscore(0.0, self.BPS) == 0.0

    def test_at_first_breakpoint(self):
        assert _zscore_to_subscore(1.0, self.BPS) == pytest.approx(25.0)

    def test_at_last_breakpoint(self):
        assert _zscore_to_subscore(4.0, self.BPS) == pytest.approx(100.0)

    def test_beyond_last_clamps_to_100(self):
        assert _zscore_to_subscore(10.0, self.BPS) == 100.0

    def test_midpoint_interpolation(self):
        # z=1.5 → halfway between (1,25) and (2,50) → 37.5
        result = _zscore_to_subscore(1.5, self.BPS)
        assert result == pytest.approx(37.5, abs=0.01)

    def test_monotone_increasing(self):
        zscores = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
        scores  = [_zscore_to_subscore(z, self.BPS) for z in zscores]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], \
                f"Score not monotone at index {i}: {scores[i]} > {scores[i+1]}"

    def test_requires_at_least_two_breakpoints(self):
        with pytest.raises(ValueError):
            _zscore_to_subscore(1.0, [[0, 0]])

    def test_subscore_never_negative(self):
        for z in [0.0, 0.1, 1.0, 2.5, 4.0, 99.0]:
            assert _zscore_to_subscore(z, self.BPS) >= 0.0


# ---------------------------------------------------------------------------
# _renormalise_weights
# ---------------------------------------------------------------------------

class TestRenormaliseWeights:

    def test_sums_to_one(self):
        raw    = {"temperature": 0.70, "salinity": 0.30}
        result = _renormalise_weights(raw, {"temperature", "salinity"})
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)

    def test_subset_available(self):
        # Only temperature available — it gets weight 1.0
        raw    = {"temperature": 0.70, "salinity": 0.30}
        result = _renormalise_weights(raw, {"temperature"})
        assert result["temperature"] == pytest.approx(1.0)
        assert "salinity" not in result

    def test_unequal_raw_weights_preserved_proportionally(self):
        raw    = {"temperature": 3.0, "salinity": 1.0}
        result = _renormalise_weights(raw, {"temperature", "salinity"})
        assert result["temperature"] == pytest.approx(0.75)
        assert result["salinity"]    == pytest.approx(0.25)

    def test_all_zero_weights_falls_back_to_equal(self):
        raw    = {"temperature": 0.0, "salinity": 0.0}
        result = _renormalise_weights(raw, {"temperature", "salinity"})
        assert result["temperature"] == pytest.approx(0.5)
        assert result["salinity"]    == pytest.approx(0.5)

    def test_empty_available_returns_empty(self):
        raw    = {"temperature": 0.7, "salinity": 0.3}
        result = _renormalise_weights(raw, set())
        assert result == {}


# ---------------------------------------------------------------------------
# _score_to_category
# ---------------------------------------------------------------------------

class TestScoreToCategory:

    def test_zero_is_low(self):
        assert _score_to_category(0.0) == "LOW"

    def test_24_9_is_low(self):
        assert _score_to_category(24.9) == "LOW"

    def test_25_is_moderate(self):
        assert _score_to_category(25.0) == "MODERATE"

    def test_49_9_is_moderate(self):
        assert _score_to_category(49.9) == "MODERATE"

    def test_50_is_high(self):
        assert _score_to_category(50.0) == "HIGH"

    def test_74_9_is_high(self):
        assert _score_to_category(74.9) == "HIGH"

    def test_75_is_severe(self):
        assert _score_to_category(75.0) == "SEVERE"

    def test_100_is_severe(self):
        assert _score_to_category(100.0) == "SEVERE"


# ---------------------------------------------------------------------------
# compute_scores end-to-end
# ---------------------------------------------------------------------------

class TestComputeScores:

    def _run(self, observed=None, baseline=None, weights=None):
        obs  = observed or make_observed(temp=29.5, sal=36.0)
        base = baseline or make_baseline(temp_mean=27.0, temp_std=1.0,
                                          sal_mean=35.0,  sal_std=0.5)
        w    = weights  or make_weights("general")
        return compute_scores(obs, base, w, STANDARD_BREAKPOINTS)

    def test_stress_score_bounded_0_100(self):
        result = self._run()
        assert 0.0 <= result.stress_score <= 100.0

    def test_variables_used_populated(self):
        result = self._run()
        assert "temperature" in result.variables_used
        assert "salinity"    in result.variables_used

    def test_no_missing_when_all_available(self):
        result = self._run()
        # oxygen/ph are always missing (BGC only) — temp/sal should be present
        assert "temperature" not in result.variables_missing
        assert "salinity"    not in result.variables_missing

    def test_bgc_vars_always_missing(self):
        result = self._run()
        assert "oxygen" in result.variables_missing
        assert "ph"     in result.variables_missing

    def test_larger_anomaly_gives_higher_score(self):
        # +1σ anomaly
        low = self._run(
            observed=make_observed(temp=28.0, sal=35.5),
            baseline=make_baseline(temp_mean=27.0, temp_std=1.0,
                                    sal_mean=35.0,  sal_std=0.5),
        )
        # +3σ anomaly
        high = self._run(
            observed=make_observed(temp=30.0, sal=36.5),
            baseline=make_baseline(temp_mean=27.0, temp_std=1.0,
                                    sal_mean=35.0,  sal_std=0.5),
        )
        assert high.stress_score > low.stress_score

    def test_at_mean_gives_zero_stress(self):
        # All three vars at their baseline mean → zero stress
        result = self._run(
            observed=make_observed(temp=27.0, sal=35.0, sst=27.5),
            baseline=make_baseline(temp_mean=27.0, temp_std=1.0,
                                    sal_mean=35.0,  sal_std=0.5,
                                    sst_mean=27.5,  sst_std=0.8),
        )
        assert result.stress_score == pytest.approx(0.0, abs=0.01)

    def test_negative_anomaly_still_produces_stress(self):
        # Cold anomaly should also stress marine life
        result = self._run(
            observed=make_observed(temp=24.0, sal=35.0),   # -3σ
            baseline=make_baseline(temp_mean=27.0, temp_std=1.0,
                                    sal_mean=35.0,  sal_std=0.5),
        )
        assert result.stress_score > 0.0

    def test_drivers_sorted_by_sub_score_descending(self):
        result = self._run(
            observed=make_observed(temp=30.0, sal=35.1),   # high temp anomaly, low sal
            baseline=make_baseline(temp_mean=27.0, temp_std=1.0,
                                    sal_mean=35.0,  sal_std=0.5),
        )
        assert len(result.drivers) >= 2
        assert result.drivers[0].sub_score >= result.drivers[1].sub_score

    def test_category_matches_score(self):
        result = self._run(
            observed=make_observed(temp=30.0, sal=36.5),   # high anomaly → HIGH/SEVERE
            baseline=make_baseline(temp_mean=27.0, temp_std=1.0,
                                    sal_mean=35.0,  sal_std=0.5),
        )
        assert result.category in ("HIGH", "SEVERE", "MODERATE")

    def test_effective_weights_sum_to_one(self):
        result = self._run()
        total = sum(result.effective_weights.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_coral_reef_weights_temperature_dominant(self):
        result = self._run(
            observed=make_observed(temp=29.5, sal=36.0),
            baseline=make_baseline(temp_mean=27.0, temp_std=1.0,
                                    sal_mean=35.0,  sal_std=0.5),
            weights=make_weights("coral_reef"),
        )
        temp_driver = next(d for d in result.drivers if d.variable == "temperature")
        sal_driver  = next(d for d in result.drivers if d.variable == "salinity")
        assert temp_driver.weight > sal_driver.weight

    def test_no_observed_returns_indeterminate(self):
        result = compute_scores(
            observed=make_observed(temp=None, sal=None, sst=None),
            baseline=make_baseline(),
            raw_weights=make_weights("general"),
            breakpoints=STANDARD_BREAKPOINTS,
        )
        assert result.stress_score == 0.0
        assert len(result.variables_used) == 0
        assert any("indeterminate" in c.lower() for c in result.caveats)
