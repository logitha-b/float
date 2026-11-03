"""Tests for confidence.py — degradation from each factor."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pytest
from fisher_advisory_engine.confidence import compute_confidence
from fisher_advisory_engine.tests.mock_data import (
    make_all_indicators, make_indicators_missing_chl,
    make_indicators_missing_subsurface, make_inputs, _no_data,
)


class TestComputeConfidence:

    def test_unverified_coords_gives_low(self):
        inputs  = make_inputs(verified=False)
        inds    = make_all_indicators(verified=False)
        result  = compute_confidence(inds, inputs, nearest_argo_km=50.0)
        assert result.label == "LOW"
        assert any("unverified" in r.lower() for r in result.reasons)

    def test_missing_chl_degrades_confidence(self):
        inputs = make_inputs(verified=True)
        inds   = make_indicators_missing_chl()
        result = compute_confidence(inds, inputs, nearest_argo_km=50.0)
        # Missing chl injects MEDIUM factor → cannot be HIGH
        assert result.label in ("MEDIUM", "LOW")
        assert any("chlorophyll" in r.lower() or "chl" in r.lower()
                   for r in result.reasons)

    def test_missing_argo_degrades_confidence(self):
        inputs = make_inputs(verified=True)
        inds   = make_all_indicators()
        result = compute_confidence(inds, inputs, nearest_argo_km=None)
        assert result.label in ("MEDIUM", "LOW")
        assert any("argo" in r.lower() for r in result.reasons)

    def test_far_argo_degrades_confidence(self):
        inputs = make_inputs(verified=True)
        inds   = make_all_indicators()
        result = compute_confidence(inds, inputs, nearest_argo_km=300.0)
        assert result.label in ("MEDIUM", "LOW")
        assert any("300" in r for r in result.reasons)

    def test_nearshore_always_medium_or_lower(self):
        # Resolution mismatch factor always injects MEDIUM for nearshore
        inputs = make_inputs(verified=True, trip_type="nearshore")
        inds   = make_all_indicators()
        result = compute_confidence(inds, inputs, nearest_argo_km=50.0)
        assert result.label in ("MEDIUM", "LOW")
        assert any("28 km" in r or "OISST" in r for r in result.reasons)

    def test_stale_sst_degrades_confidence(self):
        inputs = make_inputs(verified=True)
        inds   = make_all_indicators()
        # Override SST age to 200 hours
        inds["sst"].data_age_hours = 200.0
        inds["sst"].status = "stale"
        result = compute_confidence(inds, inputs, nearest_argo_km=50.0)
        assert result.label in ("MEDIUM", "LOW")
        assert any("200" in r for r in result.reasons)

    def test_few_indicators_gives_lower_confidence(self):
        inputs = make_inputs(verified=True)
        # Only 1 indicator with data
        inds = {k: _no_data(k, "missing") for k in
                ["sst","currents","thermal_front","productivity","subsurface"]}
        from fisher_advisory_engine.tests.mock_data import make_bathymetry_indicator
        inds["bathymetry"] = make_bathymetry_indicator()
        result = compute_confidence(inds, inputs, nearest_argo_km=None)
        assert result.label == "LOW"

    def test_reasons_are_strings(self):
        inputs = make_inputs(verified=False)
        inds   = make_all_indicators()
        result = compute_confidence(inds, inputs, nearest_argo_km=200.0)
        assert all(isinstance(r, str) for r in result.reasons)

    def test_score_is_0_to_100(self):
        inputs = make_inputs(verified=True)
        inds   = make_all_indicators()
        result = compute_confidence(inds, inputs, nearest_argo_km=50.0)
        assert 0.0 <= result.score <= 100.0
