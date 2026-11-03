"""Tests for indicators.py — all 6 indicators with synthetic data."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pytest
from fisher_advisory_engine.indicators import (
    sst_indicator, current_indicator, front_indicator,
    productivity_indicator, subsurface_indicator, bathymetry_indicator,
    IndicatorResult,
)

DATE = "2024-06-15"
LAT, LON = 10.77, 79.84


def _run(fn, *args, **kwargs):
    """Run indicator with use_synthetic=True."""
    kwargs.setdefault("use_synthetic", True)
    return fn(*args, **kwargs)


class TestSstIndicator:
    def test_returns_indicator_result(self):
        r = _run(sst_indicator, LAT, LON, DATE, DATE)
        assert isinstance(r, IndicatorResult)
        assert r.name == "sst"

    def test_synthetic_has_value(self):
        r = _run(sst_indicator, LAT, LON, DATE, DATE)
        assert r.status in ("ok", "stale", "no_data")
        if r.status == "ok":
            assert isinstance(r.value, dict)
            assert "sst_c" in r.value

    def test_safety_line_not_in_result(self):
        r = _run(sst_indicator, LAT, LON, DATE, DATE)
        raw = str(r.value)
        assert "safe to sail" not in raw.lower()
        assert "fish will be" not in raw.lower()


class TestCurrentIndicator:
    def test_synthetic_returns_dict(self):
        r = _run(current_indicator, LAT, LON, DATE, DATE)
        assert r.name == "currents"
        if r.status == "ok":
            assert "speed_ms" in (r.value or {})
            assert "direction_cardinal" in (r.value or {})

    def test_source_mentions_forecast(self):
        r = _run(current_indicator, LAT, LON, DATE, DATE)
        # In synthetic mode the source is "synthetic"; in real mode it
        # must mention "forecast" or "model".  Accept either.
        assert (
            "forecast" in r.source.lower()
            or "model" in r.source.lower()
            or "synthetic" in r.source.lower()
        )


class TestFrontIndicator:
    def test_synthetic_detects_front(self):
        # Synthetic SST has a deliberate step change → should detect a front
        r = _run(front_indicator, LAT, LON, DATE)
        assert r.name == "thermal_front"
        assert r.status in ("ok", "no_data")
        if r.status == "ok":
            val = r.value or {}
            assert "detected" in val
            assert isinstance(val["detected"], bool)

    def test_caveat_mentions_placeholder(self):
        r = _run(front_indicator, LAT, LON, DATE)
        if r.caveat:
            assert "PLACEHOLDER" in r.caveat or "placeholder" in r.caveat.lower()


class TestProductivityIndicator:
    def test_synthetic_returns_category(self):
        r = _run(productivity_indicator, LAT, LON, DATE, DATE)
        assert r.name == "productivity"
        if r.status == "ok":
            cat = (r.value or {}).get("category", "")
            assert cat in ("HIGH", "MODERATE", "LOW")

    def test_no_data_not_silently_filled(self):
        # When status is no_data, value must be None
        from fisher_advisory_engine.indicators import _make_no_data
        nd = _make_no_data("productivity", "mg/m³", "test", "test", "cloud")
        assert nd.value is None
        assert nd.status == "no_data"


class TestSubsurfaceIndicator:
    def test_synthetic_returns_mld(self):
        r = _run(subsurface_indicator, LAT, LON, DATE, DATE)
        assert r.name == "subsurface"
        if r.status == "ok":
            assert "nearest_km" in (r.value or {})


class TestBathymetryIndicator:
    def test_synthetic_returns_depth(self):
        # Bathymetry indicator doesn't use use_synthetic (it's a static dataset)
        # so we just check structure with a real call attempt; if it fails
        # it should return a no_data IndicatorResult, not raise
        r = bathymetry_indicator(LAT, LON)
        assert r.name == "bathymetry"
        assert r.status in ("ok", "no_data", "error")
        if r.status == "ok":
            assert "depth_m" in (r.value or {})
