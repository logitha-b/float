"""Tests for front_detection.py — synthetic gradient grid."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pytest
import numpy as np
from fisher_advisory_engine.front_detection import (
    detect_front, FrontResult,
    _build_grid, _compute_gradient, _haversine_km, _bearing_deg,
    _synthetic_sst_grid,
)


class TestSyntheticGrid:
    def test_detect_front_synthetic(self):
        result = detect_front(10.77, 79.84, "2024-06-15", use_synthetic=True)
        assert isinstance(result, FrontResult)
        # Synthetic grid has a +2°C step — should detect something
        assert result.status in ("ok", "no_data")
        if result.status == "ok":
            assert result.gradient_c_km is not None
            assert result.gradient_c_km >= 0.0

    def test_synthetic_grid_has_step(self):
        df = _synthetic_sst_grid(10.0, 80.0, "2024-06-15", radius=1.0, seed=42)
        assert not df.empty
        north = df[df["lat"] >= 10.0]["sst"].mean()
        south = df[df["lat"] < 10.0]["sst"].mean()
        assert north > south + 1.0  # deliberate +2°C step

    def test_result_has_caveat(self):
        result = detect_front(10.77, 79.84, "2024-06-15", use_synthetic=True)
        if result.status == "ok":
            assert result.caveat is not None
            assert len(result.caveat) > 10


class TestBuildGrid:
    def test_builds_2d_array(self):
        import pandas as pd
        rows = []
        for la in [10.0, 10.25, 10.5]:
            for lo in [79.75, 80.0]:
                rows.append({"date": pd.Timestamp("2024-06-15"),
                             "lat": la, "lon": lo, "sst": 28.0 + la * 0.1})
        df = pd.DataFrame(rows)
        grid, lats, lons = _build_grid(df)
        assert grid.shape == (3, 2)
        assert len(lats) == 3
        assert len(lons) == 2


class TestComputeGradient:
    def test_uniform_field_zero_gradient(self):
        grid = np.full((5, 5), 28.0)
        lats = np.arange(10.0, 11.25, 0.25)
        lons = np.arange(79.5, 80.75, 0.25)
        peak, pi, pj = _compute_gradient(grid, lats, lons)
        assert peak == pytest.approx(0.0, abs=1e-9)

    def test_step_change_gives_nonzero_gradient(self):
        grid = np.zeros((5, 5))
        grid[2:, :] = 2.0   # +2°C step in lat
        lats = np.linspace(10.0, 11.0, 5)
        lons = np.linspace(79.5, 80.5, 5)
        peak, _, _ = _compute_gradient(grid, lats, lons)
        assert peak > 0.0

    def test_single_row_returns_zero(self):
        grid = np.array([[28.0, 28.5]])
        lats = np.array([10.0])
        lons = np.array([79.5, 80.0])
        peak, _, _ = _compute_gradient(grid, lats, lons)
        assert peak == 0.0


class TestGeometry:
    def test_haversine_same_point(self):
        assert _haversine_km(10.0, 80.0, 10.0, 80.0) == pytest.approx(0.0, abs=0.01)

    def test_haversine_one_degree_lat(self):
        # 1° latitude ≈ 111 km
        d = _haversine_km(10.0, 80.0, 11.0, 80.0)
        assert 109 < d < 113

    def test_bearing_north(self):
        b = _bearing_deg(10.0, 80.0, 11.0, 80.0)
        assert b == pytest.approx(0.0, abs=1.0)

    def test_bearing_east(self):
        b = _bearing_deg(10.0, 80.0, 10.0, 81.0)
        assert abs(b - 90.0) < 2.0
