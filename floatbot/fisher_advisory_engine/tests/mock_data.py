"""
fisher_advisory_engine/tests/mock_data.py
-------------------------------------------
Seeded synthetic factories for all FAE unit tests.
No network calls anywhere in this module.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import date
from typing import Dict, Optional

from fisher_advisory_engine.indicators import IndicatorResult
from fisher_advisory_engine.inputs import (
    ParsedLocation, ParsedDay, ValidatedInputs,
)


# ---------------------------------------------------------------------------
# Location helpers
# ---------------------------------------------------------------------------

def make_location(verified: bool = False) -> ParsedLocation:
    return ParsedLocation(
        lat=10.77 if verified else 0.0,
        lon=79.84 if verified else 0.0,
        name="Test Port",
        district="Nagapattinam",
        verified=verified,
    )


def make_day(future: bool = False) -> ParsedDay:
    d = date(2024, 6, 15)
    return ParsedDay(
        start=d, end=d, label="2024-06-15",
        is_future=future, is_past=not future,
    )


def make_inputs(verified: bool = False, future: bool = False,
                trip_type: str = "nearshore") -> ValidatedInputs:
    return ValidatedInputs(
        location=make_location(verified),
        day=make_day(future),
        trip_type=trip_type,
        boat_type=None,
        languages=["en"],
        raw_language=None,
    )


# ---------------------------------------------------------------------------
# IndicatorResult helpers
# ---------------------------------------------------------------------------

def _ok(name: str, value: dict, unit: str = "", source: str = "synthetic",
        age_h: float = 0.0, caveat: str = None) -> IndicatorResult:
    return IndicatorResult(
        name=name, value=value, unit=unit, status="ok",
        source=source, resolution="synthetic",
        timestamp="2024-06-15T12:00:00Z",
        data_age_hours=age_h, caveat=caveat,
    )


def _no_data(name: str, caveat: str = "no data") -> IndicatorResult:
    return IndicatorResult(
        name=name, value=None, unit="", status="no_data",
        source="synthetic", resolution="synthetic",
        timestamp=None, data_age_hours=None, caveat=caveat,
    )


def make_sst_indicator(
    sst_c: float = 28.5,
    anomaly_c: float = 0.4,
    heatwave: bool = False,
    age_h: float = 10.0,
) -> IndicatorResult:
    return _ok("sst", {
        "sst_c": sst_c, "anomaly_c": anomaly_c, "heatwave": heatwave,
        "baseline_mean_c": sst_c - anomaly_c, "baseline_std_c": 0.8,
        "heatwave_events": [],
    }, unit="°C", age_h=age_h)


def make_current_indicator(
    speed_ms: float = 0.30,
    direction_cardinal: str = "SW",
    age_h: float = 12.0,
) -> IndicatorResult:
    return _ok("currents", {
        "speed_ms": speed_ms, "direction_deg": 225.0,
        "direction_cardinal": direction_cardinal,
        "u_ms": -0.21, "v_ms": -0.21,
        "drifter_speed_ms": None, "drifter_sanity": "not_available",
    }, unit="m/s", age_h=age_h)


def make_front_indicator(detected: bool = False,
                          strength: str = "not_detected") -> IndicatorResult:
    return _ok("thermal_front", {
        "detected": detected, "strength": strength,
        "gradient_c_km": 0.08, "dist_km": 45.0, "bearing_deg": 270.0,
    }, age_h=10.0)


def make_productivity_indicator(
    category: str = "MODERATE",
    chl: float = 0.45,
) -> IndicatorResult:
    return _ok("productivity", {
        "chl_mg_m3": chl, "category": category,
    }, unit="mg/m³", age_h=18.0)


def make_subsurface_indicator(
    mld_m: float = 35.0,
    nearest_km: float = 80.0,
) -> IndicatorResult:
    return _ok("subsurface", {
        "mld_m": mld_m, "nearest_km": nearest_km,
        "float_id": "9999001", "n_profiles": 5,
    }, unit="m", age_h=24.0)


def make_bathymetry_indicator(depth_m: float = 45.0) -> IndicatorResult:
    return _ok("bathymetry", {
        "depth_m": depth_m, "description": f"~{depth_m:.0f} m (continental shelf)",
    }, unit="m", age_h=0.0)


def make_all_indicators(verified: bool = True) -> Dict[str, IndicatorResult]:
    return {
        "sst":           make_sst_indicator(),
        "currents":      make_current_indicator(),
        "thermal_front": make_front_indicator(),
        "productivity":  make_productivity_indicator(),
        "subsurface":    make_subsurface_indicator(),
        "bathymetry":    make_bathymetry_indicator(),
    }


def make_indicators_missing_chl() -> Dict[str, IndicatorResult]:
    inds = make_all_indicators()
    inds["productivity"] = _no_data("productivity", "cloud cover")
    return inds


def make_indicators_missing_subsurface() -> Dict[str, IndicatorResult]:
    inds = make_all_indicators()
    inds["subsurface"] = _no_data("subsurface", "no float nearby")
    return inds


def make_rules_context(
    sst_c: float = 28.5,
    sst_anomaly_c: float = 0.4,
    heatwave: bool = False,
    current_speed_ms: float = 0.30,
    front_detected: bool = False,
    front_strength: str = "not_detected",
    front_dist_km: float = 45.0,
    front_bearing_deg: float = 270.0,
    chl_category: str = "MODERATE",
    mld_m: float = 35.0,
    depth_m: float = 45.0,
) -> dict:
    return dict(
        sst_c=sst_c, sst_anomaly_c=sst_anomaly_c, heatwave=heatwave,
        current_speed_ms=current_speed_ms, current_dir_deg=225.0,
        front_detected=front_detected, front_strength=front_strength,
        front_dist_km=front_dist_km, front_bearing_deg=front_bearing_deg,
        chl_category=chl_category, mld_m=mld_m, depth_m=depth_m,
    )
