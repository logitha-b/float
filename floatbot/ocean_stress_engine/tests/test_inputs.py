"""
ocean_stress_engine/tests/test_inputs.py
-----------------------------------------
Unit tests for inputs.py:
  - Named region resolution (longest-match, case-insensitive)
  - Bounding box validation
  - Time window parsing (all four formats)
  - Depth layer parsing
  - Ecosystem type validation
  - Baseline period parsing
  - validate_inputs() end-to-end

No network calls.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from datetime import date

from ocean_stress_engine.inputs import (
    validate_inputs,
    _parse_region,
    _parse_time_window,
    _parse_depth_layer,
    _parse_ecosystem,
    _parse_baseline,
    list_regions,
)


# ---------------------------------------------------------------------------
# Region parsing
# ---------------------------------------------------------------------------

class TestParseRegion:

    def test_named_region_case_insensitive(self):
        r = _parse_region("ARABIAN SEA")
        assert r.lat_min == 8
        assert r.lat_max == 26
        assert r.lon_min == 55
        assert r.lon_max == 78

    def test_named_region_lowercase(self):
        r = _parse_region("arabian sea")
        assert r.name == "Arabian Sea"

    def test_named_region_with_surrounding_text(self):
        r = _parse_region("temperature in the arabian sea last month")
        assert r.lon_min == 55

    def test_longest_match_wins_over_shorter(self):
        # "coral sea" is more specific than "sea" alone; ensure correct match
        r = _parse_region("coral sea")
        assert r.lat_min == -26
        assert r.lat_max == -10

    def test_bbox_dict_valid(self):
        r = _parse_region({"lat_min": 10, "lat_max": 20, "lon_min": 60, "lon_max": 70})
        assert r.lat_min == 10
        assert r.lat_max == 20
        assert "Custom bbox" in r.name

    def test_bbox_dict_missing_key_raises(self):
        with pytest.raises(ValueError, match="missing keys"):
            _parse_region({"lat_min": 10, "lat_max": 20, "lon_min": 60})

    def test_bbox_invalid_lat_range_raises(self):
        with pytest.raises(ValueError, match="latitude"):
            _parse_region({"lat_min": 30, "lat_max": 10,
                           "lon_min": 60, "lon_max": 70})

    def test_bbox_invalid_lon_range_raises(self):
        with pytest.raises(ValueError, match="longitude"):
            _parse_region({"lat_min": 10, "lat_max": 20,
                           "lon_min": 70, "lon_max": 60})

    def test_unknown_region_raises_with_suggestion(self):
        with pytest.raises(ValueError, match="Unknown region"):
            _parse_region("magic ocean")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _parse_region("")

    def test_none_raises(self):
        with pytest.raises(ValueError):
            _parse_region(None)

    def test_centre_property(self):
        r = _parse_region("arabian sea")
        lat_c, lon_c = r.centre
        assert lat_c == pytest.approx((8 + 26) / 2)
        assert lon_c == pytest.approx((55 + 78) / 2)

    def test_area_positive(self):
        r = _parse_region("arabian sea")
        assert r.area_deg2 > 0

    def test_list_regions_nonempty(self):
        regions = list_regions()
        assert len(regions) > 0
        assert "arabian sea" in regions


# ---------------------------------------------------------------------------
# Time window parsing
# ---------------------------------------------------------------------------

class TestParseTimeWindow:

    def test_yyyy_mm_format(self):
        tw = _parse_time_window("2023-07")
        assert tw.start == date(2023, 7, 1)
        assert tw.end   == date(2023, 7, 31)

    def test_february_leap_year(self):
        tw = _parse_time_window("2024-02")
        assert tw.end == date(2024, 2, 29)

    def test_february_non_leap_year(self):
        tw = _parse_time_window("2023-02")
        assert tw.end == date(2023, 2, 28)

    def test_date_range_slash(self):
        tw = _parse_time_window("2023-07-01/2023-07-31")
        assert tw.start == date(2023, 7, 1)
        assert tw.end   == date(2023, 7, 31)

    def test_date_range_to(self):
        tw = _parse_time_window("2023-07-01 to 2023-07-31")
        assert tw.start == date(2023, 7, 1)
        assert tw.end   == date(2023, 7, 31)

    def test_last_n_days(self):
        tw = _parse_time_window("last 30 days")
        delta = (tw.end - tw.start).days
        assert delta == 29   # start = end - 29 days → 30 days total

    def test_last_1_day(self):
        tw = _parse_time_window("last 1 day")
        assert tw.start == tw.end

    def test_full_year(self):
        tw = _parse_time_window("2022")
        assert tw.start == date(2022, 1, 1)
        assert tw.end   == date(2022, 12, 31)

    def test_start_before_end(self):
        tw = _parse_time_window("2023-01-01/2023-12-31")
        assert tw.start < tw.end

    def test_calendar_months_single_month(self):
        tw = _parse_time_window("2023-07")
        assert tw.calendar_months == [(2023, 7)]

    def test_calendar_months_multi_month(self):
        tw = _parse_time_window("2023-06-01/2023-08-31")
        assert (2023, 6) in tw.calendar_months
        assert (2023, 7) in tw.calendar_months
        assert (2023, 8) in tw.calendar_months

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError, match="start"):
            _parse_time_window("2023-12-01/2023-01-01")

    def test_invalid_month_raises(self):
        with pytest.raises(ValueError):
            _parse_time_window("2023-13")

    def test_bad_format_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_time_window("July 2023 approximately")

    def test_zero_days_raises(self):
        with pytest.raises(ValueError):
            _parse_time_window("last 0 days")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _parse_time_window("")


# ---------------------------------------------------------------------------
# Depth layer parsing
# ---------------------------------------------------------------------------

class TestParseDepthLayer:

    def test_surface(self):
        dl = _parse_depth_layer("surface")
        assert dl.min_dbar == 0
        assert dl.max_dbar == 50

    def test_mid(self):
        dl = _parse_depth_layer("mid")
        assert dl.min_dbar == 50
        assert dl.max_dbar == 200

    def test_deep(self):
        dl = _parse_depth_layer("deep")
        assert dl.min_dbar == 200
        assert dl.max_dbar == 1000

    def test_none_defaults_to_surface(self):
        dl = _parse_depth_layer(None)
        assert dl.min_dbar == 0
        assert dl.max_dbar == 50

    def test_unknown_layer_raises(self):
        with pytest.raises(ValueError, match="Unknown depth_layer"):
            _parse_depth_layer("abyss")

    def test_label_is_string(self):
        dl = _parse_depth_layer("surface")
        assert isinstance(dl.label, str)
        assert len(dl.label) > 0


# ---------------------------------------------------------------------------
# Ecosystem type parsing
# ---------------------------------------------------------------------------

class TestParseEcosystem:

    def test_valid_types(self):
        for eco in ("coral_reef", "fisheries", "general"):
            assert _parse_ecosystem(eco) == eco

    def test_whitespace_stripped(self):
        assert _parse_ecosystem("  general  ") == "general"

    def test_case_insensitive(self):
        assert _parse_ecosystem("CORAL_REEF") == "coral_reef"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown ecosystem_type"):
            _parse_ecosystem("kelp_forest")


# ---------------------------------------------------------------------------
# Baseline period parsing
# ---------------------------------------------------------------------------

class TestParseBaseline:

    def test_default_none_returns_2005_2020(self):
        start, end = _parse_baseline(None)
        assert start.year == 2005
        assert end.year   == 2020

    def test_custom_period(self):
        start, end = _parse_baseline("2010-2015")
        assert start == date(2010, 1, 1)
        assert end   == date(2015, 12, 31)

    def test_start_must_be_before_end(self):
        with pytest.raises(ValueError, match="start year"):
            _parse_baseline("2020-2010")

    def test_equal_years_raises(self):
        with pytest.raises(ValueError, match="start year"):
            _parse_baseline("2015-2015")

    def test_before_1981_raises(self):
        with pytest.raises(ValueError, match="1981"):
            _parse_baseline("1975-1990")

    def test_bad_format_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_baseline("2005 to 2020")


# ---------------------------------------------------------------------------
# validate_inputs end-to-end
# ---------------------------------------------------------------------------

class TestValidateInputs:

    def test_full_valid_call(self):
        result = validate_inputs(
            region="Arabian Sea",
            time_window="2023-07",
            depth_layer="surface",
            ecosystem_type="coral_reef",
            baseline_period="2005-2020",
        )
        assert result.region.lat_min == 8
        assert result.time_window.start == date(2023, 7, 1)
        assert result.depth_layer.max_dbar == 50
        assert result.ecosystem_type == "coral_reef"
        assert result.baseline_start.year == 2005

    def test_defaults_applied(self):
        result = validate_inputs(
            region="Arabian Sea",
            time_window="2023-07",
        )
        assert result.depth_layer.min_dbar == 0     # surface default
        assert result.ecosystem_type == "general"    # general default
        assert result.baseline_start.year == 2005    # 2005-2020 default

    def test_bbox_region_accepted(self):
        result = validate_inputs(
            region={"lat_min": 10, "lat_max": 20,
                    "lon_min": 60, "lon_max": 70},
            time_window="2023-07",
        )
        assert result.region.lat_min == 10

    def test_invalid_region_propagates(self):
        with pytest.raises(ValueError):
            validate_inputs(region="nowhere ocean", time_window="2023-07")

    def test_invalid_time_propagates(self):
        with pytest.raises(ValueError):
            validate_inputs(region="Arabian Sea", time_window="not-a-date")
