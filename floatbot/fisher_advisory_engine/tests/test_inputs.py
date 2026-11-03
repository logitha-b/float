"""Tests for inputs.py — port lookup, NFC normalization, did-you-mean,
time/trip/language parsing. No network calls."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pytest
from datetime import date
from fisher_advisory_engine.inputs import (
    validate_inputs, _parse_location, _parse_day, _parse_trip_type,
    _parse_boat_type, list_ports,
)


class TestParseDay:
    def test_today(self):
        d = _parse_day("today")
        assert d.start == date.today()
        assert not d.is_future

    def test_tomorrow_is_future(self):
        d = _parse_day("tomorrow")
        assert d.is_future

    def test_last_7_days(self):
        d = _parse_day("last 7 days")
        assert (d.end - d.start).days == 6

    def test_specific_date(self):
        d = _parse_day("2024-06-15")
        assert d.start == date(2024, 6, 15)

    def test_bad_format_raises(self):
        with pytest.raises(ValueError):
            _parse_day("June 2024 sometime")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _parse_day("")

    def test_zero_days_raises(self):
        with pytest.raises(ValueError):
            _parse_day("last 0 days")


class TestParseTripType:
    def test_valid(self):
        assert _parse_trip_type("nearshore") == "nearshore"
        assert _parse_trip_type("offshore")  == "offshore"

    def test_case_insensitive(self):
        assert _parse_trip_type("NEARSHORE") == "nearshore"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            _parse_trip_type("coastal")


class TestParseBoatType:
    def test_none_ok(self):
        from fisher_advisory_engine.inputs import _parse_boat_type
        assert _parse_boat_type(None) is None

    def test_valid(self):
        from fisher_advisory_engine.inputs import _parse_boat_type
        assert _parse_boat_type("traditional") == "traditional"

    def test_unknown_raises(self):
        from fisher_advisory_engine.inputs import _parse_boat_type
        with pytest.raises(ValueError):
            _parse_boat_type("canoe")


class TestParseLocation:
    def test_dict_valid(self):
        loc = _parse_location({"lat": 10.77, "lon": 79.84})
        assert loc.lat == 10.77
        assert loc.verified is True

    def test_dict_missing_key_raises(self):
        with pytest.raises(ValueError, match="missing keys"):
            _parse_location({"lat": 10.77})

    def test_dict_invalid_lat_raises(self):
        with pytest.raises(ValueError, match="latitude"):
            _parse_location({"lat": 999.0, "lon": 79.84})

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _parse_location("")

    def test_unknown_port_suggests(self):
        with pytest.raises(ValueError, match="Unknown location"):
            _parse_location("totally unknown xyz123")

    def test_case_insensitive_match(self):
        # "nagapattinam" in ports.csv (unverified, but resolvable by name)
        loc = _parse_location("NAGAPATTINAM")
        assert "nagapattinam" in loc.name.lower()
        assert loc.verified is False   # all ports unverified

    def test_alias_match(self):
        # "Rameswaram" has alias "Rameshwaram"
        loc = _parse_location("Rameshwaram")
        assert "Rameswaram" in loc.name or "rameswaram" in loc.name.lower()


class TestParseLanguage:
    def test_none_falls_back_to_en(self):
        from fisher_advisory_engine.inputs import _parse_language
        loc = _parse_location({"lat": 10.0, "lon": 80.0})
        langs = _parse_language(None, loc)
        assert "en" in langs

    def test_explicit_en(self):
        from fisher_advisory_engine.inputs import _parse_language
        loc = _parse_location({"lat": 10.0, "lon": 80.0})
        assert _parse_language("en", loc) == ["en"]

    def test_unknown_code_raises(self):
        from fisher_advisory_engine.inputs import _parse_language
        loc = _parse_location({"lat": 10.0, "lon": 80.0})
        with pytest.raises(ValueError, match="Unknown language"):
            _parse_language("xx", loc)

    def test_all_enabled(self):
        from fisher_advisory_engine.inputs import _parse_language
        loc = _parse_location({"lat": 10.0, "lon": 80.0})
        langs = _parse_language("all_enabled", loc)
        assert "en" in langs
        assert isinstance(langs, list)


class TestValidateInputs:
    def test_full_valid_call(self):
        result = validate_inputs(
            location={"lat": 10.77, "lon": 79.84},
            day="today",
            trip_type="nearshore",
            language="en",
        )
        assert result.trip_type == "nearshore"
        assert "en" in result.languages

    def test_defaults(self):
        result = validate_inputs(
            location={"lat": 10.0, "lon": 80.0},
            day="today",
        )
        assert result.trip_type == "nearshore"
        assert result.boat_type is None

    def test_invalid_trip_propagates(self):
        with pytest.raises(ValueError):
            validate_inputs(location={"lat": 10.0, "lon": 80.0},
                            day="today", trip_type="coastal")

    def test_list_ports_nonempty(self):
        ports = list_ports()
        assert len(ports) > 0
