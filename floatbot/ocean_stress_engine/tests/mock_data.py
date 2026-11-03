"""
ocean_stress_engine/tests/mock_data.py
----------------------------------------
Seeded synthetic data factories for OSE unit tests.

All functions are deterministic given the same seed.
No network calls are made anywhere in this module.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ocean_stress_engine.climatology import BaselineStats
from ocean_stress_engine.inputs      import ParsedRegion, ParsedTimeWindow, ParsedDepthLayer
from ocean_stress_engine.inputs      import ValidatedInputs


# ---------------------------------------------------------------------------
# Argo DataFrame factory
# ---------------------------------------------------------------------------

def make_argo_df(
    n_profiles:    int   = 10,
    n_levels:      int   = 5,
    temp_mean:     float = 28.0,
    temp_std:      float = 0.5,
    sal_mean:      float = 35.5,
    sal_std:       float = 0.3,
    bad_frac:      float = 0.05,   # fraction of readings with QC flag 4
    seed:          int   = 42,
) -> pd.DataFrame:
    """
    Build a synthetic Argo DataFrame matching the standard schema used by
    data/argo_loader.py. Fully deterministic given the same seed.

    Parameters
    ----------
    n_profiles  : number of float profiles
    n_levels    : pressure levels per profile
    temp_mean   : mean temperature (°C)
    temp_std    : std of temperature
    sal_mean    : mean salinity (PSU)
    sal_std     : std of salinity
    bad_frac    : fraction of readings to mark QC=4 (bad)
    seed        : random seed for reproducibility
    """
    rng  = np.random.default_rng(seed)
    rows = []

    for p in range(n_profiles):
        platform = str(1000000 + p)
        cycle    = p + 1
        lat      = float(rng.uniform(10.0, 20.0))
        lon      = float(rng.uniform(60.0, 70.0))
        t        = pd.Timestamp("2023-07-15")

        for level in range(n_levels):
            pres = float(level * 10)
            temp = float(rng.normal(temp_mean, temp_std))
            sal  = float(rng.normal(sal_mean,  sal_std))

            # Assign QC flags: mostly good (1), a few bad (4)
            is_bad = rng.random() < bad_frac
            temp_qc = 4 if is_bad else 1
            sal_qc  = 4 if is_bad else 1

            rows.append({
                "PLATFORM_NUMBER": platform,
                "CYCLE_NUMBER":    cycle,
                "LATITUDE":        lat,
                "LONGITUDE":       lon,
                "TIME":            t,
                "PRES":            pres,
                "TEMP":            round(temp, 4),
                "PSAL":            round(sal,  4),
                "TEMP_QC":         temp_qc,
                "PSAL_QC":         sal_qc,
                "DATA_MODE":       "D",
            })

    df = pd.DataFrame(rows)
    df["TIME"] = pd.to_datetime(df["TIME"])
    return df


# ---------------------------------------------------------------------------
# BaselineStats factory
# ---------------------------------------------------------------------------

def make_baseline(
    temp_mean:  float = 27.0,
    temp_std:   float = 1.0,
    sal_mean:   float = 35.0,
    sal_std:    float = 0.5,
    sst_mean:   float = 27.5,
    sst_std:    float = 0.8,
    source:     str   = "synthetic",
    caveat:     Optional[str] = None,
) -> BaselineStats:
    """Return a deterministic BaselineStats object."""
    return BaselineStats(
        temperature_mean=temp_mean,
        temperature_std=temp_std,
        salinity_mean=sal_mean,
        salinity_std=sal_std,
        sst_mean=sst_mean,
        sst_std=sst_std,
        source=source,
        calendar_month=7,
        n_profiles=0,
        n_readings=0,
        caveat=caveat,
    )


def make_baseline_no_temp() -> BaselineStats:
    """Baseline with temperature missing (std=None)."""
    return BaselineStats(
        temperature_mean=None,
        temperature_std=None,
        salinity_mean=35.0,
        salinity_std=0.5,
        sst_mean=27.5,
        sst_std=0.8,
        source="synthetic",
        calendar_month=7,
        n_profiles=0,
        n_readings=0,
        caveat="Temperature baseline unavailable.",
    )


def make_baseline_zero_std() -> BaselineStats:
    """Baseline with temperature std=0 (indeterminate — should be excluded)."""
    return BaselineStats(
        temperature_mean=27.0,
        temperature_std=0.0,
        salinity_mean=35.0,
        salinity_std=0.5,
        sst_mean=27.5,
        sst_std=0.8,
        source="synthetic",
        calendar_month=7,
        n_profiles=0,
        n_readings=0,
        caveat="Temperature std is zero.",
    )


# ---------------------------------------------------------------------------
# ValidatedInputs factory
# ---------------------------------------------------------------------------

def make_inputs(
    lat_min: float = 8.0,
    lat_max: float = 26.0,
    lon_min: float = 55.0,
    lon_max: float = 78.0,
    start:   date  = date(2023, 7, 1),
    end:     date  = date(2023, 7, 31),
    depth_min: float = 0.0,
    depth_max: float = 50.0,
    ecosystem: str   = "general",
) -> ValidatedInputs:
    """Return a ValidatedInputs object for testing."""
    return ValidatedInputs(
        region=ParsedRegion(
            lat_min=lat_min, lat_max=lat_max,
            lon_min=lon_min, lon_max=lon_max,
            name="Arabian Sea (test)",
        ),
        time_window=ParsedTimeWindow(start=start, end=end),
        depth_layer=ParsedDepthLayer(
            min_dbar=depth_min, max_dbar=depth_max,
            label=f"Surface ({depth_min:.0f}–{depth_max:.0f} m)",
        ),
        ecosystem_type=ecosystem,
        baseline_start=date(2005, 1, 1),
        baseline_end=date(2020, 12, 31),
    )


# ---------------------------------------------------------------------------
# Observed value dict helpers
# ---------------------------------------------------------------------------

def make_observed(
    temp: Optional[float] = 29.5,
    sal:  Optional[float] = 36.0,
    sst:  Optional[float] = 30.0,
    oxy:  Optional[float] = None,
    ph:   Optional[float] = None,
) -> Dict[str, Optional[float]]:
    return {"temperature": temp, "salinity": sal, "sst": sst, "oxygen": oxy, "ph": ph}


def make_weights(ecosystem: str = "general") -> Dict[str, float]:
    presets = {
        "general":    {"temperature": 0.30, "salinity": 0.20, "sst": 0.50, "oxygen": 0.00, "ph": 0.00},
        "coral_reef": {"temperature": 0.35, "salinity": 0.15, "sst": 0.50, "oxygen": 0.00, "ph": 0.00},
        "fisheries":  {"temperature": 0.30, "salinity": 0.20, "sst": 0.30, "oxygen": 0.20, "ph": 0.00},
    }
    return presets.get(ecosystem, presets["general"])


STANDARD_BREAKPOINTS = [
    [0.0,   0],
    [1.0,  25],
    [2.0,  50],
    [3.0,  75],
    [4.0, 100],
]
