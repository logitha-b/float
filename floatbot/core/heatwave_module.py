"""
core/heatwave_module.py
-----------------------
Feature 2: Cross-Dataset Reasoning — Marine Heatwave Detection

Implements the published Hobday et al. (2016) marine heatwave algorithm:

  Hobday, A. J., et al. (2016). A hierarchical approach to defining marine
  heatwaves. Progress in Oceanography, 141, 227–238.
  https://doi.org/10.1016/j.pocean.2015.12.014

Algorithm summary
-----------------
1. Build a day-of-year climatology from historical daily SST using an
   11-day centred window across all available years → mean and 90th
   percentile for each calendar day (1–366).
2. Detect periods where SST exceeds the 90th-percentile threshold for
   ≥5 consecutive days — each such period is a marine heatwave (MHW).
3. For each event, report start, end, duration, peak SST, and mean anomaly.

Public API
----------
    result = detect_heatwaves(sst_df, query_start, query_end)
    # sst_df has columns: date (datetime64), sst (float)
    # result is a HeatwaveResult dataclass
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Hobday 2016 parameters
CLIMATOLOGY_WINDOW = 11    # days centred on each calendar day
THRESHOLD_PERCENTILE = 90  # percentile to define the threshold
MIN_DURATION_DAYS = 5      # minimum consecutive days above threshold

METHOD_CITATION = (
    "Hobday, A. J., et al. (2016). A hierarchical approach to defining marine "
    "heatwaves. Progress in Oceanography, 141, 227–238. "
    "https://doi.org/10.1016/j.pocean.2015.12.014"
)


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass
class HeatwaveEvent:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    duration_days: int
    peak_sst: float          # °C — highest SST during the event
    peak_date: pd.Timestamp
    mean_anomaly: float      # °C above climatological mean (averaged over event)
    mean_threshold: float    # °C — mean 90th-percentile threshold over event period

    def to_dict(self) -> dict:
        return {
            "start_date": self.start_date.strftime("%Y-%m-%d"),
            "end_date": self.end_date.strftime("%Y-%m-%d"),
            "duration_days": self.duration_days,
            "peak_sst": round(self.peak_sst, 2),
            "peak_date": self.peak_date.strftime("%Y-%m-%d"),
            "mean_anomaly": round(self.mean_anomaly, 2),
            "mean_threshold": round(self.mean_threshold, 2),
        }

    def describe(self) -> str:
        return (
            f"{self.duration_days}-day heatwave from "
            f"{self.start_date.strftime('%d %b %Y')} to "
            f"{self.end_date.strftime('%d %b %Y')}, "
            f"peaking at {self.peak_sst:.1f}°C on "
            f"{self.peak_date.strftime('%d %b %Y')} "
            f"({self.mean_anomaly:+.1f}°C above the climatological mean)"
        )


@dataclass
class HeatwaveResult:
    query_start: Optional[pd.Timestamp]
    query_end: Optional[pd.Timestamp]
    all_events: List[HeatwaveEvent]          # all detected events in full series
    overlapping_events: List[HeatwaveEvent]  # events overlapping query window
    climatology_years: int                   # number of years used to build climatology
    method_citation: str = METHOD_CITATION

    @property
    def heatwave_detected(self) -> bool:
        return len(self.overlapping_events) > 0

    def to_answer_text(self) -> str:
        if self.query_start is None:
            # No date filter — report all events
            if not self.all_events:
                return (
                    "No marine heatwave events were detected in the SST record "
                    f"for this region ({self.climatology_years} years of data analysed, "
                    f"using the {METHOD_CITATION})."
                )
            descs = "\n".join(f"  • {e.describe()}" for e in self.all_events)
            return (
                f"{len(self.all_events)} marine heatwave event(s) detected "
                f"(Hobday et al. 2016 method, {self.climatology_years} years of climatology):\n"
                f"{descs}"
            )

        window = (
            f"{self.query_start.strftime('%d %b %Y')} – "
            f"{self.query_end.strftime('%d %b %Y')}"  # type: ignore[union-attr]
        )
        if not self.overlapping_events:
            return (
                f"No marine heatwave conditions were detected near this region "
                f"during {window}. SST remained below the 90th-percentile "
                "climatological threshold throughout the period "
                f"(Hobday et al. 2016 method, "
                f"{self.climatology_years} years of climatology)."
            )

        descs = "\n".join(f"  • {e.describe()}" for e in self.overlapping_events)
        return (
            f"Yes — {len(self.overlapping_events)} marine heatwave event(s) "
            f"detected during {window} using the Hobday et al. (2016) method "
            f"({self.climatology_years} years of SST climatology):\n{descs}"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_heatwaves(
    sst_df: pd.DataFrame,
    query_start: Optional[str] = None,
    query_end: Optional[str] = None,
) -> HeatwaveResult:
    """
    Detect marine heatwaves in a daily SST time series.

    Parameters
    ----------
    sst_df      : pd.DataFrame   Columns: date (datetime64), sst (float).
                                  Must span multiple years for a meaningful
                                  climatology (≥3 years recommended).
    query_start : str or None    ISO date string. If provided, filter detected
                                  events to those overlapping this window.
    query_end   : str or None    ISO date string (inclusive end).

    Returns
    -------
    HeatwaveResult
    """
    df = sst_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["sst"])

    if len(df) < 365:
        logger.warning(
            "SST series has only %d days — climatology may be unreliable "
            "(Hobday 2016 recommends ≥30 years; ≥3 years is a practical minimum).",
            len(df),
        )

    # --- Build climatology ----------------------------------------------------
    clim = _build_climatology(df)
    climatology_years = df["date"].dt.year.nunique()

    # --- Merge climatology with the SST series --------------------------------
    df["doy"] = df["date"].dt.dayofyear
    df = df.merge(clim, on="doy", how="left")

    # --- Detect heatwave events -----------------------------------------------
    df["above_threshold"] = df["sst"] > df["clim_p90"]
    all_events = _find_events(df)

    # --- Filter to query window -----------------------------------------------
    qs = pd.Timestamp(query_start) if query_start else None
    qe = pd.Timestamp(query_end) if query_end else None

    if qs is not None and qe is not None:
        overlapping = [
            e for e in all_events
            if e.start_date <= qe and e.end_date >= qs
        ]
    else:
        overlapping = list(all_events)

    return HeatwaveResult(
        query_start=qs,
        query_end=qe,
        all_events=all_events,
        overlapping_events=overlapping,
        climatology_years=climatology_years,
    )


# ---------------------------------------------------------------------------
# Climatology builder (Hobday 2016 §2.2)
# ---------------------------------------------------------------------------

def _build_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute day-of-year climatological mean and 90th-percentile threshold.

    Uses an 11-day centred window (±5 days) pooling all years — exactly as
    described in Hobday et al. (2016).

    Returns a DataFrame with columns: doy, clim_mean, clim_p90
    """
    df = df.copy()
    df["doy"] = df["date"].dt.dayofyear

    half_win = CLIMATOLOGY_WINDOW // 2   # 5

    records = []
    for doy in range(1, 367):
        # Build the window of ±half_win days, wrapping around year boundaries
        window_doys = set()
        for delta in range(-half_win, half_win + 1):
            d = ((doy - 1 + delta) % 365) + 1  # 1-indexed, wraps 1–365
            window_doys.add(d)

        subset = df[df["doy"].isin(window_doys)]["sst"]
        if len(subset) >= 2:
            records.append({
                "doy": doy,
                "clim_mean": float(subset.mean()),
                "clim_p90": float(np.percentile(subset, THRESHOLD_PERCENTILE)),
            })
        else:
            records.append({"doy": doy, "clim_mean": np.nan, "clim_p90": np.nan})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Event finder
# ---------------------------------------------------------------------------

def _find_events(df: pd.DataFrame) -> List[HeatwaveEvent]:
    """
    Scan the merged SST+climatology DataFrame for contiguous runs of days
    where sst > clim_p90, keeping only those ≥ MIN_DURATION_DAYS.
    """
    events: List[HeatwaveEvent] = []
    in_event = False
    event_rows: list = []

    for _, row in df.iterrows():
        if row["above_threshold"]:
            in_event = True
            event_rows.append(row)
        else:
            if in_event:
                _maybe_add_event(event_rows, events)
                event_rows = []
            in_event = False

    # Handle event running to the end of the series
    if in_event and event_rows:
        _maybe_add_event(event_rows, events)

    return events


def _maybe_add_event(rows: list, events: List[HeatwaveEvent]) -> None:
    """Emit a HeatwaveEvent if the run is long enough."""
    if len(rows) < MIN_DURATION_DAYS:
        return

    subset = pd.DataFrame(rows)
    start = subset["date"].min()
    end = subset["date"].max()
    duration = (end - start).days + 1
    peak_idx = subset["sst"].idxmax()
    peak_sst = float(subset.loc[peak_idx, "sst"])
    peak_date = subset.loc[peak_idx, "date"]
    mean_anomaly = float((subset["sst"] - subset["clim_mean"]).mean())
    mean_threshold = float(subset["clim_p90"].mean())

    events.append(HeatwaveEvent(
        start_date=start,
        end_date=end,
        duration_days=duration,
        peak_sst=peak_sst,
        peak_date=peak_date,
        mean_anomaly=mean_anomaly,
        mean_threshold=mean_threshold,
    ))


# ---------------------------------------------------------------------------
# Convenience: build climatology DataFrame for plotting
# ---------------------------------------------------------------------------

def get_climatology_df(sst_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expose the climatology (mean + 90th-percentile threshold) for a given
    SST series as a DataFrame — useful for Streamlit plotting.

    Returns columns: doy, clim_mean, clim_p90
    """
    df = sst_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=["sst"])
    return _build_climatology(df)
