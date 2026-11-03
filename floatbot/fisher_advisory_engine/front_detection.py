"""
fisher_advisory_engine/front_detection.py
-------------------------------------------
Thermal front detection from the NOAA OISST v2.1 spatial SST gradient.

Algorithm
---------
1. Fetch a square SST grid of radius `analysis_radius_deg` (config) centred
   on the query location for the query date.
2. Pivot to a 2-D lat×lon grid; fill isolated NaN cells with the local mean
   (do not interpolate if more than half the cells are NaN — return no_data).
3. Compute the SST gradient magnitude using numpy finite differences:
       G(i,j) = sqrt( (∂T/∂lat)² + (∂T/∂lon)² )
   divided by the grid spacing converted to km (1° ≈ 111 km).
4. Find the location of the maximum gradient in the grid.
5. Classify front strength against config thresholds (PLACEHOLDER):
       > gradient_threshold_c_per_km  → "detected" (moderate or strong)
       > gradient_weak_c_per_km       → "weak"
       else                           → "not_detected"
6. Compute bearing and distance from the query point to the peak gradient cell.

All thresholds are in config.yaml and labeled PLACEHOLDER.

Public API
----------
    from fisher_advisory_engine.front_detection import detect_front, FrontResult
    result = detect_front(lat, lon, date_str)
    # result.detected, result.strength, result.gradient_c_km,
    # result.dist_km, result.bearing_deg
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class FrontResult:
    detected:       bool
    strength:       str             # "not_detected" | "weak" | "moderate" | "strong"
    gradient_c_km:  Optional[float] # peak SST gradient magnitude (°C/km)
    dist_km:        Optional[float] # km from query point to peak gradient
    bearing_deg:    Optional[float] # bearing from query point to peak gradient
    peak_lat:       Optional[float] # latitude of peak gradient cell
    peak_lon:       Optional[float] # longitude of peak gradient cell
    n_grid_points:  int             # number of valid SST cells used
    status:         str             # "ok" | "no_data" | "error"
    timestamp:      Optional[str]   # ISO-8601 UTC of the SST data used
    caveat:         Optional[str]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_front(
    lat: float,
    lon: float,
    date_str: str,
    use_synthetic: bool = False,
) -> FrontResult:
    """
    Detect a thermal front near (lat, lon) on date_str.

    Parameters
    ----------
    lat, lon   : float  Query location (decimal degrees).
    date_str   : str    ISO date, e.g. '2024-06-15'.
    use_synthetic: bool  Use synthetic SST instead of live fetch (for tests).

    Returns
    -------
    FrontResult
    """
    cfg       = _load_config()
    front_cfg = cfg.get("front", {})
    radius    = float(front_cfg.get("analysis_radius_deg", 1.0))
    thresh    = float(front_cfg.get("gradient_threshold_c_per_km", 0.1))
    weak_t    = float(front_cfg.get("gradient_weak_c_per_km",      0.05))
    min_pts   = int(front_cfg.get("min_grid_points", 9))

    # ── 1. Fetch SST grid ────────────────────────────────────────────────
    lat_min = lat - radius
    lat_max = lat + radius
    lon_min = lon - radius
    lon_max = lon + radius

    if use_synthetic:
        sst_df = _synthetic_sst_grid(lat, lon, date_str, radius)
    else:
        try:
            from data.sst_loader import fetch_sst
            sst_df = fetch_sst(
                lat_min=lat_min, lat_max=lat_max,
                lon_min=lon_min, lon_max=lon_max,
                start_date=date_str, end_date=date_str,
            )
        except Exception as exc:
            return FrontResult(
                detected=False, strength="not_detected",
                gradient_c_km=None, dist_km=None, bearing_deg=None,
                peak_lat=None, peak_lon=None, n_grid_points=0,
                status="no_data",
                timestamp=None,
                caveat=f"SST fetch for front detection failed: {str(exc)[:200]}",
            )

    if sst_df is None or sst_df.empty:
        return FrontResult(
            detected=False, strength="not_detected",
            gradient_c_km=None, dist_km=None, bearing_deg=None,
            peak_lat=None, peak_lon=None, n_grid_points=0,
            status="no_data",
            timestamp=None,
            caveat="No SST data available for front detection.",
        )

    # ── 2. Build 2-D grid ────────────────────────────────────────────────
    # Average across the date dimension (should be one day, but in case of
    # multi-day fetches we take the spatial snapshot for the target date)
    target_date = pd.Timestamp(date_str).normalize()
    daily_df    = sst_df.copy()
    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.normalize()
    # Use the target date if available; fall back to the latest available date
    on_target = daily_df[daily_df["date"] == target_date]
    if on_target.empty:
        on_target = daily_df[daily_df["date"] == daily_df["date"].max()]
    actual_date   = on_target["date"].max()
    timestamp_iso = actual_date.isoformat() + "Z"

    try:
        grid, lats_ax, lons_ax = _build_grid(on_target)
    except Exception as exc:
        return FrontResult(
            detected=False, strength="not_detected",
            gradient_c_km=None, dist_km=None, bearing_deg=None,
            peak_lat=None, peak_lon=None, n_grid_points=0,
            status="no_data",
            timestamp=timestamp_iso,
            caveat=f"Could not build SST grid: {exc}",
        )

    n_valid = int(np.sum(~np.isnan(grid)))

    if n_valid < min_pts:
        return FrontResult(
            detected=False, strength="not_detected",
            gradient_c_km=None, dist_km=None, bearing_deg=None,
            peak_lat=None, peak_lon=None, n_grid_points=n_valid,
            status="no_data",
            timestamp=timestamp_iso,
            caveat=(
                f"Only {n_valid} valid SST grid cells (need ≥ {min_pts}). "
                "Front detection not possible — insufficient spatial coverage."
            ),
        )

    # Fill sparse NaN cells only if < 50% of cells are missing
    nan_frac = np.sum(np.isnan(grid)) / grid.size
    if nan_frac > 0.5:
        return FrontResult(
            detected=False, strength="not_detected",
            gradient_c_km=None, dist_km=None, bearing_deg=None,
            peak_lat=None, peak_lon=None, n_grid_points=n_valid,
            status="no_data",
            timestamp=timestamp_iso,
            caveat=(
                f"{nan_frac:.0%} of SST grid cells are masked (cloud or no data). "
                "Front detection requires > 50% valid cells."
            ),
        )

    # Simple fill: replace NaN with grid mean
    grid_filled = np.where(np.isnan(grid), np.nanmean(grid), grid)

    # ── 3. Compute gradient magnitude ────────────────────────────────────
    grad_c_km, grad_lat_idx, grad_lon_idx = _compute_gradient(
        grid_filled, lats_ax, lons_ax
    )

    # ── 4. Classify strength ─────────────────────────────────────────────
    if grad_c_km >= thresh:
        strength = "strong" if grad_c_km >= thresh * 1.5 else "moderate"
        detected = True
    elif grad_c_km >= weak_t:
        strength = "weak"
        detected = False   # weak gradient noted but not classified as a front
    else:
        strength = "not_detected"
        detected = False

    # ── 5. Distance and bearing from query point to peak ─────────────────
    peak_lat = float(lats_ax[grad_lat_idx])
    peak_lon = float(lons_ax[grad_lon_idx])
    dist_km  = _haversine_km(lat, lon, peak_lat, peak_lon)
    bearing  = _bearing_deg(lat, lon, peak_lat, peak_lon)

    caveat = (
        "Front detection uses OISST 0.25° (~28 km) grid — cannot resolve "
        "fronts smaller than approximately 50 km. "
        "Gradient threshold is PLACEHOLDER (config.yaml). "
        "This indicator requires validation against in-situ observations "
        "before operational use."
    )

    return FrontResult(
        detected=detected,
        strength=strength,
        gradient_c_km=round(grad_c_km, 4),
        dist_km=round(dist_km, 1),
        bearing_deg=round(bearing, 1),
        peak_lat=round(peak_lat, 4),
        peak_lon=round(peak_lon, 4),
        n_grid_points=n_valid,
        status="ok",
        timestamp=timestamp_iso,
        caveat=caveat,
    )


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def _build_grid(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pivot SST DataFrame (columns: lat, lon, sst) into a 2-D numpy array.
    Returns (grid, lats_axis, lons_axis).
    """
    # Round to OISST 0.25° grid to avoid float precision issues
    df = df.copy()
    df["lat"] = df["lat"].round(2)
    df["lon"] = df["lon"].round(2)

    pivot = df.groupby(["lat", "lon"])["sst"].mean().reset_index()
    if pivot.empty:
        raise ValueError("Empty SST DataFrame after grouping.")

    lats = np.sort(pivot["lat"].unique())
    lons = np.sort(pivot["lon"].unique())

    grid = np.full((len(lats), len(lons)), np.nan)
    lat_idx = {v: i for i, v in enumerate(lats)}
    lon_idx = {v: i for i, v in enumerate(lons)}

    for _, row in pivot.iterrows():
        i = lat_idx.get(row["lat"])
        j = lon_idx.get(row["lon"])
        if i is not None and j is not None:
            grid[i, j] = row["sst"]

    return grid, lats, lons


# ---------------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------------

def _compute_gradient(
    grid: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[float, int, int]:
    """
    Compute the SST gradient magnitude at each grid cell and return
    (peak_gradient_c_per_km, peak_lat_index, peak_lon_index).

    Uses numpy gradient (central differences at interior, one-sided at edges).
    Grid spacing is converted from degrees to km using the 111 km/° approximation
    for latitude; longitude spacing uses cos(lat) correction.
    """
    if grid.shape[0] < 2 or grid.shape[1] < 2:
        # Not enough cells for a meaningful gradient
        return 0.0, 0, 0

    # Mean grid spacing in degrees
    dlat_deg = float(np.mean(np.diff(lats))) if len(lats) > 1 else 0.25
    dlon_deg = float(np.mean(np.diff(lons))) if len(lons) > 1 else 0.25
    lat_c    = float(np.mean(lats))

    km_per_lat_deg = 111.0
    km_per_lon_deg = 111.0 * np.cos(np.radians(lat_c))

    dlat_km = dlat_deg * km_per_lat_deg
    dlon_km = dlon_deg * km_per_lon_deg

    # Gradients in °C per grid cell → convert to °C per km
    grad_lat_raw, grad_lon_raw = np.gradient(grid)
    grad_lat_c_km = grad_lat_raw / dlat_km if dlat_km > 0 else grad_lat_raw
    grad_lon_c_km = grad_lon_raw / dlon_km if dlon_km > 0 else grad_lon_raw

    magnitude = np.sqrt(grad_lat_c_km**2 + grad_lon_c_km**2)

    # Find peak
    peak_flat = int(np.nanargmax(magnitude))
    peak_i, peak_j = np.unravel_index(peak_flat, magnitude.shape)
    peak_val = float(magnitude[peak_i, peak_j])

    return peak_val, int(peak_i), int(peak_j)


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two points."""
    R = 6371.0
    phi1, phi2   = np.radians(lat1), np.radians(lat2)
    dphi         = np.radians(lat2 - lat1)
    dlambda      = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return float(R * 2 * np.arcsin(np.sqrt(a)))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing (degrees from north) from point 1 to point 2."""
    phi1  = np.radians(lat1)
    phi2  = np.radians(lat2)
    dl    = np.radians(lon2 - lon1)
    x     = np.sin(dl) * np.cos(phi2)
    y     = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dl)
    return float((np.degrees(np.arctan2(x, y)) + 360) % 360)


# ---------------------------------------------------------------------------
# Synthetic SST grid (for tests — no network)
# ---------------------------------------------------------------------------

def _synthetic_sst_grid(
    lat: float,
    lon: float,
    date_str: str,
    radius: float,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic 0.25° SST grid with a deliberate front
    (step change) running east–west across the middle of the grid.
    Used only when use_synthetic=True.
    """
    rng  = np.random.default_rng(seed)
    rows = []
    step = 0.25
    lat_vals = np.arange(lat - radius, lat + radius + step, step)
    lon_vals = np.arange(lon - radius, lon + radius + step, step)
    mid_lat  = lat   # front runs at the query latitude

    for la in lat_vals:
        for lo in lon_vals:
            # Introduce a +2°C step at the mid_lat (synthetic front)
            base = 28.5 + (2.0 if la >= mid_lat else 0.0)
            sst  = base + float(rng.normal(0, 0.15))
            rows.append({"date": pd.Timestamp(date_str), "lat": round(float(la), 2),
                         "lon": round(float(lo), 2), "sst": round(sst, 3)})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

_CFG_CACHE: Optional[dict] = None

def _load_config() -> dict:
    global _CFG_CACHE
    if _CFG_CACHE is not None:
        return _CFG_CACHE
    import yaml  # type: ignore
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        _CFG_CACHE = yaml.safe_load(fh)
    return _CFG_CACHE
