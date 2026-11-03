"""
data/ocean_db.py
----------------
OceanDB — unified multi-database coordinator for FloatAi.

This module is the single point of entry for all ocean data fetches.
It fans out to every connected database, merges the results, and returns
a standardised OceanSnapshot that the query engine can render directly.

Connected databases
-------------------
| # | Database           | Organisation | Variables                    | Access   |
|---|--------------------|-------------|-------------------------------|----------|
| 1 | Argo GDAC          | Ifremer/UCSD | T, S, QC flags, depth profiles| argopy   |
| 2 | NOAA OISST v2.1    | NOAA NCEI   | Daily satellite SST           | ERDDAP   |
| 3 | NOAA Global Drifter| NOAA AOML   | Surface T, u/v currents       | ERDDAP   |
| 4 | NOAA HYCOM model   | NOAA NCEI   | T, S, u/v, SSH (model)        | ERDDAP   |
| 5 | NOAA WOD           | NOAA NCEI   | Historical T/S profiles       | ERDDAP   |
| 6 | NOAA Coral Reef W. | NOAA CoRTAD | DHW, bleaching alerts         | ERDDAP   |
| 7 | GEBCO bathymetry   | GEBCO/BODC  | Ocean depth                   | ERDDAP   |
| 8 | CMEMS / Copernicus | EU Copernicus| T, S, u/v, Chl-a (optional)  | ERDDAP + toolbox |

Usage
-----
    from data.ocean_db import OceanDB
    db = OceanDB()
    snap = db.fetch(region, start_date, end_date)
    # snap.argo_df        — Argo profiles (standard schema)
    # snap.sst_ts         — daily SST time series
    # snap.drifter_df     — GDP surface drifter observations
    # snap.hycom_df       — HYCOM model surface fields
    # snap.wod_df         — WOD historical profiles
    # snap.coral_df       — CRW bleaching alert data
    # snap.gebco_depth_m  — mean bathymetry depth (float, metres)
    # snap.cmems_df       — CMEMS physics (if credentials available)
    # snap.source_report  — dict of {db_name: status_string}
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Snapshot dataclass
# ---------------------------------------------------------------------------

@dataclass
class OceanSnapshot:
    """Container for multi-source ocean data for a single region/time window."""
    region: dict          # {lat_min, lat_max, lon_min, lon_max}
    start_date: str
    end_date: str

    # Per-database DataFrames / values (None = not attempted or failed)
    argo_df:       Optional[pd.DataFrame] = None
    sst_ts:        Optional[pd.DataFrame] = None   # date, sst
    drifter_df:    Optional[pd.DataFrame] = None
    hycom_df:      Optional[pd.DataFrame] = None
    wod_df:        Optional[pd.DataFrame] = None
    coral_df:      Optional[pd.DataFrame] = None
    cmems_df:      Optional[pd.DataFrame] = None
    gebco_depth_m: Optional[float]        = None   # mean depth over region

    source_report: Dict[str, str] = field(default_factory=dict)

    def combined_profiles(self) -> pd.DataFrame:
        """
        Merge Argo + WOD profiles into a single DataFrame with the standard
        schema. Both share the same column names so concat works directly.
        WOD rows are included if their SOURCE column exists.
        """
        frames = []
        if self.argo_df is not None and not self.argo_df.empty:
            frames.append(self.argo_df)
        if self.wod_df is not None and not self.wod_df.empty:
            frames.append(self.wod_df)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        return combined.reset_index(drop=True)

    def has_data(self) -> bool:
        """Return True if at least one database returned data."""
        return any([
            self.argo_df is not None and not self.argo_df.empty,
            self.sst_ts is not None and not self.sst_ts.empty,
            self.drifter_df is not None and not self.drifter_df.empty,
            self.hycom_df is not None and not self.hycom_df.empty,
            self.wod_df is not None and not self.wod_df.empty,
            self.coral_df is not None and not self.coral_df.empty,
            self.cmems_df is not None and not self.cmems_df.empty,
        ])

    def summary_text(self) -> str:
        """Plain-language summary of what was fetched from each database."""
        lines = ["**Data sources queried:**"]
        for db, status in self.source_report.items():
            lines.append(f"  • {db}: {status}")
        if self.gebco_depth_m is not None:
            lines.append(f"  • GEBCO bathymetry: mean depth {self.gebco_depth_m:.0f} m")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# OceanDB coordinator
# ---------------------------------------------------------------------------

class OceanDB:
    """
    Multi-database coordinator.

    Each fetch method is independent and wrapped in try/except so a failure
    in one database never prevents the others from being queried.

    Parameters
    ----------
    use_synthetic : bool
        If True, all databases return synthetic data — no network calls.
    enable_argo   : bool  (default True)
    enable_sst    : bool  (default True)
    enable_drifter: bool  (default True)
    enable_hycom  : bool  (default True)
    enable_wod    : bool  (default True)
    enable_coral  : bool  (default True)
    enable_gebco  : bool  (default True)
    enable_cmems  : bool  (default True)
    """

    def __init__(
        self,
        use_synthetic: bool = False,
        enable_argo:    bool = True,
        enable_sst:     bool = True,
        enable_drifter: bool = True,
        enable_hycom:   bool = True,
        enable_wod:     bool = True,
        enable_coral:   bool = True,
        enable_gebco:   bool = True,
        enable_cmems:   bool = True,
    ):
        self.use_synthetic  = use_synthetic
        self.enable_argo    = enable_argo
        self.enable_sst     = enable_sst
        self.enable_drifter = enable_drifter
        self.enable_hycom   = enable_hycom
        self.enable_wod     = enable_wod
        self.enable_coral   = enable_coral
        self.enable_gebco   = enable_gebco
        self.enable_cmems   = enable_cmems

    def fetch(
        self,
        region: dict,
        start_date: str,
        end_date: str,
        sst_clim_start: Optional[str] = None,
    ) -> OceanSnapshot:
        """
        Fan out to all enabled databases in parallel and return an OceanSnapshot.

        Parameters
        ----------
        region         : dict  {lat_min, lat_max, lon_min, lon_max}
        start_date     : str   ISO date
        end_date       : str   ISO date
        sst_clim_start : str   Optional earlier start date for SST climatology.
                               If None, uses start_date.
        """
        snap = OceanSnapshot(region=region, start_date=start_date, end_date=end_date)
        lat_min, lat_max = region["lat_min"], region["lat_max"]
        lon_min, lon_max = region["lon_min"], region["lon_max"]
        lat_c = (lat_min + lat_max) / 2
        lon_c = (lon_min + lon_max) / 2
        sst_start = sst_clim_start or start_date

        # ── Define per-database fetch callables ───────────────────────────
        def _fetch_argo():
            if self.use_synthetic:
                from data.argo_loader import make_synthetic_profiles
                df = make_synthetic_profiles(lat_center=lat_c, lon_center=lon_c)
                df["TIME"] = pd.to_datetime(df["TIME"])
                return df[(df["TIME"] >= start_date) & (df["TIME"] <= end_date)].reset_index(drop=True)
            from data.argo_loader import fetch_argo_profiles
            return fetch_argo_profiles(lon_min, lon_max, lat_min, lat_max, start_date, end_date)

        def _fetch_sst():
            if self.use_synthetic:
                from data.sst_loader import make_synthetic_sst
                return make_synthetic_sst(lat_center=lat_c, lon_center=lon_c,
                                          start_date=sst_start, end_date=end_date)
            from data.sst_loader import fetch_sst_timeseries
            return fetch_sst_timeseries(lat=lat_c, lon=lon_c,
                                        start_date=sst_start, end_date=end_date)

        def _fetch_drifter():
            from data.drifter_loader import fetch_drifter_sst
            return fetch_drifter_sst(lat_min, lat_max, lon_min, lon_max, start_date, end_date)

        def _fetch_hycom():
            from data.hycom_loader import fetch_hycom_surface
            return fetch_hycom_surface(lat_min, lat_max, lon_min, lon_max, start_date, end_date)

        def _fetch_wod():
            from data.wod_loader import fetch_wod_profiles
            return fetch_wod_profiles(lat_min, lat_max, lon_min, lon_max, start_date, end_date)

        def _fetch_coral():
            from data.coral_watch_loader import fetch_coral_bleaching_alert
            return fetch_coral_bleaching_alert(lat_min, lat_max, lon_min, lon_max, start_date, end_date)

        def _fetch_gebco():
            from data.gebco_loader import fetch_gebco_depth
            return fetch_gebco_depth(lat_min, lat_max, lon_min, lon_max)

        def _fetch_cmems():
            from data.cmems_loader import fetch_cmems_physics
            return fetch_cmems_physics(lat_min, lat_max, lon_min, lon_max, start_date, end_date)

        # ── Submit all enabled databases in parallel ───────────────────────
        enabled: dict = {}
        if self.enable_argo:    enabled["argo"]    = _fetch_argo
        if self.enable_sst:     enabled["sst"]     = _fetch_sst
        if self.enable_drifter: enabled["drifter"] = _fetch_drifter
        if self.enable_hycom:   enabled["hycom"]   = _fetch_hycom
        if self.enable_wod:     enabled["wod"]     = _fetch_wod
        if self.enable_coral:   enabled["coral"]   = _fetch_coral
        if self.enable_gebco:   enabled["gebco"]   = _fetch_gebco
        if self.enable_cmems:   enabled["cmems"]   = _fetch_cmems

        futures: dict = {}
        with ThreadPoolExecutor(max_workers=min(len(enabled), 8)) as executor:
            for name, fn in enabled.items():
                futures[name] = executor.submit(fn)

            # ── Collect results ────────────────────────────────────────────
            # Argo
            if "argo" in futures:
                try:
                    snap.argo_df = futures["argo"].result()
                    n = len(snap.argo_df) if snap.argo_df is not None else 0
                    snap.source_report["Argo GDAC"] = (
                        f"✅ {n} profile observations" if n > 0 else "⚠️ No floats in this region/period"
                    )
                except Exception as exc:
                    snap.source_report["Argo GDAC"] = f"❌ {_short(exc)}"
                    logger.warning("Argo fetch failed: %s", exc)

            # SST
            if "sst" in futures:
                try:
                    snap.sst_ts = futures["sst"].result()
                    n = len(snap.sst_ts) if snap.sst_ts is not None else 0
                    snap.source_report["NOAA OISST v2.1"] = (
                        f"✅ {n} daily SST values" if n > 0 else "⚠️ No SST data returned"
                    )
                except Exception as exc:
                    snap.source_report["NOAA OISST v2.1"] = f"❌ {_short(exc)}"
                    logger.warning("SST fetch failed: %s", exc)

            # Drifters
            if "drifter" in futures:
                try:
                    snap.drifter_df = futures["drifter"].result()
                    n = len(snap.drifter_df) if snap.drifter_df is not None else 0
                    snap.source_report["NOAA Global Drifter Program"] = (
                        f"✅ {n} drifter observations" if n > 0 else "⚠️ No drifters in region/period"
                    )
                except Exception as exc:
                    snap.source_report["NOAA Global Drifter Program"] = f"❌ {_short(exc)}"
                    logger.warning("Drifter fetch failed: %s", exc)

            # HYCOM
            if "hycom" in futures:
                try:
                    snap.hycom_df = futures["hycom"].result()
                    n = len(snap.hycom_df) if snap.hycom_df is not None else 0
                    snap.source_report["NOAA HYCOM Model"] = (
                        f"✅ {n} model grid points" if n > 0 else "⚠️ No HYCOM data returned"
                    )
                except Exception as exc:
                    snap.source_report["NOAA HYCOM Model"] = f"❌ {_short(exc)}"
                    logger.warning("HYCOM fetch failed: %s", exc)

            # WOD
            if "wod" in futures:
                try:
                    snap.wod_df = futures["wod"].result()
                    n = len(snap.wod_df) if snap.wod_df is not None else 0
                    snap.source_report["NOAA World Ocean Database"] = (
                        f"✅ {n} WOD profile observations" if n > 0 else "⚠️ No WOD data in region/period"
                    )
                except Exception as exc:
                    snap.source_report["NOAA World Ocean Database"] = f"❌ {_short(exc)}"
                    logger.warning("WOD fetch failed: %s", exc)

            # Coral Reef Watch
            if "coral" in futures:
                try:
                    snap.coral_df = futures["coral"].result()
                    n = len(snap.coral_df) if snap.coral_df is not None else 0
                    if n > 0:
                        max_dhw = float(snap.coral_df["DHW"].max())
                        snap.source_report["NOAA Coral Reef Watch"] = (
                            f"✅ {n} grid points, max DHW={max_dhw:.1f}°C-wks"
                        )
                    else:
                        snap.source_report["NOAA Coral Reef Watch"] = "⚠️ No CRW data (region may be non-tropical)"
                except Exception as exc:
                    snap.source_report["NOAA Coral Reef Watch"] = f"❌ {_short(exc)}"
                    logger.warning("CRW fetch failed: %s", exc)

            # GEBCO
            if "gebco" in futures:
                try:
                    bathy = futures["gebco"].result()
                    if not bathy.empty:
                        ocean_cells = bathy[bathy["DEPTH_M"] > 0]
                        snap.gebco_depth_m = float(ocean_cells["DEPTH_M"].mean()) if not ocean_cells.empty else 0.0
                        snap.source_report["GEBCO Bathymetry 2022"] = (
                            f"✅ Mean depth {snap.gebco_depth_m:.0f} m over region"
                        )
                    else:
                        snap.source_report["GEBCO Bathymetry 2022"] = "⚠️ No depth data returned"
                except Exception as exc:
                    snap.source_report["GEBCO Bathymetry 2022"] = f"❌ {_short(exc)}"
                    logger.warning("GEBCO fetch failed: %s", exc)

            # CMEMS
            if "cmems" in futures:
                try:
                    snap.cmems_df = futures["cmems"].result()
                    n = len(snap.cmems_df) if snap.cmems_df is not None else 0
                    snap.source_report["CMEMS Global Ocean Physics"] = (
                        f"✅ {n} model grid points" if n > 0
                        else "⚠️ No data (set COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD for full access)"
                    )
                except Exception as exc:
                    snap.source_report["CMEMS Global Ocean Physics"] = f"❌ {_short(exc)}"
                    logger.debug("CMEMS fetch failed: %s", exc)

        return snap


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _short(exc: Exception, max_len: int = 120) -> str:
    """Shorten an exception message for display."""
    msg = str(exc)
    return msg[:max_len] + "…" if len(msg) > max_len else msg
