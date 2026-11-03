"""
core/query_engine.py
--------------------
Feature 3: Natural Language Understanding / Query Routing

Orchestrates the full pipeline:
  1. Parse the plain-English question (LLM if available, else rule-based).
  2. Resolve the region / dates into actual data fetches.
  3. Route to Feature 1 (QC), Feature 2 (heatwave), or both (compound).
  4. Return a unified EngineResult for the Streamlit app to render.

Public API
----------
    engine = QueryEngine(use_synthetic=False)
    result = engine.answer(question)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from core.qc_module import QCResult, apply_qc_filter
from core.heatwave_module import HeatwaveResult, detect_heatwaves
from nlu import llm_parser, rule_parser

logger = logging.getLogger(__name__)

# Default depth window for surface comparisons against SST heatwave data
SURFACE_DEPTH_MIN = 0.0
SURFACE_DEPTH_MAX = 10.0

# SST climatology: 1 year back from query start.
# 1 year = 12 monthly chunks. With parallel fetching this completes in ~20–40s.
# Hobday (2016) recommends 30 years but 1 year still detects seasonal anomalies.
SST_CLIMATOLOGY_YEARS = 1

# Databases enabled by default (can be toggled per-instance)
_DEFAULT_DB_FLAGS = dict(
    enable_argo=True,
    enable_sst=True,
    enable_drifter=True,
    enable_hycom=True,
    enable_wod=True,
    enable_coral=True,
    enable_gebco=True,
    enable_cmems=True,
)


# ---------------------------------------------------------------------------
# Unified result type
# ---------------------------------------------------------------------------

@dataclass
class EngineResult:
    question: str
    parsed_query: dict

    # Primary results
    qc_result: Optional[QCResult] = None
    qc_result_salinity: Optional[QCResult] = None
    heatwave_result: Optional[HeatwaveResult] = None

    # Multi-database snapshot (populated when use_ocean_db=True)
    ocean_snapshot: Optional[object] = None   # data.ocean_db.OceanSnapshot

    error: Optional[str] = None
    warnings: list = field(default_factory=list)

    @property
    def parser_used(self) -> str:
        return self.parsed_query.get("parser_used", "unknown")

    @property
    def intent(self) -> str:
        return self.parsed_query.get("intent", "unknown")

    def to_answer_text(self) -> str:
        if self.error:
            return f"Sorry, I couldn't answer that question: {self.error}"

        parts = []
        if self.qc_result:
            parts.append(self.qc_result.to_answer_text())
        if self.qc_result_salinity:
            parts.append(self.qc_result_salinity.to_answer_text())
        if self.heatwave_result:
            parts.append(self.heatwave_result.to_answer_text())

        # Supplement with extra database context
        if self.ocean_snapshot is not None:
            extras = _build_extra_context(self.ocean_snapshot)
            if extras:
                parts.append(extras)

        if not parts:
            return "No results were found for the requested region and time period."
        return "\n\n".join(parts)


def _build_extra_context(snap) -> str:
    """
    Build supplementary answer text from non-Argo databases in an OceanSnapshot.
    Only adds lines for databases that returned actual data.
    """
    lines = []

    # Ocean depth from GEBCO
    if snap.gebco_depth_m is not None and snap.gebco_depth_m > 0:
        depth = snap.gebco_depth_m
        if depth < 200:
            zone = "continental shelf"
        elif depth < 1000:
            zone = "upper continental slope"
        elif depth < 3500:
            zone = "continental rise"
        elif depth < 6000:
            zone = "abyssal plain"
        else:
            zone = "hadal zone / trench"
        lines.append(f"🌊 **Ocean depth** (GEBCO 2022): mean {depth:.0f} m ({zone})")

    # HYCOM model currents
    if snap.hycom_df is not None and not snap.hycom_df.empty:
        hdf = snap.hycom_df
        if "U_CURR" in hdf.columns and "V_CURR" in hdf.columns:
            import numpy as _np
            speeds = _np.sqrt(hdf["U_CURR"].dropna()**2 + hdf["V_CURR"].dropna()**2)
            if len(speeds) > 0:
                lines.append(
                    f"🌀 **Surface currents** (NOAA HYCOM model): "
                    f"mean speed {speeds.mean():.2f} m/s, "
                    f"max {speeds.max():.2f} m/s"
                )
        if "TEMP" in hdf.columns:
            mean_t = hdf["TEMP"].dropna().mean()
            if not _np.isnan(mean_t):
                lines.append(
                    f"🌡️ **HYCOM model temperature**: mean {mean_t:.2f}°C at surface"
                )

    # GDP Drifter SST
    if snap.drifter_df is not None and not snap.drifter_df.empty:
        ddf = snap.drifter_df
        if "SST" in ddf.columns and ddf["SST"].notna().any():
            mean_sst = float(ddf["SST"].dropna().mean())
            n_drifters = ddf["DRIFTER_ID"].nunique() if "DRIFTER_ID" in ddf.columns else "?"
            lines.append(
                f"🛟 **Surface drifter SST** (NOAA GDP): "
                f"mean {mean_sst:.2f}°C from {n_drifters} drifter(s)"
            )

    # Coral bleaching alerts
    if snap.coral_df is not None and not snap.coral_df.empty:
        cdf = snap.coral_df
        if "DHW" in cdf.columns and cdf["DHW"].notna().any():
            max_dhw = float(cdf["DHW"].max())
            max_alert = int(cdf["ALERT_LEVEL"].max()) if "ALERT_LEVEL" in cdf.columns else 0
            alert_labels = {
                0: "No Stress", 1: "Bleaching Watch",
                2: "Bleaching Warning", 3: "Alert Level 1", 4: "Alert Level 2",
            }
            alert_name = alert_labels.get(max_alert, "Unknown")
            emoji = "✅" if max_alert == 0 else ("⚠️" if max_alert <= 2 else "🚨")
            lines.append(
                f"{emoji} **Coral bleaching stress** (NOAA CRW): "
                f"max DHW={max_dhw:.1f}°C-weeks — {alert_name}"
            )

    # WOD historical profiles
    if snap.wod_df is not None and not snap.wod_df.empty:
        n = len(snap.wod_df)
        lines.append(
            f"📚 **World Ocean Database (WOD)**: {n} historical profile observations available"
        )

    # CMEMS
    if snap.cmems_df is not None and not snap.cmems_df.empty:
        cdf = snap.cmems_df
        if "TEMP" in cdf.columns and cdf["TEMP"].notna().any():
            lines.append(
                f"🇪🇺 **CMEMS ocean model temperature**: "
                f"mean {cdf['TEMP'].dropna().mean():.2f}°C"
            )

    return "\n".join(lines) if lines else ""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class QueryEngine:
    """
    Main orchestration class.

    Parameters
    ----------
    use_synthetic : bool   Use synthetic data (no network). Default False.
    db_flags      : dict   Per-database enable/disable flags. Keys match
                           OceanDB constructor kwargs (enable_argo, etc.).
                           Defaults to all enabled.
    """

    def __init__(
        self,
        use_synthetic: bool = False,
        db_flags: Optional[dict] = None,
    ):
        self.use_synthetic = use_synthetic
        self.db_flags = {**_DEFAULT_DB_FLAGS, **(db_flags or {})}
        self._dataset_latest_date: Optional[str] = None

        # Lazily-created OceanDB instance
        self._ocean_db: Optional[object] = None

        # Per-answer Argo data cache: keyed by (region_tuple, start, end)
        # Reused within a single answer() call so compound queries (temp + sal)
        # only fetch Argo once instead of twice. Cleared at the start of answer().
        self._argo_cache: dict = {}

    def _get_ocean_db(self):
        """Return (or create) the OceanDB coordinator."""
        if self._ocean_db is None:
            from data.ocean_db import OceanDB
            self._ocean_db = OceanDB(
                use_synthetic=self.use_synthetic,
                **self.db_flags,
            )
        return self._ocean_db

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def answer(self, question: str) -> EngineResult:
        """
        Parse a plain-English question and return a fully populated EngineResult.
        All exceptions are caught and surfaced as EngineResult.error strings.
        """
        # Clear the per-answer Argo cache so each new question starts fresh
        self._argo_cache = {}

        # --- 1. Parse ---------------------------------------------------------
        parsed = self._parse(question)
        result = EngineResult(question=question, parsed_query=parsed)

        # --- 2. Validate minimum fields ---------------------------------------
        region = parsed.get("region")
        start_date = parsed.get("start_date")
        end_date = parsed.get("end_date")

        if region is None:
            result.warnings.append(
                "No region was identified in your question. "
                "Try naming an ocean region (e.g. 'Gulf of Mexico') or "
                "giving lat/lon coordinates."
            )
            result.error = (
                "I couldn't identify a geographic region in your question. "
                "Please name a region or provide coordinates."
            )
            return result

        if start_date is None or end_date is None:
            result.warnings.append(
                "No time period was identified — defaulting to the last 30 days "
                "of available data."
            )
            anchor = self._get_latest_date()
            if anchor:
                end_date = anchor
                start_date = (pd.Timestamp(anchor) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
                parsed["start_date"] = start_date
                parsed["end_date"] = end_date
            else:
                result.error = "No time period provided and no dataset anchor available."
                return result

        # Warn if the requested dates are in the future (Argo/OISST lag)
        today = pd.Timestamp.now().normalize()
        if pd.Timestamp(end_date) > today:
            capped = (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            result.warnings.append(
                f"The requested end date ({end_date}) is in the future. "
                f"Both Argo and OISST have a processing lag of ~2 weeks, so the "
                f"query has been capped to {capped}."
            )
            parsed["end_date"] = capped
            end_date = capped

        # --- 3. Route ---------------------------------------------------------
        intent = parsed.get("intent", "profile_query")

        try:
            if intent == "profile_query":
                from data.argo_loader import ArgoNoDataError
                # Run Argo fetch and fast snapshot in parallel
                with ThreadPoolExecutor(max_workers=2) as ex:
                    fut_argo = ex.submit(self._run_qc, parsed)
                    fut_snap = ex.submit(self._fetch_fast_snapshot, parsed)
                    try:
                        result.qc_result = fut_argo.result()
                    except ArgoNoDataError as exc:
                        result.error = (
                            f"**No Argo float data available for this region.**\n\n"
                            f"{exc}\n\n"
                            "**What you can do:**\n"
                            "• Try a nearby region with better coverage "
                            "(Arabian Sea, Indian Ocean, Pacific, Atlantic)\n"
                            "• Switch to **Offline mode** to test with synthetic data"
                        )
                        return result
                    except Exception as exc:
                        err_str = str(exc)
                        if not err_str.startswith("Could not retrieve"):
                            err_str = f"Could not retrieve Argo profile data: {err_str}"
                        result.error = (
                            f"{err_str}\n\n"
                            "💡 Try **Offline mode** or a **shorter date range**."
                        )
                        return result
                    try:
                        result.ocean_snapshot = fut_snap.result()
                    except Exception as exc:
                        logger.debug("Fast snapshot failed (non-fatal): %s", exc)
                # Skip the duplicate fast snapshot fetch below
                return result

            elif intent == "heatwave_query":
                # Run heatwave and fast snapshot in parallel
                with ThreadPoolExecutor(max_workers=2) as ex:
                    fut_hw   = ex.submit(self._run_heatwave, parsed)
                    fut_snap = ex.submit(self._fetch_fast_snapshot, parsed)
                    result.heatwave_result = fut_hw.result()
                    try:
                        result.ocean_snapshot = fut_snap.result()
                    except Exception as exc:
                        logger.debug("Fast snapshot failed (non-fatal): %s", exc)
                return result

            elif intent == "compound_query":
                raw_q = parsed.get("raw_question", "").lower()
                wants_temp = any(k in raw_q for k in
                                 ("temperature", "temp", "warm", "hot", "cold", "cool"))
                wants_sal  = any(k in raw_q for k in
                                 ("salinity", "salt", "salty", "saline", "fresh", "psal"))

                from data.argo_loader import ArgoNoDataError
                argo_exc: Optional[Exception] = None
                no_data_msg: Optional[str] = None

                # ── Run Argo (temp + sal) and SST heatwave in PARALLEL ──────
                tasks: dict = {}
                executor = ThreadPoolExecutor(max_workers=3)
                if wants_temp or not wants_sal:
                    temp_parsed = dict(parsed)
                    temp_parsed["parameter"] = "temperature"
                    tasks["temp"] = executor.submit(self._run_qc, temp_parsed, True)
                if wants_sal:
                    sal_parsed = dict(parsed)
                    sal_parsed["parameter"] = "salinity"
                    tasks["sal"] = executor.submit(self._run_qc, sal_parsed, True)
                tasks["hw"] = executor.submit(self._run_heatwave, parsed)

                # Collect temperature result
                if "temp" in tasks:
                    try:
                        result.qc_result = tasks["temp"].result()
                    except ArgoNoDataError as exc:
                        no_data_msg = str(exc)
                        logger.info("No Argo temperature data: %s", exc)
                    except Exception as exc:
                        argo_exc = exc
                        logger.warning("Temperature QC sub-query failed: %s", exc)

                # Collect salinity result
                if "sal" in tasks:
                    try:
                        result.qc_result_salinity = tasks["sal"].result()
                    except ArgoNoDataError as exc:
                        no_data_msg = str(exc)
                        logger.info("No Argo salinity data: %s", exc)
                    except Exception as exc:
                        if argo_exc is None:
                            argo_exc = exc
                        logger.warning("Salinity QC sub-query failed: %s", exc)

                executor.shutdown(wait=False)

                # If no data found — give a scientific explanation, not a red error
                if result.qc_result is None and result.qc_result_salinity is None:
                    if no_data_msg:
                        result.warnings.append(
                            f"ℹ️ **No Argo float data available:** {no_data_msg}"
                        )
                    elif argo_exc:
                        err_str = str(argo_exc)
                        if not err_str.startswith("Could not retrieve"):
                            err_str = f"Could not retrieve Argo profile data: {err_str}"
                        result.error = (
                            f"{err_str}\n\n"
                            "💡 Try **Offline mode** or a **shorter date range**."
                        )
                        return result

                # Collect heatwave result (ran in parallel above)
                try:
                    result.heatwave_result = tasks["hw"].result()
                except Exception as exc:
                    result.warnings.append(
                        f"Heatwave check failed (SST data unavailable): {exc}"
                    )
                    logger.warning("Heatwave sub-query failed: %s", exc)

            else:
                result.qc_result = self._run_qc(parsed)

            # Auto-fetch FAST databases (GEBCO ~2s, Drifters ~3s, CRW ~5s)
            # Run in a background thread so it doesn't block result display.
            try:
                result.ocean_snapshot = self._fetch_fast_snapshot(parsed)
            except Exception as exc:
                logger.debug("Fast snapshot failed (non-fatal): %s", exc)

        except Exception as exc:
            logger.exception("Query engine error for question: %s", question)
            result.error = (
                f"{exc}\n\n"
                "💡 **Try:** switching to Offline mode, or using a shorter date range."
            )

        return result

    def fetch_snapshot(self, parsed: dict) -> "object":
        """
        Fetch the FULL OceanDB snapshot (HYCOM, WOD, CMEMS, drifters, GEBCO, CRW).
        SST and Argo are excluded — already fetched during answer().
        Called on-demand from the UI 'Load extra databases' button.
        """
        from data.ocean_db import OceanDB
        region     = parsed["region"]
        start_date = parsed["start_date"]
        end_date   = parsed["end_date"]

        yesterday = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end_date  = min(pd.Timestamp(end_date), pd.Timestamp(yesterday)).strftime("%Y-%m-%d")

        db = OceanDB(
            use_synthetic=self.use_synthetic,
            enable_argo=False,      # already fetched
            enable_sst=False,       # already fetched for heatwave
            enable_drifter=True,
            enable_hycom=True,
            enable_wod=True,
            enable_coral=True,
            enable_gebco=True,
            enable_cmems=True,
        )
        return db.fetch(region, start_date, end_date)

    def _fetch_fast_snapshot(self, parsed: dict) -> "object":
        """
        Fetch only the FAST databases — GEBCO (~2s), GDP Drifters (~3s),
        Coral Reef Watch (~5s). Called automatically on every answer().
        """
        from data.ocean_db import OceanDB
        region     = parsed["region"]
        start_date = parsed["start_date"]
        end_date   = parsed["end_date"]

        db = OceanDB(
            use_synthetic=self.use_synthetic,
            enable_argo=False,      # already fetched separately
            enable_sst=False,       # already fetched for heatwave
            enable_drifter=True,    # fast ~3s
            enable_hycom=False,     # slow — on-demand only
            enable_wod=False,       # slow — on-demand only
            enable_coral=True,      # fast ~5s
            enable_gebco=True,      # fast ~2s
            enable_cmems=False,     # slow — on-demand only
        )
        return db.fetch(region, start_date, end_date)

    # ------------------------------------------------------------------
    # Sub-runners
    # ------------------------------------------------------------------

    def _run_qc(self, parsed: dict, surface_only: bool = False) -> QCResult:
        """Fetch Argo data and run QC filtering.

        Uses an in-memory cache keyed by (region, start_date, end_date) so that
        compound queries (temperature + salinity) only hit the GDAC once.
        """
        from data.argo_loader import ArgoNoDataError

        region     = parsed["region"]
        start_date = parsed["start_date"]
        end_date   = parsed["end_date"]
        parameter  = parsed.get("parameter") or "temperature"

        if surface_only:
            depth_min, depth_max = SURFACE_DEPTH_MIN, SURFACE_DEPTH_MAX
        else:
            depth_min = float(parsed.get("depth_min") or 0.0)
            depth_max = float(parsed.get("depth_max") or 2000.0)

        cache_key = (
            region["lat_min"], region["lat_max"],
            region["lon_min"], region["lon_max"],
            start_date, end_date,
        )
        if cache_key in self._argo_cache:
            logger.info("Argo cache hit — reusing fetched data for %s", parameter)
            df = self._argo_cache[cache_key]
        else:
            try:
                df = self._fetch_argo(region, start_date, end_date)
            except ArgoNoDataError as e:
                # Store None in cache so salinity doesn't re-fetch
                self._argo_cache[cache_key] = None
                raise
            self._argo_cache[cache_key] = df
            logger.info("Argo cache stored — %d rows for %s..%s", len(df), start_date, end_date)

        if df is None:
            raise RuntimeError("No Argo data available for this region.")

        return apply_qc_filter(
            df, parameter=parameter, statistic="mean",
            depth_min=depth_min, depth_max=depth_max,
        )

    def _run_heatwave(self, parsed: dict) -> HeatwaveResult:
        """Fetch SST climatology and run heatwave detection.

        Uses the full region bounding box (not just the centre ±1°) so the
        climatology represents the whole query area.
        """
        region     = parsed["region"]
        start_date = parsed["start_date"]
        end_date   = parsed["end_date"]

        clim_start = max(
            pd.Timestamp(start_date) - pd.DateOffset(years=SST_CLIMATOLOGY_YEARS),
            pd.Timestamp("1981-09-01"),
        ).strftime("%Y-%m-%d")

        yesterday = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        sst_end   = min(pd.Timestamp(end_date), pd.Timestamp(yesterday)).strftime("%Y-%m-%d")

        if pd.Timestamp(clim_start) > pd.Timestamp(sst_end):
            raise RuntimeError(
                f"SST query end date ({end_date}) is before the OISST archive start "
                "(1981-09-01) or in the future. Please use a date before today."
            )

        # Fetch over the full region box, spatially averaged into a time series
        sst_ts = self._fetch_sst_region(region, clim_start, sst_end)
        return detect_heatwaves(sst_ts, query_start=start_date, query_end=sst_end)

    def _run_full_snapshot(self, parsed: dict) -> "object":
        """Fetch full OceanSnapshot (all databases) for a query."""
        region = parsed["region"]
        start_date = parsed["start_date"]
        end_date = parsed["end_date"]

        clim_start = max(
            pd.Timestamp(start_date) - pd.DateOffset(years=SST_CLIMATOLOGY_YEARS),
            pd.Timestamp("1981-09-01"),
        ).strftime("%Y-%m-%d")

        yesterday = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        sst_end = min(pd.Timestamp(end_date), pd.Timestamp(yesterday)).strftime("%Y-%m-%d")

        return self._get_ocean_db().fetch(
            region, start_date, sst_end,
            sst_clim_start=clim_start,
        )

    # ------------------------------------------------------------------
    # Data fetchers (swappable real ↔ synthetic)
    # ------------------------------------------------------------------

    def _fetch_argo(
        self,
        region: dict,
        start_date: str,
        end_date: str,
    ) -> "pd.DataFrame":
        if self.use_synthetic:
            from data.argo_loader import make_synthetic_profiles
            logger.info("Using synthetic Argo data")
            df = make_synthetic_profiles(
                lat_center=(region["lat_min"] + region["lat_max"]) / 2,
                lon_center=(region["lon_min"] + region["lon_max"]) / 2,
            )
            # Filter to requested date range
            df["TIME"] = pd.to_datetime(df["TIME"])
            mask = (df["TIME"] >= start_date) & (df["TIME"] <= end_date)
            return df[mask].reset_index(drop=True)
        else:
            from data.argo_loader import fetch_argo_profiles
            return fetch_argo_profiles(
                lon_min=region["lon_min"],
                lon_max=region["lon_max"],
                lat_min=region["lat_min"],
                lat_max=region["lat_max"],
                start_date=start_date,
                end_date=end_date,
            )

    def _fetch_sst(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
    ) -> "pd.DataFrame":
        if self.use_synthetic:
            from data.sst_loader import make_synthetic_sst
            logger.info("Using synthetic SST data")
            return make_synthetic_sst(
                lat_center=lat, lon_center=lon,
                start_date=start_date, end_date=end_date,
            )
        else:
            from data.sst_loader import fetch_sst_timeseries
            return fetch_sst_timeseries(
                lat=lat, lon=lon,
                start_date=start_date, end_date=end_date,
            )

    def _fetch_sst_region(
        self,
        region: dict,
        start_date: str,
        end_date: str,
    ) -> "pd.DataFrame":
        """Fetch SST over the full region box and return a spatially averaged time series."""
        lat_min = region["lat_min"]
        lat_max = region["lat_max"]
        lon_min = region["lon_min"]
        lon_max = region["lon_max"]

        # Cap the box to 2°×2° centred sub-box for large regions.
        # Smaller box = fewer grid cells = faster ERDDAP download per chunk.
        lat_span = lat_max - lat_min
        lon_span = lon_max - lon_min
        if lat_span > 4 or lon_span > 4:
            lat_c = (lat_min + lat_max) / 2
            lon_c = (lon_min + lon_max) / 2
            lat_min = lat_c - 1.0
            lat_max = lat_c + 1.0
            lon_min = lon_c - 1.0
            lon_max = lon_c + 1.0
            logger.info(
                "SST region capped to 2°×2° centre box: lat=[%.1f,%.1f] lon=[%.1f,%.1f]",
                lat_min, lat_max, lon_min, lon_max,
            )

        if self.use_synthetic:
            from data.sst_loader import make_synthetic_sst
            lat_c = (lat_min + lat_max) / 2
            lon_c = (lon_min + lon_max) / 2
            return make_synthetic_sst(
                lat_center=lat_c, lon_center=lon_c,
                start_date=start_date, end_date=end_date,
            )
        else:
            from data.sst_loader import fetch_sst
            df = fetch_sst(
                lat_min=lat_min, lat_max=lat_max,
                lon_min=lon_min, lon_max=lon_max,
                start_date=start_date, end_date=end_date,
            )
            # Spatially average into a daily time series
            ts = df.groupby("date")["sst"].mean().reset_index()
            return ts.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # NLU helpers
    # ------------------------------------------------------------------

    def _parse(self, question: str) -> dict:
        """
        Try LLM parser first; fall back silently to rule-based parser.
        Always returns a valid ParsedQuery dict.
        """
        latest = self._get_latest_date()

        if llm_parser.is_available():
            try:
                parsed = llm_parser.parse(question, dataset_latest_date=latest)
                if parsed:
                    logger.debug("LLM parser succeeded (provider=%s)", parsed.get("parser_used"))
                    return parsed
            except Exception as exc:
                logger.warning("LLM parser raised an exception: %s", exc)

        logger.debug("Using rule-based parser")
        return rule_parser.parse(question, dataset_latest_date=latest)

    def _get_latest_date(self) -> Optional[str]:
        """
        Return an ISO date string representing the most recent data point in
        the dataset. Used to anchor relative date phrases.

        For synthetic mode, returns today's date minus a fixed offset.
        For real mode, defaults to a fixed anchor that we know the Argo
        archive has good coverage for (can be overridden by the caller).
        """
        if self._dataset_latest_date:
            return self._dataset_latest_date

        if self.use_synthetic:
            # Synthetic profiles cover 2023-01-01 → ~2023-09-01 (24 cycles × 10 days)
            return "2023-08-01"

        # Real dataset: use yesterday as a conservative anchor
        # (Argo has a ~2-week processing lag, but we can always narrow the query)
        anchor = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        return anchor

    def set_latest_date(self, iso_date: str) -> None:
        """Override the dataset anchor date (e.g. after probing the actual data)."""
        self._dataset_latest_date = iso_date


# =============================================================================
# OSE integration (optional)
# -----------------------------------------------------------------------------
# This block adds one tool function that FloatAi can call to get an
# Ocean Stress Engine score for a region + time window.
#
# CONTRACT:
#   - Wrapped in try/except at every level: if the OSE package is missing,
#     deleted, or throws any error, FloatAi continues without it.
#   - read-only: imports compute_stress() which never writes to Argo data.
#   - One function added: get_stress_score(region, time_window) -> dict | None
#
# To disable: comment out or delete this entire block.
# FloatAi will run identically either way.
# =============================================================================

def get_stress_score(
    region: str,
    time_window: str,
    depth_layer: str = "surface",
    ecosystem_type: str = "general",
) -> "Optional[dict]":
    """
    OSE integration (optional) — query the Ocean Stress Engine for a
    thermohaline stress score and return a compact dict for FloatAi to render.

    Returns None silently on any failure so FloatAi is never broken by OSE.

    Parameters
    ----------
    region        : str   Named region (e.g. "Arabian Sea") or bbox dict.
    time_window   : str   "YYYY-MM", "YYYY-MM-DD/YYYY-MM-DD", "last N days".
    depth_layer   : str   "surface" | "mid" | "deep".
    ecosystem_type: str   "coral_reef" | "fisheries" | "general".

    Returns
    -------
    dict with keys: stress_score, category, confidence, summary, drivers,
                    variables_used, variables_missing, data_quality, caveats
    or None on any error.
    """
    try:
        from ocean_stress_engine.engine import compute_stress as _compute_stress

        result = _compute_stress(
            region=region,
            time_window=time_window,
            depth_layer=depth_layer,
            ecosystem_type=ecosystem_type,
        )

        return {
            "stress_score":      result.stress_score,
            "category":          result.category,
            "confidence":        result.confidence,
            "summary":           result.summary_text(),
            "drivers": [
                {
                    "variable":       d.variable,
                    "sub_score":      d.sub_score,
                    "observed":       d.observed,
                    "baseline_mean":  d.baseline_mean,
                    "anomaly_zscore": d.anomaly_zscore,
                    "unit":           d.unit,
                    "weight":         d.weight,
                }
                for d in result.drivers
            ],
            "variables_used":    result.variables_used,
            "variables_missing": result.variables_missing,
            "data_quality": {
                "floats":             result.n_floats,
                "profiles":           result.n_profiles,
                "readings_total":     result.n_readings_total,
                "readings_excluded":  result.n_readings_excluded,
            },
            "caveats":           result.caveats,
        }

    except Exception as _ose_exc:
        import logging as _logging
        _logging.getLogger("FloatAi.ose_hook").warning(
            "OSE integration failed (non-fatal): %s", _ose_exc
        )
        return None


# =============================================================================
# FAE integration (optional)
# -----------------------------------------------------------------------------
# Appended at the bottom of query_engine.py — mirrors the OSE hook pattern.
#
# CONTRACT:
#   - Wrapped in try/except: if the FAE package is missing, deleted, or throws
#     any error, FloatAi continues without it.
#   - Read-only: calls compute_advisory() which never writes to Argo data.
#   - One function added: get_fisher_advisory(location, day, language)
#     → dict | None
#
# To disable: comment out or delete this entire block.
# FloatAi will run identically either way.
# =============================================================================

def get_fisher_advisory(
    location: str,
    day: str = "today",
    language: str = "en",
) -> "Optional[dict]":
    """
    FAE integration (optional) — query the Fisher Advisory Engine for a
    plain-language sea-condition advisory and return a compact dict.

    Returns None silently on any failure so FloatAi is never broken by FAE.

    Parameters
    ----------
    location : str   Port name (e.g. "Nagapattinam") or lat/lon dict.
    day      : str   "today", "tomorrow", "last 7 days", or "YYYY-MM-DD".
    language : str   BCP-47 language code ("en", "ta"). Default: "en".

    Returns
    -------
    dict with keys: location, day, confidence, short_message,
                    indicators_ok, missing_indicators, caveats,
                    languages_served, fallback_used
    or None on any error.
    """
    try:
        from fisher_advisory_engine.engine import (
            compute_advisory as _fae_compute,
        )

        result = _fae_compute(
            location=location,
            day=day,
            language=language,
        )

        return {
            "location":          result.location_name,
            "day":               result.day_label,
            "trip_type":         result.trip_type,
            "confidence":        result.confidence.label,
            "confidence_reasons": result.confidence.reasons,
            "short_message":     result.short_message(language),
            "indicators_ok": {
                name: (ind.status in ("ok", "stale"))
                for name, ind in result.indicators.items()
            },
            "missing_indicators": result.missing_indicators,
            "caveats":            result.caveats[:5],
            "languages_served":   result.languages_served,
            "fallback_used":      result.fallback_used,
            "location_verified":  result.location_verified,
        }

    except Exception as _fae_exc:
        import logging as _logging
        _logging.getLogger("FloatAi.fae_hook").warning(
            "FAE integration failed (non-fatal): %s", _fae_exc
        )
        return None
