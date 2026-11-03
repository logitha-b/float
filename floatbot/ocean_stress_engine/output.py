"""
ocean_stress_engine/output.py
-------------------------------
StressResult — the unified result object for the Ocean Stress Engine.

Holds all computed data and provides three output formats from the same object:
  1. to_json()         — full JSON string (all fields)
  2. summary_text()    — plain-language paragraph for chat/CLI
  3. dashboard_card()  — HTML card for Streamlit or web embedding

All formatting lives here; engine.py just populates a StressResult.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Dict, List, Optional

from ocean_stress_engine.scoring   import ScoringResult, VariableDriver
from ocean_stress_engine.confidence import ConfidenceResult


# ---------------------------------------------------------------------------
# Evidence record
# ---------------------------------------------------------------------------

@dataclass
class EvidenceRecord:
    float_id: str
    cycle:    int
    depth_m:  float
    lat:      float
    lon:      float


# ---------------------------------------------------------------------------
# Main result object
# ---------------------------------------------------------------------------

@dataclass
class StressResult:
    # ── Identity ──────────────────────────────────────────────────────────
    region_name:     str
    region_bbox:     Dict[str, float]   # lat_min/max, lon_min/max
    period_start:    date
    period_end:      date
    depth_layer_min: float
    depth_layer_max: float
    depth_layer_lbl: str
    ecosystem_type:  str
    baseline_source: str               # "woa23" | "argo_historical" | "none"
    baseline_period: str               # e.g. "2005-2020"

    # ── Score ─────────────────────────────────────────────────────────────
    stress_score:    float             # 0-100
    category:        str               # LOW / MODERATE / HIGH / SEVERE
    confidence:      str               # HIGH / MEDIUM / LOW
    confidence_score: float            # numeric proxy 0-100

    # ── Variable breakdown ────────────────────────────────────────────────
    drivers:           List[VariableDriver]
    variables_used:    List[str]
    variables_missing: List[str]

    # ── Data quality ──────────────────────────────────────────────────────
    n_floats:            int
    n_profiles:          int
    n_readings_total:    int
    n_readings_excluded: int
    exclusion_reasons:   Dict[str, int]   # reason → count

    # ── Evidence ──────────────────────────────────────────────────────────
    evidence: List[EvidenceRecord] = field(default_factory=list)

    # ── Caveats ───────────────────────────────────────────────────────────
    caveats:            List[str] = field(default_factory=list)
    confidence_reasons: List[str] = field(default_factory=list)


    # -----------------------------------------------------------------------
    # Output 1: JSON
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict with all result fields."""
        drivers_list = [
            {
                "variable":       d.variable,
                "sub_score":      d.sub_score,
                "observed":       d.observed,
                "baseline_mean":  d.baseline_mean,
                "baseline_std":   d.baseline_std,
                "anomaly":        d.anomaly,
                "anomaly_zscore": d.anomaly_zscore,
                "weight":         d.weight,
                "unit":           d.unit,
            }
            for d in self.drivers
        ]
        evidence_list = [
            {"float_id": e.float_id, "cycle": e.cycle,
             "depth_m": e.depth_m, "lat": e.lat, "lon": e.lon}
            for e in self.evidence[:20]   # cap at 20 for JSON size
        ]
        return {
            "ose_version":    "1.0.0",
            "score_name":     "thermohaline_stress",
            "region": {
                "name":    self.region_name,
                "lat_min": self.region_bbox["lat_min"],
                "lat_max": self.region_bbox["lat_max"],
                "lon_min": self.region_bbox["lon_min"],
                "lon_max": self.region_bbox["lon_max"],
            },
            "period": {
                "start": self.period_start.isoformat(),
                "end":   self.period_end.isoformat(),
            },
            "depth_layer_m": {
                "min":   self.depth_layer_min,
                "max":   self.depth_layer_max,
                "label": self.depth_layer_lbl,
            },
            "ecosystem_type":    self.ecosystem_type,
            "baseline_source":   self.baseline_source,
            "baseline_period":   self.baseline_period,
            "stress_score":      self.stress_score,
            "category":          self.category,
            "confidence":        self.confidence,
            "confidence_score":  self.confidence_score,
            "variables_used":    self.variables_used,
            "variables_missing": self.variables_missing,
            "drivers":           drivers_list,
            "data_quality": {
                "floats":             self.n_floats,
                "profiles":           self.n_profiles,
                "readings_total":     self.n_readings_total,
                "readings_excluded":  self.n_readings_excluded,
                "exclusion_fraction": (
                    round(self.n_readings_excluded / self.n_readings_total, 4)
                    if self.n_readings_total > 0 else 0.0
                ),
                "exclusion_reasons":  self.exclusion_reasons,
            },
            "evidence":  evidence_list,
            "caveats":   self.caveats,
            "confidence_reasons": self.confidence_reasons,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


    # -----------------------------------------------------------------------
    # Output 2: Plain-language summary paragraph
    # -----------------------------------------------------------------------

    def summary_text(self) -> str:
        """
        Return a concise plain-English paragraph suitable for chat output or CLI.

        Structure:
          Sentence 1: score + category + region + period
          Sentence 2: main driver (variable + observed vs baseline + anomaly)
          Sentence 3: data quality (floats, readings, QC)
          Sentence 4: missing variables (if any)
          Sentence 5: confidence + reason (if degraded)
          Sentence 6: baseline source
        """
        lines: List[str] = []

        # Sentence 1 — headline
        cat_emoji = {"LOW": "🟢", "MODERATE": "🟡", "HIGH": "🟠", "SEVERE": "🔴"}.get(
            self.category, "⚪"
        )
        lines.append(
            f"{cat_emoji} The thermohaline stress score for {self.region_name} "
            f"({self.period_start.strftime('%d %b %Y')}–{self.period_end.strftime('%d %b %Y')}, "
            f"{self.depth_layer_lbl}) is "
            f"**{self.stress_score:.1f}/100** — **{self.category}** stress."
        )

        # Sentence 2 — main driver
        if self.drivers:
            d = self.drivers[0]
            direction = "above" if d.anomaly > 0 else "below"
            lines.append(
                f"The primary driver is **{d.variable}**: observed mean "
                f"{d.observed:.2f} {d.unit}, which is "
                f"{abs(d.anomaly):.2f} {d.unit} {direction} the climatological baseline "
                f"({d.baseline_mean:.2f} {d.unit}), a z-score of "
                f"{d.anomaly_zscore:+.2f} σ (sub-score {d.sub_score:.0f}/100, "
                f"weight {d.weight:.0%})."
            )

        # Sentence 3 — data quality
        excl_pct = (
            round(self.n_readings_excluded / self.n_readings_total * 100, 1)
            if self.n_readings_total > 0 else 0.0
        )
        lines.append(
            f"Based on {self.n_floats} Argo float(s), {self.n_profiles} profile(s), "
            f"{self.n_readings_total} readings "
            f"({self.n_readings_excluded} excluded by QC, {excl_pct:.0f}%)."
        )

        # Sentence 4 — missing variables
        if self.variables_missing:
            lines.append(
                f"Variables not available: {', '.join(self.variables_missing)}. "
                "Weights were re-normalised across available variables."
            )

        # Sentence 5 — confidence
        conf_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(self.confidence, "⚪")
        conf_note = (
            " " + self.confidence_reasons[0] if self.confidence_reasons else ""
        )
        lines.append(
            f"Confidence: {conf_emoji} **{self.confidence}**.{conf_note}"
        )

        # Sentence 6 — baseline source
        src_labels = {
            "woa23":            "World Ocean Atlas 2023 (Garcia et al., 2024)",
            "argo_historical":  f"historical Argo data {self.baseline_period}",
            "none":             "unavailable — anomalies could not be computed",
            "synthetic":        "synthetic test data",
        }
        src = src_labels.get(self.baseline_source, self.baseline_source)
        lines.append(f"Climatological baseline: {src}.")

        return "\n\n".join(lines)


    # -----------------------------------------------------------------------
    # Output 3: HTML dashboard card
    # -----------------------------------------------------------------------

    def dashboard_card(self) -> str:
        """
        Return a self-contained HTML string for embedding in Streamlit
        (st.markdown(..., unsafe_allow_html=True)) or any web page.
        """
        cat_color = {
            "LOW":      "#1b7a34",
            "MODERATE": "#b35c00",
            "HIGH":     "#c8580a",
            "SEVERE":   "#b71c1c",
        }.get(self.category, "#555")

        cat_bg = {
            "LOW":      "#e8f5e9",
            "MODERATE": "#fff3e0",
            "HIGH":     "#fce4d6",
            "SEVERE":   "#ffebee",
        }.get(self.category, "#f5f5f5")

        conf_color = {
            "HIGH":   "#1b7a34",
            "MEDIUM": "#b35c00",
            "LOW":    "#b71c1c",
        }.get(self.confidence, "#555")

        # Score gauge bar
        bar_width = int(self.stress_score)
        bar_color = cat_color

        # Per-variable bars
        var_bars = ""
        for d in self.drivers:
            bar_w = int(d.sub_score)
            direction = "▲" if d.anomaly > 0 else "▼"
            var_bars += f"""
            <div style="margin:0.4rem 0">
              <div style="display:flex;justify-content:space-between;
                          font-size:0.78rem;color:#1a2b3c;margin-bottom:2px">
                <span><b>{d.variable.title()}</b>
                  &nbsp;<span style="color:#5c7a99">
                    {d.observed:.2f}{d.unit} {direction} baseline {d.baseline_mean:.2f}{d.unit}
                    (z={d.anomaly_zscore:+.2f}σ)
                  </span>
                </span>
                <span style="font-weight:700;color:{bar_color}">{d.sub_score:.0f}</span>
              </div>
              <div style="background:#e0e0e0;border-radius:4px;height:8px">
                <div style="background:{bar_color};width:{bar_w}%;
                            height:8px;border-radius:4px"></div>
              </div>
            </div>"""

        # Missing variables note
        missing_html = ""
        if self.variables_missing:
            missing_html = (
                f'<div style="font-size:0.74rem;color:#888;margin-top:0.4rem">'
                f'⚠ Missing: {", ".join(self.variables_missing)} '
                f'(weights re-normalised)</div>'
            )

        # Caveats
        caveat_html = ""
        if self.caveats or self.confidence_reasons:
            all_notes = self.caveats + self.confidence_reasons
            items = "".join(f"<li>{c}</li>" for c in all_notes)
            caveat_html = (
                f'<details style="margin-top:0.6rem">'
                f'<summary style="font-size:0.75rem;color:#888;cursor:pointer">'
                f'ℹ️ {len(all_notes)} caveat(s)</summary>'
                f'<ul style="font-size:0.73rem;color:#888;margin:0.3rem 0 0 1rem">'
                f'{items}</ul></details>'
            )

        excl_pct = (
            round(self.n_readings_excluded / self.n_readings_total * 100, 1)
            if self.n_readings_total > 0 else 0.0
        )

        html = f"""
<div style="background:#fff;border:1px solid #d0e4f7;border-radius:14px;
            padding:1.4rem 1.6rem;max-width:520px;
            box-shadow:0 2px 12px rgba(21,101,192,0.07);font-family:'Segoe UI',sans-serif">

  <!-- Header -->
  <div style="display:flex;justify-content:space-between;align-items:flex-start;
              margin-bottom:0.8rem">
    <div>
      <div style="font-size:0.72rem;font-weight:700;color:#5c7a99;
                  text-transform:uppercase;letter-spacing:0.8px">
        🌊 Ocean Stress Engine v1.0 · Thermohaline Stress
      </div>
      <div style="font-size:0.88rem;font-weight:700;color:#0a2342;margin-top:2px">
        {self.region_name}
      </div>
      <div style="font-size:0.73rem;color:#5c7a99">
        {self.period_start.strftime('%d %b %Y')} – {self.period_end.strftime('%d %b %Y')}
        &nbsp;·&nbsp; {self.depth_layer_lbl}
        &nbsp;·&nbsp; {self.ecosystem_type.replace('_',' ').title()}
      </div>
    </div>
    <div style="text-align:center;min-width:72px">
      <div style="font-size:2rem;font-weight:900;color:{cat_color};line-height:1">
        {self.stress_score:.0f}
      </div>
      <div style="font-size:0.6rem;color:#5c7a99">/ 100</div>
      <div style="background:{cat_bg};color:{cat_color};border-radius:20px;
                  padding:1px 10px;font-size:0.7rem;font-weight:700;margin-top:2px">
        {self.category}
      </div>
    </div>
  </div>

  <!-- Gauge bar -->
  <div style="background:#e8f0fe;border-radius:6px;height:10px;margin-bottom:1rem">
    <div style="background:{bar_color};width:{bar_width}%;height:10px;
                border-radius:6px;transition:width 0.4s"></div>
  </div>

  <!-- Per-variable bars -->
  <div style="font-size:0.75rem;font-weight:700;color:#5c7a99;
              text-transform:uppercase;letter-spacing:0.7px;margin-bottom:0.4rem">
    Variable Drivers
  </div>
  {var_bars}
  {missing_html}

  <!-- Data quality -->
  <div style="display:flex;gap:1rem;margin-top:0.9rem;flex-wrap:wrap">
    <div style="background:#f4f8ff;border-radius:8px;padding:0.4rem 0.7rem;
                font-size:0.73rem;color:#1a2b3c">
      <b>{self.n_floats}</b> float(s)
    </div>
    <div style="background:#f4f8ff;border-radius:8px;padding:0.4rem 0.7rem;
                font-size:0.73rem;color:#1a2b3c">
      <b>{self.n_profiles}</b> profile(s)
    </div>
    <div style="background:#f4f8ff;border-radius:8px;padding:0.4rem 0.7rem;
                font-size:0.73rem;color:#1a2b3c">
      <b>{self.n_readings_total}</b> readings
      <span style="color:#888">({excl_pct:.0f}% excl.)</span>
    </div>
    <div style="background:#f4f8ff;border-radius:8px;padding:0.4rem 0.7rem;
                font-size:0.73rem;color:{conf_color}">
      Confidence: <b>{self.confidence}</b>
    </div>
  </div>

  <!-- Caveats -->
  {caveat_html}

  <!-- Footer -->
  <div style="font-size:0.68rem;color:#aaa;margin-top:0.8rem;border-top:1px solid #f0f0f0;
              padding-top:0.5rem">
    Baseline: {self.baseline_source} · {self.baseline_period}
    &nbsp;·&nbsp; Weights: PLACEHOLDER (see config.yaml)
  </div>
</div>
"""
        return html


# ---------------------------------------------------------------------------
# Factory: build StressResult from component pieces
# ---------------------------------------------------------------------------

def build_result(
    inputs,             # ValidatedInputs
    scoring:            ScoringResult,
    confidence:         ConfidenceResult,
    n_floats:           int,
    n_profiles:         int,
    n_readings_total:   int,
    n_readings_excluded: int,
    exclusion_reasons:  Dict[str, int],
    evidence:           List[EvidenceRecord],
    baseline_source:    str,
) -> StressResult:
    """
    Assemble a StressResult from the outputs of all pipeline stages.
    Called by engine.py.
    """
    return StressResult(
        region_name=inputs.region.name,
        region_bbox={
            "lat_min": inputs.region.lat_min,
            "lat_max": inputs.region.lat_max,
            "lon_min": inputs.region.lon_min,
            "lon_max": inputs.region.lon_max,
        },
        period_start=inputs.time_window.start,
        period_end=inputs.time_window.end,
        depth_layer_min=inputs.depth_layer.min_dbar,
        depth_layer_max=inputs.depth_layer.max_dbar,
        depth_layer_lbl=inputs.depth_layer.label,
        ecosystem_type=inputs.ecosystem_type,
        baseline_source=baseline_source,
        baseline_period=(
            f"{inputs.baseline_start.year}–{inputs.baseline_end.year}"
        ),
        stress_score=scoring.stress_score,
        category=scoring.category,
        confidence=confidence.label,
        confidence_score=confidence.score,
        drivers=scoring.drivers,
        variables_used=scoring.variables_used,
        variables_missing=scoring.variables_missing,
        n_floats=n_floats,
        n_profiles=n_profiles,
        n_readings_total=n_readings_total,
        n_readings_excluded=n_readings_excluded,
        exclusion_reasons=exclusion_reasons,
        evidence=evidence,
        caveats=scoring.caveats,
        confidence_reasons=confidence.reasons,
    )
