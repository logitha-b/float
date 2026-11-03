"""
ocean_stress_engine
-------------------
Ocean Stress Engine (OSE) — a self-contained module that computes an
explainable thermohaline stress score (0–100) for an ocean region and
time window, with per-variable driver breakdown.

Public API
----------
    from ocean_stress_engine import compute_stress
    result = compute_stress(
        region="Arabian Sea",
        time_window="2023-07",
        depth_layer="surface",
        ecosystem_type="coral_reef",
    )
    print(result.stress_score)
    print(result.summary_text())
    print(result.dashboard_card())

Deleting this package does NOT affect FloatAi. The only external touch is
an optional integration hook at the bottom of core/query_engine.py.
"""

from ocean_stress_engine.engine import compute_stress
from ocean_stress_engine.output import StressResult

__all__ = ["compute_stress", "StressResult"]
__version__ = "1.0.0"
