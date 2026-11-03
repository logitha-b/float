"""
ocean_stress_engine/rest_api.py
---------------------------------
FastAPI REST endpoint for the Ocean Stress Engine.

Endpoint:  POST /ose/stress
Returns:   JSON (StressResult.to_dict())

Optional — requires:  pip install fastapi uvicorn
If FastAPI is not installed this module fails on import silently;
FloatAi is unaffected (it never imports this module).

Run:
    uvicorn ocean_stress_engine.rest_api:app --host 0.0.0.0 --port 8000

Example:
    curl -X POST http://localhost:8000/ose/stress \\
      -H "Content-Type: application/json" \\
      -d '{
            "region": "Arabian Sea",
            "time_window": "2023-07",
            "depth_layer": "surface",
            "ecosystem_type": "coral_reef"
          }'
"""

from __future__ import annotations

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from typing import Optional


# ---------------------------------------------------------------------------
# Only define the app if FastAPI is installed
# ---------------------------------------------------------------------------

if not _FASTAPI_AVAILABLE:
    raise ImportError(
        "FastAPI is not installed. "
        "Install it with:  pip install fastapi uvicorn\n"
        "FloatAi itself does not require FastAPI."
    )


app = FastAPI(
    title="Ocean Stress Engine",
    description=(
        "Computes an explainable thermohaline stress score (0–100) "
        "for an ocean region and time window using Argo float data."
    ),
    version="1.0.0",
    docs_url="/ose/docs",
    redoc_url="/ose/redoc",
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class StressRequest(BaseModel):
    region: object = Field(
        ...,
        description=(
            "Named region string (e.g. 'Arabian Sea') or bbox dict "
            "{lat_min, lat_max, lon_min, lon_max}."
        ),
        examples=["Arabian Sea"],
    )
    time_window: str = Field(
        ...,
        description=(
            "Time window: 'YYYY-MM', 'YYYY-MM-DD/YYYY-MM-DD', "
            "'last N days', or 'YYYY'."
        ),
        examples=["2023-07"],
    )
    depth_layer: Optional[str] = Field(
        default="surface",
        description="'surface' (0–50 m), 'mid' (50–200 m), or 'deep' (200–1000 m).",
    )
    ecosystem_type: Optional[str] = Field(
        default="general",
        description="'coral_reef', 'fisheries', or 'general'.",
    )
    baseline_period: Optional[str] = Field(
        default=None,
        description="Climatology baseline, e.g. '2005-2020'. Default: 2005-2020.",
    )
    use_synthetic: Optional[bool] = Field(
        default=False,
        description="Use synthetic data (no network). For testing only.",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/ose/health", tags=["meta"])
def health():
    """Simple liveness check."""
    return {"status": "ok", "service": "ocean_stress_engine", "version": "1.0.0"}


@app.get("/ose/regions", tags=["meta"])
def list_regions():
    """List all supported named regions."""
    from ocean_stress_engine.inputs import list_regions as _lr
    return {"regions": _lr()}


@app.post("/ose/stress", tags=["stress"])
def compute_stress_endpoint(req: StressRequest):
    """
    Compute the thermohaline stress score for a region and time window.

    Returns a full StressResult JSON with:
    - stress_score (0-100), category, confidence
    - per-variable drivers with observed, baseline, anomaly, weight
    - data quality metrics
    - plain-text summary and caveats
    """
    from ocean_stress_engine.engine import compute_stress

    try:
        result = compute_stress(
            region=req.region,
            time_window=req.time_window,
            depth_layer=req.depth_layer,
            ecosystem_type=req.ecosystem_type or "general",
            baseline_period=req.baseline_period,
            use_synthetic=req.use_synthetic or False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Internal OSE error: {exc}",
        )

    payload = result.to_dict()
    payload["summary"] = result.summary_text()
    return JSONResponse(content=payload)


@app.post("/ose/stress/summary", tags=["stress"])
def compute_stress_summary(req: StressRequest):
    """
    Same as /ose/stress but returns only the plain-text summary + score,
    suitable for chat integration.
    """
    from ocean_stress_engine.engine import compute_stress

    try:
        result = compute_stress(
            region=req.region,
            time_window=req.time_window,
            depth_layer=req.depth_layer,
            ecosystem_type=req.ecosystem_type or "general",
            baseline_period=req.baseline_period,
            use_synthetic=req.use_synthetic or False,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {
        "stress_score":  result.stress_score,
        "category":      result.category,
        "confidence":    result.confidence,
        "summary":       result.summary_text(),
        "variables_used": result.variables_used,
        "variables_missing": result.variables_missing,
    }
