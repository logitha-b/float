"""
fisher_advisory_engine/rest_api.py
------------------------------------
Optional FastAPI REST endpoint for the Fisher Advisory Engine.

Requires:  pip install fastapi uvicorn
If FastAPI is not installed this module raises ImportError on import;
FloatAi is never affected (it never imports this module directly).

Run (from FloatAi/ directory):
    uvicorn fisher_advisory_engine.rest_api:app --host 0.0.0.0 --port 8001

Endpoints:
    GET  /fae/health          — liveness check
    GET  /fae/ports           — list all registered ports
    GET  /fae/languages       — list enabled languages
    POST /fae/advisory        — compute advisory (full JSON)
    POST /fae/advisory/short  — short message only

Example:
    curl -X POST http://localhost:8001/fae/advisory \\
      -H "Content-Type: application/json" \\
      -d '{"location":"Nagapattinam","day":"today","trip_type":"nearshore","language":"en"}'
"""

from __future__ import annotations

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from typing import List, Optional

if not _FASTAPI_AVAILABLE:
    raise ImportError(
        "FastAPI is not installed. "
        "Install with: pip install fastapi uvicorn\n"
        "FloatAi does not require FastAPI."
    )


app = FastAPI(
    title="Fisher Advisory Engine",
    description=(
        "Plain-language sea-condition advisories for coastal fishers "
        "in Tamil Nadu, in multiple languages."
    ),
    version="1.0.0",
    docs_url="/fae/docs",
    redoc_url="/fae/redoc",
)


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class AdvisoryRequest(BaseModel):
    location: object = Field(
        ...,
        description=(
            "Port name (e.g. 'Nagapattinam') or "
            "{\"lat\": float, \"lon\": float}."
        ),
        examples=["Nagapattinam"],
    )
    day: str = Field(
        default="today",
        description="'today', 'tomorrow', 'last 7 days', or 'YYYY-MM-DD'.",
    )
    trip_type: str = Field(
        default="nearshore",
        description="'nearshore' | 'offshore'.",
    )
    boat_type: Optional[str] = Field(
        default=None,
        description="'traditional' | 'mechanized' | null.",
    )
    language: object = Field(
        default=None,
        description=(
            "BCP-47 code ('en'), list of codes, 'all_enabled', or null "
            "(uses district default, falls back to 'en')."
        ),
    )
    use_synthetic: bool = Field(
        default=False,
        description="Use synthetic data (no network). For testing only.",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/fae/health", tags=["meta"])
def health():
    return {"status": "ok", "service": "fisher_advisory_engine", "version": "1.0.0"}


@app.get("/fae/ports", tags=["meta"])
def list_ports():
    """List all registered landing centres from ports.csv."""
    from fisher_advisory_engine.inputs import list_ports as _lp
    ports = _lp()
    return {
        "count": len(ports),
        "note": "All ports have verified=false until coordinates are confirmed.",
        "ports": [
            {
                "name":     p.get("name"),
                "district": p.get("district"),
                "verified": p.get("verified"),
                "lat":      p.get("lat") or None,
                "lon":      p.get("lon") or None,
            }
            for p in ports
        ],
    }


@app.get("/fae/languages", tags=["meta"])
def list_languages():
    """List enabled languages and their review status."""
    from fisher_advisory_engine.i18n.registry import get_language_config, get_mode
    lang_cfg = get_language_config()
    mode     = get_mode()
    result   = []
    for code, entry in lang_cfg.items():
        result.append({
            "code":          code,
            "native_name":   entry.get("native_name", ""),
            "enabled":       entry.get("enabled", False),
            "review_status": entry.get("review_status", "draft"),
            "served_in_mode": (
                entry.get("enabled") and (
                    mode == "demo" or entry.get("review_status") == "reviewed"
                )
            ),
        })
    return {"mode": mode, "languages": result}


@app.post("/fae/advisory", tags=["advisory"])
def compute_advisory_endpoint(req: AdvisoryRequest):
    """
    Compute a full fisher condition advisory.

    Returns JSON with indicators, rules, confidence, advisory text in
    all requested languages, and the safety line.
    """
    from fisher_advisory_engine.engine import compute_advisory

    try:
        result = compute_advisory(
            location=req.location,
            day=req.day,
            trip_type=req.trip_type,
            boat_type=req.boat_type,
            language=req.language,
            use_synthetic=req.use_synthetic,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"FAE error: {exc}")

    payload = result.to_dict()
    # Add rendered text for each served language
    payload["advisory_text"] = {}
    for lang in result.languages_served:
        try:
            payload["advisory_text"][lang] = result.short_message(lang)
        except Exception:
            payload["advisory_text"][lang] = None

    return JSONResponse(content=payload)


@app.post("/fae/advisory/short", tags=["advisory"])
def compute_advisory_short(req: AdvisoryRequest):
    """
    Compute a short phone-readable advisory message only.
    Returns {lang: text} for each served language + safety confirmation.
    """
    from fisher_advisory_engine.engine import compute_advisory

    try:
        result = compute_advisory(
            location=req.location,
            day=req.day,
            trip_type=req.trip_type,
            boat_type=req.boat_type,
            language=req.language,
            use_synthetic=req.use_synthetic,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    messages = {}
    for lang in result.languages_served:
        try:
            messages[lang] = result.short_message(lang)
        except Exception as exc:
            messages[lang] = f"[render error: {exc}]"

    return {
        "location":         result.location_name,
        "day":              result.day_label,
        "trip_type":        result.trip_type,
        "confidence":       result.confidence.label,
        "languages_served": result.languages_served,
        "messages":         messages,
        "safety_confirmed": True,   # safety_line is always last in every message
        "fallback_used":    result.fallback_used,
    }
