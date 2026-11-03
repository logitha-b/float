# Ocean Stress Engine (OSE)

A self-contained FloatAi module that computes an explainable
**thermohaline stress score** (0–100) for any ocean region and time window,
with a per-variable driver breakdown.

---

## What it does

Given a region, time window, and optional depth/ecosystem settings, the OSE:

1. Fetches Argo float profiles for the region and period
2. Applies the same QC filtering as FloatAi (imported, never modified)
3. Fetches a climatological baseline from WOA23 (or historical Argo 2005–2020)
4. Computes z-score anomalies per variable (temperature, salinity)
5. Maps anomalies to 0–100 sub-scores via a configurable piecewise function
6. Combines sub-scores with ecosystem-specific weights into one stress score
7. Returns JSON, a plain-text paragraph, and an HTML dashboard card

**Score categories:** LOW (0–25) · MODERATE (25–50) · HIGH (50–75) · SEVERE (75–100)

The score is named **thermohaline stress** because v1.0 uses temperature
and salinity only. Oxygen and pH are listed as optional future variables
(BGC-Argo coverage is currently too sparse for most regions).

---

## Quick start

```bash
# From the FloatAi/ directory
python -m ocean_stress_engine \
    --region "Arabian Sea" \
    --time_window "2023-07" \
    --depth_layer surface \
    --ecosystem coral_reef
```

### Python API

```python
from ocean_stress_engine import compute_stress

result = compute_stress(
    region="Arabian Sea",
    time_window="2023-07",
    depth_layer="surface",          # surface | mid | deep | None
    ecosystem_type="coral_reef",    # coral_reef | fisheries | general
)

print(result.stress_score)          # e.g. 67.4
print(result.category)              # e.g. "HIGH"
print(result.confidence)            # e.g. "MEDIUM"
print(result.summary_text())        # plain-English paragraph
print(result.to_json())             # full JSON string
result.dashboard_card()             # HTML string
```

### REST API

```bash
# Start the server (requires: pip install fastapi uvicorn)
uvicorn ocean_stress_engine.rest_api:app --host 0.0.0.0 --port 8000

# Query
curl -X POST http://localhost:8000/ose/stress \
  -H "Content-Type: application/json" \
  -d '{"region":"Arabian Sea","time_window":"2023-07","ecosystem_type":"coral_reef"}'
```

---

## Inputs

| Parameter | Required | Type | Description |
|---|---|---|---|
| `region` | ✅ | str or dict | Named region (e.g. `"Arabian Sea"`) or bbox `{"lat_min":8,"lat_max":26,"lon_min":55,"lon_max":78}` |
| `time_window` | ✅ | str | `"2023-07"`, `"2023-07-01/2023-07-31"`, or `"last 30 days"` |
| `depth_layer` | ✗ | str | `"surface"` (0–50 m), `"mid"` (50–200 m), `"deep"` (200–1000 m). Default: surface |
| `ecosystem_type` | ✗ | str | `"coral_reef"`, `"fisheries"`, `"general"`. Default: general |
| `baseline_period` | ✗ | str | `"2005-2020"` (default) or `"2010-2015"` etc. |

---

## Output (JSON fields)

```json
{
  "region": {...},
  "period": {"start": "...", "end": "..."},
  "depth_layer_m": {"min": 0, "max": 50},
  "stress_score": 67.4,
  "category": "HIGH",
  "confidence": "MEDIUM",
  "variables_used": ["temperature", "salinity"],
  "variables_missing": ["oxygen", "ph"],
  "drivers": [
    {
      "variable": "temperature",
      "sub_score": 72.1,
      "observed": 29.8,
      "baseline": 27.9,
      "anomaly": 1.9,
      "anomaly_zscore": 2.3,
      "weight": 0.70
    }
  ],
  "data_quality": {
    "floats": 12,
    "profiles": 38,
    "readings_total": 4820,
    "readings_excluded": 143,
    "exclusion_reasons": {"spike": 80, "gradient": 63}
  },
  "evidence": [{"float_id": "1234567", "cycle": 42, "depth_m": 5.0}],
  "caveats": ["oxygen and ph unavailable — weights re-normalised across temperature and salinity"]
}
```

---

## How weights work

Weights live in `config.yaml` under `ecosystem_weights`. They are **relative**
— the engine re-normalises them to sum to 1.0 across available variables.

If a variable has no data, its weight is redistributed proportionally
to the other variables and a caveat is added to the output.

**Current weights are labeled PLACEHOLDERS** — see `config.yaml` for details
on calibration methodology.

---

## How to change weights

1. Edit `config.yaml` → `ecosystem_weights`
2. Run the validation script:
   ```bash
   python ocean_stress_engine/validate_stress_events.py
   ```
3. Check that known severe events (2016 bleaching) score ≥ 50
4. Check that non-event years score ≤ 30

---

## Climatology baseline

The OSE uses a **two-tier approach**:

1. **WOA23** (preferred): fetches monthly climatological mean + std from
   NOAA NCEI ERDDAP (`woa23_decav_t_mn`, `woa23_decav_s_mn`).
   Citation: Garcia et al. (2024), *NOAA Atlas NESDIS 89*.

2. **Historical Argo fallback**: if WOA23 is unavailable, computes mean ± std
   from real Argo profiles in the same region, same calendar month,
   over the baseline period (default 2005–2020).

Both sources cache results in `ose_cache/` (JSON, with `ose_` prefix,
TTL = 30 days by default).

---

## Running tests

```bash
# From FloatAi/
python -m pytest ocean_stress_engine/tests/ -v
```

Tests use seeded synthetic data only — no network calls.

---

## Files added (none modified except integration hook)

```
ocean_stress_engine/
├── __init__.py
├── config.yaml
├── README.md
├── inputs.py
├── climatology.py
├── scoring.py
├── confidence.py
├── output.py
├── engine.py
├── rest_api.py
├── cli.py
├── validate_stress_events.py
└── tests/
    ├── __init__.py
    ├── mock_data.py
    ├── test_scoring.py
    ├── test_inputs.py
    └── test_missing_variables.py
```

**Only existing file touched:** `core/query_engine.py` — one optional
`try/except` block appended at the very bottom, clearly marked
`# OSE integration (optional)`.
