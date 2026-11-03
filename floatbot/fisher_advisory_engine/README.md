# Fisher Advisory Engine (FAE)

A self-contained FloatAi module that turns real ocean-condition data into
short, plain-language **condition advisories** for coastal fishers in Tamil Nadu,
in multiple languages.

---

## What it does

Given a landing centre (or lat/lon point), a day, and a trip type, the FAE:

1. Fetches sea-surface temperature, currents, chlorophyll, and bathymetry
   from existing FloatAi data loaders (read-only)
2. Runs the Hobday (2016) marine heatwave check via the existing heatwave module
3. Computes a thermal front indicator from the SST spatial gradient
4. Retrieves Argo subsurface context (mixed layer depth) if a float is nearby
5. Evaluates a rules engine (`rules.yaml`) — every advisory line traces to a
   rule id and the indicator values that triggered it
6. Computes confidence (HIGH/MEDIUM/LOW) from data age, gaps, and coverage
7. Renders the result in one or more languages

## What it does NOT do

- Never predicts exact fish locations
- Never guarantees a catch
- Never declares any trip safe or unsafe
- Never uses machine learning
- Never presents observation data as a forecast
- Never machine-translates at runtime

Every output ends with a fixed **safety line** telling the user to check
official IMD and INCOIS weather, wave, and cyclone warnings before sailing.

---

## Quick start

```bash
# From the FloatAi/ directory

# Short advisory (English)
python -m fisher_advisory_engine --location Nagapattinam --day today

# Tamil
python -m fisher_advisory_engine --location Nagapattinam --day today --lang ta

# Offshore trip, full JSON
python -m fisher_advisory_engine --location Nagapattinam --trip offshore --output json

# Offline / no network (synthetic data)
python -m fisher_advisory_engine --location Nagapattinam --day today --synthetic

# List ports
python -m fisher_advisory_engine --list-ports

# Rules status report
python -m fisher_advisory_engine --rules-report
```

### Python API

```python
from fisher_advisory_engine import compute_advisory

result = compute_advisory(
    location="Nagapattinam",
    day="today",
    trip_type="nearshore",   # "nearshore" | "offshore"
    language="en",           # "en" | "ta" | ["en","ta"] | "all_enabled"
)

print(result.short_message("en"))     # phone-readable message
print(result.speech_text("en"))       # spoken-word version
print(result.dashboard_card("en"))    # HTML card
print(result.to_json())               # full JSON
```

### REST API (optional — requires FastAPI)

```bash
pip install fastapi uvicorn
uvicorn fisher_advisory_engine.rest_api:app --host 0.0.0.0 --port 8001
```

```bash
curl -X POST http://localhost:8001/fae/advisory \
  -H "Content-Type: application/json" \
  -d '{"location":"Nagapattinam","day":"today","trip_type":"nearshore","language":"en"}'
```

---

## Inputs

| Parameter | Required | Type | Description |
|---|---|---|---|
| `location` | ✅ | str or dict | Port name or `{"lat": float, "lon": float}` |
| `day` | ✗ | str | `"today"` (default), `"tomorrow"`, `"last 7 days"`, `"YYYY-MM-DD"` |
| `trip_type` | ✗ | str | `"nearshore"` (default) or `"offshore"` |
| `boat_type` | ✗ | str | `"traditional"` or `"mechanized"` (metadata only in v1) |
| `language` | ✗ | str/list | `"en"`, `"ta"`, `["en","ta"]`, `"all_enabled"`, or None |

---

## Output (JSON fields)

```json
{
  "fae_version": "1.0.0",
  "location": {"name":"Nagapattinam","lat":0.0,"lon":0.0,"verified":false},
  "period": {"start":"2024-06-15","end":"2024-06-15","label":"today"},
  "trip_type": "nearshore",
  "indicators": {
    "sst":     {"value":{"sst_c":28.5,"anomaly_c":0.4,"heatwave":false},...},
    "currents":{"value":{"speed_ms":0.32,"direction_cardinal":"SW"},...},
    "thermal_front":{"value":{"detected":false},...},
    "productivity":{"value":{"chl_mg_m3":0.45,"category":"MODERATE"},...},
    "subsurface":{"value":{"mld_m":35.0,"nearest_km":120.0},...},
    "bathymetry":{"value":{"depth_m":45.0},...}
  },
  "missing_indicators": ["oxygen","ph"],
  "rules_triggered": [],
  "rules_disabled": ["A001_warm_anomaly_caution", "...14 more"],
  "confidence": {"level":"LOW","score":20.0,"reasons":["..."]},
  "advisory_text": {"en":"...","ta":"..."},
  "safety_confirmed": true,
  "disclaimer": "...",
  "caveats": ["..."]
}
```

---

## Indicators

| # | Name | Dataset | Source |
|---|---|---|---|
| 1 | SST + anomaly + heatwave | NOAA OISST v2.1 | Huang et al. (2020) |
| 2 | Surface currents | NOAA HYCOM model | NCEI ERDDAP |
| 3 | Thermal front | Derived from OISST gradient | FAE algorithm |
| 4 | Productivity (Chl-a) | CMEMS satellite | Copernicus Marine Service |
| 5 | Subsurface (MLD) | Argo GDAC | argopy |
| 6 | Bathymetry | ETOPO1 via CoastWatch | Amante & Eakins (2009) |

---

## Rules

Rules live in `rules.yaml`. A rule is **ACTIVE** only when its `reference`
field contains a real citation — not `"UNSOURCED - needs citation"`.

**All 14 rules are currently disabled.** Run `--rules-report` to see them.

To enable a rule:
1. Find a peer-reviewed or agency source for the condition-to-biology link.
2. Add the citation to the `reference:` field in `rules.yaml`.
3. Re-run tests: `python -m pytest fisher_advisory_engine/tests/ -v`

---

## How thresholds are set

All numeric thresholds are in `config.yaml` and labeled **PLACEHOLDER**.
They are initial estimates pending calibration against:
- Published oceanographic literature for the Bay of Bengal / Gulf of Mannar
- INCOIS PFZ reference data (see `validate_incois.py`)
- Domain-expert review

To change a threshold:
1. Edit `config.yaml` with the new value and add a `# source:` comment.
2. Run the validation script if INCOIS data is available.
3. Run unit tests.

---

## Languages

| Code | Name | Status | Notes |
|---|---|---|---|
| `en` | English | reviewed | Fully enabled |
| `ta` | தமிழ் (Tamil) | draft | Enabled in demo mode; **needs native review** |
| `te` | తెలుగు (Telugu) | draft | Disabled scaffold |
| `ml` | മലയാളം (Malayalam) | draft | Disabled scaffold |
| `hi` | हिन्दी (Hindi) | draft | Disabled scaffold |

### Adding a new language in 5 steps

1. Copy `i18n/en.yaml` to `i18n/<code>.yaml`
2. Translate every value (never change the keys)
3. Add an entry to `config.yaml` under `languages:`
4. Run tests: `python -m pytest fisher_advisory_engine/tests/test_i18n.py -v`
5. Run review workflow:
   ```bash
   python -m fisher_advisory_engine.i18n export-review --lang <code> --out review.csv
   # reviewer fills in review.csv
   python -m fisher_advisory_engine.i18n import-review --lang <code> --file review.csv
   ```

---

## Running tests

```bash
# All FAE unit tests (no network — synthetic data only)
python -m pytest fisher_advisory_engine/tests/ -v

# Real-data validation (requires INCOIS PFZ files in validation_data/)
python fisher_advisory_engine/validate_incois.py
```

**Synthetic-data tests** (in `tests/`) use seeded mock data and never call
any live data source. They run in < 5 seconds with no network.

**Real-data validation** (`validate_incois.py`) uses live ERDDAP fetches
and INCOIS reference files. It is kept separate from unit tests.

---

## Known limitations

- **Port coordinates are all unverified.** Every port in `ports.csv` has
  `verified=false` and placeholder (empty) coordinates. All indicator values
  computed for a named port are meaningless until the operator confirms
  the coordinates.

- **All rules are disabled.** No advisory condition lines will appear until
  citations are added to `rules.yaml`.

- **All thresholds are placeholders.** The numeric values in `config.yaml`
  are first-pass estimates, not validated thresholds.

- **Tamil translation is draft.** The `ta.yaml` file contains a first-pass
  draft that has NOT been reviewed by a native Tamil speaker.

- **OISST resolution is 0.25° (~28 km).** Small-scale coastal features
  (fronts, plumes) may not be resolved. Confidence is automatically lowered
  for nearshore queries.

- **Chlorophyll is frequently unavailable.** Cloud cover and nearshore
  turbidity (CDOM) cause frequent data gaps. The FAE returns "no data"
  rather than interpolating.

- **Argo floats are sparse nearshore.** Mixed layer depth is often
  unavailable for Tamil Nadu coastal waters.

- **No target-species logic in v1.** Species-specific thermal preferences
  are out of scope until sourced from published literature.

---

## What the module does NOT claim

- It does not claim that any conditions will produce a specific catch.
- It does not claim that any trip is safe.
- It does not replace official IMD or INCOIS weather and wave advisories.
- It does not provide navigation guidance.
- The front detection and chlorophyll indicators have not been validated
  against independent observations for Tamil Nadu coastal waters.

---

## File structure

```
fisher_advisory_engine/
├── __init__.py              # Public API
├── __main__.py              # python -m entry point
├── config.yaml              # All thresholds and language registry
├── rules.yaml               # All rules (currently all disabled)
├── ports.csv                # Tamil Nadu fish landing centres
├── engine.py                # compute_advisory() orchestrator
├── inputs.py                # Input validation, port resolver
├── indicators.py            # 6 ocean-condition indicators
├── front_detection.py       # SST gradient thermal front
├── rules_engine.py          # Rule evaluation
├── confidence.py            # Confidence computation
├── output.py                # AdvisoryResult, JSON, card, speech
├── rest_api.py              # FastAPI endpoint (optional)
├── cli.py                   # CLI
├── validate_incois.py       # Real-data validation (separate from tests)
├── validation_data/         # Place INCOIS PFZ CSV files here
├── i18n/
│   ├── registry.py          # Language loader, key validation
│   ├── renderer.py          # Text rendering
│   ├── review_workflow.py   # export/import review CSV
│   ├── en.yaml              # English (reviewed)
│   ├── ta.yaml              # Tamil (draft)
│   ├── te.yaml              # Telugu (scaffold)
│   ├── ml.yaml              # Malayalam (scaffold)
│   ├── hi.yaml              # Hindi (scaffold)
│   ├── forbidden_en.yaml    # Forbidden phrases (English)
│   └── forbidden_ta.yaml    # Forbidden phrases (Tamil)
└── tests/
    ├── mock_data.py
    ├── test_inputs.py
    ├── test_indicators.py
    ├── test_front_detection.py
    ├── test_rules_engine.py
    ├── test_confidence.py
    └── test_i18n.py
```

---

## Deliverables report

### Disabled rules (all 14 — need citations)

Run `python -m fisher_advisory_engine --rules-report` for the full list.

### Unverified ports

All 20 ports in `ports.csv` have `verified=false`. Coordinates are empty
(TODO). No port-level advisory is meaningful until coordinates are confirmed.

### Placeholder thresholds

See every `# PLACEHOLDER` comment in `config.yaml`.

### Language status

| Language | Total keys | Reviewed | Draft | Missing |
|---|---|---|---|---|
| en | 70+ | 70+ | 0 | 0 |
| ta | 70+ | 0 | 70+ | 0 |
| te | 70+ | 0 | 0 (TODO) | 0 |
| ml | 70+ | 0 | 0 (TODO) | 0 |
| hi | 70+ | 0 | 0 (TODO) | 0 |

*Run `python -m pytest fisher_advisory_engine/tests/test_i18n.py -v`
to get the exact key counts.*
