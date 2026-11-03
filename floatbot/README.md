# FloatAi — Conversational AI for Argo Ocean Data

FloatAi lets you ask plain-English questions about ocean data from the global [Argo float network](https://argo.ucsd.edu/) and get answers that are quality-controlled, explainable, and grounded in real climate science.

## Three core innovations

| Pillar | What it does |
|--------|-------------|
| **Explainable QC** | Every reading is checked against its official Argo QC flag. If data is excluded, the system explains *which test failed* and *why*, in plain language. |
| **Cross-dataset reasoning** | Compound questions ("was this unusual?") are answered by comparing Argo readings against NOAA OISST climatology and detecting marine heatwaves using the published [Hobday et al. (2016)](https://doi.org/10.1016/j.pocean.2015.12.014) method. |
| **Benchmarked trust** | Accuracy is evaluated against a naive (QC-blind) baseline using narrow single-profile queries with known ground truth, producing real accuracy numbers. |

---

## Project structure

```
FloatAi/
├── requirements.txt
├── README.md
├── app.py                   # Streamlit chat interface (Feature 5)
├── data/
│   ├── argo_loader.py       # Real Argo data via argopy (+ synthetic fallback)
│   └── sst_loader.py        # Real SST data via NOAA OISST/ERDDAP (+ synthetic)
├── core/
│   ├── qc_module.py         # Feature 1: QC-aware filtering & explanation
│   ├── heatwave_module.py   # Feature 2: Hobday 2016 marine heatwave detection
│   └── query_engine.py      # Feature 3: NLU routing & orchestration
├── nlu/
│   ├── llm_parser.py        # Gemini / OpenAI NLU parser
│   └── rule_parser.py       # No-API-key regex fallback parser
└── eval/
    └── benchmark.py         # Feature 4: accuracy benchmark
```

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows SSL note:** If you get `unable to get local issuer certificate` errors
> (common behind corporate firewalls / antivirus), install `pip-system-certs`:
> ```bash
> pip install pip-system-certs
> ```
> This is already listed in `requirements.txt`.

> **Critical version pin:** `erddapy<3` is required. erddapy 3.x removed
> `_quote_string_constraints`, which argopy 1.4.x still imports, causing an
> `ImportError`. The pin in `requirements.txt` prevents this.

### 3. Configure API keys (optional)

LLM-powered query parsing is optional. The app is **fully functional** without
any API key using the built-in rule-based parser.

```bash
# For Gemini (recommended):
export GOOGLE_API_KEY="your-key-here"      # macOS/Linux
$env:GOOGLE_API_KEY="your-key-here"        # Windows PowerShell

# For OpenAI (alternative):
export OPENAI_API_KEY="your-key-here"
$env:OPENAI_API_KEY="your-key-here"
```

**Never hardcode API keys in the source code.**

---

## Running the app

```bash
cd FloatAi
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Offline / development mode

Toggle **"Use synthetic data"** in the sidebar, or launch with synthetic data
pre-selected by setting `use_synthetic=True` in a custom script:

```python
from core.query_engine import QueryEngine
engine = QueryEngine(use_synthetic=True)
result = engine.answer("What was the temperature in the Gulf of Mexico in July 2023?")
print(result.to_answer_text())
```

---

## Running the benchmark

```bash
# Offline (no network required):
python -m eval.benchmark --synthetic

# Real Argo data (requires internet):
python -m eval.benchmark

# Save results to a custom path:
python -m eval.benchmark --synthetic --output my_results.csv
```

The benchmark reports:
- **Naive baseline accuracy** — using all readings, ignoring QC flags
- **QC-aware accuracy** — Feature 1 active
- **Delta** — improvement in percentage points
- Full per-question CSV saved to `eval/benchmark_results.csv`

---

## Data sources & citations

**Argo float data**
> Argo (2000). Argo float data and metadata from Global Data Assembly Centre (Argo GDAC). SEANOE. https://doi.org/10.17882/42182

**Sea surface temperature (SST)**
> Huang, B., Liu, C., Banzon, V., Freeman, E., Graham, G., Hankins, B., Smith, T., & Zhang, H.-M. (2020). Improvements of the Daily Optimum Interpolation Sea Surface Temperature (DOISST) Version 2.1. *Journal of Climate*, 34(8), 2923–2939. https://doi.org/10.25921/RE9P-PT57

**Marine heatwave detection**
> Hobday, A. J., Alexander, L. V., Perkins, S. E., Smale, D. A., Straub, S. C., Oliver, E. C. J., ... Wernberg, T. (2016). A hierarchical approach to defining marine heatwaves. *Progress in Oceanography*, 141, 227–238. https://doi.org/10.1016/j.pocean.2015.12.014

---

## Architecture notes

### Swappable data layer

The data-loading layer (`argo_loader`, `sst_loader`) is fully decoupled from all
downstream logic. Both real and synthetic loaders return DataFrames with an
identical column schema:

| Column | Type | Description |
|--------|------|-------------|
| `PLATFORM_NUMBER` | str | Float WMO number |
| `CYCLE_NUMBER` | int | Dive cycle |
| `LATITUDE` | float | Decimal degrees |
| `LONGITUDE` | float | Decimal degrees |
| `TIME` | datetime64 | UTC timestamp |
| `PRES` | float | Pressure / depth (dbar) |
| `TEMP` | float | In-situ temperature (°C) |
| `PSAL` | float | Practical salinity (PSU) |
| `TEMP_QC` | int | Argo QC flag (1=good … 4=bad, 9=missing) |
| `PSAL_QC` | int | Argo QC flag |
| `DATA_MODE` | str | 'R' real-time, 'D' delayed-mode |

Switching between real and synthetic data requires only changing `use_synthetic`
on the `QueryEngine` — zero changes to QC, heatwave, or NLU logic.

### QC flag semantics

| Flag | Meaning | FloatAi treatment |
|------|---------|-------------------|
| 1 | Good | Used |
| 2 | Probably good | Used |
| 3 | Probably bad | Excluded + explained |
| 4 | Bad | Excluded + explained |
| 9 | Missing | Excluded silently |

### Depth awareness

FloatAi always reports the depth range used for any statistic. Compound
questions ("was this unusual?") automatically restrict the Argo query to
surface readings (0–10 m) to match the SST-based heatwave baseline.
Mixing full-depth-profile averages with surface SST would give misleading
comparisons.

---

## Known issues / pitfalls

1. **argopy + erddapy conflict** — pin `erddapy<3` (already in requirements.txt).
2. **Windows SSL errors** — install `pip-system-certs`.
3. **Large-sample benchmarking** — averaging hundreds of readings washes out the
   QC effect. The benchmark deliberately uses narrow single-profile queries.
4. **Synthetic data density** — profiles are spaced ~10 days apart (not random)
   so narrow date-range queries always return results.
5. **OISST coverage lag** — the final OISST product has a ~2-week lag; the NRT
   product is used automatically as a fallback for recent dates.
