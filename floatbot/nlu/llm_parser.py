"""
nlu/llm_parser.py
-----------------
Feature 3 (LLM branch): Gemini / OpenAI-based natural language parser.

Converts a plain-English question into a structured ParsedQuery dict.
Falls through to rule_parser automatically on any API error.

Environment variables (never hardcoded)
---------------------------------------
GOOGLE_API_KEY   — enables Gemini (google-generativeai)
OPENAI_API_KEY   — enables OpenAI (openai>=1.0), tried if no Google key

The caller should always wrap this in a try/except and fall back to
rule_parser.parse() if this module raises or returns None.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ParsedQuery schema (shared with rule_parser)
# ---------------------------------------------------------------------------
# {
#   "intent"     : "profile_query" | "heatwave_query" | "compound_query",
#   "parameter"  : "temperature" | "salinity" | None,
#   "region"     : {"lat_min", "lat_max", "lon_min", "lon_max"} | None,
#   "start_date" : "YYYY-MM-DD" | None,
#   "end_date"   : "YYYY-MM-DD" | None,
#   "depth_min"  : float | None,
#   "depth_max"  : float | None,
#   "raw_question": str,
#   "parser_used" : "llm_gemini" | "llm_openai" | "rule_based",
# }

_SYSTEM_PROMPT = """You are a query parser for an ocean science assistant.
Convert the user's plain-English question into a JSON object with exactly these keys:
  intent       : "profile_query" | "heatwave_query" | "compound_query"
  parameter    : "temperature" | "salinity" | null
  region       : {"lat_min": float, "lat_max": float, "lon_min": float, "lon_max": float} | null
  start_date   : "YYYY-MM-DD" | null
  end_date     : "YYYY-MM-DD" | null
  depth_min    : float | null
  depth_max    : float | null

Rules:
- Use "compound_query" when the question asks BOTH for a measurement AND asks if it
  was unusual / anomalous / a heatwave.
- Use "heatwave_query" for questions primarily about unusual conditions, anomalies,
  or marine heatwaves.
- Use "profile_query" for questions about temperature or salinity values.
- If a month/year is mentioned (e.g. "July 2023"), set start_date to the first day
  and end_date to the last day of that month.
- If "last month", "this month", "this year", etc. is used, output them literally
  as the special tokens: "last_month", "this_month", "this_year" — the caller will
  resolve them against the dataset's latest date.
- If no depth is mentioned, leave depth_min and depth_max null.
- Return ONLY the JSON object, no markdown, no explanation.
"""


def is_available() -> bool:
    """Return True if at least one LLM API key is configured."""
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def active_provider() -> Optional[str]:
    """Return 'gemini', 'openai', or None."""
    if os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


def parse(question: str, dataset_latest_date: Optional[str] = None) -> Optional[dict]:
    """
    Parse a plain-English question using an LLM.

    Parameters
    ----------
    question            : str   The user's raw question.
    dataset_latest_date : str   ISO date of the most recent data point, used to
                                resolve relative date phrases like 'last month'.

    Returns
    -------
    dict matching the ParsedQuery schema, or None on failure.
    """
    provider = active_provider()
    if provider is None:
        logger.debug("No LLM API key found; LLM parser unavailable.")
        return None

    try:
        if provider == "gemini":
            raw = _call_gemini(question)
        else:
            raw = _call_openai(question)
    except Exception as exc:
        logger.warning("LLM API call failed (%s): %s", provider, exc)
        return None

    parsed = _extract_json(raw)
    if parsed is None:
        logger.warning("LLM returned non-JSON: %s", raw[:200])
        return None

    parsed["raw_question"] = question
    parsed["parser_used"] = f"llm_{provider}"

    # Resolve relative date tokens
    if dataset_latest_date:
        parsed = _resolve_relative_dates(parsed, dataset_latest_date)

    return parsed


# ---------------------------------------------------------------------------
# Provider-specific callers
# ---------------------------------------------------------------------------

def _call_gemini(question: str) -> str:
    import google.generativeai as genai  # type: ignore

    api_key = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=_SYSTEM_PROMPT,
    )
    response = model.generate_content(question)
    return response.text


def _call_openai(question: str) -> str:
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=400,
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Optional[dict]:
    """Extract and parse the first JSON object found in a string."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Find first { … }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _resolve_relative_dates(parsed: dict, dataset_latest_date: str) -> dict:
    """
    Replace special relative-date tokens with real YYYY-MM-DD strings,
    anchored to the dataset's latest date (not wall-clock today).
    """
    anchor = pd.Timestamp(dataset_latest_date)

    token_map = {
        "this_month": (
            anchor.replace(day=1).strftime("%Y-%m-%d"),
            anchor.strftime("%Y-%m-%d"),
        ),
        "last_month": (
            (anchor - pd.offsets.MonthBegin(1)).replace(day=1).strftime("%Y-%m-%d"),
            (anchor.replace(day=1) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        ),
        "this_year": (
            anchor.replace(month=1, day=1).strftime("%Y-%m-%d"),
            anchor.strftime("%Y-%m-%d"),
        ),
    }

    for field in ("start_date", "end_date"):
        val = parsed.get(field)
        if isinstance(val, str) and val in token_map:
            start, end = token_map[val]
            parsed[field] = start if field == "start_date" else end

    return parsed


# Import pandas here to avoid circular issues (used only in _resolve_relative_dates)
try:
    import pandas as pd
except ImportError:
    pass
