"""
Automated Data Profiling Card
-------------------------------
Computes a compact "data health" profile for the uploaded dataset: per-column
type breakdown, missingness, cardinality, skewness, and outlier counts for
numeric columns -- all pure pandas/numpy arithmetic, no plotting library
involved at all (deliberately, after two prior features hung due to
matplotlib/kaleido backend issues in the user's environment). Renders via
Streamlit's native st.metric/st.dataframe only.

Mirrors audit_agent.py's two-layer pattern:
  1. Deterministic layer: compute_profile() does all the math, fully
     testable in isolation, returns a plain dict/DataFrame.
  2. LLM layer: get_profile_narrative() takes only the compact computed
     profile (never raw data) and asks an LLM for a one-paragraph summary.
     Falls back to a rule-based summary on any failure.
"""

import os
import json

import numpy as np
import pandas as pd
from google import genai
from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Layer 1: Deterministic profiling
# ---------------------------------------------------------------------------

def compute_profile(df: pd.DataFrame) -> dict:
    """Computes a compact data health profile. Pure pandas/numpy, no
    plotting, no model fitting -- just descriptive statistics. Returns a
    dict with overall counts plus a per-column DataFrame."""

    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    total_missing = int(df.isnull().sum().sum())
    total_cells = n_rows * n_cols if n_cols else 1
    overall_missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells else 0.0

    duplicate_rows = int(df.duplicated().sum())

    rows = []
    for col in df.columns:
        series = df[col]
        missing_count = int(series.isnull().sum())
        missing_pct = round((missing_count / n_rows) * 100, 1) if n_rows else 0.0
        nunique = int(series.nunique(dropna=True))
        cardinality_pct = round((nunique / n_rows) * 100, 1) if n_rows else 0.0

        col_type = "numeric" if col in numeric_cols else "categorical"

        skew_val = None
        outlier_count = None
        outlier_pct = None
        if col_type == "numeric":
            clean = series.dropna()
            if len(clean) >= 3 and clean.std() > 0:
                skew_val = round(float(clean.skew()), 2)

                q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = clean[(clean < lower_bound) | (clean > upper_bound)]
                    outlier_count = int(len(outliers))
                    outlier_pct = round((outlier_count / len(clean)) * 100, 1)
                else:
                    outlier_count = 0
                    outlier_pct = 0.0
            else:
                skew_val = 0.0
                outlier_count = 0
                outlier_pct = 0.0

        rows.append({
            "Column": col,
            "Type": col_type,
            "Missing %": missing_pct,
            "Unique Values": nunique,
            "Cardinality %": cardinality_pct,
            "Skewness": skew_val,
            "Outlier %": outlier_pct,
        })

    profile_table = pd.DataFrame(rows)

    if profile_table.empty:
        return {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "numeric_col_count": 0,
            "categorical_col_count": 0,
            "overall_missing_pct": 0.0,
            "duplicate_rows": duplicate_rows,
            "profile_table": profile_table,
            "flagged_high_missing": [],
            "flagged_high_cardinality": [],
            "flagged_skewed": [],
            "flagged_outliers": [],
        }

    flagged_high_missing = profile_table[profile_table["Missing %"] >= 30]["Column"].tolist()
    flagged_high_cardinality = profile_table[
        (profile_table["Type"] == "categorical") & (profile_table["Cardinality %"] >= 90)
    ]["Column"].tolist()

    skew_numeric = pd.to_numeric(profile_table["Skewness"], errors="coerce")
    flagged_skewed = profile_table[skew_numeric.abs() >= 1.0]["Column"].tolist()

    outlier_numeric = pd.to_numeric(profile_table["Outlier %"], errors="coerce")
    flagged_outliers = profile_table[outlier_numeric >= 5]["Column"].tolist()

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "numeric_col_count": len(numeric_cols),
        "categorical_col_count": len(categorical_cols),
        "overall_missing_pct": overall_missing_pct,
        "duplicate_rows": duplicate_rows,
        "profile_table": profile_table,
        "flagged_high_missing": flagged_high_missing,
        "flagged_high_cardinality": flagged_high_cardinality,
        "flagged_skewed": flagged_skewed,
        "flagged_outliers": flagged_outliers,
    }


# ---------------------------------------------------------------------------
# Layer 2: LLM narrative
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a data analyst writing a single short paragraph \
summarizing a dataset's overall health, based ONLY on pre-computed summary \
statistics you are given (never raw data). Be specific about which columns \
have which issues, in plain English suitable for someone not deeply technical.

Respond ONLY with valid JSON, no markdown fences, no preamble, in this shape:
{
  "headline": "one short phrase capturing overall data health, e.g. 'Mostly clean with a few skewed columns'",
  "summary": "2-4 sentence plain-English paragraph"
}
"""


def get_profile_narrative(profile: dict, api_key: str = None) -> dict:
    """Sends the compact profile dict (minus the full per-column DataFrame,
    which is summarized down to flagged column lists) to an LLM via
    OpenRouter for a narrative summary. Falls back to a rule-based summary
    on any failure (missing key, network error, bad response)."""

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        return _fallback_narrative(profile, error="No GEMINI_API_KEY found.")

    compact_input = {
        "n_rows": profile["n_rows"],
        "n_cols": profile["n_cols"],
        "numeric_col_count": profile["numeric_col_count"],
        "categorical_col_count": profile["categorical_col_count"],
        "overall_missing_pct": profile["overall_missing_pct"],
        "duplicate_rows": profile["duplicate_rows"],
        "flagged_high_missing": profile["flagged_high_missing"],
        "flagged_high_cardinality": profile["flagged_high_cardinality"],
        "flagged_skewed": profile["flagged_skewed"],
        "flagged_outliers": profile["flagged_outliers"],
    }

    try:
        client = genai.Client(api_key=key)

        prompt = f"""
    {_SYSTEM_PROMPT}

    Profile Statistics:
    {json.dumps(compact_input, indent=2)}
    """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        text = response.text.strip()

        if text.startswith("```"):
            text = (
                text.replace("```json", "")
                .replace("```", "")
                .strip()
            )

        return json.loads(text)

    except Exception as e:
        return _fallback_narrative(profile, error=str(e))


def _fallback_narrative(profile: dict, error: str = "") -> dict:
    issues = []
    if profile["flagged_high_missing"]:
        issues.append(f"{len(profile['flagged_high_missing'])} column(s) with heavy missing data")
    if profile["flagged_high_cardinality"]:
        issues.append(f"{len(profile['flagged_high_cardinality'])} column(s) that look like identifiers")
    if profile["flagged_skewed"]:
        issues.append(f"{len(profile['flagged_skewed'])} skewed numeric column(s)")
    if profile["flagged_outliers"]:
        issues.append(f"{len(profile['flagged_outliers'])} column(s) with notable outliers")
    if profile["duplicate_rows"] > 0:
        issues.append(f"{profile['duplicate_rows']} duplicate row(s)")

    if not issues:
        headline = "Clean dataset"
        summary = (
            f"AI reasoning was unavailable, so a rule-based summary was generated."
            f"This dataset has {profile['n_rows']} rows and {profile['n_cols']} columns "
            f"with no significant data quality issues detected."
        )
    else:
        headline = "A few things worth reviewing"
        summary = (
            f"AI reasoning was unavailable, so a rule-based summary was generated."
            f"This dataset has {profile['n_rows']} rows and {profile['n_cols']} columns. "
            f"Notable findings: {', '.join(issues)}."
        )

    if error:
        summary += f" (Reasoning step error: {error})"

    return {"headline": headline, "summary": summary}