"""
Data Quality & Trust Auditor Agent
-----------------------------------
Runs before model training. Performs a battery of statistical probes on the
raw dataset (leakage, imbalance, cardinality, missingness, constants,
duplicates), then hands the structured findings to an LLM which reasons
over them holistically and returns a prioritized, plain-English verdict.

This is intentionally split into two layers:
  1. `run_probes()`      -> deterministic, fast, no LLM. Pure pandas/numpy.
  2. `get_agent_verdict()` -> sends the probe findings (not raw data) to
                              Claude, which reasons about severity, ordering,
                              and what to actually do about each finding.

Keeping probes deterministic means the LLM never has to (and never does)
eyeball raw rows -- it only reasons over compact, pre-computed evidence.
This keeps cost low and avoids ever sending the user's raw data off-app
unnecessarily beyond small samples needed for context.
"""

import json
import os
import numpy as np
import pandas as pd
from google import genai

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Layer 1: Deterministic statistical probes
# ---------------------------------------------------------------------------

def _check_missingness(df: pd.DataFrame) -> list:
    findings = []
    miss = df.isnull().mean().sort_values(ascending=False)
    for col, pct in miss.items():
        if pct >= 0.5:
            findings.append({
                "type": "high_missingness",
                "column": col,
                "severity": "high",
                "detail": f"{pct*100:.1f}% of values are missing."
            })
        elif pct >= 0.15:
            findings.append({
                "type": "moderate_missingness",
                "column": col,
                "severity": "medium",
                "detail": f"{pct*100:.1f}% of values are missing."
            })
    return findings


def _check_constant_columns(df: pd.DataFrame) -> list:
    findings = []
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if nunique <= 1:
            findings.append({
                "type": "constant_column",
                "column": col,
                "severity": "medium",
                "detail": "Column has a single unique value (or is empty) and carries no signal."
            })
    return findings


def _check_high_cardinality(df: pd.DataFrame, target_column: str) -> list:
    findings = []
    n = len(df)
    for col in df.columns:
        if col == target_column:
            continue
        if df[col].dtype == "object" or str(df[col].dtype).startswith("category"):
            nunique = df[col].nunique(dropna=True)
            ratio = nunique / n if n else 0
            if ratio > 0.9 and nunique > 20:
                findings.append({
                    "type": "high_cardinality",
                    "column": col,
                    "severity": "medium",
                    "detail": f"{nunique} unique values across {n} rows ({ratio*100:.0f}% unique) — "
                              f"looks like an identifier (e.g. ID, name) rather than a real category."
                })
    return findings


def _check_duplicates(df: pd.DataFrame) -> list:
    findings = []
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        pct = dup_count / len(df) * 100
        findings.append({
            "type": "duplicate_rows",
            "column": None,
            "severity": "medium" if pct < 10 else "high",
            "detail": f"{dup_count} duplicate rows found ({pct:.1f}% of dataset)."
        })
    return findings


def _check_target_leakage(df: pd.DataFrame, target_column: str, is_classification: bool) -> list:
    """Flags columns suspiciously predictive of the target -- a common cause
    of leaderboard scores that look great but won't hold up in production."""
    findings = []
    if target_column not in df.columns:
        return findings

    y = df[target_column]
    num_cols = df.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c != target_column]

    if not is_classification and pd.api.types.is_numeric_dtype(y):
        # Numeric target: check Pearson correlation magnitude.
        for col in num_cols:
            try:
                corr = df[[col, target_column]].dropna().corr().iloc[0, 1]
            except Exception:
                continue
            if pd.notna(corr) and abs(corr) > 0.97:
                findings.append({
                    "type": "potential_leakage",
                    "column": col,
                    "severity": "high",
                    "detail": f"Correlation with target is {corr:.3f} — near-perfect. "
                              f"This column may be derived from or duplicate the target."
                })
    else:
        # Classification target: check if any single column near-perfectly
        # separates classes (e.g. via group-wise variance collapse) or is
        # literally a near-duplicate / encoded copy of the target.
        try:
            y_codes = pd.factorize(y)[0]
        except Exception:
            y_codes = None

        for col in num_cols:
            try:
                series = df[col]
                if series.nunique(dropna=True) <= 1:
                    continue
                corr = np.corrcoef(series.fillna(series.median()), y_codes)[0, 1]
            except Exception:
                continue
            if pd.notna(corr) and abs(corr) > 0.95:
                findings.append({
                    "type": "potential_leakage",
                    "column": col,
                    "severity": "high",
                    "detail": f"Strongly correlates with the target classes (|corr|≈{abs(corr):.2f}). "
                              f"Verify this isn't an encoded version of the target or a post-outcome field."
                })

        # Exact duplicate column check (covers categorical leakage too)
        for col in df.columns:
            if col == target_column:
                continue
            try:
                if df[col].astype(str).equals(df[target_column].astype(str)):
                    findings.append({
                        "type": "potential_leakage",
                        "column": col,
                        "severity": "high",
                        "detail": "This column is identical to the target column."
                    })
            except Exception:
                continue

    return findings


def _check_class_imbalance(df: pd.DataFrame, target_column: str, is_classification: bool) -> list:
    findings = []
    if not is_classification or target_column not in df.columns:
        return findings

    counts = df[target_column].value_counts(normalize=True)
    if len(counts) < 2:
        return findings

    majority = counts.iloc[0]
    minority = counts.iloc[-1]
    if majority >= 0.95:
        severity = "high"
    elif majority >= 0.80:
        severity = "medium"
    else:
        return findings

    findings.append({
        "type": "class_imbalance",
        "column": target_column,
        "severity": severity,
        "detail": f"Majority class makes up {majority*100:.1f}% of rows; minority class only "
                  f"{minority*100:.1f}%. Accuracy alone will be a misleading metric here."
    })
    return findings


def _check_sample_size(df: pd.DataFrame) -> list:
    findings = []
    n_rows, n_cols = df.shape
    if n_rows < 100:
        findings.append({
            "type": "small_sample",
            "column": None,
            "severity": "high",
            "detail": f"Only {n_rows} rows available. Model evaluation will be noisy and unreliable."
        })
    elif n_rows < 30 * n_cols:
        findings.append({
            "type": "low_rows_to_features_ratio",
            "column": None,
            "severity": "medium",
            "detail": f"{n_rows} rows vs {n_cols} columns — a low row-to-feature ratio raises "
                      f"overfitting risk, especially for tree ensembles."
        })
    return findings


def run_probes(df: pd.DataFrame, target_column: str, is_classification: bool) -> list:
    """Runs the full battery of deterministic checks and returns a flat list
    of finding dicts: {type, column, severity, detail}."""
    findings = []
    findings += _check_missingness(df)
    findings += _check_constant_columns(df)
    findings += _check_high_cardinality(df, target_column)
    findings += _check_duplicates(df)
    findings += _check_target_leakage(df, target_column, is_classification)
    findings += _check_class_imbalance(df, target_column, is_classification)
    findings += _check_sample_size(df)
    return findings


# ---------------------------------------------------------------------------
# Layer 2: LLM reasoning over the findings
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a meticulous data quality auditor reviewing findings \
from an automated statistical scan of a dataset, prior to machine learning model \
training. You will be given a JSON list of findings (each with a type, column, \
severity, and detail) plus basic dataset metadata.

Your job:
1. Decide an overall verdict: "safe_to_train", "train_with_caution", or "fix_before_training".
2. Write a short (2-3 sentence) plain-English summary of the dataset's overall health.
3. Re-rank the findings by actual real-world risk to model validity (not just the \
   severity label they arrived with) and write one short, concrete, plain-English \
   recommendation per finding -- what the user should actually do about it.
4. If there are no findings at all, say so plainly and give a short confidence note.

Respond ONLY with valid JSON in this exact shape, no markdown fences, no preamble:
{
  "verdict": "safe_to_train" | "train_with_caution" | "fix_before_training",
  "summary": "string",
  "ranked_findings": [
    {"column": "string or null", "issue": "string", "recommendation": "string", "risk": "high" | "medium" | "low"}
  ]
}
"""


def get_agent_verdict(findings: list, dataset_meta: dict, api_key: str = None) -> dict:
    """Sends probe findings to an LLM (via OpenRouter) for reasoning. Returns
    a dict matching the schema in _SYSTEM_PROMPT. Falls back to a
    deterministic verdict if the API call fails for any reason (e.g. no key,
    no network), so the feature degrades gracefully rather than breaking
    the app.

    api_key: pass explicitly, or set OPENROUTER_API_KEY in your environment
    / .env file and leave this as None. Get a key at openrouter.ai/keys.
    """

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        return _fallback_verdict(
            findings,
            error="No GEMINI_API_KEY found. Set it in a .env file or pass it explicitly."
        )

    # OpenRouter uses an OpenAI-compatible chat completions schema, not
    # Anthropic's native Messages API shape. Model id is prefixed with the
    # provider name. "anthropic/claude-sonnet-4.6" routes to Claude Sonnet
    # 4.6 through OpenRouter; swap for a ":free" suffixed model id to run
    # this at zero cost (e.g. a free Llama/Qwen model) if preferred.
    client = genai.Client(api_key=key)

    prompt = f"""
    System Instructions:
    {_SYSTEM_PROMPT}

    Dataset Information:
    {json.dumps({
        "dataset_meta": dataset_meta,
        "findings": findings
    }, indent=2)}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text = response.text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)


def _fallback_verdict(findings: list, error: str = "") -> dict:
    """Deterministic backup if the LLM call fails -- never leaves the user
    with a broken tab."""
    high = [f for f in findings if f.get("severity") == "high"]
    if high:
        verdict = "fix_before_training"
    elif findings:
        verdict = "train_with_caution"
    else:
        verdict = "safe_to_train"

    ranked = [
        {
            "column": f.get("column"),
            "issue": f.get("type", "").replace("_", " "),
            "recommendation": f.get("detail", ""),
            "risk": f.get("severity", "medium")
        }
        for f in findings
    ]

    summary = (
        "AI reasoning was unavailable, so a rule-based audit summary was generated. "
        f"{len(findings)} issue(s) detected."
        if findings
        else
        "AI reasoning was unavailable, so a rule-based audit summary was generated. "
        "No issues were detected by the statistical probes."
    )

    if error:
        summary += f" (Reasoning step error: {error})"

    return {
        "verdict": verdict,
        "summary": summary,
        "ranked_findings": ranked
    }
def _fallback_verdict(findings: list, error: str = "") -> dict:
    high = [f for f in findings if f.get("severity") == "high"]

    if high:
        verdict = "fix_before_training"
    elif findings:
        verdict = "train_with_caution"
    else:
        verdict = "safe_to_train"

    ranked = [
        {
            "column": f.get("column"),
            "issue": f.get("type", "").replace("_", " "),
            "recommendation": f.get("detail", ""),
            "risk": f.get("severity", "medium")
        }
        for f in findings
    ]

    summary = (
        "AI reasoning was unavailable, so a rule-based audit summary was generated. "
        f"{len(findings)} issue(s) detected."
        if findings
        else
        "AI reasoning was unavailable, so a rule-based audit summary was generated. "
        "No issues were detected by the statistical probes."
    )

    if error:
        summary += f" (Reasoning step error: {error})"

    return {
        "verdict": verdict,
        "summary": summary,
        "ranked_findings": ranked
    }
