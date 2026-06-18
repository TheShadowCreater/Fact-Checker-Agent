"""
report_generator.py
--------------------
Builds a downloadable CSV report and executive summary stats from results.
"""

import io
import pandas as pd


VERDICT_ORDER = ["Verified", "Inaccurate", "False"]


def build_dataframe(results: list[dict]) -> pd.DataFrame:
    """
    Convert list of result dicts to a clean pandas DataFrame.

    Each result dict should have:
        id, claim, category, source_snippet,
        verdict, confidence, reasoning, sources
    """
    rows = []
    for r in results:
        sources = r.get("sources", [])
        sources_str = " | ".join(sources) if sources else "No sources"
        rows.append(
            {
                "ID": r.get("id", ""),
                "Claim": r.get("claim", ""),
                "Category": r.get("category", ""),
                "Verdict": r.get("verdict", ""),
                "Confidence": round(float(r.get("confidence", 0)), 2),
                "Reasoning": r.get("reasoning", ""),
                "Sources": sources_str,
                "Source Snippet": r.get("source_snippet", ""),
            }
        )
    return pd.DataFrame(rows)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Return DataFrame as UTF-8 CSV bytes for Streamlit download."""
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    return buf.getvalue().encode("utf-8")


def compute_summary(results: list[dict]) -> dict:
    """
    Compute executive-summary statistics.

    Returns:
        dict with keys: total, verified, inaccurate, false,
                        avg_confidence, accuracy_rate
    """
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "verified": 0,
            "inaccurate": 0,
            "false": 0,
            "avg_confidence": 0.0,
            "accuracy_rate": 0.0,
        }

    verdicts = [r.get("verdict", "Inaccurate") for r in results]
    confidences = [float(r.get("confidence", 0.5)) for r in results]

    verified = verdicts.count("Verified")
    inaccurate = verdicts.count("Inaccurate")
    false = verdicts.count("False")

    return {
        "total": total,
        "verified": verified,
        "inaccurate": inaccurate,
        "false": false,
        "avg_confidence": round(sum(confidences) / total, 2),
        "accuracy_rate": round(verified / total * 100, 1),
    }
