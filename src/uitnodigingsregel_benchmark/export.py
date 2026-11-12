"""Build and write standardized benchmark JSON results."""

import json
import math
import re
from pathlib import Path

import pandas as pd


def sanitize(obj):
    """Recursively replace float NaN/Inf with None for valid JSON."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


BENCHMARK_PCTS = [5, 10, 15, 20, 25, 30, 40, 50]
# Precision weighted higher: intervention cost per false positive exceeds cost of missed student.
PRECISION_WEIGHT = 0.6
RECALL_WEIGHT = 0.4


def extract_precision_recall_curve(
    eval_df: pd.DataFrame,
    model_short_name: str,
    benchmark_pcts: list[int] = BENCHMARK_PCTS,
) -> list[dict]:
    """Sample precision/recall from a full evaluation curve at standard percentages.

    Args:
        eval_df: DataFrame from student_signal.evaluate.prepare_model_predictions().
        model_short_name: Lowercase model key (e.g. 'random_forest').
        benchmark_pcts: Invitation percentages to sample.

    Returns:
        List of dicts with 'pct', 'precision', 'recall' keys.
    """
    pct_col = "perc_uitgenodigde_studenten"
    prec_col = f"precision{model_short_name}"
    rec_col = f"recall{model_short_name}"

    curve = []
    for pct in benchmark_pcts:
        idx = (eval_df[pct_col] - pct).abs().idxmin()
        row = eval_df.loc[idx]
        curve.append(
            {
                "pct": pct,
                "precision": round(float(row[prec_col]) * 100, 1),
                "recall": round(float(row[rec_col]) * 100, 1),
            }
        )
    return curve


def build_result(metadata: dict, model_results: dict) -> dict:
    """Assemble the full benchmark result dict.

    Args:
        metadata: Instelling metadata (n_train, dropout_rate, features_available, etc.).
        model_results: Dict mapping model names to their results.

    Returns:
        Complete result dict ready for JSON serialisation.
    """
    best_name = max(
        model_results,
        key=lambda n: (
            model_results[n]["evaluation"]["precision"] * PRECISION_WEIGHT
            + model_results[n]["evaluation"]["recall"] * RECALL_WEIGHT
        ),
    )
    best = model_results[best_name]
    return {
        "metadata": metadata,
        "models": model_results,
        "recommendation": {
            "model": best_name,
            "precision_at_20": best["evaluation"]["precision"],
            "recall_at_20": best["evaluation"]["recall"],
            "lift_over_baseline": best["evaluation"].get("lift_over_baseline"),
            "status": best["evaluation"]["status"],
        },
    }


def write_result(result: dict, results_dir: str | Path = "results") -> Path:
    """Write benchmark result to a named JSON file derived from instelling_id and dropout_year.

    Filename format: benchmark_{instelling_id}_{dropout_year}.json
    Re-running for the same instelling/year overwrites the previous file cleanly.

    Args:
        result: Complete benchmark result dict.
        results_dir: Directory to write the file.

    Returns:
        Path to the written file.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    meta = result["metadata"]
    instelling_id = meta.get("instelling_id") or "anoniem"
    dropout_year = meta.get("dropout_year") or "onbekend"
    # A cohort such as "2023/2024" must not end up as a folder in the file name
    name = re.sub(r"[^\w.-]+", "-", f"benchmark_{instelling_id}_{dropout_year}")
    path = results_dir / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitize(result), f, indent=2, ensure_ascii=False)
    return path
