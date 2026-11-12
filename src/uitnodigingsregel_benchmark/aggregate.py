"""Combine the shared results of institutions into tables for the overview and the app."""

import json
from pathlib import Path
from statistics import mean, median

from uitnodigingsregel_benchmark.shared_variables import is_own_name

DEMO_INSTELLING_ID = "demo"


def is_shareable(result: dict) -> bool:
    """Whether a result counts in the collective overview: a full run on own data."""
    metadata = result["metadata"]
    return not metadata.get("quick_mode") and metadata.get("instelling_id") != DEMO_INSTELLING_ID


def load_shared_results(results_dir: str | Path = "results") -> list[dict]:
    """Read the result files in the results folder that count in the overview.

    Args:
        results_dir: Folder with benchmark_*.json files; subfolders are not read.

    Returns:
        The results of full runs on own data, ordered by file name.
    """
    results = []
    for path in sorted(Path(results_dir).glob("benchmark_*.json")):
        with open(path, encoding="utf-8") as f:
            result = json.load(f)
        if is_shareable(result):
            results.append(result)
    return results


def summarize(results: list[dict]) -> dict:
    """Count the institutions, results and students in the shared results."""
    students: dict[str, int] = {}
    for result in results:
        metadata = result["metadata"]
        n = metadata["n_train"] + metadata["n_test"]
        # An institution with several results counts once, with its largest data set
        students[metadata["instelling_id"]] = max(students.get(metadata["instelling_id"], 0), n)
    return {
        "n_institutions": len(students),
        "n_results": len(results),
        "n_students": sum(students.values()),
    }


def model_table(results: list[dict]) -> list[dict]:
    """List precision and recall at 20% per institution and model.

    Returns:
        One row per result and model, with the kind of split, best precision first.
    """
    rows = []
    for result in results:
        metadata = result["metadata"]
        for name, model in result["models"].items():
            rows.append(
                {
                    "instelling": metadata["instelling_id"],
                    "year": metadata.get("dropout_year"),
                    "split": metadata.get("split", {}).get("type", "random"),
                    "model": name,
                    "precision": model["evaluation"]["precision"],
                    "recall": model["evaluation"]["recall"],
                    "dropout_rate": round(metadata["dropout_rate_test"] * 100, 1),
                }
            )
    return sorted(rows, key=lambda r: (r["split"], r["model"], -r["precision"]))


def variable_impact_table(results: list[dict]) -> list[dict]:
    """Combine the impact per variable over institutions.

    Per institution the impact of a variable is the mean precision difference over its
    models and results. The table gives the mean, lowest and highest of those values.

    Returns:
        One row per variable, largest mean impact first.
    """
    per_institution: dict[str, dict[str, list[float]]] = {}
    for result in results:
        instelling = result["metadata"]["instelling_id"]
        for model in result["models"].values():
            for entry in model.get("variable_impact", []):
                # An own variable of one institution is not the same variable elsewhere
                if "precision_diff" in entry and not is_own_name(entry["variable"]):
                    per_institution.setdefault(entry["variable"], {}).setdefault(
                        instelling, []
                    ).append(entry["precision_diff"])

    rows = []
    for variable, institutions in per_institution.items():
        values = [mean(diffs) for diffs in institutions.values()]
        rows.append(
            {
                "variable": variable,
                "n_institutions": len(values),
                # + 0.0 turns a rounded -0.0 into 0.0
                "mean": round(mean(values), 1) + 0.0,
                "median": round(median(values), 1) + 0.0,
                "lowest": round(min(values), 1) + 0.0,
                "highest": round(max(values), 1) + 0.0,
            }
        )
    return sorted(rows, key=lambda r: (-r["mean"], r["variable"]))


def build_shared_data(results: list[dict]) -> dict:
    """Assemble everything the overview page and the app need from the shared results."""
    return {
        "summary": summarize(results),
        "models": model_table(results),
        "variables": variable_impact_table(results),
    }
