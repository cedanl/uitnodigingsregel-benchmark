"""Shared test helpers."""

import pytest

from uitnodigingsregel_benchmark import sharing
from uitnodigingsregel_benchmark.check import FORMAT_VERSION
from uitnodigingsregel_benchmark.export import build_result


def make_result(instelling: str = "test", precision_diff: float = 2.0, **metadata) -> dict:
    """A small result that passes the check."""
    model = {
        "best_params": {"alpha": 0.1},
        "precision_recall_curve": [{"pct": 20, "precision": 50.0, "recall": 40.0}],
        "feature_importance": [{"feature": "StudentAge", "importance": 0.5}],
        "evaluation": {
            "precision": 50.0,
            "recall": 40.0,
            "lift_over_baseline": 2.5,
            "status": "Betrouwbaar",
        },
        "variable_impact": [
            {"variable": "StudentAge", "columns": ["StudentAge"], "precision_diff": precision_diff},
            {"variable": "eigen_variabele_1", "columns": ["eigen_variabele_1"], "error": "mislukt"},
        ],
        "r2_train": 0.1,
        "r2_val": 0.1,
        "mse_train": 0.1,
        "mse_val": None,
    }
    meta = {
        "format_version": FORMAT_VERSION,
        "benchmark_version": "0.1.0",
        "student_signal_version": "0.2.0",
        "instelling_id": instelling,
        "dropout_year": 2023,
        "split": {"type": "random", "test_size": 0.2},
        "quick_mode": False,
        "impact_measured": True,
        "n_train": 800,
        "n_test": 200,
        "dropout_rate_train": 0.2,
        "dropout_rate_test": 0.2,
        "features_available": ["StudentAge", "eigen_variabele_1"],
        "n_own_variables": 1,
        "run_date": "2026-04-24",
        **metadata,
    }
    return build_result(meta, {"Lasso": model})


@pytest.fixture
def valid_result() -> dict:
    return make_result()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Tests never reach the internet."""
    monkeypatch.setattr(sharing, "fetch_shared_data", lambda: None)
