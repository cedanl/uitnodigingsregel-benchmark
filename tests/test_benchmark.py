"""Basic smoke tests for the benchmark package."""

import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import student_signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from uitnodigingsregel_benchmark.export import build_result, write_result
from uitnodigingsregel_benchmark.features import extract_feature_importance
from uitnodigingsregel_benchmark.split import split_train_test

DEMO_TRAIN = Path(__file__).parent.parent / "data/01-raw/demo/synth_data_train.csv"
DROPOUT_COL = "Dropout"
ID_COL = "Studentnummer"


@pytest.fixture(scope="module")
def prepared():
    if not DEMO_TRAIN.exists():
        pytest.skip("Demo data niet aanwezig")
    train_df = pd.read_csv(DEMO_TRAIN, sep="\t")
    train_split, test_split = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df[DROPOUT_COL]
    )
    return student_signal.prepare(
        train_split.reset_index(drop=True),
        test_split.reset_index(drop=True),
        target_col=DROPOUT_COL,
        id_col=ID_COL,
    )


def test_id_column_is_not_a_feature(prepared):
    assert ID_COL not in prepared.X_train.columns
    assert ID_COL not in prepared.pred_df.columns
    assert len(prepared.train_ids) == len(prepared.X_train)
    assert len(prepared.pred_ids) == len(prepared.X_pred)


def test_feature_importance_tree(prepared):
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(prepared.X_train, prepared.y_train)
    result = extract_feature_importance(model, prepared, use_scaled=False)
    assert len(result) == len(prepared.X_train.columns)
    assert all("feature" in r and "importance" in r for r in result)
    importances = [r["importance"] for r in result]
    assert importances == sorted(importances, reverse=True)


def test_feature_importance_linear(prepared):
    model = Lasso(alpha=0.01, max_iter=10_000)
    model.fit(prepared.X_train_scaled, prepared.y_train)
    result = extract_feature_importance(model, prepared, use_scaled=True)
    assert len(result) == len(prepared.X_train.columns)
    assert all("coefficient" in r for r in result)


def test_write_result(tmp_path):
    metadata = {
        "instelling_id": "test",
        "n_train": 100,
        "dropout_rate_train": 0.2,
        "run_date": "2026-04-24",
    }
    model_results = {
        "Test Model": {
            "best_params": {},
            "precision_recall_curve": [{"pct": 20, "precision": 50.0, "recall": 50.0}],
            "feature_importance": [{"feature": "x", "importance": 0.5}],
            "evaluation": {
                "precision": 50.0,
                "recall": 50.0,
                "lift_over_baseline": 2.5,
                "status": "Betrouwbaar",
            },
            "r2_train": 0.1,
            "r2_val": 0.1,
            "mse_train": 0.1,
            "mse_val": 0.1,
        }
    }
    result = build_result(metadata, model_results)
    path = write_result(result, tmp_path)
    assert path.exists()
    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["recommendation"]["model"] == "Test Model"
    assert loaded["recommendation"]["lift_over_baseline"] == 2.5
    assert loaded["metadata"]["instelling_id"] == "test"


def test_models_without_xgboost(monkeypatch, capsys):
    import models

    monkeypatch.setitem(sys.modules, "xgboost", None)
    try:
        reloaded = importlib.reload(models)
        assert "XGBoost" not in reloaded.MODELS
        assert "Random Forest" in reloaded.MODELS
        assert "brew install libomp" in capsys.readouterr().out
    finally:
        monkeypatch.undo()
        importlib.reload(models)


def _students(cohorts: list) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Studentnummer": range(len(cohorts)),
            "Cohort": cohorts,
            "Leeftijd": [16 + i % 5 for i in range(len(cohorts))],
            "Dropout": [i % 2 for i in range(len(cohorts))],
        }
    )


def test_split_on_cohort_tests_on_last_cohort():
    data = _students([2021] * 10 + [2023] * 6 + [2022] * 10)
    train, test, info = split_train_test(data, "Dropout", "Cohort")
    assert info == {"type": "cohort", "train_cohorts": [2021, 2022], "test_cohort": 2023}
    assert len(train) == 20
    assert len(test) == 6
    assert "Cohort" not in train.columns
    assert "Cohort" not in test.columns
    json.dumps(info)


def test_split_without_cohort_column_is_random():
    data = _students([2021] * 20).drop(columns="Cohort")
    train, test, info = split_train_test(data, "Dropout")
    assert info["type"] == "random"
    assert len(train) == 16
    assert len(test) == 4


@pytest.mark.parametrize("cohorts", [[2021] * 20, None])
def test_split_falls_back_to_random(cohorts, capsys):
    data = _students(cohorts or [2021] * 20)
    cohort_col = "Cohort" if cohorts else "Startjaar"
    train, test, info = split_train_test(data, "Dropout", cohort_col)
    assert info["type"] == "random"
    assert len(train) + len(test) == 20
    assert "willekeurig gesplitst" in capsys.readouterr().out


def test_split_leaves_out_rows_without_cohort():
    data = _students([2021] * 10 + [2022] * 6 + [None] * 4)
    train, test, info = split_train_test(data, "Dropout", "Cohort")
    assert info["type"] == "cohort"
    assert len(train) + len(test) == 16


def test_write_result_with_cohort_in_file_name(tmp_path):
    result = {"metadata": {"instelling_id": "test", "dropout_year": "2023/2024"}}
    assert write_result(result, tmp_path).name == "benchmark_test_2023-2024.json"
