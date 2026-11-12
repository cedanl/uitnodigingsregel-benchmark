"""Basic smoke tests for the benchmark package."""

import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import student_signal
from conftest import make_result
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from uitnodigingsregel_benchmark.cli import main
from uitnodigingsregel_benchmark.dataset import (
    DataValidationError,
    ensure_id_column,
    validate_student_data,
)
from uitnodigingsregel_benchmark.export import write_result
from uitnodigingsregel_benchmark.features import extract_feature_importance
from uitnodigingsregel_benchmark.impact import resolve_variables
from uitnodigingsregel_benchmark.runner import run_benchmark_on_data, select_models
from uitnodigingsregel_benchmark.settings import DEFAULTS, SettingsError, load_config
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


def test_write_result(tmp_path, valid_result):
    path = write_result(valid_result, tmp_path)
    assert path.name == "benchmark_test_2023.json"
    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["recommendation"]["model"] == "Lasso"
    assert loaded["recommendation"]["lift_over_baseline"] == 2.5
    assert loaded["metadata"]["instelling_id"] == "test"


def test_models_without_xgboost(monkeypatch, capsys):
    from uitnodigingsregel_benchmark import models

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
    result = make_result(dropout_year="2023/2024")
    assert write_result(result, tmp_path).name == "benchmark_test_2023-2024.json"


def test_load_config_fills_defaults(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("instelling_id: abc\nn_iter: 3\nmodels: null\n", encoding="utf-8")
    config = load_config(config_file)
    assert config["instelling_id"] == "abc"
    assert config["n_iter"] == 3
    assert config["models"] is None
    assert config["dropout_column"] == DEFAULTS["dropout_column"]
    assert load_config(tmp_path / "ontbreekt.yaml") == DEFAULTS


def test_load_config_rejects_unknown_setting(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("n_iterr: 3\n", encoding="utf-8")
    with pytest.raises(SettingsError, match="n_iterr"):
        load_config(config_file)


def test_select_models_rejects_unknown_model():
    assert list(select_models(["Lasso"])) == ["Lasso"]
    with pytest.raises(SettingsError, match="Beschikbare modellen"):
        select_models(["Lasso", "Bestaat niet"])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.drop(columns="Dropout"), "ontbreekt de kolom 'Dropout'"),
        (lambda d: d.assign(Dropout=2), "alleen 0 en 1"),
        (lambda d: d.assign(Dropout=0), "0 uitvallers"),
        (lambda d: d.assign(Dropout=[1] * 3 + [0] * 37), "3 uitvallers"),
        (lambda d: d.assign(Leeftijd="onbekend"), "geen kolommen met getallen"),
        (lambda d: d.iloc[0:0], "geen rijen"),
    ],
)
def test_validate_student_data_names_the_problem(change, message):
    data = change(_students([2021] * 40).drop(columns="Cohort"))
    with pytest.raises(DataValidationError, match=message):
        validate_student_data(data)


def test_validate_student_data_accepts_usable_data():
    validate_student_data(_students([2021] * 40))


def test_ensure_id_column_adds_row_numbers():
    data = _students([2021] * 4)
    assert ensure_id_column(data)[1] == "Studentnummer"
    without_id, id_column = ensure_id_column(data.drop(columns="Studentnummer"))
    assert id_column in without_id.columns
    assert list(without_id[id_column]) == [0, 1, 2, 3]


def test_cli_quick_run_on_demo_data(tmp_path, monkeypatch):
    if not DEMO_TRAIN.exists():
        pytest.skip("Demo data niet aanwezig")
    monkeypatch.chdir(DEMO_TRAIN.parents[3])
    config_file = tmp_path / "config.yaml"
    config_file.write_text("n_jobs: 1\n", encoding="utf-8")
    args = ["--demo", "--quick", "--config", str(config_file), "--results-dir", str(tmp_path)]
    assert main(args) == 0
    with open(tmp_path / "benchmark_demo_onbekend.json", encoding="utf-8") as f:
        result = json.load(f)
    assert result["metadata"]["quick_mode"] is True
    assert result["metadata"]["instelling_id"] == "demo"
    assert result["metadata"]["split"]["type"] == "random"
    assert set(result["models"]) == {"Random Forest", "Lasso", "Logistic Regression"}
    # Own column names stay out of the result file, but are in the local report
    features = result["metadata"]["features_available"]
    assert "ROCMondriaan" not in features
    assert "ROCMondriaan" not in json.dumps(result)
    assert "eigen_variabele_1" in features
    report = (tmp_path / "benchmark_demo_onbekend.html").read_text(encoding="utf-8")
    assert "ROCMondriaan" in report


def test_cli_reports_missing_file_without_traceback(tmp_path, capsys):
    exit_code = main(["--train", str(tmp_path / "geen.csv"), "--results-dir", str(tmp_path)])
    assert exit_code == 1
    assert "bestaat niet" in capsys.readouterr().err


def test_resolve_variables_joins_groups():
    columns = ["Leeftijd", "Niveau_HAVO", "Niveau_MBO", "Techniek", "Zorg"]
    groups = {"Vooropleiding": ["Niveau_*"], "Richting": ["Techniek", "Zorg"]}
    assert resolve_variables(columns, groups) == {
        "Vooropleiding": ["Niveau_HAVO", "Niveau_MBO"],
        "Richting": ["Techniek", "Zorg"],
        "Leeftijd": ["Leeftijd"],
    }
    assert resolve_variables(columns, groups, ["Leeftijd", "Vooropleiding"]) == {
        "Leeftijd": ["Leeftijd"],
        "Vooropleiding": ["Niveau_HAVO", "Niveau_MBO"],
    }
    assert list(resolve_variables(columns)) == columns


@pytest.mark.parametrize(
    ("groups", "variables", "message"),
    [
        ({"Vooropleiding": ["Bestaat_niet_*"]}, None, "past op geen enkele kolom"),
        ({"A": ["Niveau_*"], "B": ["Niveau_MBO"]}, None, "staat in twee groepen"),
        (None, ["Bestaat niet"], "Onbekende variabele"),
    ],
)
def test_resolve_variables_rejects_wrong_settings(groups, variables, message):
    with pytest.raises(SettingsError, match=message):
        resolve_variables(["Leeftijd", "Niveau_HAVO", "Niveau_MBO"], groups, variables)


def test_cli_measures_impact_per_variable(tmp_path, monkeypatch):
    if not DEMO_TRAIN.exists():
        pytest.skip("Demo data niet aanwezig")
    monkeypatch.chdir(DEMO_TRAIN.parents[3])
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "n_jobs: 1\n"
        "impact_variables: [StudentAge, Vooropleiding]\n"
        "variable_groups:\n  Vooropleiding: ['VooroplNiveau_*']\n",
        encoding="utf-8",
    )
    args = ["--demo", "--quick", "--impact", "--config", str(config_file)]
    assert main([*args, "--results-dir", str(tmp_path)]) == 0
    with open(tmp_path / "benchmark_demo_onbekend.json", encoding="utf-8") as f:
        result = json.load(f)
    assert result["metadata"]["impact_measured"] is True
    for model in result["models"].values():
        impact = {e["variable"]: e for e in model["variable_impact"]}
        # The own group of shared columns gets the shared group name
        assert set(impact) == {"StudentAge", "Vooropleiding_niveau"}
        assert len(impact["Vooropleiding_niveau"]["columns"]) > 1
        entry = impact["StudentAge"]
        assert entry["precision_diff"] == pytest.approx(
            model["evaluation"]["precision"] - entry["precision_without"]
        )


def test_cli_without_impact_leaves_it_out(tmp_path, monkeypatch):
    if not DEMO_TRAIN.exists():
        pytest.skip("Demo data niet aanwezig")
    monkeypatch.chdir(DEMO_TRAIN.parents[3])
    config_file = tmp_path / "config.yaml"
    config_file.write_text("n_jobs: 1\nmodels: [Lasso]\n", encoding="utf-8")
    args = ["--demo", "--config", str(config_file), "--results-dir", str(tmp_path)]
    assert main(args) == 0
    with open(tmp_path / "benchmark_demo_onbekend.json", encoding="utf-8") as f:
        result = json.load(f)
    assert result["metadata"]["impact_measured"] is False
    assert "variable_impact" not in result["models"]["Lasso"]


def test_repeated_splits_give_mean_and_spread(tmp_path, monkeypatch):
    if not DEMO_TRAIN.exists():
        pytest.skip("Demo data niet aanwezig")
    monkeypatch.chdir(DEMO_TRAIN.parents[3])
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "n_jobs: 1\nmodels: [Lasso, Logistic Regression]\nn_iter: 3\ncv_folds: 3\nrepeats: 3\n"
        "impact_variables: [StudentAge, Vooropleiding_niveau]\n",
        encoding="utf-8",
    )
    args = ["--demo", "--impact", "--config", str(config_file), "--results-dir", str(tmp_path)]
    assert main(args) == 0
    result = json.loads((tmp_path / "benchmark_demo_onbekend.json").read_text(encoding="utf-8"))
    assert result["metadata"]["split"]["repeats"] == 3
    for model in result["models"].values():
        evaluation = model["evaluation"]
        assert evaluation["precision_sd"] >= 0 and evaluation["recall_sd"] >= 0
        point_20 = next(p for p in model["precision_recall_curve"] if p["pct"] == 20)
        assert point_20["precision"] == pytest.approx(evaluation["precision"], abs=0.1)
        impact = {e["variable"]: e for e in model["variable_impact"]}
        assert set(impact) == {"StudentAge", "Vooropleiding_niveau"}
        assert all("precision_diff_sd" in e for e in impact.values())


def test_cohort_split_is_not_repeated():
    data = _students([2021] * 30 + [2022] * 30 + [2023] * 30)
    config = {**load_config("geen.yaml"), "cohort_column": "Cohort", "repeats": 4, "n_jobs": 1}
    config.update({"models": ["Lasso"], "n_iter": 2, "cv_folds": 2})
    result, _ = run_benchmark_on_data(data, config)
    assert result["metadata"]["split"]["repeats"] == 1
    assert result["models"]["Lasso"]["evaluation"]["precision_sd"] == 0


def test_references_for_random_and_one_variable():
    data = _students([2021] * 100)
    data["Verzuim"] = data["Dropout"] * 5  # absence that predicts dropout perfectly
    config = {**load_config("geen.yaml"), "n_jobs": 1, "models": ["Lasso"], "n_iter": 2}
    config.update({"cv_folds": 2, "repeats": 2, "reference_variable": "Verzuim"})
    config["column_mapping"] = {"Verzuim": "absence_unauthorized"}
    result, local_names = run_benchmark_on_data(data, config)

    references = result["references"]
    assert references["random"]["precision"] == pytest.approx(50.0)
    assert references["random"]["recall"] == 20.0
    single = references["single_variable"]
    assert single["variable"] == "absence_unauthorized"
    assert single["precision"] == 100.0
    assert single["recall"] == 40.0
    evaluation = result["models"]["Lasso"]["evaluation"]
    assert evaluation["precision_vs_single_variable"] == pytest.approx(
        evaluation["precision"] - 100.0, abs=0.1
    )
    assert evaluation["precision_vs_random"] == pytest.approx(
        evaluation["precision"] - 50.0, abs=0.1
    )


def test_reference_variable_must_be_a_numeric_feature():
    data = _students([2021] * 40)
    data["Opleiding"] = "Zorg"
    config = {**load_config("geen.yaml"), "n_jobs": 1, "models": ["Lasso"], "n_iter": 2}
    config.update({"cv_folds": 2, "repeats": 1, "reference_variable": "Opleiding"})
    with pytest.raises(SettingsError, match="getallen"):
        run_benchmark_on_data(data, config)
    config["reference_variable"] = "Dropout"
    with pytest.raises(SettingsError, match="geen kolom met kenmerken"):
        run_benchmark_on_data(data, config)
