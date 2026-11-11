"""Run the benchmark: prepare data, train models, write standardized JSON results."""

from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import student_signal
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from student_signal.evaluate import get_stoplight_evaluation, prepare_model_predictions

from uitnodigingsregel_benchmark.dataset import (
    ensure_id_column,
    read_student_data,
    validate_student_data,
)
from uitnodigingsregel_benchmark.export import (
    BENCHMARK_PCTS,
    build_result,
    extract_precision_recall_curve,
    write_result,
)
from uitnodigingsregel_benchmark.features import extract_feature_importance
from uitnodigingsregel_benchmark.models import MODELS
from uitnodigingsregel_benchmark.settings import SettingsError
from uitnodigingsregel_benchmark.split import SPLIT_COHORT, split_train_test

RANDOM_STATE = 42


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def _compute_metrics(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    if hasattr(model, "predict_proba"):
        y_train_pred = model.predict_proba(X_train)[:, 1]
        y_val_pred = model.predict_proba(X_val)[:, 1]
    else:
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

    return {
        "r2_train": round(_r2(y_train.values, y_train_pred), 3),
        "r2_val": round(_r2(y_val.values, y_val_pred), 3),
        "mse_train": round(_mse(y_train.values, y_train_pred), 3),
        "mse_val": round(_mse(y_val.values, y_val_pred), 3),
    }


def _precision_at(curve: list[dict], pct: int) -> tuple[float, float]:
    match = next((p for p in curve if p["pct"] == pct), None)
    return (match["precision"], match["recall"]) if match else (0.0, 0.0)


def _fit_with_gridsearch(
    estimator: Any,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> tuple[Any, dict]:
    """Fit estimator with RandomizedSearchCV. Returns (best_estimator, best_params)."""
    scoring = "roc_auc" if hasattr(estimator, "predict_proba") else "neg_mean_squared_error"
    cv = StratifiedKFold(n_splits=config["cv_folds"], shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator,
        param_grid,
        n_iter=config["n_iter"],
        cv=cv,
        scoring=scoring,
        random_state=RANDOM_STATE,
        n_jobs=config["n_jobs"],
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def select_models(names: list[str] | None) -> dict[str, dict[str, Any]]:
    """Pick the models to run from the registry.

    Args:
        names: Model names from the settings, or None for all available models.

    Returns:
        The chosen entries from the model registry.

    Raises:
        SettingsError: If a name is not in the registry.
    """
    if names is None:
        return MODELS
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        raise SettingsError(
            f"Onbekend model in de instellingen: {', '.join(unknown)}. "
            f"Beschikbare modellen: {', '.join(MODELS)}."
        )
    return {n: MODELS[n] for n in names}


def run_benchmark(
    train_path: str | Path,
    config: dict,
    results_dir: str | Path = "results",
    quick_mode: bool = False,
) -> Path:
    """Run the benchmark on one data file and write the result.

    Args:
        train_path: Path to the CSV with known outcomes.
        config: Settings from load_config().
        results_dir: Directory to write the result file.
        quick_mode: Whether the settings are those of quick mode; stated in the result.

    Returns:
        Path to the written result file.

    Raises:
        DataValidationError: If the data cannot be used.
        SettingsError: If the settings name an unknown model.
    """
    dropout_col = config["dropout_column"]
    models = select_models(config["models"])

    print(f"Laden van data: {train_path}")
    data = read_student_data(train_path)
    validate_student_data(data, dropout_col, config["id_column"], config["cv_folds"])
    data, id_col = ensure_id_column(data, config["id_column"])

    test_size = config["test_size"]
    train_split, test_split, split_info = split_train_test(
        data, dropout_col, config["cohort_column"], test_size, RANDOM_STATE
    )
    if split_info["type"] == SPLIT_COHORT:
        dropout_year = split_info["test_cohort"]
        training_years = split_info["train_cohorts"]
        print(f"Splitsing op cohort: trainen op {training_years}, toetsen op {dropout_year}.")
    else:
        dropout_year = config["dropout_year"]
        training_years = config["training_years"]
        print(f"Willekeurige splitsing ({round((1 - test_size) * 100)}/{round(test_size * 100)}).")

    print("Data voorbereiden...")
    prepared = student_signal.prepare(
        train_split,
        test_split,
        target_col=dropout_col,
        id_col=id_col,
    )

    metadata = {
        "instelling_id": config["instelling_id"],
        "dropout_year": dropout_year,
        "training_years": training_years,
        "split": split_info,
        "quick_mode": quick_mode,
        "n_train": len(prepared.X_train),
        "n_test": len(prepared.X_pred),
        "dropout_rate_train": round(float(prepared.y_train.mean()), 3),
        "dropout_rate_test": round(float(prepared.pred_df[dropout_col].mean()), 3),
        "features_available": list(prepared.X_train.columns),
        "run_date": datetime.now().strftime("%Y-%m-%d"),
        "student_signal_version": version("student_signal"),
        "cv_folds": config["cv_folds"],
        "n_iter_search": config["n_iter"],
    }

    x_val = {
        False: prepared.pred_df.drop(dropout_col, axis=1),
        True: prepared.pred_df_scaled.drop(dropout_col, axis=1),
    }
    y_val = {
        False: prepared.pred_df[dropout_col],
        True: prepared.pred_df_scaled[dropout_col],
    }

    dropout_rate_pct = metadata["dropout_rate_train"] * 100

    model_results = {}
    for name, model_cfg in models.items():
        estimator = model_cfg["estimator"]
        use_scaled = model_cfg["use_scaled"]
        param_grid = model_cfg.get("param_grid", {})

        print(f"Trainen: {name} (gridsearch {config['n_iter']} iteraties)...")
        X_train = prepared.X_train_scaled if use_scaled else prepared.X_train

        if param_grid:
            best_estimator, best_params = _fit_with_gridsearch(
                estimator, param_grid, X_train, prepared.y_train, config
            )
        else:
            estimator.fit(X_train, prepared.y_train)
            best_estimator, best_params = estimator, {}

        val_data = prepared.pred_df_scaled if use_scaled else prepared.pred_df
        short = name.lower().replace(" ", "_")
        eval_df = prepare_model_predictions(val_data, best_estimator, short, dropout_col)
        curve = extract_precision_recall_curve(eval_df, short, BENCHMARK_PCTS)

        prec20, rec20 = _precision_at(curve, 20)
        _, status, _ = get_stoplight_evaluation(prec20, rec20)

        lift = round(prec20 / dropout_rate_pct, 2) if dropout_rate_pct > 0 else None

        metrics = _compute_metrics(
            best_estimator, X_train, prepared.y_train, x_val[use_scaled], y_val[use_scaled]
        )

        print(f"  Feature importance berekenen voor {name}...")
        importance = extract_feature_importance(best_estimator, prepared, use_scaled, dropout_col)

        model_results[name] = {
            "best_params": best_params,
            "precision_recall_curve": curve,
            "feature_importance": importance,
            "evaluation": {
                "precision": prec20,
                "recall": rec20,
                "lift_over_baseline": lift,
                "status": status,
            },
            **metrics,
        }
        print(f"  Precisie@20%: {prec20:.1f}% | Recall@20%: {rec20:.1f}% | {status}")

    result = build_result(metadata, model_results)
    path = write_result(result, results_dir)
    print(f"\nBenchmark resultaat opgeslagen: {path}")
    return path
