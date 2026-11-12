"""Run the benchmark: prepare data, train models, write standardized JSON results."""

from collections.abc import Callable
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import student_signal
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from student_signal.evaluate import get_stoplight_evaluation, prepare_model_predictions

from uitnodigingsregel_benchmark.check import FORMAT_VERSION, check_result
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
from uitnodigingsregel_benchmark.impact import measure_variable_impact, resolve_variables
from uitnodigingsregel_benchmark.models import MODELS
from uitnodigingsregel_benchmark.report import build_html_report
from uitnodigingsregel_benchmark.settings import SettingsError
from uitnodigingsregel_benchmark.shared_variables import (
    apply_shared_names,
    is_own_name,
    load_mapping,
    match_columns,
    merge_variable_groups,
    public_names,
)
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


def _prepare(train: pd.DataFrame, test: pd.DataFrame, dropout_col: str, id_col: str):
    return student_signal.prepare(train, test, target_col=dropout_col, id_col=id_col)


def _curve(model: Any, prepared: Any, use_scaled: bool, dropout_col: str, name: str) -> list[dict]:
    """Precision and recall of a fitted model at the standard invitation percentages."""
    val_data = prepared.pred_df_scaled if use_scaled else prepared.pred_df
    short = name.lower().replace(" ", "_")
    eval_df = prepare_model_predictions(val_data, model, short, dropout_col)
    return extract_precision_recall_curve(eval_df, short, BENCHMARK_PCTS)


def _mean_sd(values: list[float]) -> tuple[float, float]:
    """Mean and standard deviation over the splits; the deviation is 0 for one split."""
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return round(float(np.mean(values)), 1), round(sd, 1)


def _mean_curve(curves: list[list[dict]]) -> list[dict]:
    return [
        {
            "pct": points[0]["pct"],
            "precision": round(float(np.mean([p["precision"] for p in points])), 1),
            "recall": round(float(np.mean([p["recall"] for p in points])), 1),
        }
        for points in zip(*curves)
    ]


def _combine_impact(per_split: list[list[dict]]) -> list[dict]:
    """Combine the impact of each variable over the splits into a mean and a spread.

    Args:
        per_split: For each split the output of measure_variable_impact() for one model.

    Returns:
        One entry per variable, largest mean precision difference first. A variable for
        which retraining failed on any split keeps only the error.
    """
    by_variable: dict[str, list[dict]] = {}
    for entries in per_split:
        for entry in entries:
            by_variable.setdefault(entry["variable"], []).append(entry)

    combined = []
    for variable, entries in by_variable.items():
        entry = {"variable": variable, "columns": entries[0]["columns"]}
        failed = next((e for e in entries if "error" in e), None)
        if failed:
            entry["error"] = failed["error"]
        else:
            for field in ("precision_without", "recall_without"):
                entry[field] = _mean_sd([e[field] for e in entries])[0]
            for field in ("precision_diff", "recall_diff"):
                entry[field], entry[f"{field}_sd"] = _mean_sd([e[field] for e in entries])
        combined.append(entry)
    combined.sort(key=lambda e: e.get("precision_diff", float("-inf")), reverse=True)
    return combined


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
    print(f"Laden van data: {train_path}")
    if config["column_mapping"] is None:
        config = {**config, "column_mapping": load_mapping()}
    result, local_names = run_benchmark_on_data(read_student_data(train_path), config, quick_mode)
    own = [name for public, name in local_names.items() if is_own_name(public)]
    if own:
        print(
            f"\n{len(own)} kolommen of groepen staan niet in de gedeelde variabelenlijst en "
            f"komen onder een neutrale naam in het resultaatbestand: {', '.join(own)}."
        )
    for warning in check_result(result):
        print(f"\nLet op: {warning}")
    path = write_result(result, results_dir)
    report_path = path.with_suffix(".html")
    report_path.write_text(build_html_report(result, local_names), encoding="utf-8")
    print(f"\nBenchmark resultaat opgeslagen: {path}")
    print(f"Rapport opgeslagen: {report_path}")
    return path


def run_benchmark_on_data(
    data: pd.DataFrame,
    config: dict,
    quick_mode: bool = False,
    on_progress: Callable[[float, str], None] | None = None,
) -> tuple[dict, dict[str, str]]:
    """Run the benchmark on student data and return the result.

    Args:
        data: Student data with known outcomes.
        config: Settings from load_config().
        quick_mode: Whether the settings are those of quick mode; stated in the result.
        on_progress: Called with the share of the work done (0 to 1) and a description.

    Returns:
        The complete result, ready to write with write_result(), and the names in it
        that replace a name of the institution, mapped to that own name. The result
        holds shared variable names only; the second value stays with the institution.

    Raises:
        DataValidationError: If the data cannot be used.
        SettingsError: If the settings name an unknown model or variable.
    """
    dropout_col = config["dropout_column"]
    models = select_models(config["models"])

    validate_student_data(data, dropout_col, config["id_column"], config["cv_folds"])
    data, id_col = ensure_id_column(data, config["id_column"])

    test_size = config["test_size"]
    train_split, test_split, split_info = split_train_test(
        data, dropout_col, config["cohort_column"], test_size, RANDOM_STATE
    )
    splits = [(train_split, test_split)]
    if split_info["type"] == SPLIT_COHORT:
        dropout_year = split_info["test_cohort"]
        training_years = split_info["train_cohorts"]
        print(f"Splitsing op cohort: trainen op {training_years}, toetsen op {dropout_year}.")
    else:
        dropout_year = config["dropout_year"]
        training_years = config["training_years"]
        # A cohort split has one possible test cohort; a random split can be repeated
        for repeat in range(1, config["repeats"]):
            train, test, _ = split_train_test(
                data,
                dropout_col,
                config["cohort_column"],
                test_size,
                RANDOM_STATE + repeat,
                verbose=False,
            )
            splits.append((train, test))
        print(
            f"Willekeurige splitsing ({round((1 - test_size) * 100)}/{round(test_size * 100)}), "
            f"{len(splits)} keer herhaald."
        )
    split_info["repeats"] = len(splits)

    feature_columns = [c for c in train_split.columns if c not in (dropout_col, id_col)]
    linked = match_columns(feature_columns, config["column_mapping"])
    variables = None
    if config["impact"]:
        variables = resolve_variables(
            feature_columns,
            merge_variable_groups(linked, config["variable_groups"]),
            config["impact_variables"],
        )

    n_steps = 1 + len(models) + (len(splits) - 1) + len(variables or []) * len(splits)
    steps_done = 0

    def report(message: str) -> None:
        nonlocal steps_done
        if on_progress:
            on_progress(steps_done / n_steps, message)
        steps_done += 1

    report("Data voorbereiden")
    print("Data voorbereiden...")
    prepared = _prepare(train_split, test_split, dropout_col, id_col)

    metadata = {
        "instelling_id": config["instelling_id"],
        "dropout_year": dropout_year,
        "training_years": training_years,
        "split": split_info,
        "quick_mode": quick_mode,
        "impact_measured": bool(config["impact"]),
        "n_train": len(prepared.X_train),
        "n_test": len(prepared.X_pred),
        "dropout_rate_train": round(float(prepared.y_train.mean()), 3),
        "dropout_rate_test": round(float(prepared.pred_df[dropout_col].mean()), 3),
        "features_available": list(prepared.X_train.columns),
        "run_date": datetime.now().strftime("%Y-%m-%d"),
        "student_signal_version": version("student_signal"),
        "benchmark_version": version("uitnodigingsregel-benchmark"),
        "format_version": FORMAT_VERSION,
        "cv_folds": config["cv_folds"],
        "n_iter_search": config["n_iter"],
    }

    model_results = {}
    # Per split: model name mapped to (fitted estimator, uses scaled data)
    fitted: list[dict[str, tuple[Any, bool]]] = [{}]
    curves: dict[str, list[list[dict]]] = {}
    for name, model_cfg in models.items():
        estimator = model_cfg["estimator"]
        use_scaled = model_cfg["use_scaled"]
        param_grid = model_cfg.get("param_grid", {})

        report(f"Model trainen: {name}")
        print(f"Trainen: {name} (gridsearch {config['n_iter']} iteraties)...")
        X_train = prepared.X_train_scaled if use_scaled else prepared.X_train

        if param_grid:
            best_estimator, best_params = _fit_with_gridsearch(
                estimator, param_grid, X_train, prepared.y_train, config
            )
        else:
            estimator.fit(X_train, prepared.y_train)
            best_estimator, best_params = estimator, {}

        curves[name] = [_curve(best_estimator, prepared, use_scaled, dropout_col, name)]
        X_val = (prepared.pred_df_scaled if use_scaled else prepared.pred_df).drop(
            dropout_col, axis=1
        )
        metrics = _compute_metrics(
            best_estimator, X_train, prepared.y_train, X_val, prepared.pred_df[dropout_col]
        )

        print(f"  Feature importance berekenen voor {name}...")
        importance = extract_feature_importance(best_estimator, prepared, use_scaled, dropout_col)
        model_results[name] = {
            "best_params": best_params,
            "feature_importance": importance,
            **metrics,
        }
        fitted[0][name] = (best_estimator, use_scaled)

    # The other splits reuse the settings found in the search on the first split
    for number, (train, test) in enumerate(splits[1:], start=2):
        report(f"Splitsing {number} van {len(splits)}: modellen opnieuw trainen")
        print(f"Splitsing {number} van {len(splits)}: modellen opnieuw trainen...")
        prepared_split = _prepare(train, test, dropout_col, id_col)
        fitted.append({})
        for name, (estimator, use_scaled) in fitted[0].items():
            X_train = prepared_split.X_train_scaled if use_scaled else prepared_split.X_train
            model = clone(estimator).fit(X_train, prepared_split.y_train)
            curves[name].append(_curve(model, prepared_split, use_scaled, dropout_col, name))
            fitted[-1][name] = (model, use_scaled)

    dropout_rate_pct = metadata["dropout_rate_train"] * 100
    for name, model_curves in curves.items():
        per_split = [_precision_at(curve, 20) for curve in model_curves]
        precision, precision_sd = _mean_sd([p for p, _ in per_split])
        recall, recall_sd = _mean_sd([r for _, r in per_split])
        _, status, _ = get_stoplight_evaluation(precision, recall)
        model_results[name].update(
            {
                "precision_recall_curve": _mean_curve(model_curves),
                "evaluation": {
                    "precision": precision,
                    "recall": recall,
                    "precision_sd": precision_sd,
                    "recall_sd": recall_sd,
                    "lift_over_baseline": (
                        round(precision / dropout_rate_pct, 2) if dropout_rate_pct > 0 else None
                    ),
                    "status": status,
                },
            }
        )
        spread = f" (± {precision_sd:.1f})" if len(splits) > 1 else ""
        print(f"  {name}: precisie@20% {precision:.1f}%{spread} | recall@20% {recall:.1f}%")

    if variables:
        print(
            f"Impact meten: elk model opnieuw trainen zonder één van {len(variables)} variabelen"
            f", op {len(splits)} splitsingen."
        )
        impact_per_split = []
        for number, ((train, test), models_split) in enumerate(zip(splits, fitted), start=1):
            baseline = {name: _precision_at(curves[name][number - 1], 20) for name in models_split}
            prefix = f"Splitsing {number} van {len(splits)}: " if len(splits) > 1 else ""
            impact_per_split.append(
                measure_variable_impact(
                    train,
                    test,
                    variables,
                    models_split,
                    baseline,
                    dropout_col,
                    id_col,
                    lambda message, prefix=prefix: report(prefix + message),
                )
            )
        for name in model_results:
            model_results[name]["variable_impact"] = _combine_impact(
                [split_impact[name] for split_impact in impact_per_split]
            )

    if on_progress:
        on_progress(1.0, "Klaar")
    result = build_result(metadata, model_results)
    local_names = apply_shared_names(result, public_names(linked))
    return result, local_names
