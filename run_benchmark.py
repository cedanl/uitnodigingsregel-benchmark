"""Benchmark entrypoint: prepare data, train models, write standardized JSON results."""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import student_signal
import yaml
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from student_signal.evaluate import get_stoplight_evaluation, prepare_model_predictions

from models import MODELS
from uitnodigingsregel_benchmark.export import (
    BENCHMARK_PCTS,
    build_result,
    extract_precision_recall_curve,
    write_result,
)
from uitnodigingsregel_benchmark.features import extract_feature_importance

DROPOUT_COL = "Dropout"
ID_COL = "Studentnummer"
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ITER = 20
RANDOM_STATE = 42


def _load_config(config_path: Path = Path("config.yaml")) -> dict:
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
) -> tuple[Any, dict]:
    """Fit estimator with RandomizedSearchCV. Returns (best_estimator, best_params)."""
    scoring = "roc_auc" if hasattr(estimator, "predict_proba") else "neg_mean_squared_error"
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator,
        param_grid,
        n_iter=N_ITER,
        cv=cv,
        scoring=scoring,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def main() -> None:
    parser = argparse.ArgumentParser(description="Run uitnodigingsregel benchmark")
    parser.add_argument("--train", required=True, help="Path to training CSV (with known outcomes)")
    parser.add_argument("--results-dir", default="results", help="Output directory")
    args = parser.parse_args()

    config = _load_config()

    print(f"Laden van data: {args.train}")
    train_df = pd.read_csv(args.train, sep=None, engine="python")

    print(f"Train/test split ({int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)})...")
    train_split, test_split = train_test_split(
        train_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_df[DROPOUT_COL],
    )

    print("Data voorbereiden...")
    prepared = student_signal.prepare(
        train_split.reset_index(drop=True),
        test_split.reset_index(drop=True),
        target_col=DROPOUT_COL,
        id_col=ID_COL,
    )

    metadata = {
        "instelling_id": config.get("instelling_id", "anoniem"),
        "dropout_year": config.get("dropout_year"),
        "training_years": config.get("training_years"),
        "n_train": len(prepared.X_train),
        "n_test": len(prepared.X_pred),
        "dropout_rate_train": round(float(prepared.y_train.mean()), 3),
        "dropout_rate_test": round(float(prepared.pred_df[DROPOUT_COL].mean()), 3),
        "features_available": list(prepared.X_train.columns),
        "run_date": datetime.now().strftime("%Y-%m-%d"),
        "cv_folds": CV_FOLDS,
        "n_iter_search": N_ITER,
    }

    x_val = {
        False: prepared.pred_df.drop(DROPOUT_COL, axis=1),
        True: prepared.pred_df_scaled.drop(DROPOUT_COL, axis=1),
    }
    y_val = {
        False: prepared.pred_df[DROPOUT_COL],
        True: prepared.pred_df_scaled[DROPOUT_COL],
    }

    dropout_rate_pct = metadata["dropout_rate_train"] * 100

    model_results = {}
    for name, model_cfg in MODELS.items():
        estimator = model_cfg["estimator"]
        use_scaled = model_cfg["use_scaled"]
        param_grid = model_cfg.get("param_grid", {})

        print(f"Trainen: {name} (gridsearch {N_ITER} iteraties)...")
        X_train = prepared.X_train_scaled if use_scaled else prepared.X_train

        if param_grid:
            best_estimator, best_params = _fit_with_gridsearch(
                estimator, param_grid, X_train, prepared.y_train
            )
        else:
            estimator.fit(X_train, prepared.y_train)
            best_estimator, best_params = estimator, {}

        val_data = prepared.pred_df_scaled if use_scaled else prepared.pred_df
        short = name.lower().replace(" ", "_")
        eval_df = prepare_model_predictions(val_data, best_estimator, short, DROPOUT_COL)
        curve = extract_precision_recall_curve(eval_df, short, BENCHMARK_PCTS)

        prec20, rec20 = _precision_at(curve, 20)
        _, status, _ = get_stoplight_evaluation(prec20, rec20)

        lift = round(prec20 / dropout_rate_pct, 2) if dropout_rate_pct > 0 else None

        metrics = _compute_metrics(
            best_estimator, X_train, prepared.y_train, x_val[use_scaled], y_val[use_scaled]
        )

        print(f"  Feature importance berekenen voor {name}...")
        importance = extract_feature_importance(best_estimator, prepared, use_scaled, DROPOUT_COL)

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
    path = write_result(result, args.results_dir)
    print(f"\nBenchmark resultaat opgeslagen: {path}")


if __name__ == "__main__":
    main()
