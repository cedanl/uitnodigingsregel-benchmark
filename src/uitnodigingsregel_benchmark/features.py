"""Full feature importance extraction for any sklearn-compatible model."""

from typing import Any

import numpy as np
import pandas as pd
from student_signal.pipeline import PreparedData


def extract_feature_importance(
    model: Any,
    prepared: PreparedData,
    use_scaled: bool = False,
    dropout_column: str = "Dropout",
) -> list[dict]:
    """Extract full feature importance ranking for any sklearn-compatible model.

    Auto-detects model type:
    - feature_importances_ (tree-based) → Gini importance, all features
    - coef_ (linear) → absolute coefficient, all features
    - other (SVM, etc.) → model-agnostic perturbation importance

    Args:
        model: Fitted sklearn-compatible estimator.
        prepared: PreparedData from student_signal.prepare().
        use_scaled: Whether model was trained on scaled data.
        dropout_column: Name of the target column.

    Returns:
        List of dicts with 'feature' and 'importance' keys, sorted descending.
        Linear models also include a 'coefficient' key with the signed value.
    """
    X = prepared.X_train_scaled if use_scaled else prepared.X_train

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X.columns)
        sorted_imp = importances.sort_values(ascending=False)
        return [{"feature": f, "importance": round(float(v), 6)} for f, v in sorted_imp.items()]

    if hasattr(model, "coef_"):
        raw = model.coef_.ravel() if model.coef_.ndim > 1 else model.coef_
        coefs = pd.Series(raw, index=X.columns)
        sorted_coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)
        return [
            {
                "feature": f,
                "importance": round(float(abs(v)), 6),
                "coefficient": round(float(v), 6),
            }
            for f, v in sorted_coefs.items()
        ]

    return _perturbation_importance(model, X)


def _perturbation_importance(model: Any, X: pd.DataFrame) -> list[dict]:
    """Model-agnostic perturbation importance: perturb each feature by 1 std."""
    if hasattr(model, "predict_proba"):

        def pred_fn(x: pd.DataFrame) -> np.ndarray:
            return model.predict_proba(x)[:, 1]
    else:
        pred_fn = model.predict

    baseline = pred_fn(X)
    feature_stds = X.std()
    importances = []

    X_work = X.copy()
    for col in X.columns:
        original = X_work[col].copy()
        X_work[col] = X_work[col] + feature_stds[col]
        perturbed = pred_fn(X_work)
        X_work[col] = original
        importance = float(np.abs(perturbed - baseline).mean())
        importances.append({"feature": col, "importance": round(importance, 6)})

    importances.sort(key=lambda x: x["importance"], reverse=True)
    return importances
