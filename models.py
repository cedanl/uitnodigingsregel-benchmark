"""Algorithm registry for the benchmark.

Each entry defines an estimator, whether it needs scaled input, and a
parameter grid for RandomizedSearchCV. Add, remove, or extend freely —
any sklearn-compatible estimator works.

Models marked 'core' are used for cross-institutional comparison by CEDA.
Additional models can be added for local experimentation.
"""

from typing import Any

from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception as error:  # noqa: BLE001 - xgboost raises its own error when its library fails to load
    XGBClassifier = None
    print(
        "XGBoost kan niet worden geladen en wordt overgeslagen. "
        "De andere modellen draaien wel.\n"
        "Op een Mac ontbreekt meestal OpenMP. Installeer het met: brew install libomp\n"
        f"Melding van XGBoost: {str(error).strip().splitlines()[0]}"
    )

# fmt: off
MODELS: dict[str, dict[str, Any]] = {
    # --- Core models (comparable across instellingen) ---
    # Parameter grids match student-signal for cross-institutional comparability.
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=42),
        "use_scaled": False,
        "param_grid": {
            "bootstrap":         [True, False],
            "max_depth":         [2, 3, 4],
            "max_features":      [3, 4, 5],
            "min_samples_leaf":  [3, 4, 5],
            "min_samples_split": [2, 3, 5],
            "n_estimators":      [100, 200, 300],
        },
    },
    "Lasso": {
        "estimator": Lasso(max_iter=10_000),
        "use_scaled": True,
        "param_grid": {
            "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                      1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
        },
    },
    "Logistic Regression": {
        "estimator": LogisticRegression(solver="saga", max_iter=10_000),
        "use_scaled": True,
        "param_grid": {
            "C":        [0.01, 0.1, 1, 10, 100],
            "l1_ratio": [0.0, 0.5, 1.0],
        },
    },
    "SVM": {
        "estimator": SVC(probability=True, random_state=42),
        "use_scaled": True,
        "param_grid": {
            "C":      [0.1, 1, 10, 100, 1000],
            "gamma":  [0.0001, 0.001, 0.01, 0.1, 1],
            "kernel": ["rbf"],
        },
    },
    # --- Extended models ---
    "Gradient Boosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "use_scaled": False,
        "param_grid": {
            "n_estimators":  [100, 200],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth":     [3, 5, 7],
            "subsample":     [0.8, 1.0],
        },
    },
    # Same kind of model as XGBoost, part of scikit-learn: runs without extra installation.
    "Hist Gradient Boosting": {
        "estimator": HistGradientBoostingClassifier(random_state=42),
        "use_scaled": False,
        "param_grid": {
            "max_iter":          [100, 200],
            "learning_rate":     [0.05, 0.1, 0.2],
            "max_depth":         [3, 5, 7],
            "min_samples_leaf":  [10, 20],
        },
    },
}

if XGBClassifier is not None:
    MODELS["XGBoost"] = {
        "estimator": XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
        "use_scaled": False,
        "param_grid": {
            "n_estimators":      [100, 200],
            "learning_rate":     [0.05, 0.1, 0.2],
            "max_depth":         [3, 5, 7],
            "subsample":         [0.8, 1.0],
            "colsample_bytree":  [0.8, 1.0],
        },
    }
# fmt: on
