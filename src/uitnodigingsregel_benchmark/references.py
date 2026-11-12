"""Reference rules without a model: what inviting at random or on one variable achieves."""

import numpy as np
import pandas as pd

from uitnodigingsregel_benchmark.impact import INVITE_PCT
from uitnodigingsregel_benchmark.settings import SettingsError

RANDOM_STATE = 42


def _mean_sd(values: list[float]) -> tuple[float, float]:
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return round(float(np.mean(values)), 1), round(sd, 1)


def random_reference(tests: list[pd.DataFrame], dropout_col: str) -> dict:
    """Expected precision and recall when 20% of the students is invited at random.

    At random the share of dropouts among the invited equals the dropout rate, and the
    invited reach 20% of all dropouts.
    """
    precision, precision_sd = _mean_sd([test[dropout_col].mean() * 100 for test in tests])
    return {
        "precision": precision,
        "recall": float(INVITE_PCT),
        "precision_sd": precision_sd,
        "recall_sd": 0.0,
    }


def check_reference_variable(data: pd.DataFrame, variable: str, excluded: list[str]) -> None:
    """Check that a column can serve as the reference variable.

    Raises:
        SettingsError: If the column is missing, not a feature or not numeric.
    """
    if variable not in data.columns or variable in excluded:
        raise SettingsError(f"De referentievariabele '{variable}' is geen kolom met kenmerken.")
    if not pd.api.types.is_numeric_dtype(data[variable]):
        raise SettingsError(
            f"De referentievariabele '{variable}' moet een kolom met getallen zijn."
        )


def _invite_on_variable(train: pd.DataFrame, test: pd.DataFrame, variable: str, dropout_col: str):
    # Direction from the training data: invite the highest or the lowest values first
    correlation = train[variable].corr(train[dropout_col])
    direction = -1.0 if correlation < 0 else 1.0
    score = test[variable].fillna(train[variable].median()).to_numpy() * direction
    # Equal values, such as many students without absence, are invited in random order
    tie_break = np.random.default_rng(RANDOM_STATE).random(len(test))
    order = np.lexsort((tie_break, -score))
    n_invited = round(len(test) * INVITE_PCT / 100)
    dropouts = test[dropout_col].to_numpy()
    reached = dropouts[order[:n_invited]].sum()
    return reached / n_invited * 100, reached / max(dropouts.sum(), 1) * 100


def single_variable_reference(
    splits: list[tuple[pd.DataFrame, pd.DataFrame]], variable: str, dropout_col: str
) -> dict:
    """Precision and recall when the 20% students with the highest risk on one variable is invited.

    Whether a high or a low value means more risk follows from the training data.

    Args:
        splits: Train and test set of each split.
        variable: Column to invite on.
        dropout_col: Name of the dropout column.

    Returns:
        Mean and spread over the splits, and the name of the variable.
    """
    per_split = [_invite_on_variable(train, test, variable, dropout_col) for train, test in splits]
    precision, precision_sd = _mean_sd([p for p, _ in per_split])
    recall, recall_sd = _mean_sd([r for _, r in per_split])
    return {
        "variable": variable,
        "precision": precision,
        "recall": recall,
        "precision_sd": precision_sd,
        "recall_sd": recall_sd,
    }
