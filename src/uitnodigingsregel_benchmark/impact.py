"""Measure the impact of a variable: train the models again without it and compare."""

from collections.abc import Callable
from fnmatch import fnmatch
from typing import Any

import pandas as pd
import student_signal
from sklearn.base import clone
from student_signal.evaluate import prepare_model_predictions

from uitnodigingsregel_benchmark.export import extract_precision_recall_curve
from uitnodigingsregel_benchmark.settings import SettingsError

INVITE_PCT = 20
RETRAIN_FAILED = "Het model kon niet opnieuw worden getraind zonder deze variabele."


def precision_recall_at_invite_pct(
    model: Any,
    val_data: pd.DataFrame,
    dropout_col: str,
) -> tuple[float, float]:
    """Precision and recall (in %) when the top 20% of students is invited."""
    eval_df = prepare_model_predictions(val_data, model, "model", dropout_col)
    point = extract_precision_recall_curve(eval_df, "model", [INVITE_PCT])[0]
    return point["precision"], point["recall"]


def resolve_variables(
    columns: list[str],
    variable_groups: dict[str, list[str]] | None = None,
    impact_variables: list[str] | None = None,
) -> dict[str, list[str]]:
    """Determine which variables to test and which columns each one covers.

    A group joins columns that belong to one original variable, such as the columns
    per level of prior education. Columns outside a group are a variable of their own.

    Args:
        columns: Feature columns in the data (without dropout, id and cohort column).
        variable_groups: Variable name mapped to column names or patterns ('Vooropl_*').
        impact_variables: Names of the variables to test, or None for all.

    Returns:
        Variable name mapped to its columns, in the order of the data.

    Raises:
        SettingsError: If a group matches no column, a column is in two groups or a
            requested variable does not exist.
    """
    grouped: dict[str, str] = {}
    variables: dict[str, list[str]] = {}
    for name, patterns in (variable_groups or {}).items():
        members = [c for c in columns if any(fnmatch(c, p) for p in patterns)]
        if not members:
            raise SettingsError(
                f"De groep '{name}' in variable_groups past op geen enkele kolom in de data."
            )
        for column in members:
            if column in grouped:
                raise SettingsError(
                    f"De kolom '{column}' staat in twee groepen in variable_groups: "
                    f"'{grouped[column]}' en '{name}'."
                )
            grouped[column] = name
        variables[name] = members
    for column in columns:
        if column not in grouped:
            variables[column] = [column]

    if impact_variables is None:
        return variables
    unknown = [v for v in impact_variables if v not in variables]
    if unknown:
        raise SettingsError(
            f"Onbekende variabele in impact_variables: {', '.join(unknown)}. "
            "Gebruik een kolomnaam of de naam van een groep uit variable_groups."
        )
    return {v: variables[v] for v in impact_variables}


def measure_variable_impact(
    train: pd.DataFrame,
    test: pd.DataFrame,
    variables: dict[str, list[str]],
    fitted_models: dict[str, tuple[Any, bool]],
    baseline: dict[str, tuple[float, float]],
    dropout_col: str,
    id_col: str,
    on_step: Callable[[str], None] | None = None,
) -> dict[str, list[dict]]:
    """Train each model again without one variable at a time and compare with the baseline.

    The models keep the settings found in the parameter search; only the data changes.
    A positive difference means the model does better with the variable than without.

    Args:
        train: Train set before preparation.
        test: Test set before preparation.
        variables: Variable name mapped to its columns, from resolve_variables().
        fitted_models: Model name mapped to (fitted estimator, uses scaled data).
        baseline: Model name mapped to (precision, recall) at 20% with all variables.
        dropout_col: Name of the dropout column.
        id_col: Name of the identification column.
        on_step: Called with a description before each variable, to show progress.

    Returns:
        Per model a list with one entry per variable, largest precision difference first.
    """
    impact: dict[str, list[dict]] = {name: [] for name in fitted_models}
    for number, (variable, columns) in enumerate(variables.items(), start=1):
        print(f"  Zonder '{variable}' ({number} van {len(variables)})...")
        if on_step:
            on_step(f"Impact meten: zonder '{variable}' ({number} van {len(variables)})")
        prepared = student_signal.prepare(
            train.drop(columns=columns),
            test.drop(columns=columns),
            target_col=dropout_col,
            id_col=id_col,
        )
        for name, (estimator, use_scaled) in fitted_models.items():
            X_train = prepared.X_train_scaled if use_scaled else prepared.X_train
            val_data = prepared.pred_df_scaled if use_scaled else prepared.pred_df
            entry: dict = {"variable": variable, "columns": columns}
            try:
                model = clone(estimator).fit(X_train, prepared.y_train)
            except ValueError as error:
                # For example a model setting that needs more columns than are left. The
                # message may name columns, so it is printed here and not stored.
                print(f"    {name} kon niet opnieuw worden getraind: {error}")
                entry["error"] = RETRAIN_FAILED
            else:
                precision, recall = precision_recall_at_invite_pct(model, val_data, dropout_col)
                entry.update(
                    {
                        "precision_without": precision,
                        "recall_without": recall,
                        "precision_diff": round(baseline[name][0] - precision, 1),
                        "recall_diff": round(baseline[name][1] - recall, 1),
                    }
                )
            impact[name].append(entry)

    for entries in impact.values():
        entries.sort(key=lambda e: e.get("precision_diff", float("-inf")), reverse=True)
    return impact
