"""Split the data in a train and a test set: by cohort when possible, else at random."""

import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_COHORT = "cohort"
SPLIT_RANDOM = "random"


def _plain(value: object) -> object:
    """Convert a numpy scalar to a plain Python value, so it can be written to JSON."""
    return value.item() if hasattr(value, "item") else value


def split_train_test(
    data: pd.DataFrame,
    dropout_col: str,
    cohort_col: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split the data in a train and a test set.

    With a cohort column that holds at least two cohorts, the models train on the
    earlier cohorts and are tested on the last one. Without it the data is split at
    random, stratified on the dropout column. The cohort column is never a feature.

    Args:
        data: Student data with the dropout column.
        dropout_col: Name of the dropout column.
        cohort_col: Name of the cohort column, or None for a random split.
        test_size: Share of the data in the test set for a random split.
        random_state: Seed for a random split.
        verbose: Whether to print why rows are left out or the split falls back.

    Returns:
        Train set, test set and a description of the split for the result file.

    Raises:
        ValueError: If the test cohort also occurs in the train set.
    """
    reason = None
    if cohort_col and cohort_col not in data.columns:
        reason = f"kolom '{cohort_col}' staat niet in de data"
    elif cohort_col:
        n_missing = int(data[cohort_col].isna().sum())
        if n_missing and verbose:
            print(f"{n_missing} rijen zonder waarde in '{cohort_col}' doen niet mee.")
            data = data[data[cohort_col].notna()]
        cohorts = sorted(data[cohort_col].unique())
        if len(cohorts) < 2:
            reason = f"kolom '{cohort_col}' bevat maar één cohort"

    if cohort_col and reason is None:
        test_cohort = cohorts[-1]
        train = data[data[cohort_col] != test_cohort]
        test = data[data[cohort_col] == test_cohort]
        if test_cohort in set(train[cohort_col]):
            raise ValueError(f"Het testcohort {test_cohort} zit ook in de trainingsdata.")
        info = {
            "type": SPLIT_COHORT,
            "train_cohorts": [_plain(c) for c in cohorts[:-1]],
            "test_cohort": _plain(test_cohort),
        }
    else:
        if reason and verbose:
            print(f"Geen splitsing op cohort: {reason}. De data wordt willekeurig gesplitst.")
        train, test = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            stratify=data[dropout_col],
        )
        info = {"type": SPLIT_RANDOM, "test_size": test_size}

    if cohort_col and cohort_col in data.columns:
        train = train.drop(columns=cohort_col)
        test = test.drop(columns=cohort_col)
    return train.reset_index(drop=True), test.reset_index(drop=True), info
