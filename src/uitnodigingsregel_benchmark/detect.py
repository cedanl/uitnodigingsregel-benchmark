"""Recognise the role of columns in student data, so the user only has to confirm."""

import pandas as pd

DROPOUT_NAMES = ("dropout", "uitval", "uitgevallen", "uitvaller")
ID_NAMES = ("studentnummer", "student_id", "studentid", "leerlingnummer", "deelnemernummer")
COHORT_NAMES = ("cohort", "startjaar", "schooljaar", "studiejaar", "instroomjaar", "jaar")
MAX_COHORTS = 15


def _is_binary(column: pd.Series) -> bool:
    return set(column.dropna().unique()) <= {0, 1}


def _first_match(columns: list[str], names: tuple[str, ...]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for name in names:
        if name in lowered:
            return lowered[name]
    for name in names:
        for lower, column in lowered.items():
            if name in lower:
                return column
    return None


def detect_dropout_column(data: pd.DataFrame) -> str | None:
    """Find the dropout column: a column with only 0 and 1 and a name such as 'Dropout'."""
    binary = [c for c in data.columns if _is_binary(data[c])]
    return _first_match(binary, DROPOUT_NAMES)


def detect_id_column(data: pd.DataFrame) -> str | None:
    """Find the column with student numbers by its name."""
    return _first_match(list(data.columns), ID_NAMES)


def detect_cohort_column(data: pd.DataFrame) -> str | None:
    """Find a cohort column: a name such as 'Startjaar' and between 2 and 15 values."""
    candidates = [c for c in data.columns if 2 <= data[c].nunique() <= MAX_COHORTS]
    return _first_match(candidates, COHORT_NAMES)


def suggest_variable_groups(data: pd.DataFrame, exclude: list[str]) -> dict[str, list[str]]:
    """Suggest columns that together form one variable.

    Columns with only 0 and 1 that share the text before the first underscore, such as
    'VooroplNiveau_HAVO' and 'VooroplNiveau_MBO', are usually one variable split per value.

    Args:
        data: Student data.
        exclude: Columns that are not a feature (dropout, id and cohort column).

    Returns:
        Group name mapped to its columns, for groups of at least two columns.
    """
    groups: dict[str, list[str]] = {}
    for column in data.columns:
        if column in exclude or "_" not in column or not _is_binary(data[column]):
            continue
        groups.setdefault(column.split("_", 1)[0], []).append(column)
    return {name: columns for name, columns in groups.items() if len(columns) > 1}
