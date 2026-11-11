"""Read the student data and check that the benchmark can use it."""

from pathlib import Path

import pandas as pd

MAX_LISTED_COLUMNS = 5
ROW_NUMBER_ID = "rijnummer"


class DataValidationError(ValueError):
    """The data cannot be used; the message is written for the end user."""


def _list_columns(columns: object) -> str:
    names = [f"'{c}'" for c in columns]
    listed = ", ".join(names[:MAX_LISTED_COLUMNS])
    extra = len(names) - MAX_LISTED_COLUMNS
    return f"{listed} en {extra} andere" if extra > 0 else listed


def read_student_data(path: str | Path) -> pd.DataFrame:
    """Read a student CSV file; the separator is detected automatically.

    Args:
        path: Path to the CSV file.

    Returns:
        The data without duplicate rows.

    Raises:
        DataValidationError: If the file does not exist or cannot be read.
    """
    path = Path(path)
    if not path.exists():
        raise DataValidationError(
            f"Het bestand {path} bestaat niet. Zet je CSV in data/01-raw/user_data/ "
            "en geef het pad op met --train, of gebruik --demo voor de demo-data."
        )
    try:
        data = pd.read_csv(path, sep=None, engine="python")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as e:
        raise DataValidationError(
            f"Het bestand {path} kan niet worden gelezen als CSV ({e})."
        ) from e
    return data.drop_duplicates().reset_index(drop=True)


def validate_student_data(
    data: pd.DataFrame,
    dropout_column: str = "Dropout",
    id_column: str = "Studentnummer",
    cv_folds: int = 5,
) -> None:
    """Check that the data can be used to train and test the models.

    Args:
        data: Student data.
        dropout_column: Name of the dropout column.
        id_column: Name of the student number column.
        cv_folds: Number of folds in the parameter search; sets the minimum group size.

    Raises:
        DataValidationError: With a Dutch message that names the problem and the columns.
    """
    if data.empty:
        raise DataValidationError("De data bevat geen rijen.")
    if dropout_column not in data.columns:
        raise DataValidationError(
            f"In de data ontbreekt de kolom '{dropout_column}'. "
            f"Gevonden kolommen: {_list_columns(data.columns)}. "
            "Heet de kolom anders, zet de naam dan bij dropout_column in config.yaml. "
            "Staat alles in één kolom, dan gebruikt het bestand een scheidingsteken "
            "dat niet herkend wordt. Gebruik tab, puntkomma of komma."
        )

    values = set(data[dropout_column].dropna().unique())
    if data[dropout_column].isna().any() or not values <= {0, 1}:
        raise DataValidationError(
            f"De kolom '{dropout_column}' mag alleen 0 en 1 bevatten, zonder lege cellen."
        )
    counts = data[dropout_column].value_counts()
    # Both groups must be large enough to end up in every fold of the train set
    minimum = 2 * cv_folds
    if len(counts) < 2 or counts.min() < minimum:
        raise DataValidationError(
            f"De data bevat {int(counts.get(1, 0))} uitvallers (1) en "
            f"{int(counts.get(0, 0))} niet-uitvallers (0). "
            f"Van beide zijn er minstens {minimum} nodig."
        )

    features = data.drop(columns=[c for c in (id_column, dropout_column) if c in data.columns])
    if features.select_dtypes(include=["number"]).empty:
        raise DataValidationError(
            f"De data bevat geen kolommen met getallen naast '{dropout_column}'. "
            "Controleer of getallen als getal zijn opgeslagen."
        )


def ensure_id_column(
    data: pd.DataFrame, id_column: str = "Studentnummer"
) -> tuple[pd.DataFrame, str]:
    """Return the data with an identification column, adding row numbers if it is missing.

    The benchmark does not use the identification, so a file without student numbers works.

    Args:
        data: Student data.
        id_column: Name of the student number column.

    Returns:
        The data and the name of the identification column to use.
    """
    if id_column in data.columns:
        return data, id_column
    print(f"Geen kolom '{id_column}' gevonden; de benchmark gebruikt rijnummers.")
    data = data.copy()
    data.insert(0, ROW_NUMBER_ID, range(len(data)))
    return data, ROW_NUMBER_ID
