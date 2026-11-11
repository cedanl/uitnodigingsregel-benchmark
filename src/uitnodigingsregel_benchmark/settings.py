"""Benchmark settings: defaults, the user's config.yaml and quick mode."""

from pathlib import Path

import yaml

DEFAULTS: dict = {
    "instelling_id": "anoniem",
    "dropout_year": None,
    "training_years": None,
    "cohort_column": None,
    "dropout_column": "Dropout",
    "id_column": "Studentnummer",
    "models": None,
    "n_iter": 20,
    "cv_folds": 5,
    "test_size": 0.2,
    "n_jobs": -1,
}

# Fewer models and fewer search rounds: a result in seconds instead of minutes.
QUICK_SETTINGS: dict = {
    "models": ["Random Forest", "Lasso", "Logistic Regression"],
    "n_iter": 5,
    "cv_folds": 3,
}


class SettingsError(ValueError):
    """The settings cannot be used; the message is written for the end user."""


def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Read the settings from a YAML file and fill in the defaults.

    Args:
        config_path: Path to the YAML file. A missing file gives the defaults.

    Returns:
        Settings with a value for every key in DEFAULTS.

    Raises:
        SettingsError: If the file holds a setting that does not exist.
    """
    config_path = Path(config_path)
    user_config: dict = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}

    unknown = sorted(set(user_config) - set(DEFAULTS))
    if unknown:
        raise SettingsError(
            f"Onbekende instelling in {config_path}: {', '.join(unknown)}. "
            f"Mogelijke instellingen: {', '.join(DEFAULTS)}."
        )
    return {**DEFAULTS, **{k: v for k, v in user_config.items() if v is not None}}
