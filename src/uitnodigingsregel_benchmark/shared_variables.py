"""Shared variable names: a result file holds these names, never the institution's own."""

import json
import re
from fnmatch import fnmatch
from importlib.resources import files
from pathlib import Path

import yaml

from uitnodigingsregel_benchmark.settings import SettingsError

MAPPING_FILE = Path("koppeling.json")
OWN_VARIABLE_PREFIX = "eigen_variabele_"
OWN_GROUP_PREFIX = "eigen_groep_"
OWN_PREFIXES = (OWN_VARIABLE_PREFIX, OWN_GROUP_PREFIX)


def _load_shared_variables() -> list[dict]:
    path = files("uitnodigingsregel_benchmark.metadata") / "variabelen.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["variabelen"]


SHARED_VARIABLES: list[dict] = _load_shared_variables()
SHARED_NAMES: list[str] = [v["naam"] for v in SHARED_VARIABLES]
SHARED_GROUP: dict[str, str] = {v["naam"]: v["groep"] for v in SHARED_VARIABLES if v.get("groep")}


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


_BY_NORMALISED = {_normalise(name): name for name in SHARED_NAMES}


def is_own_name(name: str) -> bool:
    """Whether a name in a result stands for a variable outside the shared list."""
    return str(name).startswith(OWN_PREFIXES)


def match_columns(
    columns: list[str], mapping: dict[str, str | None] | None = None
) -> dict[str, str | None]:
    """Link each column to a shared variable.

    A column in the mapping gets the name stated there (None: an own variable). Other
    columns are linked when their name equals a shared name, ignoring case and punctuation.

    Args:
        columns: Feature columns in the data.
        mapping: Column name mapped to a shared name or None, as chosen by the institution.

    Returns:
        Column name mapped to its shared name, or None without a shared name.

    Raises:
        SettingsError: If the mapping names an unknown shared variable, or links two
            columns to the same one.
    """
    mapping = mapping or {}
    linked: dict[str, str | None] = {}
    for column in columns:
        if column in mapping:
            linked[column] = mapping[column]
        else:
            linked[column] = _BY_NORMALISED.get(_normalise(column))

    unknown = sorted({name for name in linked.values() if name and name not in SHARED_NAMES})
    if unknown:
        raise SettingsError(
            f"Onbekende gedeelde variabele in de koppeling: {', '.join(unknown)}. "
            "De gedeelde variabelen staan in metadata/variabelen.yaml."
        )
    seen: dict[str, str] = {}
    for column, name in linked.items():
        if name and name in seen:
            raise SettingsError(
                f"De kolommen '{seen[name]}' en '{column}' zijn allebei gekoppeld aan de "
                f"gedeelde variabele '{name}'. Koppel er één aan een andere variabele."
            )
        if name:
            seen[name] = column
    return linked


def public_names(linked: dict[str, str | None]) -> dict[str, str]:
    """Give each column the name it gets in the result file.

    Args:
        linked: Output of match_columns().

    Returns:
        Column name mapped to its shared name, or to 'eigen_variabele_N' without one.
    """
    names: dict[str, str] = {}
    for column, shared in linked.items():
        if shared:
            names[column] = shared
        else:
            n_own = sum(1 for name in names.values() if is_own_name(name))
            names[column] = f"{OWN_VARIABLE_PREFIX}{n_own + 1}"
    return names


def merge_variable_groups(
    linked: dict[str, str | None], user_groups: dict[str, list[str]] | None = None
) -> dict[str, list[str]]:
    """Join the groups of the shared list with the institution's own groups.

    Shared variables that belong to one group, such as the columns per level of prior
    education, always count as one variable so results of institutions can be combined.
    A column the institution placed in a group of its own stays there.

    Args:
        linked: Output of match_columns().
        user_groups: Group name mapped to column names or patterns, from the settings.

    Returns:
        Group name mapped to column names or patterns, for resolve_variables().
    """
    user_groups = user_groups or {}
    in_user_group = {
        column
        for patterns in user_groups.values()
        for column in linked
        if any(fnmatch(column, pattern) for pattern in patterns)
    }
    shared: dict[str, list[str]] = {}
    for column, name in linked.items():
        group = SHARED_GROUP.get(name or "")
        if group and column not in in_user_group:
            shared.setdefault(group, []).append(column)
    groups = {name: columns for name, columns in shared.items() if len(columns) > 1}
    return {**groups, **user_groups}


def apply_shared_names(result: dict, names: dict[str, str]) -> dict[str, str]:
    """Replace the institution's column and group names in a result by public names.

    Changes the result in place.

    Args:
        result: Result from build_result(), with the institution's own names.
        names: Output of public_names().

    Returns:
        Public name mapped to the institution's own name, for every name that changed.
    """
    names = dict(names)
    used = set(names.values())

    def column_name(column: str) -> str:
        if column not in names:
            # A column made during preparation, such as one per value of a text column
            n_own = sum(1 for name in used if name.startswith(OWN_VARIABLE_PREFIX))
            names[column] = f"{OWN_VARIABLE_PREFIX}{n_own + 1}"
            used.add(names[column])
        return names[column]

    group_names: dict[str, str] = {}

    def variable_name(variable: str, columns: list[str]) -> str:
        if columns == [variable]:
            return column_name(variable)
        if variable not in group_names:
            shared_groups = {SHARED_GROUP.get(column_name(c)) for c in columns}
            name = next(iter(shared_groups)) if len(shared_groups) == 1 else None
            if not name or name in used:
                n_own = sum(1 for n in used if n.startswith(OWN_GROUP_PREFIX))
                name = f"{OWN_GROUP_PREFIX}{n_own + 1}"
            group_names[variable] = name
            used.add(name)
        return group_names[variable]

    metadata = result["metadata"]
    metadata["features_available"] = [column_name(c) for c in metadata["features_available"]]
    for model in result["models"].values():
        for entry in model.get("feature_importance", []):
            entry["feature"] = column_name(entry["feature"])
        for entry in model.get("variable_impact", []):
            entry["variable"] = variable_name(entry["variable"], entry["columns"])
            entry["columns"] = [column_name(c) for c in entry["columns"]]
    single_variable = result.get("references", {}).get("single_variable")
    if single_variable:
        single_variable["variable"] = column_name(single_variable["variable"])
    metadata["n_own_variables"] = sum(
        1 for name in metadata["features_available"] if is_own_name(name)
    )

    local = {public: own for own, public in {**names, **group_names}.items() if public != own}
    return dict(sorted(local.items()))


def load_mapping(path: str | Path = MAPPING_FILE) -> dict[str, str | None]:
    """Read the institution's column mapping; a missing file gives an empty mapping."""
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_mapping(mapping: dict[str, str | None], path: str | Path = MAPPING_FILE) -> None:
    """Write the institution's column mapping, so the next run starts from it."""
    Path(path).write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
