"""Check a result file before it is written or shared."""

import argparse
import json
import re
import sys
from importlib.resources import files
from pathlib import Path

from jsonschema import Draft202012Validator

from uitnodigingsregel_benchmark.shared_variables import SHARED_GROUP, SHARED_NAMES

FORMAT_VERSION = "1"
SUPPORTED_FORMAT_VERSIONS = {FORMAT_VERSION}
# Fewer dropouts in the test group make precision at 20% depend on a handful of students
MIN_TEST_DROPOUTS = 30

_OWN_VARIABLE = re.compile(r"^eigen_variabele_\d+$")
_OWN_GROUP = re.compile(r"^eigen_groep_\d+$")
_COLUMN_NAMES = set(SHARED_NAMES)
_VARIABLE_NAMES = _COLUMN_NAMES | set(SHARED_GROUP.values())

_VALIDATOR = Draft202012Validator(
    json.loads(
        (files("uitnodigingsregel_benchmark.metadata") / "result_schema.json").read_text(
            encoding="utf-8"
        )
    )
)


class ResultError(ValueError):
    """The result file cannot be shared; the message states why."""


def _unknown_names(result: dict) -> list[str]:
    """Names in the result that are neither shared nor neutral, such as own column names."""
    columns: set[str] = set(result["metadata"]["features_available"])
    single_variable = result.get("references", {}).get("single_variable")
    if single_variable:
        columns.add(single_variable["variable"])
    variables: set[str] = set()
    for model in result["models"].values():
        columns |= {e["feature"] for e in model.get("feature_importance", [])}
        for entry in model.get("variable_impact", []):
            variables.add(entry["variable"])
            columns |= set(entry["columns"])
    unknown = {c for c in columns if c not in _COLUMN_NAMES and not _OWN_VARIABLE.match(c)}
    unknown |= {
        v
        for v in variables
        if v not in _VARIABLE_NAMES and not _OWN_VARIABLE.match(v) and not _OWN_GROUP.match(v)
    }
    return sorted(unknown)


def check_result(result: dict) -> list[str]:
    """Check that a result can be shared.

    Args:
        result: A complete result, as written to the result file.

    Returns:
        Warnings that do not block sharing, such as a small test group.

    Raises:
        ResultError: If the format version is unknown, the result does not follow the
            schema, or it holds names outside the shared variable list.
    """
    version = result.get("metadata", {}).get("format_version")
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise ResultError(
            f"Onbekende versie van het formaat: {version or 'geen'}. Ondersteund: "
            f"{', '.join(sorted(SUPPORTED_FORMAT_VERSIONS))}. Maak het bestand opnieuw met de "
            "laatste versie van de benchmark."
        )
    errors = sorted(_VALIDATOR.iter_errors(result), key=lambda e: list(e.absolute_path))
    if errors:
        problems = "; ".join(
            f"{'/'.join(str(p) for p in e.absolute_path) or 'bestand'}: {e.message}"
            for e in errors[:5]
        )
        raise ResultError(f"Het bestand volgt het vastgelegde formaat niet. {problems}")
    unknown = _unknown_names(result)
    if unknown:
        raise ResultError(
            "Het bestand bevat namen die niet in de gedeelde variabelenlijst staan: "
            f"{', '.join(unknown[:10])}. Maak het bestand opnieuw met de laatste versie."
        )

    warnings = []
    metadata = result["metadata"]
    n_dropouts = round(metadata["n_test"] * metadata["dropout_rate_test"])
    if n_dropouts < MIN_TEST_DROPOUTS:
        warnings.append(
            f"De testgroep heeft {metadata['n_test']} studenten, van wie {n_dropouts} "
            f"uitgevallen. Bij minder dan {MIN_TEST_DROPOUTS} uitvallers hangt de precisie af "
            "van een handvol studenten: verschillen van enkele procentpunten zijn dan toeval."
        )
    return warnings


def main(argv: list[str] | None = None) -> int:
    """Check result files from the command line; exit code 1 if one cannot be shared."""
    parser = argparse.ArgumentParser(description="Check result files before sharing")
    parser.add_argument("paths", nargs="+", help="Result files")
    args = parser.parse_args(argv)
    failed = 0
    for path in args.paths:
        try:
            result = json.loads(Path(path).read_text(encoding="utf-8"))
            warnings = check_result(result)
        except (ResultError, json.JSONDecodeError, OSError) as error:
            failed += 1
            print(f"Afgekeurd: {path}\n  {error}")
            continue
        print(f"In orde: {path}")
        for warning in warnings:
            print(f"  Let op: {warning}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
