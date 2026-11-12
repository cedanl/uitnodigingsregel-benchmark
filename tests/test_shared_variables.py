"""Shared variable names in the result file."""

import pytest

from uitnodigingsregel_benchmark.settings import SettingsError
from uitnodigingsregel_benchmark.shared_variables import (
    SHARED_NAMES,
    apply_shared_names,
    match_columns,
    merge_variable_groups,
    public_names,
)


def test_shared_list_has_unique_names():
    assert len(SHARED_NAMES) == len(set(SHARED_NAMES))
    assert "StudentAge" in SHARED_NAMES


def test_match_columns_ignores_case_and_punctuation():
    linked = match_columns(["studentage", "Aanmel aantal", "Reistijd_min"])
    assert linked == {
        "studentage": "StudentAge",
        "Aanmel aantal": "Aanmel_aantal",
        "Reistijd_min": None,
    }


def test_match_columns_uses_the_mapping_first():
    linked = match_columns(
        ["Leeftijd", "StudentAge"], {"Leeftijd": "StudentAge", "StudentAge": None}
    )
    assert linked == {"Leeftijd": "StudentAge", "StudentAge": None}


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({"Leeftijd": "Bestaat_niet"}, "Onbekende gedeelde variabele"),
        ({"Leeftijd": "StudentAge"}, "allebei"),
    ],
)
def test_match_columns_rejects_wrong_mapping(mapping, message):
    with pytest.raises(SettingsError, match=message):
        match_columns(["Leeftijd", "StudentAge"], mapping)


def test_public_names_number_the_own_variables():
    names = public_names({"Leeftijd": "StudentAge", "ROC_X": None, "Postcode": None})
    assert names == {
        "Leeftijd": "StudentAge",
        "ROC_X": "eigen_variabele_1",
        "Postcode": "eigen_variabele_2",
    }


def test_shared_groups_are_joined_unless_the_user_grouped_them():
    linked = match_columns(["VooroplNiveau_HAVO", "VooroplNiveau_MBO", "Economie", "Techniek"])
    groups = merge_variable_groups(linked, {"Mijn_richting": ["Economie"]})
    assert groups == {
        "Vooropleiding_niveau": ["VooroplNiveau_HAVO", "VooroplNiveau_MBO"],
        "Mijn_richting": ["Economie"],
    }


def test_apply_shared_names_removes_own_names():
    result = {
        "metadata": {
            "features_available": ["Leeftijd", "ROC_X", "VooroplNiveau_HAVO", "VooroplNiveau_MBO"]
        },
        "models": {
            "Lasso": {
                "feature_importance": [{"feature": "ROC_X", "importance": 1.0}],
                "variable_impact": [
                    {"variable": "Leeftijd", "columns": ["Leeftijd"], "precision_diff": 1.0},
                    {"variable": "ROC_X", "columns": ["ROC_X"], "precision_diff": 2.0},
                    {"variable": "Vooropl", "columns": ["VooroplNiveau_HAVO", "VooroplNiveau_MBO"]},
                    {"variable": "Mix", "columns": ["Leeftijd", "ROC_X"]},
                ],
            }
        },
    }
    linked = match_columns(result["metadata"]["features_available"], {"Leeftijd": "StudentAge"})
    local = apply_shared_names(result, public_names(linked))

    assert "ROC_X" not in str(result) and "Leeftijd" not in str(result)
    assert result["metadata"]["n_own_variables"] == 1
    impact = [e["variable"] for e in result["models"]["Lasso"]["variable_impact"]]
    assert impact == ["StudentAge", "eigen_variabele_1", "Vooropleiding_niveau", "eigen_groep_1"]
    assert local == {
        "StudentAge": "Leeftijd",
        "Vooropleiding_niveau": "Vooropl",
        "eigen_groep_1": "Mix",
        "eigen_variabele_1": "ROC_X",
    }
