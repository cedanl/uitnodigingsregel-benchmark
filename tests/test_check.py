"""Checking a result file before it is written or shared."""

import copy
import json

import pytest
from conftest import make_result

from uitnodigingsregel_benchmark.check import ResultError, check_result, main
from uitnodigingsregel_benchmark.export import write_result


def test_valid_result_passes(valid_result):
    assert check_result(valid_result) == []


@pytest.mark.parametrize("version", [None, "0", "2"])
def test_unknown_format_version_is_refused(valid_result, version):
    valid_result["metadata"]["format_version"] = version
    with pytest.raises(ResultError, match="Onbekende versie"):
        check_result(valid_result)


def test_extra_fields_are_refused(valid_result):
    valid_result["metadata"]["studenten"] = [{"Studentnummer": 123, "Dropout": 1}]
    with pytest.raises(ResultError, match="formaat"):
        check_result(valid_result)


def test_own_column_names_are_refused(valid_result):
    valid_result["metadata"]["features_available"].append("ROCMondriaan")
    with pytest.raises(ResultError, match="ROCMondriaan"):
        check_result(valid_result)


def test_shared_group_and_own_group_names_are_allowed(valid_result):
    impact = valid_result["models"]["Lasso"]["variable_impact"]
    impact.append({"variable": "Vooropleiding_niveau", "columns": ["VooroplNiveau_HAVO"]})
    impact.append({"variable": "eigen_groep_1", "columns": ["eigen_variabele_1"]})
    assert check_result(valid_result) == []


def test_small_test_group_gives_a_warning():
    warnings = check_result(make_result(n_test=100, dropout_rate_test=0.1))
    assert len(warnings) == 1
    assert "10 uitgevallen" in warnings[0]


def test_write_result_refuses_and_writes_nothing(tmp_path, valid_result):
    broken = copy.deepcopy(valid_result)
    broken["metadata"]["features_available"] = ["Postcode"]
    with pytest.raises(ResultError):
        write_result(broken, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_command_reports_each_file(tmp_path, valid_result, capsys):
    good = tmp_path / "goed.json"
    good.write_text(json.dumps(valid_result), encoding="utf-8")
    old = tmp_path / "oud.json"
    del valid_result["metadata"]["format_version"]
    old.write_text(json.dumps(valid_result), encoding="utf-8")
    assert main([str(good)]) == 0
    assert main([str(good), str(old)]) == 1
    output = capsys.readouterr().out
    assert "In orde" in output and "Afgekeurd" in output and "oud.json" in output
