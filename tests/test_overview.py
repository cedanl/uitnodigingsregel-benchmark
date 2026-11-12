"""Combining shared results and building the overview page."""

import copy
import json
from datetime import date
from pathlib import Path

from conftest import make_result

from uitnodigingsregel_benchmark.aggregate import (
    build_shared_data,
    is_shareable,
    load_shared_results,
    variable_impact_table,
)
from uitnodigingsregel_benchmark.overview import build_overview_html, build_site

EXAMPLE_RESULT = Path(__file__).parent.parent / "results/voorbeeld/benchmark_demo_onbekend.json"


def _result(instelling: str, diff: float, split: str = "random", n: int = 1000) -> dict:
    return make_result(instelling, diff, split={"type": split}, n_train=n - 200, n_test=200)


def test_demo_and_quick_results_do_not_count():
    result = _result("a", 2.0)
    assert is_shareable(result)
    quick = copy.deepcopy(result)
    quick["metadata"]["quick_mode"] = True
    assert not is_shareable(quick)
    assert not is_shareable(_result("demo", 2.0))


def test_load_shared_results_skips_subfolders_and_demo(tmp_path):
    (tmp_path / "voorbeeld").mkdir()
    for path, result in (
        ("benchmark_a_2023.json", _result("a", 2.0)),
        ("benchmark_demo_2023.json", _result("demo", 2.0)),
        ("voorbeeld/benchmark_b_2023.json", _result("b", 2.0)),
    ):
        (tmp_path / path).write_text(json.dumps(result), encoding="utf-8")
    assert [r["metadata"]["instelling_id"] for r in load_shared_results(tmp_path)] == ["a"]


def test_variable_impact_is_combined_per_institution():
    results = [_result("a", 4.0), _result("a", 2.0), _result("b", -1.0)]
    rows = variable_impact_table(results)
    assert rows == [
        {
            "variable": "StudentAge",
            "n_institutions": 2,
            "mean": 1.0,
            "median": 1.0,
            "lowest": -1.0,
            "highest": 3.0,
        }
    ]


def test_summary_counts_an_institution_once():
    shared = build_shared_data([_result("a", 4.0, n=1000), _result("a", 2.0, n=600)])
    assert shared["summary"] == {"n_institutions": 1, "n_results": 2, "n_students": 1000}


def test_overview_page_shows_results_per_kind_of_split():
    shared = build_shared_data([_result("a", 4.0, "cohort"), _result("<b>", -1.0)])
    page = build_overview_html(shared, date(2026, 1, 2))
    assert page.startswith("<!doctype html>")
    assert "<script" not in page
    assert "Getoetst op een later cohort" in page
    assert "Willekeurig gesplitst" in page
    assert "Variabelen met de grootste bijdrage" in page
    assert "StudentAge" in page
    assert "<b>" not in page
    assert "2026-01-02" in page


def test_overview_page_without_results():
    page = build_overview_html(build_shared_data([]))
    assert "nog geen gedeelde resultaten" in page
    assert "Impact per variabele" not in page


def test_build_site_writes_page_and_data(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "benchmark_a_2023.json").write_text(
        json.dumps(_result("a", 2.0)), encoding="utf-8"
    )
    index = build_site(results_dir, tmp_path / "site")
    assert index.exists()
    shared = json.loads((tmp_path / "site/shared_results.json").read_text(encoding="utf-8"))
    assert shared["summary"]["n_institutions"] == 1


def test_example_result_does_not_count_in_the_overview():
    assert not is_shareable(json.loads(EXAMPLE_RESULT.read_text(encoding="utf-8")))
