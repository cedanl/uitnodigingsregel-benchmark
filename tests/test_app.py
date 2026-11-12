"""The app, from choosing the data to the result on screen."""

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from uitnodigingsregel_benchmark import sharing
from uitnodigingsregel_benchmark.aggregate import build_shared_data
from uitnodigingsregel_benchmark.cli import APP_FILE
from uitnodigingsregel_benchmark.detect import (
    detect_cohort_column,
    detect_dropout_column,
    detect_id_column,
    suggest_variable_groups,
)
from uitnodigingsregel_benchmark.report import build_html_report

REPO_ROOT = Path(__file__).parent.parent
DEMO_TRAIN = REPO_ROOT / "data/01-raw/demo/synth_data_train.csv"
EXAMPLE_RESULT = REPO_ROOT / "results/voorbeeld/benchmark_demo_onbekend.json"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """The app never reaches the internet in tests."""
    monkeypatch.setattr(sharing, "fetch_shared_data", lambda: None)


def _students() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Studentnummer": [1, 2, 3, 4],
            "Startjaar": [2021, 2021, 2022, 2022],
            "Leeftijd": [16, 17, 18, 19],
            "Niveau_HAVO": [1, 0, 0, 1],
            "Niveau_MBO": [0, 1, 1, 0],
            "verzuim_uren": [3, 0, 8, 1],
            "Dropout": [0, 1, 0, 1],
        }
    )


def test_detect_columns():
    data = _students()
    assert detect_dropout_column(data) == "Dropout"
    assert detect_id_column(data) == "Studentnummer"
    assert detect_cohort_column(data) == "Startjaar"
    assert detect_cohort_column(data.assign(Startjaar=2021)) is None
    assert detect_dropout_column(data.drop(columns="Dropout")) is None


def test_suggest_variable_groups():
    data = _students()
    assert suggest_variable_groups(data, ["Dropout"]) == {"Niveau": ["Niveau_HAVO", "Niveau_MBO"]}
    assert suggest_variable_groups(data, ["Dropout", "Niveau_MBO"]) == {}


def test_app_runs_benchmark_on_demo_data(tmp_path, monkeypatch):
    if not DEMO_TRAIN.exists():
        pytest.skip("Demo data niet aanwezig")
    # The app reads the demo data from, and writes results to, the working directory
    (tmp_path / "data/01-raw").mkdir(parents=True)
    (tmp_path / "data/01-raw/demo").symlink_to(DEMO_TRAIN.parent)
    (tmp_path / "config.yaml").write_text("n_jobs: 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    app = AppTest.from_file(str(APP_FILE), default_timeout=300).run()
    assert not app.exception
    assert len(app.subheader) == 1

    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    selected = {box.label: box.value for box in app.selectbox}
    assert selected["Kolom met uitval"] == "Dropout"
    assert selected["Kolom met studentnummer"] == "Studentnummer"
    assert selected["Kolom met cohort of startjaar"] == "(geen)"
    assert any("VooroplNiveau" in box.label for box in app.checkbox)

    start = next(button for button in app.button if button.label == "Start benchmark")
    start.click().run()
    assert not app.exception
    assert not app.error
    assert "Uitkomst" in [s.value for s in app.subheader]
    assert "Impact per variabele" in [s.value for s in app.subheader]
    assert len(app.metric) == 3

    result_files = list((tmp_path / "results").glob("benchmark_*.json"))
    assert len(result_files) == 1
    result = json.loads(result_files[0].read_text(encoding="utf-8"))
    assert result["metadata"]["quick_mode"] is True
    impact = result["models"]["Lasso"]["variable_impact"]
    assert "VooroplNiveau" in {entry["variable"] for entry in impact}

    assert [button.label for button in app.get("download_button")] == [
        "Download rapport",
        "Download resultaatbestand",
    ]
    report = result_files[0].with_suffix(".html").read_text(encoding="utf-8")
    assert report == build_html_report(result)


def test_report_is_standalone_and_shows_the_result():
    result = json.loads(EXAMPLE_RESULT.read_text(encoding="utf-8"))
    report = build_html_report(result)
    assert report.startswith("<!doctype html>")
    assert "http://" not in report
    assert "https://" not in report
    assert "<script" not in report
    assert result["recommendation"]["model"] in report
    for name in result["models"]:
        assert name in report
    assert "Impact per variabele" in report
    assert "VooroplNiveau_HAVO" in report or "VooroplNiveau" in report


def test_report_without_impact_and_with_cohort_split():
    result = json.loads(EXAMPLE_RESULT.read_text(encoding="utf-8"))
    result["metadata"]["impact_measured"] = False
    result["metadata"]["split"] = {
        "type": "cohort",
        "train_cohorts": [2021, 2022],
        "test_cohort": 2023,
    }
    result["metadata"]["instelling_id"] = "<b>naam</b>"
    report = build_html_report(result)
    assert "Impact per variabele" not in report
    assert "getoetst op cohort 2023" in report
    assert "<b>naam</b>" not in report


def _run_app_on_demo(tmp_path, monkeypatch) -> AppTest:
    (tmp_path / "data/01-raw").mkdir(parents=True)
    (tmp_path / "data/01-raw/demo").symlink_to(DEMO_TRAIN.parent)
    (tmp_path / "config.yaml").write_text("n_jobs: 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(APP_FILE), default_timeout=300).run()
    app.button[0].click().run()
    next(b for b in app.button if b.label == "Start benchmark").click().run()
    assert not app.exception
    return app


def test_app_quick_run_is_not_for_sharing(tmp_path, monkeypatch):
    if not DEMO_TRAIN.exists():
        pytest.skip("Demo data niet aanwezig")
    app = _run_app_on_demo(tmp_path, monkeypatch)
    text = " ".join(m.value for m in app.markdown)
    assert "Een snelle run is niet bedoeld om te delen" in text
    assert "nog geen gedeelde resultaten" in text


def test_app_compares_with_shared_results(tmp_path, monkeypatch):
    if not DEMO_TRAIN.exists():
        pytest.skip("Demo data niet aanwezig")
    other = json.loads(EXAMPLE_RESULT.read_text(encoding="utf-8"))
    other["metadata"]["instelling_id"] = "andere-instelling"
    monkeypatch.setattr(sharing, "fetch_shared_data", lambda: build_shared_data([other]))
    app = _run_app_on_demo(tmp_path, monkeypatch)
    text = " ".join(m.value for m in app.markdown)
    assert "Impact per variabele, in procentpunten precisie" in text
    compared = app.dataframe[-1].value
    assert "StudentAge" in set(compared["Variabele"])
    assert set(compared["Aantal instellingen"]) == {1}
