"""Build the collective overview page from the shared results."""

import argparse
import json
from datetime import date
from html import escape
from pathlib import Path

from uitnodigingsregel_benchmark.aggregate import build_shared_data, load_shared_results
from uitnodigingsregel_benchmark.report import COLOR_HELPS, COLOR_HURTS, CSS

REPO_URL = "https://github.com/cedanl/uitnodigingsregel-benchmark"
SHARED_DATA_FILE = "shared_results.json"
SPLIT_LABELS = {
    "cohort": (
        "Getoetst op een later cohort",
        "Getraind op eerdere cohorten en getoetst op het laatste cohort. Dit lijkt het meest "
        "op het gebruik in de praktijk.",
    ),
    "random": (
        "Willekeurig gesplitst",
        "Train- en testdata zijn willekeurig getrokken uit dezelfde cohorten. Deze uitkomst "
        "valt vaak gunstiger uit en is niet te vergelijken met een toets op een later cohort.",
    ),
}
TOP_VARIABLES = 5

OVERVIEW_CSS = """
.range { position: relative; height: 18px; }
.range .line { position: absolute; top: 8px; height: 2px; background: #999; }
.range .dot { position: absolute; top: 3px; width: 12px; height: 12px; border-radius: 50%;
              margin-left: -6px; }
.range .zero { position: absolute; top: 0; bottom: 0; width: 1px; background: #888; }
.bars td.count { width: 90px; text-align: right; color: #5a5a5a; font-size: 14px; }
ol.top { padding-left: 20px; }
"""


def _range_row(row: dict, low: float, high: float) -> str:
    """One variable as a dot (mean) on a line (lowest to highest institution)."""

    def position(value: float) -> float:
        return (value - low) / (high - low) * 100

    color = COLOR_HELPS if row["mean"] > 0 else (COLOR_HURTS if row["mean"] < 0 else "#9a9a9a")
    left, right = position(row["lowest"]), position(row["highest"])
    return (
        f'<tr><td class="name">{escape(str(row["variable"]))}</td>'
        f'<td><div class="range"><div class="zero" style="left:{position(0):.1f}%"></div>'
        f'<div class="line" style="left:{left:.1f}%;width:{right - left:.1f}%"></div>'
        f'<div class="dot" style="left:{position(row["mean"]):.1f}%;background:{color}"></div>'
        f"</div></td>"
        f'<td class="value">{row["mean"]:+.1f}</td>'
        f'<td class="count">{row["n_institutions"]} inst.</td></tr>'
    )


def _variables_section(variables: list[dict]) -> str:
    if not variables:
        return "<p>Nog geen resultaten met een meting van de impact per variabele.</p>"
    low = min(0.0, *(r["lowest"] for r in variables))
    high = max(0.0, *(r["highest"] for r in variables))
    if high == low:
        high = low + 1
    rows = "".join(_range_row(row, low, high) for row in variables)
    return (
        '<p class="note">Per variabele: het verschil in precisie bij 20% uitgenodigde '
        "studenten tussen een model met en zonder de variabele, in procentpunten. De stip is "
        "het gemiddelde over de instellingen, de lijn loopt van de laagste tot de hoogste "
        "instelling. Rechts van de streep: modellen voorspellen beter mét de variabele. Een "
        "variabele die bij één instelling is getoetst zegt nog weinig.</p>"
        f'<table class="bars">{rows}</table>'
    )


def _models_section(models: list[dict]) -> str:
    parts = []
    for split, (title, explanation) in SPLIT_LABELS.items():
        rows = [r for r in models if r["split"] == split]
        if not rows:
            continue
        body = "".join(
            f"<tr><td>{escape(r['model'])}</td><td>{escape(str(r['instelling']))}</td>"
            f"<td>{escape(str(r['year'] or ''))}</td>"
            f'<td class="number">{r["precision"]:.0f}%</td>'
            f'<td class="number">{r["recall"]:.0f}%</td>'
            f'<td class="number">{r["dropout_rate"]:.0f}%</td></tr>'
            for r in rows
        )
        parts.append(
            f'<h3>{title}</h3><p class="note">{explanation}</p>'
            "<table><tr><th>Model</th><th>Instelling</th><th>Jaar</th>"
            '<th class="number">Precisie</th><th class="number">Recall</th>'
            f'<th class="number">Uitval in de data</th></tr>{body}</table>'
        )
    return "".join(parts)


def build_overview_html(shared: dict, built_on: date | None = None) -> str:
    """Build the overview page.

    Args:
        shared: Output of build_shared_data().
        built_on: Date shown on the page; defaults to today.

    Returns:
        The page as one HTML document without external files or scripts.
    """
    summary = shared["summary"]
    built_on = built_on or date.today()
    parts = [
        "<h1>Uitnodigingsregel benchmark: gezamenlijk overzicht</h1>",
        f'<p class="meta">Bijgewerkt op {built_on.isoformat()}. Instellingen meten lokaal hoe '
        "goed modellen uitval voorspellen en welke variabelen daaraan bijdragen, en delen "
        "alleen de uitkomst. Er staat geen studentdata op deze pagina.</p>",
    ]
    if summary["n_results"] == 0:
        parts.append(
            "<p>Er zijn nog geen gedeelde resultaten. Zodra een instelling een resultaat "
            "aanlevert, verschijnt het hier.</p>"
        )
    else:
        positive = [r for r in shared["variables"] if r["mean"] > 0][:TOP_VARIABLES]
        parts += [
            '<div class="metrics">',
            f'<div class="metric"><div class="label">Instellingen</div>'
            f'<div class="value">{summary["n_institutions"]}</div></div>',
            f'<div class="metric"><div class="label">Resultaten</div>'
            f'<div class="value">{summary["n_results"]}</div></div>',
            f'<div class="metric"><div class="label">Studenten in de data</div>'
            f'<div class="value">{summary["n_students"]}</div></div>',
            "</div>",
        ]
        if positive:
            items = "".join(
                f"<li><strong>{escape(str(r['variable']))}</strong>: gemiddeld "
                f"{r['mean']:+.1f} procentpunt precisie, getoetst bij {r['n_institutions']} "
                f"instelling{'en' if r['n_institutions'] != 1 else ''}</li>"
                for r in positive
            )
            parts.append(
                f'<h2>Variabelen met de grootste bijdrage</h2><ol class="top">{items}</ol>'
            )
        parts += [
            "<h2>Impact per variabele</h2>",
            _variables_section(shared["variables"]),
            "<h2>Hoe goed voorspellen de modellen</h2>",
            '<p class="note">Als de 20% studenten met de hoogste voorspelde kans worden '
            "uitgenodigd: precisie is het deel van de uitgenodigde studenten dat ook echt "
            "uitviel, recall het deel van alle uitvallers dat daarmee wordt bereikt. Vergelijk "
            "de precisie met het uitvalpercentage in de data: ligt die er dichtbij, dan is het "
            "model niet beter dan willekeurig uitnodigen.</p>",
            _models_section(shared["models"]),
        ]
    parts.append(
        "<h2>Meedoen</h2>"
        f"<p>De benchmark draait op je eigen computer. Installatie en uitleg staan in de "
        f'<a href="{REPO_URL}#readme">README</a>. Gedeelde resultaten staan hier onder de naam '
        "of code die de instelling zelf heeft gekozen.</p>"
    )
    body = "\n".join(parts)
    return (
        '<!doctype html>\n<html lang="nl">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Uitnodigingsregel benchmark: gezamenlijk overzicht</title>\n"
        f"<style>{CSS}{OVERVIEW_CSS}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def build_site(results_dir: str | Path = "results", output_dir: str | Path = "site") -> Path:
    """Write the overview page and the shared data file for the app.

    Args:
        results_dir: Folder with the shared result files.
        output_dir: Folder to write index.html and shared_results.json.

    Returns:
        Path to the written index.html.
    """
    shared = build_shared_data(load_shared_results(results_dir))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SHARED_DATA_FILE).write_text(
        json.dumps(shared, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    index = output_dir / "index.html"
    index.write_text(build_overview_html(shared), encoding="utf-8")
    return index


def main() -> None:
    """Build the overview from the command line."""
    parser = argparse.ArgumentParser(description="Build the collective overview page")
    parser.add_argument("--results-dir", default="results", help="Folder with shared results")
    parser.add_argument("--output-dir", default="site", help="Folder to write the page to")
    args = parser.parse_args()
    print(f"Overzicht geschreven: {build_site(args.results_dir, args.output_dir)}")


if __name__ == "__main__":
    main()
