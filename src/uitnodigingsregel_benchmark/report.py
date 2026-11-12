"""Build a standalone HTML report of one benchmark result."""

import re
from html import escape

from student_signal.evaluate import get_stoplight_evaluation

# Shown in the app and in the report; ** marks bold text
PRECISION_RECALL_TEXT = (
    "Als je de 20% studenten met de hoogste voorspelde kans uitnodigt: **precisie** is het "
    "deel van de uitgenodigde studenten dat ook echt uitviel, **recall** is het deel van "
    "alle uitvallers dat je daarmee bereikt. Een precisie rond het uitvalpercentage in de "
    "data betekent dat het model niet beter is dan willekeurig uitnodigen."
)
IMPACT_TEXT = (
    "Elke balk is het verschil in precisie tussen het model met en zonder de variabele. "
    "Naar rechts: het model voorspelt beter mét de variabele. Een verschil van enkele "
    "procentpunten kan toeval zijn, zeker bij een kleine groep studenten. Twee variabelen "
    "die hetzelfde zeggen kunnen elk onbelangrijk lijken, omdat de ander het overneemt."
)
QUICK_TEXT = "Dit is een snelle run. Kies 'Volledig' voor een uitkomst om te delen."

COLOR_HELPS = "#2a7f62"
COLOR_HURTS = "#c8785a"

CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, Arial, sans-serif; color: #1a1a1a;
       max-width: 860px; margin: 0 auto; padding: 32px 20px 48px; line-height: 1.5; }
h1 { font-size: 28px; margin: 0 0 4px; }
h2 { font-size: 20px; margin: 36px 0 8px; }
h3 { font-size: 16px; margin: 24px 0 8px; }
.meta, .note { color: #5a5a5a; font-size: 14px; }
.warning { background: #fff3cd; border-left: 4px solid #e0a800; padding: 10px 14px;
           margin: 16px 0; }
.metrics { display: flex; gap: 16px; margin: 16px 0; }
.metric { flex: 1; border: 1px solid #ddd; border-radius: 12px; padding: 12px 16px; }
.metric .value { font-size: 30px; font-weight: 600; }
.metric .label { color: #5a5a5a; font-size: 14px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 15px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #e3e3e3; }
th { font-weight: 600; background: #f6f6f6; }
td.number, th.number { text-align: right; font-variant-numeric: tabular-nums; }
.bars td { border-bottom: none; padding: 3px 8px; }
.bars td.name { width: 30%; text-align: right; color: #333; }
.bars td.value { width: 64px; text-align: right; font-variant-numeric: tabular-nums; }
.track { display: flex; height: 18px; }
.half { flex: 1; display: flex; }
.half.left { justify-content: flex-end; border-right: 1px solid #888; }
.bar { height: 100%; border-radius: 2px; }
details { margin: 8px 0; }
summary { cursor: pointer; font-weight: 600; }
footer { margin-top: 40px; color: #5a5a5a; font-size: 13px; border-top: 1px solid #ddd;
         padding-top: 12px; }
@media print { body { padding: 0; } details { break-inside: avoid; } }
"""


def _html(text: str) -> str:
    """Escape text for HTML and turn **bold** into <strong>."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escape(text))


def split_description(metadata: dict) -> str:
    """Describe in one sentence how the data was split and how many students were used."""
    split = metadata["split"]
    if split["type"] == "cohort":
        cohorts = ", ".join(str(c) for c in split["train_cohorts"])
        how = f"Getraind op cohort {cohorts} en getoetst op cohort {split['test_cohort']}."
    else:
        how = "De data is willekeurig gesplitst in een deel om te trainen en te toetsen."
    return f"{how} Getraind op {metadata['n_train']} studenten, getoetst op {metadata['n_test']}."


def _model_rows(result: dict) -> str:
    models = sorted(
        result["models"].items(), key=lambda item: item[1]["evaluation"]["precision"], reverse=True
    )
    rows = []
    for name, model in models:
        evaluation = model["evaluation"]
        emoji = get_stoplight_evaluation(evaluation["precision"], evaluation["recall"])[0]
        rows.append(
            f"<tr><td>{escape(name)}</td><td>{emoji} {escape(evaluation['status'])}</td>"
            f'<td class="number">{evaluation["precision"]:.0f}%</td>'
            f'<td class="number">{evaluation["recall"]:.0f}%</td></tr>'
        )
    return "\n".join(rows)


def _impact_bars(entries: list[dict]) -> str:
    """Draw the impact per variable as bars to the left and right of zero."""
    measured = [e for e in entries if "precision_diff" in e]
    largest = max((abs(e["precision_diff"]) for e in measured), default=0) or 1
    rows = []
    for entry in measured:
        diff = entry["precision_diff"]
        width = abs(diff) / largest * 100
        left = f'<div class="bar" style="width:{width:.1f}%;background:{COLOR_HURTS}"></div>'
        right = f'<div class="bar" style="width:{width:.1f}%;background:{COLOR_HELPS}"></div>'
        rows.append(
            f'<tr><td class="name">{escape(str(entry["variable"]))}</td>'
            f'<td><div class="track"><div class="half left">{left if diff < 0 else ""}</div>'
            f'<div class="half">{right if diff > 0 else ""}</div></div></td>'
            f'<td class="value">{diff:+.1f}</td></tr>'
        )
    return f'<table class="bars">{"".join(rows)}</table>'


def build_html_report(result: dict) -> str:
    """Build a standalone HTML report of a benchmark result.

    The report needs no internet connection and holds no data of individual students.

    Args:
        result: Complete benchmark result from run_benchmark_on_data().

    Returns:
        The report as one HTML document.
    """
    metadata = result["metadata"]
    best = result["recommendation"]
    title = f"Uitnodigingsregel benchmark: {metadata['instelling_id']}"
    year = f", uitvaljaar {metadata['dropout_year']}" if metadata.get("dropout_year") else ""

    parts = [
        f"<h1>{escape(title)}</h1>",
        f'<p class="meta">Gemaakt op {escape(str(metadata["run_date"]))}{escape(year)}. '
        f"{escape(split_description(metadata))}</p>",
    ]
    if metadata.get("quick_mode"):
        parts.append(f'<p class="warning">{escape(QUICK_TEXT)}</p>')

    parts += [
        "<h2>Uitkomst</h2>",
        f"<p>Best voorspellende model: <strong>{escape(best['model'])}</strong></p>",
        '<div class="metrics">',
        f'<div class="metric"><div class="label">Precisie</div>'
        f'<div class="value">{best["precision_at_20"]:.0f}%</div></div>',
        f'<div class="metric"><div class="label">Recall</div>'
        f'<div class="value">{best["recall_at_20"]:.0f}%</div></div>',
        f'<div class="metric"><div class="label">Uitval in de data</div>'
        f'<div class="value">{metadata["dropout_rate_test"] * 100:.0f}%</div></div>',
        "</div>",
        f'<p class="note">{_html(PRECISION_RECALL_TEXT)}</p>',
        '<table><tr><th>Model</th><th>Oordeel</th><th class="number">Precisie</th>'
        f'<th class="number">Recall</th></tr>{_model_rows(result)}</table>',
    ]

    if metadata.get("impact_measured"):
        parts += [
            "<h2>Impact per variabele</h2>",
            f'<p class="note">{_html(IMPACT_TEXT)} De getallen zijn procentpunten.</p>',
            f"<h3>{escape(best['model'])} (best voorspellend)</h3>",
            _impact_bars(result["models"][best["model"]]["variable_impact"]),
        ]
        for name, model in result["models"].items():
            if name != best["model"]:
                parts.append(
                    f"<details><summary>{escape(name)}</summary>"
                    f"{_impact_bars(model['variable_impact'])}</details>"
                )

    parts.append(
        "<footer>Dit rapport bevat uitkomsten per model, namen van kolommen en aantallen "
        "studenten. Het bevat geen gegevens van individuele studenten. Gemaakt met "
        f"uitnodigingsregel-benchmark en student_signal "
        f"{escape(str(metadata.get('student_signal_version', '')))}.</footer>"
    )
    body = "\n".join(parts)
    return (
        '<!doctype html>\n<html lang="nl">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(title)}</title>\n<style>{CSS}</style>\n</head>\n"
        f"<body>\n{body}\n</body>\n</html>\n"
    )
