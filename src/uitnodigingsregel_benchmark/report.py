"""Build a standalone HTML report of one benchmark result."""

import re
from html import escape

from student_signal.evaluate import get_stoplight_evaluation

from uitnodigingsregel_benchmark.shared_variables import is_own_name

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
LOCAL_NAMES_TEXT = (
    "De uitkomst gebruikt de namen van de gedeelde variabelenlijst. Kolommen die daar niet "
    "in staan hebben een neutrale naam. Deze tabel staat alleen in dit rapport en op je "
    "eigen scherm, niet in het resultaatbestand."
)
SPREAD_TEXT = (
    "De getallen zijn gemiddelden over meerdere splitsingen van de data; achter ± staat hoeveel "
    "ze per splitsing verschilden. Een grijze balk valt binnen die spreiding: het verschil is "
    "niet te onderscheiden van toeval."
)
QUICK_TEXT = "Dit is een snelle run. Kies 'Volledig' voor een uitkomst om te delen."

COLOR_HELPS = "#2a7f62"
COLOR_HURTS = "#c8785a"
COLOR_NOISE = "#9a9a9a"

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
.bars td.value { width: 96px; text-align: right; font-variant-numeric: tabular-nums; }
.track { display: flex; height: 18px; }
.half { flex: 1; display: flex; }
.half.left { justify-content: flex-end; border-right: 1px solid #888; }
.bar { height: 100%; border-radius: 2px; }
tr.reference td { color: #5a5a5a; font-style: italic; }
details { margin: 8px 0; }
summary { cursor: pointer; font-weight: 600; }
footer { margin-top: 40px; color: #5a5a5a; font-size: 13px; border-top: 1px solid #ddd;
         padding-top: 12px; }
@media print { body { padding: 0; } details { break-inside: avoid; } }
"""


def display_name(name: str, local_names: dict[str, str] | None) -> str:
    """Show an own variable with the institution's own name next to its neutral name."""
    if local_names and is_own_name(name) and name in local_names:
        return f"{local_names[name]} ({name})"
    return name


def within_spread(entry: dict) -> bool:
    """Whether the impact of a variable is not distinguishable from chance over the splits."""
    spread = entry.get("precision_diff_sd") or 0
    return spread > 0 and abs(entry["precision_diff"]) <= spread


def close_to_best(result: dict) -> list[str]:
    """Models whose precision differs from the best model's by less than its spread."""
    best = result["recommendation"]["model"]
    evaluation = result["models"][best]["evaluation"]
    spread = evaluation.get("precision_sd") or 0
    return [
        name
        for name, model in result["models"].items()
        if name != best
        and spread > 0
        and evaluation["precision"] - model["evaluation"]["precision"] <= spread
    ]


def with_spread(value: float, spread: float | None, unit: str = "%") -> str:
    """Format a value with its spread over the splits, if there is one."""
    text = f"{value:.0f}{unit}"
    return f"{text} ± {spread:.0f}" if spread else text


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
        if split.get("repeats", 1) > 1:
            how += f" Dat is {split['repeats']} keer herhaald met een andere splitsing."
    return f"{how} Getraind op {metadata['n_train']} studenten, getoetst op {metadata['n_test']}."


REFERENCE_TEXT = (
    "Ter vergelijking twee regels zonder model: willekeurig uitnodigen, en de 20% studenten "
    "uitnodigen met de hoogste waarde op één variabele. Haalt een model niet meer dan zo'n "
    "regel, dan levert het weinig op."
)
REFERENCE_LABEL = "Referentie zonder model"


def reference_label(key: str, reference: dict, local_names: dict[str, str] | None) -> str:
    """Name of a reference rule as shown to the user."""
    if key == "random":
        return "Willekeurig uitnodigen"
    return f"Uitnodigen op {display_name(reference['variable'], local_names)}"


def _reference_rows(result: dict, local_names: dict[str, str] | None) -> str:
    rows = []
    for key, reference in result.get("references", {}).items():
        precision = with_spread(reference["precision"], reference.get("precision_sd"))
        recall = with_spread(reference["recall"], reference.get("recall_sd"))
        rows.append(
            f'<tr class="reference"><td>{escape(reference_label(key, reference, local_names))}'
            f"</td><td>{REFERENCE_LABEL}</td>"
            f'<td class="number">{precision}</td><td class="number">{recall}</td></tr>'
        )
    return "\n".join(rows)


def comparison_section(comparison: dict, local_names: dict[str, str] | None) -> str:
    """The own result next to the shared results, for the report."""
    parts = [
        "<h2>Vergelijking met andere instellingen</h2>",
        f'<p class="note">Gedeelde resultaten van {comparison["n_institutions"]} instellingen '
        "uit het gezamenlijke overzicht.</p>",
    ]
    others = comparison["other_precisions"]
    if others:
        parts.append(
            f"<p>{escape(comparison['model'])} haalt bij jou een precisie van "
            f"{comparison['own_precision']:.0f}%. Bij {len(others)} gedeelde resultaten met "
            f"dezelfde soort splitsing ligt die tussen {min(others):.0f}% en "
            f"{max(others):.0f}%.</p>"
        )
    if comparison["variables"]:
        rows = "".join(
            f"<tr><td>{escape(display_name(row['variable'], local_names))}</td>"
            f'<td class="number">{row["own"]:+.1f}</td>'
            f'<td class="number">{row["shared_mean"]:+.1f}</td>'
            f'<td class="number">{row["n_institutions"]}</td></tr>'
            for row in comparison["variables"]
        )
        parts.append(
            '<table><tr><th>Variabele</th><th class="number">Impact bij jou</th>'
            '<th class="number">Gemiddeld bij anderen</th>'
            f'<th class="number">Instellingen</th></tr>{rows}</table>'
        )
    return "\n".join(parts)


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
            f'<td class="number">'
            f"{with_spread(evaluation['precision'], evaluation.get('precision_sd'))}</td>"
            f'<td class="number">{with_spread(evaluation["recall"], evaluation.get("recall_sd"))}'
            "</td></tr>"
        )
    return "\n".join(rows)


def _impact_bars(entries: list[dict], local_names: dict[str, str] | None = None) -> str:
    """Draw the impact per variable as bars to the left and right of zero."""
    measured = [e for e in entries if "precision_diff" in e]
    largest = max((abs(e["precision_diff"]) for e in measured), default=0) or 1
    rows = []
    for entry in measured:
        diff = entry["precision_diff"]
        width = abs(diff) / largest * 100
        noise = within_spread(entry)
        left_color = COLOR_NOISE if noise else COLOR_HURTS
        right_color = COLOR_NOISE if noise else COLOR_HELPS
        left = f'<div class="bar" style="width:{width:.1f}%;background:{left_color}"></div>'
        right = f'<div class="bar" style="width:{width:.1f}%;background:{right_color}"></div>'
        spread = entry.get("precision_diff_sd")
        value = f"{diff:+.1f}" + (f" ± {spread:.1f}" if spread else "")
        rows.append(
            f'<tr><td class="name">{escape(display_name(entry["variable"], local_names))}</td>'
            f'<td><div class="track"><div class="half left">{left if diff < 0 else ""}</div>'
            f'<div class="half">{right if diff > 0 else ""}</div></div></td>'
            f'<td class="value">{value}</td></tr>'
        )
    return f'<table class="bars">{"".join(rows)}</table>'


def _repeated(metadata: dict) -> bool:
    return metadata["split"].get("repeats", 1) > 1


def _close_note(result: dict) -> str:
    close = close_to_best(result)
    if not close:
        return ""
    return f". Het verschil met {', '.join(close)} valt binnen de spreiding."


def build_html_report(
    result: dict,
    local_names: dict[str, str] | None = None,
    comparison: dict | None = None,
) -> str:
    """Build a standalone HTML report of a benchmark result.

    The report needs no internet connection and holds no data of individual students.

    Args:
        result: Complete benchmark result from run_benchmark_on_data().
        local_names: Names in the result mapped to the institution's own names; shown
            in the report so the reader knows which variable is meant.
        comparison: Output of compare_with_shared(), or None without shared results.

    Returns:
        The report as one HTML document.
    """
    metadata = result["metadata"]
    best = result["recommendation"]
    best_eval = result["models"][best["model"]]["evaluation"]
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
        f"<p>Best voorspellende model: <strong>{escape(best['model'])}</strong>"
        f"{escape(_close_note(result))}</p>",
        '<div class="metrics">',
        f'<div class="metric"><div class="label">Precisie</div>'
        f'<div class="value">{with_spread(best["precision_at_20"], best_eval.get("precision_sd"))}'
        "</div></div>",
        f'<div class="metric"><div class="label">Recall</div>'
        f'<div class="value">{with_spread(best["recall_at_20"], best_eval.get("recall_sd"))}'
        "</div></div>",
        f'<div class="metric"><div class="label">Uitval in de data</div>'
        f'<div class="value">{metadata["dropout_rate_test"] * 100:.0f}%</div></div>',
        "</div>",
        f'<p class="note">{_html(PRECISION_RECALL_TEXT)}</p>',
        '<table><tr><th>Model</th><th>Oordeel</th><th class="number">Precisie</th>'
        f'<th class="number">Recall</th></tr>{_model_rows(result)}'
        f"{_reference_rows(result, local_names)}</table>",
        *([f'<p class="note">{escape(REFERENCE_TEXT)}</p>'] if result.get("references") else []),
    ]

    if metadata.get("impact_measured"):
        parts += [
            "<h2>Impact per variabele</h2>",
            f'<p class="note">{_html(IMPACT_TEXT)} De getallen zijn procentpunten.</p>',
            *([f'<p class="note">{_html(SPREAD_TEXT)}</p>'] if _repeated(metadata) else []),
            f"<h3>{escape(best['model'])} (best voorspellend)</h3>",
            _impact_bars(result["models"][best["model"]]["variable_impact"], local_names),
        ]
        for name, model in result["models"].items():
            if name != best["model"]:
                parts.append(
                    f"<details><summary>{escape(name)}</summary>"
                    f"{_impact_bars(model['variable_impact'], local_names)}</details>"
                )

    if comparison:
        parts.append(comparison_section(comparison, local_names))

    if local_names:
        rows = "".join(
            f"<tr><td>{escape(public)}</td><td>{escape(own)}</td></tr>"
            for public, own in local_names.items()
        )
        parts += [
            "<h2>Namen van je eigen kolommen</h2>",
            f'<p class="note">{escape(LOCAL_NAMES_TEXT)}</p>',
            f"<table><tr><th>Naam in de uitkomst</th><th>Je eigen kolom of groep</th></tr>"
            f"{rows}</table>",
        ]

    parts.append(
        "<footer>Dit rapport bevat uitkomsten per model, namen van variabelen en aantallen "
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
