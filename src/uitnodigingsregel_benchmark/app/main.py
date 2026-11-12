"""Streamlit app: run the benchmark without commands or settings files."""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from student_signal.evaluate import get_stoplight_evaluation

from uitnodigingsregel_benchmark import sharing
from uitnodigingsregel_benchmark.app.styles import APP_CSS
from uitnodigingsregel_benchmark.check import ResultError, check_result
from uitnodigingsregel_benchmark.dataset import (
    DataValidationError,
    read_student_data,
    validate_student_data,
)
from uitnodigingsregel_benchmark.detect import (
    detect_cohort_column,
    detect_dropout_column,
    detect_id_column,
    suggest_variable_groups,
)
from uitnodigingsregel_benchmark.export import sanitize, write_result
from uitnodigingsregel_benchmark.models import MODELS
from uitnodigingsregel_benchmark.report import (
    IMPACT_TEXT,
    LOCAL_NAMES_TEXT,
    PRECISION_RECALL_TEXT,
    QUICK_TEXT,
    build_html_report,
    display_name,
    split_description,
)
from uitnodigingsregel_benchmark.runner import run_benchmark_on_data
from uitnodigingsregel_benchmark.settings import QUICK_SETTINGS, SettingsError, load_config
from uitnodigingsregel_benchmark.shared_variables import (
    SHARED_NAMES,
    load_mapping,
    match_columns,
    save_mapping,
)

DEMO_DATA = Path("data/01-raw/demo/synth_data_train.csv")
RESULTS_DIR = Path("results")
NONE = "(geen)"
QUICK = "Snel"
FULL = "Volledig"
OWN = "(eigen variabele, niet in de lijst)"

st.set_page_config(page_title="Uitnodigingsregel benchmark", page_icon="📊", layout="centered")
st.markdown(APP_CSS, unsafe_allow_html=True)

for key in (
    "data",
    "data_label",
    "upload_seen",
    "result",
    "result_path",
    "shared",
    "local_names",
    "warnings",
):
    st.session_state.setdefault(key, None)
st.session_state.setdefault("own_groups", {})


def _use_data(source: bytes | Path, label: str) -> None:
    """Read the chosen file and keep it for the next steps; show the problem if it fails."""
    try:
        st.session_state.data = read_student_data(source)
        st.session_state.data_label = label
    except DataValidationError as error:
        st.session_state.data = None
        st.error(str(error))
    st.session_state.result = None
    st.session_state.own_groups = {}


def _select(label: str, options: list[str], detected: str | None, help_text: str) -> str:
    index = options.index(detected) if detected in options else 0
    return st.selectbox(label, options, index=index, help=help_text)


def _result_table(result: dict) -> pd.DataFrame:
    rows = []
    for name, model in result["models"].items():
        evaluation = model["evaluation"]
        emoji = get_stoplight_evaluation(evaluation["precision"], evaluation["recall"])[0]
        rows.append(
            {
                "Model": name,
                "Oordeel": f"{emoji} {evaluation['status']}",
                "Precisie (%)": evaluation["precision"],
                "Recall (%)": evaluation["recall"],
            }
        )
    return pd.DataFrame(rows).sort_values("Precisie (%)", ascending=False).reset_index(drop=True)


def _impact_chart(entries: list[dict], local_names: dict[str, str]) -> alt.Chart:
    data = pd.DataFrame([e for e in entries if "precision_diff" in e])
    data["variable"] = data["variable"].apply(lambda name: display_name(name, local_names))
    data["Effect"] = data["precision_diff"].apply(
        lambda diff: "Helpt" if diff > 0 else ("Geen verschil" if diff == 0 else "Helpt niet")
    )
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("precision_diff:Q", title="Verschil in precisie met en zonder (procentpunten)"),
            y=alt.Y(
                "variable:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelLimit=260, labelOverlap=False),
            ),
            color=alt.Color(
                "Effect:N",
                scale=alt.Scale(
                    domain=["Helpt", "Geen verschil", "Helpt niet"],
                    range=["#2a7f62", "#9a9a9a", "#c8785a"],
                ),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=[
                alt.Tooltip("variable:N", title="Variabele"),
                alt.Tooltip("precision_diff:Q", title="Verschil precisie"),
                alt.Tooltip("recall_diff:Q", title="Verschil recall"),
                alt.Tooltip("precision_without:Q", title="Precisie zonder"),
            ],
        )
        .properties(height=alt.Step(26), padding={"left": 5, "right": 30, "top": 5, "bottom": 5})
    )


st.title("Uitnodigingsregel benchmark")
st.write(
    "Meet hoe goed modellen uitval voorspellen op je eigen data en welke variabelen "
    "daaraan bijdragen. Alles draait op deze computer; er gaat geen studentdata naar buiten."
)

# ── Stap 1: data ──────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("1. Kies je data")
    st.write(
        "Een CSV-bestand met één rij per student, een kolom die aangeeft of de student is "
        "uitgevallen (1) of niet (0), en de kenmerken van de student."
    )
    uploaded = st.file_uploader("CSV-bestand", type=["csv"], label_visibility="collapsed")
    if uploaded is not None and st.session_state.upload_seen != uploaded.file_id:
        st.session_state.upload_seen = uploaded.file_id
        _use_data(uploaded.getvalue(), uploaded.name)
    if st.button("Of probeer het met de demo-data", disabled=not DEMO_DATA.exists()):
        _use_data(DEMO_DATA, "demo-data")

data = st.session_state.data
if data is None:
    st.stop()

# ── Stap 2: kolommen ──────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("2. Controleer de kolommen")
    st.caption(
        f"{st.session_state.data_label}: {len(data)} studenten, {len(data.columns)} kolommen."
    )
    columns = list(data.columns)
    dropout_col = _select(
        "Kolom met uitval",
        columns,
        detect_dropout_column(data),
        "De kolom met 1 voor een uitgevallen student en 0 voor een student die bleef.",
    )
    id_choice = _select(
        "Kolom met studentnummer",
        [NONE, *columns],
        detect_id_column(data),
        "Wordt niet gebruikt om te voorspellen. Mag ontbreken.",
    )
    cohort_choice = _select(
        "Kolom met cohort of startjaar",
        [NONE, *columns],
        detect_cohort_column(data),
        "Met minstens twee cohorten traint de benchmark op de eerdere cohorten en toetst "
        "op het laatste. Zonder deze kolom wordt de data willekeurig gesplitst.",
    )
    id_col = None if id_choice == NONE else id_choice
    cohort_col = None if cohort_choice == NONE else cohort_choice

    config = load_config()
    data_ok = True
    try:
        validate_student_data(data, dropout_col, id_col or "", config["cv_folds"])
    except DataValidationError as error:
        data_ok = False
        st.error(str(error))

    variable_groups: dict[str, list[str]] = {}
    if data_ok:
        n_dropout = int(data[dropout_col].sum())
        st.caption(f"{n_dropout} van de {len(data)} studenten zijn uitgevallen.")
        suggestions = suggest_variable_groups(
            data, [c for c in (dropout_col, id_col, cohort_col) if c]
        )
        if suggestions:
            st.write(
                "Deze kolommen lijken samen één variabele te vormen. "
                "Aangevinkte groepen tellen als één variabele bij het meten van de impact."
            )
            for name, members in suggestions.items():
                label = f"**{name}**: {len(members)} kolommen ({', '.join(members[:3])}, …)"
                if st.checkbox(label, value=True, key=f"group_{name}"):
                    variable_groups[name] = members

        own_groups = st.session_state.own_groups
        for name, members in own_groups.items():
            st.write(f"**{name}**: {', '.join(members)}")
        with st.expander("Zelf kolommen samenvoegen tot één variabele"):
            st.write(
                "Voor een variabele die over meerdere kolommen is verdeeld zonder gedeeld "
                "begin van de naam, bijvoorbeeld een kolom per opleidingsrichting."
            )
            taken = {
                c for members in [*variable_groups.values(), *own_groups.values()] for c in members
            }
            free = [
                c for c in columns if c not in taken and c not in (dropout_col, id_col, cohort_col)
            ]
            group_name = st.text_input("Naam van de variabele", placeholder="Richting")
            group_columns = st.multiselect("Kolommen", free)
            if st.button(
                "Voeg samen", disabled=not (group_name.strip() and len(group_columns) > 1)
            ):
                own_groups[group_name.strip()] = group_columns
                st.rerun()
            if own_groups and st.button("Wis eigen groepen"):
                st.session_state.own_groups = {}
                st.rerun()
        variable_groups.update(own_groups)

    column_mapping: dict[str, str | None] = {}
    if data_ok:
        features = [c for c in columns if c not in (dropout_col, id_col, cohort_col)]
        saved = load_mapping()
        try:
            linked = match_columns(features, saved)
        except SettingsError:
            # A mapping saved for other data; start from the automatic recognition
            saved = {}
            linked = match_columns(features)
        n_linked = sum(1 for name in linked.values() if name)
        st.write(
            f"**Gedeelde variabelen:** {n_linked} van de {len(features)} kolommen horen bij "
            "een variabele uit de gedeelde lijst. Het resultaatbestand gebruikt alleen die "
            "namen. De andere kolommen tellen gewoon mee, maar heten in het resultaatbestand "
            "eigen_variabele_1, eigen_variabele_2, enzovoort."
        )
        with st.expander("Kolommen koppelen aan de gedeelde lijst"):
            st.write(
                "Heet een kolom anders dan in de gedeelde lijst maar bedoelt hij hetzelfde, "
                "kies dan de gedeelde variabele. Je keuze wordt op deze computer bewaard in "
                "koppeling.json en niet gedeeld."
            )
            options = [OWN, *SHARED_NAMES]
            for column in features:
                if linked[column] and column not in saved:
                    continue
                choice = st.selectbox(
                    column,
                    options,
                    index=options.index(linked[column]) if linked[column] else 0,
                    key=f"map_{column}",
                )
                column_mapping[column] = None if choice == OWN else choice
        try:
            match_columns(features, column_mapping)
        except SettingsError as error:
            data_ok = False
            st.error(str(error))

# ── Stap 3: draaien ───────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("3. Draai de benchmark")
    instelling_id = st.text_input(
        "Naam of code van je instelling",
        value=config["instelling_id"],
        help="Komt in het resultaatbestand. Gebruik een code als je niet herkenbaar wilt zijn.",
    )
    year = st.text_input(
        "Jaar waarover de uitval gaat (optioneel)",
        value=str(config["dropout_year"] or ""),
        help="Komt in de naam van het resultaatbestand. Bij een cohortkolom volgt het uit de data.",
    )
    mode = st.radio(
        "Hoe uitgebreid?",
        [QUICK, FULL],
        captions=[
            f"{len(QUICK_SETTINGS['models'])} modellen, klaar in ongeveer een minuut. "
            "Om te kijken of alles werkt.",
            f"Alle {len(MODELS)} modellen. Gebruik dit voor een uitkomst om te delen.",
        ],
        horizontal=True,
    )
    measure_impact = st.checkbox(
        "Meet de impact per variabele",
        value=True,
        help="Traint elk model opnieuw zonder één variabele en vergelijkt de uitkomst. "
        "Duurt langer bij veel variabelen.",
    )
    start = st.button("Start benchmark", type="primary", disabled=not data_ok)

if start:
    config.update(
        {
            "instelling_id": instelling_id.strip().replace(" ", "-") or "anoniem",
            "dropout_year": year.strip() or None,
            "dropout_column": dropout_col,
            "id_column": id_col or "",
            "cohort_column": cohort_col,
            "impact": measure_impact,
            "variable_groups": variable_groups or None,
            "column_mapping": column_mapping,
        }
    )
    save_mapping(column_mapping)
    if mode == QUICK:
        config.update(QUICK_SETTINGS)
    progress = st.progress(0.0, text="Starten")
    try:
        result, local_names = run_benchmark_on_data(
            data,
            config,
            quick_mode=mode == QUICK,
            on_progress=lambda share, message: progress.progress(share, text=message),
        )
    except (DataValidationError, SettingsError) as error:
        progress.empty()
        st.error(str(error))
    else:
        progress.empty()
        try:
            st.session_state.warnings = check_result(result)
            st.session_state.result_path = write_result(result, RESULTS_DIR)
        except ResultError as error:
            st.session_state.result = None
            st.error(f"Het resultaatbestand kan niet worden gemaakt. {error}")
        else:
            st.session_state.result = result
            st.session_state.local_names = local_names
            st.session_state.shared = sharing.fetch_shared_data()
            st.session_state.result_path.with_suffix(".html").write_text(
                build_html_report(result, local_names), encoding="utf-8"
            )

result = st.session_state.result
if result is None:
    st.stop()

# ── Uitkomst ──────────────────────────────────────────────────────────────────
metadata = result["metadata"]
best = result["recommendation"]
local_names = st.session_state.local_names
with st.container(border=True):
    st.subheader("Uitkomst")
    st.caption(split_description(metadata))
    if metadata["quick_mode"]:
        st.info(QUICK_TEXT)
    for warning in st.session_state.warnings or []:
        st.warning(warning)

    st.write(f"Best voorspellende model: **{best['model']}**")
    left, middle, right = st.columns(3)
    left.metric("Precisie", f"{best['precision_at_20']:.0f}%")
    middle.metric("Recall", f"{best['recall_at_20']:.0f}%")
    right.metric("Uitval in de data", f"{metadata['dropout_rate_test'] * 100:.0f}%")
    st.caption(PRECISION_RECALL_TEXT)
    st.dataframe(_result_table(result), hide_index=True, width="stretch")

if metadata["impact_measured"]:
    with st.container(border=True):
        st.subheader("Impact per variabele")
        model_names = list(result["models"])
        chosen = st.selectbox("Model", model_names, index=model_names.index(best["model"]))
        st.altair_chart(
            _impact_chart(result["models"][chosen]["variable_impact"], local_names),
            width="stretch",
        )
        st.caption(IMPACT_TEXT)
        if local_names:
            with st.expander("Namen van je eigen kolommen in de uitkomst"):
                st.caption(LOCAL_NAMES_TEXT)
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"Naam in de uitkomst": public, "Je eigen kolom of groep": own}
                            for public, own in local_names.items()
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )

with st.container(border=True):
    st.subheader("Rapport en resultaatbestand")
    result_path = st.session_state.result_path
    st.write(
        "Het **rapport** is één HTML-bestand met de uitkomst en de grafieken, om te bewaren "
        "of door te sturen naar collega's. Het opent in elke browser, ook zonder internet."
    )
    st.download_button(
        "Download rapport",
        data=build_html_report(result, local_names),
        file_name=result_path.with_suffix(".html").name,
        mime="text/html",
    )
    st.write(
        "Het **resultaatbestand** bevat de uitkomsten per model met de namen uit de gedeelde "
        "variabelenlijst, en aantallen studenten. Het bevat geen namen van je eigen kolommen "
        "en geen gegevens van individuele studenten."
    )
    st.download_button(
        "Download resultaatbestand",
        data=json.dumps(sanitize(result), indent=2, ensure_ascii=False),
        file_name=result_path.name,
        mime="application/json",
    )
    st.caption(f"Beide bestanden staan ook in de map {result_path.parent}.")

with st.container(border=True):
    st.subheader("Delen voor de gezamenlijke benchmark")
    if metadata["quick_mode"]:
        st.write(
            "Een snelle run is niet bedoeld om te delen. Draai de benchmark met 'Volledig' "
            "voor een uitkomst die meetelt in het gezamenlijke overzicht."
        )
    else:
        st.write(
            f"Met het resultaatbestand telt je uitkomst mee in het "
            f"[gezamenlijke overzicht]({sharing.OVERVIEW_URL}). Dat overzicht is openbaar: je "
            f"uitkomst staat er onder de naam **{metadata['instelling_id']}**. Wil je niet "
            "herkenbaar zijn, draai dan opnieuw met een code als naam."
        )
        if sharing.UPLOAD_URL:
            st.write(
                "Zet het resultaatbestand in de uploadmap. CEDA controleert het bestand en "
                "neemt het daarna op in het overzicht. De app verstuurt zelf niets."
            )
            st.link_button("Open de uploadmap", sharing.UPLOAD_URL)
        else:
            st.write(
                "Stuur het resultaatbestand naar je contactpersoon bij CEDA. CEDA controleert "
                "het bestand en neemt het daarna op in het overzicht. De app verstuurt zelf niets."
            )

with st.container(border=True):
    st.subheader("Vergelijk met andere instellingen")
    shared = st.session_state.shared
    if shared is None:
        st.write(
            "Er zijn nog geen gedeelde resultaten om mee te vergelijken, of er is geen "
            f"internetverbinding. Het overzicht staat op {sharing.OVERVIEW_URL}"
        )
    else:
        summary = shared["summary"]
        st.caption(
            f"Gedeelde resultaten van {summary['n_institutions']} instellingen, opgehaald uit "
            "het gezamenlijke overzicht. Er is niets over je eigen data verstuurd."
        )
        split_type = metadata["split"]["type"]
        others = [
            row["precision"]
            for row in shared["models"]
            if row["model"] == best["model"] and row["split"] == split_type
        ]
        if others:
            st.write(
                f"**{best['model']}** haalt bij jou een precisie van "
                f"{best['precision_at_20']:.0f}%. Bij {len(others)} gedeelde resultaten met "
                f"dezelfde soort splitsing ligt die tussen {min(others):.0f}% en "
                f"{max(others):.0f}%."
            )
        if metadata["impact_measured"]:
            own = {
                entry["variable"]: entry["precision_diff"]
                for entry in result["models"][best["model"]]["variable_impact"]
                if "precision_diff" in entry
            }
            rows = [
                {
                    "Variabele": row["variable"],
                    "Jouw data": own[row["variable"]],
                    "Gemiddeld bij anderen": row["mean"],
                    "Aantal instellingen": row["n_institutions"],
                }
                for row in shared["variables"]
                if row["variable"] in own
            ]
            if rows:
                st.write("Impact per variabele, in procentpunten precisie:")
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            else:
                st.write(
                    "Geen van je variabelen komt onder dezelfde naam voor in de gedeelde "
                    "resultaten."
                )
