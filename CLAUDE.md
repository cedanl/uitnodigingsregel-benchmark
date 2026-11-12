# uitnodigingsregel-benchmark

## Overview
Benchmarking harness for the uitnodigingsregel methodology. Runs locally at each MBO instelling, produces a standardized JSON results file that CEDA aggregates across institutions to derive minimum feature recommendations and algorithm insights.

## Standards
Follow CEDA technical standards: https://github.com/cedanl/.github/tree/main/standards/README.md

## Tech Stack
Python 3.13, student-signal, scikit-learn, pandas, Streamlit.

## Project Structure
```
├── config.yaml                      ← all user settings (columns, models, search rounds, processes)
├── run_benchmark.py                 ← same as the uitnodigingsregel-benchmark command
├── src/uitnodigingsregel_benchmark/
│   ├── cli.py                       ← command entrypoint: starts the app, or runs with --train/--demo
│   ├── app/main.py                  ← Streamlit app: the primary interface for institutions
│   ├── app/styles.py                ← CSS of the app
│   ├── detect.py                    ← recognises dropout, id and cohort column and variable groups
│   ├── settings.py                  ← defaults, config.yaml and quick mode
│   ├── dataset.py                   ← reads the CSV and checks it before training
│   ├── split.py                     ← train/test split: by cohort when possible, else random
│   ├── models.py                    ← algorithm registry
│   ├── runner.py                    ← trains the models and assembles the result (used by app and CLI)
│   ├── features.py                  ← full feature importance extraction
│   ├── impact.py                    ← impact per variable: retrain without it and compare (--impact)
│   └── export.py                    ← builds standardized JSON output
├── analysis/aggregate.qmd           ← CEDA-side Quarto aggregation doc
├── results/                         ← JSON results files land here
│   └── voorbeeld/                   ← example result, made on the demo data
├── data/01-raw/demo/                ← synthetic demo data
└── data/01-raw/user_data/           ← instelling puts their CSV here (gitignored)
```

## How to Run
```bash
uv sync
uv run uitnodigingsregel-benchmark                                             # app (primary interface)
uv run uitnodigingsregel-benchmark --demo --quick                               # demo data, seconds
uv run uitnodigingsregel-benchmark --train data/01-raw/user_data/data.csv      # own data
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

A run never overwrites a tracked file: `--demo` writes `results/benchmark_demo_*.json` (gitignored) and the example lives in `results/voorbeeld/`.

The app is the primary interface: a feature for institutions must work there, not only on the command line. App and command line both call `run_benchmark_on_data` in `runner.py`. `tests/test_app.py` clicks through the app on the demo data.

## Data
Input: one CSV with a `Dropout` column (0/1) and any student features. With `cohort_column` set (in `config.yaml` or `--cohort-column`) and at least two cohorts in the data, the tool trains on the earlier cohorts and tests on the last one. Otherwise it splits 80/20 at random. `metadata.split` in the result states which split was used.
Output: `results/benchmark_{instelling_id}_{dropout_year}.json`
