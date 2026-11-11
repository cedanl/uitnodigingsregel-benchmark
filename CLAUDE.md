# uitnodigingsregel-benchmark

## Overview
Benchmarking harness for the uitnodigingsregel methodology. Runs locally at each MBO instelling, produces a standardized JSON results file that CEDA aggregates across institutions to derive minimum feature recommendations and algorithm insights.

## Standards
Follow CEDA technical standards: https://github.com/cedanl/.github/tree/main/standards/README.md

## Tech Stack
Python 3.13, student-signal, scikit-learn, pandas.

## Project Structure
```
├── models.py                        ← user-configurable algorithm registry
├── run_benchmark.py                 ← entrypoint: reads CSV, writes JSON
├── src/uitnodigingsregel_benchmark/
│   ├── split.py                     ← train/test split: by cohort when possible, else random
│   ├── features.py                  ← full feature importance extraction
│   └── export.py                    ← builds standardized JSON output
├── analysis/aggregate.qmd           ← CEDA-side Quarto aggregation doc
├── results/                         ← JSON results files land here
└── data/01-raw/user_data/           ← instelling puts their CSV here (gitignored)
```

## How to Run
```bash
uv sync
uv run python run_benchmark.py --train data/01-raw/user_data/data.csv
```

## Data
Input: one CSV with a `Dropout` column (0/1) and any student features. With `cohort_column` set (in `config.yaml` or `--cohort-column`) and at least two cohorts in the data, the tool trains on the earlier cohorts and tests on the last one. Otherwise it splits 80/20 at random. `metadata.split` in the result states which split was used.
Output: `results/benchmark_{instelling_id}_{dropout_year}.json`
