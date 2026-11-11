# uitnodigingsregel-benchmark

Benchmarking harness for the uitnodigingsregel dropout prediction methodology. Each instelling runs this locally on their own data — no student data leaves the institution. The resulting JSON file can be shared with CEDA for cross-institutional analysis.

## Installation

Requires Python 3.13+ and [uv](https://astral.sh/uv/).

```bash
git clone https://github.com/cedanl/uitnodigingsregel-benchmark.git
cd uitnodigingsregel-benchmark
uv sync
```

On a Mac, XGBoost needs OpenMP, which `uv sync` does not install. Without it the benchmark skips XGBoost and runs the other models, including Hist Gradient Boosting: the same kind of model, from scikit-learn. To include XGBoost:

```bash
brew install libomp
```

## Usage

First try the benchmark on the synthetic demo data. This needs no data of your own and takes a few seconds:

```bash
uv run uitnodigingsregel-benchmark --demo --quick
```

Then run it on your own data:

1. Place your CSV in `data/01-raw/user_data/`: one file with a `Dropout` column (1 = dropped out, 0 = not) and the student features. A column with student numbers is not needed.
2. Set your `instelling_id` and `dropout_year` in `config.yaml`.
3. Run:

```bash
uv run uitnodigingsregel-benchmark --train data/01-raw/user_data/data.csv
```

Output: `results/benchmark_{instelling_id}_{dropout_year}.json`. Share this file with CEDA to contribute to the collective benchmark. `results/voorbeeld/` holds an example of the file, made on the demo data.

Before training, the benchmark checks the data. If something is missing or wrong, it stops and states what to change.

| Option | What it does |
|---|---|
| `--train PATH` | Run on your own CSV |
| `--demo` | Run on the synthetic demo data; writes `results/benchmark_demo_*.json` |
| `--quick` | Three models and fewer search rounds, for a first check. The result file states `quick_mode: true`; do not share a quick result |
| `--cohort-column NAME` | Test on a later cohort (see below) |
| `--config PATH` | Use another settings file than `config.yaml` |
| `--results-dir PATH` | Write the result to another folder |

## Settings

All settings are in `config.yaml`; each one is explained there. No Python files need to be changed.

| Setting | Meaning | Default |
|---|---|---|
| `instelling_id` | Short code for your institution, used in the file name | `anoniem` |
| `dropout_year`, `training_years` | Years the data covers | none |
| `dropout_column` | Name of the dropout column | `Dropout` |
| `id_column` | Name of the student number column; may be absent | `Studentnummer` |
| `cohort_column` | Name of the cohort column (see below) | none |
| `models` | Models to run | all |
| `n_iter`, `cv_folds` | Search rounds per model and folds per round; lower is faster | 20, 5 |
| `test_size` | Share of the data used for testing in a random split | 0.2 |
| `n_jobs` | Processes to run at once; set to 2 if the computer runs out of memory | all cores |

## Testing on a later cohort

By default the data is split at random: 80% to train the models, 20% to test them. In practice an institution trains on earlier cohorts and predicts the next one, and a random split often gives a more favourable result than that.

If your data has a column with the cohort (for example the start year) and holds at least two cohorts, the benchmark can train on the earlier cohorts and test on the last one. Set the column name in `config.yaml`:

```yaml
cohort_column: "Startjaar"
```

or pass it on the command line with `--cohort-column Startjaar`. The cohort column is not used as a predictor, and `dropout_year` and `training_years` in the result then follow from the data.

This is optional. Without a cohort column, or with only one cohort in the data, the benchmark uses the random split. The result file states which split was used in `metadata.split`, and the aggregation only compares results with the same kind of split.

## Comparing results across versions

Since `student_signal` 0.2.0 the student number (`Studentnummer`) is no longer used as a predictor. Results made with an earlier version are not comparable with results made with 0.2.0 or later. The JSON file states the version used in `metadata.student_signal_version`; a file without that field was made with an earlier version. Run the benchmark again to compare.

## Adding algorithms

The available models are listed in `src/uitnodigingsregel_benchmark/models.py`. Any sklearn-compatible estimator can be added there:

```python
MODELS = {
    ...
    "Gradient Boosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "use_scaled": False,
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
        },
    },
}
```

Results with other models than the standard ones are for local use; the collective benchmark compares the standard models.

## Development

```bash
uv run pytest
```

## License

MIT
