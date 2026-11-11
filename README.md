# uitnodigingsregel-benchmark

Benchmarking harness for the uitnodigingsregel dropout prediction methodology. Each instelling runs this locally on their own data — no student data leaves the institution. The resulting JSON file can be shared with CEDA for cross-institutional analysis.

## Installation

Requires Python 3.13+ and [uv](https://astral.sh/uv/).

```bash
git clone https://github.com/cedanl/uitnodigingsregel-benchmark.git
cd uitnodigingsregel-benchmark
uv sync
```

On a Mac, XGBoost needs OpenMP, which `uv sync` does not install. Without it the benchmark skips XGBoost and runs the other models. To include XGBoost:

```bash
brew install libomp
```

## Usage

1. Place your CSV in `data/01-raw/user_data/` — one file with a `Dropout` column and any student features
2. Configure `config.yaml` with your `instelling_id` and `dropout_year`
3. Configure algorithms in `models.py` (defaults: Random Forest, Lasso, SVM, Gradient Boosting, XGBoost)
4. Run:

```bash
uv run python run_benchmark.py --train data/01-raw/user_data/data.csv
```

Output: `results/benchmark_{instelling_id}_{dropout_year}.json`

The tool splits your data 80/20 internally and trains all configured models. Share the resulting JSON with CEDA to contribute to the collective benchmark.

To try the tool without your own data, run it on the synthetic demo data:

```bash
uv run python run_benchmark.py --train data/01-raw/demo/synth_data_train.csv
```

## Comparing results across versions

Since `student_signal` 0.2.0 the student number (`Studentnummer`) is no longer used as a predictor. Results made with an earlier version are not comparable with results made with 0.2.0 or later. The JSON file states the version used in `metadata.student_signal_version`; a file without that field was made with an earlier version. Run the benchmark again to compare.

## Adding algorithms

Edit `models.py` — any sklearn-compatible estimator works:

```python
from sklearn.ensemble import GradientBoostingClassifier

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

## Development

```bash
uv run pytest
```

## License

MIT
