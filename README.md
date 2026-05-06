# uitnodigingsregel-benchmark

Benchmarking harness for the uitnodigingsregel dropout prediction methodology. Each instelling runs this locally on their own data — no student data leaves the institution. The resulting JSON file can be shared with CEDA for cross-institutional analysis.

## Installation

Requires Python 3.13+ and [uv](https://astral.sh/uv/).

```bash
git clone https://github.com/cedanl/uitnodigingsregel-benchmark.git
cd uitnodigingsregel-benchmark
uv sync
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
