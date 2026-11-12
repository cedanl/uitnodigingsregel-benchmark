"""Command line entrypoint (uitnodigingsregel-benchmark command).

Without --train or --demo the command starts the app.
"""

import argparse
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

from uitnodigingsregel_benchmark.dataset import DataValidationError
from uitnodigingsregel_benchmark.runner import run_benchmark
from uitnodigingsregel_benchmark.settings import QUICK_SETTINGS, SettingsError, load_config

DEMO_DATA = Path("data/01-raw/demo/synth_data_train.csv")
DEMO_INSTELLING_ID = "demo"
APP_FILE = Path(__file__).parent / "app" / "main.py"
APP_PORT = 8502


def launch_app(open_browser: bool = True) -> int:
    """Start the Streamlit app and open it in the browser.

    Args:
        open_browser: Whether to open the app in the default browser.

    Returns:
        Exit code of the app.
    """
    url = f"http://localhost:{APP_PORT}"
    print(f"De benchmark-app start op {url}. Stoppen: Ctrl+C in dit venster.")
    if open_browser:

        def _open() -> None:
            time.sleep(2)
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP_FILE),
            "--server.headless=true",
            f"--server.port={APP_PORT}",
            "--browser.gatherUsageStats=false",
            "--server.maxUploadSize=1500",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark from the command line.

    Args:
        argv: Command line arguments; defaults to those of the process.

    Returns:
        Exit code: 0 after a completed run, 1 if the data or settings cannot be used.
        Without --train or --demo: the exit code of the app.
    """
    parser = argparse.ArgumentParser(description="Run uitnodigingsregel benchmark")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--train", help="Path to the CSV with known outcomes")
    source.add_argument("--demo", action="store_true", help="Run on the synthetic demo data")
    parser.add_argument("--config", default="config.yaml", help="Path to the settings file")
    parser.add_argument("--results-dir", default="results", help="Output directory")
    parser.add_argument(
        "--cohort-column",
        help="Column with the cohort; train on earlier cohorts, test on the last one",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fewer models and fewer search rounds, for a first check",
    )
    parser.add_argument(
        "--impact",
        action="store_true",
        help="Measure the impact of each variable by training again without it",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the app without opening the browser",
    )
    args = parser.parse_args(argv)

    if not args.train and not args.demo:
        return launch_app(open_browser=not args.no_browser)

    try:
        config = load_config(args.config)
        if args.cohort_column:
            config["cohort_column"] = args.cohort_column
        if args.impact:
            config["impact"] = True
        if args.quick:
            config.update(QUICK_SETTINGS)
            print("Snelle modus: minder modellen en minder zoekrondes.")
        if args.demo:
            config["instelling_id"] = DEMO_INSTELLING_ID
        train_path = DEMO_DATA if args.demo else args.train
        run_benchmark(train_path, config, args.results_dir, quick_mode=args.quick)
    except (DataValidationError, SettingsError) as error:
        print(f"\nDe benchmark kan niet starten.\n{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
