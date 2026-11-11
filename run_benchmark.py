"""Benchmark entrypoint; same as the uitnodigingsregel-benchmark command."""

import sys

from uitnodigingsregel_benchmark.cli import main

if __name__ == "__main__":
    sys.exit(main())
