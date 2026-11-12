"""Where results are shared and how the app gets the collective results."""

import json
import urllib.request

from uitnodigingsregel_benchmark.overview import SHARED_DATA_FILE

OVERVIEW_URL = "https://cedanl.github.io/uitnodigingsregel-benchmark/"
FETCH_TIMEOUT_SECONDS = 5


def fetch_shared_data() -> dict | None:
    """Get the collective results from the overview site.

    Only downloads; nothing about the user's data is sent.

    Returns:
        Output of build_shared_data(), or None without connection or without results.
    """
    try:
        with urllib.request.urlopen(  # noqa: S310 - fixed https address
            OVERVIEW_URL + SHARED_DATA_FILE, timeout=FETCH_TIMEOUT_SECONDS
        ) as response:
            shared = json.load(response)
    except (OSError, ValueError):
        return None
    return shared if shared.get("summary", {}).get("n_results") else None
