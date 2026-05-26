import time
import requests
import pandas as pd


def fetch_opensky_arrivals(
    airport="ESSA",
    begin_unix=None,
    end_unix=None,
    access_token=None,
    max_retries=3,
):
    """
    Fetch airport arrivals from the OpenSky REST API.

    This function uses retries because OpenSky public API requests can timeout
    or be throttled, especially from cloud-hosted apps.
    """

    if begin_unix is None or end_unix is None:
        raise ValueError("begin_unix and end_unix are required.")

    if begin_unix >= end_unix:
        raise ValueError("begin_unix must be smaller than end_unix.")

    max_interval_seconds = 2 * 24 * 60 * 60

    if end_unix - begin_unix > max_interval_seconds:
        raise ValueError("OpenSky arrivals endpoint supports intervals up to two days.")

    url = "https://opensky-network.org/api/flights/arrival"

    params = {
        "airport": airport,
        "begin": int(begin_unix),
        "end": int(end_unix),
    }

    headers = {
        "User-Agent": "atco-workload-green-arrivals-dashboard/0.1"
    }

    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=60,
            )

            if response.status_code == 404:
                return pd.DataFrame()

            if response.status_code != 200:
                raise RuntimeError(
                    f"OpenSky request failed with status {response.status_code}: {response.text}"
                )

            data = response.json()

            if not data:
                return pd.DataFrame()

            return pd.DataFrame(data)

        except Exception as error:
            last_error = error
            time.sleep(2 * attempt)

    raise RuntimeError(f"OpenSky request failed after {max_retries} attempts: {last_error}")
