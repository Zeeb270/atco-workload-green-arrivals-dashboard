import pandas as pd
import numpy as np


def generate_synthetic_arrivals(
    scenario="Moderate",
    n_aircraft=80,
    start_time="2026-01-01 08:00:00",
    duration_minutes=90,
    random_seed=42
):
    """
    Generate synthetic airport arrival traffic data.

    This is not operational ATC data. It is a research prototype dataset
    designed to test feature engineering, workload prediction, and dashboard
    visualization.

    Parameters
    ----------
    scenario : str
        Light, Moderate, or Heavy.
    n_aircraft : int
        Number of aircraft to generate.
    start_time : str
        Scenario start time.
    duration_minutes : int
        Duration of the traffic scenario.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Synthetic arrival traffic dataset.
    """

    rng = np.random.default_rng(random_seed)
    start = pd.to_datetime(start_time)

    scenario = scenario.lower()

    if scenario == "light":
        n_aircraft = max(20, min(n_aircraft, 50))
        traffic_compression = 1.25
        speed_std = 18
        altitude_std = 2500
    elif scenario == "heavy":
        n_aircraft = max(90, n_aircraft)
        traffic_compression = 0.65
        speed_std = 35
        altitude_std = 5500
    else:
        n_aircraft = max(50, min(n_aircraft, 100))
        traffic_compression = 0.9
        speed_std = 25
        altitude_std = 4000

    aircraft_prefixes = [
        "SAS", "FIN", "DLH", "KLM", "BAW", "NAX", "SWR", "AFR", "RYR", "LOT",
        "AUA", "IBE", "TAP", "EZY", "WZZ"
    ]

    aircraft_ids = [
        f"{rng.choice(aircraft_prefixes)}{rng.integers(100, 999)}"
        for _ in range(n_aircraft)
    ]

    # Observation timestamps spread over the scenario duration
    timestamp_offsets = np.sort(
        rng.uniform(0, duration_minutes, size=n_aircraft)
    )

    timestamps = [
        start + pd.Timedelta(minutes=float(x))
        for x in timestamp_offsets
    ]

    # Estimated arrival times are later than observation timestamps.
    # Heavy traffic compresses arrivals more strongly.
    eta_offsets = rng.uniform(
        8 * traffic_compression,
        35 * traffic_compression,
        size=n_aircraft
    )

    estimated_arrival_times = [
        timestamps[i] + pd.Timedelta(minutes=float(eta_offsets[i]))
        for i in range(n_aircraft)
    ]

    distance_to_airport_km = rng.uniform(40, 180, size=n_aircraft)

    altitude_ft = np.clip(
        3500 + distance_to_airport_km * 190 + rng.normal(0, altitude_std, size=n_aircraft),
        2500,
        39000
    )

    speed_kt = np.clip(
        250 + distance_to_airport_km * 1.2 + rng.normal(0, speed_std, size=n_aircraft),
        180,
        490
    )

    route_angle_deg = rng.choice(
        [20, 35, 50, 65, 80, 110, 130, 150, 210, 240, 280, 320],
        size=n_aircraft
    )

    runway = rng.choice(
        ["01L", "01R"],
        size=n_aircraft,
        p=[0.55, 0.45]
    )

    df = pd.DataFrame(
        {
            "aircraft_id": aircraft_ids,
            "timestamp": timestamps,
            "distance_to_airport_km": np.round(distance_to_airport_km, 2),
            "altitude_ft": np.round(altitude_ft, 0).astype(int),
            "speed_kt": np.round(speed_kt, 1),
            "estimated_arrival_time": estimated_arrival_times,
            "route_angle_deg": route_angle_deg,
            "runway": runway,
        }
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df
