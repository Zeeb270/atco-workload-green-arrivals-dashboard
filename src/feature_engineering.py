import pandas as pd
import numpy as np


def create_time_window_features(df, window_minutes=3):
    """
    Convert aircraft-level arrival data into time-window traffic-complexity features.

    Parameters
    ----------
    df : pandas.DataFrame
        Arrival traffic data.
    window_minutes : int
        Size of each time window in minutes.

    Returns
    -------
    pandas.DataFrame
        One row per time window with engineered traffic-complexity features.
    """

    df = df.copy()

    required_columns = [
        "aircraft_id",
        "timestamp",
        "distance_to_airport_km",
        "altitude_ft",
        "speed_kt",
        "estimated_arrival_time",
        "runway",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["estimated_arrival_time"] = pd.to_datetime(df["estimated_arrival_time"])

    df = df.sort_values("timestamp")

    df["time_window"] = df["timestamp"].dt.floor(f"{window_minutes}min")

    grouped = df.groupby("time_window")

    features = grouped.agg(
        n_aircraft=("aircraft_id", "count"),
        avg_distance_km=("distance_to_airport_km", "mean"),
        std_distance_km=("distance_to_airport_km", "std"),
        avg_altitude_ft=("altitude_ft", "mean"),
        std_altitude_ft=("altitude_ft", "std"),
        avg_speed_kt=("speed_kt", "mean"),
        std_speed_kt=("speed_kt", "std"),
        n_runways=("runway", "nunique"),
    ).reset_index()

    features = features.fillna(0)

    # Estimated arrival time spacing
    spacing_features = []

    for window, group in grouped:
        arrival_times = pd.to_datetime(group["estimated_arrival_time"]).sort_values()

        if len(arrival_times) > 1:
            gaps = arrival_times.diff().dropna().dt.total_seconds() / 60.0
            min_gap = gaps.min()
            mean_gap = gaps.mean()
            n_tight_gaps = int((gaps < 3).sum())
        else:
            min_gap = 999
            mean_gap = 999
            n_tight_gaps = 0

        spacing_features.append(
            {
                "time_window": window,
                "min_arrival_gap_min": min_gap,
                "mean_arrival_gap_min": mean_gap,
                "n_tight_arrival_gaps": n_tight_gaps,
            }
        )

    spacing_df = pd.DataFrame(spacing_features)

    features = features.merge(spacing_df, on="time_window", how="left")

    # Simple operational complexity score
    features["complexity_score"] = (
        1.5 * features["n_aircraft"]
        + 2.0 * features["n_tight_arrival_gaps"]
        + 0.03 * features["std_speed_kt"]
        + 0.001 * features["std_altitude_ft"]
        + 1.0 * features["n_runways"]
    )

    # Workload labels based on score thresholds
    q_low = features["complexity_score"].quantile(0.33)
    q_high = features["complexity_score"].quantile(0.66)

    def label_workload(score):
        if score <= q_low:
            return "LOW"
        elif score <= q_high:
            return "MEDIUM"
        else:
            return "HIGH"

    features["workload_label"] = features["complexity_score"].apply(label_workload)

    # High-workload risk as a simple normalized score
    min_score = features["complexity_score"].min()
    max_score = features["complexity_score"].max()

    if max_score > min_score:
        features["high_workload_risk"] = (
            (features["complexity_score"] - min_score) / (max_score - min_score)
        )
    else:
        features["high_workload_risk"] = 0.0

    return features
