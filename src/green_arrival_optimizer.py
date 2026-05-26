import pandas as pd
import numpy as np
from src.environmental_metrics import add_environmental_metrics, summarize_environmental_metrics

def prepare_arrival_jobs(df, separation_minutes=3.0):
    """
    Prepare aircraft arrival jobs from arrival traffic data.

    The function estimates each aircraft's unconstrained arrival time and creates
    a simple scheduling problem where aircraft must be assigned feasible landing
    times while respecting minimum runway separation.

    Parameters
    ----------
    df : pandas.DataFrame
        Arrival traffic data.
    separation_minutes : float
        Minimum landing-time separation between consecutive aircraft.

    Returns
    -------
    pandas.DataFrame
        Prepared aircraft arrival jobs.
    """

    jobs = df.copy()

    jobs["timestamp"] = pd.to_datetime(jobs["timestamp"], errors="coerce")
    jobs["estimated_arrival_time"] = pd.to_datetime(
        jobs["estimated_arrival_time"],
        errors="coerce"
    )

    jobs = jobs.dropna(subset=["timestamp", "estimated_arrival_time"])

    jobs = jobs.sort_values("estimated_arrival_time").reset_index(drop=True)

    scenario_start = jobs["estimated_arrival_time"].min()

    jobs["eta_minutes"] = (
        jobs["estimated_arrival_time"] - scenario_start
    ).dt.total_seconds() / 60.0

    jobs["distance_to_airport_km"] = pd.to_numeric(
        jobs["distance_to_airport_km"],
        errors="coerce"
    ).fillna(jobs["distance_to_airport_km"].median())

    jobs["altitude_ft"] = pd.to_numeric(
        jobs["altitude_ft"],
        errors="coerce"
    ).fillna(jobs["altitude_ft"].median())

    jobs["speed_kt"] = pd.to_numeric(
        jobs["speed_kt"],
        errors="coerce"
    ).fillna(jobs["speed_kt"].median())

    jobs["separation_minutes"] = separation_minutes

    return jobs


def schedule_fcfs(jobs, separation_minutes=3.0):
    """
    First-Come-First-Served arrival sequence.
    """

    scheduled = jobs.sort_values("eta_minutes").copy()
    return assign_landing_times(scheduled, separation_minutes)


def schedule_delay_minimizing(jobs, separation_minutes=3.0):
    """
    Delay-minimizing strategy.

    In this simplified prototype, this is close to FCFS but prioritizes aircraft
    with earlier estimated arrival times and shorter remaining distance.
    """

    scheduled = jobs.sort_values(
        ["eta_minutes", "distance_to_airport_km"],
        ascending=[True, True]
    ).copy()

    return assign_landing_times(scheduled, separation_minutes)


def schedule_green_only(jobs, separation_minutes=3.0):
    """
    Green-only strategy.

    Prioritizes aircraft that are closer, lower, and slower, because these are
    assumed to be more sensitive to inefficient holding or level-flight penalties.
    """

    scheduled = jobs.copy()

    scheduled["green_priority_score"] = (
        0.50 * scheduled["distance_to_airport_km"].rank(ascending=True)
        + 0.30 * scheduled["altitude_ft"].rank(ascending=True)
        + 0.20 * scheduled["speed_kt"].rank(ascending=True)
    )

    scheduled = scheduled.sort_values(
        ["green_priority_score", "eta_minutes"],
        ascending=[True, True]
    ).copy()

    return assign_landing_times(scheduled, separation_minutes)


def schedule_balanced_green(jobs, separation_minutes=3.0):
    """
    Balanced green strategy.

    Balances estimated arrival time and environmental sensitivity.
    """

    scheduled = jobs.copy()

    scheduled["eta_rank"] = scheduled["eta_minutes"].rank(ascending=True)
    scheduled["distance_rank"] = scheduled["distance_to_airport_km"].rank(ascending=True)
    scheduled["altitude_rank"] = scheduled["altitude_ft"].rank(ascending=True)
    scheduled["speed_rank"] = scheduled["speed_kt"].rank(ascending=True)

    scheduled["balanced_priority_score"] = (
        0.45 * scheduled["eta_rank"]
        + 0.25 * scheduled["distance_rank"]
        + 0.20 * scheduled["altitude_rank"]
        + 0.10 * scheduled["speed_rank"]
    )

    scheduled = scheduled.sort_values(
        ["balanced_priority_score", "eta_minutes"],
        ascending=[True, True]
    ).copy()

    return assign_landing_times(scheduled, separation_minutes)


def assign_landing_times(sequence_df, separation_minutes=3.0):
    """
    Assign feasible landing times to a proposed arrival sequence.

    Each aircraft lands no earlier than its estimated arrival time and no earlier
    than the previous aircraft plus the separation requirement.
    """

    scheduled = sequence_df.copy().reset_index(drop=True)

    landing_times = []

    for i, row in scheduled.iterrows():
        eta = row["eta_minutes"]

        if i == 0:
            landing_time = eta
        else:
            landing_time = max(
                eta,
                landing_times[-1] + separation_minutes
            )

        landing_times.append(landing_time)

    scheduled["scheduled_landing_min"] = landing_times
    scheduled["delay_min"] = scheduled["scheduled_landing_min"] - scheduled["eta_minutes"]

    scheduled["holding_proxy_min"] = scheduled["delay_min"].clip(lower=0)

    # Extra distance proxy:
    # assume each holding minute creates additional flown distance.
    # 1 minute at approach speed roughly corresponds to several km.
    scheduled["extra_distance_proxy_km"] = scheduled["holding_proxy_min"] * (
        scheduled["speed_kt"] * 1.852 / 60.0
    )

    # Level-flight proxy:
    # delay at lower altitude is penalized more strongly.
    scheduled["low_altitude_factor"] = np.where(
        scheduled["altitude_ft"] < 12000,
        1.5,
        np.where(scheduled["altitude_ft"] < 24000, 1.2, 1.0)
    )

    scheduled["level_flight_proxy_min"] = (
        scheduled["holding_proxy_min"] * scheduled["low_altitude_factor"]
    )

    # Environmental proxy:
    # simplified cost based on holding, extra distance, and low-altitude level flight.
        # Legacy simple emission proxy kept for comparison.
    scheduled["emission_proxy"] = (
        1.00 * scheduled["holding_proxy_min"]
        + 0.08 * scheduled["extra_distance_proxy_km"]
        + 0.70 * scheduled["level_flight_proxy_min"]
    )

    # Add improved environmental metrics.
    scheduled = add_environmental_metrics(scheduled)

    return scheduled


def evaluate_strategy(scheduled_df, strategy_name):
    """
    Aggregate strategy-level metrics.
    """

    total_delay = scheduled_df["delay_min"].sum()
    avg_delay = scheduled_df["delay_min"].mean()
    max_delay = scheduled_df["delay_min"].max()

    total_holding = scheduled_df["holding_proxy_min"].sum()
    total_extra_distance = scheduled_df["extra_distance_proxy_km"].sum()
    total_level_flight = scheduled_df["level_flight_proxy_min"].sum()
    total_emission_proxy = scheduled_df["emission_proxy"].sum()

    delayed_aircraft = int((scheduled_df["delay_min"] > 0.1).sum())

    env_summary = summarize_environmental_metrics(scheduled_df)

    result = {
        "strategy": strategy_name,
        "total_delay_min": round(total_delay, 2),
        "avg_delay_min": round(avg_delay, 2),
        "max_delay_min": round(max_delay, 2),
        "holding_proxy_min": round(total_holding, 2),
        "extra_distance_proxy_km": round(total_extra_distance, 2),
        "level_flight_proxy_min": round(total_level_flight, 2),
        "emission_proxy": round(total_emission_proxy, 2),
        "delayed_aircraft": delayed_aircraft,
    }

    result.update(env_summary)

    return result
def compare_green_arrival_strategies(df, separation_minutes=3.0):
    """
    Compare arrival strategies for environmental performance.

    Returns
    -------
    strategy_summary : pandas.DataFrame
        Strategy-level comparison table.
    schedules : dict
        Aircraft-level schedules for each strategy.
    """

    jobs = prepare_arrival_jobs(df, separation_minutes=separation_minutes)

    schedules = {
        "FCFS baseline": schedule_fcfs(jobs, separation_minutes),
        "Delay-minimizing": schedule_delay_minimizing(jobs, separation_minutes),
        "Green-only": schedule_green_only(jobs, separation_minutes),
        "Balanced green": schedule_balanced_green(jobs, separation_minutes),
    }

    results = []

    for strategy_name, scheduled_df in schedules.items():
        results.append(evaluate_strategy(scheduled_df, strategy_name))

    strategy_summary = pd.DataFrame(results)

    # Lower is better.
    for col in ["total_delay_min", "environmental_cost"]:
        min_val = strategy_summary[col].min()
        max_val = strategy_summary[col].max()

        if max_val > min_val:
            strategy_summary[f"normalized_{col}"] = (
                (strategy_summary[col] - min_val) / (max_val - min_val)
            )
        else:
            strategy_summary[f"normalized_{col}"] = 0.0

    strategy_summary["balanced_score"] = (
        0.40 * strategy_summary["normalized_total_delay_min"]
        + 0.60 * strategy_summary["normalized_environmental_cost"]
    )

    strategy_summary = strategy_summary.sort_values(
        "balanced_score",
        ascending=True
    ).reset_index(drop=True)

    return strategy_summary, schedules
