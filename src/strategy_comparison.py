import pandas as pd
import numpy as np


def compare_arrival_strategies(features_df):
    """
    Compare simplified arrival-management strategies.

    Strategies:
    1. FCFS baseline
    2. Green-only
    3. Workload-aware green

    This is a research prototype. The values are proxy metrics derived from
    traffic-complexity features, not operational ATC outputs.
    """

    df = features_df.copy()

    total_aircraft = df["n_aircraft"].sum()
    avg_complexity = df["complexity_score"].mean()
    max_complexity = df["complexity_score"].max()
    tight_gaps = df["n_tight_arrival_gaps"].sum()

    speed_variability = df["std_speed_kt"].mean()
    altitude_variability = df["std_altitude_ft"].mean()

    # Baseline: First-Come-First-Served
    fcfs_delay = 0.35 * total_aircraft + 1.2 * tight_gaps
    fcfs_emissions = 1.00 * total_aircraft + 0.08 * speed_variability + 0.002 * altitude_variability
    fcfs_workload = avg_complexity + 0.3 * tight_gaps

    # Green-only: prioritizes reduced emissions, but may create workload concentration
    green_delay = 0.42 * total_aircraft + 1.0 * tight_gaps
    green_emissions = 0.78 * fcfs_emissions
    green_workload = fcfs_workload * 1.18

    # Workload-aware green: slightly less green than green-only, but reduces high workload
    workload_aware_delay = 0.40 * total_aircraft + 0.9 * tight_gaps
    workload_aware_emissions = 0.84 * fcfs_emissions
    workload_aware_workload = fcfs_workload * 0.82

    strategy_df = pd.DataFrame(
        {
            "strategy": [
                "FCFS baseline",
                "Green-only",
                "Workload-aware green"
            ],
            "delay_proxy": [
                round(fcfs_delay, 2),
                round(green_delay, 2),
                round(workload_aware_delay, 2)
            ],
            "emission_proxy": [
                round(fcfs_emissions, 2),
                round(green_emissions, 2),
                round(workload_aware_emissions, 2)
            ],
            "workload_risk_proxy": [
                round(fcfs_workload, 2),
                round(green_workload, 2),
                round(workload_aware_workload, 2)
            ],
            "high_workload_windows_proxy": [
                int((df["complexity_score"] > df["complexity_score"].quantile(0.66)).sum()),
                int(np.ceil((df["complexity_score"] > df["complexity_score"].quantile(0.60)).sum())),
                int((df["complexity_score"] > df["complexity_score"].quantile(0.75)).sum())
            ]
        }
    )

    # Lower is better for all three normalized metrics.
    for col in ["delay_proxy", "emission_proxy", "workload_risk_proxy"]:
        min_val = strategy_df[col].min()
        max_val = strategy_df[col].max()

        if max_val > min_val:
            strategy_df[f"normalized_{col}"] = (
                (strategy_df[col] - min_val) / (max_val - min_val)
            )
        else:
            strategy_df[f"normalized_{col}"] = 0.0

    strategy_df["balanced_score"] = (
        0.30 * strategy_df["normalized_delay_proxy"]
        + 0.35 * strategy_df["normalized_emission_proxy"]
        + 0.35 * strategy_df["normalized_workload_risk_proxy"]
    )

    strategy_df = strategy_df.sort_values("balanced_score").reset_index(drop=True)

    return strategy_df
