def generate_scenario_report(
    n_aircraft,
    avg_distance,
    avg_altitude,
    avg_speed,
    emission_proxy,
    workload_label,
    workload_score,
    model_choice,
    ml_metrics,
    strategy_df,
    model_comparison_df,
):
    """
    Generate a downloadable text report for the current dashboard scenario.
    """

    best_strategy = strategy_df.iloc[0]
    best_model = model_comparison_df.iloc[0]

    report = f"""
ATCO WORKLOAD-AWARE GREEN ARRIVAL DASHBOARD
Scenario Analysis Report

1. Scenario Summary
-------------------
Number of aircraft: {n_aircraft}
Average distance to airport: {avg_distance} km
Average altitude: {avg_altitude:.0f} ft
Average speed: {avg_speed} kt
Emission proxy: {emission_proxy}

2. Workload Prediction
----------------------
Selected model: {model_choice}
Predicted workload level: {workload_label}
Average workload complexity score: {workload_score:.3f}

Model performance:
- Accuracy: {ml_metrics['accuracy']:.3f}
- Macro precision: {ml_metrics['macro_precision']:.3f}
- Macro recall: {ml_metrics['macro_recall']:.3f}
- Macro F1: {ml_metrics['macro_f1']:.3f}
- Evaluation mode: {ml_metrics['evaluation_mode']}

3. Best Model in Current Comparison
-----------------------------------
Best model: {best_model['model']}
Best macro F1 score: {best_model['macro_f1']:.3f}

4. Arrival Strategy Comparison
------------------------------
Best balanced strategy: {best_strategy['strategy']}
Balanced score: {best_strategy['balanced_score']:.3f}

Best strategy metrics:
- Delay proxy: {best_strategy['delay_proxy']}
- Emission proxy: {best_strategy['emission_proxy']}
- Workload risk proxy: {best_strategy['workload_risk_proxy']}
- Estimated high-workload windows: {best_strategy['high_workload_windows_proxy']}

5. Interpretation
-----------------
The dashboard suggests that sustainable arrival operations should not be assessed using environmental metrics alone. A strategy with lower emissions may still be operationally unsuitable if it increases traffic complexity or controller workload.

The workload-aware green strategy is designed to balance environmental performance with operational feasibility.

6. Limitations
--------------
- The current dataset may be synthetic or user-uploaded.
- Workload labels are proxy labels, not real ATCO workload ratings.
- Environmental impact is represented using simplified proxy metrics.
- The model is a research prototype and is not intended for operational ATC use.
- Reliable validation would require operational data and expert or controller workload labels.

7. Future Work
--------------
- Integrate OpenSky or SCAT-derived Swedish air traffic data.
- Improve environmental modelling with fuel-burn or descent-profile estimates.
- Replace proxy workload labels with expert-labelled or measured ATCO workload data.
- Add uncertainty estimation and stronger model validation.
- Extend the explanation assistant with a free or local LLM.
"""

    return report.strip()
