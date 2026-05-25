def generate_rule_based_explanation(
    question,
    n_aircraft,
    avg_distance,
    avg_altitude,
    emission_proxy,
    workload_label,
    workload_score,
    top_features=None,
    strategy_df=None
):
    """
    Rule-based explanation assistant.

    This is a safe fallback when no LLM API key is available.
    It explains dashboard results using only the provided scenario data.
    """

    if top_features is None:
        top_features = [
            "number of aircraft",
            "speed variability",
            "altitude variability",
            "arrival spacing"
        ]

    question_lower = question.lower()

    # Strategy explanation
    if (
        "strategy" in question_lower
        or "best" in question_lower
        or "green" in question_lower
        or "arrival" in question_lower
        or "compare" in question_lower
    ):
        if strategy_df is not None and len(strategy_df) > 0:
            best = strategy_df.iloc[0]

            return f"""
The best balanced strategy in the current scenario is **{best['strategy']}**.

It has:

- delay proxy: **{best['delay_proxy']}**
- emission proxy: **{best['emission_proxy']}**
- workload risk proxy: **{best['workload_risk_proxy']}**
- estimated high-workload windows: **{best['high_workload_windows_proxy']}**
- balanced score: **{best['balanced_score']:.3f}**

The balanced score combines delay, environmental efficiency, and workload risk. A lower balanced score is better.

In this prototype, the **green-only** strategy may reduce emissions, but it can increase workload risk if it compresses traffic or creates tighter arrival spacing. The **workload-aware green** strategy is usually more operationally realistic because it tries to reduce environmental impact while avoiding excessive controller workload.

These values are proxy metrics, not operational ATC measurements.
"""
        else:
            return """
The strategy comparison module is not available for the current scenario.

In general, the best strategy should not be selected only by minimizing emissions. A useful arrival strategy should balance delay, environmental efficiency, runway throughput, safety constraints, and controller workload.
"""

    # Workload explanation
    if "why" in question_lower and "workload" in question_lower:
        return f"""
The predicted workload is **{workload_label}** because the current scenario contains **{n_aircraft} aircraft** with an average distance of **{avg_distance} km** from the airport and an average altitude of **{avg_altitude:.0f} ft**.

The estimated workload score is **{workload_score:.2f}**.

The main operational drivers are likely:

1. {top_features[0]}
2. {top_features[1]}
3. {top_features[2]}

In this prototype, workload is estimated from traffic-complexity features rather than real controller self-reports. Therefore, the result should be interpreted as a research-style workload risk indicator, not an operational ATC decision.
"""

    # Environmental explanation
    if "emission" in question_lower or "environment" in question_lower:
        return f"""
The current environmental proxy value is **{emission_proxy}**.

This value is based on a simplified approximation using aircraft distance and speed. A lower value indicates a more efficient arrival situation, while a higher value may indicate more distance, higher speeds, or less efficient arrival flow.

In a future version, this proxy can be improved using more realistic fuel-burn or descent-profile models.
"""

    # Limitation explanation
    if "limitation" in question_lower or "weakness" in question_lower:
        return """
The main limitations of this prototype are:

1. It currently uses synthetic or sample arrival data.
2. Workload labels are proxy labels, not real ATCO workload ratings.
3. The environmental metric is a simplified proxy.
4. The model is not validated on operational Arlanda data yet.
5. The dashboard is a research prototype, not an operational ATC tool.
6. The current LLM assistant is rule-based unless API integration is enabled.

These limitations should be clearly stated in the GitHub README and project report.
"""

    # Model explanation
    if "model" in question_lower or "machine learning" in question_lower or "ml" in question_lower:
        return f"""
The dashboard uses a machine-learning workload model to classify traffic windows into workload levels.

The current predicted workload is **{workload_label}**, and the average workload score is **{workload_score:.2f}**.

The model uses traffic-complexity features such as aircraft count, speed variability, altitude variability, runway use, and arrival spacing.

The current version is a prototype. Since the workload labels are generated from proxy rules, the model demonstrates the pipeline rather than validated operational accuracy.
"""

    # Default answer
    return f"""
This scenario contains **{n_aircraft} aircraft**.

The predicted workload level is **{workload_label}**, with an estimated workload score of **{workload_score:.2f}**. The emission proxy is **{emission_proxy}**.

The dashboard suggests that workload and sustainability should be analysed together. A low-emission strategy is not necessarily ideal if it creates high traffic complexity or increases controller workload.

You can ask questions such as:

- Why is workload high?
- Which strategy is best?
- What are the limitations?
- How does the model work?
"""
