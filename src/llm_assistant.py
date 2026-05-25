def generate_rule_based_explanation(
    question,
    n_aircraft,
    avg_distance,
    avg_altitude,
    emission_proxy,
    workload_label,
    workload_score,
    top_features=None
):
    """
    Rule-based explanation assistant.

    This is a safe fallback when no LLM API key is available.
    It explains the dashboard results using only the provided scenario data.
    """

    if top_features is None:
        top_features = [
            "number of aircraft",
            "speed variability",
            "altitude variability",
            "arrival spacing"
        ]

    question_lower = question.lower()

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

    elif "emission" in question_lower or "environment" in question_lower or "green" in question_lower:
        return f"""
The current environmental proxy value is **{emission_proxy}**.

This value is based on a simplified approximation using aircraft distance and speed. A lower value indicates a more efficient arrival situation, while a higher value may indicate more distance, higher speeds, or less efficient arrival flow.

In a future version, this proxy can be improved using more realistic fuel-burn or descent-profile models.
"""

    elif "strategy" in question_lower or "best" in question_lower:
        return """
The best strategy should not be selected only by minimizing emissions.

A good arrival strategy should balance:

1. environmental efficiency,
2. delay,
3. runway throughput,
4. controller workload,
5. operational safety constraints.

The most useful strategy is usually the **workload-aware green strategy**, because it attempts to reduce environmental impact while avoiding high controller workload.
"""

    elif "limitation" in question_lower:
        return """
The main limitations of this prototype are:

1. It currently uses synthetic or sample arrival data.
2. Workload labels are proxy labels, not real ATCO workload ratings.
3. The environmental metric is a simplified proxy.
4. The model is not validated on operational Arlanda data yet.
5. The dashboard is a research prototype, not an operational ATC tool.

These limitations should be clearly stated in the GitHub README and project report.
"""

    else:
        return f"""
This scenario contains **{n_aircraft} aircraft**.

The predicted workload level is **{workload_label}**, with an estimated workload score of **{workload_score:.2f}**. The emission proxy is **{emission_proxy}**.

The dashboard suggests that workload and sustainability should be analysed together. A low-emission strategy is not necessarily ideal if it creates high traffic complexity or increases controller workload.
"""
