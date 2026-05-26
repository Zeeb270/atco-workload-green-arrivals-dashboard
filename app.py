import streamlit as st
import pandas as pd
import plotly.express as px
from src.feature_engineering import create_time_window_features
from src.llm_assistant import generate_rule_based_explanation, generate_groq_explanation
from src.workload_model import train_and_evaluate_model, compare_models
from src.synthetic_data import generate_synthetic_arrivals
from src.strategy_comparison import compare_arrival_strategies
from src.report_generator import generate_scenario_report
from src.data_preprocessing import (
    load_uploaded_data,
    suggest_column_mapping,
    apply_column_mapping,
    clean_aviation_data,
    validate_required_columns,
)
from src.green_arrival_optimizer import compare_green_arrival_strategies
from src.opensky_adapter import (
    convert_opensky_arrivals_to_dashboard_format,
    is_opensky_arrival_format,
)
from datetime import datetime, time
from src.opensky_fetcher import fetch_opensky_arrivals
from src.ui_components import (
    inject_global_css,
    hero,
    kpi_card,
    research_note,
    warning_note,
)

st.set_page_config(
    page_title="ATCO Workload-Aware Green Arrival Dashboard",
    page_icon="✈️",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
inject_global_css()
hero()
# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">✈️ ATCO Workload-Aware Green Arrival Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Machine-learning prototype for sustainable airport arrival operations, controller workload prediction, and LLM-based operational explanation.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Control Panel")
st.sidebar.markdown("Upload arrival traffic data or use the built-in sample dataset.")


dataset_structure = st.sidebar.radio(
    "Dataset structure",
    [
        "Raw aircraft arrival data",
        "OpenSky airport arrivals",
        "Preprocessed ML feature table"
    ]
)

data_mode = st.sidebar.radio(
    "Data source",
    ["Use sample CSV", "Generate synthetic scenario", "Upload file",
        "Fetch OpenSky ESSA arrivals"]
)
opensky_airport = "ESSA"
opensky_start_date = None
opensky_end_date = None

if data_mode == "Fetch OpenSky ESSA arrivals":
    st.sidebar.markdown("### OpenSky ESSA Arrivals")

    opensky_airport = st.sidebar.text_input(
        "Airport ICAO code",
        value="ESSA"
    )

    opensky_start_date = st.sidebar.date_input(
        "Start date",
        value=pd.Timestamp.today().date() - pd.Timedelta(days=7)
    )

    opensky_start_time = st.sidebar.time_input(
        "Start time",
        value=time(6, 0)
    )

    opensky_end_date = st.sidebar.date_input(
        "End date",
        value=pd.Timestamp.today().date() - pd.Timedelta(days=7)
    )

    opensky_end_time = st.sidebar.time_input(
        "End time",
        value=time(12, 0)
    )

    st.sidebar.caption(
        "Use a small historical interval first, for example 06:00–12:00. "
        "OpenSky direct fetching may timeout for large intervals."
    )

    st.sidebar.caption(
        "Use dates from yesterday or earlier. OpenSky airport arrivals are updated after nightly batch processing."
    )
traffic_mode = st.sidebar.selectbox(
    "Traffic scenario",
    ["Light", "Moderate", "Heavy"]
)

n_synthetic_aircraft = st.sidebar.slider(
    "Number of synthetic aircraft",
    min_value=20,
    max_value=180,
    value=90,
    step=10
)

separation_minutes = st.sidebar.slider(
    "Minimum runway separation in minutes",
    min_value=2.0,
    max_value=6.0,
    value=3.0,
    step=0.5
)

scenario_duration = st.sidebar.slider(
    "Scenario duration in minutes",
    min_value=30,
    max_value=180,
    value=90,
    step=15
)

random_seed = st.sidebar.number_input(
    "Random seed",
    min_value=1,
    max_value=9999,
    value=42,
    step=1
)

model_choice = st.sidebar.selectbox(
    "Workload model",
    ["Rule-based prototype", "kNN placeholder", "SVC placeholder", "Random Forest placeholder"]
)

show_raw_data = st.sidebar.checkbox("Show raw data", value=True)

# -----------------------------
# Load data
# -----------------------------
uploaded_file = None

if data_mode == "Upload file":
    uploaded_file = st.sidebar.file_uploader(
        "Upload arrival traffic file",
        type=["csv", "xlsx", "xls", "json"]
    )

if data_mode == "Upload file" and uploaded_file is not None:
    raw_df = load_uploaded_data(uploaded_file)
    st.sidebar.success("Uploaded dataset loaded.")

    st.subheader("Raw Uploaded Data Preview")
    st.dataframe(raw_df.head(20), use_container_width=True)

    if dataset_structure == "OpenSky airport arrivals":
        if not is_opensky_arrival_format(raw_df):
            st.warning(
                "The uploaded file does not look like a standard OpenSky arrivals file. "
                "The converter will still try to process it, but check the output carefully."
            )

        df = convert_opensky_arrivals_to_dashboard_format(raw_df)
        df, cleaning_report = clean_aviation_data(df)

        st.subheader("Converted OpenSky Arrival Data")
        st.dataframe(df.head(20), use_container_width=True)

    elif dataset_structure == "Preprocessed ML feature table":
        st.subheader("Preprocessed ML Feature Table Mode")
        st.warning(
            "This mode is under development. For now, use Raw aircraft arrival data or OpenSky airport arrivals."
        )
        st.stop()

    else:
        suggested_mapping = suggest_column_mapping(raw_df)

        st.subheader("Column Mapping")

        column_mapping = {}

        for standard_col in [
            "aircraft_id",
            "timestamp",
            "distance_to_airport_km",
            "altitude_ft",
            "speed_kt",
            "estimated_arrival_time",
            "route_angle_deg",
            "runway",
        ]:
            options = ["Not available"] + list(raw_df.columns)

            suggested_value = suggested_mapping.get(standard_col)

            if suggested_value in options:
                default_index = options.index(suggested_value)
            else:
                default_index = 0

            column_mapping[standard_col] = st.selectbox(
                f"Map column for: {standard_col}",
                options=options,
                index=default_index
            )

        mapped_df = apply_column_mapping(raw_df, column_mapping)
        df, cleaning_report = clean_aviation_data(mapped_df)
elif data_mode == "Fetch OpenSky ESSA arrivals":
    begin_dt = datetime.combine(opensky_start_date, opensky_start_time)
    end_dt = datetime.combine(opensky_end_date, opensky_end_time)

    begin_unix = int(pd.Timestamp(begin_dt).timestamp())
    end_unix = int(pd.Timestamp(end_dt).timestamp())

    try:
        with st.spinner("Fetching OpenSky arrivals..."):
            raw_df = fetch_opensky_arrivals(
                airport=opensky_airport,
                begin_unix=begin_unix,
                end_unix=end_unix,
            )

    except Exception as error:
        st.error(
            "OpenSky request failed. This may be due to timeout, API throttling, "
            "or temporary OpenSky unavailability."
        )

        st.info(
            """
            Suggested fixes:
            1. Try a one-day interval only.
            2. Try an older date.
            3. Use the OpenSky sample file upload mode.
            4. Add OpenSky authentication later for more reliable access.
            """
        )

        st.code(str(error))
        st.stop()

    if raw_df.empty:
        st.error(
            "No OpenSky arrivals returned for this airport/date interval. "
            "Try an older date, a one-day interval, or check the airport code."
        )
        st.stop()

    st.sidebar.success(f"Fetched {len(raw_df)} arrivals from OpenSky.")

    st.subheader("Raw OpenSky Arrivals")
    st.dataframe(raw_df.head(20), use_container_width=True)

    df = convert_opensky_arrivals_to_dashboard_format(raw_df)
    df, cleaning_report = clean_aviation_data(df)

    st.subheader("Converted OpenSky Arrival Data")
    st.dataframe(df.head(20), use_container_width=True)
elif data_mode == "Generate synthetic scenario":
    raw_df = generate_synthetic_arrivals(
        scenario=traffic_mode,
        n_aircraft=n_synthetic_aircraft,
        duration_minutes=scenario_duration,
        random_seed=random_seed
    )

    df, cleaning_report = clean_aviation_data(raw_df)
    st.sidebar.success(f"Generated {traffic_mode} synthetic scenario.")

else:
    raw_df = pd.read_csv("data_sample/sample_arrivals.csv")
    df, cleaning_report = clean_aviation_data(raw_df)
    st.sidebar.info("Using sample arrival dataset.")

# Convert times if available
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

if "estimated_arrival_time" in df.columns:
    df["estimated_arrival_time"] = pd.to_datetime(df["estimated_arrival_time"])

# Create 3-minute traffic-complexity features
features_df = create_time_window_features(df, window_minutes=3)

green_strategy_df, green_schedules = compare_green_arrival_strategies(
    df,
    separation_minutes=separation_minutes
)

ml_model, ml_metrics, confusion_df, predictions_df, feature_importance_df = train_and_evaluate_model(
    features_df,
    model_name=model_choice
)

model_comparison_df = compare_models(features_df)

strategy_df = compare_arrival_strategies(features_df)
# -----------------------------
# Basic derived metrics
# -----------------------------
n_aircraft = len(df)
avg_distance = round(df["distance_to_airport_km"].mean(), 2)
avg_altitude = round(df["altitude_ft"].mean(), 0)
avg_speed = round(df["speed_kt"].mean(), 2)

# Workload score from engineered traffic-complexity features
workload_score = features_df["complexity_score"].mean()
latest_workload_label = features_df.iloc[-1]["workload_label"]

if latest_workload_label == "LOW":
    workload_label = "LOW"
    workload_class = "status-low"
elif latest_workload_label == "MEDIUM":
    workload_label = "MEDIUM"
    workload_class = "status-medium"
else:
    workload_label = "HIGH"
    workload_class = "status-high"

# Environmental proxy
emission_proxy = round((df["distance_to_airport_km"].sum() * df["speed_kt"].mean()) / 1000, 2)

# -----------------------------
# Metric cards
# -----------------------------
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Aircraft</div>
            <div class="metric-value">{n_aircraft}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Avg Distance</div>
            <div class="metric-value">{avg_distance} km</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Avg Altitude</div>
            <div class="metric-value">{avg_altitude:.0f} ft</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Emission Proxy</div>
            <div class="metric-value">{emission_proxy}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Workload</div>
            <div class="metric-value"><span class="{workload_class}">{workload_label}</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Mission Control",
        "Data & Scenario",
        "Green Arrival Optimizer",
        "ML Workload Intelligence",
        "LLM Explanation Assistant",
        "Research Report"
    ]
)

# -----------------------------
# Tab 1
# -----------------------------
with tab1:
    st.subheader("Mission Control")

    st.markdown(
        """
        This dashboard is a research prototype for evaluating environmentally efficient
        arrival strategies at Stockholm Arlanda Airport.

        The current Phase 1 focus is **green arrival optimization**: comparing arrival
        sequencing strategies using delay, holding, extra-distance, and environmental-cost
        proxy metrics.

        Phase 2 extends the framework with **machine-learning-based workload intelligence**.
        Phase 3 adds an **LLM-ready explanation and reporting layer**.
        """
    )

    best_green_strategy = green_strategy_df.iloc[0]["strategy"]
    best_environmental_cost = green_strategy_df.iloc[0]["environmental_cost"]
    best_delay = green_strategy_df.iloc[0]["total_delay_min"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Aircraft Analysed", n_aircraft)
    col2.metric("Best Green Strategy", best_green_strategy)
    col3.metric("Environmental Cost", f"{best_environmental_cost:,.0f}")
    col4.metric("Total Delay", f"{best_delay:,.1f} min")

    st.markdown("### Research Workflow")

    st.markdown(
        """
        ```text
        Real or simulated arrival data
                ↓
        Arrival sequencing strategy comparison
                ↓
        Environmental proxy evaluation
                ↓
        ML workload-risk extension
                ↓
        LLM-assisted explanation and reporting
        ```
        """
    )

    st.info(
        "Current environmental values are proxy metrics for strategy comparison, "
        "not certified fuel-burn or CO₂ estimates."
    )
# -----------------------------
# Tab 2
# -----------------------------
with tab2:
    st.subheader("Data & Scenario")

    st.markdown(
        """
        This section shows the loaded dataset, cleaning diagnostics, and scenario setup.
        """
    )

    with st.expander("Show data quality report"):
        st.json(cleaning_report)

    with st.expander("Show processed arrival data"):
        st.dataframe(df, use_container_width=True)

    with st.expander("Show engineered traffic-complexity features"):
        st.dataframe(features_df, use_container_width=True)
# -----------------------------
# Tab 3
# -----------------------------
with tab3:
    st.subheader("Green Arrival Optimization")

    st.markdown(
        """
        This section compares simplified arrival sequencing strategies for environmental performance.
        The current module focuses on delay, holding, extra-distance, level-flight, and emissions-proxy metrics.
        """
    )

    st.subheader("Strategy-Level Results")
    st.dataframe(green_strategy_df, use_container_width=True)

    best_green_strategy = green_strategy_df.iloc[0]["strategy"]
    best_green_score = green_strategy_df.iloc[0]["balanced_score"]

    st.success(
        f"Best green-arrival strategy in this scenario: {best_green_strategy} "
        f"with balanced score = {best_green_score:.3f}"
    )

    col_a, col_b = st.columns(2)

    with col_a:
        fig_green_bar = px.bar(
            green_strategy_df,
            x="strategy",
            y=["total_delay_min", "holding_proxy_min", "environmental_cost"],
            barmode="group",
            title="Delay, Holding, and Emission Proxy by Strategy"
        )
        fig_green_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_green_bar, use_container_width=True)

    with col_b:
        fig_green_scatter = px.scatter(
            green_strategy_df,
            x="environmental_cost",
            y="total_delay_min",
            size="delayed_aircraft",
            color="strategy",
            title="Environment Cost vs Total Delay"
        )
        fig_green_scatter.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_green_scatter, use_container_width=True)

    st.subheader("Aircraft-Level Schedule")

    selected_schedule_strategy = st.selectbox(
        "Select strategy schedule to inspect",
        options=list(green_schedules.keys())
    )

    st.dataframe(
        green_schedules[selected_schedule_strategy][
            [
                "aircraft_id",
                "eta_minutes",
                "scheduled_landing_min",
                "delay_min",
                "holding_proxy_min",
                "extra_distance_proxy_km",
                "level_flight_proxy_min",
                "emission_proxy",
                "fuel_burn_proxy",
                "co2_proxy",
                "environmental_cost",
                "descent_inefficiency_score",
                "speed_inefficiency_score",
                "runway",
            ]
        ],
        use_container_width=True
    )

    fig_schedule = px.line(
        green_schedules[selected_schedule_strategy],
        x="scheduled_landing_min",
        y="delay_min",
        markers=True,
        hover_name="aircraft_id",
        title=f"Aircraft Delay Profile: {selected_schedule_strategy}"
    )
    fig_schedule.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_schedule, use_container_width=True)

    st.info(
        """
        Interpretation: lower emission proxy is better, but a strategy must also maintain acceptable delay.
        This first module evaluates green-arrival efficiency before adding workload impact in a later phase.
        """
    )
# -----------------------------
# Tab 4
# -----------------------------
with tab4:
    st.info(
    "Phase 2 extension: this module evaluates workload-risk prediction using "
    "traffic-complexity features. It is not the main green-optimization module yet."
    )
    st.subheader("Model Explainability")

    st.markdown(
        """
        This section compares workload prediction models and shows which
        traffic-complexity features are most influential in the current model.
        """
    )

    st.subheader("Model Comparison")

    st.dataframe(model_comparison_df, use_container_width=True)

    fig_model_comparison = px.bar(
        model_comparison_df,
        x="model",
        y="macro_f1",
        title="Model Comparison by Macro F1 Score"
    )
    fig_model_comparison.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_model_comparison, use_container_width=True)

    best_model_name = model_comparison_df.iloc[0]["model"]
    best_macro_f1 = model_comparison_df.iloc[0]["macro_f1"]

    st.success(
        f"Best model in current scenario: {best_model_name} "
        f"with macro F1 = {best_macro_f1:.3f}"
    )

    st.subheader("Feature Importance")

    st.dataframe(feature_importance_df, use_container_width=True)

    fig_importance = px.bar(
        feature_importance_df,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Feature Importance: {model_choice}"
    )
    fig_importance.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis={"categoryorder": "total ascending"}
    )
    st.plotly_chart(fig_importance, use_container_width=True)

    st.info(
        """
        For Random Forest and Gradient Boosting, feature importance comes from the model.
        For kNN and SVC, this prototype currently uses feature variance as a placeholder.
        A future version can use permutation importance or SHAP.
        """
    )

# -----------------------------
# Tab 5
# -----------------------------
with tab5:
    st.info(
    "This assistant explains dashboard outputs. It does not make operational ATC decisions."
    )
    st.subheader("LLM Research Assistant")

    st.markdown(
        """
        This assistant explains the dashboard results using the current scenario data.
        
        Current version: **rule-based explanation assistant**  
        Future version: OpenAI API / local LLM integration
        """
    )

    explanation_mode = st.selectbox(
        "Explanation mode",
        ["Rule-based assistant", "OpenAI LLM assistant coming later", "Free Groq Llama assistant"]
    )

    user_question = st.text_area(
        "Ask a question about the current scenario",
        placeholder="Example: Why is the workload predicted as high?"
    )

    top_features = [
        "number of aircraft",
        "arrival spacing",
        "speed variability",
        "altitude variability"
    ]

    scenario_context = {
    "number_of_aircraft": int(n_aircraft),
    "average_distance_km": float(avg_distance),
    "average_altitude_ft": float(avg_altitude),
    "emission_proxy": float(emission_proxy),
    "workload_label": str(workload_label),
    "workload_score": float(round(workload_score, 3)),
    "selected_ml_model": str(model_choice),
    "model_metrics": {
        "accuracy": float(round(ml_metrics["accuracy"], 3)),
        "macro_precision": float(round(ml_metrics["macro_precision"], 3)),
        "macro_recall": float(round(ml_metrics["macro_recall"], 3)),
        "macro_f1": float(round(ml_metrics["macro_f1"], 3)),
        "evaluation_mode": ml_metrics["evaluation_mode"],
    },
    "best_strategy": strategy_df.iloc[0].to_dict(),
    "top_features": top_features,
}

    if st.button("Generate explanation"):
        if user_question.strip() == "":
            st.error("Please enter a question.")
        else:
            if explanation_mode == "Rule-based assistant":
                answer = generate_rule_based_explanation(
                    question=user_question,
                    n_aircraft=n_aircraft,
                    avg_distance=avg_distance,
                    avg_altitude=avg_altitude,
                    emission_proxy=emission_proxy,
                    workload_label=workload_label,
                    workload_score=workload_score,
                    top_features=top_features,
                    strategy_df=strategy_df
                )

                st.markdown("### Assistant Answer")
                st.markdown(answer)

            else:
                try:
                    groq_api_key = st.secrets["GROQ_API_KEY"]

                    answer = generate_groq_explanation(
                        question=user_question,
                        scenario_context=scenario_context,
                        api_key=groq_api_key
                    )

                    st.markdown("### Groq Llama Assistant Answer")
                    st.markdown(answer)

                except KeyError:
                    st.error(
                        "Groq API key is missing. Add GROQ_API_KEY in Streamlit Cloud secrets, "
                        "or use the rule-based assistant."
                    )

                except Exception as error:
                    st.error(f"Groq assistant failed: {error}")
# -----------------------------
# Tab 6
# -----------------------------
with tab6:
    st.subheader("Downloadable Scenario Report")

    st.markdown(
        """
        This section generates a short research-style report from the current dashboard state.
        The report can be downloaded and included as supporting material in a portfolio or project documentation.
        """
    )

    report_text = generate_scenario_report(
        n_aircraft=n_aircraft,
        avg_distance=avg_distance,
        avg_altitude=avg_altitude,
        avg_speed=avg_speed,
        emission_proxy=emission_proxy,
        workload_label=workload_label,
        workload_score=workload_score,
        model_choice=model_choice,
        ml_metrics=ml_metrics,
        strategy_df=strategy_df,
        model_comparison_df=model_comparison_df,
    )

    st.text_area(
        "Generated report preview",
        report_text,
        height=500
    )

    st.download_button(
        label="Download Scenario Report",
        data=report_text,
        file_name="scenario_report.txt",
        mime="text/plain"
    )
