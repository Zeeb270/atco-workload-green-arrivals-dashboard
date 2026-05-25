import streamlit as st
import pandas as pd
import plotly.express as px
from src.feature_engineering import create_time_window_features
from src.llm_assistant import generate_rule_based_explanation
from src.workload_model import train_and_evaluate_model, compare_models
from src.synthetic_data import generate_synthetic_arrivals

st.set_page_config(
    page_title="ATCO Workload-Aware Green Arrival Dashboard",
    page_icon="✈️",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #050816 0%, #0B1026 45%, #111827 100%);
        color: #E5E7EB;
    }

    section[data-testid="stSidebar"] {
        background-color: #050816;
        border-right: 1px solid #1F2937;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #E5E7EB;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 17px;
        color: #9CA3AF;
        margin-top: 0px;
        margin-bottom: 25px;
    }

    .metric-card {
        background: rgba(15, 23, 42, 0.92);
        border: 1px solid rgba(56, 189, 248, 0.35);
        padding: 22px;
        border-radius: 18px;
        box-shadow: 0px 0px 20px rgba(56, 189, 248, 0.08);
    }

    .metric-label {
        font-size: 14px;
        color: #9CA3AF;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #38BDF8;
    }

    .section-card {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.18);
        padding: 20px;
        border-radius: 18px;
        margin-top: 12px;
        margin-bottom: 18px;
    }

    .status-low {
        color: #22C55E;
        font-weight: 800;
    }

    .status-medium {
        color: #FACC15;
        font-weight: 800;
    }

    .status-high {
        color: #EF4444;
        font-weight: 800;
    }

    h1, h2, h3 {
        color: #F9FAFB;
    }

    .stDataFrame {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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


data_mode = st.sidebar.radio(
    "Data source",
    ["Use sample CSV", "Generate synthetic scenario", "Upload CSV"]
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

if data_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload arrival traffic CSV",
        type=["csv"]
    )

if data_mode == "Upload CSV" and uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Uploaded dataset loaded.")

elif data_mode == "Generate synthetic scenario":
    df = generate_synthetic_arrivals(
        scenario=traffic_mode,
        n_aircraft=n_synthetic_aircraft,
        duration_minutes=scenario_duration,
        random_seed=random_seed
    )
    st.sidebar.success(f"Generated {traffic_mode} synthetic scenario.")

else:
    df = pd.read_csv("data_sample/sample_arrivals.csv")
    st.sidebar.info("Using sample arrival dataset.")

# Convert times if available
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

if "estimated_arrival_time" in df.columns:
    df["estimated_arrival_time"] = pd.to_datetime(df["estimated_arrival_time"])
# Create 3-minute traffic-complexity features
features_df = create_time_window_features(df, window_minutes=3)
ml_model, ml_metrics, confusion_df, predictions_df, feature_importance_df = train_and_evaluate_model(
    features_df,
    model_name=model_choice
)
model_comparison_df = compare_models(features_df)
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
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Traffic Overview",
        "Workload Prediction",
        "Green Arrival Strategy",
        "Model Explainability",
        "LLM Assistant"
    ]
)

# -----------------------------
# Tab 1
# -----------------------------
with tab1:
    st.subheader("Arrival Traffic Overview")

    left, right = st.columns([1.1, 1])

    with left:
        fig_radar = px.scatter_polar(
            df,
            r="distance_to_airport_km",
            theta="route_angle_deg",
            color="runway",
            size="speed_kt",
            hover_name="aircraft_id",
            title="Radar-Style Arrival Distribution"
        )
        fig_radar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with right:
        fig_altitude = px.scatter(
            df,
            x="distance_to_airport_km",
            y="altitude_ft",
            size="speed_kt",
            color="runway",
            hover_name="aircraft_id",
            title="Altitude Profile by Distance"
        )
        fig_altitude.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_altitude, use_container_width=True)

    if show_raw_data:
        st.subheader("Raw Arrival Dataset")
        st.dataframe(df, use_container_width=True)

# -----------------------------
# Tab 2
# -----------------------------
with tab2:
    st.subheader("Machine-Learning Workload Prediction")

    st.markdown(
        f"""
        <div class="section-card">
        Current model: <b>{model_choice}</b><br>
        Estimated workload score: <b>{workload_score:.2f}</b><br>
        Predicted workload level: <span class="{workload_class}">{workload_label}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("3-Minute Traffic-Complexity Features")
    st.dataframe(features_df, use_container_width=True)

    fig_window_workload = px.line(
        features_df,
        x="time_window",
        y="complexity_score",
        markers=True,
        color="workload_label",
        title="Traffic Complexity Score by 3-Minute Window"
    )
    fig_window_workload.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_window_workload, use_container_width=True)

    st.subheader("Model Performance")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    metric_col1.metric("Accuracy", f"{ml_metrics['accuracy']:.3f}")
    metric_col2.metric("Macro Precision", f"{ml_metrics['macro_precision']:.3f}")
    metric_col3.metric("Macro Recall", f"{ml_metrics['macro_recall']:.3f}")
    metric_col4.metric("Macro F1", f"{ml_metrics['macro_f1']:.3f}")

    st.caption(ml_metrics["evaluation_mode"])

    st.subheader("Predictions by 3-Minute Window")
    st.dataframe(predictions_df, use_container_width=True)

    fig_ml_workload = px.line(
        predictions_df,
        x="time_window",
        y="complexity_score",
        markers=True,
        color="ml_predicted_workload",
        title="ML-Predicted Workload by 3-Minute Window"
    )
    fig_ml_workload.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_ml_workload, use_container_width=True)

    st.subheader("Confusion Matrix")
    st.dataframe(confusion_df, use_container_width=True)
# -----------------------------
# Tab 3
# -----------------------------
with tab3:
    st.subheader("Green Arrival Strategy Comparison")

    strategy_df = pd.DataFrame(
        {
            "strategy": ["FCFS baseline", "Green-only", "Workload-aware green"],
            "delay_proxy": [18, 12, 15],
            "emission_proxy": [100, 74, 82],
            "high_workload_windows": [3, 5, 1]
        }
    )

    col_a, col_b = st.columns(2)

    with col_a:
        fig_strategy = px.bar(
            strategy_df,
            x="strategy",
            y=["delay_proxy", "emission_proxy", "high_workload_windows"],
            barmode="group",
            title="Strategy Trade-Off Comparison"
        )
        fig_strategy.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_strategy, use_container_width=True)

    with col_b:
        fig_pareto = px.scatter(
            strategy_df,
            x="emission_proxy",
            y="delay_proxy",
            size="high_workload_windows",
            color="strategy",
            title="Delay vs Emission Proxy"
        )
        fig_pareto.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

    st.dataframe(strategy_df, use_container_width=True)

# -----------------------------
# Tab 4
# -----------------------------
with tab4:
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
        ["Rule-based assistant", "OpenAI LLM assistant coming later"]
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
                    top_features=top_features
                )

                st.markdown("### Assistant Answer")
                st.markdown(answer)

            else:
                st.warning(
                    "OpenAI LLM integration will be added in the next version. "
                    "For now, use the rule-based assistant."
                )
