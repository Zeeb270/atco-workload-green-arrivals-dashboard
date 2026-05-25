import streamlit as st
import pandas as pd
import plotly.express as px

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

uploaded_file = st.sidebar.file_uploader(
    "Upload arrival traffic CSV",
    type=["csv"]
)

traffic_mode = st.sidebar.selectbox(
    "Traffic scenario",
    ["Sample traffic", "Light", "Moderate", "Heavy"]
)

model_choice = st.sidebar.selectbox(
    "Workload model",
    ["Rule-based prototype", "kNN placeholder", "SVC placeholder", "Random Forest placeholder"]
)

show_raw_data = st.sidebar.checkbox("Show raw data", value=True)

# -----------------------------
# Load data
# -----------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Uploaded dataset loaded.")
else:
    df = pd.read_csv("data_sample/sample_arrivals.csv")
    st.sidebar.info("Using sample arrival dataset.")

# Convert times if available
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

if "estimated_arrival_time" in df.columns:
    df["estimated_arrival_time"] = pd.to_datetime(df["estimated_arrival_time"])

# -----------------------------
# Basic derived metrics
# -----------------------------
n_aircraft = len(df)
avg_distance = round(df["distance_to_airport_km"].mean(), 2)
avg_altitude = round(df["altitude_ft"].mean(), 0)
avg_speed = round(df["speed_kt"].mean(), 2)

# Prototype workload score
# This is deliberately simple for the first version.
workload_score = (
    0.35 * n_aircraft
    + 0.02 * df["speed_kt"].std()
    + 0.001 * df["altitude_ft"].std()
)

if workload_score < 5:
    workload_label = "LOW"
    workload_class = "status-low"
elif workload_score < 9:
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
    st.subheader("Prototype Workload Prediction")

    st.markdown(
        f"""
        <div class="section-card">
        Current prototype model: <b>{model_choice}</b><br>
        Estimated workload score: <b>{workload_score:.2f}</b><br>
        Predicted workload level: <span class="{workload_class}">{workload_label}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    workload_df = pd.DataFrame(
        {
            "feature": [
                "Number of aircraft",
                "Speed variability",
                "Altitude variability",
                "Traffic scenario factor"
            ],
            "importance_proxy": [
                n_aircraft,
                round(df["speed_kt"].std(), 2),
                round(df["altitude_ft"].std() / 1000, 2),
                {"Sample traffic": 1, "Light": 0.7, "Moderate": 1.0, "Heavy": 1.4}[traffic_mode]
            ]
        }
    )

    fig_workload = px.bar(
        workload_df,
        x="feature",
        y="importance_proxy",
        title="Prototype Workload Drivers"
    )
    fig_workload.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_workload, use_container_width=True)

    st.warning(
        "This is currently a prototype workload score. The next version will replace it with a trained machine-learning model."
    )

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
        This section will later show feature importance, confusion matrices, macro F1 scores,
        and model comparison results.
        """
    )

    explain_df = pd.DataFrame(
        {
            "model": ["kNN", "SVC", "Random Forest", "Gradient Boosting"],
            "macro_f1_placeholder": [0.78, 0.76, 0.73, 0.75],
            "interpretability": ["Medium", "Medium", "High", "Medium"]
        }
    )

    st.dataframe(explain_df, use_container_width=True)

    fig_model = px.bar(
        explain_df,
        x="model",
        y="macro_f1_placeholder",
        title="Placeholder Model Comparison"
    )
    fig_model.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_model, use_container_width=True)

# -----------------------------
# Tab 5
# -----------------------------
with tab5:
    st.subheader("LLM Research Assistant")

    st.markdown(
        """
        The LLM assistant will explain dashboard results using structured data from the current scenario.
        In the final version, this section can connect to OpenAI API or a local LLM.
        """
    )

    user_question = st.text_area(
        "Ask a question about the current scenario",
        placeholder="Example: Why is the workload predicted as high?"
    )

    if st.button("Generate prototype explanation"):
        if user_question.strip() == "":
            st.error("Please enter a question.")
        else:
            st.markdown("### Prototype Answer")
            st.write(
                f"""
                Based on the current dashboard state, the scenario contains {n_aircraft} aircraft,
                an average distance of {avg_distance} km, an average altitude of {avg_altitude:.0f} ft,
                and an emission proxy of {emission_proxy}. The prototype workload label is {workload_label}.
                
                A full LLM integration will later provide a more detailed explanation using model outputs,
                feature importance values, and strategy-comparison results.
                """
            )
