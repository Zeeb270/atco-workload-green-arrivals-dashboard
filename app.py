"""
ATCO Workload-Aware Green Arrival Dashboard
Stockholm Arlanda (ESSA) — Research Prototype

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time

from src.data_preprocessing import (
    load_uploaded_data,
    suggest_column_mapping,
    apply_column_mapping,
    clean_aviation_data,
)
from src.feature_engineering import create_time_window_features
from src.synthetic_data import generate_synthetic_arrivals
from src.green_arrival_optimizer import compare_green_arrival_strategies
from src.workload_model import train_and_evaluate_model, compare_models
from src.strategy_comparison import compare_arrival_strategies
from src.report_generator import generate_scenario_report
from src.environmental_metrics import compute_cdo_rate, compute_fuel_saved_kg
from src.opensky_adapter import convert_opensky_arrivals_to_dashboard_format, is_opensky_arrival_format
from src.opensky_fetcher import fetch_opensky_arrivals
from src.llm_assistant import generate_rule_based_explanation, generate_groq_explanation
from src.ui import inject_css, render_header, kpi_card, panel, small_note, warning_note

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ESSA Green Arrivals · ATCO Workload Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
render_header()

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("Control Panel")
st.sidebar.markdown("Configure data source, scenario, and model.")

data_mode = st.sidebar.radio(
    "Data source",
    ["Sample CSV", "Synthetic scenario", "Upload file", "Fetch OpenSky (ESSA)"],
)

dataset_structure = st.sidebar.radio(
    "Uploaded file format",
    ["Raw aircraft arrival data", "OpenSky airport arrivals"],
    help="Only used when uploading a file.",
)

# OpenSky date pickers (shown only when relevant)
opensky_airport = "ESSA"
opensky_start_date = opensky_end_date = None
opensky_start_time = opensky_end_time = None

if data_mode == "Fetch OpenSky (ESSA)":
    st.sidebar.markdown("### OpenSky query")
    opensky_airport = st.sidebar.text_input("Airport ICAO", value="ESSA")
    opensky_start_date = st.sidebar.date_input(
        "Start date", value=pd.Timestamp.today().date() - pd.Timedelta(days=7)
    )
    opensky_start_time = st.sidebar.time_input("Start time (UTC)", value=time(6, 0))
    opensky_end_date = st.sidebar.date_input(
        "End date", value=pd.Timestamp.today().date() - pd.Timedelta(days=7)
    )
    opensky_end_time = st.sidebar.time_input("End time (UTC)", value=time(12, 0))
    st.sidebar.caption(
        "Use a small interval (e.g. 6 h). OpenSky airport arrivals are "
        "updated after nightly batch processing — use dates from yesterday or earlier."
    )

st.sidebar.markdown("### Scenario settings")
traffic_mode = st.sidebar.selectbox("Traffic scenario", ["Light", "Moderate", "Heavy"])
n_synthetic_aircraft = st.sidebar.slider("Synthetic aircraft", 20, 180, 90, 10)
separation_minutes = st.sidebar.slider("Runway separation (min)", 2.0, 6.0, 3.0, 0.5)
scenario_duration = st.sidebar.slider("Scenario duration (min)", 30, 180, 90, 15)
random_seed = st.sidebar.number_input("Random seed", 1, 9999, 42, 1)

st.sidebar.markdown("### Model")
model_choice = st.sidebar.selectbox(
    "Workload classifier",
    ["Random Forest", "Gradient Boosting", "kNN", "SVC"],
)

# ── Load data ────────────────────────────────────────────────────────────────
uploaded_file = None
if data_mode == "Upload file":
    uploaded_file = st.sidebar.file_uploader(
        "Upload arrival file", type=["csv", "xlsx", "xls", "json"]
    )

@st.cache_data(show_spinner=False)
def load_sample():
    return pd.read_csv("data_sample/sample_arrivals.csv")

@st.cache_data(show_spinner=False)
def load_synthetic(scenario, n, duration, seed):
    return generate_synthetic_arrivals(
        scenario=scenario, n_aircraft=n, duration_minutes=duration, random_seed=seed
    )

with st.spinner("Loading data…"):
    if data_mode == "Upload file" and uploaded_file is not None:
        raw_df = load_uploaded_data(uploaded_file)
        if dataset_structure == "OpenSky airport arrivals":
            if not is_opensky_arrival_format(raw_df):
                st.warning(
                    "File does not look like a standard OpenSky arrivals export. "
                    "The converter will try anyway — check the output carefully."
                )
            raw_df = convert_opensky_arrivals_to_dashboard_format(raw_df)
        else:
            mapping = suggest_column_mapping(raw_df)
            with st.expander("Column mapping (edit if needed)", expanded=False):
                col_opts = ["Not available"] + list(raw_df.columns)
                column_mapping = {}
                for standard_col, suggested in mapping.items():
                    default = col_opts.index(suggested) if suggested in col_opts else 0
                    column_mapping[standard_col] = st.selectbox(
                        f"{standard_col}", col_opts, index=default, key=standard_col
                    )
            raw_df = apply_column_mapping(raw_df, column_mapping)
        df, cleaning_report = clean_aviation_data(raw_df)
        st.sidebar.success(f"Loaded {len(df)} rows from upload.")

    elif data_mode == "Fetch OpenSky (ESSA)":
        begin_dt = datetime.combine(opensky_start_date, opensky_start_time)
        end_dt = datetime.combine(opensky_end_date, opensky_end_time)
        begin_unix = int(pd.Timestamp(begin_dt).timestamp())
        end_unix = int(pd.Timestamp(end_dt).timestamp())
        try:
            with st.spinner("Fetching from OpenSky…"):
                raw_df = fetch_opensky_arrivals(
                    airport=opensky_airport, begin_unix=begin_unix, end_unix=end_unix
                )
        except Exception as err:
            st.error("OpenSky request failed.")
            st.info(
                "**Suggested fixes:** use a shorter interval, an older date, "
                "or switch to the Sample CSV / Upload mode."
            )
            st.code(str(err))
            st.stop()
        if raw_df.empty:
            st.error("No arrivals returned. Try a different date or airport code.")
            st.stop()
        df = convert_opensky_arrivals_to_dashboard_format(raw_df)
        df, cleaning_report = clean_aviation_data(df)
        st.sidebar.success(f"Fetched {len(df)} arrivals from OpenSky.")

    elif data_mode == "Synthetic scenario":
        raw_df = load_synthetic(traffic_mode, n_synthetic_aircraft, scenario_duration, random_seed)
        df, cleaning_report = clean_aviation_data(raw_df)
        st.sidebar.success(f"Generated {traffic_mode} synthetic scenario ({len(df)} aircraft).")

    else:  # Sample CSV
        raw_df = load_sample()
        df, cleaning_report = clean_aviation_data(raw_df)
        st.sidebar.info("Using built-in sample dataset.")

# Parse timestamps
for col in ("timestamp", "estimated_arrival_time"):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])

# ── Feature engineering + models ────────────────────────────────────────────
features_df = create_time_window_features(df, window_minutes=3)

green_strategy_df, green_schedules = compare_green_arrival_strategies(
    df, separation_minutes=separation_minutes
)

ml_model, ml_metrics, confusion_df, predictions_df, feature_importance_df = (
    train_and_evaluate_model(features_df, model_name=model_choice)
)

model_comparison_df = compare_models(features_df)
strategy_df = compare_arrival_strategies(features_df)

# ── Derived KPIs ─────────────────────────────────────────────────────────────
n_aircraft = df["aircraft_id"].nunique() if "aircraft_id" in df.columns else len(df)
avg_distance = df["distance_to_airport_km"].mean() if "distance_to_airport_km" in df.columns else None
avg_altitude = df["altitude_ft"].mean() if "altitude_ft" in df.columns else None
avg_speed = df["speed_kt"].mean() if "speed_kt" in df.columns else None

workload_score = features_df["complexity_score"].mean()
latest_workload_label = features_df.iloc[-1]["workload_label"]
workload_class = {"LOW": "status-low", "MEDIUM": "status-medium", "HIGH": "status-high"}.get(
    latest_workload_label, "status-medium"
)

cdo_rate = compute_cdo_rate(df)
fuel_saved = compute_fuel_saved_kg(df)
emission_proxy = round(
    (df["distance_to_airport_km"].sum() * df["speed_kt"].mean()) / 1000, 2
) if avg_distance and avg_speed else 0.0

# ── Global KPI strip ─────────────────────────────────────────────────────────
cols = st.columns(5)
kpi_definitions = [
    ("CDO Rate", f"{cdo_rate}%", "of arrivals on green descent", "green"),
    ("Flights", f"{n_aircraft:,}", "in current scenario", "white"),
    ("Avg Altitude", f"{avg_altitude:,.0f} ft" if avg_altitude else "N/A", "arrival state", "blue"),
    ("Fuel Saved", f"{fuel_saved:,.0f} kg" if fuel_saved else "N/A", "vs. step-down baseline", "amber"),
    ("Workload", f"<span class='{workload_class}'>{latest_workload_label}</span>", "predicted level", "red"),
]
for col, (label, value, subtitle, color) in zip(cols, kpi_definitions):
    with col:
        kpi_card(label, value, subtitle, color)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📡 Phase 1 — Traffic View",
    "🗂 Data & Diagnostics",
    "🌿 Green Arrival Optimizer",
    "🤖 ML Workload Model",
    "💬 Research Assistant",
    "📄 Report",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Phase 1 Demo
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    small_note(
        "Phase 1 — Real or sample ESSA arrival data, arrival-region trajectory features, "
        "and environmental evidence. ML workload and LLM explanation are later phases."
    )

    left, right = st.columns([1.6, 1])

    with left:
        with panel("Arrival traffic map"):
            if {"lat", "lon"}.issubset(df.columns):
                map_df = df.dropna(subset=["lat", "lon"])
                if len(map_df) > 3000:
                    map_df = map_df.sample(3000, random_state=42)
                st.map(map_df, latitude="lat", longitude="lon", size=8, use_container_width=True)
            else:
                st.info(
                    "Map view activates when the dataset contains **lat** and **lon** columns. "
                    "Upload a SCAT trajectory file to enable it."
                )

    with right:
        with panel("Arrival list"):
            display_cols = [
                c for c in
                ["aircraft_id", "adep", "aircraft_type", "runway", "star",
                 "distance_to_airport_km", "altitude_ft", "speed_kt"]
                if c in df.columns
            ]
            if display_cols:
                arrival_list = df[display_cols].copy()
                if "aircraft_id" in arrival_list.columns:
                    arrival_list = arrival_list.drop_duplicates("aircraft_id")
                st.dataframe(arrival_list.head(25), use_container_width=True, hide_index=True)
            else:
                st.info("No standard arrival columns found in the current dataset.")

    b1, b2, b3 = st.columns(3)

    with b1:
        with panel("Dataset evidence"):
            st.markdown(
                "**Source:** SCAT / OpenSky / synthetic  \n"
                "**Airport:** Stockholm Arlanda / ESSA  \n"
                "**Scale:** production-ready (up to thousands of flights)  \n"
                "**Purpose:** arrival-region green strategy research"
            )

    with b2:
        with panel("Environmental evidence"):
            if "low_altitude_level_time_min" in df.columns:
                st.metric("Low-altitude level time", f"{df['low_altitude_level_time_min'].sum():.1f} min")
            elif "estimated_co2_kg" in df.columns:
                st.metric("Estimated CO₂", f"{df['estimated_co2_kg'].sum():,.0f} kg")
            else:
                st.markdown(
                    f"**Emission proxy:** {emission_proxy}  \n"
                    "Derived from distance × speed — lower is better.  \n"
                    "Replace with BADA/OpenAP fuel-burn model for real values."
                )

    with b3:
        with panel("Research roadmap"):
            st.markdown(
                "**Phase 1 (now):** Data pipeline, trajectory features, env. proxies  \n"
                "**Phase 2:** ML workload prediction with labelled ATCO data  \n"
                "**Phase 3:** LLM explanation and rolling-horizon optimizer"
            )

    warning_note(
        "Research prototype — not an operational ATC system. "
        "Environmental and workload values are proxy metrics."
    )

    with st.expander("Show raw data sample (first 50 rows)"):
        st.dataframe(df.head(50), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Data & Diagnostics
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Data & Diagnostics")
    with st.expander("Data quality report"):
        st.json(cleaning_report)
    with st.expander("Processed arrival data"):
        st.dataframe(df, use_container_width=True)
    with st.expander("Engineered traffic-complexity features"):
        st.dataframe(features_df, use_container_width=True)

    # Complexity score over time
    if "time_window" in features_df.columns:
        fig = px.line(
            features_df,
            x="time_window",
            y="complexity_score",
            color="workload_label",
            title="Traffic complexity over time",
            color_discrete_map={"LOW": "#34d399", "MEDIUM": "#fbbf24", "HIGH": "#f87171"},
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Green Arrival Optimizer
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Green Arrival Optimizer")
    st.markdown(
        "Compares four arrival sequencing strategies on delay, holding, "
        "extra distance, level-flight time, and CO₂ proxy."
    )

    st.dataframe(green_strategy_df, use_container_width=True)

    best = green_strategy_df.iloc[0]
    st.success(
        f"Best strategy: **{best['strategy']}** "
        f"(balanced score = {best['balanced_score']:.3f})"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        fig_bar = px.bar(
            green_strategy_df,
            x="strategy",
            y=["total_delay_min", "holding_proxy_min", "environmental_cost"],
            barmode="group",
            title="Delay, Holding, and Environmental Cost by Strategy",
            color_discrete_sequence=["#60a5fa", "#fbbf24", "#f87171"],
        )
        fig_bar.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        fig_scatter = px.scatter(
            green_strategy_df,
            x="environmental_cost",
            y="total_delay_min",
            size="delayed_aircraft",
            color="strategy",
            title="Environmental Cost vs Total Delay",
        )
        fig_scatter.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Aircraft-level schedule")
    selected_strategy = st.selectbox("Inspect schedule for", options=list(green_schedules.keys()))
    schedule_cols = [
        c for c in [
            "aircraft_id", "eta_minutes", "scheduled_landing_min", "delay_min",
            "holding_proxy_min", "extra_distance_proxy_km", "level_flight_proxy_min",
            "fuel_burn_proxy", "co2_proxy", "environmental_cost",
            "descent_inefficiency_score", "speed_inefficiency_score", "runway",
        ] if c in green_schedules[selected_strategy].columns
    ]
    st.dataframe(green_schedules[selected_strategy][schedule_cols], use_container_width=True)

    fig_sched = px.line(
        green_schedules[selected_strategy],
        x="scheduled_landing_min",
        y="delay_min",
        markers=True,
        hover_name="aircraft_id",
        title=f"Delay profile — {selected_strategy}",
    )
    fig_sched.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_sched, use_container_width=True)

    st.info(
        "Lower environmental cost is better, but a strategy is only suitable if delay "
        "remains acceptable. This module evaluates green efficiency before adding "
        "ATCO workload impact in Phase 2."
    )

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — ML Workload Model
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.info(
        "Phase 2 extension: workload-risk prediction from traffic-complexity features. "
        "Labels are currently proxy-derived, not from real ATCO ratings."
    )
    st.subheader("Model comparison")
    st.dataframe(model_comparison_df, use_container_width=True)

    fig_cmp = px.bar(
        model_comparison_df,
        x="model", y="macro_f1",
        title="Macro F1 by model",
        color="model",
    )
    fig_cmp.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_cmp, use_container_width=True)

    best_model = model_comparison_df.iloc[0]
    st.success(f"Best model: **{best_model['model']}** (macro F1 = {best_model['macro_f1']:.3f})")

    st.subheader("Feature importance")
    fig_imp = px.bar(
        feature_importance_df,
        x="importance", y="feature",
        orientation="h",
        title=f"Feature importance — {model_choice}",
    )
    fig_imp.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("Confusion matrix")
    st.dataframe(confusion_df, use_container_width=True)

    st.info(
        "For Random Forest / Gradient Boosting, importance comes from the model. "
        "For kNN / SVC, feature variance is used as a placeholder. "
        "A future version will use SHAP values."
    )

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Research Assistant
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.info("This assistant explains dashboard outputs — it does not make ATC decisions.")
    st.subheader("Research assistant")

    explanation_mode = st.selectbox(
        "Mode",
        ["Rule-based (no API key needed)", "Groq Llama (free API key required)"],
    )

    user_question = st.text_area(
        "Ask a question about the current scenario",
        placeholder="Example: Why is workload predicted as HIGH?",
        height=100,
    )

    top_features = list(feature_importance_df["feature"].head(4)) if not feature_importance_df.empty else [
        "n_aircraft", "n_tight_arrival_gaps", "std_speed_kt", "std_altitude_ft"
    ]

    scenario_context = {
        "number_of_aircraft": int(n_aircraft),
        "average_distance_km": float(avg_distance) if avg_distance else None,
        "average_altitude_ft": float(avg_altitude) if avg_altitude else None,
        "emission_proxy": float(emission_proxy),
        "workload_label": latest_workload_label,
        "workload_score": float(round(workload_score, 3)),
        "selected_ml_model": model_choice,
        "model_metrics": {
            "accuracy": float(round(ml_metrics["accuracy"], 3)),
            "macro_f1": float(round(ml_metrics["macro_f1"], 3)),
        },
        "best_strategy": strategy_df.iloc[0].to_dict() if not strategy_df.empty else {},
        "top_features": top_features,
    }

    if st.button("Generate explanation", type="primary"):
        if not user_question.strip():
            st.error("Please enter a question.")
        elif explanation_mode.startswith("Rule-based"):
            answer = generate_rule_based_explanation(
                question=user_question,
                n_aircraft=n_aircraft,
                avg_distance=avg_distance or 0,
                avg_altitude=avg_altitude or 0,
                emission_proxy=emission_proxy,
                workload_label=latest_workload_label,
                workload_score=workload_score,
                top_features=top_features,
                strategy_df=strategy_df,
            )
            st.markdown("### Answer")
            st.markdown(answer)
        else:
            try:
                groq_api_key = st.secrets["GROQ_API_KEY"]
                answer = generate_groq_explanation(
                    question=user_question,
                    scenario_context=scenario_context,
                    api_key=groq_api_key,
                )
                st.markdown("### Groq Llama answer")
                st.markdown(answer)
            except KeyError:
                st.error(
                    "Add **GROQ_API_KEY** to Streamlit Cloud secrets "
                    "(Settings → Secrets) or use the rule-based mode."
                )
            except Exception as err:
                st.error(f"Groq request failed: {err}")

# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — Report
# ════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Downloadable scenario report")
    report_text = generate_scenario_report(
        n_aircraft=n_aircraft,
        avg_distance=avg_distance or 0,
        avg_altitude=avg_altitude or 0,
        avg_speed=avg_speed or 0,
        emission_proxy=emission_proxy,
        workload_label=latest_workload_label,
        workload_score=workload_score,
        model_choice=model_choice,
        ml_metrics=ml_metrics,
        strategy_df=strategy_df,
        model_comparison_df=model_comparison_df,
    )
    st.text_area("Preview", report_text, height=500)
    st.download_button(
        "⬇ Download report (.txt)",
        data=report_text,
        file_name=f"essa_green_arrivals_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
    )
