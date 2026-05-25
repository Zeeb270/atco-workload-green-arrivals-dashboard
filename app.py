import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="ATCO Workload-Aware Green Arrival Dashboard",
    layout="wide"
)

st.title("ATCO Workload-Aware Green Arrival Dashboard")

st.write(
    """
    This dashboard is a research prototype for analysing sustainable airport arrival operations
    while considering predicted air traffic controller workload.
    """
)

st.sidebar.header("Data Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload arrival traffic CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Uploaded dataset loaded.")
else:
    df = pd.read_csv("data_sample/sample_arrivals.csv")
    st.sidebar.info("Using sample arrival dataset.")

st.header("1. Arrival Traffic Data")

st.dataframe(df, use_container_width=True)

st.header("2. Basic Traffic Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Number of Aircraft", len(df))
col2.metric("Average Distance to Airport (km)", round(df["distance_to_airport_km"].mean(), 2))
col3.metric("Average Altitude (ft)", round(df["altitude_ft"].mean(), 0))
col4.metric("Average Speed (kt)", round(df["speed_kt"].mean(), 2))

st.header("3. Aircraft Distance to Airport")

fig_distance = px.bar(
    df,
    x="aircraft_id",
    y="distance_to_airport_km",
    color="runway",
    title="Aircraft Distance to Airport"
)

st.plotly_chart(fig_distance, use_container_width=True)

st.header("4. Altitude and Speed Profile")

fig_scatter = px.scatter(
    df,
    x="distance_to_airport_km",
    y="altitude_ft",
    size="speed_kt",
    color="runway",
    hover_name="aircraft_id",
    title="Arrival Traffic: Distance, Altitude, and Speed"
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.info(
    """
    Next step: extract traffic-complexity features in 3-minute windows and use them
    for machine-learning workload prediction.
    """
)
