import streamlit as st

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

st.header("Planned Workflow")

st.markdown(
    """
    1. Upload or load arrival traffic data  
    2. Extract traffic-complexity features  
    3. Predict controller workload using machine learning  
    4. Compare baseline and green arrival strategies  
    5. Use an LLM assistant to explain results  
    """
)

st.info("Dashboard implementation is in progress.")
