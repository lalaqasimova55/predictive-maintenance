"""Streamlit controls for the predictive maintenance demonstration."""

try:
    import streamlit as st
except ImportError:
    st = None


def health_label(rul):
    if rul > 60:
        return "Healthy"
    if rul > 30:
        return "Warning"
    return "Critical"


def run_app():
    if st is None:
        raise RuntimeError("Install streamlit to run the dashboard.")
    st.title("Predictive Maintenance — FD001")
    engine = st.number_input("Engine ID", min_value=1, step=1, value=1)
    cycle = st.number_input("Observed cycle", min_value=1, step=1, value=1)
    model = st.selectbox("Model", ["Median", "Random Forest", "XGBoost", "LSTM"])
    scenario = st.selectbox("Missingness scenario", ["S0 Clean", "S1 Random 10%", "S2 Random 20%", "S3 Short gap"])
    recovery = st.selectbox("Recovery method", ["Training median", "Causal forward fill"])
    k = st.selectbox("Inspection queue K", [5, 10])
    st.write({"engine": engine, "observed_cycle": cycle, "model": model, "scenario": scenario, "recovery": recovery, "queue_k": k})


if __name__ == "__main__":
    run_app()
