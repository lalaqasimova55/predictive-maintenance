"""Streamlit dashboard for causal NASA C-MAPSS FD001 RUL inference."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import xgboost as xgb

from predictive_maintenance.artifacts import load_joblib
from predictive_maintenance.constants import CYCLE_COLUMN, ID_COLUMN, RUL_TARGET_COLUMN
from predictive_maintenance.dashboard import (
    HEALTH_STATUS_NOTE,
    QUEUE_NOTE,
    health_label,
    observed_prefix,
    plot_rul_history,
    plot_sensor_history,
    top_k_queue,
)
from predictive_maintenance.features import build_tabular_features
from predictive_maintenance.masks import apply_sensor_mask, nested_random_masks, short_gap_mask
from predictive_maintenance.models.lstm import LSTMConfig, RULLSTM, predict_lstm
from predictive_maintenance.sequences import build_prefix_window, scale_sequence_frame


PREPARED_PATH = ROOT / "artifacts/preprocessing/prepared_bundle.joblib"
MODEL_DIR = ROOT / "artifacts/models"


@st.cache_resource
def load_bundle():
    if not PREPARED_PATH.exists():
        raise FileNotFoundError(
            "Prepared data was not found. Run download_data.py and prepare_data.py first."
        )
    return joblib.load(PREPARED_PATH)


@st.cache_resource
def load_models():
    missing = [
        str(path)
        for path in (
            MODEL_DIR / "baseline.joblib",
            MODEL_DIR / "random_forest.joblib",
            MODEL_DIR / "xgboost.json",
            MODEL_DIR / "lstm_meta.joblib",
            MODEL_DIR / "lstm.pt",
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Model artifacts are missing. Run scripts/train_models.py first. Missing: "
            + ", ".join(missing)
        )

    baseline = load_joblib(MODEL_DIR / "baseline.joblib")
    random_forest = load_joblib(MODEL_DIR / "random_forest.joblib")
    xgboost_model = xgb.XGBRegressor()
    xgboost_model.load_model(MODEL_DIR / "xgboost.json")

    metadata = load_joblib(MODEL_DIR / "lstm_meta.joblib")
    config = LSTMConfig(**metadata["config"])
    lstm = RULLSTM(config.input_size, config.hidden_units, config.dense_units)
    lstm.load_state_dict(torch.load(MODEL_DIR / "lstm.pt", map_location="cpu"))
    lstm.eval()

    return {
        "Median baseline": baseline,
        "Random Forest": random_forest,
        "XGBoost": xgboost_model,
        "LSTM": lstm,
    }


def corrupt_prefix(prefix: pd.DataFrame, bundle, scenario: str, seed: int) -> pd.DataFrame:
    """Apply a deterministic sensor-missingness scenario to the visible prefix only."""
    sensors = bundle.selected_sensors
    if scenario == "Clean":
        return prefix.copy()
    if scenario in {"Random 10%", "Random 20%"}:
        low_mask, high_mask = nested_random_masks(prefix, sensors, seed)
        mask = low_mask if scenario == "Random 10%" else high_mask
    elif scenario == "Five-cycle sensor gap":
        mask, _ = short_gap_mask(prefix, sensors, preferred_sensor="sensor_11", gap_cycles=5)
    else:
        raise ValueError(f"Unknown missingness scenario: {scenario}")
    return apply_sensor_mask(prefix, mask, sensors)


def recover_prefix(prefix: pd.DataFrame, bundle, recovery: str) -> pd.DataFrame:
    if recovery == "Training median":
        return bundle.medians.median_fill(prefix, bundle.selected_sensors)
    if recovery == "Causal forward fill":
        return bundle.medians.causal_forward_fill(prefix, bundle.selected_sensors)
    raise ValueError(f"Unknown recovery method: {recovery}")


def predict_prefix(model_name: str, model, prefix: pd.DataFrame, bundle) -> float:
    """Predict RUL using only the rows supplied in the causal prefix."""
    if prefix.empty:
        raise ValueError("At least one observed cycle is required.")

    if model_name in {"Random Forest", "XGBoost"}:
        features, _ = build_tabular_features(
            prefix, bundle.selected_sensors, bundle.rolling_window
        )
        latest = features.iloc[[-1]][bundle.feature_columns]
        matrix = bundle.tabular_scaler.transform(latest)
        prediction = float(model.predict(matrix)[0])
    elif model_name == "LSTM":
        scaled = scale_sequence_frame(
            prefix, bundle.selected_sensors, bundle.sequence_scaler
        )
        window = build_prefix_window(
            scaled, bundle.selected_sensors, bundle.lstm_window
        )
        prediction = float(predict_lstm(model, np.expand_dims(window, axis=0))[0])
    else:
        prediction = float(model.predict([prefix])[0])

    return max(0.0, prediction)


def predict_with_scenario(
    model_name: str,
    model,
    prefix: pd.DataFrame,
    bundle,
    scenario: str,
    recovery: str,
    seed: int,
) -> float:
    corrupted = corrupt_prefix(prefix, bundle, scenario, seed)
    recovered = recover_prefix(corrupted, bundle, recovery)
    if recovered[bundle.selected_sensors].isna().any().any():
        raise ValueError("Missing sensor values remain after recovery.")
    return predict_prefix(model_name, model, recovered, bundle)


def build_replay(
    engine: pd.DataFrame,
    observed_cycle: int,
    model_name: str,
    model,
    bundle,
    scenario: str,
    recovery: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the visible prefix and cycle-by-cycle predictions without future rows."""
    visible = observed_prefix(
        engine, int(engine[ID_COLUMN].iloc[0]), observed_cycle
    )
    corrupted = corrupt_prefix(visible, bundle, scenario, seed)
    rows = []
    for cycle in corrupted[CYCLE_COLUMN].tolist():
        cycle_prefix = corrupted[corrupted[CYCLE_COLUMN] <= cycle].copy()
        recovered = recover_prefix(cycle_prefix, bundle, recovery)
        rows.append(
            {
                CYCLE_COLUMN: int(cycle),
                "rul_pred": predict_prefix(model_name, model, recovered, bundle),
            }
        )
    return visible, pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="wide")
    st.title("Predictive Maintenance — Remaining Useful Life")
    st.caption(
        "NASA C-MAPSS FD001 demonstration. Predictions estimate remaining operating cycles; "
        "they are not validated real-world maintenance thresholds."
    )

    try:
        bundle = load_bundle()
        models = load_models()
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(str(exc))
        st.info(
            "From the repository root, install requirements and run: "
            "python scripts/download_data.py → python scripts/prepare_data.py → "
            "python scripts/train_models.py"
        )
        st.stop()

    engine_ids = sorted(map(int, bundle.official_test_df[ID_COLUMN].unique()))
    model_name = st.sidebar.selectbox(
        "Prediction model", ["XGBoost", "Random Forest", "LSTM", "Median baseline"]
    )
    scenario = st.sidebar.selectbox(
        "Sensor availability",
        ["Clean", "Random 10%", "Random 20%", "Five-cycle sensor gap"],
    )
    recovery = st.sidebar.selectbox(
        "Missing-value recovery", ["Causal forward fill", "Training median"]
    )
    seed = st.sidebar.selectbox("Random-mask seed", [7, 17, 27])
    engine_id = st.sidebar.selectbox("Engine ID", engine_ids)

    engine = bundle.official_test_df[
        bundle.official_test_df[ID_COLUMN] == engine_id
    ].sort_values(CYCLE_COLUMN).copy()
    max_cycle = int(engine[CYCLE_COLUMN].max())
    observed_cycle = st.sidebar.slider(
        "Observed cycle (replay cutoff)",
        min_value=int(engine[CYCLE_COLUMN].min()),
        max_value=max_cycle,
        value=max_cycle,
    )
    k = st.sidebar.radio("Inspection queue size", [5, 10], horizontal=True)

    model = models[model_name]
    visible, replay = build_replay(
        engine, observed_cycle, model_name, model, bundle, scenario, recovery, int(seed)
    )
    current_prediction = float(replay.iloc[-1]["rul_pred"])
    status = health_label(current_prediction)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Engine", str(engine_id))
    metric_cols[1].metric("Observed cycle", str(observed_cycle))
    metric_cols[2].metric("Predicted RUL", f"{current_prediction:.1f} cycles")
    metric_cols[3].metric("Display status", status)

    st.subheader("Observed sensor history")
    sensor = st.selectbox("Sensor to inspect", bundle.selected_sensors)
    st.pyplot(plot_sensor_history(visible, sensor), clear_figure=True)

    st.subheader("RUL replay")
    st.pyplot(plot_rul_history(replay), clear_figure=True)
    st.dataframe(replay.tail(10), use_container_width=True, hide_index=True)

    st.caption(HEALTH_STATUS_NOTE)
    st.caption(
        f"Scenario: {scenario} · Recovery: {recovery} · Mask seed: {seed}. "
        "All predictions and charts are restricted to the selected observed cycle."
    )

    with st.expander("Fleet inspection queue"):
        st.write(QUEUE_NOTE)
        if st.button("Calculate fleet queue", type="primary"):
            progress = st.progress(0)
            queue_rows = []
            for index, fleet_engine_id in enumerate(engine_ids, start=1):
                fleet_engine = bundle.official_test_df[
                    bundle.official_test_df[ID_COLUMN] == fleet_engine_id
                ].sort_values(CYCLE_COLUMN).copy()
                last_cycle = int(fleet_engine[CYCLE_COLUMN].max())
                pred = predict_with_scenario(
                    model_name,
                    model,
                    fleet_engine,
                    bundle,
                    scenario,
                    recovery,
                    int(seed),
                )
                queue_rows.append(
                    {
                        ID_COLUMN: fleet_engine_id,
                        "observed_cycle": last_cycle,
                        "rul_pred": pred,
                    }
                )
                progress.progress(index / len(engine_ids))
            queue = top_k_queue(pd.DataFrame(queue_rows), k=k)
            st.dataframe(queue, use_container_width=True, hide_index=True)
            st.download_button(
                "Download inspection queue CSV",
                queue.to_csv(index=False).encode("utf-8"),
                file_name=f"inspection_queue_top_{k}.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
