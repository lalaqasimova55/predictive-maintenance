# Predictive Maintenance — NASA C-MAPSS FD001

Remaining Useful Life (RUL) prediction using Random Forest, XGBoost and a compact LSTM, with a median baseline and robustness tests under artificial sensor missingness.

## ML Engineer 4 deliverables

This branch contains repository setup, leakage-safe split contracts, integrity checks, the median baseline, a common experiment manifest, metric comparison utilities and the Streamlit control layer.

## Reproducibility contract

- Dataset: NASA C-MAPSS FD001.
- Engine split: 80/20 with seed 42.
- Validation history: first 70% of each held-out engine trajectory.
- Engine ID is metadata only and never a model feature.
- RUL is measured in operating cycles.
- Clean models are frozen before missingness experiments.
- Random 10% and 20% missingness use seeds 7, 17 and 27.
- Missing sensors are masked before imputation and feature generation.
- Forward fill is causal and falls back to training medians.
- Predictions are clamped to zero.
- Dashboard labels: RUL > 60 Healthy, 30 < RUL <= 60 Warning, RUL <= 30 Critical.

## Run

Generate the shared experiment manifest with:
    python run_experiments.py

Run the dashboard with:
    streamlit run app.py

Model-specific training and preprocessing modules can consume config.yaml, split_manifest.json, validation.py and model_interface.py.

## Scope

The project evaluates FD001 RUL prediction and robustness to artificial missing sensor measurements. It does not claim a new ML algorithm or real-world maintenance thresholds.
