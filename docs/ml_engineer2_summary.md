# ML Engineer 2 Summary

## Responsibilities

This part covers the ML Engineer 2 tasks for the NASA C-MAPSS FD001 predictive-maintenance project:

- Initial EDA
- Sensor screening
- Missing-value recovery
- Shared preprocessing contract
- XGBoost training
- XGBoost robustness evaluation

## Sensor Screening

Sensor screening is performed using only the 80 training engines from the fixed development split.

Constant and near-constant sensors are removed before model training. The selected sensor list is then frozen and reused for validation and testing.

This avoids using validation or test information to decide which sensors to keep.

## Missing-Value Recovery

Two recovery strategies are implemented:

1. Training median replacement
2. Causal per-engine forward fill with training-median fallback

The median values are calculated using training engines only.

Forward fill uses only past observations from the same engine. Backward fill and future interpolation are not used because they would introduce future information.

## Processing Contract

For robustness experiments, preprocessing follows this order:

1. Apply the artificial sensor mask
2. Impute missing sensor values
3. Recompute causal features
4. Apply the saved scaler
5. Generate the RUL prediction

The same frozen preprocessing rules are used for all compared models.

## XGBoost Model

XGBoost is trained as a regression model for Remaining Useful Life prediction.

Model selection is based on clean validation-prefix RMSE, with MAE used as a tie-breaker.

The selected model is frozen before official test evaluation and missing-sensor robustness experiments.

## Clean Test Result

XGBoost produced the strongest clean official FD001 test result among the tested models:

- MAE: 15.15 cycles
- RMSE: 19.59 cycles

## Missing-Sensor Robustness

XGBoost was evaluated under:

- 10% random missing sensor values
- 20% random missing sensor values
- A five-cycle short sensor gap

Both median replacement and causal forward fill were compared.

Forward fill generally preserved XGBoost performance better than median replacement, especially under 20% missingness and the short sensor-gap scenario.

The missingness experiment is an artificial robustness stress test and does not represent naturally missing NASA sensor data.