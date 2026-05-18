# V8 Wind-Tech Research Notes

These notes capture what is actually useful from the public wind-power repositories reviewed for V8.

## WindFM

- Source: https://github.com/shiyu-coder/WindFM
- Useful idea: pretrained wind-power foundation model with multivariate inputs: wind speed, wind direction, power, density, temperature, pressure.
- Implementation choice: V8 keeps WindFM diagnostic-only. If `windfm_oof_predictions.csv` and `windfm_valid_predictions.csv` exist, `--enable-windfm-diagnostic` joins them as fold-safe features. It does not train/download WindFM inside the main pipeline because that would add large optional dependencies and possible Colab instability.

## Lag and Rolling Priors

- Source: https://github.com/EnesAgirman/Wind-Power-Generation-Forecasting
- Useful idea: lagged and rolling time-series features improve stability, but target-derived lags can leak.
- Implementation choice: V8 adds `--enable-lag-residual-features`, which fits month/hour and wind-regime residual priors inside each fold from the fold train slice only. Validation and final prediction rows never contribute to the priors.

## Hybrid Regime Handling

- Source: https://github.com/ShashwatArghode/Wind-Energy-Prediction-using-LSTM
- Useful idea: pure LSTM struggles with hard turbine rules such as cut-in/cut-out; hybrid tree/rule handling is better.
- Implementation choice: V8 adds `wind_regime_code` and optional `hgb_regime_direct` / `hgb_regime_residual` models with `--enable-regime-models`. Each regime-specific HGB falls back to a global HGB when the regime has too few rows.

## Physical Interactions and Feature Selection

- Source: https://github.com/vchaparro/wind-power-forecasting
- Useful idea: derive wind speed/direction from vector components, cyclic direction encodings, inverse/temperature interactions, and feature selection.
- Implementation choice: V8 adds `--enable-source-interactions` for external-source speed x direction, speed x temperature, speed x density, source-source deltas, and shear x density interactions. Mutual-information selection is not added to the main pipeline because the blend already uses model-side feature selection/regularization and a fixed feature matrix improves reproducibility.

## Generic LSTM/XGBoost Demo Repos

- Sources:
  - https://github.com/Devtech99/Wind-Power-Forecasting-ML
  - https://github.com/Sk70249/Wind-Energy-Analysis-and-Forecast-using-Deep-Learning-LSTM
  - https://github.com/MainakVerse/Wind-Energy-Predictor
- Useful idea: XGBoost/tree ensembles and theoretical-curve/lag features usually beat generic sequence models on small tabular wind datasets.
- Implementation choice: V8 does not add a generic LSTM path. The current CatBoost/HGB ensemble is already the stronger tabular track; the extracted improvements are physics priors, residual priors, regime models, and optional WindFM diagnostics.
