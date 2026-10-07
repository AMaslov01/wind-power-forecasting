# Method

This project forecasts hourly output for the 90.09 MW Azov wind farm. The
evaluation period covers the first quarter of 2026, with an additional 24-hour
forecast for 18 May 2026.

## Data

The solution uses three organizer-provided files:

- historical weather, turbine-status, and generation observations from
  2022-2025;
- Q1 2026 features without the generation target;
- an operational 2026 file containing post-Q1 observations and the 24 rows to
  forecast for 18 May.

The original CSV files are not distributed in this repository. Public weather
and regional-energy sources are used only when their coverage and time range
are compatible with the forecast setting.

## Physics prior

The physical baseline approximates the Siemens Gamesa SG 3.4-132 power curve.
It combines rotor-equivalent wind speed, a piecewise power curve, air-density
correction, turbine availability, and farm capacity limits.

The baseline is used as both a prediction anchor and a feature. Residual models
learn corrections to it instead of relearning the wind-to-power relation from
scratch.

## Features

The final pipeline uses a compact set of feature groups:

- calendar and cyclic time features;
- wind speed and direction at multiple heights;
- gust, pressure, temperature, precipitation, and cloud measurements;
- physical power, operating-zone, density, and availability features;
- short lags, rolling statistics, and weather-change features;
- guarded public-weather features from Open-Meteo/GFS, Meteostat, and NASA
  POWER when available;
- monthly regional renewable-energy context.

Experimental sources and feature families remain in the research code, but
they are not required by the documented final configuration.

## Validation and models

Validation follows the deployment season: Q1 2023, Q1 2024, and Q1 2025 are
used as temporal backtests. This avoids the leakage that a random split would
introduce in a forecasting problem.

The final ensemble combines direct and physics-residual CatBoost and
HistGradientBoosting regressors. Blend weights are non-negative and selected
from out-of-fold predictions. Specialist regressors cover different turbine
operating zones.

All final predictions are checked for shape, missing values, and the physical
range `[0, 90.09]` MW.

## 2026 adaptation

Known post-Q1 observations can estimate a conservative scalar correction for
the Q1 forecast. The correction is accepted only when validation constraints on
mean shift, maximum adjustment, and zero-output behavior are satisfied.

## Reproducibility limits

This repository publishes the code and experiment design, not a self-contained
artifact bundle. The organizer data and trained weights cannot be redistributed.
An end-to-end run therefore requires the local files listed in
[`dataset/README.md`](../dataset/README.md). Results may also vary when external
weather providers revise their archives.
