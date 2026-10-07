# Physics-Informed Wind Farm Generation Forecasting

A reproducible pipeline for hourly generation forecasting at the 90.09 MW Azov wind farm. This repository contains our hackathon solution for the Q1 evaluation period from **2026-01-01 to 2026-03-31**.

## Approach

The final public solution uses a stable `V14 no-CDS` pipeline:

- a physics-informed baseline derived from the Siemens Gamesa SG 3.4-132 turbine power curve;
- public weather features from Open-Meteo/GFS, Meteostat cache, and NASA POWER when available;
- 45 months of monthly renewable-energy context;
- temporal Q1 backtests for 2023, 2024, and 2025;
- CatBoost and HistGradientBoosting models combined with non-negative validation-optimized weights;
- regime-specific HGB experts for different wind operating zones;
- weather-dynamics features;
- a guarded April-May 2026 actuals adapter whose Q1 correction is rejected automatically when validation constraints fail.

The pipeline validates output shape, missing values, expected artifacts, and physical bounds. Forecasts outside `[0, 90.09]` MW are rejected.

## Quick start

Place the official hackathon CSV files in `dataset/`, keep the supplied `model_weights/` directory next to this README, and run:

```bash
bash run_solution.sh
```

The command creates:

```text
outputs/predictions_q1.csv       # 2,126 hourly values for leaderboard evaluation
outputs/predictions_may18.csv    # 24 hourly values for 2026-05-18
outputs/RUN_REPORT.json          # checksums, package versions, validation statistics
```

To rebuild the full solution from source data:

```bash
bash run_solution.sh --retrain
```

For a Colab/GPU-style rebuild:

```bash
bash run_solution.sh --retrain --gpu
```

To validate the package before running any expensive step:

```bash
bash run_solution.sh --check-only
```

## Required local files

The original datasets are not stored on GitHub. Place them locally as follows:

```text
dataset/train_dataset.csv
dataset/valid_features.csv
dataset/3888f9f2-9bda-4b2c-94af-5562668bce86_test_dataset.csv
model_weights/ensemble_manifest.json
model_weights/predictions_q1.csv
model_weights/predictions_may18.csv
```

If `model_weights/` is absent, run `bash run_solution.sh --retrain` to rebuild the stable V14 no-CDS ensemble and write a fresh artifact set.

The standard one-command path restores verified forecasts from the artifact bundle and uses only the Python standard library. The `--retrain` mode installs the complete ML dependencies, rebuilds the model from local datasets, rewrites `model_weights/`, and validates the resulting files.

## Output validation

`run_solution.sh` invokes `physics/validate_solution.py` and checks:

- Q1 forecast: one CSV column, exactly 2,126 rows, no NaNs, values within `[0, 90.09]`;
- May 18 forecast: one CSV column, exactly 24 rows, no NaNs, values within `[0, 90.09]`;
- required dataset files and expected row counts;
- `ensemble_manifest.json` and both forecast snapshots in the artifact bundle.

## Repository notes

Generated forecasts, model artifacts, local datasets, cache directories, and the final submission directory are intentionally excluded from Git. They belong in the hackathon ZIP/package rather than the public repository.

Optional hooks for Copernicus, Renewables Ninja, and other research sources remain in the code for future work, but the final one-command pipeline does not depend on them.
