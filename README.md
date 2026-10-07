# Physics-Informed Wind Power Forecasting

A hackathon solution for hourly generation forecasting at the 90.09 MW Azov
wind farm. The evaluation period covers **1 January-31 March 2026**.

> This is a code release, not a standalone demo. The organizer data and trained
> model artifacts cannot be redistributed, so a fresh clone cannot run
> end-to-end without those local files.

## Approach

The final solution combines:

- a physics-informed baseline derived from the Siemens Gamesa SG 3.4-132 power
  curve;
- public weather features from Open-Meteo/GFS, Meteostat, and NASA POWER when
  available;
- 45 months of regional renewable-energy context;
- temporal Q1 backtests for 2023, 2024, and 2025;
- CatBoost and HistGradientBoosting models with non-negative,
  validation-optimized blend weights;
- specialist regressors for different wind operating zones;
- a guarded 2026 calibration step whose correction is rejected when validation
  constraints fail.

Predictions are checked for shape, missing values, and the physical range
`[0, 90.09]` MW. See [`docs/method.md`](docs/method.md) for the technical
summary.

## Running the code

The required private files are listed in
[`dataset/README.md`](dataset/README.md).

If you have both the organizer data and the trained artifact bundle, run:

```bash
bash run_solution.sh
```

This restores and validates:

```text
outputs/predictions_q1.csv       # 2,126 hourly Q1 values
outputs/predictions_may18.csv    # 24 hourly values for 18 May 2026
outputs/RUN_REPORT.json          # checksums, versions, validation statistics
```

If you have the organizer data but not the artifact bundle, retrain the models:

```bash
bash run_solution.sh --retrain
```

GPU-enabled CatBoost training is available with:

```bash
bash run_solution.sh --retrain --gpu
```

To validate the local data and artifact layout before an expensive run:

```bash
bash run_solution.sh --check-only
```

## Required local files

For inference from the original artifact bundle:

```text
dataset/train_dataset.csv
dataset/valid_features.csv
dataset/3888f9f2-9bda-4b2c-94af-5562668bce86_test_dataset.csv
model_weights/ensemble_manifest.json
model_weights/predictions_q1.csv
model_weights/predictions_may18.csv
```

`model_weights/` can be rebuilt with `bash run_solution.sh --retrain` when all
three organizer files are available.

## Output validation

`run_solution.sh` invokes `physics/validate_solution.py` and checks:

- Q1 output: one CSV column, exactly 2,126 rows, no missing values;
- 18 May output: one CSV column, exactly 24 rows, no missing values;
- every prediction is within `[0, 90.09]` MW;
- expected data files and artifact metadata are present.

## Scope and limitations

Generated forecasts, trained weights, organizer data, caches, and submission
files are intentionally excluded from Git.

External weather archives may change after the competition, so a later retrain
is not guaranteed to reproduce the original predictions byte-for-byte. Optional
source adapters remain as research code and are not required by the documented
final configuration.
