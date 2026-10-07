# Dataset

The organizer data is not licensed for redistribution. To run the pipeline,
place the following files in this directory:

```text
dataset/train_dataset.csv
dataset/valid_features.csv
dataset/3888f9f2-9bda-4b2c-94af-5562668bce86_test_dataset.csv
```

- `train_dataset.csv` contains 2022-2025 features and the target column
  `Выработка. Результирующий расчет`.
- `valid_features.csv` contains the Q1 2026 forecast rows without the target.
- The operational test file contains known post-Q1 observations followed by
  24 rows for the 18 May 2026 forecast.

Farm metadata used by the solution: 26 Siemens Gamesa turbines, 90.09 MW
installed capacity, near `46.8268, 38.7179`.
