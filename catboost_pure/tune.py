"""Optuna hyperparameter search using last CV fold as objective."""
import pandas as pd
import numpy as np
import optuna
from pathlib import Path
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "dataset"
TRAIN_PATH = DATA_DIR / "train_dataset.csv"
VALID_PATH = DATA_DIR / "valid_features.csv"
CATBOOST_INFO_DIR = ROOT / "catboost_info"
MAX_CAPACITY = 90.09
N_TURBINES = 26
TURBINE_CAPACITY = 3.465
SEED = 42
TARGET = "Выработка. Результирующий расчет"

exec(open(ROOT / "train_predict.py").read().split('print("Loading data...")')[0])

# Remove near-zero importance features (precipitation is noise for wind power)
LOW_IMPORTANCE = {"rain", "showers", "snowfall", "any_precip"}

FEATURE_COLS_PRUNED = [c for c in FEATURE_COLS_BASE if c not in LOW_IMPORTANCE]

print("Preparing data...")
train_raw = pd.read_csv(TRAIN_PATH)
valid_raw  = pd.read_csv(VALID_PATH)
train_feat = engineer_features(train_raw)
valid_feat = engineer_features(valid_raw)
combined = pd.concat([train_feat, valid_feat], ignore_index=True)
combined, rolling_cols = add_rolling_features(combined)

# Prune noisy short-window rolling std cols; keep all lag/delta/long-window features
LOW_ROLLING = {c for c in rolling_cols if c.endswith("_r3s") or c.endswith("_r6s")}
FEATURE_COLS = FEATURE_COLS_PRUNED + [c for c in rolling_cols if c not in LOW_ROLLING]
print(f"Features: {len(FEATURE_COLS)} (pruned {len(LOW_IMPORTANCE) + len(LOW_ROLLING)} noisy cols)")

n_train = len(train_feat)
train_df = combined.iloc[:n_train].sort_values("datetime").reset_index(drop=True)
X_all = train_df[FEATURE_COLS]
y_all = train_df[TARGET]

# Use only the last fold for tuning (most representative of test period)
folds = list(TimeSeriesSplit(n_splits=5).split(X_all))
tr_idx, val_idx = folds[-1]
X_tr, y_tr = X_all.iloc[tr_idx], y_all.iloc[tr_idx]
X_val, y_val = X_all.iloc[val_idx], y_all.iloc[val_idx]
print(f"Tuning on last fold: train={len(tr_idx)}, val={len(val_idx)} rows")

pool_tr  = Pool(X_tr, y_tr)
pool_val = Pool(X_val, y_val)


def objective(trial):
    params = dict(
        iterations=4000,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        depth=trial.suggest_int("depth", 5, 9),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 10, 80),
        random_strength=trial.suggest_float("random_strength", 0.05, 3.0, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 2.0),
        subsample=trial.suggest_float("subsample", 0.55, 1.0),
        colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.55, 1.0),
        loss_function="RMSE",
        eval_metric="RMSE",
        early_stopping_rounds=200,
        verbose=0,
        random_seed=SEED,
        thread_count=-1,
        train_dir=str(CATBOOST_INFO_DIR),
    )
    model = CatBoostRegressor(**params)
    model.fit(pool_tr, eval_set=pool_val, use_best_model=True)
    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))


N_TRIALS = 80
print(f"\nRunning {N_TRIALS} Optuna trials...\n")

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best = study.best_params
best_rmse = study.best_value
print(f"\nBest val RMSE: {best_rmse:.4f}")
print("Best params:")
for k, v in best.items():
    print(f"  {k}: {v}")

# Save best params for use in train_predict.py
import json
result = {"rmse": best_rmse, "params": best, "feature_cols": FEATURE_COLS}
with open(ROOT / "best_params.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nSaved to best_params.json")
