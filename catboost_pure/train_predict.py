import json
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "dataset"
TRAIN_PATH = DATA_DIR / "train_dataset.csv"
VALID_PATH = DATA_DIR / "valid_features.csv"
OUTPUT_PATH = ROOT / "predictions.csv"
CATBOOST_INFO_DIR = ROOT / "catboost_info"

MAX_CAPACITY = 90.09
N_TURBINES = 26
TURBINE_CAPACITY = 3.465
SEED = 42

TARGET = "Выработка. Результирующий расчет"


def engineer_features(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["METEOFORECASTHOUR_OPENM_Datetime"])
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["week_of_year"] = df["datetime"].dt.isocalendar().week.astype(int)
    df["season"] = ((df["month"] % 12) // 3).astype(int)

    df["hour_sin"] = np.sin(df["hour_of_day"] * 2 * np.pi / 24)
    df["hour_cos"] = np.cos(df["hour_of_day"] * 2 * np.pi / 24)
    df["month_sin"] = np.sin(df["month"] * 2 * np.pi / 12)
    df["month_cos"] = np.cos(df["month"] * 2 * np.pi / 12)
    df["doy_sin"] = np.sin(df["day_of_year"] * 2 * np.pi / 365)
    df["doy_cos"] = np.cos(df["day_of_year"] * 2 * np.pi / 365)

    for h in [10, 80, 120, 180]:
        col = f"wind_direction_{h}m"
        df[f"wdir_{h}m_sin"] = np.sin(df[col] * 2 * np.pi)
        df[f"wdir_{h}m_cos"] = np.cos(df[col] * 2 * np.pi)

    for h in [10, 80, 120, 180]:
        df[f"ws_{h}m_sq"] = df[f"wind_speed_{h}m"] ** 2
        df[f"ws_{h}m_cb"] = df[f"wind_speed_{h}m"] ** 3

    # Wind direction veer between heights (atmospheric stability proxy)
    for (h1, h2) in [(10, 80), (80, 120), (80, 180)]:
        diff = (df[f"wind_direction_{h2}m"] - df[f"wind_direction_{h1}m"]) % 1.0
        diff = diff.where(diff <= 0.5, diff - 1.0)
        df[f"veer_{h1}_{h2}"] = diff * 360.0

    R = 287.05
    T_k = df["temperature_80m"] + 273.15
    P_pa = df["pressure_msl"] * 100
    df["air_density"] = P_pa / (R * T_k)
    df["wpd_80m"] = 0.5 * df["air_density"] * df["ws_80m_cb"]

    df["shear_10_80"] = df["wind_speed_80m"] - df["wind_speed_10m"]
    df["shear_80_120"] = df["wind_speed_120m"] - df["wind_speed_80m"]
    df["shear_80_180"] = df["wind_speed_180m"] - df["wind_speed_80m"]
    df["shear_10_180"] = df["wind_speed_180m"] - df["wind_speed_10m"]
    df["shear_ratio"] = df["wind_speed_180m"] / (df["wind_speed_10m"] + 0.1)
    df["gust_ratio"] = df["wind_gusts_10m"] / (df["wind_speed_10m"] + 0.1)
    df["turbulence"] = df["shear_10_180"] / (df["wind_speed_80m"] + 0.1)

    df["available_turbines"] = N_TURBINES - df["Кол-во_ВЭУ_в_ремонте"]
    df["available_capacity"] = df["available_turbines"] * TURBINE_CAPACITY
    df["repair_ratio"] = df["Кол-во_ВЭУ_в_ремонте"] / N_TURBINES

    # SG 3.4-132 (Gamesa G132-3.465): cut-in=3.0, rated=10.3, cut-out=25.0 m/s
    v = df["wind_speed_80m"].values
    v_cutin, v_rated, v_cutout = 3.0, 10.3, 25.0
    cf = np.zeros_like(v)
    ramp = (v >= v_cutin) & (v < v_rated)
    rated_mask = (v >= v_rated) & (v <= v_cutout)
    cf[ramp] = ((v[ramp] - v_cutin) / (v_rated - v_cutin)) ** 3
    cf[rated_mask] = 1.0
    df["theoretical_cf"] = cf
    df["theoretical_power"] = cf * df["available_capacity"].values

    df["wpd_120m"] = 0.5 * df["air_density"] * df["ws_120m_cb"]

    df["precip_total"] = df["rain"] + df["showers"] + df["snowfall"]
    df["any_precip"] = (df["precip_total"] > 0).astype(int)
    df["temp_diff"] = df["temperature_80m"] - df["temperature_120m"]

    return df


def add_rolling_features(df_combined):
    """Add rolling wind persistence features. df_combined must be sorted by datetime."""
    df = df_combined.sort_values("datetime").reset_index(drop=True)

    rolling_configs = [
        ("wind_speed_80m",  [3, 6, 12, 24, 48]),
        ("wind_speed_120m", [6, 24]),
        ("wind_speed_180m", [6, 24]),
        ("wind_gusts_10m",  [3, 6]),
        ("wpd_80m",         [3, 6, 24]),
        ("wpd_120m",        [6, 24]),
    ]
    new_cols = []
    for col, windows in rolling_configs:
        for w in windows:
            mean_col = f"{col}_r{w}m"
            std_col  = f"{col}_r{w}s"
            df[mean_col] = df[col].rolling(w, min_periods=1).mean()
            df[std_col]  = df[col].rolling(w, min_periods=1).std().fillna(0)
            new_cols += [mean_col, std_col]

    # Lags for ws_80m: 1-6h captures strong temporal autocorrelation (r=0.95/0.88/0.82 at lag 1/2/3)
    for lag in [1, 2, 3, 6]:
        col = f"wind_speed_80m_lag{lag}"
        df[col] = df["wind_speed_80m"].shift(lag).fillna(df["wind_speed_80m"])
        new_cols.append(col)

    # WPD lags: directly physics-related (WPD ∝ v³ ∝ power)
    for lag in [1, 2, 3]:
        col = f"wpd_80m_lag{lag}"
        df[col] = df["wpd_80m"].shift(lag).fillna(df["wpd_80m"])
        new_cols.append(col)

    # Wind ramp: change in wind speed over last N hours (ramp-event detector)
    for delta_h in [3, 6]:
        col = f"ws_80m_delta{delta_h}"
        df[col] = (df["wind_speed_80m"]
                   - df["wind_speed_80m"].shift(delta_h).fillna(df["wind_speed_80m"]))
        new_cols.append(col)

    # lag-1 for 120m and gusts
    for col in ["wind_speed_120m", "wind_gusts_10m"]:
        lag_col = f"{col}_lag1"
        df[lag_col] = df[col].shift(1).fillna(df[col])
        new_cols.append(lag_col)

    return df, new_cols


FEATURE_COLS_BASE = [
    "month", "hour_of_day", "day_of_year", "day_of_week", "week_of_year", "season",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
    "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m",
    "ws_10m_sq", "ws_80m_sq", "ws_120m_sq", "ws_180m_sq",
    "ws_10m_cb", "ws_80m_cb", "ws_120m_cb", "ws_180m_cb",
    "wdir_10m_sin", "wdir_10m_cos", "wdir_80m_sin", "wdir_80m_cos",
    "wdir_120m_sin", "wdir_120m_cos", "wdir_180m_sin", "wdir_180m_cos",
    "wind_gusts_10m", "gust_ratio", "turbulence",
    "shear_10_80", "shear_80_120", "shear_80_180", "shear_10_180", "shear_ratio",
    "veer_10_80", "veer_80_120", "veer_80_180",
    "temperature_80m", "temperature_120m", "temp_diff",
    "pressure_msl", "air_density", "wpd_80m", "wpd_120m",
    "precip_total",
    "cloud_cover_low",
    "Кол-во_ВЭУ_в_ремонте", "available_turbines", "available_capacity", "repair_ratio",
    "theoretical_cf", "theoretical_power",
]

with open(ROOT / "best_params.json") as _f:
    _bp = json.load(_f)

# Tuned params from Optuna + new physics features added to the feature set
CATBOOST_PARAMS = dict(
    iterations=5000,
    **_bp["params"],
    loss_function="RMSE",
    eval_metric="RMSE",
    early_stopping_rounds=200,
    verbose=500,
    random_seed=SEED,
    thread_count=-1,
    train_dir=str(CATBOOST_INFO_DIR),
)

# Feature list from tuned run; wpd_120m and its rolling cols are new additions
_TUNED_FEATURES = set(_bp["feature_cols"])
_NEW_FEATURES = {"wpd_120m", "wpd_120m_r6m", "wpd_120m_r24m", "wpd_120m_r6s", "wpd_120m_r24s"}


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.clip(np.array(y_pred), 0, MAX_CAPACITY)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    nonzero = y_true > 0.5
    mape = (
        np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
        if nonzero.sum() > 0
        else float("nan")
    )

    error_rate = np.mean(np.abs(y_true - y_pred) > MAX_CAPACITY * 0.10) * 100

    return rmse, mae, mape, error_rate


print("Loading data...")
train_raw = pd.read_csv(TRAIN_PATH)
valid_raw = pd.read_csv(VALID_PATH)
print(f"Train: {len(train_raw)} rows | Valid: {len(valid_raw)} rows")

print("Engineering features...")
train_feat = engineer_features(train_raw)
valid_feat = engineer_features(valid_raw)

# Rolling features require full sorted sequence (train then valid contiguous)
combined = pd.concat([train_feat, valid_feat], ignore_index=True)
combined, rolling_cols = add_rolling_features(combined)

FEATURE_COLS = [
    f for f in FEATURE_COLS_BASE + rolling_cols
    if f in _TUNED_FEATURES or f in _NEW_FEATURES
]

n_train = len(train_feat)
train_df = combined.iloc[:n_train].copy()
valid_df = combined.iloc[n_train:].copy()

train_df = train_df.sort_values("datetime").reset_index(drop=True)

X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET]
X_valid = valid_df[FEATURE_COLS]

N_SPLITS = 5
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

oof_preds = np.zeros(len(X_train))
cv_rows = []
models = []
best_iterations = []

header = f"{'Fold':<5} {'RMSE':>9} {'MAE':>9} {'MAPE%':>9} {'ErrRate%':>10} {'Iters':>7}"
print("\n" + header)
print("-" * len(header))

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
    X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(
        Pool(X_tr, y_tr),
        eval_set=Pool(X_val, y_val),
        use_best_model=True,
    )
    models.append(model)
    best_iterations.append(model.get_best_iteration() + 1)

    raw_preds = model.predict(X_val)
    preds = np.clip(raw_preds, 0, MAX_CAPACITY)
    oof_preds[val_idx] = preds

    rmse, mae, mape, error_rate = compute_metrics(y_val.values, raw_preds)
    cv_rows.append(dict(rmse=rmse, mae=mae, mape=mape, error_rate=error_rate))

    print(
        f"{fold:<5} {rmse:>9.3f} {mae:>9.3f} {mape:>9.2f} {error_rate:>10.2f} {best_iterations[-1]:>7}"
    )

cv_df = pd.DataFrame(cv_rows)
print("-" * len(header))
print(
    f"{'Mean':<5} {cv_df['rmse'].mean():>9.3f} {cv_df['mae'].mean():>9.3f} "
    f"{cv_df['mape'].mean():>9.2f} {cv_df['error_rate'].mean():>10.2f}"
)

oof_rmse, oof_mae, oof_mape, oof_er = compute_metrics(y_train.values, oof_preds)
print(f"\nOOF  {oof_rmse:>9.3f} {oof_mae:>9.3f} {oof_mape:>9.2f} {oof_er:>10.2f}")

# Final model on ALL training data.
# Last fold is most representative (largest training set, closest to full data size).
# Scale up slightly to account for the additional ~20% data vs last fold.
final_iters = max(int(best_iterations[-1] * 1.15), 500)
print(f"\nTraining final model on full dataset ({final_iters} iterations, no early stopping)...")

final_params = {**CATBOOST_PARAMS}
final_params.pop("early_stopping_rounds")
final_params["iterations"] = final_iters
final_params["verbose"] = 500

final_model = CatBoostRegressor(**final_params)
final_model.fit(Pool(X_train, y_train))

valid_preds = np.clip(final_model.predict(X_valid), 0, MAX_CAPACITY)

output_df = pd.DataFrame({TARGET: valid_preds})
output_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nPredictions saved to {OUTPUT_PATH}")
print(f"Rows: {len(output_df)} | min={valid_preds.min():.2f} | max={valid_preds.max():.2f} | mean={valid_preds.mean():.2f} MW")
