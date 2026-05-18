"""Fold-local latent availability and wind-residual model for physics_first_v2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


TARGET = "Выработка. Результирующий расчет"
REPAIR_COL = "Кол-во_ВЭУ_в_ремонте"
N_TOTAL_TURBINES = 26
P_RATED_FARM = 90.09
RHO_STANDARD = 1.225
SEED = 42


PHYSICS_FIRST_V2_FEATURES = [
    "month",
    "hour_of_day",
    REPAIR_COL,
    "n_working_monthly_avg",
    "wind_speed_eq",
    "wind_speed_eq_weather_corrected",
    "weather_bias_delta_100m",
    "weather_bias_agreement_100m",
    "weather_bias_smooth_delta_100m",
    "weather_bias_delta_10m",
    "weather_bias_agreement_10m",
    "P_physics_farm",
    "P_physics_farm_weather_corrected",
    "P_physics_weather_delta",
    "physics_v2_potential_bin",
    "physics_v2_midpower_flag",
    "physics_v2_q1_flag",
    "physics_v2_direction_sector",
    "shear_120_80",
    "shear_180_80",
    "shear_80_10",
    "direction_shear_120_80",
    "temperature_80m",
    "temperature_120m",
    "pressure_msl",
    "air_density_ratio",
    "rain",
    "showers",
    "snowfall",
    "cloud_cover_low",
    "ice_risk",
    "turbulence_intensity",
    "gust_factor",
]


@dataclass
class PhysicsFirstV2Calibrator:
    wind_delta_model: HistGradientBoostingRegressor | None
    availability_delta_model: HistGradientBoostingRegressor | None
    wind_delta_global: float
    availability_delta_global: float
    feature_cols: list[str]
    wind_delta_shrink: float = 0.45
    availability_delta_shrink: float = 0.55


def initialize_physics_first_v2_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    base = out.get("P_physics_farm_weather_corrected", out["P_physics_farm"])
    out["wind_speed_eq_physics_v2"] = out.get(
        "wind_speed_eq_weather_corrected",
        out["wind_speed_eq"],
    )
    out["wind_delta_physics_v2"] = 0.0
    out["availability_delta_physics_v2"] = 0.0
    out["n_working_latent_v2"] = out["n_working_monthly_avg"]
    out["availability_latent_v2"] = out["availability_monthly_avg"]
    out["P_physics_per_turbine_v2"] = out.get(
        "P_physics_per_turbine_weather_corrected",
        out["P_physics_per_turbine"],
    )
    out["P_physics_farm_latent_v2"] = np.clip(base, 0.0, P_RATED_FARM)
    out["P_physics_farm_latent_v2_delta"] = out["P_physics_farm_latent_v2"] - out["P_physics_farm"]
    return out


def _feature_cols(frame: pd.DataFrame) -> list[str]:
    return [col for col in PHYSICS_FIRST_V2_FEATURES if col in frame.columns]


def _hgb(random_offset: int, *, max_leaf_nodes: int, min_samples_leaf: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=260,
        learning_rate=0.035,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=0.35,
        early_stopping=True,
        validation_fraction=0.18,
        random_state=SEED + random_offset,
    )


def _implied_partial_wind_delta(frame: pd.DataFrame, cfg) -> pd.Series:
    monthly_n = frame["n_working_monthly_avg"].clip(lower=1.0)
    target_per_turbine = (frame[TARGET] / monthly_n).clip(lower=0.0)
    density_corr = (frame["air_density"].values / RHO_STANDARD) ** cfg.density_correction_exp
    gross_rated = cfg.p_rated_per_turbine * density_corr * cfg.efficiency_factor
    ratio = (target_per_turbine.values / np.maximum(gross_rated, 1e-6)).clip(0.0, 0.985)
    implied = cfg.v_cut_in + (cfg.v_rated - cfg.v_cut_in) * ratio ** (1.0 / cfg.cubic_exponent)
    base_wind = frame.get("wind_speed_eq_weather_corrected", frame["wind_speed_eq"]).values
    delta = pd.Series(implied - base_wind, index=frame.index)
    reliable = (
        (target_per_turbine > 0.35)
        & (target_per_turbine < pd.Series(gross_rated, index=frame.index) * 0.96)
        & frame.get("wind_speed_eq_weather_corrected", frame["wind_speed_eq"]).between(
            cfg.v_cut_in - 0.8,
            cfg.v_rated + 0.8,
        )
    )
    return delta.where(reliable).clip(-2.5, 2.5)


def _effective_availability_delta(frame: pd.DataFrame) -> pd.Series:
    per_turbine = frame.get(
        "P_physics_per_turbine_weather_corrected",
        frame["P_physics_per_turbine"],
    )
    reliable = (
        (per_turbine > 0.75)
        & np.isfinite(per_turbine)
        & np.isfinite(frame[TARGET])
        & (frame[TARGET] > 0.0)
    )
    effective_n = (frame[TARGET] / per_turbine.replace(0.0, np.nan)).clip(0.0, N_TOTAL_TURBINES)
    delta = effective_n - frame["n_working_monthly_avg"]
    return delta.where(reliable).clip(-4.0, 3.0)


def fit_physics_first_v2_calibrator(train_frame: pd.DataFrame, cfg) -> PhysicsFirstV2Calibrator:
    frame = initialize_physics_first_v2_features(train_frame)
    cols = _feature_cols(frame)
    if not cols or TARGET not in frame:
        return PhysicsFirstV2Calibrator(None, None, 0.0, 0.0, cols)

    wind_target = _implied_partial_wind_delta(frame, cfg)
    wind_mask = wind_target.notna()
    wind_global = float(wind_target[wind_mask].mean()) if bool(wind_mask.any()) else 0.0
    wind_model = None
    if int(wind_mask.sum()) >= 500:
        wind_model = _hgb(101, max_leaf_nodes=16, min_samples_leaf=80)
        wind_model.fit(frame.loc[wind_mask, cols], wind_target.loc[wind_mask])

    availability_target = _effective_availability_delta(frame)
    availability_mask = availability_target.notna()
    availability_global = (
        float(availability_target[availability_mask].mean())
        if bool(availability_mask.any())
        else 0.0
    )
    availability_model = None
    if int(availability_mask.sum()) >= 500:
        availability_model = _hgb(211, max_leaf_nodes=18, min_samples_leaf=90)
        availability_model.fit(
            frame.loc[availability_mask, cols],
            availability_target.loc[availability_mask],
        )

    return PhysicsFirstV2Calibrator(
        wind_delta_model=wind_model,
        availability_delta_model=availability_model,
        wind_delta_global=float(np.clip(wind_global, -1.25, 1.25)),
        availability_delta_global=float(np.clip(availability_global, -2.0, 1.5)),
        feature_cols=cols,
    )


def _empirical_power_curve(v: np.ndarray, cfg) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    p = np.zeros_like(v)
    partial = (v >= cfg.v_cut_in) & (v < cfg.v_rated)
    span = cfg.v_rated - cfg.v_cut_in
    p[partial] = cfg.p_rated_per_turbine * (
        (v[partial] - cfg.v_cut_in) / span
    ) ** cfg.cubic_exponent
    rated = (v >= cfg.v_rated) & (v < cfg.v_cut_out)
    p[rated] = cfg.p_rated_per_turbine
    return p


def apply_physics_first_v2_calibrator(
    frame: pd.DataFrame,
    calibrator: PhysicsFirstV2Calibrator,
    cfg,
) -> pd.DataFrame:
    out = initialize_physics_first_v2_features(frame)
    cols = [col for col in calibrator.feature_cols if col in out.columns]

    if calibrator.wind_delta_model is not None and cols:
        wind_delta = calibrator.wind_delta_model.predict(out[cols])
    else:
        wind_delta = np.full(len(out), calibrator.wind_delta_global, dtype=float)
    wind_delta = np.clip(wind_delta, -2.5, 2.5) * calibrator.wind_delta_shrink
    out["wind_delta_physics_v2"] = wind_delta
    out["wind_speed_eq_physics_v2"] = (
        out.get("wind_speed_eq_weather_corrected", out["wind_speed_eq"]).values + wind_delta
    ).clip(min=0.0)

    if calibrator.availability_delta_model is not None and cols:
        availability_delta = calibrator.availability_delta_model.predict(out[cols])
    else:
        availability_delta = np.full(len(out), calibrator.availability_delta_global, dtype=float)
    availability_delta = np.clip(availability_delta, -4.0, 3.0) * calibrator.availability_delta_shrink
    out["availability_delta_physics_v2"] = availability_delta
    out["n_working_latent_v2"] = (
        out["n_working_monthly_avg"].astype(float).values + availability_delta
    ).clip(0.0, N_TOTAL_TURBINES)
    out["availability_latent_v2"] = out["n_working_latent_v2"] / N_TOTAL_TURBINES

    density_corr = (out["air_density"].values / RHO_STANDARD) ** cfg.density_correction_exp
    p_std = _empirical_power_curve(out["wind_speed_eq_physics_v2"].values, cfg)
    out["P_physics_per_turbine_v2"] = np.clip(
        p_std * density_corr * cfg.efficiency_factor,
        0.0,
        cfg.p_rated_per_turbine,
    )
    out["P_physics_farm_latent_v2"] = np.clip(
        out["P_physics_per_turbine_v2"].values * out["n_working_latent_v2"].values,
        0.0,
        P_RATED_FARM,
    )
    out["P_physics_farm_latent_v2_delta"] = out["P_physics_farm_latent_v2"] - out["P_physics_farm"]
    return out
