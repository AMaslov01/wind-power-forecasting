"""Weather-bias features for the physics-first model family.

This module deliberately does not use the target. It turns external/reanalysis
weather disagreement into corrected rotor-speed physics columns that can be
used by train folds and the public validation set in the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DATETIME_COL = "METEOFORECASTHOUR_OPENM_Datetime"
REPAIR_COL = "Кол-во_ВЭУ_в_ремонте"
N_TOTAL_TURBINES = 26
P_RATED_FARM = 90.09
ROTOR_DIAMETER = 132.0
ROTOR_RADIUS = ROTOR_DIAMETER / 2.0
RHO_STANDARD = 1.225


@dataclass(frozen=True)
class WeatherBiasConfig:
    """Conservative, target-free weather correction controls."""

    agreement_scale_mps: float = 4.0
    max_speed_delta_mps: float = 2.25
    era5_100m_blend: float = 0.55
    era5_10m_blend: float = 0.35


def _rotor_equivalent_speed(ws10, ws80, ws120, ws180, hub_height: float) -> np.ndarray:
    measured_heights = np.array([10.0, 80.0, 120.0, 180.0])
    log_h = np.log(measured_heights)
    n_bins = 21
    rotor_h = np.linspace(hub_height - ROTOR_RADIUS, hub_height + ROTOR_RADIUS, n_bins)
    chords = 2.0 * np.sqrt(np.maximum(ROTOR_RADIUS**2 - (rotor_h - hub_height) ** 2, 0.0))
    weights = chords * (rotor_h[1] - rotor_h[0])
    log_rotor_h = np.log(np.clip(rotor_h, 1.0, None))

    speeds = np.stack(
        [
            np.asarray(ws10, dtype=float),
            np.asarray(ws80, dtype=float),
            np.asarray(ws120, dtype=float),
            np.asarray(ws180, dtype=float),
        ],
        axis=1,
    )
    v_at_h = np.empty((speeds.shape[0], n_bins))
    for j, lh in enumerate(log_rotor_h):
        idx = np.clip(np.searchsorted(log_h, lh) - 1, 0, len(log_h) - 2)
        t = (lh - log_h[idx]) / (log_h[idx + 1] - log_h[idx])
        v_at_h[:, j] = speeds[:, idx] + (speeds[:, idx + 1] - speeds[:, idx]) * t
    return np.cbrt(np.maximum((v_at_h**3 * weights).sum(axis=1) / weights.sum(), 0.0))


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


def _local_speed_at_100m(frame: pd.DataFrame) -> pd.Series:
    eps = 0.1
    ws80 = frame["wind_speed_80m"].clip(lower=eps)
    ws120 = frame["wind_speed_120m"].clip(lower=eps)
    alpha = np.log(ws120 / ws80) / np.log(120.0 / 80.0)
    return ws80 * (100.0 / 80.0) ** alpha


def _agreement(delta: pd.Series, scale: float) -> pd.Series:
    return np.exp(-((delta.abs() / scale) ** 2)).clip(0.0, 1.0)


def add_weather_bias_features(
    frame: pd.DataFrame,
    cfg,
    *,
    config: WeatherBiasConfig = WeatherBiasConfig(),
) -> pd.DataFrame:
    """Add target-free corrected weather and physics columns.

    The correction is intentionally conservative: external ERA5 100 m wind can
    shift the rotor profile only when it agrees reasonably with the local
    forecast. The result is a physical alternative baseline, not a leaderboard
    offset.
    """

    out = frame.copy()
    local_100m = _local_speed_at_100m(out)
    out["weather_local_speed_100m"] = local_100m

    if "ext_era5_wind_speed_100m" in out:
        delta_100m = pd.to_numeric(out["ext_era5_wind_speed_100m"], errors="coerce") - local_100m
    else:
        delta_100m = pd.Series(0.0, index=out.index)
    delta_100m = delta_100m.fillna(0.0).clip(
        -config.max_speed_delta_mps,
        config.max_speed_delta_mps,
    )
    agreement_100m = _agreement(delta_100m, config.agreement_scale_mps)
    smooth_delta_100m = config.era5_100m_blend * agreement_100m * delta_100m
    out["weather_bias_delta_100m"] = delta_100m
    out["weather_bias_agreement_100m"] = agreement_100m
    out["weather_bias_smooth_delta_100m"] = smooth_delta_100m

    if "ext_era5_wind_speed_10m" in out:
        delta_10m = pd.to_numeric(out["ext_era5_wind_speed_10m"], errors="coerce") - out["wind_speed_10m"]
    else:
        delta_10m = pd.Series(0.0, index=out.index)
    delta_10m = delta_10m.fillna(0.0).clip(
        -config.max_speed_delta_mps,
        config.max_speed_delta_mps,
    )
    agreement_10m = _agreement(delta_10m, config.agreement_scale_mps)
    smooth_delta_10m = config.era5_10m_blend * agreement_10m * delta_10m
    out["weather_bias_delta_10m"] = delta_10m
    out["weather_bias_agreement_10m"] = agreement_10m

    out["wind_speed_10m_weather_corrected"] = (out["wind_speed_10m"] + smooth_delta_10m).clip(lower=0.0)
    out["wind_speed_80m_weather_corrected"] = (out["wind_speed_80m"] + smooth_delta_100m).clip(lower=0.0)
    out["wind_speed_120m_weather_corrected"] = (out["wind_speed_120m"] + smooth_delta_100m).clip(lower=0.0)
    out["wind_speed_180m_weather_corrected"] = (out["wind_speed_180m"] + 0.85 * smooth_delta_100m).clip(lower=0.0)

    out["wind_speed_eq_weather_corrected"] = _rotor_equivalent_speed(
        out["wind_speed_10m_weather_corrected"].values,
        out["wind_speed_80m_weather_corrected"].values,
        out["wind_speed_120m_weather_corrected"].values,
        out["wind_speed_180m_weather_corrected"].values,
        cfg.hub_height,
    )
    out["wind_speed_eq_weather_corrected_cubed"] = out["wind_speed_eq_weather_corrected"] ** 3

    density_corr = (out["air_density"].values / RHO_STANDARD) ** cfg.density_correction_exp
    p_std = _empirical_power_curve(out["wind_speed_eq_weather_corrected"].values, cfg)
    p_per_turbine = np.clip(
        p_std * density_corr * cfg.efficiency_factor,
        0.0,
        cfg.p_rated_per_turbine,
    )
    out["P_physics_per_turbine_weather_corrected"] = p_per_turbine
    out["P_physics_farm_weather_corrected"] = np.clip(
        p_per_turbine * out["n_working"].values,
        0.0,
        P_RATED_FARM,
    )
    out["P_physics_weather_delta"] = (
        out["P_physics_farm_weather_corrected"] - out["P_physics_farm"]
    )

    try:
        pbin = pd.qcut(
            out["P_physics_farm_weather_corrected"].rank(method="first"),
            q=9,
            labels=False,
            duplicates="drop",
        )
        out["physics_v2_potential_bin"] = pd.Series(pbin, index=out.index).astype(float)
    except ValueError:
        out["physics_v2_potential_bin"] = 0.0
    out["physics_v2_midpower_flag"] = out["physics_v2_potential_bin"].between(3, 6).astype(float)
    out["physics_v2_q1_flag"] = out["month"].isin([1, 2, 3]).astype(float)

    direction_deg = (out["wind_direction_120m"] % 1.0) * 360.0
    out["physics_v2_direction_sector"] = np.floor(direction_deg / 45.0).clip(0, 7).astype(int)
    return out
