"""
Temporal ensemble for hourly wind farm output.

Physics baseline uses the Siemens Gamesa SG 3.4-132 spec power curve
(empirical cubic ramp, V_RATED=10.3 m/s), modulated by air density and a
turbine efficiency factor. CatBoost models learn either the target directly
or the residual against the physical baseline; their predictions are blended
with validator-calibrated weights.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import random
import tempfile
import warnings
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from time import perf_counter
from time import sleep
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

from empirical_power_curve_model import (
    add_empirical_curve_v4_guarded_features,
    apply_empirical_curve_features,
    apply_empirical_curve_v4_features,
    default_empirical_curve_v4_config,
    fit_empirical_curve_calibrator,
    fit_empirical_curve_v4_calibrator,
    initialize_empirical_curve_features,
    initialize_empirical_curve_v4_features,
    load_empirical_curve_v4_config,
    prediction_similarity_weights,
)
from latent_availability_model import (
    apply_physics_first_v2_calibrator,
    fit_physics_first_v2_calibrator,
    initialize_physics_first_v2_features,
)
from weather_bias_model import add_weather_bias_features


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "dataset"
TRAIN_PATH = DATA_DIR / "train_dataset.csv"
VALID_PATH = DATA_DIR / "valid_features.csv"
OUTPUT_PATH = ROOT / "predictions.csv"
EXTERNAL_WEATHER_DIR = DATA_DIR / "external_weather"
METEOSTAT_CACHE_PATH = EXTERNAL_WEATHER_DIR / "meteostat_hourly_azov.csv"
HGB_FAMILY_PARAMS_PATH = ROOT / "hgb_family_params.json"
EMPIRICAL_CURVE_V4_CONFIG_PATH = ROOT / "empirical_curve_v4_config.json"
OPENMETEO_FORECAST_COVERAGE_REPORT_PATH = ROOT / "openmeteo_forecast_coverage_report.json"
SO_UPS_RES_CACHE_PATH = DATA_DIR / "external_energy" / "so_ups_res_monthly.csv"

DATETIME_COL = "METEOFORECASTHOUR_OPENM_Datetime"
TARGET = "Выработка. Результирующий расчет"
REPAIR_COL = "Кол-во_ВЭУ_в_ремонте"

N_TOTAL_TURBINES = 26
P_RATED_FARM = 90.09
HUB_HEIGHT_M = 84.0
ROTOR_DIAMETER = 132.0
ROTOR_RADIUS = ROTOR_DIAMETER / 2.0
A_ROTOR = np.pi * ROTOR_RADIUS**2

R_SPECIFIC_AIR = 287.05
RHO_STANDARD = 1.225
SEED = 42
BEST_ITER_MULTIPLIER = 1.10
CATBOOST_TASK_TYPE = "CPU"
CATBOOST_DEVICES: str | None = None
AZOV_LAT = 46.8268455973
AZOV_LON = 38.7179393185
LOCAL_TIMEZONE = "Europe/Moscow"
HTTP_TIMEOUT_SECONDS = 90
DEFAULT_EXTERNAL_SOURCE_TIMEOUT_SECONDS = 45
MIN_METEOSTAT_COVERAGE = 0.90
SAFE_METEOSTAT_COLUMNS = [
    "ext_meteostat_temperature",
    "ext_meteostat_dew_point",
    "ext_meteostat_humidity",
    "ext_meteostat_wind_direction",
    "ext_meteostat_wind_speed",
    "ext_meteostat_pressure_msl",
    "ext_meteostat_condition_code",
]
OPENMETEO_HISTORICAL_FORECAST_MODELS = {
    "gfs": "gfs_global",
    "icon": "icon_global",
    "ecmwf": "ecmwf_ifs025",
}
OPENMETEO_FORECAST_MODEL_CATALOG = {
    **OPENMETEO_HISTORICAL_FORECAST_MODELS,
    "icon_eu": "icon_eu",
    "cma": "cma_grapes_global",
    "arpege": "arpege_world",
}
OPENMETEO_FORECAST_SETS = {
    "gfs_only": ("gfs",),
    "icon_only": ("icon",),
    "ecmwf_only": ("ecmwf",),
    "core_open_forecast": ("gfs", "icon", "ecmwf"),
    "core_pressure_forecast": ("gfs", "icon", "ecmwf"),
    "pressure_level_forecast": ("gfs", "icon", "ecmwf"),
    "core_plus_icon_eu": ("gfs", "icon", "ecmwf", "icon_eu"),
    "core_plus_cma": ("gfs", "icon", "ecmwf", "cma"),
    "core_plus_arpege": ("gfs", "icon", "ecmwf", "arpege"),
    "expanded_open_forecast": ("gfs", "icon", "ecmwf", "icon_eu", "cma", "arpege"),
    "expanded_pressure_forecast": ("gfs", "icon", "ecmwf", "icon_eu", "cma", "arpege"),
}
OPENMETEO_PRESSURE_FORECAST_SETS = {
    "core_pressure_forecast",
    "pressure_level_forecast",
    "expanded_pressure_forecast",
}
OPENMETEO_PRESSURE_LEVELS = ("1000hPa", "925hPa", "900hPa")
OPENMETEO_FILL_POLICIES = {"legacy", "strict"}
OPENMETEO_COVERAGE_MIN_DEFAULT = 0.80
EXTERNAL_CACHE_POLICIES = {"cache_first", "refresh", "offline"}
EXTERNAL_SOURCE_SETS = {
    "baseline": (),
    "strict_open": ("nasa_power",),
    "strict_copernicus": ("nasa_power", "copernicus_era5"),
    "strict_keyed": ("nasa_power", "copernicus_era5", "meteoblue"),
    "tech_features": (),
    "all_strict": ("nasa_power", "copernicus_era5", "meteoblue"),
    "climatology_probe": ("global_wind_atlas", "renewables_ninja", "windatlas_xyz"),
    "diagnostic_forecast": ("windy", "noaa_nomads_gfs"),
    # Backward-compatible aliases from V7. They stay strict-safe: forecast-only
    # and pre-2020 sources no longer enter all_available by accident.
    "open_free": ("nasa_power",),
    "keyed_safe": ("nasa_power", "copernicus_era5", "meteoblue"),
    "all_available": ("nasa_power", "copernicus_era5", "meteoblue"),
}
SO_UPS_RES_POLICIES = {"legacy_best_gap", "clean_48", "custom_exclude"}
SO_UPS_RES_LEGACY_BEST_GAP_MONTHS = ("2023-01", "2023-05", "2024-05")
EXTERNAL_SOURCE_REPORT_PATH = ROOT / "external_source_report.json"
EXTERNAL_SOURCE_REPORT_VERSION = "v9"
ENABLE_SOURCE_INTERACTIONS = False
ENABLE_LAG_RESIDUAL_FEATURES = False
ENABLE_REGIME_PRIOR_V2 = False
ENABLE_REGIME_MODELS = False
ENABLE_REGIME_V2 = False
ENABLE_WINDFM_DIAGNOSTIC = False
ENABLE_HGB_QUANTILE_SISTERS = False
ENABLE_DIRECTION_SECTOR_FEATURES = False
ENABLE_WEATHER_ANALOG_RESIDUAL = False
ENABLE_WEATHER_DYNAMICS_FEATURES = False
ENABLE_MULTI_REGIME_FEATURES = False
ENABLE_MULTI_REGIME_EXPERTS = False
REGIME_CALIBRATION_MODES = {"none", "affine", "isotonic"}
REGIME_CALIBRATION_MODE = "none"
REGIME_V2_SOFT_BOUNDARY_WIDTH = 0.75
REGIME_V2_MIN_EXPERT_WEIGHT = 0.35
REGIME_CALIBRATION_MIN_ROWS = 500
REGIME_AFFINE_SHRINKAGE = 0.35
REGIME_ISOTONIC_SHRINKAGE = 0.25
BASELINE_EXTRA_SOURCE_FEATURES = {
    "ext_nasa_",
    "ext_cds_era5_",
    "ext_meteoblue_",
    "ext_gwa_",
    "ext_renewables_ninja_",
    "ext_windy_",
    "ext_noaa_nomads_",
    "ext_windatlas_xyz_",
}
NASA_POWER_PARAMETERS = (
    "T2M",
    "T2MDEW",
    "RH2M",
    "PS",
    "PRECTOTCORR",
    "WS10M",
    "WS50M",
    "WD10M",
    "WD50M",
)
OPENMETEO_SURFACE_HOURLY = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "snowfall",
    "cloud_cover",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_100m",
    "wind_speed_120m",
    "wind_speed_180m",
    "wind_direction_10m",
    "wind_direction_80m",
    "wind_direction_100m",
    "wind_direction_120m",
    "wind_direction_180m",
    "wind_gusts_10m",
    "boundary_layer_height",
    "cape",
    "total_column_integrated_water_vapour",
    "freezing_level_height",
]


@dataclass(frozen=True)
class PhysicsConfig:
    """Tunable physical parameters of the turbine and farm.

    Defaults match the Siemens Gamesa SG 3.4-132 manufacturer spec.
    """
    v_cut_in: float = 3.0
    v_rated: float = 10.3
    v_cut_out: float = 25.0
    p_rated_per_turbine: float = 3.465
    cubic_exponent: float = 3.0
    hub_height: float = HUB_HEIGHT_M
    efficiency_factor: float = 1.0
    ice_threshold: float = 2.0
    density_correction_exp: float = 1.0


MANUFACTURER_PHYSICS = PhysicsConfig()

PUBLIC_BEST_PHYSICS = PhysicsConfig(
    v_cut_in=2.3393477705815173,
    v_rated=10.83096885835045,
    v_cut_out=23.96363070301636,
    p_rated_per_turbine=3.3260919549786356,
    cubic_exponent=1.498293155285144,
    hub_height=HUB_HEIGHT_M,
    efficiency_factor=1.119626992504275,
    ice_threshold=3.205444959029653,
    density_correction_exp=0.45408166948375117,
)

LEGACY_TUNED_PHYSICS = PhysicsConfig(
    v_cut_in=2.7266,
    v_rated=9.6478,
    v_cut_out=24.5148,
    p_rated_per_turbine=3.4002,
    cubic_exponent=2.3246,
    hub_height=HUB_HEIGHT_M,
    efficiency_factor=1.0586,
    ice_threshold=1.9882,
    density_correction_exp=1.19,
)

DEFAULT_PHYSICS = PUBLIC_BEST_PHYSICS

PHYSICS_PRESETS = {
    "public_best": PUBLIC_BEST_PHYSICS,
    "legacy_tuned": LEGACY_TUNED_PHYSICS,
    "manufacturer": MANUFACTURER_PHYSICS,
}
ACTIVE_PHYSICS_CONFIG = DEFAULT_PHYSICS


TEMPORAL_BASE_COLS = [
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_120m",
    "wind_speed_180m",
    "wind_gusts_10m",
    "temperature_80m",
    "pressure_msl",
    "wind_speed_eq",
    "wind_speed_eq_cubed",
    "wind_speed_eq_weather_corrected",
    "wind_speed_eq_physics_v2",
    "P_physics_farm",
    "P_physics_farm_weather_corrected",
    "P_physics_farm_latent_v2",
    "ext_era5_wind_speed_10m",
    "ext_era5_wind_speed_100m",
    "ext_era5_pressure_msl",
    "ext_era5_wpd_100m",
    "ext_cds_era5_wind_speed_10m",
    "ext_cds_era5_wind_speed_100m",
    "ext_cds_era5_pressure_msl",
    "ext_cds_era5_wpd_100m",
    "ext_cds_era5_air_density_2m",
    "ext_omfc_gfs_wind_speed_100m",
    "ext_omfc_icon_wind_speed_100m",
    "ext_omfc_ecmwf_wind_speed_100m",
    "ext_omfc_icon_eu_wind_speed_100m",
    "ext_omfc_cma_wind_speed_100m",
    "ext_omfc_arpege_wind_speed_100m",
    "ext_omfc_consensus_wind_speed_100m",
    "ext_omfc_spread_wind_speed_100m",
    "ext_omfc_consensus_pressure_hub_wind_speed",
    "ext_omfc_spread_pressure_hub_wind_speed",
    "ext_nasa_ws10m",
    "ext_nasa_ws50m",
    "ext_nasa_wpd_50m",
    "ext_blend_wind_speed_10m",
    "ext_blend_wind_gusts_10m",
    "ext_blend_pressure_msl",
    "ext_blend_cloud_cover_low",
]
TEMPORAL_LAGS = [1, 2, 3, 6, 12]
ROLLING_WINDOWS = [3, 7, 13]

ROLLING_STD_COLS = [
    "wind_speed_80m",
    "wind_speed_120m",
    "wind_gusts_10m",
    "P_physics_farm",
    "P_physics_farm_weather_corrected",
    "P_physics_farm_latent_v2",
    "ext_era5_wind_speed_100m",
    "ext_cds_era5_wind_speed_100m",
    "ext_cds_era5_wpd_100m",
    "ext_omfc_gfs_wind_speed_100m",
    "ext_omfc_icon_wind_speed_100m",
    "ext_omfc_ecmwf_wind_speed_100m",
    "ext_omfc_icon_eu_wind_speed_100m",
    "ext_omfc_cma_wind_speed_100m",
    "ext_omfc_arpege_wind_speed_100m",
    "ext_omfc_consensus_wind_speed_100m",
    "ext_omfc_consensus_pressure_hub_wind_speed",
    "ext_nasa_ws50m",
    "ext_blend_wind_speed_10m",
    "ext_blend_wind_gusts_10m",
]
ROLLING_STD_WINDOWS = [12, 24]
DELTA_COLS = [
    "wind_speed_80m",
    "wind_speed_120m",
    "wind_speed_eq_weather_corrected",
    "wind_speed_eq_physics_v2",
    "ext_era5_wind_speed_100m",
    "ext_cds_era5_wind_speed_100m",
    "ext_cds_era5_wpd_100m",
    "ext_omfc_gfs_wind_speed_100m",
    "ext_omfc_icon_wind_speed_100m",
    "ext_omfc_ecmwf_wind_speed_100m",
    "ext_omfc_icon_eu_wind_speed_100m",
    "ext_omfc_cma_wind_speed_100m",
    "ext_omfc_arpege_wind_speed_100m",
    "ext_omfc_consensus_wind_speed_100m",
    "ext_omfc_consensus_pressure_hub_wind_speed",
    "ext_nasa_ws50m",
    "ext_blend_wind_speed_10m",
]
DELTA_LAGS = [3, 6]

WEATHER_DYNAMICS_BASE_COLS = [
    "wind_speed_eq",
    "P_physics_farm",
    "ext_era5_wind_speed_100m",
    "ext_omfc_gfs_wind_speed_100m",
    "ext_nasa_ws50m",
    "ext_cds_era5_wind_speed_100m",
    "ext_nasa_wpd_50m",
    "ext_cds_era5_wpd_100m",
]
WEATHER_DYNAMICS_EWM_SPANS = [3, 6]
WEATHER_DYNAMICS_MEDIAN_WINDOWS = [3, 7]
WEATHER_DYNAMICS_TREND_WINDOWS = [6, 12]
WEATHER_SOURCE_DISAGREEMENT_PAIRS = [
    ("ext_omfc_gfs_wind_speed_100m", "ext_nasa_ws50m", "gfs100_minus_nasa50"),
    ("ext_omfc_gfs_wind_speed_100m", "ext_cds_era5_wind_speed_100m", "gfs100_minus_cds100"),
    ("ext_nasa_ws50m", "ext_cds_era5_wind_speed_100m", "nasa50_minus_cds100"),
]

CAT_FEATURES = ["month", "hour_of_day"]
LEGACY_PUBLIC_BEST_BLEND_WEIGHTS = {
    "cat7_direct": 0.02000000000000001,
    "cat7_residual": 0.19719382772737326,
    "cat5_residual": 0.07190431917472785,
    "cat5_direct": 0.02402947164403205,
    "hgb_direct": 0.6868723814538669,
}
HGB_FAMILY_BLEND_WEIGHTS = {
    "cat7_direct": 0.020,
    "cat7_residual": 0.150,
    "cat5_residual": 0.070,
    "cat5_direct": 0.020,
    "hgb_direct": 0.550,
    "hgb_direct_smooth": 0.120,
    "hgb_direct_deep": 0.040,
    "hgb_residual_smooth": 0.030,
}
HGB_OPTUNA_FAMILY_BLEND_WEIGHTS = {
    "cat7_direct": 0.005,
    "cat7_residual": 0.040,
    "cat5_residual": 0.005,
    "cat5_direct": 0.040,
    "hgb_direct_deep": 0.300,
    "hgb_residual_smooth": 0.450,
    "hgb_opt_direct_1": 0.080,
    "hgb_opt_direct_2": 0.030,
    "hgb_opt_residual_1": 0.040,
    "hgb_opt_residual_2": 0.010,
}
HGB_OPTUNA_ONLY_BLEND_WEIGHTS = {
    "hgb_direct": 0.025,
    "hgb_direct_smooth": 0.050,
    "hgb_direct_deep": 0.275,
    "hgb_residual_smooth": 0.450,
    "hgb_opt_direct_1": 0.100,
    "hgb_opt_direct_2": 0.030,
    "hgb_opt_residual_1": 0.060,
    "hgb_opt_residual_2": 0.010,
}
PHYSICS_FIRST_V2_BLEND_WEIGHTS = {
    "physics_v2_prior": 0.420,
    "hgb_physics_v2_residual_smooth": 0.360,
    "hgb_physics_v2_residual_deep": 0.160,
    "hgb_direct_smooth": 0.060,
}
EMPIRICAL_CURVE_V3_BLEND_WEIGHTS = {
    "empirical_curve_v3_prior": 0.200,
    "hgb_empirical_v3_residual_smooth": 0.300,
    "hgb_empirical_v3_residual_deep": 0.150,
    "hgb_direct_smooth": 0.200,
    "hgb_residual_smooth": 0.150,
}
EMPIRICAL_CURVE_V4_BLEND_WEIGHTS = {
    "empirical_curve_v4_prior": 0.020,
    "hgb_empirical_v4_residual_deep": 0.400,
    "hgb_empirical_v4_residual_smooth": 0.080,
    "hgb_direct_smooth": 0.280,
    "hgb_residual_smooth": 0.220,
}
EMPIRICAL_CURVE_V4_GUARDED_BLEND_WEIGHTS = {
    "empirical_curve_v4_guarded_prior": 0.020,
    "hgb_empirical_v4_guarded_residual_deep": 0.180,
    "hgb_empirical_v3_residual_deep": 0.250,
    "hgb_direct_smooth": 0.280,
    "hgb_residual_smooth": 0.270,
}
LGB_PROBE_BLEND_WEIGHTS = {
    "cat7_direct": 0.018,
    "cat7_residual": 0.135,
    "cat5_residual": 0.063,
    "cat5_direct": 0.018,
    "hgb_direct": 0.495,
    "hgb_direct_smooth": 0.108,
    "hgb_direct_deep": 0.036,
    "hgb_residual_smooth": 0.027,
    "lgb_direct_l1": 0.060,
    "lgb_residual_l1": 0.040,
}
FOREST_PROBE_BLEND_WEIGHTS = {
    "cat7_direct": 0.005,
    "cat7_residual": 0.040,
    "cat5_residual": 0.005,
    "cat5_direct": 0.050,
    "hgb_direct": 0.070,
    "hgb_direct_smooth": 0.020,
    "hgb_direct_deep": 0.280,
    "hgb_residual_smooth": 0.360,
    "lgb_residual_l1": 0.060,
    "etr_direct": 0.035,
    "etr_direct_smooth": 0.035,
    "etr_residual": 0.045,
    "etr_residual_smooth": 0.035,
}
FOREST_ONLY_BLEND_WEIGHTS = {
    "etr_direct": 0.180,
    "etr_direct_smooth": 0.220,
    "etr_residual": 0.270,
    "etr_residual_smooth": 0.330,
}
ALL_FAMILY_BLEND_WEIGHTS = {
    "cat7_direct": 0.003,
    "cat7_residual": 0.030,
    "cat5_residual": 0.003,
    "cat5_direct": 0.055,
    "hgb_direct": 0.030,
    "hgb_direct_smooth": 0.003,
    "hgb_direct_deep": 0.290,
    "hgb_residual_smooth": 0.390,
    "lgb_direct_l1": 0.010,
    "lgb_residual_l1": 0.120,
    "etr_direct": 0.015,
    "etr_direct_smooth": 0.003,
    "etr_residual": 0.038,
    "etr_residual_smooth": 0.010,
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    target_mode: str
    weight: float


MODEL_SETS = {
    "legacy424": [
        ModelSpec("cat7_direct", "direct", LEGACY_PUBLIC_BEST_BLEND_WEIGHTS["cat7_direct"]),
        ModelSpec("cat7_residual", "residual", LEGACY_PUBLIC_BEST_BLEND_WEIGHTS["cat7_residual"]),
        ModelSpec("cat5_residual", "residual", LEGACY_PUBLIC_BEST_BLEND_WEIGHTS["cat5_residual"]),
        ModelSpec("cat5_direct", "direct", LEGACY_PUBLIC_BEST_BLEND_WEIGHTS["cat5_direct"]),
        ModelSpec("hgb_direct", "direct", LEGACY_PUBLIC_BEST_BLEND_WEIGHTS["hgb_direct"]),
    ],
    "hgb_family": [
        ModelSpec("cat7_direct", "direct", HGB_FAMILY_BLEND_WEIGHTS["cat7_direct"]),
        ModelSpec("cat7_residual", "residual", HGB_FAMILY_BLEND_WEIGHTS["cat7_residual"]),
        ModelSpec("cat5_residual", "residual", HGB_FAMILY_BLEND_WEIGHTS["cat5_residual"]),
        ModelSpec("cat5_direct", "direct", HGB_FAMILY_BLEND_WEIGHTS["cat5_direct"]),
        ModelSpec("hgb_direct", "direct", HGB_FAMILY_BLEND_WEIGHTS["hgb_direct"]),
        ModelSpec("hgb_direct_smooth", "direct", HGB_FAMILY_BLEND_WEIGHTS["hgb_direct_smooth"]),
        ModelSpec("hgb_direct_deep", "direct", HGB_FAMILY_BLEND_WEIGHTS["hgb_direct_deep"]),
        ModelSpec("hgb_residual_smooth", "residual", HGB_FAMILY_BLEND_WEIGHTS["hgb_residual_smooth"]),
    ],
    "hgb_optuna_family": [
        ModelSpec("cat7_direct", "direct", HGB_OPTUNA_FAMILY_BLEND_WEIGHTS["cat7_direct"]),
        ModelSpec("cat7_residual", "residual", HGB_OPTUNA_FAMILY_BLEND_WEIGHTS["cat7_residual"]),
        ModelSpec("cat5_residual", "residual", HGB_OPTUNA_FAMILY_BLEND_WEIGHTS["cat5_residual"]),
        ModelSpec("cat5_direct", "direct", HGB_OPTUNA_FAMILY_BLEND_WEIGHTS["cat5_direct"]),
        ModelSpec("hgb_direct_deep", "direct", HGB_OPTUNA_FAMILY_BLEND_WEIGHTS["hgb_direct_deep"]),
        ModelSpec("hgb_residual_smooth", "residual", HGB_OPTUNA_FAMILY_BLEND_WEIGHTS["hgb_residual_smooth"]),
        ModelSpec("hgb_opt_direct_1", "direct", HGB_OPTUNA_FAMILY_BLEND_WEIGHTS["hgb_opt_direct_1"]),
        ModelSpec("hgb_opt_direct_2", "direct", HGB_OPTUNA_FAMILY_BLEND_WEIGHTS["hgb_opt_direct_2"]),
        ModelSpec("hgb_opt_residual_1", "residual", HGB_OPTUNA_FAMILY_BLEND_WEIGHTS["hgb_opt_residual_1"]),
        ModelSpec("hgb_opt_residual_2", "residual", HGB_OPTUNA_FAMILY_BLEND_WEIGHTS["hgb_opt_residual_2"]),
    ],
    "hgb_optuna_only": [
        ModelSpec("hgb_direct", "direct", HGB_OPTUNA_ONLY_BLEND_WEIGHTS["hgb_direct"]),
        ModelSpec("hgb_direct_smooth", "direct", HGB_OPTUNA_ONLY_BLEND_WEIGHTS["hgb_direct_smooth"]),
        ModelSpec("hgb_direct_deep", "direct", HGB_OPTUNA_ONLY_BLEND_WEIGHTS["hgb_direct_deep"]),
        ModelSpec("hgb_residual_smooth", "residual", HGB_OPTUNA_ONLY_BLEND_WEIGHTS["hgb_residual_smooth"]),
        ModelSpec("hgb_opt_direct_1", "direct", HGB_OPTUNA_ONLY_BLEND_WEIGHTS["hgb_opt_direct_1"]),
        ModelSpec("hgb_opt_direct_2", "direct", HGB_OPTUNA_ONLY_BLEND_WEIGHTS["hgb_opt_direct_2"]),
        ModelSpec("hgb_opt_residual_1", "residual", HGB_OPTUNA_ONLY_BLEND_WEIGHTS["hgb_opt_residual_1"]),
        ModelSpec("hgb_opt_residual_2", "residual", HGB_OPTUNA_ONLY_BLEND_WEIGHTS["hgb_opt_residual_2"]),
    ],
    "physics_first_v2": [
        ModelSpec(
            "physics_v2_prior",
            "physics_v2_prior",
            PHYSICS_FIRST_V2_BLEND_WEIGHTS["physics_v2_prior"],
        ),
        ModelSpec(
            "hgb_physics_v2_residual_smooth",
            "physics_v2_residual",
            PHYSICS_FIRST_V2_BLEND_WEIGHTS["hgb_physics_v2_residual_smooth"],
        ),
        ModelSpec(
            "hgb_physics_v2_residual_deep",
            "physics_v2_residual",
            PHYSICS_FIRST_V2_BLEND_WEIGHTS["hgb_physics_v2_residual_deep"],
        ),
        ModelSpec("hgb_direct_smooth", "direct", PHYSICS_FIRST_V2_BLEND_WEIGHTS["hgb_direct_smooth"]),
    ],
    "empirical_curve_v3": [
        ModelSpec(
            "empirical_curve_v3_prior",
            "empirical_v3_prior",
            EMPIRICAL_CURVE_V3_BLEND_WEIGHTS["empirical_curve_v3_prior"],
        ),
        ModelSpec(
            "hgb_empirical_v3_residual_smooth",
            "empirical_v3_residual",
            EMPIRICAL_CURVE_V3_BLEND_WEIGHTS["hgb_empirical_v3_residual_smooth"],
        ),
        ModelSpec(
            "hgb_empirical_v3_residual_deep",
            "empirical_v3_residual",
            EMPIRICAL_CURVE_V3_BLEND_WEIGHTS["hgb_empirical_v3_residual_deep"],
        ),
        ModelSpec("hgb_direct_smooth", "direct", EMPIRICAL_CURVE_V3_BLEND_WEIGHTS["hgb_direct_smooth"]),
        ModelSpec("hgb_residual_smooth", "residual", EMPIRICAL_CURVE_V3_BLEND_WEIGHTS["hgb_residual_smooth"]),
    ],
    "empirical_curve_v4": [
        ModelSpec(
            "empirical_curve_v4_prior",
            "empirical_v4_prior",
            EMPIRICAL_CURVE_V4_BLEND_WEIGHTS["empirical_curve_v4_prior"],
        ),
        ModelSpec(
            "hgb_empirical_v4_residual_deep",
            "empirical_v4_residual",
            EMPIRICAL_CURVE_V4_BLEND_WEIGHTS["hgb_empirical_v4_residual_deep"],
        ),
        ModelSpec(
            "hgb_empirical_v4_residual_smooth",
            "empirical_v4_residual",
            EMPIRICAL_CURVE_V4_BLEND_WEIGHTS["hgb_empirical_v4_residual_smooth"],
        ),
        ModelSpec("hgb_direct_smooth", "direct", EMPIRICAL_CURVE_V4_BLEND_WEIGHTS["hgb_direct_smooth"]),
        ModelSpec("hgb_residual_smooth", "residual", EMPIRICAL_CURVE_V4_BLEND_WEIGHTS["hgb_residual_smooth"]),
    ],
    "empirical_curve_v4_guarded": [
        ModelSpec(
            "empirical_curve_v4_guarded_prior",
            "empirical_v4_guarded_prior",
            EMPIRICAL_CURVE_V4_GUARDED_BLEND_WEIGHTS["empirical_curve_v4_guarded_prior"],
        ),
        ModelSpec(
            "hgb_empirical_v4_guarded_residual_deep",
            "empirical_v4_guarded_residual",
            EMPIRICAL_CURVE_V4_GUARDED_BLEND_WEIGHTS["hgb_empirical_v4_guarded_residual_deep"],
        ),
        ModelSpec(
            "hgb_empirical_v3_residual_deep",
            "empirical_v3_residual",
            EMPIRICAL_CURVE_V4_GUARDED_BLEND_WEIGHTS["hgb_empirical_v3_residual_deep"],
        ),
        ModelSpec("hgb_direct_smooth", "direct", EMPIRICAL_CURVE_V4_GUARDED_BLEND_WEIGHTS["hgb_direct_smooth"]),
        ModelSpec("hgb_residual_smooth", "residual", EMPIRICAL_CURVE_V4_GUARDED_BLEND_WEIGHTS["hgb_residual_smooth"]),
    ],
    "lgb_probe": [
        ModelSpec("cat7_residual", "residual", LGB_PROBE_BLEND_WEIGHTS["cat7_residual"]),
        ModelSpec("cat5_residual", "residual", LGB_PROBE_BLEND_WEIGHTS["cat5_residual"]),
        ModelSpec("cat5_direct", "direct", LGB_PROBE_BLEND_WEIGHTS["cat5_direct"]),
        ModelSpec("hgb_direct", "direct", LGB_PROBE_BLEND_WEIGHTS["hgb_direct"]),
        ModelSpec("hgb_direct_smooth", "direct", LGB_PROBE_BLEND_WEIGHTS["hgb_direct_smooth"]),
        ModelSpec("hgb_direct_deep", "direct", LGB_PROBE_BLEND_WEIGHTS["hgb_direct_deep"]),
        ModelSpec("hgb_residual_smooth", "residual", LGB_PROBE_BLEND_WEIGHTS["hgb_residual_smooth"]),
        ModelSpec("lgb_direct_l1", "direct", LGB_PROBE_BLEND_WEIGHTS["lgb_direct_l1"]),
        ModelSpec("lgb_residual_l1", "residual", LGB_PROBE_BLEND_WEIGHTS["lgb_residual_l1"]),
    ],
    "forest_probe": [
        ModelSpec("cat7_direct", "direct", FOREST_PROBE_BLEND_WEIGHTS["cat7_direct"]),
        ModelSpec("cat7_residual", "residual", FOREST_PROBE_BLEND_WEIGHTS["cat7_residual"]),
        ModelSpec("cat5_residual", "residual", FOREST_PROBE_BLEND_WEIGHTS["cat5_residual"]),
        ModelSpec("cat5_direct", "direct", FOREST_PROBE_BLEND_WEIGHTS["cat5_direct"]),
        ModelSpec("hgb_direct", "direct", FOREST_PROBE_BLEND_WEIGHTS["hgb_direct"]),
        ModelSpec("hgb_direct_smooth", "direct", FOREST_PROBE_BLEND_WEIGHTS["hgb_direct_smooth"]),
        ModelSpec("hgb_direct_deep", "direct", FOREST_PROBE_BLEND_WEIGHTS["hgb_direct_deep"]),
        ModelSpec("hgb_residual_smooth", "residual", FOREST_PROBE_BLEND_WEIGHTS["hgb_residual_smooth"]),
        ModelSpec("lgb_residual_l1", "residual", FOREST_PROBE_BLEND_WEIGHTS["lgb_residual_l1"]),
        ModelSpec("etr_direct", "direct", FOREST_PROBE_BLEND_WEIGHTS["etr_direct"]),
        ModelSpec("etr_direct_smooth", "direct", FOREST_PROBE_BLEND_WEIGHTS["etr_direct_smooth"]),
        ModelSpec("etr_residual", "residual", FOREST_PROBE_BLEND_WEIGHTS["etr_residual"]),
        ModelSpec("etr_residual_smooth", "residual", FOREST_PROBE_BLEND_WEIGHTS["etr_residual_smooth"]),
    ],
    "forest_only_probe": [
        ModelSpec("etr_direct", "direct", FOREST_ONLY_BLEND_WEIGHTS["etr_direct"]),
        ModelSpec("etr_direct_smooth", "direct", FOREST_ONLY_BLEND_WEIGHTS["etr_direct_smooth"]),
        ModelSpec("etr_residual", "residual", FOREST_ONLY_BLEND_WEIGHTS["etr_residual"]),
        ModelSpec("etr_residual_smooth", "residual", FOREST_ONLY_BLEND_WEIGHTS["etr_residual_smooth"]),
    ],
    "all_family_probe": [
        ModelSpec("cat7_direct", "direct", ALL_FAMILY_BLEND_WEIGHTS["cat7_direct"]),
        ModelSpec("cat7_residual", "residual", ALL_FAMILY_BLEND_WEIGHTS["cat7_residual"]),
        ModelSpec("cat5_residual", "residual", ALL_FAMILY_BLEND_WEIGHTS["cat5_residual"]),
        ModelSpec("cat5_direct", "direct", ALL_FAMILY_BLEND_WEIGHTS["cat5_direct"]),
        ModelSpec("hgb_direct", "direct", ALL_FAMILY_BLEND_WEIGHTS["hgb_direct"]),
        ModelSpec("hgb_direct_smooth", "direct", ALL_FAMILY_BLEND_WEIGHTS["hgb_direct_smooth"]),
        ModelSpec("hgb_direct_deep", "direct", ALL_FAMILY_BLEND_WEIGHTS["hgb_direct_deep"]),
        ModelSpec("hgb_residual_smooth", "residual", ALL_FAMILY_BLEND_WEIGHTS["hgb_residual_smooth"]),
        ModelSpec("lgb_direct_l1", "direct", ALL_FAMILY_BLEND_WEIGHTS["lgb_direct_l1"]),
        ModelSpec("lgb_residual_l1", "residual", ALL_FAMILY_BLEND_WEIGHTS["lgb_residual_l1"]),
        ModelSpec("etr_direct", "direct", ALL_FAMILY_BLEND_WEIGHTS["etr_direct"]),
        ModelSpec("etr_direct_smooth", "direct", ALL_FAMILY_BLEND_WEIGHTS["etr_direct_smooth"]),
        ModelSpec("etr_residual", "residual", ALL_FAMILY_BLEND_WEIGHTS["etr_residual"]),
        ModelSpec("etr_residual_smooth", "residual", ALL_FAMILY_BLEND_WEIGHTS["etr_residual_smooth"]),
    ],}
ACTIVE_MODEL_SET = "hgb_family"
MODEL_SPECS = list(MODEL_SETS[ACTIVE_MODEL_SET])

VALIDATOR_WEIGHT_LOWER = 0.001
VALIDATOR_WEIGHT_SHRINKAGE = 0.35
VALIDATOR_WEIGHT_L2 = 0.03

HGB_STATIC_PARAM_OVERRIDES = {
    "hgb_direct_smooth": {
        "max_iter": 1200,
        "learning_rate": 0.02,
        "max_leaf_nodes": 18,
        "min_samples_leaf": 90,
        "l2_regularization": 0.7,
        "validation_fraction": 0.20,
    },
    "hgb_direct_deep": {
        "max_iter": 750,
        "learning_rate": 0.04,
        "max_leaf_nodes": 40,
        "min_samples_leaf": 35,
        "l2_regularization": 0.15,
        "validation_fraction": 0.15,
    },
    "hgb_residual_smooth": {
        "max_iter": 1100,
        "learning_rate": 0.025,
        "max_leaf_nodes": 18,
        "min_samples_leaf": 85,
        "l2_regularization": 0.6,
        "validation_fraction": 0.20,
    },
    "hgb_regime_low_direct": {
        "max_iter": 900,
        "learning_rate": 0.024,
        "max_leaf_nodes": 14,
        "min_samples_leaf": 130,
        "l2_regularization": 1.15,
        "validation_fraction": 0.20,
    },
    "hgb_regime_partial_direct": {
        "max_iter": 850,
        "learning_rate": 0.036,
        "max_leaf_nodes": 44,
        "min_samples_leaf": 38,
        "l2_regularization": 0.18,
        "validation_fraction": 0.15,
    },
    "hgb_regime_rated_direct": {
        "max_iter": 720,
        "learning_rate": 0.024,
        "max_leaf_nodes": 16,
        "min_samples_leaf": 120,
        "l2_regularization": 1.00,
        "validation_fraction": 0.20,
    },
    "hgb_regime_cutout_direct": {
        "max_iter": 560,
        "learning_rate": 0.020,
        "max_leaf_nodes": 10,
        "min_samples_leaf": 180,
        "l2_regularization": 1.60,
        "validation_fraction": 0.22,
    },
    "hgb_regime_low_residual": {
        "max_iter": 850,
        "learning_rate": 0.022,
        "max_leaf_nodes": 12,
        "min_samples_leaf": 140,
        "l2_regularization": 1.20,
        "validation_fraction": 0.20,
    },
    "hgb_regime_partial_residual": {
        "max_iter": 1100,
        "learning_rate": 0.026,
        "max_leaf_nodes": 24,
        "min_samples_leaf": 70,
        "l2_regularization": 0.55,
        "validation_fraction": 0.18,
    },
    "hgb_regime_rated_residual": {
        "max_iter": 820,
        "learning_rate": 0.022,
        "max_leaf_nodes": 14,
        "min_samples_leaf": 130,
        "l2_regularization": 1.10,
        "validation_fraction": 0.22,
    },
    "hgb_regime_cutout_residual": {
        "max_iter": 560,
        "learning_rate": 0.018,
        "max_leaf_nodes": 10,
        "min_samples_leaf": 180,
        "l2_regularization": 1.60,
        "validation_fraction": 0.22,
    },
    "hgb_q50_direct_deep": {
        "loss": "quantile",
        "quantile": 0.5,
        "max_iter": 850,
        "learning_rate": 0.028,
        "max_leaf_nodes": 34,
        "min_samples_leaf": 55,
        "l2_regularization": 0.35,
        "validation_fraction": 0.18,
    },
    "hgb_q50_residual_smooth": {
        "loss": "quantile",
        "quantile": 0.5,
        "max_iter": 1100,
        "learning_rate": 0.020,
        "max_leaf_nodes": 16,
        "min_samples_leaf": 110,
        "l2_regularization": 0.85,
        "validation_fraction": 0.20,
    },
    "hgb_physics_v2_residual_smooth": {
        "max_iter": 950,
        "learning_rate": 0.025,
        "max_leaf_nodes": 18,
        "min_samples_leaf": 100,
        "l2_regularization": 0.9,
        "validation_fraction": 0.20,
    },
    "hgb_physics_v2_residual_deep": {
        "max_iter": 700,
        "learning_rate": 0.035,
        "max_leaf_nodes": 36,
        "min_samples_leaf": 45,
        "l2_regularization": 0.35,
        "validation_fraction": 0.16,
    },
    "hgb_empirical_v3_residual_smooth": {
        "max_iter": 1000,
        "learning_rate": 0.024,
        "max_leaf_nodes": 18,
        "min_samples_leaf": 95,
        "l2_regularization": 0.85,
        "validation_fraction": 0.20,
    },
    "hgb_empirical_v3_residual_deep": {
        "max_iter": 720,
        "learning_rate": 0.034,
        "max_leaf_nodes": 34,
        "min_samples_leaf": 48,
        "l2_regularization": 0.40,
        "validation_fraction": 0.16,
    },
    "hgb_empirical_v4_residual_deep": {
        "max_iter": 760,
        "learning_rate": 0.032,
        "max_leaf_nodes": 36,
        "min_samples_leaf": 46,
        "l2_regularization": 0.42,
        "validation_fraction": 0.16,
    },
    "hgb_empirical_v4_residual_smooth": {
        "max_iter": 1000,
        "learning_rate": 0.024,
        "max_leaf_nodes": 18,
        "min_samples_leaf": 95,
        "l2_regularization": 0.85,
        "validation_fraction": 0.20,
    },
    "hgb_empirical_v4_guarded_residual_deep": {
        "max_iter": 720,
        "learning_rate": 0.030,
        "max_leaf_nodes": 28,
        "min_samples_leaf": 65,
        "l2_regularization": 0.70,
        "validation_fraction": 0.18,
    },
}
HGB_OPT_FALLBACK_SOURCES = {
    "hgb_opt_direct_1": "hgb_direct_deep",
    "hgb_opt_direct_2": "hgb_direct_smooth",
    "hgb_opt_residual_1": "hgb_residual_smooth",
    "hgb_opt_residual_2": "hgb_residual_smooth",
}
HGB_DYNAMIC_PARAMS_PATH = HGB_FAMILY_PARAMS_PATH
_HGB_DYNAMIC_MODELS_CACHE: dict[str, dict] | None = None
_HGB_DYNAMIC_MODELS_CACHE_PATH: Path | None = None
EMPIRICAL_CURVE_CONFIG_PATH = EMPIRICAL_CURVE_V4_CONFIG_PATH
ACTIVE_EMPIRICAL_CURVE_V4_CONFIG = default_empirical_curve_v4_config()

LATENT_AVAILABILITY_COLS = [
    "n_working_latent",
    "availability_latent",
    "availability_delta_from_monthly",
    "P_physics_farm_latent",
]
LATENT_MIN_PER_TURBINE_MW = 0.75
LATENT_MIN_GROUP_ROWS = 48
LATENT_DELTA_CLIP = 6.0


@dataclass(frozen=True)
class LatentAvailabilityCalibrator:
    """Fold-local calibration from monthly average availability to latent hourly capacity."""

    global_delta: float
    monthly_delta: dict[int, float] = field(default_factory=dict)


def default_blend_weights() -> dict[str, float]:
    return {spec.name: float(spec.weight) for spec in MODEL_SPECS}


def _ordered_weight_vector(weights: dict[str, float]) -> np.ndarray:
    return np.array([weights[spec.name] for spec in MODEL_SPECS], dtype=float)


def normalize_blend_weights(weights: dict[str, float]) -> dict[str, float]:
    ordered = np.array(
        [max(float(weights.get(spec.name, spec.weight)), 0.0) for spec in MODEL_SPECS],
        dtype=float,
    )
    total = float(ordered.sum())
    if total <= 0:
        ordered = _ordered_weight_vector(default_blend_weights())
    else:
        ordered /= total
    return {spec.name: float(weight) for spec, weight in zip(MODEL_SPECS, ordered)}


def format_blend_weights(weights: dict[str, float]) -> str:
    weights = normalize_blend_weights(weights)
    return ", ".join(f"{spec.name}={weights[spec.name]:.3f}" for spec in MODEL_SPECS)


def blend_weight_lower_bound(spec_name: str) -> float:
    if ACTIVE_MODEL_SET == "empirical_curve_v3" and spec_name == "empirical_curve_v3_prior":
        return 0.10
    if ACTIVE_MODEL_SET == "physics_first_v2":
        if spec_name == "physics_v2_prior":
            return 0.25
        if spec_name.startswith("hgb_physics_v2_residual"):
            return 0.05
    return VALIDATOR_WEIGHT_LOWER


def blend_model_predictions(
    predictions_by_model: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    weights = normalize_blend_weights(weights)
    blended = np.zeros_like(next(iter(predictions_by_model.values())), dtype=float)
    for spec in MODEL_SPECS:
        blended += weights[spec.name] * predictions_by_model[spec.name]
    return blended


def score_blend_weights(
    weights: dict[str, float],
    fold_predictions: dict[int, dict[str, np.ndarray]],
    fold_targets: dict[int, np.ndarray],
    fold_weights: dict[int, float],
) -> float:
    if not fold_predictions:
        return float("nan")
    total = 0.0
    weight_total = 0.0
    for year, predictions_by_model in fold_predictions.items():
        fold_weight = float(fold_weights.get(year, 1.0))
        blended = blend_model_predictions(predictions_by_model, weights)
        total += fold_weight * platform_error_percent(fold_targets[year], blended)
        weight_total += fold_weight
    return float(total / weight_total)


def optimize_validator_blend_weights(
    fold_predictions: dict[int, dict[str, np.ndarray]],
    fold_targets: dict[int, np.ndarray],
    fold_weights: dict[int, float],
    base_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Calibrate convex ensemble weights on already-finished historical folds only."""
    if not fold_predictions:
        return normalize_blend_weights(base_weights or default_blend_weights())

    base_weights = normalize_blend_weights(base_weights or default_blend_weights())
    base_vector = _ordered_weight_vector(base_weights)
    lower_bounds = np.array(
        [min(blend_weight_lower_bound(spec.name), 1.0) for spec in MODEL_SPECS],
        dtype=float,
    )
    if float(lower_bounds.sum()) >= 1.0:
        raise ValueError("Blend weight lower bounds sum to >= 1.0")

    def loss(weight_vector: np.ndarray) -> float:
        candidate = {spec.name: float(weight) for spec, weight in zip(MODEL_SPECS, weight_vector)}
        mae_proxy = score_blend_weights(candidate, fold_predictions, fold_targets, fold_weights)
        regularization = VALIDATOR_WEIGHT_L2 * float(np.sum((weight_vector - base_vector) ** 2))
        return mae_proxy + regularization

    result = minimize(
        loss,
        x0=base_vector,
        method="SLSQP",
        constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
        bounds=[(float(bound), 1.0) for bound in lower_bounds],
        options={"ftol": 1e-7, "maxiter": 300},
    )
    optimized = result.x if result.success else base_vector
    stable = (1.0 - VALIDATOR_WEIGHT_SHRINKAGE) * optimized + VALIDATOR_WEIGHT_SHRINKAGE * base_vector
    return normalize_blend_weights({spec.name: float(weight) for spec, weight in zip(MODEL_SPECS, stable)})


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def set_catboost_runtime(task_type: str = "CPU", devices: str | None = None) -> None:
    global CATBOOST_TASK_TYPE, CATBOOST_DEVICES
    CATBOOST_TASK_TYPE = task_type.upper()
    CATBOOST_DEVICES = devices


def set_validator_weight_lower(weight_lower: float) -> None:
    global VALIDATOR_WEIGHT_LOWER
    if weight_lower < 0.0:
        raise ValueError("Blend weight lower bound must be non-negative.")
    if weight_lower * len(MODEL_SPECS) >= 1.0:
        raise ValueError("Blend weight lower bound is too large for the active model set.")
    VALIDATOR_WEIGHT_LOWER = float(weight_lower)


def set_hgb_dynamic_params_path(path: Path) -> None:
    global HGB_DYNAMIC_PARAMS_PATH, _HGB_DYNAMIC_MODELS_CACHE, _HGB_DYNAMIC_MODELS_CACHE_PATH
    HGB_DYNAMIC_PARAMS_PATH = Path(path)
    _HGB_DYNAMIC_MODELS_CACHE = None
    _HGB_DYNAMIC_MODELS_CACHE_PATH = None


def set_empirical_curve_config_path(path: Path) -> None:
    global EMPIRICAL_CURVE_CONFIG_PATH, ACTIVE_EMPIRICAL_CURVE_V4_CONFIG
    EMPIRICAL_CURVE_CONFIG_PATH = Path(path)
    ACTIVE_EMPIRICAL_CURVE_V4_CONFIG = load_empirical_curve_v4_config(EMPIRICAL_CURVE_CONFIG_PATH)


def set_active_physics_config(cfg: PhysicsConfig) -> None:
    global ACTIVE_PHYSICS_CONFIG
    ACTIVE_PHYSICS_CONFIG = cfg


def set_model_set(model_set: str = ACTIVE_MODEL_SET) -> None:
    global ACTIVE_MODEL_SET, MODEL_SPECS
    if model_set not in MODEL_SETS:
        raise ValueError(f"Unknown model set {model_set!r}; choose from {sorted(MODEL_SETS)}")
    ACTIVE_MODEL_SET = model_set
    MODEL_SPECS = list(MODEL_SETS[model_set])


def configure_v8_experiments(
    *,
    source_interactions: bool = False,
    lag_residual_features: bool = False,
    regime_prior_v2: bool = False,
    regime_models: bool = False,
    regime_v2: bool = False,
    windfm_diagnostic: bool = False,
    hgb_quantile_sisters: bool = False,
    direction_sector_features: bool = False,
    weather_analog_residual: bool = False,
    weather_dynamics_features: bool = False,
    multi_regime_features: bool = False,
    multi_regime_experts: bool = False,
    regime_calibration: str = "none",
) -> None:
    global ENABLE_SOURCE_INTERACTIONS
    global ENABLE_LAG_RESIDUAL_FEATURES
    global ENABLE_REGIME_PRIOR_V2
    global ENABLE_REGIME_MODELS
    global ENABLE_REGIME_V2
    global ENABLE_WINDFM_DIAGNOSTIC
    global ENABLE_HGB_QUANTILE_SISTERS
    global ENABLE_DIRECTION_SECTOR_FEATURES
    global ENABLE_WEATHER_ANALOG_RESIDUAL
    global ENABLE_WEATHER_DYNAMICS_FEATURES
    global ENABLE_MULTI_REGIME_FEATURES
    global ENABLE_MULTI_REGIME_EXPERTS
    global REGIME_CALIBRATION_MODE
    global MODEL_SPECS
    if regime_calibration not in REGIME_CALIBRATION_MODES:
        raise ValueError(
            f"Unknown regime calibration mode {regime_calibration!r}; "
            f"expected one of {sorted(REGIME_CALIBRATION_MODES)}"
        )
    ENABLE_SOURCE_INTERACTIONS = bool(source_interactions)
    ENABLE_LAG_RESIDUAL_FEATURES = bool(lag_residual_features)
    ENABLE_REGIME_PRIOR_V2 = bool(regime_prior_v2)
    ENABLE_REGIME_MODELS = bool(regime_models)
    ENABLE_REGIME_V2 = bool(regime_v2)
    ENABLE_WINDFM_DIAGNOSTIC = bool(windfm_diagnostic)
    ENABLE_HGB_QUANTILE_SISTERS = bool(hgb_quantile_sisters)
    ENABLE_DIRECTION_SECTOR_FEATURES = bool(direction_sector_features)
    ENABLE_WEATHER_ANALOG_RESIDUAL = bool(weather_analog_residual)
    ENABLE_WEATHER_DYNAMICS_FEATURES = bool(weather_dynamics_features)
    ENABLE_MULTI_REGIME_FEATURES = bool(multi_regime_features or multi_regime_experts)
    ENABLE_MULTI_REGIME_EXPERTS = bool(multi_regime_experts)
    REGIME_CALIBRATION_MODE = regime_calibration
    base_specs = list(MODEL_SETS[ACTIVE_MODEL_SET])
    if ENABLE_REGIME_MODELS:
        base_specs.extend(
            [
                ModelSpec("hgb_regime_direct", "direct", 0.020),
                ModelSpec("hgb_regime_residual", "residual", 0.080),
            ]
        )
    if ENABLE_HGB_QUANTILE_SISTERS:
        base_specs.extend(
            [
                ModelSpec("hgb_q50_direct_deep", "direct", 0.030),
                ModelSpec("hgb_q50_residual_smooth", "residual", 0.030),
            ]
        )
    MODEL_SPECS = base_specs


def physics_config_from_preset(preset: str = "public_best") -> PhysicsConfig:
    if preset not in PHYSICS_PRESETS:
        raise ValueError(f"Unknown physics preset {preset!r}; choose from {sorted(PHYSICS_PRESETS)}")
    return PHYSICS_PRESETS[preset]


def load_hgb_dynamic_models(path: Path | None = None) -> dict[str, dict]:
    global _HGB_DYNAMIC_MODELS_CACHE, _HGB_DYNAMIC_MODELS_CACHE_PATH
    path = Path(path or HGB_DYNAMIC_PARAMS_PATH)
    if _HGB_DYNAMIC_MODELS_CACHE is not None and _HGB_DYNAMIC_MODELS_CACHE_PATH == path:
        return _HGB_DYNAMIC_MODELS_CACHE
    if not path.exists():
        _HGB_DYNAMIC_MODELS_CACHE = {}
        _HGB_DYNAMIC_MODELS_CACHE_PATH = path
        return _HGB_DYNAMIC_MODELS_CACHE
    payload = json.loads(path.read_text())
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise ValueError(f"{path} must contain a list under 'models'")
    loaded: dict[str, dict] = {}
    for model in models:
        if not isinstance(model, dict):
            continue
        name = model.get("name")
        params = model.get("params")
        if isinstance(name, str) and isinstance(params, dict):
            loaded[name] = {
                "target_mode": model.get("target_mode"),
                "params": params,
                "cv_score": model.get("cv_score"),
            }
    _HGB_DYNAMIC_MODELS_CACHE = loaded
    _HGB_DYNAMIC_MODELS_CACHE_PATH = path
    return _HGB_DYNAMIC_MODELS_CACHE


def format_duration(seconds: float) -> str:
    minutes, seconds = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _fetch_json(
    url: str,
    *,
    retries: int = 3,
    timeout_seconds: int | float | None = None,
    headers: dict[str, str] | None = None,
    safe_url: str | None = None,
    data: bytes | None = None,
) -> dict:
    last_error: Exception | None = None
    request_headers = {"User-Agent": "azov-windfarm-forecast/1.0"}
    if headers:
        request_headers.update(headers)
    timeout = timeout_seconds if timeout_seconds is not None else HTTP_TIMEOUT_SECONDS
    display_url = safe_url or url
    for attempt in range(retries):
        try:
            request = Request(url, data=data, headers=request_headers)
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:  # pragma: no cover - depends on external APIs
            body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {body[:1000]}")
            break
        except Exception as exc:  # pragma: no cover - depends on external APIs
            last_error = exc
            if attempt + 1 < retries:
                sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch external weather data from {display_url}") from last_error


def _date_bounds(df: pd.DataFrame) -> tuple[str, str]:
    dt = pd.to_datetime(df[DATETIME_COL])
    return dt.min().strftime("%Y-%m-%d"), dt.max().strftime("%Y-%m-%d")


def _load_cached_external(path: Path, start_date: str, end_date: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    cached = pd.read_csv(path, parse_dates=[DATETIME_COL])
    if cached.empty:
        return None
    have_start = cached[DATETIME_COL].min().strftime("%Y-%m-%d")
    have_end = cached[DATETIME_COL].max().strftime("%Y-%m-%d")
    if have_start <= start_date and have_end >= end_date:
        return cached
    return None


def _validate_optional_external_frame(
    frame: pd.DataFrame,
    source_name: str,
    start_date: str,
    end_date: str,
    min_coverage: float,
) -> tuple[pd.DataFrame | None, dict]:
    report = {
        "source": source_name,
        "status": "skipped",
        "reason": "",
        "rows": int(len(frame)) if frame is not None else 0,
        "coverage": 0.0,
        "columns": [],
    }
    if frame is None or frame.empty:
        report["reason"] = "empty frame"
        return None, report
    if DATETIME_COL not in frame.columns:
        report["reason"] = f"missing {DATETIME_COL}"
        return None, report

    out = frame.copy()
    out[DATETIME_COL] = _normalize_external_datetime(out[DATETIME_COL])
    out = out.dropna(subset=[DATETIME_COL]).sort_values(DATETIME_COL)
    out = out.drop_duplicates(DATETIME_COL, keep="last").reset_index(drop=True)
    feature_cols = [col for col in out.columns if col != DATETIME_COL]
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    feature_cols = [col for col in feature_cols if not out[col].isna().all()]
    if not feature_cols:
        report["reason"] = "no non-empty numeric feature columns"
        return None, report
    out = out[[DATETIME_COL] + feature_cols]
    coverage = _requested_hour_coverage(out, start_date, end_date)
    report.update(
        {
            "rows": int(len(out)),
            "coverage": float(coverage),
            "columns": feature_cols,
            "start": out[DATETIME_COL].min().strftime("%Y-%m-%d %H:%M:%S"),
            "end": out[DATETIME_COL].max().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    if coverage < min_coverage:
        report["reason"] = f"coverage {coverage:.1%} below {min_coverage:.1%}"
        return None, report
    report["status"] = "available"
    return out, report


def _optional_external_fetchers() -> dict[str, tuple[str, object]]:
    return {
        "copernicus_era5": ("copernicus_era5_azov.csv", fetch_copernicus_era5),
        "meteoblue": ("meteoblue_history_azov.csv", fetch_meteoblue_history),
        "renewables_ninja": ("renewables_ninja_wind_azov.csv", fetch_renewables_ninja_wind),
        "windy": ("windy_point_forecast_azov.csv", fetch_windy_point_forecast),
        "global_wind_atlas": ("global_wind_atlas_static_azov.csv", fetch_global_wind_atlas_static),
        "noaa_nomads_gfs": ("noaa_nomads_gfs_azov.csv", fetch_noaa_nomads_gfs),
        "windatlas_xyz": ("windatlas_xyz_azov.csv", fetch_windatlas_xyz),
    }


def _month_ranges(start_date: str, end_date: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current = start.replace(day=1)
    while current <= end:
        month_start = max(current, start)
        month_end = min(current + pd.offsets.MonthEnd(0), end)
        ranges.append((pd.Timestamp(month_start), pd.Timestamp(month_end)))
        current = pd.Timestamp(current + pd.offsets.MonthBegin(1))
    return ranges


def _nasa_month_cache_path(cache_dir: Path, month_start: pd.Timestamp) -> Path:
    return cache_dir / f"nasa_power_azov_{month_start.strftime('%Y%m')}.csv"


def _load_nasa_power_chunked(
    cache_dir: Path,
    start_date: str,
    end_date: str,
    cache_policy: str,
    timeout_seconds: int | float,
    min_coverage: float,
) -> tuple[pd.DataFrame | None, dict]:
    report = {
        "source": "nasa_power",
        "cache_path": str(cache_dir / "nasa_power_azov.csv"),
        "chunked": True,
        "status": "skipped",
    }
    if cache_policy != "refresh":
        full_cache = _load_cached_external(cache_dir / "nasa_power_azov.csv", start_date, end_date)
        if full_cache is not None:
            frame, validation = _validate_optional_external_frame(
                full_cache,
                "nasa_power",
                start_date,
                end_date,
                min_coverage,
            )
            validation["cache_path"] = str(cache_dir / "nasa_power_azov.csv")
            validation["from_cache"] = True
            validation["chunked"] = True
            if frame is not None:
                return frame, validation
            report = validation

    if cache_policy == "offline":
        report.setdefault("reason", "cache unavailable in offline mode")
        return None, report

    frames: list[pd.DataFrame] = []
    chunk_reports: list[dict] = []
    for month_start, month_end in _month_ranges(start_date, end_date):
        chunk_path = _nasa_month_cache_path(cache_dir, month_start)
        chunk = None if cache_policy == "refresh" else _load_cached_external(
            chunk_path,
            month_start.strftime("%Y-%m-%d"),
            month_end.strftime("%Y-%m-%d"),
        )
        from_cache = chunk is not None
        if chunk is None:
            try:
                # NASA POWER timestamps are UTC; requesting the previous UTC day
                # covers the first local-hours after conversion to Europe/Moscow.
                query_start = (month_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                query_end = month_end.strftime("%Y-%m-%d")
                chunk = fetch_nasa_power(query_start, query_end, timeout_seconds=timeout_seconds)
            except Exception as exc:
                chunk_reports.append(
                    {
                        "month": month_start.strftime("%Y-%m"),
                        "status": "skipped",
                        "reason": str(exc),
                        "cache_path": str(chunk_path),
                    }
                )
                continue
        frame, validation = _validate_optional_external_frame(
            chunk,
            "nasa_power",
            month_start.strftime("%Y-%m-%d"),
            month_end.strftime("%Y-%m-%d"),
            min_coverage=min(min_coverage, 0.95),
        )
        validation["month"] = month_start.strftime("%Y-%m")
        validation["cache_path"] = str(chunk_path)
        validation["from_cache"] = from_cache
        chunk_reports.append(validation)
        if frame is not None:
            if not from_cache:
                frame.to_csv(chunk_path, index=False)
            frames.append(frame)

    if not frames:
        report["reason"] = "no NASA POWER monthly chunks were available"
        report["chunks"] = chunk_reports
        return None, report

    merged = pd.concat(frames, ignore_index=True).sort_values(DATETIME_COL)
    merged = merged.drop_duplicates(DATETIME_COL, keep="last").reset_index(drop=True)
    frame, validation = _validate_optional_external_frame(
        merged,
        "nasa_power",
        start_date,
        end_date,
        min_coverage,
    )
    validation["cache_path"] = str(cache_dir / "nasa_power_azov.csv")
    validation["from_cache"] = all(item.get("from_cache") for item in chunk_reports if item.get("status") == "available")
    validation["chunked"] = True
    validation["chunks"] = chunk_reports
    if frame is not None:
        frame.to_csv(cache_dir / "nasa_power_azov.csv", index=False)
    return frame, validation


def _load_optional_external_source(
    source_name: str,
    cache_dir: Path,
    start_date: str,
    end_date: str,
    cache_policy: str,
    timeout_seconds: int | float,
    min_coverage: float,
) -> tuple[pd.DataFrame | None, dict]:
    fetchers = _optional_external_fetchers()
    if source_name == "nasa_power":
        return _load_nasa_power_chunked(
            cache_dir,
            start_date,
            end_date,
            cache_policy=cache_policy,
            timeout_seconds=timeout_seconds,
            min_coverage=min_coverage,
        )
    elif source_name in fetchers:
        cache_name, fetcher = fetchers[source_name]
    else:
        return None, {"source": source_name, "status": "skipped", "reason": "unknown source"}

    cache_path = cache_dir / cache_name
    report = {"source": source_name, "cache_path": str(cache_path), "status": "skipped"}
    if cache_policy not in EXTERNAL_CACHE_POLICIES:
        raise ValueError(f"Unknown external cache policy: {cache_policy}")

    if cache_policy != "refresh" and cache_path.exists():
        cached = _load_cached_external(cache_path, start_date, end_date)
        if cached is not None:
            frame, validation = _validate_optional_external_frame(
                cached,
                source_name,
                start_date,
                end_date,
                min_coverage,
            )
            validation["cache_path"] = str(cache_path)
            validation["from_cache"] = True
            if frame is not None:
                return frame, validation
            report = validation

    if cache_policy == "offline":
        report.setdefault("reason", "cache unavailable in offline mode")
        return None, report

    try:
        print(f"  fetching optional {source_name} external source ({start_date} -> {end_date})", flush=True)
        fetched = fetcher(start_date, end_date, timeout_seconds)
        frame, validation = _validate_optional_external_frame(
            fetched,
            source_name,
            start_date,
            end_date,
            min_coverage,
        )
        validation["cache_path"] = str(cache_path)
        validation["from_cache"] = False
        if frame is None:
            return None, validation
        frame.to_csv(cache_path, index=False)
        return frame, validation
    except Exception as exc:
        report["reason"] = str(exc)
        return None, report


def resolve_openmeteo_forecast_sources(source_set: str) -> tuple[str, ...]:
    if source_set not in OPENMETEO_FORECAST_SETS:
        valid = ", ".join(sorted(OPENMETEO_FORECAST_SETS))
        raise ValueError(f"Unknown Open-Meteo forecast set {source_set!r}; expected one of: {valid}")
    return OPENMETEO_FORECAST_SETS[source_set]


def openmeteo_pressure_enabled(source_set: str, explicit: bool = False) -> bool:
    return explicit or source_set in OPENMETEO_PRESSURE_FORECAST_SETS


def openmeteo_forecast_columns(prefix: str) -> list[str]:
    return [f"ext_omfc_{prefix}_{col}" for col in OPENMETEO_SURFACE_HOURLY]


def openmeteo_pressure_columns(prefix: str) -> list[str]:
    cols = []
    for level in OPENMETEO_PRESSURE_LEVELS:
        cols.extend(
            [
                f"ext_omfc_{prefix}_wind_speed_{level}",
                f"ext_omfc_{prefix}_wind_direction_{level}",
                f"ext_omfc_{prefix}_temperature_{level}",
                f"ext_omfc_{prefix}_geopotential_height_{level}",
            ]
        )
    return cols


def _load_cached_external_with_columns(
    path: Path,
    start_date: str,
    end_date: str,
    required_columns: list[str] | None = None,
) -> pd.DataFrame | None:
    cached = _load_cached_external(path, start_date, end_date)
    if cached is None:
        return None
    if required_columns:
        missing = [col for col in required_columns if col not in cached.columns]
        if missing:
            return None
    return cached


def fetch_openmeteo_era5(start_date: str, end_date: str) -> pd.DataFrame:
    hourly = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "precipitation",
        "rain",
        "snowfall",
        "cloud_cover",
        "cloud_cover_low",
        "pressure_msl",
        "surface_pressure",
        "wind_speed_10m",
        "wind_speed_100m",
        "wind_direction_10m",
        "wind_direction_100m",
        "wind_gusts_10m",
    ]
    params = {
        "latitude": AZOV_LAT,
        "longitude": AZOV_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly),
        "timezone": LOCAL_TIMEZONE,
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(params)
    payload = _fetch_json(url)
    hourly_payload = payload.get("hourly", {})
    if "time" not in hourly_payload:
        raise RuntimeError(f"Open-Meteo response has no hourly time axis: {payload}")

    out = pd.DataFrame({DATETIME_COL: pd.to_datetime(hourly_payload["time"])})
    for col in hourly:
        out[f"ext_era5_{col}"] = hourly_payload.get(col)
    return out


def fetch_openmeteo_historical_forecast(
    start_date: str,
    end_date: str,
    model: str = "gfs_global",
    prefix: str = "gfs",
    include_pressure_levels: bool = False,
) -> pd.DataFrame:
    hourly = list(OPENMETEO_SURFACE_HOURLY)
    if include_pressure_levels:
        for level in OPENMETEO_PRESSURE_LEVELS:
            hourly.extend(
                [
                    f"wind_speed_{level}",
                    f"wind_direction_{level}",
                    f"temperature_{level}",
                    f"geopotential_height_{level}",
                ]
            )
    params = {
        "latitude": AZOV_LAT,
        "longitude": AZOV_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly),
        "timezone": LOCAL_TIMEZONE,
        "wind_speed_unit": "ms",
        "models": model,
    }
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast?" + urlencode(params)
    payload = _fetch_json(url)
    hourly_payload = payload.get("hourly", {})
    if "time" not in hourly_payload:
        raise RuntimeError(f"Open-Meteo historical forecast response has no hourly time axis: {payload}")

    out = pd.DataFrame({DATETIME_COL: pd.to_datetime(hourly_payload["time"])})
    for col in hourly:
        out[f"ext_omfc_{prefix}_{col}"] = hourly_payload.get(col)
    out[f"ext_omfc_{prefix}_observed"] = 1.0
    return out


def fetch_nasa_power(
    start_date: str,
    end_date: str,
    timeout_seconds: int | float | None = None,
) -> pd.DataFrame:
    parameters = list(NASA_POWER_PARAMETERS)
    params = {
        "parameters": ",".join(parameters),
        "community": "RE",
        "longitude": AZOV_LON,
        "latitude": AZOV_LAT,
        "start": start_date.replace("-", ""),
        "end": end_date.replace("-", ""),
        "format": "JSON",
        "time-standard": "UTC",
    }
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point?" + urlencode(params)
    payload = _fetch_json(url, timeout_seconds=timeout_seconds)
    values = payload.get("properties", {}).get("parameter", {})
    if not values:
        raise RuntimeError(f"NASA POWER response has no parameter payload: {payload}")

    first_param = parameters[0]
    time_index = sorted(values[first_param].keys())
    out = pd.DataFrame(
        {
            DATETIME_COL: pd.to_datetime(time_index, format="%Y%m%d%H", utc=True)
            .tz_convert(LOCAL_TIMEZONE)
            .tz_localize(None)
            .floor("h")
        }
    )
    for param in parameters:
        out[f"ext_nasa_{param.lower()}"] = [values.get(param, {}).get(ts) for ts in time_index]
    out = out.replace(-999.0, np.nan)
    return out


def resolve_external_source_set(source_set: str, include_nasa_power: bool = False) -> tuple[str, ...]:
    if source_set not in EXTERNAL_SOURCE_SETS:
        valid = ", ".join(sorted(EXTERNAL_SOURCE_SETS))
        raise ValueError(f"Unknown external source set {source_set!r}; expected one of: {valid}")
    sources = list(EXTERNAL_SOURCE_SETS[source_set])
    if include_nasa_power and "nasa_power" not in sources:
        sources.append("nasa_power")
    return tuple(sources)


def _env_required(name: str, source_name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{source_name} skipped: environment variable {name} is not set")
    return value


def _meteoblue_signed_url(path: str, params: dict[str, object], api_key: str) -> tuple[str, str]:
    params_with_key = {**params, "apikey": api_key}
    secret = os.environ.get("METEOBLUE_API_SECRET", "").strip()
    if secret:
        params_with_key["expire"] = int(pd.Timestamp.utcnow().timestamp()) + 3600
    query = urlencode(params_with_key)
    url_path = f"{path}?{query}"
    if secret:
        signature = hmac.new(secret.encode("utf-8"), url_path.encode("utf-8"), hashlib.sha256).hexdigest()
        url_path = f"{url_path}&sig={signature}"
    safe_params = {**params_with_key, "apikey": "<hidden>"}
    if secret:
        safe_params["sig"] = "<hidden>"
    safe_path = f"{path}?{urlencode(safe_params)}"
    return "https://my.meteoblue.com" + url_path, "https://my.meteoblue.com" + safe_path


def _local_naive_from_utc(values: pd.Series | pd.DatetimeIndex) -> pd.Series:
    dt = pd.to_datetime(values, utc=True, errors="coerce")
    if isinstance(dt, pd.DatetimeIndex):
        dt = pd.Series(dt)
    return dt.dt.tz_convert(LOCAL_TIMEZONE).dt.tz_localize(None).dt.floor("h")


def _wind_speed_from_uv(u: pd.Series, v: pd.Series) -> pd.Series:
    return np.sqrt(pd.to_numeric(u, errors="coerce") ** 2 + pd.to_numeric(v, errors="coerce") ** 2)


def _wind_direction_from_uv(u: pd.Series, v: pd.Series) -> pd.Series:
    # Meteorological direction: degrees from which the wind blows.
    return (np.degrees(np.arctan2(-pd.to_numeric(u, errors="coerce"), -pd.to_numeric(v, errors="coerce"))) + 360.0) % 360.0


def fetch_copernicus_era5(start_date: str, end_date: str, timeout_seconds: int | float) -> pd.DataFrame:
    _env_required("CDSAPI_KEY", "Copernicus CDS")
    os.environ.setdefault("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
    try:
        import cdsapi
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "Copernicus CDS skipped: install optional dependencies with "
            "pip install cdsapi xarray netCDF4"
        ) from exc

    client = cdsapi.Client(quiet=True, timeout=timeout_seconds)

    def _fetch_month(month_start: pd.Timestamp, month_end: pd.Timestamp) -> pd.DataFrame:
        days = pd.date_range(month_start, month_end, freq="D")
        request = {
            "product_type": ["reanalysis"],
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "100m_u_component_of_wind",
                "100m_v_component_of_wind",
                "2m_temperature",
                "surface_pressure",
                "mean_sea_level_pressure",
            ],
            "year": [month_start.strftime("%Y")],
            "month": [month_start.strftime("%m")],
            "day": [d.strftime("%d") for d in days],
            "time": [f"{hour:02d}:00" for hour in range(24)],
            "area": [AZOV_LAT + 0.15, AZOV_LON - 0.15, AZOV_LAT - 0.15, AZOV_LON + 0.15],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
            try:
                client.retrieve("reanalysis-era5-single-levels", request, tmp.name)
            except Exception as exc:
                message = str(exc)
                if "licence" in message.lower() or "license" in message.lower() or "403" in message:
                    raise RuntimeError(
                        "Copernicus CDS ERA5 skipped: accept the required ERA5 licence first at "
                        "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels"
                        "?tab=download#manage-licences"
                    ) from exc
                raise
            ds = xr.open_dataset(tmp.name)
            try:
                return _copernicus_dataset_to_frame(ds)
            finally:
                ds.close()

    def _copernicus_dataset_to_frame(ds) -> pd.DataFrame:
        time_name = "valid_time" if "valid_time" in ds.coords else "time"
        ds = ds.sel(latitude=AZOV_LAT, longitude=AZOV_LON, method="nearest")
        out = pd.DataFrame({DATETIME_COL: _local_naive_from_utc(ds[time_name].to_index())})
        rename = {
            "t2m": "ext_cds_era5_temperature_2m",
            "sp": "ext_cds_era5_surface_pressure",
            "msl": "ext_cds_era5_pressure_msl",
        }
        for source, target in rename.items():
            if source in ds:
                out[target] = np.asarray(ds[source]).reshape(-1)
        if {"u10", "v10"}.issubset(ds):
            u10 = pd.Series(np.asarray(ds["u10"]).reshape(-1))
            v10 = pd.Series(np.asarray(ds["v10"]).reshape(-1))
            out["ext_cds_era5_wind_speed_10m"] = _wind_speed_from_uv(u10, v10)
            out["ext_cds_era5_wind_direction_10m"] = _wind_direction_from_uv(u10, v10)
        if {"u100", "v100"}.issubset(ds):
            u100 = pd.Series(np.asarray(ds["u100"]).reshape(-1))
            v100 = pd.Series(np.asarray(ds["v100"]).reshape(-1))
            out["ext_cds_era5_wind_speed_100m"] = _wind_speed_from_uv(u100, v100)
            out["ext_cds_era5_wind_direction_100m"] = _wind_direction_from_uv(u100, v100)
        return out

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    frames = []
    month_start = start
    while month_start <= end:
        month_end = min(month_start + pd.offsets.MonthEnd(0), end)
        frames.append(_fetch_month(month_start, month_end))
        month_start = (month_start + pd.offsets.MonthBegin(1)).normalize()
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if "ext_cds_era5_temperature_2m" in out:
        out["ext_cds_era5_temperature_2m"] = out["ext_cds_era5_temperature_2m"] - 273.15
    if "ext_cds_era5_pressure_msl" in out:
        out["ext_cds_era5_pressure_msl"] = out["ext_cds_era5_pressure_msl"] / 100.0
    if "ext_cds_era5_surface_pressure" in out:
        out["ext_cds_era5_surface_pressure"] = out["ext_cds_era5_surface_pressure"] / 100.0
    return out


def fetch_meteoblue_history(start_date: str, end_date: str, timeout_seconds: int | float) -> pd.DataFrame:
    api_key = _env_required("METEOBLUE_API_KEY", "meteoblue")
    variable_map = {
        "temperature": "ext_meteoblue_temperature_2m",
        "relative_humidity": "ext_meteoblue_relative_humidity_2m",
        "sealevel_pressure": "ext_meteoblue_pressure_msl",
        "wind_speed": "ext_meteoblue_wind_speed_10m",
        "wind_direction": "ext_meteoblue_wind_direction_10m",
        "precipitation": "ext_meteoblue_precipitation",
        "dewpoint": "ext_meteoblue_dew_point_2m",
    }

    def _extract_variable(payload: dict, variable: str) -> tuple[pd.Series, pd.Series]:
        data = payload.get("data") or payload.get("data_1h") or payload
        if not isinstance(data, dict):
            raise RuntimeError(f"meteoblue {variable} response has unsupported data shape")
        times = (
            data.get("time")
            or data.get("timestamp")
            or data.get("datetime")
            or payload.get("time")
            or payload.get("timestamp")
        )
        if not times:
            raise RuntimeError(f"meteoblue {variable} response has no hourly time axis; keys={sorted(payload)[:20]}")
        candidates = [
            variable,
            variable.replace("_", ""),
            "value",
            "values",
            "data",
        ]
        lower_map = {str(key).lower(): key for key in data}
        values = None
        for candidate in candidates:
            key = lower_map.get(candidate.lower())
            if key is not None and key != "time":
                values = data[key]
                break
        if values is None:
            arrays = [
                value for key, value in data.items()
                if key not in {"time", "timestamp", "datetime"} and isinstance(value, list)
            ]
            if arrays:
                values = arrays[0]
        if values is None:
            raise RuntimeError(f"meteoblue {variable} response has no numeric value column; keys={sorted(data)[:20]}")
        return pd.Series(times), pd.Series(values)

    base_params = {
        "start": start_date,
        "end": end_date,
        "lat": AZOV_LAT,
        "lon": AZOV_LON,
        "format": "json",
        "timeformat": "iso8601",
        "utc_offset": 3,
        "domain": "NEMSAUTO",
        "temperature": "C",
        "windspeed": "ms-1",
        "winddirection": "degree",
        "precipitationamount": "mm",
        "aggregation": "none",
    }
    out: pd.DataFrame | None = None
    errors: list[str] = []
    for variable, target_col in variable_map.items():
        params = {**base_params, "variable": variable}
        url, safe_url = _meteoblue_signed_url("/history/point", params, api_key)
        try:
            payload = _fetch_json(url, timeout_seconds=timeout_seconds, safe_url=safe_url)
            times, values = _extract_variable(payload, variable)
        except Exception as exc:
            errors.append(f"{variable}: {exc}")
            continue
        partial = pd.DataFrame(
            {
                DATETIME_COL: pd.to_datetime(times, errors="coerce").dt.floor("h"),
                target_col: pd.to_numeric(values, errors="coerce"),
            }
        )
        out = partial if out is None else out.merge(partial, on=DATETIME_COL, how="outer")
    if out is None or len(out.columns) <= 1:
        raise RuntimeError("meteoblue History API produced no usable variables; " + "; ".join(errors[:3]))
    return out.sort_values(DATETIME_COL).drop_duplicates(DATETIME_COL, keep="last").reset_index(drop=True)


def fetch_renewables_ninja_wind(start_date: str, end_date: str, timeout_seconds: int | float) -> pd.DataFrame:
    token = _env_required("RENEWABLES_NINJA_TOKEN", "Renewables.ninja")
    params = {
        "lat": AZOV_LAT,
        "lon": AZOV_LON,
        "date_from": start_date,
        "date_to": end_date,
        "capacity": 1.0,
        "height": int(round(HUB_HEIGHT_M)),
        "turbine": "Vestas V112 3000",
        "format": "json",
        "local_time": "true",
    }
    url = "https://www.renewables.ninja/api/data/wind?" + urlencode(params)
    payload = _fetch_json(
        url,
        timeout_seconds=timeout_seconds,
        headers={"Authorization": f"Token {token}"},
    )
    data = payload.get("data", {})
    if not data:
        raise RuntimeError(f"Renewables.ninja response has no data payload; keys={sorted(payload)[:20]}")
    records = []
    for ts, values in data.items():
        row = {DATETIME_COL: pd.to_datetime(ts).floor("h")}
        if isinstance(values, dict):
            row.update(values)
        records.append(row)
    out = pd.DataFrame(records)
    rename = {
        "wind_speed": "ext_renewables_ninja_wind_speed_hub",
        "electricity": "ext_renewables_ninja_capacity_factor",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    keep = [DATETIME_COL] + [c for c in out.columns if c.startswith("ext_renewables_ninja_")]
    return out.loc[:, keep]


def fetch_windy_point_forecast(start_date: str, end_date: str, timeout_seconds: int | float) -> pd.DataFrame:
    api_key = _env_required("WINDY_API_KEY", "Windy")
    payload = {
        "lat": AZOV_LAT,
        "lon": AZOV_LON,
        "model": "gfs",
        "parameters": ["wind", "temp", "rh", "pressure"],
        "levels": ["surface", "100m"],
        "key": api_key,
    }
    safe_payload = {**payload, "key": "<hidden>"}
    body = json.dumps(payload).encode("utf-8")
    response = _fetch_json(
        "https://api.windy.com/api/point-forecast/v2",
        timeout_seconds=timeout_seconds,
        headers={"Content-Type": "application/json"},
        safe_url=f"https://api.windy.com/api/point-forecast/v2 payload={safe_payload}",
        data=body,
    )
    ts_values = response.get("ts")
    if not ts_values:
        raise RuntimeError(f"Windy response has no ts axis; keys={sorted(response)[:20]}")
    out = pd.DataFrame({DATETIME_COL: _local_naive_from_utc(pd.to_datetime(ts_values, unit="ms"))})
    for source, target in {
        "wind_u-surface": "ext_windy_wind_u_surface",
        "wind_v-surface": "ext_windy_wind_v_surface",
        "wind_u-100m": "ext_windy_wind_u_100m",
        "wind_v-100m": "ext_windy_wind_v_100m",
        "temp-surface": "ext_windy_temperature_surface",
        "rh-surface": "ext_windy_relative_humidity_surface",
        "pressure-surface": "ext_windy_pressure_surface",
    }.items():
        if source in response:
            out[target] = response[source]
    if {"ext_windy_wind_u_surface", "ext_windy_wind_v_surface"}.issubset(out.columns):
        out["ext_windy_wind_speed_surface"] = _wind_speed_from_uv(
            out["ext_windy_wind_u_surface"], out["ext_windy_wind_v_surface"]
        )
        out["ext_windy_wind_direction_surface"] = _wind_direction_from_uv(
            out["ext_windy_wind_u_surface"], out["ext_windy_wind_v_surface"]
        )
    if {"ext_windy_wind_u_100m", "ext_windy_wind_v_100m"}.issubset(out.columns):
        out["ext_windy_wind_speed_100m"] = _wind_speed_from_uv(
            out["ext_windy_wind_u_100m"], out["ext_windy_wind_v_100m"]
        )
        out["ext_windy_wind_direction_100m"] = _wind_direction_from_uv(
            out["ext_windy_wind_u_100m"], out["ext_windy_wind_v_100m"]
        )
    return out


def fetch_global_wind_atlas_static(start_date: str, end_date: str, timeout_seconds: int | float) -> pd.DataFrame:
    ws100 = os.environ.get("GLOBAL_WIND_ATLAS_WS100M")
    wpd100 = os.environ.get("GLOBAL_WIND_ATLAS_WPD100M")
    if not ws100 and not wpd100:
        raise RuntimeError(
            "Global Wind Atlas skipped: set GLOBAL_WIND_ATLAS_WS100M and/or "
            "GLOBAL_WIND_ATLAS_WPD100M from the downloaded point/GIS value"
        )
    out = pd.DataFrame({DATETIME_COL: _hourly_index(start_date, end_date)})
    if ws100:
        out["ext_gwa_wind_speed_100m_mean"] = float(ws100)
    if wpd100:
        out["ext_gwa_power_density_100m_mean"] = float(wpd100)
    return out


def fetch_noaa_nomads_gfs(start_date: str, end_date: str, timeout_seconds: int | float) -> pd.DataFrame:
    raise RuntimeError(
        "NOAA NOMADS GFS filter skipped for training: NOMADS is a current forecast feed here, "
        "not a complete 2022-2026 historical cache"
    )


def fetch_windatlas_xyz(start_date: str, end_date: str, timeout_seconds: int | float) -> pd.DataFrame:
    if pd.Timestamp(end_date) > pd.Timestamp("2019-12-31"):
        raise RuntimeError(
            "windatlas.xyz skipped for strict training: observed endpoint currently serves "
            "older 2018/2019 sample-style history but returns errors for 2022-2026"
        )
    params = {
        "lat": AZOV_LAT,
        "lon": AZOV_LON,
        "height": int(round(HUB_HEIGHT_M)),
        "date_from": start_date,
        "date_to": end_date,
    }
    url = "http://windatlas.xyz/api/wind/?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "azov-windfarm-forecast/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:
        text = response.read().decode("utf-8", errors="replace")
    lines = [line for line in text.splitlines() if line.strip() and not line.lower().startswith("metadata")]
    if not lines or not lines[0].startswith("datetime"):
        raise RuntimeError(f"windatlas.xyz response has no CSV datetime header: {text[:300]}")
    from io import StringIO

    frame = pd.read_csv(StringIO("\n".join(lines)))
    if "datetime" not in frame or "wind_speed" not in frame:
        raise RuntimeError(f"windatlas.xyz CSV missing required columns: {frame.columns.tolist()}")
    out = pd.DataFrame(
        {
            DATETIME_COL: pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
            .dt.tz_convert(LOCAL_TIMEZONE)
            .dt.tz_localize(None)
            .dt.floor("h"),
            "ext_windatlas_xyz_wind_speed_hub": pd.to_numeric(frame["wind_speed"], errors="coerce"),
        }
    )
    return out


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_col = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        found = lower_to_col.get(candidate.lower())
        if found is not None:
            return found
    return None


def _normalize_external_datetime(values: pd.Series) -> pd.Series:
    dt = pd.to_datetime(values, errors="coerce")
    if dt.isna().any():
        bad_count = int(dt.isna().sum())
        raise ValueError(f"External weather has {bad_count} unparseable datetime values")
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert(LOCAL_TIMEZONE).dt.tz_localize(None)
    return dt.dt.floor("h")


def _normalize_pressure_hpa(values: pd.Series) -> pd.Series:
    pressure = pd.to_numeric(values, errors="coerce")
    finite = pressure[np.isfinite(pressure)]
    if finite.empty:
        return pressure
    median = finite.median()
    if median > 2000.0:
        return pressure / 100.0
    if 650.0 <= median <= 800.0:
        return pressure * 1.33322
    return pressure


def _normalize_fraction(values: pd.Series) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    finite = out[np.isfinite(out)]
    if not finite.empty and finite.max() > 1.0:
        out = out / 100.0
    return out.clip(0.0, 1.0)


def fetch_meteostat_hourly(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        import meteostat as ms
    except ImportError as exc:
        raise RuntimeError(
            "Meteostat support requires the 'meteostat' package. "
            "Install it with: pip install -r physics/requirements.txt"
        ) from exc

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(hours=23)
    point_cls = getattr(ms, "Point", None)
    if point_cls is None:
        raise RuntimeError(
            "Installed Meteostat package does not expose Point. "
            "Install dependencies with: pip install -r physics/requirements.txt"
        )
    point = point_cls(AZOV_LAT, AZOV_LON)

    config = getattr(ms, "config", None)
    if config is not None and hasattr(config, "block_large_requests"):
        config.block_large_requests = False

    hourly_cls = getattr(ms, "Hourly", None)
    if hourly_cls is None:
        version = getattr(ms, "__version__", "unknown")
        raise RuntimeError(
            f"Installed Meteostat {version} exposes the newer API, but this project "
            "requires the Meteostat 1.x Hourly(Point, ...) API for Azov point weather. "
            "Install the pinned dependency with: "
            "pip install --force-reinstall 'meteostat>=1.6,<2.0'"
        )

    def _fetch_chunk(chunk_start: pd.Timestamp, chunk_end: pd.Timestamp) -> pd.DataFrame:
        start_dt = chunk_start.to_pydatetime()
        end_dt = chunk_end.to_pydatetime()
        time_series = hourly_cls(point, start_dt, end_dt, timezone=LOCAL_TIMEZONE)
        chunk = time_series.fetch()
        if chunk is None:
            raise RuntimeError(
                "Meteostat returned no hourly data for the Azov point query. "
                "Use the pinned dependency with: "
                "pip install --force-reinstall 'meteostat>=1.6,<2.0'"
            )
        return chunk.reset_index()

    chunk_frames = []
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Support for nested sequences for 'parse_dates'.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="'H' is deprecated.*",
            category=FutureWarning,
        )
        chunk_start = start_ts
        while chunk_start <= end_ts:
            chunk_end = min(
                chunk_start + pd.DateOffset(years=1) - pd.Timedelta(hours=1),
                end_ts,
            )
            chunk_frames.append(_fetch_chunk(chunk_start, pd.Timestamp(chunk_end)))
            chunk_start = pd.Timestamp(chunk_end) + pd.Timedelta(hours=1)
    data = pd.concat(chunk_frames, ignore_index=True) if chunk_frames else pd.DataFrame()
    if data.empty:
        raise RuntimeError("Meteostat returned no hourly rows for the Azov wind farm point")

    out = pd.DataFrame({DATETIME_COL: _normalize_external_datetime(data["time"])})
    rename_map = {
        "temp": "ext_meteostat_temperature",
        "dwpt": "ext_meteostat_dew_point",
        "rhum": "ext_meteostat_humidity",
        "wdir": "ext_meteostat_wind_direction",
        "wspd": "ext_meteostat_wind_speed",
        "pres": "ext_meteostat_pressure_msl",
        "coco": "ext_meteostat_condition_code",
    }
    for source_col, target_col in rename_map.items():
        if source_col in data:
            out[target_col] = pd.to_numeric(data[source_col], errors="coerce")

    if "ext_meteostat_wind_speed" in out:
        out["ext_meteostat_wind_speed"] = out["ext_meteostat_wind_speed"] / 3.6

    out = out.sort_values(DATETIME_COL).drop_duplicates(DATETIME_COL, keep="last")
    return out.reset_index(drop=True)


def _select_safe_meteostat_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only dense Meteostat signals; old caches may still contain sparse columns."""
    keep_cols = [DATETIME_COL] + [col for col in SAFE_METEOSTAT_COLUMNS if col in frame.columns]
    return frame.loc[:, keep_cols].copy()


def _requested_hour_coverage(frame: pd.DataFrame, start_date: str, end_date: str) -> float:
    if frame.empty:
        return 0.0
    dt = _normalize_external_datetime(frame[DATETIME_COL])
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(hours=23)
    requested = int(((end - start) / pd.Timedelta(hours=1)) + 1)
    in_range = dt.loc[(dt >= start) & (dt <= end)].nunique()
    return float(in_range / max(requested, 1))


def _hourly_index(start_date: str, end_date: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(hours=23)
    return pd.date_range(start, end, freq="h")


def _source_coverage_by_year(
    frame: pd.DataFrame,
    prefix: str,
    start_date: str,
    end_date: str,
) -> dict:
    requested = pd.DataFrame({DATETIME_COL: _hourly_index(start_date, end_date)})
    cols = [
        col
        for col in frame.columns
        if col.startswith(f"ext_omfc_{prefix}_") and not col.endswith("_observed")
    ]
    if not cols:
        requested["covered"] = False
    else:
        source = frame[[DATETIME_COL] + cols].copy()
        source[DATETIME_COL] = _normalize_external_datetime(source[DATETIME_COL])
        source = source.drop_duplicates(DATETIME_COL, keep="last")
        source["covered"] = source[cols].notna().any(axis=1)
        requested = requested.merge(source[[DATETIME_COL, "covered"]], on=DATETIME_COL, how="left")
        requested["covered"] = requested["covered"].fillna(False)

    requested["year"] = requested[DATETIME_COL].dt.year
    by_year = requested.groupby("year")["covered"].mean()
    return {
        "overall": float(requested["covered"].mean()),
        "by_year": {str(int(year)): float(value) for year, value in by_year.items()},
        "columns": cols,
    }


def build_openmeteo_forecast_coverage_report(
    frame: pd.DataFrame,
    prefixes: tuple[str, ...],
    start_date: str,
    end_date: str,
) -> dict:
    sources = {
        prefix: _source_coverage_by_year(frame, prefix, start_date, end_date)
        for prefix in prefixes
    }
    return {
        "start_date": start_date,
        "end_date": end_date,
        "sources": sources,
    }


def apply_openmeteo_coverage_guard(
    frame: pd.DataFrame,
    prefixes: tuple[str, ...],
    start_date: str,
    end_date: str,
    min_coverage: float,
    coverage_policy: str,
) -> tuple[pd.DataFrame, dict]:
    report = build_openmeteo_forecast_coverage_report(frame, prefixes, start_date, end_date)
    dropped: list[str] = []
    if coverage_policy == "strict":
        for prefix, metrics in report["sources"].items():
            yearly_values = list(metrics["by_year"].values())
            yearly_ok = bool(yearly_values) and min(yearly_values) >= min_coverage
            if metrics["overall"] < min_coverage or not yearly_ok:
                drop_cols = [col for col in frame.columns if col.startswith(f"ext_omfc_{prefix}_")]
                frame = frame.drop(columns=drop_cols)
                dropped.append(prefix)
    report["coverage_policy"] = coverage_policy
    report["min_coverage"] = float(min_coverage)
    report["dropped_sources"] = dropped
    return frame, report


def load_meteostat_hourly(
    base_df: pd.DataFrame,
    cache_path: Path = METEOSTAT_CACHE_PATH,
    refresh: bool = False,
    allow_partial: bool = True,
    require: bool = False,
) -> pd.DataFrame | None:
    start_date, end_date = _date_bounds(base_df)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not refresh:
        frame = pd.read_csv(cache_path, parse_dates=[DATETIME_COL])
        frame[DATETIME_COL] = _normalize_external_datetime(frame[DATETIME_COL])
        have_start = frame[DATETIME_COL].min().strftime("%Y-%m-%d")
        have_end = frame[DATETIME_COL].max().strftime("%Y-%m-%d")
        if have_start <= start_date and have_end >= end_date:
            return _select_safe_meteostat_columns(frame)
        coverage = _requested_hour_coverage(frame, start_date, end_date)
        if allow_partial and have_start <= start_date and coverage >= MIN_METEOSTAT_COVERAGE:
            print(
                f"  warning: cached Meteostat covers {have_start} -> {have_end} "
                f"({coverage:.1%} of requested hours); using partial source and filling gaps",
                flush=True,
            )
            return _select_safe_meteostat_columns(frame)

    try:
        print(f"  fetching meteostat external weather ({start_date} -> {end_date})")
        frame = fetch_meteostat_hourly(start_date, end_date)
    except Exception as exc:
        if require:
            raise RuntimeError(
                f"Meteostat weather is required for this run but unavailable: {exc}. "
                "Install dependencies with: pip install -r physics/requirements.txt"
            ) from exc
        print(
            f"  warning: Meteostat weather unavailable ({exc}); continuing without it",
            flush=True,
        )
        return None

    have_start = frame[DATETIME_COL].min().strftime("%Y-%m-%d")
    have_end = frame[DATETIME_COL].max().strftime("%Y-%m-%d")
    if have_start > start_date or have_end < end_date:
        coverage = _requested_hour_coverage(frame, start_date, end_date)
        if allow_partial and have_start <= start_date and coverage >= MIN_METEOSTAT_COVERAGE:
            print(
                f"  warning: Meteostat covers {have_start} -> {have_end} "
                f"({coverage:.1%} of requested hours); using partial source and filling gaps",
                flush=True,
            )
            frame.to_csv(cache_path, index=False)
            return _select_safe_meteostat_columns(frame)
        print(
            f"  warning: Meteostat covers {have_start} -> {have_end}, "
            f"but model needs {start_date} -> {end_date}; continuing without it",
            flush=True,
        )
        if require:
            raise RuntimeError(
                f"Meteostat weather is required, but coverage is {have_start} -> {have_end}; "
                f"requested {start_date} -> {end_date}."
            )
        return None
    frame.to_csv(cache_path, index=False)
    return _select_safe_meteostat_columns(frame)


def load_external_weather(
    base_df: pd.DataFrame,
    cache_dir: Path = EXTERNAL_WEATHER_DIR,
    refresh: bool = False,
    include_nasa_power: bool = False,
    external_source_set: str = "baseline",
    external_source_timeout_sec: int | float = DEFAULT_EXTERNAL_SOURCE_TIMEOUT_SECONDS,
    external_cache_policy: str = "cache_first",
    require_extra_sources: bool = False,
    include_openmeteo_forecast: bool = True,
    openmeteo_forecast_set: str = "core_open_forecast",
    openmeteo_fill_policy: str = "legacy",
    openmeteo_min_coverage: float = OPENMETEO_COVERAGE_MIN_DEFAULT,
    include_openmeteo_pressure: bool = False,
    include_meteostat: bool = True,
    meteostat_cache_path: Path = METEOSTAT_CACHE_PATH,
    allow_partial_meteostat: bool = True,
    require_meteostat: bool = False,
) -> pd.DataFrame:
    start_date, end_date = _date_bounds(base_df)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if external_cache_policy not in EXTERNAL_CACHE_POLICIES:
        raise ValueError(f"Unknown external cache policy: {external_cache_policy}")
    effective_refresh = refresh or external_cache_policy == "refresh"
    fetch_allowed = external_cache_policy != "offline"
    sources = {
        "openmeteo_era5": (cache_dir / "openmeteo_era5_azov.csv", fetch_openmeteo_era5),
    }

    frames = []
    for source_name, (cache_path, fetcher) in sources.items():
        frame = None if effective_refresh else _load_cached_external(cache_path, start_date, end_date)
        if frame is None:
            if not fetch_allowed:
                print(
                    f"  warning: {source_name} cache unavailable in offline mode; continuing without it",
                    flush=True,
                )
                continue
            print(f"  fetching {source_name} external weather ({start_date} -> {end_date})")
            frame = fetcher(start_date, end_date)
            frame.to_csv(cache_path, index=False)
        if frame is not None:
            frames.append(frame)

    if openmeteo_fill_policy not in OPENMETEO_FILL_POLICIES:
        raise ValueError(f"Unknown Open-Meteo fill policy: {openmeteo_fill_policy}")

    forecast_prefixes = resolve_openmeteo_forecast_sources(openmeteo_forecast_set)
    pressure_enabled = openmeteo_pressure_enabled(openmeteo_forecast_set, include_openmeteo_pressure)
    if include_openmeteo_forecast:
        for short_name in forecast_prefixes:
            model_name = OPENMETEO_FORECAST_MODEL_CATALOG[short_name]
            suffix = "pressure" if pressure_enabled else "surface"
            cache_name = (
                f"openmeteo_historical_forecast_{short_name}_azov.csv"
                if suffix == "surface"
                else f"openmeteo_historical_forecast_{short_name}_{suffix}_azov.csv"
            )
            cache_path = cache_dir / cache_name
            required_cols = [f"ext_omfc_{short_name}_wind_speed_10m"]
            if pressure_enabled:
                required_cols.append(f"ext_omfc_{short_name}_wind_speed_{OPENMETEO_PRESSURE_LEVELS[0]}")
            frame = None if effective_refresh else _load_cached_external_with_columns(
                cache_path,
                start_date,
                end_date,
                required_columns=required_cols,
            )
            if frame is None:
                if not fetch_allowed:
                    print(
                        f"  warning: Open-Meteo historical forecast {short_name} cache unavailable "
                        "in offline mode; continuing without it",
                        flush=True,
                    )
                else:
                    try:
                        print(
                            f"  fetching openmeteo_historical_forecast_{short_name} "
                            f"external weather ({start_date} -> {end_date})"
                            + (" + pressure levels" if pressure_enabled else "")
                        )
                        frame = fetch_openmeteo_historical_forecast(
                            start_date,
                            end_date,
                            model=model_name,
                            prefix=short_name,
                            include_pressure_levels=pressure_enabled,
                        )
                        frame.to_csv(cache_path, index=False)
                    except Exception as exc:
                        print(
                            f"  warning: Open-Meteo historical forecast {short_name} unavailable "
                            f"({exc}); continuing without it",
                            flush=True,
                        )
                        frame = None
            if frame is not None:
                frames.append(frame)

    if include_meteostat:
        meteostat_frame = load_meteostat_hourly(
            base_df,
            cache_path=meteostat_cache_path,
            refresh=effective_refresh,
            allow_partial=allow_partial_meteostat,
            require=require_meteostat,
        )
        if meteostat_frame is not None:
            frames.append(meteostat_frame)

    requested_extra_sources = resolve_external_source_set(external_source_set, include_nasa_power=include_nasa_power)
    extra_reports = []
    available_extra_sources = []
    for source_name in requested_extra_sources:
        frame, report = _load_optional_external_source(
            source_name,
            cache_dir,
            start_date,
            end_date,
            cache_policy=external_cache_policy,
            timeout_seconds=external_source_timeout_sec,
            min_coverage=openmeteo_min_coverage,
        )
        extra_reports.append(report)
        if frame is not None:
            frames.append(frame)
            available_extra_sources.append(source_name)
            print(
                f"  optional source {source_name}: available "
                f"rows={report.get('rows', len(frame))} coverage={report.get('coverage', 0.0):.1%}",
                flush=True,
            )
        else:
            print(
                f"  optional source {source_name}: skipped ({report.get('reason', 'unavailable')})",
                flush=True,
            )

    source_report_payload = {
        "version": EXTERNAL_SOURCE_REPORT_VERSION,
        "source_set": external_source_set,
        "cache_policy": external_cache_policy,
        "timeout_seconds": float(external_source_timeout_sec),
        "start_date": start_date,
        "end_date": end_date,
        "requested_sources": list(requested_extra_sources),
        "available_sources": available_extra_sources,
        "reports": extra_reports,
    }
    EXTERNAL_SOURCE_REPORT_PATH.write_text(json.dumps(source_report_payload, indent=2, ensure_ascii=False))

    if require_extra_sources and requested_extra_sources and not available_extra_sources:
        skipped = "; ".join(
            f"{item.get('source')}: {item.get('reason', 'unavailable')}" for item in extra_reports
        )
        raise RuntimeError(
            "No optional external source passed validation, so this run would reproduce the baseline. "
            f"Requested source_set={external_source_set}. Skips: {skipped}"
        )

    if not frames:
        raise ValueError(
            "External weather requested, but no source cache/fetch produced usable rows. "
            "Use --skip-external-weather or provide/refresh external caches."
        )

    external = frames[0]
    for frame in frames[1:]:
        external = external.merge(frame, on=DATETIME_COL, how="outer")
    external = external.sort_values(DATETIME_COL).drop_duplicates(DATETIME_COL, keep="last")
    if include_openmeteo_forecast:
        external, coverage_report = apply_openmeteo_coverage_guard(
            external,
            forecast_prefixes,
            start_date,
            end_date,
            min_coverage=openmeteo_min_coverage,
            coverage_policy=openmeteo_fill_policy,
        )
        OPENMETEO_FORECAST_COVERAGE_REPORT_PATH.write_text(
            json.dumps(coverage_report, indent=2, ensure_ascii=False)
        )
        dropped = coverage_report.get("dropped_sources") or []
        if dropped:
            print(
                "  warning: dropped Open-Meteo forecast sources by strict coverage guard: "
                + ", ".join(dropped),
                flush=True,
            )
    return external.reset_index(drop=True)


def _guarded_source_blend(
    local_values: pd.Series,
    external_values: pd.Series,
    agreement_scale: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Average weather sources when they agree, but fall back to local forecast on outliers."""
    local_numeric = pd.to_numeric(local_values, errors="coerce")
    external_numeric = pd.to_numeric(external_values, errors="coerce")
    delta = external_numeric - local_numeric
    agreement = np.exp(-((delta.abs() / agreement_scale) ** 2)).clip(0.0, 1.0)
    blended = local_numeric * (1.0 - 0.5 * agreement) + external_numeric * (0.5 * agreement)
    return blended, delta, agreement


def add_guarded_external_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ext_era5_cloud_cover_low" in df:
        df["ext_era5_cloud_cover_low_fraction"] = df["ext_era5_cloud_cover_low"] / 100.0

    blend_specs = [
        ("wind_speed_10m", "ext_era5_wind_speed_10m", "ext_blend_wind_speed_10m", 3.0),
        ("wind_gusts_10m", "ext_era5_wind_gusts_10m", "ext_blend_wind_gusts_10m", 4.0),
        ("pressure_msl", "ext_era5_pressure_msl", "ext_blend_pressure_msl", 6.0),
        ("rain", "ext_era5_rain", "ext_blend_rain", 2.0),
        ("snowfall", "ext_era5_snowfall", "ext_blend_snowfall", 1.0),
        ("cloud_cover_low", "ext_era5_cloud_cover_low_fraction", "ext_blend_cloud_cover_low", 0.35),
    ]
    for local_col, external_col, output_col, scale in blend_specs:
        if local_col not in df or external_col not in df:
            continue
        blended, delta, agreement = _guarded_source_blend(df[local_col], df[external_col], scale)
        df[output_col] = blended
        df[f"{output_col}_delta"] = delta
        df[f"{output_col}_agreement"] = agreement

    if {"wind_direction_10m", "ext_era5_wind_direction_10m"}.issubset(df.columns):
        local_deg = pd.to_numeric(df["wind_direction_10m"], errors="coerce") * 360.0
        external_deg = pd.to_numeric(df["ext_era5_wind_direction_10m"], errors="coerce")
        circular_delta = ((external_deg - local_deg + 180.0) % 360.0) - 180.0
        agreement = np.exp(-((circular_delta.abs() / 45.0) ** 2)).clip(0.0, 1.0)
        local_rad = np.deg2rad(local_deg)
        external_rad = np.deg2rad(external_deg)
        x = np.cos(local_rad) * (1.0 - 0.5 * agreement) + np.cos(external_rad) * (0.5 * agreement)
        y = np.sin(local_rad) * (1.0 - 0.5 * agreement) + np.sin(external_rad) * (0.5 * agreement)
        df["ext_blend_wind_direction_10m_sin"] = y
        df["ext_blend_wind_direction_10m_cos"] = x
        df["ext_blend_wind_direction_10m_delta_deg"] = circular_delta
        df["ext_blend_wind_direction_10m_agreement"] = agreement

    active_prefixes = [
        prefix for prefix in OPENMETEO_FORECAST_MODEL_CATALOG if f"ext_omfc_{prefix}_observed" in df.columns
    ]
    forecast_ws100_cols = [
        col
        for col in [f"ext_omfc_{prefix}_wind_speed_100m" for prefix in active_prefixes]
        if col in df.columns
    ]
    forecast_ws10_cols = [
        col
        for col in [f"ext_omfc_{prefix}_wind_speed_10m" for prefix in active_prefixes]
        if col in df.columns
    ]
    forecast_pressure_cols = [
        col
        for col in [f"ext_omfc_{prefix}_pressure_msl" for prefix in active_prefixes]
        if col in df.columns
    ]
    forecast_pressure_hub_cols = [
        col
        for col in [f"ext_omfc_{prefix}_pressure_hub_wind_speed" for prefix in active_prefixes]
        if col in df.columns
    ]
    if forecast_ws100_cols:
        values = df[forecast_ws100_cols].apply(pd.to_numeric, errors="coerce")
        df["ext_omfc_consensus_wind_speed_100m"] = values.mean(axis=1)
        df["ext_omfc_spread_wind_speed_100m"] = values.std(axis=1).fillna(0.0)
        if "wind_speed_80m" in df:
            blended, delta, agreement = _guarded_source_blend(
                df["wind_speed_80m"],
                df["ext_omfc_consensus_wind_speed_100m"],
                3.0,
            )
            df["ext_blend_omfc_wind_speed_100m"] = blended
            df["ext_blend_omfc_wind_speed_100m_delta"] = delta
            df["ext_blend_omfc_wind_speed_100m_agreement"] = agreement
    if forecast_pressure_hub_cols:
        values = df[forecast_pressure_hub_cols].apply(pd.to_numeric, errors="coerce")
        df["ext_omfc_consensus_pressure_hub_wind_speed"] = values.mean(axis=1)
        df["ext_omfc_spread_pressure_hub_wind_speed"] = values.std(axis=1).fillna(0.0)
        if "wind_speed_80m" in df:
            blended, delta, agreement = _guarded_source_blend(
                df["wind_speed_80m"],
                df["ext_omfc_consensus_pressure_hub_wind_speed"],
                3.0,
            )
            df["ext_blend_omfc_pressure_hub_wind_speed"] = blended
            df["ext_blend_omfc_pressure_hub_wind_speed_delta"] = delta
            df["ext_blend_omfc_pressure_hub_wind_speed_agreement"] = agreement
    if forecast_ws10_cols:
        values = df[forecast_ws10_cols].apply(pd.to_numeric, errors="coerce")
        df["ext_omfc_consensus_wind_speed_10m"] = values.mean(axis=1)
        df["ext_omfc_spread_wind_speed_10m"] = values.std(axis=1).fillna(0.0)
        if "wind_speed_10m" in df:
            blended, delta, agreement = _guarded_source_blend(
                df["wind_speed_10m"],
                df["ext_omfc_consensus_wind_speed_10m"],
                2.2,
            )
            df["ext_blend_omfc_wind_speed_10m"] = blended
            df["ext_blend_omfc_wind_speed_10m_delta"] = delta
            df["ext_blend_omfc_wind_speed_10m_agreement"] = agreement
    if forecast_pressure_cols:
        values = df[forecast_pressure_cols].apply(pd.to_numeric, errors="coerce")
        df["ext_omfc_consensus_pressure_msl"] = values.mean(axis=1)
        df["ext_omfc_spread_pressure_msl"] = values.std(axis=1).fillna(0.0)
        if "pressure_msl" in df:
            blended, delta, agreement = _guarded_source_blend(
                df["pressure_msl"],
                df["ext_omfc_consensus_pressure_msl"],
                5.0,
            )
            df["ext_blend_omfc_pressure_msl"] = blended
            df["ext_blend_omfc_pressure_msl_delta"] = delta
            df["ext_blend_omfc_pressure_msl_agreement"] = agreement

    return df


def add_source_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Targeted wind-source interactions inspired by public wind-power pipelines."""
    out = df.copy()
    features: dict[str, pd.Series] = {}

    speed_specs = [
        ("ext_nasa_ws50m", "ext_nasa_t2m", "ext_nasa_air_density_2m", "ext_nasa_wd50m"),
        ("ext_cds_era5_wind_speed_100m", "ext_cds_era5_temperature_2m", "ext_cds_era5_air_density_2m", "ext_cds_era5_wind_direction_100m"),
        ("ext_meteoblue_wind_speed_10m", "ext_meteoblue_temperature_2m", "ext_meteoblue_air_density_2m", "ext_meteoblue_wind_direction_10m"),
        ("ext_meteostat_wind_speed", "ext_meteostat_temperature", "ext_meteostat_air_density_2m", "ext_meteostat_wind_direction"),
        ("ext_omfc_consensus_wind_speed_100m", "temperature_80m", "air_density", None),
        ("ext_blend_omfc_wind_speed_100m", "temperature_80m", "air_density", None),
    ]
    for speed_col, temp_col, density_col, direction_col in speed_specs:
        if speed_col not in out:
            continue
        speed = pd.to_numeric(out[speed_col], errors="coerce")
        prefix = speed_col.removeprefix("ext_")
        if temp_col in out:
            temp = pd.to_numeric(out[temp_col], errors="coerce")
            features[f"ext_interact_{prefix}_x_temp"] = speed * temp
        if density_col in out:
            density = pd.to_numeric(out[density_col], errors="coerce")
            features[f"ext_interact_{prefix}_x_density"] = speed * density
            features[f"ext_interact_{prefix}_wpd_like"] = 0.5 * density * speed**3
        if direction_col and f"{direction_col}_sin" in out and f"{direction_col}_cos" in out:
            features[f"ext_interact_{prefix}_x_wdir_sin"] = speed * out[f"{direction_col}_sin"]
            features[f"ext_interact_{prefix}_x_wdir_cos"] = speed * out[f"{direction_col}_cos"]

    delta_specs = [
        ("ext_nasa_ws50m", "ext_omfc_gfs_wind_speed_100m"),
        ("ext_nasa_ws50m", "ext_blend_omfc_wind_speed_100m"),
        ("ext_cds_era5_wind_speed_100m", "ext_omfc_gfs_wind_speed_100m"),
        ("ext_meteoblue_wind_speed_10m", "ext_omfc_gfs_wind_speed_10m"),
        ("ext_meteostat_wind_speed", "wind_speed_10m"),
    ]
    for left_col, right_col in delta_specs:
        if left_col in out and right_col in out:
            left = pd.to_numeric(out[left_col], errors="coerce")
            right = pd.to_numeric(out[right_col], errors="coerce")
            safe_name = f"{left_col}_minus_{right_col}".replace("ext_", "").replace("_wind_speed", "_ws")
            features[f"ext_interact_{safe_name}"] = left - right
            features[f"ext_interact_{safe_name}_ratio"] = left / (right.abs() + 0.1)

    shear_density_specs = [
        ("ext_nasa_ws50m", "ext_nasa_ws10m", "ext_nasa_air_density_2m"),
        ("ext_cds_era5_wind_speed_100m", "ext_cds_era5_wind_speed_10m", "ext_cds_era5_air_density_2m"),
        ("ext_omfc_consensus_wind_speed_100m", "ext_omfc_consensus_wind_speed_10m", "air_density"),
    ]
    for high_col, low_col, density_col in shear_density_specs:
        if {high_col, low_col, density_col}.issubset(out.columns):
            shear = pd.to_numeric(out[high_col], errors="coerce") - pd.to_numeric(out[low_col], errors="coerce")
            density = pd.to_numeric(out[density_col], errors="coerce")
            name = f"{high_col}_minus_{low_col}".replace("ext_", "").replace("_wind_speed", "_ws")
            features[f"ext_interact_{name}_x_density"] = shear * density

    if features:
        out = pd.concat([out, pd.DataFrame(features)], axis=1)
        print(f"  source interaction features: added {len(features)} columns", flush=True)
    return out


def add_external_weather_features(
    df: pd.DataFrame,
    cache_dir: Path = EXTERNAL_WEATHER_DIR,
    refresh: bool = False,
    include_nasa_power: bool = False,
    external_source_set: str = "baseline",
    external_source_timeout_sec: int | float = DEFAULT_EXTERNAL_SOURCE_TIMEOUT_SECONDS,
    external_cache_policy: str = "cache_first",
    require_extra_sources: bool = False,
    include_openmeteo_forecast: bool = True,
    openmeteo_forecast_set: str = "core_open_forecast",
    openmeteo_fill_policy: str = "legacy",
    openmeteo_min_coverage: float = OPENMETEO_COVERAGE_MIN_DEFAULT,
    include_openmeteo_pressure: bool = False,
    include_meteostat: bool = True,
    meteostat_cache_path: Path = METEOSTAT_CACHE_PATH,
    allow_partial_meteostat: bool = True,
    require_meteostat: bool = False,
    include_meteostat_derived: bool = False,
) -> pd.DataFrame:
    external = load_external_weather(
        df,
        cache_dir=cache_dir,
        refresh=refresh,
        include_nasa_power=include_nasa_power,
        external_source_set=external_source_set,
        external_source_timeout_sec=external_source_timeout_sec,
        external_cache_policy=external_cache_policy,
        require_extra_sources=require_extra_sources,
        include_openmeteo_forecast=include_openmeteo_forecast,
        openmeteo_forecast_set=openmeteo_forecast_set,
        openmeteo_fill_policy=openmeteo_fill_policy,
        openmeteo_min_coverage=openmeteo_min_coverage,
        include_openmeteo_pressure=include_openmeteo_pressure,
        include_meteostat=include_meteostat,
        meteostat_cache_path=meteostat_cache_path,
        allow_partial_meteostat=allow_partial_meteostat,
        require_meteostat=require_meteostat,
    )
    out = df.merge(external, on=DATETIME_COL, how="left")
    external_cols = [c for c in out.columns if c.startswith("ext_")]
    meteostat_source_cols = [c for c in external_cols if c.startswith("ext_meteostat_")]
    meteostat_observed = None
    if meteostat_source_cols:
        meteostat_observed = out[meteostat_source_cols].notna().any(axis=1).astype(float)
    missing_rows = out[external_cols].isna().all(axis=1).sum()
    if missing_rows:
        raise ValueError(f"External weather is missing for {missing_rows} timestamp rows")

    out = out.sort_values(DATETIME_COL).reset_index(drop=True)
    out[external_cols] = out[external_cols].apply(pd.to_numeric, errors="coerce")
    observed_cols = [col for col in external_cols if col.startswith("ext_omfc_") and col.endswith("_observed")]
    if observed_cols:
        out[observed_cols] = out[observed_cols].fillna(0.0)
    empty_external_cols = [col for col in external_cols if out[col].isna().all()]
    if empty_external_cols:
        print(
            "  warning: dropping empty external weather columns: "
            + ", ".join(empty_external_cols[:10])
            + ("..." if len(empty_external_cols) > 10 else ""),
            flush=True,
        )
        out = out.drop(columns=empty_external_cols)
        external_cols = [col for col in external_cols if col not in empty_external_cols]
    forecast_cols = [
        col
        for col in external_cols
        if col.startswith("ext_omfc_") and not col.endswith("_observed")
    ]
    other_external_cols = [col for col in external_cols if col not in forecast_cols]
    if other_external_cols:
        out[other_external_cols] = out[other_external_cols].interpolate(limit_direction="both").ffill().bfill()
    if forecast_cols:
        if openmeteo_fill_policy == "legacy":
            out[forecast_cols] = out[forecast_cols].interpolate(limit_direction="both").ffill().bfill()
        else:
            out[forecast_cols] = out[forecast_cols].interpolate(
                limit=3,
                limit_area="inside",
                limit_direction="both",
            )
    if meteostat_observed is not None:
        out["ext_meteostat_observed"] = meteostat_observed

    direction_feature_frames = []
    static_direction_cols = [
        "ext_era5_wind_direction_10m",
        "ext_era5_wind_direction_100m",
        "ext_cds_era5_wind_direction_10m",
        "ext_cds_era5_wind_direction_100m",
        "ext_meteoblue_wind_direction_10m",
        "ext_windy_wind_direction_surface",
        "ext_windy_wind_direction_100m",
        "ext_nasa_wd10m",
        "ext_nasa_wd50m",
        "ext_meteostat_wind_direction",
    ]
    dynamic_direction_cols = [
        col
        for col in out.columns
        if col.startswith("ext_omfc_") and "_wind_direction_" in col
    ]
    for full_col in [c for c in static_direction_cols + dynamic_direction_cols if c in out.columns]:
        radians = np.deg2rad(out[full_col])
        direction_feature_frames.append(
            pd.DataFrame(
                {
                    f"{full_col}_sin": np.sin(radians),
                    f"{full_col}_cos": np.cos(radians),
                }
            )
        )
    if direction_feature_frames:
        out = pd.concat([out] + direction_feature_frames, axis=1)

    if {"ext_era5_surface_pressure", "ext_era5_temperature_2m", "ext_era5_wind_speed_100m"}.issubset(out.columns):
        era5_t_k = out["ext_era5_temperature_2m"] + 273.15
        era5_p_pa = out["ext_era5_surface_pressure"] * 100.0
        out["ext_era5_air_density_2m"] = era5_p_pa / (R_SPECIFIC_AIR * era5_t_k)
        out["ext_era5_wpd_100m"] = 0.5 * out["ext_era5_air_density_2m"] * out["ext_era5_wind_speed_100m"] ** 3

    if {"ext_nasa_ps", "ext_nasa_t2m", "ext_nasa_ws50m"}.issubset(out.columns):
        nasa_t_k = out["ext_nasa_t2m"] + 273.15
        nasa_p_pa = out["ext_nasa_ps"] * 1000.0
        out["ext_nasa_air_density_2m"] = nasa_p_pa / (R_SPECIFIC_AIR * nasa_t_k)
        out["ext_nasa_wpd_50m"] = 0.5 * out["ext_nasa_air_density_2m"] * out["ext_nasa_ws50m"] ** 3

    if {"ext_cds_era5_surface_pressure", "ext_cds_era5_temperature_2m", "ext_cds_era5_wind_speed_100m"}.issubset(out.columns):
        cds_t_k = out["ext_cds_era5_temperature_2m"] + 273.15
        cds_p_pa = out["ext_cds_era5_surface_pressure"] * 100.0
        out["ext_cds_era5_air_density_2m"] = cds_p_pa / (R_SPECIFIC_AIR * cds_t_k)
        out["ext_cds_era5_wpd_100m"] = 0.5 * out["ext_cds_era5_air_density_2m"] * out["ext_cds_era5_wind_speed_100m"] ** 3

    if {"ext_meteoblue_pressure_msl", "ext_meteoblue_temperature_2m", "ext_meteoblue_wind_speed_10m"}.issubset(out.columns):
        mb_t_k = out["ext_meteoblue_temperature_2m"] + 273.15
        mb_p_pa = out["ext_meteoblue_pressure_msl"] * 100.0
        out["ext_meteoblue_air_density_2m"] = mb_p_pa / (R_SPECIFIC_AIR * mb_t_k)
        out["ext_meteoblue_wpd_10m"] = 0.5 * out["ext_meteoblue_air_density_2m"] * out["ext_meteoblue_wind_speed_10m"] ** 3

    if {"ext_meteostat_pressure_msl", "ext_meteostat_temperature", "ext_meteostat_wind_speed"}.issubset(out.columns):
        meteostat_t_k = out["ext_meteostat_temperature"] + 273.15
        meteostat_p_pa = out["ext_meteostat_pressure_msl"] * 100.0
        out["ext_meteostat_air_density_2m"] = meteostat_p_pa / (R_SPECIFIC_AIR * meteostat_t_k)
        out["ext_meteostat_wpd_10m"] = 0.5 * out["ext_meteostat_air_density_2m"] * out["ext_meteostat_wind_speed"] ** 3

    active_forecast_prefixes = [
        prefix for prefix in OPENMETEO_FORECAST_MODEL_CATALOG if f"ext_omfc_{prefix}_observed" in out.columns
    ]
    omfc_feature_frames = []
    for short_name in active_forecast_prefixes:
        pressure_col = f"ext_omfc_{short_name}_surface_pressure"
        temp_col = f"ext_omfc_{short_name}_temperature_2m"
        ws100_col = f"ext_omfc_{short_name}_wind_speed_100m"
        ws180_col = f"ext_omfc_{short_name}_wind_speed_180m"
        ws80_col = f"ext_omfc_{short_name}_wind_speed_80m"
        new_features = {}
        if {pressure_col, temp_col, ws100_col}.issubset(out.columns):
            t_k = out[temp_col] + 273.15
            p_pa = out[pressure_col] * 100.0
            density = p_pa / (R_SPECIFIC_AIR * t_k)
            new_features[f"ext_omfc_{short_name}_air_density_2m"] = density
            new_features[f"ext_omfc_{short_name}_wpd_100m"] = 0.5 * density * out[ws100_col] ** 3
        if {ws80_col, ws180_col}.issubset(out.columns):
            new_features[f"ext_omfc_{short_name}_rotor_shear_180_80"] = (
                (out[ws180_col] - out[ws80_col]) / (out[ws80_col] + 0.1)
            )

        pressure_ws_parts = []
        pressure_height_parts = []
        pressure_density_parts = []
        pressure_weights = []
        for level in OPENMETEO_PRESSURE_LEVELS:
            ws_col = f"ext_omfc_{short_name}_wind_speed_{level}"
            temp_col_level = f"ext_omfc_{short_name}_temperature_{level}"
            height_col = f"ext_omfc_{short_name}_geopotential_height_{level}"
            if {ws_col, temp_col_level, height_col}.issubset(out.columns):
                level_hpa = float(level.removesuffix("hPa"))
                temp_k = out[temp_col_level] + 273.15
                density = (level_hpa * 100.0) / (R_SPECIFIC_AIR * temp_k)
                new_features[f"ext_omfc_{short_name}_air_density_{level}"] = density
                new_features[f"ext_omfc_{short_name}_wpd_{level}"] = 0.5 * density * out[ws_col] ** 3
                height_delta = out[height_col] - HUB_HEIGHT_M
                new_features[f"ext_omfc_{short_name}_height_delta_hub_{level}"] = height_delta
                weight = 1.0 / (height_delta.abs() + 10.0)
                pressure_ws_parts.append(out[ws_col])
                pressure_height_parts.append(out[height_col])
                pressure_density_parts.append(density)
                pressure_weights.append(weight)
        if pressure_ws_parts:
            weight_sum = sum(pressure_weights)
            hub_ws = sum(w * s for w, s in zip(pressure_weights, pressure_ws_parts)) / weight_sum
            hub_height = sum(w * s for w, s in zip(pressure_weights, pressure_height_parts)) / weight_sum
            hub_density = sum(w * s for w, s in zip(pressure_weights, pressure_density_parts)) / weight_sum
            new_features[f"ext_omfc_{short_name}_pressure_hub_wind_speed"] = hub_ws
            new_features[f"ext_omfc_{short_name}_pressure_hub_height"] = hub_height
            new_features[f"ext_omfc_{short_name}_pressure_hub_air_density"] = hub_density
            new_features[f"ext_omfc_{short_name}_pressure_hub_wpd"] = 0.5 * hub_density * hub_ws ** 3
        if new_features:
            omfc_feature_frames.append(pd.DataFrame(new_features))
    if omfc_feature_frames:
        out = pd.concat([out] + omfc_feature_frames, axis=1)

    paired_deltas = [
        ("ext_era5_wind_speed_10m", "wind_speed_10m", "ext_era5_ws10_minus_openm"),
        ("ext_era5_wind_speed_100m", "wind_speed_80m", "ext_era5_ws100_minus_openm80"),
        ("ext_era5_pressure_msl", "pressure_msl", "ext_era5_pressure_minus_openm"),
        ("ext_nasa_ws10m", "wind_speed_10m", "ext_nasa_ws10_minus_openm"),
        ("ext_nasa_ps", "pressure_msl", "ext_nasa_pressure_minus_openm"),
        ("ext_cds_era5_wind_speed_10m", "wind_speed_10m", "ext_cds_era5_ws10_minus_openm"),
        ("ext_cds_era5_wind_speed_100m", "wind_speed_80m", "ext_cds_era5_ws100_minus_openm80"),
        ("ext_cds_era5_pressure_msl", "pressure_msl", "ext_cds_era5_pressure_minus_openm"),
        ("ext_meteoblue_wind_speed_10m", "wind_speed_10m", "ext_meteoblue_ws10_minus_openm"),
        ("ext_meteoblue_pressure_msl", "pressure_msl", "ext_meteoblue_pressure_minus_openm"),
        ("ext_windy_wind_speed_surface", "wind_speed_10m", "ext_windy_wind_speed_surface_delta_10m"),
        ("ext_windy_wind_speed_100m", "wind_speed_80m", "ext_windy_wind_speed_100m_delta_80m"),
        ("ext_renewables_ninja_wind_speed_hub", "wind_speed_80m", "ext_renewables_ninja_ws_hub_delta_80m"),
        ("ext_meteostat_wind_speed", "wind_speed_10m", "ext_meteostat_wind_speed_delta_10m"),
        ("ext_meteostat_temperature", "temperature_80m", "ext_meteostat_temperature_delta_80m"),
        ("ext_meteostat_pressure_msl", "pressure_msl", "ext_meteostat_pressure_delta_msl"),
    ]
    for short_name in active_forecast_prefixes:
        paired_deltas.extend(
            [
                (
                    f"ext_omfc_{short_name}_wind_speed_10m",
                    "wind_speed_10m",
                    f"ext_omfc_{short_name}_ws10_minus_openm",
                ),
                (
                    f"ext_omfc_{short_name}_wind_speed_100m",
                    "wind_speed_80m",
                    f"ext_omfc_{short_name}_ws100_minus_openm80",
                ),
                (
                    f"ext_omfc_{short_name}_pressure_msl",
                    "pressure_msl",
                    f"ext_omfc_{short_name}_pressure_minus_openm",
                ),
            ]
        )
    for external_col, local_col, new_col in paired_deltas:
        if external_col in out and local_col in out:
            local_values = out[local_col]
            if external_col == "ext_nasa_ps":
                local_values = local_values / 10.0
            out[new_col] = out[external_col] - local_values

    external_feature_cols = [c for c in out.columns if c.startswith("ext_")]
    empty_external_feature_cols = [col for col in external_feature_cols if out[col].isna().all()]
    if empty_external_feature_cols:
        print(
            "  warning: dropping empty derived external columns: "
            + ", ".join(empty_external_feature_cols[:10])
            + ("..." if len(empty_external_feature_cols) > 10 else ""),
            flush=True,
        )
        out = out.drop(columns=empty_external_feature_cols)
        external_feature_cols = [col for col in external_feature_cols if col not in empty_external_feature_cols]
    forecast_feature_cols = [
        col
        for col in external_feature_cols
        if col.startswith("ext_omfc_") and not col.endswith("_observed")
    ]
    non_forecast_feature_cols = [col for col in external_feature_cols if col not in forecast_feature_cols]
    if non_forecast_feature_cols:
        out[non_forecast_feature_cols] = (
            out[non_forecast_feature_cols].interpolate(limit_direction="both").ffill().bfill()
        )
    if forecast_feature_cols:
        if openmeteo_fill_policy == "legacy":
            out[forecast_feature_cols] = out[forecast_feature_cols].interpolate(limit_direction="both").ffill().bfill()
        else:
            out[forecast_feature_cols] = out[forecast_feature_cols].interpolate(
                limit=3,
                limit_area="inside",
                limit_direction="both",
            )
    bad_cols = [c for c in non_forecast_feature_cols if out[c].isna().any()]
    if bad_cols:
        raise ValueError(f"External weather columns contain NaNs after filling: {bad_cols[:10]}")

    out = add_guarded_external_averages(out)
    if ENABLE_SOURCE_INTERACTIONS:
        out = add_source_interaction_features(out)

    if include_meteostat_derived:
        # Opt-in only: this 10-feature block improved Q1 CV but worsened the public score.
        if "ext_meteostat_wind_speed" in out.columns:
            ws = out["ext_meteostat_wind_speed"]
            out["ext_meteostat_wind_speed_lag_1h"] = ws.shift(1)
            out["ext_meteostat_wind_speed_lag_2h"] = ws.shift(2)
            out["ext_meteostat_wind_speed_lag_3h"] = ws.shift(3)

        if "ext_meteostat_wpd_10m" in out.columns:
            wpd = out["ext_meteostat_wpd_10m"]
            out["ext_meteostat_wpd_10m_lag_1h"] = wpd.shift(1)
            out["ext_meteostat_wpd_10m_lag_3h"] = wpd.shift(3)

        if {"ext_meteostat_temperature", "ext_meteostat_dew_point"}.issubset(out.columns):
            out["ext_meteostat_dew_point_depression"] = (
                out["ext_meteostat_temperature"] - out["ext_meteostat_dew_point"]
            )

        if "ext_meteostat_condition_code" in out.columns:
            coco = out["ext_meteostat_condition_code"]
            out["ext_meteostat_is_fog"] = coco.between(5, 6).astype(float)
            out["ext_meteostat_is_precipitation"] = coco.between(7, 19).astype(float)
            out["ext_meteostat_is_storm"] = (coco >= 20).astype(float)

        if {"ext_meteostat_wind_speed", "wind_speed_10m"}.issubset(out.columns):
            out["ext_meteostat_wind_speed_bias_factor"] = (
                out["ext_meteostat_wind_speed"] / (out["wind_speed_10m"] + 0.1)
            )

    new_meta_cols = [
        c for c in out.columns
        if c.startswith("ext_meteostat_") and out[c].isna().any()
        and c != "ext_meteostat_observed"
    ]
    if new_meta_cols:
        out[new_meta_cols] = out[new_meta_cols].ffill().bfill()

    return out


def _parse_month_tokens(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def load_so_ups_res_monthly(
    cache_path: Path = SO_UPS_RES_CACHE_PATH,
    policy: str = "legacy_best_gap",
    exclude_months: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame | None:
    if not cache_path.exists():
        return None
    if policy not in SO_UPS_RES_POLICIES:
        raise ValueError(f"Unknown SO UPS RES policy {policy!r}; expected one of {sorted(SO_UPS_RES_POLICIES)}")
    frame = pd.read_csv(cache_path)
    if frame.empty or "month" not in frame:
        return None
    frame["month"] = pd.to_datetime(frame["month"]).dt.to_period("M").dt.to_timestamp()
    frame = frame.dropna(subset=["month"])
    if frame.empty:
        return None
    before = len(frame)
    if policy == "legacy_best_gap":
        months_to_exclude = set(SO_UPS_RES_LEGACY_BEST_GAP_MONTHS)
    elif policy == "custom_exclude":
        months_to_exclude = set(exclude_months or ())
    else:
        months_to_exclude = set()
    if months_to_exclude:
        month_str = frame["month"].dt.strftime("%Y-%m")
        frame = frame.loc[~month_str.isin(months_to_exclude)].copy()
    dropped = before - len(frame)
    print(
        "  SO UPS RES policy: "
        f"{policy}; usable_months={len(frame)}/{before}; "
        f"excluded={','.join(sorted(months_to_exclude)) if months_to_exclude else 'none'}; "
        f"dropped={dropped}",
        flush=True,
    )
    rename_map = {
        "installed_mw": "ext_so_ups_res_rostov_installed_mw",
        "generation_month_mwh": "ext_so_ups_res_rostov_generation_month_mwh",
        "generation_ytd_mwh": "ext_so_ups_res_rostov_generation_ytd_mwh",
        "curtailment_hours_month": "ext_so_ups_res_rostov_curtailment_hours_month",
        "max_curtailment_mw_month": "ext_so_ups_res_rostov_max_curtailment_mw_month",
        "max_deviation_mw_month": "ext_so_ups_res_rostov_max_deviation_mw_month",
    }
    keep = ["month"] + [col for col in rename_map if col in frame.columns]
    out = frame[keep].rename(columns=rename_map).copy()
    for col in out.columns:
        if col != "month":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    empty_metric_cols = [
        col for col in out.columns
        if col != "month" and not out[col].notna().any()
    ]
    if empty_metric_cols:
        out = out.drop(columns=empty_metric_cols)
        print(
            "  SO UPS RES dropped empty metric columns: "
            + ", ".join(empty_metric_cols),
            flush=True,
        )
    if {
        "ext_so_ups_res_rostov_installed_mw",
        "ext_so_ups_res_rostov_generation_month_mwh",
    }.issubset(out.columns):
        days = out["month"].dt.days_in_month
        denom = out["ext_so_ups_res_rostov_installed_mw"] * 24.0 * days
        out["ext_so_ups_res_rostov_capacity_factor_month"] = (
            out["ext_so_ups_res_rostov_generation_month_mwh"] / denom.replace(0.0, np.nan)
        )
    return out.sort_values("month").drop_duplicates("month", keep="last").reset_index(drop=True)


def add_so_ups_res_features(
    df: pd.DataFrame,
    cache_path: Path = SO_UPS_RES_CACHE_PATH,
    include_valid_months: bool = False,
    policy: str = "legacy_best_gap",
    exclude_months: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    monthly = load_so_ups_res_monthly(cache_path, policy=policy, exclude_months=exclude_months)
    if monthly is None:
        print(f"  warning: SO UPS RES monthly cache not found at {cache_path}; continuing without it")
        return df

    out = df.copy()
    out["_so_ups_month"] = pd.to_datetime(out[DATETIME_COL]).dt.to_period("M").dt.to_timestamp()
    monthly = monthly.rename(columns={"month": "_so_ups_report_month"})
    out = out.merge(monthly, left_on="_so_ups_month", right_on="_so_ups_report_month", how="left")
    feature_cols = [col for col in out.columns if col.startswith("ext_so_ups_res_")]
    matched_months = int(out.loc[:, feature_cols].notna().any(axis=1).groupby(out["_so_ups_month"]).any().sum()) if feature_cols else 0
    print(
        f"  SO UPS RES join: feature_cols={len(feature_cols)} matched_months={matched_months}",
        flush=True,
    )
    out = out.drop(columns=["_so_ups_month", "_so_ups_report_month"])
    if not include_valid_months and TARGET in out.columns:
        train_target_mask = out[TARGET].notna()
        if train_target_mask.any():
            train_end = out.loc[train_target_mask, DATETIME_COL].max()
            valid_like_mask = out[DATETIME_COL] > train_end
            out.loc[valid_like_mask, feature_cols] = np.nan
    return out


def _rotor_equivalent_speed(ws10, ws80, ws120, ws180, hub_height: float):
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

    integral_v3 = (v_at_h**3 * weights).sum(axis=1) / weights.sum()
    return np.cbrt(np.maximum(integral_v3, 0.0))


def _empirical_power_curve(v, cfg: PhysicsConfig):
    """Manufacturer-style cubic power curve, returns MW per turbine at standard density."""
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


def _farm_power_for_config(
    df: pd.DataFrame,
    cfg: PhysicsConfig,
    wind_speed_eq: np.ndarray | pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if wind_speed_eq is None:
        wind_speed_eq = _rotor_equivalent_speed(
            df["wind_speed_10m"].values,
            df["wind_speed_80m"].values,
            df["wind_speed_120m"].values,
            df["wind_speed_180m"].values,
            hub_height=cfg.hub_height,
        )
    wind_speed_eq = np.asarray(wind_speed_eq, dtype=float)
    p_per_turbine_std = _empirical_power_curve(wind_speed_eq, cfg)
    density_corr = (df["air_density"].values / RHO_STANDARD) ** cfg.density_correction_exp
    p_per_turbine = np.clip(
        p_per_turbine_std * density_corr * cfg.efficiency_factor,
        0.0,
        cfg.p_rated_per_turbine,
    )
    p_farm = np.clip(p_per_turbine * df["n_working"].values, 0.0, P_RATED_FARM)
    return p_per_turbine, p_farm


def initialize_latent_availability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Default non-leaky latent availability features before fold calibration."""
    df = df.copy()
    df["n_working_latent"] = df["n_working_monthly_avg"]
    df["availability_latent"] = df["availability_monthly_avg"]
    df["availability_delta_from_monthly"] = 0.0
    df["P_physics_farm_latent"] = np.clip(
        df["P_physics_per_turbine"] * df["n_working_latent"], 0.0, P_RATED_FARM
    )
    return df


def fit_latent_availability_calibrator(train_df: pd.DataFrame) -> LatentAvailabilityCalibrator:
    """Estimate actual working turbines from train-only target and physics potential.

    The repair column is a monthly average, so it is unsafe as an exact hourly cap.
    This lightweight calibrator learns only month-level deltas from the training
    slice supplied by CV/final fit; validation rows never contribute to it.
    """
    if TARGET not in train_df:
        return LatentAvailabilityCalibrator(global_delta=0.0)

    per_turbine = pd.to_numeric(train_df["P_physics_per_turbine"], errors="coerce")
    monthly_n = pd.to_numeric(train_df["n_working_monthly_avg"], errors="coerce")
    target = pd.to_numeric(train_df[TARGET], errors="coerce")
    reliable = (
        np.isfinite(per_turbine)
        & np.isfinite(monthly_n)
        & np.isfinite(target)
        & (per_turbine >= LATENT_MIN_PER_TURBINE_MW)
        & (target > 0.0)
    )
    if int(reliable.sum()) < LATENT_MIN_GROUP_ROWS:
        return LatentAvailabilityCalibrator(global_delta=0.0)

    proxy_n = (target[reliable] / per_turbine[reliable]).clip(0.0, N_TOTAL_TURBINES)
    delta = (proxy_n - monthly_n[reliable]).clip(-LATENT_DELTA_CLIP, LATENT_DELTA_CLIP)
    weights = per_turbine[reliable].clip(upper=train_df["P_physics_per_turbine"].max())
    global_delta = float(np.average(delta, weights=np.maximum(weights, 1e-3)))

    work = pd.DataFrame(
        {
            "month": train_df.loc[reliable, "month"].astype(int).values,
            "delta": delta.values,
            "weight": np.maximum(weights.values, 1e-3),
        }
    )
    monthly_delta: dict[int, float] = {}
    for month, group in work.groupby("month"):
        if len(group) >= LATENT_MIN_GROUP_ROWS:
            monthly_delta[int(month)] = float(np.average(group["delta"], weights=group["weight"]))

    return LatentAvailabilityCalibrator(
        global_delta=float(np.clip(global_delta, -LATENT_DELTA_CLIP, LATENT_DELTA_CLIP)),
        monthly_delta=monthly_delta,
    )


def apply_latent_availability_calibrator(
    df: pd.DataFrame,
    calibrator: LatentAvailabilityCalibrator,
) -> pd.DataFrame:
    df = df.copy()
    month_delta = df["month"].astype(int).map(calibrator.monthly_delta)
    delta = month_delta.fillna(calibrator.global_delta).astype(float)
    n_latent = (df["n_working_monthly_avg"].astype(float) + delta).clip(0.0, N_TOTAL_TURBINES)
    df["n_working_latent"] = n_latent
    df["availability_latent"] = n_latent / N_TOTAL_TURBINES
    df["availability_delta_from_monthly"] = df["availability_latent"] - df["availability_monthly_avg"]
    df["P_physics_farm_latent"] = np.clip(
        df["P_physics_per_turbine"] * df["n_working_latent"], 0.0, P_RATED_FARM
    )
    return df


def prepare_model_frames_with_latent_availability(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    train_idx: np.ndarray | None = None,
    eval_idx: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    fit_df = train_df.iloc[train_idx].copy() if train_idx is not None else train_df.copy()
    calibrator = fit_latent_availability_calibrator(fit_df)
    fit_df = apply_latent_availability_calibrator(fit_df, calibrator)
    pred_df = apply_latent_availability_calibrator(pred_df, calibrator)
    eval_df = None
    if eval_idx is not None:
        eval_df = train_df.iloc[eval_idx].copy()
        eval_df = apply_latent_availability_calibrator(eval_df, calibrator)
    return fit_df, pred_df, eval_df


RESIDUAL_PRIOR_FEATURES = [
    "residual_prior_global",
    "residual_prior_month_hour",
    "residual_prior_regime_hour",
    "target_prior_month_hour",
    "physics_error_ratio_prior_month_hour",
]

REGIME_PRIOR_V2_FEATURES = [
    "residual_prior_regime_month_hour_v2",
    "residual_prior_regime_wsbin_hour_v2",
    "target_prior_regime_hour_v2",
    "physics_error_ratio_prior_regime_wsbin_v2",
]

WEATHER_ANALOG_RESIDUAL_FEATURES = [
    "weather_analog_residual_median",
    "weather_analog_target_median",
    "weather_analog_confidence",
]

MULTI_REGIME_PRIOR_FEATURES = [
    "multi_regime_residual_median",
    "multi_regime_target_median",
    "multi_regime_confidence",
    "multi_regime_shear_residual_median",
    "multi_regime_uncertainty_residual_median",
]


def add_windfm_diagnostic_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    oof_path: Path | None = None,
    valid_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof_path = Path(oof_path or (ROOT / "windfm_oof_predictions.csv"))
    valid_path = Path(valid_path or (ROOT / "windfm_valid_predictions.csv"))
    if not oof_path.exists() or not valid_path.exists():
        print(
            "  WindFM diagnostic: skipped "
            f"(missing {oof_path.name} and/or {valid_path.name}; run a dedicated WindFM OOF export first)",
            flush=True,
        )
        return train, valid

    def _load(path: Path, output_col: str) -> pd.DataFrame:
        frame = pd.read_csv(path, parse_dates=[DATETIME_COL])
        pred_col = "prediction" if "prediction" in frame.columns else None
        if pred_col is None:
            candidates = [col for col in frame.columns if col != DATETIME_COL]
            if not candidates:
                raise ValueError(f"{path} has no prediction column")
            pred_col = candidates[0]
        return frame[[DATETIME_COL, pred_col]].rename(columns={pred_col: output_col})

    try:
        train_add = _load(oof_path, "ext_windfm_oof_power")
        valid_add = _load(valid_path, "ext_windfm_valid_power")
    except Exception as exc:
        print(f"  WindFM diagnostic: skipped ({exc})", flush=True)
        return train, valid

    train = train.merge(train_add, on=DATETIME_COL, how="left")
    valid = valid.merge(valid_add, on=DATETIME_COL, how="left")
    if train["ext_windfm_oof_power"].isna().any() or valid["ext_windfm_valid_power"].isna().any():
        print("  WindFM diagnostic: skipped (prediction files do not cover all rows)", flush=True)
        train = train.drop(columns=[c for c in ["ext_windfm_oof_power"] if c in train])
        valid = valid.drop(columns=[c for c in ["ext_windfm_valid_power"] if c in valid])
        return train, valid
    train["ext_windfm_power"] = train["ext_windfm_oof_power"]
    valid["ext_windfm_power"] = valid["ext_windfm_valid_power"]
    train["ext_windfm_residual_vs_physics"] = train["ext_windfm_power"] - train["P_physics_farm"]
    valid["ext_windfm_residual_vs_physics"] = valid["ext_windfm_power"] - valid["P_physics_farm"]
    train = train.drop(columns=["ext_windfm_oof_power"])
    valid = valid.drop(columns=["ext_windfm_valid_power"])
    print("  WindFM diagnostic: joined fold-safe OOF/valid prediction features", flush=True)
    return train, valid


def initialize_residual_prior_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in RESIDUAL_PRIOR_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    return df


def initialize_regime_prior_v2_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in REGIME_PRIOR_V2_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    return df


def initialize_weather_analog_residual_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in WEATHER_ANALOG_RESIDUAL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    return df


def initialize_multi_regime_prior_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in MULTI_REGIME_PRIOR_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    return df


def _ensure_residual_prior_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add fold-local grouping helpers used only to build target-derived priors."""
    df = df.copy()
    if "month" not in df.columns:
        df["month"] = pd.to_datetime(df[DATETIME_COL]).dt.month.astype(int)
    if "hour" not in df.columns:
        if "hour_of_day" in df.columns:
            df["hour"] = pd.to_numeric(df["hour_of_day"], errors="coerce").fillna(0).astype(int)
        else:
            df["hour"] = pd.to_datetime(df[DATETIME_COL]).dt.hour.astype(int)
    if "wind_regime_code" not in df.columns:
        if "wind_speed_eq" in df.columns:
            v_eq = pd.to_numeric(df["wind_speed_eq"], errors="coerce")
            df["wind_regime_code"] = np.select(
                [
                    v_eq < ACTIVE_PHYSICS_CONFIG.v_cut_in + 0.4,
                    v_eq < ACTIVE_PHYSICS_CONFIG.v_rated - 1.0,
                    v_eq < ACTIVE_PHYSICS_CONFIG.v_cut_out - 2.0,
                    v_eq >= ACTIVE_PHYSICS_CONFIG.v_cut_out - 2.0,
                ],
                [0, 1, 2, 3],
                default=1,
            ).astype(int)
        else:
            df["wind_regime_code"] = 1
    return df


def _ensure_regime_prior_v2_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_residual_prior_group_columns(df)
    if "wind_speed_bin_v2" not in df.columns:
        if "wind_speed_eq" in df.columns:
            speed = pd.to_numeric(df["wind_speed_eq"], errors="coerce")
        else:
            speed = pd.to_numeric(df.get("wind_speed_100m", df.get("wind_speed_80m", 0.0)), errors="coerce")
        bins = [-np.inf, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 16.0, np.inf]
        df["wind_speed_bin_v2"] = (
            pd.cut(speed, bins=bins, labels=False)
            .astype("float")
            .fillna(0)
            .astype(int)
        )
    return df


def _ensure_weather_analog_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_regime_prior_v2_group_columns(df)
    if "direction_sector_80m" not in df.columns:
        if "wind_direction_80m" in df.columns:
            direction = pd.to_numeric(df["wind_direction_80m"], errors="coerce").fillna(0.0) % 1.0
            df["direction_sector_80m"] = np.floor(direction * 8.0).clip(0, 7).astype(int)
        else:
            df["direction_sector_80m"] = 0
    if "weather_analog_speed_bin" not in df.columns:
        speed = pd.to_numeric(df.get("wind_speed_eq", df.get("wind_speed_80m", 0.0)), errors="coerce")
        bins = [-np.inf, 3.0, 5.5, 8.0, 11.0, np.inf]
        df["weather_analog_speed_bin"] = (
            pd.cut(speed, bins=bins, labels=False)
            .astype("float")
            .fillna(0)
            .astype(int)
        )
    return df


def add_multi_regime_context_features(df: pd.DataFrame, cfg: PhysicsConfig | None = None) -> pd.DataFrame:
    cfg = cfg or ACTIVE_PHYSICS_CONFIG
    df = _ensure_weather_analog_group_columns(df)
    speed = pd.to_numeric(df.get("wind_speed_eq", df.get("wind_speed_80m", 0.0)), errors="coerce").fillna(0.0)
    direction_sector = pd.to_numeric(df["direction_sector_80m"], errors="coerce").fillna(0).astype(int).clip(0, 7)
    direction_quadrant = (direction_sector // 2).astype(int).clip(0, 3)

    bins = [
        -np.inf,
        cfg.v_cut_in - 0.5,
        cfg.v_cut_in + 0.5,
        5.0,
        7.0,
        9.0,
        cfg.v_rated - 0.5,
        cfg.v_rated + 1.0,
        np.inf,
    ]
    df["wind_speed_bin_fine"] = (
        pd.cut(speed, bins=bins, labels=False)
        .astype("float")
        .fillna(0)
        .astype(int)
    )
    df["direction_quadrant_80m"] = direction_quadrant

    shear = pd.to_numeric(df.get("shear_120_80", 0.0), errors="coerce").fillna(0.0)
    df["shear_regime_code"] = (
        pd.cut(shear, bins=[-np.inf, -1.0, 0.25, 1.5, np.inf], labels=False)
        .astype("float")
        .fillna(1)
        .astype(int)
    )
    if {"wind_direction_120m", "wind_direction_80m"}.issubset(df.columns):
        diff = (df["wind_direction_120m"] - df["wind_direction_80m"]) % 1.0
        diff = diff.where(diff <= 0.5, diff - 1.0)
        df["direction_shear_abs_120_80"] = np.abs(diff * 360.0)
    else:
        df["direction_shear_abs_120_80"] = 0.0
    df["direction_shear_regime_code"] = (
        pd.cut(df["direction_shear_abs_120_80"], bins=[-np.inf, 10.0, 25.0, np.inf], labels=False)
        .astype("float")
        .fillna(0)
        .astype(int)
    )

    spread_sources = []
    for col in [
        "wind_speed_80m",
        "wind_speed_120m",
        "ext_nasa_ws50m",
        "ext_omfc_gfs_wind_speed_80m",
        "ext_omfc_gfs_wind_speed_100m",
        "ext_omfc_gfs_wind_speed_120m",
    ]:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            spread_sources.append(pd.to_numeric(df[col], errors="coerce"))
    if len(spread_sources) >= 2:
        spread_frame = pd.concat(spread_sources, axis=1)
        df["wind_forecast_spread_mps"] = spread_frame.std(axis=1).fillna(0.0)
    else:
        df["wind_forecast_spread_mps"] = 0.0
    df["forecast_disagreement_code"] = (
        pd.cut(df["wind_forecast_spread_mps"], bins=[-np.inf, 0.6, 1.2, np.inf], labels=False)
        .astype("float")
        .fillna(0)
        .astype(int)
    )

    df["wind_multi_regime_code"] = (
        df["wind_regime_code"].astype(int) * 4 + df["direction_quadrant_80m"].astype(int)
    ).astype(int)
    df["wind_multi_context_code"] = (
        df["wind_multi_regime_code"].astype(int)
        + 16 * df["shear_regime_code"].astype(int)
        + 64 * df["forecast_disagreement_code"].astype(int)
    ).astype(int)

    partial_weight = np.clip(
        (speed - cfg.v_cut_in) / max(cfg.v_rated - cfg.v_cut_in, 1e-6),
        0.0,
        1.0,
    )
    rated_headroom = np.clip((cfg.v_cut_out - speed) / max(cfg.v_cut_out - cfg.v_rated, 1e-6), 0.0, 1.0)
    for quadrant in range(4):
        q_mask = (direction_quadrant == quadrant).astype(float)
        df[f"dir_quad_{quadrant}_partial_weight"] = q_mask * partial_weight
        df[f"dir_quad_{quadrant}_rated_headroom"] = q_mask * rated_headroom
    for bin_id in range(8):
        bin_mask = (df["wind_speed_bin_fine"].astype(int) == bin_id).astype(float)
        df[f"fine_speed_bin_{bin_id}_shear"] = bin_mask * shear
        df[f"fine_speed_bin_{bin_id}_spread"] = bin_mask * df["wind_forecast_spread_mps"]

    return df


def _group_prior_map(
    fit_df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    global_value: float,
    shrink: float = 80.0,
) -> dict[tuple, float]:
    if any(col not in fit_df.columns for col in [value_col] + group_cols):
        return {}
    work = fit_df.dropna(subset=[value_col] + group_cols).copy()
    if work.empty:
        return {}
    grouped = work.groupby(group_cols)[value_col].agg(["mean", "count"])
    values = (
        (grouped["mean"] * grouped["count"] + global_value * shrink)
        / (grouped["count"] + shrink)
    )
    return {key if isinstance(key, tuple) else (key,): float(value) for key, value in values.items()}


def _group_median_prior_map(
    fit_df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    global_value: float,
    shrink: float = 120.0,
) -> dict[tuple, float]:
    if any(col not in fit_df.columns for col in [value_col] + group_cols):
        return {}
    work = fit_df.dropna(subset=[value_col] + group_cols).copy()
    if work.empty:
        return {}
    grouped = work.groupby(group_cols)[value_col].agg(["median", "count"])
    values = (
        (grouped["median"] * grouped["count"] + global_value * shrink)
        / (grouped["count"] + shrink)
    )
    return {key if isinstance(key, tuple) else (key,): float(value) for key, value in values.items()}


def _group_count_map(fit_df: pd.DataFrame, group_cols: list[str]) -> dict[tuple, int]:
    if any(col not in fit_df.columns for col in group_cols):
        return {}
    work = fit_df.dropna(subset=group_cols).copy()
    if work.empty:
        return {}
    counts = work.groupby(group_cols).size()
    return {key if isinstance(key, tuple) else (key,): int(value) for key, value in counts.items()}


def _apply_group_map(df: pd.DataFrame, group_cols: list[str], mapping: dict[tuple, float], default: float) -> pd.Series:
    if any(col not in df.columns for col in group_cols):
        return pd.Series(default, index=df.index, dtype=float)
    if not mapping:
        return pd.Series(default, index=df.index, dtype=float)
    keys = list(zip(*(df[col].values for col in group_cols)))
    return pd.Series([mapping.get(tuple(key), default) for key in keys], index=df.index, dtype=float)


def apply_residual_prior_features(
    fit_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    eval_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    if TARGET not in fit_df or "residual" not in fit_df:
        return fit_df, pred_df, eval_df
    fit_df = _ensure_residual_prior_group_columns(initialize_residual_prior_features(fit_df))
    pred_df = _ensure_residual_prior_group_columns(initialize_residual_prior_features(pred_df))
    if eval_df is not None:
        eval_df = _ensure_residual_prior_group_columns(initialize_residual_prior_features(eval_df))

    residual = pd.to_numeric(fit_df["residual"], errors="coerce")
    target = pd.to_numeric(fit_df[TARGET], errors="coerce")
    physics = pd.to_numeric(fit_df["P_physics_farm"], errors="coerce")
    global_residual = float(residual.mean()) if residual.notna().any() else 0.0
    global_target = float(target.mean()) if target.notna().any() else 0.0
    ratio_values = residual / (physics.abs() + 1.0)
    global_ratio = float(ratio_values.mean()) if ratio_values.notna().any() else 0.0

    work = fit_df.copy()
    work["_physics_error_ratio"] = ratio_values
    maps = {
        "residual_prior_month_hour": (
            ["month", "hour"],
            _group_prior_map(work, "residual", ["month", "hour"], global_residual),
            global_residual,
        ),
        "residual_prior_regime_hour": (
            ["wind_regime_code", "hour"],
            _group_prior_map(work, "residual", ["wind_regime_code", "hour"], global_residual),
            global_residual,
        ),
        "target_prior_month_hour": (
            ["month", "hour"],
            _group_prior_map(work, TARGET, ["month", "hour"], global_target),
            global_target,
        ),
        "physics_error_ratio_prior_month_hour": (
            ["month", "hour"],
            _group_prior_map(work, "_physics_error_ratio", ["month", "hour"], global_ratio),
            global_ratio,
        ),
    }
    for frame in [fit_df, pred_df] + ([eval_df] if eval_df is not None else []):
        frame["residual_prior_global"] = global_residual
        for col, (group_cols, mapping, default) in maps.items():
            frame[col] = _apply_group_map(frame, group_cols, mapping, default)
    return fit_df, pred_df, eval_df


def apply_regime_prior_v2_features(
    fit_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    eval_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    if TARGET not in fit_df or "residual" not in fit_df:
        return fit_df, pred_df, eval_df
    fit_df = _ensure_regime_prior_v2_group_columns(initialize_regime_prior_v2_features(fit_df))
    pred_df = _ensure_regime_prior_v2_group_columns(initialize_regime_prior_v2_features(pred_df))
    if eval_df is not None:
        eval_df = _ensure_regime_prior_v2_group_columns(initialize_regime_prior_v2_features(eval_df))

    residual = pd.to_numeric(fit_df["residual"], errors="coerce")
    target = pd.to_numeric(fit_df[TARGET], errors="coerce")
    physics = pd.to_numeric(fit_df["P_physics_farm"], errors="coerce")
    global_residual = float(residual.mean()) if residual.notna().any() else 0.0
    global_target = float(target.mean()) if target.notna().any() else 0.0
    ratio_values = residual / (physics.abs() + 1.0)
    global_ratio = float(ratio_values.mean()) if ratio_values.notna().any() else 0.0

    work = fit_df.copy()
    work["_physics_error_ratio"] = ratio_values
    maps = {
        "residual_prior_regime_month_hour_v2": (
            ["wind_regime_code", "month", "hour"],
            _group_prior_map(work, "residual", ["wind_regime_code", "month", "hour"], global_residual, shrink=160.0),
            global_residual,
        ),
        "residual_prior_regime_wsbin_hour_v2": (
            ["wind_regime_code", "wind_speed_bin_v2", "hour"],
            _group_prior_map(work, "residual", ["wind_regime_code", "wind_speed_bin_v2", "hour"], global_residual, shrink=180.0),
            global_residual,
        ),
        "target_prior_regime_hour_v2": (
            ["wind_regime_code", "hour"],
            _group_prior_map(work, TARGET, ["wind_regime_code", "hour"], global_target, shrink=180.0),
            global_target,
        ),
        "physics_error_ratio_prior_regime_wsbin_v2": (
            ["wind_regime_code", "wind_speed_bin_v2"],
            _group_prior_map(work, "_physics_error_ratio", ["wind_regime_code", "wind_speed_bin_v2"], global_ratio, shrink=220.0),
            global_ratio,
        ),
    }
    for frame in [fit_df, pred_df] + ([eval_df] if eval_df is not None else []):
        for col, (group_cols, mapping, default) in maps.items():
            frame[col] = _apply_group_map(frame, group_cols, mapping, default)
    return fit_df, pred_df, eval_df


def apply_weather_analog_residual_features(
    fit_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    eval_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    if TARGET not in fit_df or "residual" not in fit_df:
        return fit_df, pred_df, eval_df
    fit_df = _ensure_weather_analog_group_columns(initialize_weather_analog_residual_features(fit_df))
    pred_df = _ensure_weather_analog_group_columns(initialize_weather_analog_residual_features(pred_df))
    if eval_df is not None:
        eval_df = _ensure_weather_analog_group_columns(initialize_weather_analog_residual_features(eval_df))

    residual = pd.to_numeric(fit_df["residual"], errors="coerce")
    target = pd.to_numeric(fit_df[TARGET], errors="coerce")
    global_residual = float(residual.median()) if residual.notna().any() else 0.0
    global_target = float(target.median()) if target.notna().any() else 0.0
    group_cols = ["month", "hour", "wind_regime_code", "weather_analog_speed_bin", "direction_sector_80m"]
    residual_map = _group_median_prior_map(fit_df, "residual", group_cols, global_residual, shrink=120.0)
    target_map = _group_median_prior_map(fit_df, TARGET, group_cols, global_target, shrink=160.0)
    count_map = _group_count_map(fit_df, group_cols)

    for frame in [fit_df, pred_df] + ([eval_df] if eval_df is not None else []):
        frame["weather_analog_residual_median"] = _apply_group_map(
            frame,
            group_cols,
            residual_map,
            global_residual,
        )
        frame["weather_analog_target_median"] = _apply_group_map(
            frame,
            group_cols,
            target_map,
            global_target,
        )
        if count_map:
            keys = list(zip(*(frame[col].values for col in group_cols)))
            counts = pd.Series([count_map.get(tuple(key), 0) for key in keys], index=frame.index, dtype=float)
            frame["weather_analog_confidence"] = (counts / (counts + 120.0)).fillna(0.0)
        else:
            frame["weather_analog_confidence"] = 0.0
    return fit_df, pred_df, eval_df


def apply_multi_regime_prior_features(
    fit_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    eval_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    if TARGET not in fit_df or "residual" not in fit_df:
        return fit_df, pred_df, eval_df
    fit_df = initialize_multi_regime_prior_features(add_multi_regime_context_features(fit_df))
    pred_df = initialize_multi_regime_prior_features(add_multi_regime_context_features(pred_df))
    if eval_df is not None:
        eval_df = initialize_multi_regime_prior_features(add_multi_regime_context_features(eval_df))

    residual = pd.to_numeric(fit_df["residual"], errors="coerce")
    target = pd.to_numeric(fit_df[TARGET], errors="coerce")
    global_residual = float(residual.median()) if residual.notna().any() else 0.0
    global_target = float(target.median()) if target.notna().any() else 0.0

    speed_dir_hour = ["hour", "wind_speed_bin_fine", "direction_quadrant_80m"]
    speed_dir_shear = ["wind_speed_bin_fine", "direction_quadrant_80m", "shear_regime_code"]
    uncertainty_group = ["wind_regime_code", "forecast_disagreement_code", "direction_quadrant_80m"]
    residual_map = _group_median_prior_map(fit_df, "residual", speed_dir_hour, global_residual, shrink=260.0)
    target_map = _group_median_prior_map(fit_df, TARGET, speed_dir_hour, global_target, shrink=300.0)
    shear_map = _group_median_prior_map(fit_df, "residual", speed_dir_shear, global_residual, shrink=300.0)
    uncertainty_map = _group_median_prior_map(fit_df, "residual", uncertainty_group, global_residual, shrink=320.0)
    count_map = _group_count_map(fit_df, speed_dir_hour)

    for frame in [fit_df, pred_df] + ([eval_df] if eval_df is not None else []):
        frame["multi_regime_residual_median"] = _apply_group_map(
            frame,
            speed_dir_hour,
            residual_map,
            global_residual,
        )
        frame["multi_regime_target_median"] = _apply_group_map(
            frame,
            speed_dir_hour,
            target_map,
            global_target,
        )
        frame["multi_regime_shear_residual_median"] = _apply_group_map(
            frame,
            speed_dir_shear,
            shear_map,
            global_residual,
        )
        frame["multi_regime_uncertainty_residual_median"] = _apply_group_map(
            frame,
            uncertainty_group,
            uncertainty_map,
            global_residual,
        )
        if count_map:
            keys = list(zip(*(frame[col].values for col in speed_dir_hour)))
            counts = pd.Series([count_map.get(tuple(key), 0) for key in keys], index=frame.index, dtype=float)
            frame["multi_regime_confidence"] = (counts / (counts + 260.0)).fillna(0.0)
        else:
            frame["multi_regime_confidence"] = 0.0
    return fit_df, pred_df, eval_df


def add_physical_features(
    df: pd.DataFrame,
    cfg: PhysicsConfig = DEFAULT_PHYSICS,
    include_physics_variants: bool = False,
) -> pd.DataFrame:
    df = df.copy()

    t_k = df["temperature_80m"] + 273.15
    p_pa = df["pressure_msl"] * 100.0
    df["air_density"] = p_pa / (R_SPECIFIC_AIR * t_k)
    df["air_density_ratio"] = df["air_density"] / RHO_STANDARD

    for h in (10, 80, 120, 180):
        df[f"wind_speed_{h}m_cubed"] = df[f"wind_speed_{h}m"] ** 3

    df["wpd_80m"] = 0.5 * df["air_density"] * df["wind_speed_80m_cubed"]
    df["wpd_120m"] = 0.5 * df["air_density"] * df["wind_speed_120m_cubed"]

    eps = 0.1
    df["alpha_80_10"] = np.log(
        df["wind_speed_80m"].clip(lower=eps) / df["wind_speed_10m"].clip(lower=eps)
    ) / np.log(80.0 / 10.0)
    df["alpha_120_80"] = np.log(
        df["wind_speed_120m"].clip(lower=eps) / df["wind_speed_80m"].clip(lower=eps)
    ) / np.log(120.0 / 80.0)
    df["alpha_180_120"] = np.log(
        df["wind_speed_180m"].clip(lower=eps) / df["wind_speed_120m"].clip(lower=eps)
    ) / np.log(180.0 / 120.0)

    df["shear_120_80"] = df["wind_speed_120m"] - df["wind_speed_80m"]
    df["shear_180_80"] = df["wind_speed_180m"] - df["wind_speed_80m"]
    df["shear_80_10"] = df["wind_speed_80m"] - df["wind_speed_10m"]

    df["wind_speed_eq"] = _rotor_equivalent_speed(
        df["wind_speed_10m"].values,
        df["wind_speed_80m"].values,
        df["wind_speed_120m"].values,
        df["wind_speed_180m"].values,
        hub_height=cfg.hub_height,
    )
    df["wind_speed_eq_cubed"] = df["wind_speed_eq"] ** 3

    p_per_turbine_std = _empirical_power_curve(df["wind_speed_eq"].values, cfg)
    density_corr = (df["air_density"].values / RHO_STANDARD) ** cfg.density_correction_exp
    p_per_turbine = p_per_turbine_std * density_corr * cfg.efficiency_factor
    df["P_physics_per_turbine"] = np.clip(p_per_turbine, 0.0, cfg.p_rated_per_turbine)

    df["repair_monthly_avg"] = df[REPAIR_COL].astype(float)
    df["n_working_monthly_avg"] = N_TOTAL_TURBINES - df["repair_monthly_avg"]
    df["availability_monthly_avg"] = df["n_working_monthly_avg"] / N_TOTAL_TURBINES
    df["n_working"] = df["n_working_monthly_avg"]
    df["availability_fraction"] = df["availability_monthly_avg"]
    df["P_physics_farm"] = np.clip(
        df["P_physics_per_turbine"] * df["n_working"], 0.0, P_RATED_FARM
    )
    if include_physics_variants:
        physics_variants = {
            "manufacturer": MANUFACTURER_PHYSICS,
            "public_best": PUBLIC_BEST_PHYSICS,
            "hub80": replace(PUBLIC_BEST_PHYSICS, hub_height=80.0),
        }
        for name, variant_cfg in physics_variants.items():
            variant_v_eq = (
                df["wind_speed_eq"].values
                if variant_cfg.hub_height == cfg.hub_height
                else None
            )
            _, variant_farm = _farm_power_for_config(df, variant_cfg, variant_v_eq)
            df[f"P_physics_farm_{name}"] = variant_farm
            df[f"P_physics_delta_{name}"] = variant_farm - df["P_physics_farm"].values
    df = initialize_latent_availability_features(df)
    df = add_weather_bias_features(df, cfg)
    df = initialize_physics_first_v2_features(df)
    if ACTIVE_MODEL_SET in {"empirical_curve_v3", "empirical_curve_v4_guarded"}:
        df = initialize_empirical_curve_features(df, cfg)
    if ACTIVE_MODEL_SET in {"empirical_curve_v4", "empirical_curve_v4_guarded"}:
        df = initialize_empirical_curve_v4_features(df, cfg, ACTIVE_EMPIRICAL_CURVE_V4_CONFIG)
    if ACTIVE_MODEL_SET == "empirical_curve_v4_guarded":
        df = add_empirical_curve_v4_guarded_features(df, ACTIVE_EMPIRICAL_CURVE_V4_CONFIG)

    safe_v3 = np.maximum(df["wind_speed_eq_cubed"].values, 1e-3)
    df["cp_effective"] = np.clip(
        p_per_turbine_std * 1e6 / (0.5 * RHO_STANDARD * A_ROTOR * safe_v3),
        0.0,
        0.6,
    )

    v_eq = df["wind_speed_eq"]
    df["is_below_cutin"] = (v_eq < cfg.v_cut_in).astype(int)
    df["is_above_cutout"] = (v_eq >= cfg.v_cut_out).astype(int)
    df["is_in_rated"] = ((v_eq >= cfg.v_rated) & (v_eq < cfg.v_cut_out)).astype(int)
    df["is_in_partial"] = ((v_eq >= cfg.v_cut_in) & (v_eq < cfg.v_rated)).astype(int)
    regime_conditions = [
        v_eq < cfg.v_cut_in,
        (v_eq >= cfg.v_cut_in) & (v_eq < cfg.v_rated),
        (v_eq >= cfg.v_rated) & (v_eq < cfg.v_cut_out),
        v_eq >= cfg.v_cut_out,
    ]
    df["wind_regime_code"] = np.select(regime_conditions, [0, 1, 2, 3], default=1).astype(int)
    df["wind_speed_to_cutin"] = v_eq - cfg.v_cut_in
    df["wind_speed_to_rated"] = v_eq - cfg.v_rated
    df["wind_speed_to_cutout"] = cfg.v_cut_out - v_eq

    for h in (10, 80, 120, 180):
        col = f"wind_direction_{h}m"
        df[f"wd_{h}m_sin"] = np.sin(2.0 * np.pi * df[col])
        df[f"wd_{h}m_cos"] = np.cos(2.0 * np.pi * df[col])

    if ENABLE_DIRECTION_SECTOR_FEATURES:
        df = _ensure_weather_analog_group_columns(df)
        sector = df["direction_sector_80m"].astype(int)
        speed_bin = df["weather_analog_speed_bin"].astype(int)
        df["direction_sector_120m"] = (
            np.floor((pd.to_numeric(df["wind_direction_120m"], errors="coerce").fillna(0.0) % 1.0) * 8.0)
            .clip(0, 7)
            .astype(int)
        )
        for sector_id in range(8):
            df[f"dir80_sector_{sector_id}"] = (sector == sector_id).astype(int)
        for bin_id in range(5):
            df[f"speed_eq_bin_{bin_id}"] = (speed_bin == bin_id).astype(int)
        for sector_id in range(8):
            sector_mask = sector == sector_id
            for bin_id in range(5):
                df[f"dir80_sector_{sector_id}_speed_bin_{bin_id}"] = (
                    sector_mask & (speed_bin == bin_id)
                ).astype(int)

    diff = (df["wind_direction_120m"] - df["wind_direction_80m"]) % 1.0
    diff = diff.where(diff <= 0.5, diff - 1.0)
    df["direction_shear_120_80"] = diff * 360.0

    if ENABLE_MULTI_REGIME_FEATURES:
        df = add_multi_regime_context_features(df, cfg)

    v10 = df["wind_speed_10m"].clip(lower=eps)
    df["turbulence_intensity"] = (df["wind_gusts_10m"] - df["wind_speed_10m"]) / v10
    df["gust_factor"] = df["wind_gusts_10m"] / v10

    df["temp_gradient_120_80"] = df["temperature_120m"] - df["temperature_80m"]
    df["is_stable"] = (df["temp_gradient_120_80"] > 0).astype(int)

    has_precip = (df["rain"] > 0) | (df["snowfall"] > 0) | (df["showers"] > 0)
    df["ice_risk"] = ((df["temperature_80m"] < cfg.ice_threshold) & has_precip).astype(int)

    df["hour_sin"] = np.sin(2.0 * np.pi * df["hour_of_day"] / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * df["hour_of_day"] / 24.0)
    df["month_sin"] = np.sin(2.0 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * df["month"] / 12.0)
    df["is_night"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] > 20)).astype(int)

    return df


def _rolling_slope(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(y)
    if int(mask.sum()) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)[mask]
    y = y[mask]
    return float(np.polyfit(x, y, 1)[0])


def _weather_dynamics_features(df: pd.DataFrame) -> list[pd.Series]:
    """Kaggle-inspired input-feature dynamics, using only weather/physics X."""
    features: list[pd.Series] = []
    for col in [c for c in WEATHER_DYNAMICS_BASE_COLS if c in df.columns]:
        s = pd.to_numeric(df[col], errors="coerce")
        if not s.notna().any():
            continue
        safe_name = col.replace("ext_", "").replace("_", "")
        for span in WEATHER_DYNAMICS_EWM_SPANS:
            features.append(
                s.ewm(span=span, min_periods=1, adjust=False)
                .mean()
                .rename(f"weather_dyn_{safe_name}_ema{span}")
            )
        for window in WEATHER_DYNAMICS_MEDIAN_WINDOWS:
            features.append(
                s.rolling(window, min_periods=1)
                .median()
                .rename(f"weather_dyn_{safe_name}_median{window}")
            )
        for window in WEATHER_DYNAMICS_TREND_WINDOWS:
            roll = s.rolling(window, min_periods=2)
            features.append(
                roll.apply(_rolling_slope, raw=True).rename(
                    f"weather_dyn_{safe_name}_trend{window}"
                )
            )
            features.append(
                (roll.std() / (roll.mean().abs() + 0.1)).rename(
                    f"weather_dyn_{safe_name}_relstd{window}"
                )
            )

    for left_col, right_col, name in WEATHER_SOURCE_DISAGREEMENT_PAIRS:
        if left_col not in df.columns or right_col not in df.columns:
            continue
        left = pd.to_numeric(df[left_col], errors="coerce")
        right = pd.to_numeric(df[right_col], errors="coerce")
        if not left.notna().any() or not right.notna().any():
            continue
        delta = left - right
        features.extend(
            [
                delta.rename(f"weather_dyn_{name}"),
                delta.clip(lower=0.0).rename(f"weather_dyn_{name}_pos"),
                delta.clip(upper=0.0).rename(f"weather_dyn_{name}_neg"),
                delta.abs().rename(f"weather_dyn_{name}_abs"),
                (left / (right.abs() + 0.1)).rename(f"weather_dyn_{name}_ratio"),
            ]
        )

    source_cols = [
        col for col in [
            "ext_omfc_gfs_wind_speed_100m",
            "ext_nasa_ws50m",
            "ext_cds_era5_wind_speed_100m",
        ]
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any()
    ]
    if len(source_cols) >= 2:
        source_frame = pd.concat(
            [pd.to_numeric(df[col], errors="coerce") for col in source_cols],
            axis=1,
        )
        source_frame.columns = source_cols
        features.extend(
            [
                source_frame.mean(axis=1).rename("weather_dyn_source_ws_mean"),
                source_frame.std(axis=1).rename("weather_dyn_source_ws_std"),
                (source_frame.max(axis=1) - source_frame.min(axis=1)).rename(
                    "weather_dyn_source_ws_range"
                ),
            ]
        )
    return features


def add_temporal_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weather-only sequence context. No generation target history is used."""
    df = df.copy().sort_values(DATETIME_COL).reset_index(drop=True)
    features = []

    for col in [c for c in TEMPORAL_BASE_COLS if c in df.columns]:
        s = df[col]
        for lag in TEMPORAL_LAGS:
            features.append(s.shift(lag).rename(f"{col}_lag{lag}"))
            features.append(s.shift(-lag).rename(f"{col}_lead{lag}"))
        for window in ROLLING_WINDOWS:
            features.append(
                s.rolling(window, center=True, min_periods=1).mean().rename(
                    f"{col}_roll{window}_mean"
                )
            )
        features.append((s - s.shift(1)).rename(f"{col}_diff1"))
        features.append((s.shift(-1) - s).rename(f"{col}_lead_diff1"))

    for col in [c for c in ROLLING_STD_COLS if c in df.columns]:
        s = df[col]
        for window in ROLLING_STD_WINDOWS:
            features.append(
                s.rolling(window, center=True, min_periods=2).std().rename(
                    f"{col}_roll{window}_std"
                )
            )

    for col in [c for c in DELTA_COLS if c in df.columns]:
        s = df[col]
        for lag in DELTA_LAGS:
            features.append((s - s.shift(lag)).rename(f"{col}_delta{lag}"))

    if ENABLE_WEATHER_DYNAMICS_FEATURES:
        features.extend(_weather_dynamics_features(df))

    extra = pd.concat(features, axis=1)
    return pd.concat([df, extra], axis=1).copy()


def apply_submission_bounds(pred, df: pd.DataFrame) -> np.ndarray:
    return np.clip(np.asarray(pred, dtype=float), 0.0, P_RATED_FARM)


def platform_error_percent(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred) / P_RATED_FARM * 100.0)


def _q1_valid_indices_by_year(train: pd.DataFrame) -> dict[int, np.ndarray]:
    return {year: valid_idx for year, _, valid_idx, _ in q1_folds(train)}


def _weighted_oof_error_percent(
    predictions_by_year: dict[int, np.ndarray],
    y_true: dict[int, np.ndarray],
    fold_weights: dict[int, float],
) -> float:
    total = 0.0
    weight_total = 0.0
    for year, pred in predictions_by_year.items():
        fold_weight = float(fold_weights.get(year, 1.0))
        total += fold_weight * platform_error_percent(y_true[year], pred)
        weight_total += fold_weight
    return float(total / max(weight_total, 1e-12))


def _blend_oof_predictions(oof: dict[int, dict[str, np.ndarray]], blend_weights: dict[str, float]) -> dict[int, np.ndarray]:
    return {year: blend_model_predictions(by_model, blend_weights) for year, by_model in oof.items()}


def _calibrate_values_for_regime(values: np.ndarray, rule: dict) -> np.ndarray:
    if rule.get("kind") == "affine":
        return float(rule.get("a", 1.0)) * values + float(rule.get("b", 0.0))
    if rule.get("kind") == "isotonic":
        x = np.asarray(rule.get("x_thresholds", []), dtype=float)
        y = np.asarray(rule.get("y_thresholds", []), dtype=float)
        if len(x) >= 2 and len(y) == len(x):
            return np.interp(values, x, y, left=y[0], right=y[-1])
    return values


def apply_regime_prediction_calibrator(
    pred: np.ndarray,
    df: pd.DataFrame,
    calibrator: dict | None,
) -> np.ndarray:
    pred = np.asarray(pred, dtype=float).copy()
    if not calibrator or calibrator.get("mode") == "none" or "wind_regime_code" not in df:
        return apply_submission_bounds(pred, df)
    regimes = pd.to_numeric(df["wind_regime_code"], errors="coerce").fillna(1).astype(int).values
    out = pred.copy()
    by_regime = calibrator.get("by_regime", {})
    for regime, rule in by_regime.items():
        mask = regimes == int(regime)
        if int(mask.sum()) == 0:
            continue
        raw_cal = _calibrate_values_for_regime(pred[mask], rule)
        shrinkage = float(rule.get("shrinkage", 0.0))
        out[mask] = pred[mask] + shrinkage * (raw_cal - pred[mask])
    return apply_submission_bounds(out, df)


def _fit_affine_regime_rule(
    pred: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
) -> dict:
    residual = target - pred
    start_b = float(np.clip(np.median(residual), -2.0, 2.0)) if len(residual) else 0.0

    def objective(params: np.ndarray) -> float:
        a, b = params
        corrected = a * pred + b
        loss = np.average(np.abs(target - corrected), weights=sample_weight)
        regularization = 0.02 * float((a - 1.0) ** 2) + 0.002 * float(b**2)
        return float(loss + regularization)

    result = minimize(
        objective,
        x0=np.array([1.0, start_b], dtype=float),
        method="SLSQP",
        bounds=[(0.92, 1.08), (-2.0, 2.0)],
        options={"ftol": 1e-8, "maxiter": 200},
    )
    a, b = result.x if result.success else np.array([1.0, start_b], dtype=float)
    return {
        "kind": "affine",
        "a": float(np.clip(a, 0.92, 1.08)),
        "b": float(np.clip(b, -2.0, 2.0)),
        "shrinkage": REGIME_AFFINE_SHRINKAGE,
        "success": bool(result.success),
    }


def _fit_isotonic_regime_rule(
    pred: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
) -> dict:
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(out_of_bounds="clip", increasing=True)
    model.fit(pred, target, sample_weight=sample_weight)
    return {
        "kind": "isotonic",
        "x_thresholds": [float(v) for v in model.X_thresholds_],
        "y_thresholds": [float(v) for v in model.y_thresholds_],
        "shrinkage": REGIME_ISOTONIC_SHRINKAGE,
        "success": True,
    }


def fit_regime_prediction_calibrator(
    train: pd.DataFrame,
    oof: dict[int, dict[str, np.ndarray]],
    y_true: dict[int, np.ndarray],
    blend_weights: dict[str, float],
    fold_weights: dict[int, float],
    mode: str = "none",
    min_rows: int = REGIME_CALIBRATION_MIN_ROWS,
) -> dict:
    if mode not in REGIME_CALIBRATION_MODES:
        raise ValueError(f"Unknown regime calibration mode {mode!r}")
    blended_by_year = _blend_oof_predictions(oof, blend_weights)
    base_error = _weighted_oof_error_percent(blended_by_year, y_true, fold_weights)
    if mode == "none":
        return {
            "mode": "none",
            "enabled": False,
            "base_weighted_q1_cv": float(base_error),
            "calibrated_weighted_q1_cv": float(base_error),
            "improvement_abs": 0.0,
            "by_regime": {},
        }

    valid_indices_by_year = _q1_valid_indices_by_year(train)
    pred_parts, target_parts, regime_parts, weight_parts = [], [], [], []
    for year, pred in blended_by_year.items():
        valid_idx = valid_indices_by_year[year]
        regimes = train.iloc[valid_idx]["wind_regime_code"].astype(int).values
        n_rows = len(pred)
        pred_parts.append(np.asarray(pred, dtype=float))
        target_parts.append(np.asarray(y_true[year], dtype=float))
        regime_parts.append(regimes)
        weight_parts.append(np.full(n_rows, float(fold_weights.get(year, 1.0)) / max(n_rows, 1)))

    pred_all = np.concatenate(pred_parts)
    target_all = np.concatenate(target_parts)
    regime_all = np.concatenate(regime_parts)
    sample_weight_all = np.concatenate(weight_parts)

    by_regime: dict[int, dict] = {}
    for regime in sorted(np.unique(regime_all).astype(int)):
        mask = regime_all == regime
        n_rows = int(mask.sum())
        raw_mae = float(np.mean(np.abs(target_all[mask] - pred_all[mask]))) if n_rows else float("nan")
        raw_bias = float(np.mean(target_all[mask] - pred_all[mask])) if n_rows else float("nan")
        if n_rows < min_rows:
            by_regime[int(regime)] = {
                "kind": "identity",
                "n_rows": n_rows,
                "raw_mae_mw": raw_mae,
                "raw_bias_mw": raw_bias,
                "reason": f"below min_rows={min_rows}",
            }
            continue
        if mode == "affine":
            rule = _fit_affine_regime_rule(pred_all[mask], target_all[mask], sample_weight_all[mask])
        else:
            rule = _fit_isotonic_regime_rule(pred_all[mask], target_all[mask], sample_weight_all[mask])
        raw_cal = _calibrate_values_for_regime(pred_all[mask], rule)
        calibrated = pred_all[mask] + float(rule["shrinkage"]) * (raw_cal - pred_all[mask])
        rule.update(
            {
                "n_rows": n_rows,
                "raw_mae_mw": raw_mae,
                "calibrated_mae_mw": float(np.mean(np.abs(target_all[mask] - calibrated))),
                "raw_bias_mw": raw_bias,
                "calibrated_bias_mw": float(np.mean(target_all[mask] - calibrated)),
            }
        )
        by_regime[int(regime)] = rule

    calibrated_by_year = {}
    for year, pred in blended_by_year.items():
        valid_idx = valid_indices_by_year[year]
        fold_df = train.iloc[valid_idx].reset_index(drop=True)
        calibrated_by_year[year] = apply_regime_prediction_calibrator(
            pred,
            fold_df,
            {"mode": mode, "by_regime": by_regime},
        )
    calibrated_error = _weighted_oof_error_percent(calibrated_by_year, y_true, fold_weights)
    return {
        "mode": mode,
        "enabled": True,
        "min_rows": int(min_rows),
        "base_weighted_q1_cv": float(base_error),
        "calibrated_weighted_q1_cv": float(calibrated_error),
        "improvement_abs": float(base_error - calibrated_error),
        "by_regime": by_regime,
    }


def compute_regime_oof_diagnostics(
    train: pd.DataFrame,
    oof: dict[int, dict[str, np.ndarray]],
    y_true: dict[int, np.ndarray],
    blend_weights: dict[str, float],
    fold_weights: dict[int, float],
) -> dict:
    valid_indices_by_year = _q1_valid_indices_by_year(train)
    blended_by_year = _blend_oof_predictions(oof, blend_weights)
    by_year: dict[str, dict] = {}
    all_parts: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for year, pred in blended_by_year.items():
        valid_idx = valid_indices_by_year[year]
        regimes = train.iloc[valid_idx]["wind_regime_code"].astype(int).values
        target = np.asarray(y_true[year], dtype=float)
        year_record: dict[str, dict] = {}
        for regime in sorted(np.unique(regimes).astype(int)):
            mask = regimes == regime
            if int(mask.sum()) == 0:
                continue
            y_reg = target[mask]
            p_reg = pred[mask]
            year_record[str(int(regime))] = {
                "rows": int(mask.sum()),
                "mae_mw": float(np.mean(np.abs(y_reg - p_reg))),
                "platform_error_percent": platform_error_percent(y_reg, p_reg),
                "residual_bias_mw": float(np.mean(y_reg - p_reg)),
                "target_mean_mw": float(np.mean(y_reg)),
                "prediction_mean_mw": float(np.mean(p_reg)),
            }
            all_parts.setdefault(int(regime), []).append((y_reg, p_reg))
        by_year[str(year)] = year_record
    by_regime: dict[str, dict] = {}
    for regime, parts in sorted(all_parts.items()):
        y_all = np.concatenate([item[0] for item in parts])
        p_all = np.concatenate([item[1] for item in parts])
        by_regime[str(regime)] = {
            "rows": int(len(y_all)),
            "mae_mw": float(np.mean(np.abs(y_all - p_all))),
            "platform_error_percent": platform_error_percent(y_all, p_all),
            "residual_bias_mw": float(np.mean(y_all - p_all)),
            "target_mean_mw": float(np.mean(y_all)),
            "prediction_mean_mw": float(np.mean(p_all)),
        }
    return {
        "fold_weights": {str(year): float(weight) for year, weight in fold_weights.items()},
        "by_year": by_year,
        "by_regime": by_regime,
    }


def cat_params(spec_name: str, verbose: bool, override_iterations: int | None = None) -> dict:
    if spec_name.startswith("cat5"):
        params = {
            "iterations": 2200,
            "learning_rate": 0.035,
            "depth": 5,
            "l2_leaf_reg": 12,
            "min_data_in_leaf": 50,
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "random_seed": SEED,
            "early_stopping_rounds": 150,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.8,
            "allow_writing_files": False,
            "thread_count": 1,
            "verbose": 200 if verbose else False,
        }
    elif spec_name.startswith("cat7"):
        params = {
            "iterations": 1800,
            "learning_rate": 0.032,
            "depth": 7,
            "l2_leaf_reg": 18,
            "min_data_in_leaf": 70,
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "random_seed": SEED + 11,
            "early_stopping_rounds": 150,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.75,
            "rsm": 0.75,
            "allow_writing_files": False,
            "thread_count": 1,
            "verbose": 200 if verbose else False,
        }
    else:
        raise ValueError(f"Unknown CatBoost spec: {spec_name}")

    if CATBOOST_TASK_TYPE == "GPU":
        params["task_type"] = "GPU"
        params["metric_period"] = 5
        if CATBOOST_DEVICES:
            params["devices"] = CATBOOST_DEVICES
        params.pop("thread_count", None)
        params.pop("rsm", None)

    if override_iterations is not None:
        params["iterations"] = int(override_iterations)
        params.pop("early_stopping_rounds", None)
    return params


def hgb_model(spec_name: str) -> HistGradientBoostingRegressor:
    params = {
        "loss": "absolute_error",
        "max_iter": 900,
        "learning_rate": 0.03,
        "max_leaf_nodes": 24,
        "min_samples_leaf": 55,
        "l2_regularization": 0.3,
        "random_state": SEED,
        "early_stopping": True,
        "validation_fraction": 0.15,
    }
    dynamic = load_hgb_dynamic_models().get(spec_name)
    if dynamic is not None:
        params.update(dynamic["params"])
    elif spec_name in HGB_STATIC_PARAM_OVERRIDES:
        params.update(HGB_STATIC_PARAM_OVERRIDES[spec_name])
    elif spec_name in HGB_OPT_FALLBACK_SOURCES:
        params.update(HGB_STATIC_PARAM_OVERRIDES[HGB_OPT_FALLBACK_SOURCES[spec_name]])
    elif spec_name != "hgb_direct":
        raise ValueError(f"Unknown HGB spec: {spec_name}")
    return HistGradientBoostingRegressor(**params)


def lgb_params(spec_name: str, override_iterations: int | None = None) -> dict:
    base = {
        "objective": "regression_l1",
        "metric": "mae",
        "n_estimators": 1200,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 80,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.05,
        "reg_lambda": 0.50,
        "random_state": SEED + 29,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if spec_name == "lgb_residual_l1":
        base.update(
            n_estimators=1400,
            learning_rate=0.020,
            num_leaves=24,
            max_depth=5,
            min_child_samples=110,
            subsample=0.90,
            colsample_bytree=0.75,
            reg_alpha=0.10,
            reg_lambda=1.00,
            random_state=SEED + 31,
        )
    elif spec_name != "lgb_direct_l1":
        raise ValueError(f"Unknown LightGBM spec: {spec_name}")

    if override_iterations is not None:
        base["n_estimators"] = int(override_iterations)
    return base


def etr_params(spec_name: str, override_iterations: int | None = None) -> dict:
    base = {
        "n_estimators": 650,
        "criterion": "squared_error",
        "max_features": 0.58,
        "min_samples_leaf": 10,
        "bootstrap": False,
        "random_state": SEED + 101,
        "n_jobs": -1,
    }
    if spec_name == "etr_direct_smooth":
        base.update(
            n_estimators=560,
            max_features=0.72,
            min_samples_leaf=28,
            max_depth=22,
            bootstrap=True,
            max_samples=0.88,
            random_state=SEED + 103,
        )
    elif spec_name == "etr_residual":
        base.update(
            n_estimators=720,
            max_features=0.52,
            min_samples_leaf=12,
            random_state=SEED + 107,
        )
    elif spec_name == "etr_residual_smooth":
        base.update(
            n_estimators=620,
            max_features=0.68,
            min_samples_leaf=30,
            max_depth=20,
            bootstrap=True,
            max_samples=0.90,
            random_state=SEED + 109,
        )
    elif spec_name != "etr_direct":
        raise ValueError(f"Unknown ExtraTrees spec: {spec_name}")

    if override_iterations is not None:
        base["n_estimators"] = int(override_iterations)
    return base


def fit_predict_etr(
    spec_name: str,
    fit_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: list[str],
    y_fit,
    override_iterations: int | None = None,
) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    x_fit = imputer.fit_transform(fit_df[feature_cols])
    x_pred = imputer.transform(pred_df[feature_cols])
    model = ExtraTreesRegressor(**etr_params(spec_name, override_iterations=override_iterations))
    model.fit(x_fit, y_fit)
    return model.predict(x_pred)


def _regime_v2_hgb_name(target_mode: str, regime: int) -> str:
    direct = target_mode == "direct"
    if regime <= 0:
        return "hgb_regime_low_direct" if direct else "hgb_regime_low_residual"
    if regime == 1:
        return "hgb_regime_partial_direct" if direct else "hgb_regime_partial_residual"
    if regime == 2:
        return "hgb_regime_rated_direct" if direct else "hgb_regime_rated_residual"
    return "hgb_regime_cutout_direct" if direct else "hgb_regime_cutout_residual"


def _regime_v2_soft_gate(pred_df: pd.DataFrame, regime: int) -> np.ndarray:
    if "wind_speed_eq" not in pred_df.columns:
        return np.ones(len(pred_df), dtype=float)
    v_eq = pd.to_numeric(pred_df["wind_speed_eq"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    cfg = ACTIVE_PHYSICS_CONFIG
    if regime <= 0:
        distances = np.abs(v_eq - cfg.v_cut_in)
    elif regime == 1:
        distances = np.minimum(np.abs(v_eq - cfg.v_cut_in), np.abs(v_eq - cfg.v_rated))
    elif regime == 2:
        distances = np.minimum(np.abs(v_eq - cfg.v_rated), np.abs(v_eq - cfg.v_cut_out))
    else:
        distances = np.abs(v_eq - cfg.v_cut_out)
    x = np.clip(distances / REGIME_V2_SOFT_BOUNDARY_WIDTH, 0.0, 1.0)
    smooth = x * x * (3.0 - 2.0 * x)
    return REGIME_V2_MIN_EXPERT_WEIGHT + (1.0 - REGIME_V2_MIN_EXPERT_WEIGHT) * smooth


def fit_predict_regime_hgb(
    spec_name: str,
    fit_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: list[str],
    y_fit: pd.Series,
    target_mode: str,
) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    x_fit_all = imputer.fit_transform(fit_df[feature_cols])
    x_pred_all = imputer.transform(pred_df[feature_cols])

    fallback_name = "hgb_direct_smooth" if target_mode == "direct" else "hgb_residual_smooth"
    fallback = hgb_model(fallback_name)
    fallback.fit(x_fit_all, y_fit)
    raw_pred = fallback.predict(x_pred_all)

    regime_col = "wind_regime_code"
    use_multi_regime = (
        ENABLE_MULTI_REGIME_EXPERTS
        and not ENABLE_REGIME_V2
        and "wind_multi_regime_code" in fit_df
        and "wind_multi_regime_code" in pred_df
    )
    if use_multi_regime:
        regime_col = "wind_multi_regime_code"
    if regime_col not in fit_df or regime_col not in pred_df:
        return total_from_model_output(raw_pred, pred_df, target_mode)

    min_regime_rows = 420 if use_multi_regime else 450
    expert_weight = 0.65 if use_multi_regime else 1.0
    for regime in sorted(pd.Series(fit_df[regime_col]).dropna().astype(int).unique()):
        fit_mask = fit_df[regime_col].astype(int).values == regime
        pred_mask = pred_df[regime_col].astype(int).values == regime
        if int(fit_mask.sum()) < min_regime_rows or int(pred_mask.sum()) == 0:
            continue
        params_name = (
            _regime_v2_hgb_name(target_mode, int(regime))
            if ENABLE_REGIME_V2
            else ("hgb_direct_deep" if target_mode == "direct" else "hgb_residual_smooth")
        )
        model = hgb_model(params_name)
        model.fit(x_fit_all[fit_mask], np.asarray(y_fit)[fit_mask])
        expert_pred = model.predict(x_pred_all[pred_mask])
        if ENABLE_REGIME_V2:
            gate = _regime_v2_soft_gate(pred_df.loc[pred_mask], int(regime))
            raw_pred[pred_mask] = raw_pred[pred_mask] * (1.0 - gate) + expert_pred * gate
        elif use_multi_regime:
            raw_pred[pred_mask] = raw_pred[pred_mask] * (1.0 - expert_weight) + expert_pred * expert_weight
        else:
            raw_pred[pred_mask] = expert_pred
    return total_from_model_output(raw_pred, pred_df, target_mode)


def target_for(train: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "direct":
        return train[TARGET]
    if mode == "residual":
        return train["residual"]
    if mode == "physics_v2_residual":
        return train[TARGET] - train["P_physics_farm_latent_v2"]
    if mode == "physics_v2_prior":
        return train["P_physics_farm_latent_v2"]
    if mode == "empirical_v3_residual":
        return train[TARGET] - train["P_empirical_curve_v3_prior"]
    if mode == "empirical_v3_prior":
        return train["P_empirical_curve_v3_prior"]
    if mode == "empirical_v4_residual":
        return train[TARGET] - train["P_empirical_curve_v4_prior"]
    if mode == "empirical_v4_prior":
        return train["P_empirical_curve_v4_prior"]
    if mode == "empirical_v4_guarded_residual":
        center = train.get("empirical_v4_guarded_residual_center", 0.0)
        return train[TARGET] - train["P_empirical_curve_v4_guarded_prior"] - center
    if mode == "empirical_v4_guarded_prior":
        return train["P_empirical_curve_v4_guarded_prior"]
    raise ValueError(f"Unknown target mode: {mode}")


def total_from_model_output(raw_pred, df: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "residual":
        raw_pred = df["P_physics_farm"].values + raw_pred
    elif mode == "physics_v2_residual":
        raw_pred = df["P_physics_farm_latent_v2"].values + raw_pred
    elif mode == "empirical_v3_residual":
        raw_pred = df["P_empirical_curve_v3_prior"].values + raw_pred
    elif mode == "empirical_v4_residual":
        raw_pred = df["P_empirical_curve_v4_prior"].values + raw_pred
    elif mode == "empirical_v4_guarded_residual":
        raw_pred = df["P_empirical_curve_v4_guarded_prior"].values + raw_pred
    return apply_submission_bounds(raw_pred, df)


def apply_empirical_v4_guarded_calibrators(
    base_fit_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    eval_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    v3_calibrator = fit_empirical_curve_calibrator(base_fit_df, pred_df, ACTIVE_PHYSICS_CONFIG)
    fit_df = apply_empirical_curve_features(base_fit_df, v3_calibrator, ACTIVE_PHYSICS_CONFIG)
    pred_df = apply_empirical_curve_features(pred_df, v3_calibrator, ACTIVE_PHYSICS_CONFIG)
    if eval_df is not None:
        eval_df = apply_empirical_curve_features(eval_df, v3_calibrator, ACTIVE_PHYSICS_CONFIG)

    v4_calibrator = fit_empirical_curve_v4_calibrator(
        base_fit_df,
        pred_df,
        ACTIVE_PHYSICS_CONFIG,
        ACTIVE_EMPIRICAL_CURVE_V4_CONFIG,
    )
    fit_df = apply_empirical_curve_v4_features(fit_df, v4_calibrator, ACTIVE_PHYSICS_CONFIG)
    pred_df = apply_empirical_curve_v4_features(pred_df, v4_calibrator, ACTIVE_PHYSICS_CONFIG)
    if eval_df is not None:
        eval_df = apply_empirical_curve_v4_features(eval_df, v4_calibrator, ACTIVE_PHYSICS_CONFIG)

    fit_df = add_empirical_curve_v4_guarded_features(fit_df, ACTIVE_EMPIRICAL_CURVE_V4_CONFIG)
    pred_df = add_empirical_curve_v4_guarded_features(pred_df, ACTIVE_EMPIRICAL_CURVE_V4_CONFIG)
    if eval_df is not None:
        eval_df = add_empirical_curve_v4_guarded_features(eval_df, ACTIVE_EMPIRICAL_CURVE_V4_CONFIG)

    residual = fit_df[TARGET] - fit_df["P_empirical_curve_v4_guarded_prior"]
    weights = prediction_similarity_weights(fit_df, pred_df)
    center = float(np.average(residual.values, weights=weights)) if float(weights.sum()) > 0.0 else 0.0
    center *= ACTIVE_EMPIRICAL_CURVE_V4_CONFIG.residual_center_strength
    center = float(np.clip(center, -ACTIVE_EMPIRICAL_CURVE_V4_CONFIG.guard_delta_clip, ACTIVE_EMPIRICAL_CURVE_V4_CONFIG.guard_delta_clip))
    fit_df["empirical_v4_guarded_residual_center"] = center
    pred_df["empirical_v4_guarded_residual_center"] = center
    if eval_df is not None:
        eval_df["empirical_v4_guarded_residual_center"] = center
    return fit_df, pred_df, eval_df


def fit_predict_spec(
    spec: ModelSpec,
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: list[str],
    train_idx: np.ndarray | None = None,
    eval_idx: np.ndarray | None = None,
    verbose: bool = False,
    override_iterations: int | None = None,
) -> tuple[np.ndarray, int | None]:
    if ACTIVE_MODEL_SET == "empirical_curve_v4_guarded" or spec.target_mode.startswith("empirical_v4_guarded"):
        base_fit_df = train_df.iloc[train_idx].copy() if train_idx is not None else train_df.copy()
        base_eval_df = train_df.iloc[eval_idx].copy() if eval_idx is not None else None
        fit_df, pred_df, eval_df = apply_empirical_v4_guarded_calibrators(base_fit_df, pred_df, base_eval_df)
    elif ACTIVE_MODEL_SET == "empirical_curve_v4" or spec.target_mode.startswith("empirical_v4"):
        base_fit_df = train_df.iloc[train_idx].copy() if train_idx is not None else train_df.copy()
        calibrator = fit_empirical_curve_v4_calibrator(
            base_fit_df,
            pred_df,
            ACTIVE_PHYSICS_CONFIG,
            ACTIVE_EMPIRICAL_CURVE_V4_CONFIG,
        )
        fit_df = apply_empirical_curve_v4_features(base_fit_df, calibrator, ACTIVE_PHYSICS_CONFIG)
        pred_df = apply_empirical_curve_v4_features(pred_df, calibrator, ACTIVE_PHYSICS_CONFIG)
        eval_df = None
        if eval_idx is not None:
            eval_df = train_df.iloc[eval_idx].copy()
            eval_df = apply_empirical_curve_v4_features(eval_df, calibrator, ACTIVE_PHYSICS_CONFIG)
    elif ACTIVE_MODEL_SET == "empirical_curve_v3" or spec.target_mode.startswith("empirical_v3"):
        base_fit_df = train_df.iloc[train_idx].copy() if train_idx is not None else train_df.copy()
        calibrator = fit_empirical_curve_calibrator(base_fit_df, pred_df, ACTIVE_PHYSICS_CONFIG)
        fit_df = apply_empirical_curve_features(base_fit_df, calibrator, ACTIVE_PHYSICS_CONFIG)
        pred_df = apply_empirical_curve_features(pred_df, calibrator, ACTIVE_PHYSICS_CONFIG)
        eval_df = None
        if eval_idx is not None:
            eval_df = train_df.iloc[eval_idx].copy()
            eval_df = apply_empirical_curve_features(eval_df, calibrator, ACTIVE_PHYSICS_CONFIG)
    elif ACTIVE_MODEL_SET == "physics_first_v2" or spec.target_mode.startswith("physics_v2"):
        base_fit_df = train_df.iloc[train_idx].copy() if train_idx is not None else train_df.copy()
        calibrator = fit_physics_first_v2_calibrator(base_fit_df, ACTIVE_PHYSICS_CONFIG)
        fit_df = apply_physics_first_v2_calibrator(base_fit_df, calibrator, ACTIVE_PHYSICS_CONFIG)
        pred_df = apply_physics_first_v2_calibrator(pred_df, calibrator, ACTIVE_PHYSICS_CONFIG)
        eval_df = None
        if eval_idx is not None:
            eval_df = train_df.iloc[eval_idx].copy()
            eval_df = apply_physics_first_v2_calibrator(eval_df, calibrator, ACTIVE_PHYSICS_CONFIG)
    else:
        fit_df, pred_df, eval_df = prepare_model_frames_with_latent_availability(
            train_df,
            pred_df,
            train_idx=train_idx,
            eval_idx=eval_idx,
        )
    if ENABLE_LAG_RESIDUAL_FEATURES:
        fit_df, pred_df, eval_df = apply_residual_prior_features(fit_df, pred_df, eval_df)
    if ENABLE_REGIME_PRIOR_V2:
        fit_df, pred_df, eval_df = apply_regime_prior_v2_features(fit_df, pred_df, eval_df)
    if ENABLE_WEATHER_ANALOG_RESIDUAL:
        fit_df, pred_df, eval_df = apply_weather_analog_residual_features(fit_df, pred_df, eval_df)
    if ENABLE_MULTI_REGIME_FEATURES:
        fit_df, pred_df, eval_df = apply_multi_regime_prior_features(fit_df, pred_df, eval_df)

    if spec.name == "physics_v2_prior":
        return apply_submission_bounds(pred_df["P_physics_farm_latent_v2"].values, pred_df), None
    if spec.name == "empirical_curve_v3_prior":
        return apply_submission_bounds(pred_df["P_empirical_curve_v3_prior"].values, pred_df), None
    if spec.name == "empirical_curve_v4_prior":
        return apply_submission_bounds(pred_df["P_empirical_curve_v4_prior"].values, pred_df), None
    if spec.name == "empirical_curve_v4_guarded_prior":
        return apply_submission_bounds(pred_df["P_empirical_curve_v4_guarded_prior"].values, pred_df), None

    y_fit = target_for(fit_df, spec.target_mode)

    best_iter: int | None = None
    eval_rows = 0 if eval_idx is None else len(eval_idx)
    print(
        f"      fitting {spec.name}: train_rows={len(fit_df)} eval_rows={eval_rows} "
        f"pred_rows={len(pred_df)} features={len(feature_cols)}",
        flush=True,
    )
    started_at = perf_counter()

    if spec.name.startswith("hgb_regime"):
        raw_total = fit_predict_regime_hgb(spec.name, fit_df, pred_df, feature_cols, y_fit, spec.target_mode)
        duration = format_duration(perf_counter() - started_at)
        print(f"      done {spec.name} in {duration}", flush=True)
        return raw_total, None
    if spec.name.startswith("cat"):
        model = CatBoostRegressor(**cat_params(spec.name, verbose=verbose, override_iterations=override_iterations))
        fit_pool = Pool(fit_df[feature_cols], y_fit, cat_features=CAT_FEATURES)
        eval_set = None
        if eval_df is not None:
            eval_y = target_for(eval_df, spec.target_mode)
            eval_set = Pool(eval_df[feature_cols], eval_y, cat_features=CAT_FEATURES)
        model.fit(fit_pool, eval_set=eval_set, use_best_model=eval_set is not None)
        raw_pred = model.predict(pred_df[feature_cols])
        if eval_set is not None:
            best_iter = int(model.get_best_iteration() or model.tree_count_)
    elif spec.name.startswith("hgb"):
        model = hgb_model(spec.name)
        model.fit(fit_df[feature_cols], y_fit)
        raw_pred = model.predict(pred_df[feature_cols])
    elif spec.name.startswith("lgb"):
        try:
            from lightgbm import LGBMRegressor, early_stopping, log_evaluation
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM model set selected, but the 'lightgbm' package is not installed. "
                "Install it with: pip install lightgbm"
            ) from exc

        model = LGBMRegressor(**lgb_params(spec.name, override_iterations=override_iterations))
        fit_kwargs = {"categorical_feature": [col for col in CAT_FEATURES if col in feature_cols]}
        if eval_df is not None:
            eval_y = target_for(eval_df, spec.target_mode)
            fit_kwargs.update(
                eval_set=[(eval_df[feature_cols], eval_y)],
                eval_metric="mae",
                callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
            )
        model.fit(fit_df[feature_cols], y_fit, **fit_kwargs)
        raw_pred = model.predict(pred_df[feature_cols])
        if eval_df is not None:
            best_iter = int(getattr(model, "best_iteration_", None) or model.n_estimators)
    elif spec.name.startswith("etr"):
        raw_pred = fit_predict_etr(
            spec.name,
            fit_df,
            pred_df,
            feature_cols,
            y_fit,
            override_iterations=override_iterations,
        )
    else:
        raise ValueError(f"Unknown model spec: {spec.name}")

    duration = format_duration(perf_counter() - started_at)
    best_iter_msg = f" best_iter={best_iter}" if best_iter is not None else ""
    print(f"      done {spec.name} in {duration}{best_iter_msg}", flush=True)
    return total_from_model_output(raw_pred, pred_df, spec.target_mode), best_iter


def drop_empty_feature_columns(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str],
    *,
    keep_empty_feature_columns: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if keep_empty_feature_columns:
        return train, valid, feature_cols
    empty_cols = [
        col for col in feature_cols
        if col in train.columns and not train[col].notna().any()
    ]
    if not empty_cols:
        return train, valid, feature_cols
    train = train.drop(columns=empty_cols)
    valid_drop_cols = [col for col in empty_cols if col in valid.columns]
    if valid_drop_cols:
        valid = valid.drop(columns=valid_drop_cols)
    empty_set = set(empty_cols)
    feature_cols = [col for col in feature_cols if col not in empty_set]
    print(
        "  dropped empty feature columns: " + ", ".join(empty_cols),
        flush=True,
    )
    return train, valid, feature_cols


def load_and_prepare(
    physics_cfg: PhysicsConfig = DEFAULT_PHYSICS,
    use_external_weather: bool = True,
    external_cache_dir: Path = EXTERNAL_WEATHER_DIR,
    refresh_external_weather: bool = False,
    include_nasa_power: bool = False,
    external_source_set: str = "baseline",
    external_source_timeout_sec: int | float = DEFAULT_EXTERNAL_SOURCE_TIMEOUT_SECONDS,
    external_cache_policy: str = "cache_first",
    require_extra_sources: bool = False,
    include_openmeteo_forecast: bool = True,
    openmeteo_forecast_set: str = "core_open_forecast",
    openmeteo_fill_policy: str = "legacy",
    openmeteo_min_coverage: float = OPENMETEO_COVERAGE_MIN_DEFAULT,
    include_openmeteo_pressure: bool = False,
    include_meteostat: bool = True,
    meteostat_cache_path: Path = METEOSTAT_CACHE_PATH,
    allow_partial_meteostat: bool = True,
    require_meteostat: bool = False,
    include_meteostat_derived: bool = False,
    include_so_ups_res: bool = False,
    so_ups_res_cache_path: Path = SO_UPS_RES_CACHE_PATH,
    include_so_ups_valid_months: bool = False,
    so_ups_res_policy: str = "legacy_best_gap",
    so_ups_res_exclude_months: str | tuple[str, ...] | list[str] | None = None,
    include_physics_variants: bool = False,
    keep_empty_feature_columns: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    train_raw = pd.read_csv(TRAIN_PATH, parse_dates=[DATETIME_COL])
    valid_raw = pd.read_csv(VALID_PATH, parse_dates=[DATETIME_COL])
    valid_original_order = valid_raw[DATETIME_COL].values.copy()

    train_raw = train_raw.sort_values(DATETIME_COL).reset_index(drop=True)
    valid_raw = valid_raw.sort_values(DATETIME_COL).reset_index(drop=True)

    n_train = len(train_raw)
    valid_no_target = valid_raw.copy()
    if TARGET in valid_no_target.columns:
        valid_no_target = valid_no_target.drop(columns=[TARGET])

    combined = pd.concat([train_raw, valid_no_target], ignore_index=True, sort=False)
    combined["wind_direction_10m"] = combined["wind_direction_10m"].fillna(combined["wind_direction_80m"])
    combined["wind_direction_180m"] = combined["wind_direction_180m"].fillna(combined["wind_direction_120m"])
    combined["wind_speed_180m"] = combined["wind_speed_180m"].fillna(combined["wind_speed_120m"])

    if use_external_weather:
        combined = add_external_weather_features(
            combined,
            cache_dir=external_cache_dir,
            refresh=refresh_external_weather,
            include_nasa_power=include_nasa_power,
            external_source_set=external_source_set,
            external_source_timeout_sec=external_source_timeout_sec,
            external_cache_policy=external_cache_policy,
            require_extra_sources=require_extra_sources,
            include_openmeteo_forecast=include_openmeteo_forecast,
            openmeteo_forecast_set=openmeteo_forecast_set,
            openmeteo_fill_policy=openmeteo_fill_policy,
            openmeteo_min_coverage=openmeteo_min_coverage,
            include_openmeteo_pressure=include_openmeteo_pressure,
            include_meteostat=include_meteostat,
            meteostat_cache_path=meteostat_cache_path,
            allow_partial_meteostat=allow_partial_meteostat,
            require_meteostat=require_meteostat,
            include_meteostat_derived=include_meteostat_derived,
        )
    if include_so_ups_res:
        combined = add_so_ups_res_features(
            combined,
            cache_path=so_ups_res_cache_path,
            include_valid_months=include_so_ups_valid_months,
            policy=so_ups_res_policy,
            exclude_months=(
                _parse_month_tokens(so_ups_res_exclude_months)
                if isinstance(so_ups_res_exclude_months, str)
                else so_ups_res_exclude_months
            ),
        )

    combined = add_physical_features(
        combined,
        physics_cfg,
        include_physics_variants=include_physics_variants,
    )
    combined = add_temporal_weather_features(combined)
    combined = combined.sort_values(DATETIME_COL).reset_index(drop=True)

    train_mask = combined[DATETIME_COL].isin(train_raw[DATETIME_COL])
    train = combined.loc[train_mask].copy()
    valid = combined.loc[~train_mask].copy()

    train[TARGET] = train_raw.set_index(DATETIME_COL).loc[
        train[DATETIME_COL].values, TARGET
    ].values
    train["residual"] = train[TARGET] - train["P_physics_farm"]
    if ENABLE_LAG_RESIDUAL_FEATURES:
        train = initialize_residual_prior_features(train)
        valid = initialize_residual_prior_features(valid)
    if ENABLE_REGIME_PRIOR_V2:
        train = initialize_regime_prior_v2_features(train)
        valid = initialize_regime_prior_v2_features(valid)
    if ENABLE_WEATHER_ANALOG_RESIDUAL:
        train = initialize_weather_analog_residual_features(train)
        valid = initialize_weather_analog_residual_features(valid)
    if ENABLE_MULTI_REGIME_FEATURES:
        train = initialize_multi_regime_prior_features(train)
        valid = initialize_multi_regime_prior_features(valid)
    if ENABLE_WINDFM_DIAGNOSTIC:
        train, valid = add_windfm_diagnostic_features(train, valid)

    for col in CAT_FEATURES:
        train[col] = train[col].astype(int)
        valid[col] = valid[col].astype(int)

    drop_cols = {TARGET, "residual", DATETIME_COL}
    feature_cols = [c for c in train.columns if c not in drop_cols]
    train, valid, feature_cols = drop_empty_feature_columns(
        train,
        valid,
        feature_cols,
        keep_empty_feature_columns=keep_empty_feature_columns,
    )
    missing = [c for c in feature_cols if c not in valid.columns]
    if missing:
        raise ValueError(f"Missing columns in valid data: {missing}")
    if require_extra_sources and resolve_external_source_set(external_source_set, include_nasa_power=include_nasa_power):
        extra_source_cols = [
            col for col in feature_cols
            if any(col.startswith(prefix) for prefix in BASELINE_EXTRA_SOURCE_FEATURES)
        ]
        if not extra_source_cols:
            raise RuntimeError(
                "--require-extra-sources was set, but no real optional source feature columns "
                "survived preprocessing. Refusing to run a baseline-equivalent training job."
            )
        print(f"  optional external source feature columns: {len(extra_source_cols)}", flush=True)

    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)

    assert len(train) == n_train, f"train length mismatch: {len(train)} vs {n_train}"
    assert len(valid) == len(valid_raw), f"valid length mismatch: {len(valid)} vs {len(valid_raw)}"

    return train, valid, valid_original_order, feature_cols


def q1_folds(train: pd.DataFrame) -> list[tuple[int, np.ndarray, np.ndarray, float]]:
    folds = []
    for year, weight in [(2023, 0.2), (2024, 0.3), (2025, 0.5)]:
        train_idx = np.where(train[DATETIME_COL] < pd.Timestamp(f"{year}-01-01"))[0]
        valid_idx = np.where(
            (train[DATETIME_COL] >= pd.Timestamp(f"{year}-01-01"))
            & (train[DATETIME_COL] <= pd.Timestamp(f"{year}-03-31 23:00:00"))
        )[0]
        folds.append((year, train_idx, valid_idx, weight))
    return folds


def run_q1_cv(
    train: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict[int | str, float], dict[str, list[int]], dict[str, float]]:
    print("Running Q1 backtest proxy...", flush=True)
    fold_scores: dict[int | str, float] = {}
    baseline_fold_scores: dict[int, float] = {}
    best_iters: dict[str, list[int]] = {spec.name: [] for spec in MODEL_SPECS}
    cv_started_at = perf_counter()
    folds = q1_folds(train)
    base_weights = default_blend_weights()
    fold_weights = {year: weight for year, _, _, weight in folds}
    historical_predictions: dict[int, dict[str, np.ndarray]] = {}
    historical_targets: dict[int, np.ndarray] = {}

    for fold_no, (year, train_idx, valid_idx, _) in enumerate(folds, start=1):
        fold_started_at = perf_counter()
        fold_df = train.iloc[valid_idx].reset_index(drop=True)
        fold_blend_weights = optimize_validator_blend_weights(
            historical_predictions,
            historical_targets,
            fold_weights,
            base_weights=base_weights,
        )
        print(
            f"  Fold {fold_no}/{len(folds)} Q1 {year}: "
            f"train_rows={len(train_idx)} valid_rows={len(valid_idx)}",
            flush=True,
        )
        if historical_predictions:
            calibration_score = score_blend_weights(
                fold_blend_weights,
                historical_predictions,
                historical_targets,
                fold_weights,
            )
            print(
                f"    validator blend weights from prior folds: "
                f"{format_blend_weights(fold_blend_weights)} "
                f"(calibration={calibration_score:.4f}%)",
                flush=True,
            )
        else:
            print(
                f"    validator blend weights: {format_blend_weights(fold_blend_weights)} "
                "(no prior folds for calibration)",
                flush=True,
            )
        fold_predictions: dict[str, np.ndarray] = {}
        for spec_no, spec in enumerate(MODEL_SPECS, start=1):
            print(
                f"    Model {spec_no}/{len(MODEL_SPECS)}: "
                f"{spec.name} ({spec.target_mode}), weight={fold_blend_weights[spec.name]:.3f}",
                flush=True,
            )
            pred, best_iter = fit_predict_spec(
                spec,
                train,
                fold_df,
                feature_cols,
                train_idx=train_idx,
                eval_idx=valid_idx,
            )
            fold_predictions[spec.name] = pred
            if best_iter is not None:
                best_iters[spec.name].append(best_iter)

        baseline_blended = blend_model_predictions(fold_predictions, base_weights)
        baseline_score = platform_error_percent(fold_df[TARGET].values, baseline_blended)
        baseline_fold_scores[year] = baseline_score
        blended = blend_model_predictions(fold_predictions, fold_blend_weights)
        score = platform_error_percent(fold_df[TARGET].values, blended)
        fold_scores[year] = score
        historical_predictions[year] = fold_predictions
        historical_targets[year] = fold_df[TARGET].values.copy()
        print(
            f"  Q1 {year}: {score:.4f}% platform error "
            f"(static={baseline_score:.4f}%, delta={baseline_score - score:+.4f}%) "
            f"({format_duration(perf_counter() - fold_started_at)})",
            flush=True,
        )

    weighted = sum(fold_scores[year] * weight for year, _, _, weight in folds)
    baseline_weighted = sum(baseline_fold_scores[year] * weight for year, _, _, weight in folds)
    fold_scores["weighted"] = weighted
    fold_scores["static_weighted"] = baseline_weighted
    print(
        f"  Weighted Q1 proxy: {weighted:.4f}% "
        f"(static={baseline_weighted:.4f}%, delta={baseline_weighted - weighted:+.4f}%) "
        f"({format_duration(perf_counter() - cv_started_at)} total)",
        flush=True,
    )

    print("  best_iter per spec (mean across folds):", flush=True)
    for name, iters in best_iters.items():
        if iters:
            print(f"    {name}: {iters} -> mean={np.mean(iters):.0f}", flush=True)
    final_blend_weights = optimize_validator_blend_weights(
        historical_predictions,
        historical_targets,
        fold_weights,
        base_weights=base_weights,
    )
    final_calibration_score = score_blend_weights(
        final_blend_weights,
        historical_predictions,
        historical_targets,
        fold_weights,
    )
    print(
        f"  final validator blend weights for full-train fit: "
        f"{format_blend_weights(final_blend_weights)} "
        f"(OOF calibration={final_calibration_score:.4f}%)",
        flush=True,
    )
    return fold_scores, best_iters, final_blend_weights


def derive_final_iterations(best_iters: dict[str, list[int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, iters in best_iters.items():
        if not iters:
            continue
        out[name] = max(int(round(float(np.mean(iters)) * BEST_ITER_MULTIPLIER)), 500)
    return out


@dataclass
class PostQ1PredictionFrame:
    frame: pd.DataFrame
    target: pd.Series
    original_order: np.ndarray
    blank_original_order: np.ndarray
    source_path: str


def _round_operational_datetimes(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.round("h")


def load_post_q1_prediction_frame(
    path: Path,
    feature_cols: list[str],
    physics_cfg: PhysicsConfig = DEFAULT_PHYSICS,
    include_so_ups_res: bool = False,
    so_ups_res_cache_path: Path = SO_UPS_RES_CACHE_PATH,
    include_so_ups_valid_months: bool = False,
    so_ups_res_policy: str = "legacy_best_gap",
    so_ups_res_exclude_months: str | tuple[str, ...] | list[str] | None = None,
    include_physics_variants: bool = False,
) -> PostQ1PredictionFrame:
    """Prepare the April-May operational dataset without changing Q1 train/valid frames."""
    raw = pd.read_csv(path)
    if DATETIME_COL not in raw.columns:
        raise ValueError(f"{path} is missing {DATETIME_COL!r}")
    raw = raw.copy()
    raw[DATETIME_COL] = _round_operational_datetimes(raw[DATETIME_COL])
    raw = raw.dropna(subset=[DATETIME_COL])
    raw["month"] = raw[DATETIME_COL].dt.month.astype(int)
    raw["hour_of_day"] = raw[DATETIME_COL].dt.hour.astype(int)
    target = (
        pd.to_numeric(raw[TARGET], errors="coerce")
        if TARGET in raw.columns
        else pd.Series(np.nan, index=raw.index, dtype=float)
    )
    original_order = raw[DATETIME_COL].values.copy()
    blank_original_order = raw.loc[target.isna(), DATETIME_COL].values.copy()

    train_raw = pd.read_csv(TRAIN_PATH, parse_dates=[DATETIME_COL]).sort_values(DATETIME_COL).reset_index(drop=True)
    pred_no_target = raw.drop(columns=[TARGET], errors="ignore")
    combined = pd.concat([train_raw, pred_no_target], ignore_index=True, sort=False)
    combined["wind_direction_10m"] = combined["wind_direction_10m"].fillna(combined["wind_direction_80m"])
    combined["wind_direction_180m"] = combined["wind_direction_180m"].fillna(combined["wind_direction_120m"])
    combined["wind_speed_180m"] = combined["wind_speed_180m"].fillna(combined["wind_speed_120m"])

    if include_so_ups_res:
        combined = add_so_ups_res_features(
            combined,
            cache_path=so_ups_res_cache_path,
            include_valid_months=include_so_ups_valid_months,
            policy=so_ups_res_policy,
            exclude_months=(
                _parse_month_tokens(so_ups_res_exclude_months)
                if isinstance(so_ups_res_exclude_months, str)
                else so_ups_res_exclude_months
            ),
        )

    combined = add_physical_features(
        combined,
        physics_cfg,
        include_physics_variants=include_physics_variants,
    )
    combined = add_temporal_weather_features(combined)
    combined = combined.sort_values(DATETIME_COL).reset_index(drop=True)
    train_mask = combined[DATETIME_COL].isin(train_raw[DATETIME_COL])
    pred = combined.loc[~train_mask].copy()
    target_by_dt = pd.Series(target.values, index=raw[DATETIME_COL].values)
    pred[TARGET] = target_by_dt.loc[pred[DATETIME_COL].values].values
    pred["residual"] = pred[TARGET] - pred["P_physics_farm"]

    if ENABLE_LAG_RESIDUAL_FEATURES:
        pred = initialize_residual_prior_features(pred)
    if ENABLE_REGIME_PRIOR_V2:
        pred = initialize_regime_prior_v2_features(pred)
    if ENABLE_WEATHER_ANALOG_RESIDUAL:
        pred = initialize_weather_analog_residual_features(pred)
    if ENABLE_MULTI_REGIME_FEATURES:
        pred = initialize_multi_regime_prior_features(pred)

    for col in CAT_FEATURES:
        if col in pred.columns:
            pred[col] = pred[col].fillna(0).astype(int)
    missing_feature_cols = [col for col in feature_cols if col not in pred.columns]
    if missing_feature_cols:
        pred = pd.concat(
            [
                pred,
                pd.DataFrame(np.nan, index=pred.index, columns=missing_feature_cols),
            ],
            axis=1,
        )
    pred = pred.copy().reset_index(drop=True)
    aligned_target = pd.to_numeric(pred[TARGET], errors="coerce")
    return PostQ1PredictionFrame(
        frame=pred,
        target=aligned_target,
        original_order=original_order,
        blank_original_order=blank_original_order,
        source_path=str(path),
    )


def _adapter_group_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "wind_regime_code" not in out.columns:
        out = _ensure_residual_prior_group_columns(out)
    speed = pd.to_numeric(out.get("wind_speed_eq", out.get("wind_speed_80m", 0.0)), errors="coerce")
    out["adapter_speed_bin"] = (
        pd.cut(speed, bins=[-np.inf, 3.0, 5.5, 8.0, 11.0, np.inf], labels=False)
        .astype("float")
        .fillna(0)
        .astype(int)
    )
    direction = pd.to_numeric(out.get("wind_direction_80m", 0.0), errors="coerce").fillna(0.0) % 1.0
    out["adapter_direction_sector"] = np.floor(direction * 8.0).clip(0, 7).astype(int)
    return out


def _fit_affine_rules(pred: np.ndarray, target: np.ndarray, df: pd.DataFrame) -> dict:
    grouped = _adapter_group_frame(df)
    regimes = pd.to_numeric(grouped["wind_regime_code"], errors="coerce").fillna(1).astype(int).to_numpy()
    residual = target - pred
    global_b = float(np.clip(np.median(residual), -8.0, 8.0))
    rules: dict[int, dict[str, float]] = {}
    for regime in sorted(np.unique(regimes)):
        mask = regimes == regime
        if int(mask.sum()) < 80:
            rules[int(regime)] = {"a": 1.0, "b": global_b, "rows": int(mask.sum()), "fallback": 1.0}
            continue
        x = pred[mask]
        y = target[mask]
        if float(np.std(x)) < 1e-6:
            a, b = 1.0, float(np.median(y - x))
        else:
            a, b = np.polyfit(x, y, deg=1)
        rules[int(regime)] = {
            "a": float(np.clip(a, 0.85, 1.15)),
            "b": float(np.clip(b, -8.0, 8.0)),
            "rows": int(mask.sum()),
            "fallback": 0.0,
        }
    return {"kind": "affine_by_regime", "rules": rules, "global_b": global_b}


def _predict_affine_rules(adapter: dict, pred: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    grouped = _adapter_group_frame(df)
    regimes = pd.to_numeric(grouped["wind_regime_code"], errors="coerce").fillna(1).astype(int).to_numpy()
    correction = np.full(len(pred), float(adapter.get("global_b", 0.0)), dtype=float)
    for regime, rule in adapter.get("rules", {}).items():
        mask = regimes == int(regime)
        correction[mask] = float(rule.get("a", 1.0)) * pred[mask] + float(rule.get("b", 0.0)) - pred[mask]
    return correction


def _fit_weather_analog_rules(pred: np.ndarray, target: np.ndarray, df: pd.DataFrame) -> dict:
    grouped = _adapter_group_frame(df)
    grouped["_adapter_residual"] = target - pred
    group_cols = ["month", "hour_of_day", "wind_regime_code", "adapter_speed_bin", "adapter_direction_sector"]
    global_resid = float(np.median(grouped["_adapter_residual"]))
    med = grouped.groupby(group_cols)["_adapter_residual"].agg(["median", "count"])
    mapping = {
        tuple(key if isinstance(key, tuple) else (key,)): {
            "median": float(row["median"]),
            "count": int(row["count"]),
        }
        for key, row in med.iterrows()
    }
    return {
        "kind": "weather_analog",
        "group_cols": group_cols,
        "mapping": mapping,
        "global_residual": global_resid,
    }


def _predict_weather_analog_rules(adapter: dict, df: pd.DataFrame) -> np.ndarray:
    grouped = _adapter_group_frame(df)
    group_cols = adapter.get("group_cols", [])
    mapping = adapter.get("mapping", {})
    default = float(adapter.get("global_residual", 0.0))
    corrections = []
    for key in zip(*(grouped[col].values for col in group_cols)):
        payload = mapping.get(tuple(key))
        if payload is None:
            corrections.append(default)
        else:
            count = float(payload.get("count", 0))
            conf = count / (count + 80.0)
            corrections.append(conf * float(payload.get("median", default)) + (1.0 - conf) * default)
    return np.asarray(corrections, dtype=float)


def _adapter_feature_columns(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    preferred = [
        "P_physics_farm",
        "P_physics_per_turbine",
        "wind_speed_eq",
        "wind_speed_80m",
        "wind_speed_120m",
        "wind_gusts_10m",
        "wind_regime_code",
        "air_density",
        "temperature_80m",
        "pressure_msl",
        "hour_of_day",
        "month",
        "direction_shear_120_80",
        "wd_80m_sin",
        "wd_80m_cos",
        "availability_fraction",
    ]
    out = [col for col in preferred if col in df.columns]
    out.extend(
        col for col in feature_cols
        if col.startswith(("ext_nasa_", "ext_so_ups_res_", "ext_omfc_gfs_")) and col in df.columns
    )
    seen = set()
    return [col for col in out if not (col in seen or seen.add(col))]


def _fit_hgb_adapter(pred: np.ndarray, target: np.ndarray, df: pd.DataFrame, feature_cols: list[str]) -> dict:
    adapter_cols = _adapter_feature_columns(df, feature_cols)
    adapter_cols = [col for col in adapter_cols if pd.to_numeric(df[col], errors="coerce").notna().any()]
    if not adapter_cols:
        return {"kind": "identity", "reason": "no adapter features"}
    x = df[adapter_cols].copy()
    x["_base_prediction"] = pred
    imputer = SimpleImputer(strategy="median")
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=160,
        learning_rate=0.035,
        max_leaf_nodes=8,
        min_samples_leaf=35,
        l2_regularization=1.0,
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.20,
    )
    model.fit(imputer.fit_transform(x), target - pred)
    return {"kind": "hgb_residual_adapter", "columns": adapter_cols, "imputer": imputer, "model": model}


def _predict_hgb_adapter(adapter: dict, pred: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    columns = adapter.get("columns", [])
    x = df[columns].copy()
    x["_base_prediction"] = pred
    return adapter["model"].predict(adapter["imputer"].transform(x))


def _adapter_raw_correction(adapter: dict, pred: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    kind = adapter.get("kind")
    if kind == "affine_by_regime":
        return _predict_affine_rules(adapter, pred, df)
    if kind == "weather_analog":
        return _predict_weather_analog_rules(adapter, df)
    if kind == "hgb_residual_adapter":
        return _predict_hgb_adapter(adapter, pred, df)
    return np.zeros(len(pred), dtype=float)


def _serializable_adapter(adapter: dict) -> dict:
    out = {key: value for key, value in adapter.items() if key not in {"model", "imputer"}}
    if "mapping" in out:
        out["mapping_sample"] = {
            "|".join(map(str, key)): value
            for key, value in list(out["mapping"].items())[:25]
        }
        out["mapping_size"] = len(out["mapping"])
        out.pop("mapping", None)
    return out


def fit_post_q1_actual_adapter(
    post_df: pd.DataFrame,
    base_pred: np.ndarray,
    feature_cols: list[str],
    correction_cap: float = 8.0,
    allowed_kinds: tuple[str, ...] | None = None,
    shrink_values: tuple[float, ...] = (0.25, 0.40, 0.55),
    min_improvement_mw: float = 0.0,
) -> tuple[dict, dict]:
    target = pd.to_numeric(post_df[TARGET], errors="coerce").to_numpy(dtype=float)
    known_mask = np.isfinite(target)
    n_known = int(known_mask.sum())
    if n_known < 240:
        adapter = {"kind": "identity", "shrink": 0.0, "correction_cap": float(correction_cap)}
        return adapter, {"enabled": False, "reason": f"not enough known post-Q1 rows: {n_known}"}

    known_df = post_df.loc[known_mask].reset_index(drop=True)
    known_pred = np.asarray(base_pred, dtype=float)[known_mask]
    known_target = target[known_mask]
    valid_n = min(max(168, n_known // 5), n_known - 120)
    fit_n = n_known - valid_n
    fit_df, val_df = known_df.iloc[:fit_n].reset_index(drop=True), known_df.iloc[fit_n:].reset_index(drop=True)
    fit_pred, val_pred = known_pred[:fit_n], known_pred[fit_n:]
    fit_target, val_target = known_target[:fit_n], known_target[fit_n:]
    base_mae = float(np.mean(np.abs(val_target - val_pred)))

    candidates = []
    for candidate in [
        _fit_affine_rules(fit_pred, fit_target, fit_df),
        _fit_weather_analog_rules(fit_pred, fit_target, fit_df),
        _fit_hgb_adapter(fit_pred, fit_target, fit_df, feature_cols),
    ]:
        if candidate.get("kind") == "identity":
            continue
        if allowed_kinds is not None and candidate.get("kind") not in set(allowed_kinds):
            continue
        raw_corr = np.clip(_adapter_raw_correction(candidate, val_pred, val_df), -correction_cap, correction_cap)
        for shrink in shrink_values:
            correction = shrink * raw_corr
            mae = float(np.mean(np.abs(val_target - apply_submission_bounds(val_pred + correction, val_df))))
            candidates.append(
                {
                    "kind": candidate["kind"],
                    "shrink": shrink,
                    "mae": mae,
                    "mean_correction": float(np.mean(correction)),
                    "max_abs_correction": float(np.max(np.abs(correction))) if len(correction) else 0.0,
                    "adapter": candidate,
                }
            )
    eligible = [
        item for item in candidates
        if (base_mae - item["mae"]) > min_improvement_mw and abs(item["mean_correction"]) <= 4.0
    ]
    if not eligible:
        adapter = {"kind": "identity", "shrink": 0.0, "correction_cap": float(correction_cap)}
        return adapter, {
            "enabled": False,
            "known_rows": n_known,
            "fit_rows": int(fit_n),
            "validation_rows": int(valid_n),
            "base_validation_mae_mw": base_mae,
            "allowed_kinds": list(allowed_kinds) if allowed_kinds is not None else None,
            "shrink_values": list(shrink_values),
            "min_improvement_mw": float(min_improvement_mw),
            "candidate_scores": [
                {key: value for key, value in item.items() if key != "adapter"}
                for item in candidates
            ],
            "reason": "no adapter improved blocked April-May validation within correction constraints",
        }

    best = min(eligible, key=lambda item: item["mae"])
    if best["kind"] == "affine_by_regime":
        final_adapter = _fit_affine_rules(known_pred, known_target, known_df)
    elif best["kind"] == "weather_analog":
        final_adapter = _fit_weather_analog_rules(known_pred, known_target, known_df)
    else:
        final_adapter = _fit_hgb_adapter(known_pred, known_target, known_df, feature_cols)
    final_adapter["shrink"] = float(best["shrink"])
    final_adapter["correction_cap"] = float(correction_cap)
    report = {
        "enabled": True,
        "known_rows": n_known,
        "fit_rows": int(fit_n),
        "validation_rows": int(valid_n),
        "base_validation_mae_mw": base_mae,
        "allowed_kinds": list(allowed_kinds) if allowed_kinds is not None else None,
        "shrink_values": list(shrink_values),
        "min_improvement_mw": float(min_improvement_mw),
        "selected_kind": best["kind"],
        "selected_shrink": float(best["shrink"]),
        "selected_validation_mae_mw": float(best["mae"]),
        "validation_improvement_mw": float(base_mae - best["mae"]),
        "candidate_scores": [
            {key: value for key, value in item.items() if key != "adapter"}
            for item in candidates
        ],
        "final_adapter": _serializable_adapter(final_adapter),
    }
    return final_adapter, report


def apply_post_q1_actual_adapter(
    pred: np.ndarray,
    df: pd.DataFrame,
    adapter: dict,
) -> tuple[np.ndarray, dict]:
    pred = np.asarray(pred, dtype=float)
    shrink = float(adapter.get("shrink", 0.0))
    cap = float(adapter.get("correction_cap", 8.0))
    raw_corr = np.clip(_adapter_raw_correction(adapter, pred, df), -cap, cap)
    correction = shrink * raw_corr
    adjusted = apply_submission_bounds(pred + correction, df)
    return adjusted, {
        "kind": adapter.get("kind", "identity"),
        "shrink": shrink,
        "correction_cap": cap,
        "mean_correction_mw": float(np.mean(adjusted - pred)) if len(pred) else 0.0,
        "max_abs_correction_mw": float(np.max(np.abs(adjusted - pred))) if len(pred) else 0.0,
    }


def _fit_scalar_transfer_stats(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    residual = target - pred
    finite = np.isfinite(pred) & np.isfinite(target)
    if not finite.any():
        return {"bias_mw": 0.0, "scale_delta": 0.0, "sum_scale_delta": 0.0}
    bias = float(np.nanmedian(residual[finite]))
    ratio_mask = finite & (pred >= 5.0)
    if int(ratio_mask.sum()) >= 120:
        ratios = np.clip(target[ratio_mask] / np.maximum(pred[ratio_mask], 1e-6), 0.25, 1.75)
        scale_delta = float(np.nanmedian(ratios) - 1.0)
        sum_scale_delta = float(np.sum(target[ratio_mask]) / max(float(np.sum(pred[ratio_mask])), 1e-6) - 1.0)
    else:
        scale_delta = 0.0
        sum_scale_delta = 0.0
    return {
        "bias_mw": float(np.clip(bias, -1.5, 1.5)),
        "scale_delta": float(np.clip(scale_delta, -0.05, 0.05)),
        "sum_scale_delta": float(np.clip(sum_scale_delta, -0.05, 0.05)),
    }


def _scalar_transfer_correction(pred: np.ndarray, stats: dict[str, float], kind: str, shrink: float, cap: float) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    shrink = float(shrink)
    if kind == "bias":
        raw = np.full(len(pred), float(stats.get("bias_mw", 0.0)), dtype=float)
    elif kind == "scale_median":
        raw = pred * float(stats.get("scale_delta", 0.0))
    elif kind == "scale_sum":
        raw = pred * float(stats.get("sum_scale_delta", 0.0))
    elif kind == "bias_plus_scale":
        raw = np.full(len(pred), 0.5 * float(stats.get("bias_mw", 0.0)), dtype=float)
        raw += pred * 0.5 * float(stats.get("scale_delta", 0.0))
    else:
        raw = np.zeros(len(pred), dtype=float)
    return np.clip(shrink * raw, -cap, cap)


def apply_q1_scalar_actual_transfer(
    q1_pred: np.ndarray,
    q1_df: pd.DataFrame,
    post_df: pd.DataFrame,
    post_base_pred: np.ndarray,
    correction_cap: float = 0.75,
    max_abs_mean_delta_mw: float = 0.35,
    max_zero_increase: int = 3,
    min_improvement_mw: float = 0.03,
) -> tuple[np.ndarray, dict]:
    """Transfer only a tiny global 2026 bias/scale signal from April-May to Q1.

    This intentionally avoids weather-local residual models: April-May rows are
    out-of-quarter and can validate well while creating unrealistic Q1 zeros.
    """
    q1_pred = np.asarray(q1_pred, dtype=float)
    post_base_pred = np.asarray(post_base_pred, dtype=float)
    target = pd.to_numeric(post_df[TARGET], errors="coerce").to_numpy(dtype=float)
    known_mask = np.isfinite(target)
    n_known = int(known_mask.sum())
    if n_known < 240:
        return q1_pred, {
            "enabled": False,
            "policy": "scalar",
            "reason": f"not enough known post-Q1 rows: {n_known}",
        }

    known_pred = post_base_pred[known_mask]
    known_target = target[known_mask]
    valid_n = min(max(168, n_known // 5), n_known - 120)
    fit_n = n_known - valid_n
    fit_pred, val_pred = known_pred[:fit_n], known_pred[fit_n:]
    fit_target, val_target = known_target[:fit_n], known_target[fit_n:]
    base_mae = float(np.mean(np.abs(val_target - val_pred)))
    fit_stats = _fit_scalar_transfer_stats(fit_pred, fit_target)

    candidates = []
    for kind in ("bias", "scale_median", "scale_sum", "bias_plus_scale"):
        for shrink in (0.10, 0.20, 0.30):
            corr = _scalar_transfer_correction(val_pred, fit_stats, kind, shrink, correction_cap)
            adjusted = apply_submission_bounds(val_pred + corr, post_df.loc[known_mask].iloc[fit_n:].reset_index(drop=True))
            mae = float(np.mean(np.abs(val_target - adjusted)))
            candidates.append(
                {
                    "kind": kind,
                    "shrink": float(shrink),
                    "mae": mae,
                    "improvement_mw": float(base_mae - mae),
                    "mean_correction_mw": float(np.mean(corr)) if len(corr) else 0.0,
                    "max_abs_correction_mw": float(np.max(np.abs(corr))) if len(corr) else 0.0,
                }
            )

    eligible = [
        item
        for item in candidates
        if item["improvement_mw"] >= min_improvement_mw
        and abs(item["mean_correction_mw"]) <= 0.75
        and item["max_abs_correction_mw"] <= correction_cap + 1e-9
    ]
    report = {
        "enabled": False,
        "policy": "scalar",
        "known_rows": n_known,
        "fit_rows": int(fit_n),
        "validation_rows": int(valid_n),
        "base_validation_mae_mw": base_mae,
        "fit_stats": fit_stats,
        "correction_cap_mw": float(correction_cap),
        "max_abs_mean_delta_mw": float(max_abs_mean_delta_mw),
        "max_zero_increase": int(max_zero_increase),
        "min_improvement_mw": float(min_improvement_mw),
        "candidate_scores": candidates,
    }
    if not eligible:
        report["reason"] = "no tiny scalar correction improved blocked April-May validation enough"
        return q1_pred, report

    best = min(eligible, key=lambda item: item["mae"])
    final_stats = _fit_scalar_transfer_stats(known_pred, known_target)
    correction = _scalar_transfer_correction(q1_pred, final_stats, best["kind"], best["shrink"], correction_cap)
    adjusted = apply_submission_bounds(q1_pred + correction, q1_df)
    applied = adjusted - q1_pred
    before_zeros = int((q1_pred <= 1e-9).sum())
    after_zeros = int((adjusted <= 1e-9).sum())
    application = {
        "kind": best["kind"],
        "shrink": float(best["shrink"]),
        "mean_correction_mw": float(np.mean(applied)) if len(applied) else 0.0,
        "max_abs_correction_mw": float(np.max(np.abs(applied))) if len(applied) else 0.0,
        "before_zeros": before_zeros,
        "after_zeros": after_zeros,
        "delta_gwh": float(np.sum(applied) / 1000.0),
    }
    guard_failures = []
    if abs(application["mean_correction_mw"]) > max_abs_mean_delta_mw:
        guard_failures.append("mean correction too large")
    if application["max_abs_correction_mw"] > correction_cap + 1e-9:
        guard_failures.append("hourly correction too large")
    if after_zeros > before_zeros + max_zero_increase:
        guard_failures.append("zero count increase too large")
    if guard_failures:
        report.update(
            {
                "reason": "; ".join(guard_failures),
                "selected_kind": best["kind"],
                "selected_shrink": float(best["shrink"]),
                "selected_validation_mae_mw": float(best["mae"]),
                "validation_improvement_mw": float(best["improvement_mw"]),
                "final_stats": final_stats,
                "application": application,
            }
        )
        return q1_pred, report

    report.update(
        {
            "enabled": True,
            "selected_kind": best["kind"],
            "selected_shrink": float(best["shrink"]),
            "selected_validation_mae_mw": float(best["mae"]),
            "validation_improvement_mw": float(best["improvement_mw"]),
            "final_stats": final_stats,
            "application": application,
        }
    )
    return adjusted, report


EL5_WIND_OUTPUT_Q1_GWH = {2023: 228.0, 2024: 226.0, 2025: 121.0, 2026: 215.0}
EL5_Q1_SOURCE_URLS = [
    "https://www.marketscreener.com/news/enel-russia-el5-energo-publishes-its-ifrs-based-financial-results-for-1q-2026-ce7f58dad880fe25",
    "https://www.marketscreener.com/quote/stock/ENEL-RUSSIA-6498987/news/Enel-Russia-EL5-Energo-publishes-its-IFRS-based-financial-results-for-1Q-2025-49757173/",
]


def _q1_mask(df: pd.DataFrame, year: int | None = None) -> pd.Series:
    dt = pd.to_datetime(df[DATETIME_COL])
    mask = dt.dt.month.isin([1, 2, 3])
    if year is not None:
        mask &= dt.dt.year == year
    return mask


def _redistribute_energy_delta(pred: np.ndarray, df: pd.DataFrame, target_mwh: float) -> tuple[np.ndarray, dict]:
    out = np.asarray(pred, dtype=float).copy()
    cfg = ACTIVE_PHYSICS_CONFIG
    v_eq = pd.to_numeric(df.get("wind_speed_eq", df.get("wind_speed_80m", 0.0)), errors="coerce").fillna(0.0).to_numpy()
    ramp_up = np.clip((v_eq - cfg.v_cut_in) / max(cfg.v_rated - cfg.v_cut_in, 1e-6), 0.0, 1.0)
    ramp_down = 1.0 - np.clip((v_eq - cfg.v_rated) / max(cfg.v_cut_out - cfg.v_rated, 1e-6), 0.0, 1.0)
    partial_weight = 0.15 + 0.85 * ramp_up * ramp_down
    before = float(np.sum(out))
    for _ in range(6):
        delta = float(target_mwh - np.sum(out))
        if abs(delta) < 1e-6:
            break
        capacity_weight = np.clip(P_RATED_FARM - out, 0.0, None) if delta > 0 else np.clip(out, 0.0, None)
        weights = capacity_weight * partial_weight
        if float(weights.sum()) <= 1e-9:
            break
        out = np.clip(out + delta * weights / float(weights.sum()), 0.0, P_RATED_FARM)
    correction = out - pred
    return out, {
        "before_gwh": before / 1000.0,
        "after_gwh": float(np.sum(out)) / 1000.0,
        "target_gwh": float(target_mwh) / 1000.0,
        "mean_correction_mw": float(np.mean(correction)),
        "max_abs_correction_mw": float(np.max(np.abs(correction))) if len(correction) else 0.0,
        "remaining_delta_mwh": float(target_mwh - np.sum(out)),
    }


def apply_q1_energy_anchor(
    pred: np.ndarray,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    source: str = "azov_weather_proxy",
) -> tuple[np.ndarray, dict]:
    if source != "azov_weather_proxy":
        raise ValueError(f"Unknown q1 energy anchor source {source!r}")
    historical = {}
    ratios = []
    for year in (2023, 2024, 2025):
        mask = _q1_mask(train, year)
        actual_mwh = float(pd.to_numeric(train.loc[mask, TARGET], errors="coerce").sum())
        proxy_mwh = float(pd.to_numeric(train.loc[mask, "P_physics_farm"], errors="coerce").sum())
        ratio = actual_mwh / proxy_mwh if proxy_mwh > 0 else np.nan
        historical[str(year)] = {
            "rows": int(mask.sum()),
            "azov_actual_gwh": actual_mwh / 1000.0,
            "azov_proxy_gwh": proxy_mwh / 1000.0,
            "actual_to_proxy_ratio": float(ratio),
            "el5_total_wind_gwh": EL5_WIND_OUTPUT_Q1_GWH.get(year),
            "implied_non_azov_wind_gwh": EL5_WIND_OUTPUT_Q1_GWH.get(year, np.nan) - actual_mwh / 1000.0,
        }
        if np.isfinite(ratio):
            ratios.append(float(ratio))
    valid_proxy_mwh = float(pd.to_numeric(valid.loc[_q1_mask(valid), "P_physics_farm"], errors="coerce").sum())
    median_ratio = float(np.median(ratios))
    target_mwh = valid_proxy_mwh * median_ratio
    adjusted, correction = _redistribute_energy_delta(pred, valid, target_mwh)
    report = {
        "enabled": True,
        "source": source,
        "method": "median historical Azov actual/P_physics_farm ratio applied to 2026 Q1 proxy",
        "source_urls": EL5_Q1_SOURCE_URLS,
        "el5_total_wind_output_q1_gwh": {str(k): v for k, v in EL5_WIND_OUTPUT_Q1_GWH.items()},
        "historical": historical,
        "ratios_used": ratios,
        "median_ratio": median_ratio,
        "valid_proxy_gwh": valid_proxy_mwh / 1000.0,
        "correction": correction,
    }
    return adjusted, report


def write_prediction_values_by_datetime(
    pred_asc: np.ndarray,
    frame: pd.DataFrame,
    original_order: np.ndarray,
    path: Path,
) -> None:
    asc_to_value = dict(zip(frame[DATETIME_COL].values, pred_asc))
    output_values = np.array([asc_to_value[ts] for ts in original_order], dtype=float)
    pd.DataFrame({TARGET: output_values}).to_csv(path, index=False)


def validate_prediction_file(path: Path, expected_rows: int | None = None) -> dict[str, float | int | str]:
    pred = pd.read_csv(path)
    if pred.shape[1] != 1:
        raise ValueError(f"Predictions must have one column, got {pred.shape[1]}")
    if list(pred.columns) != [TARGET]:
        raise ValueError(f"Bad prediction column: {list(pred.columns)}")
    if expected_rows is not None and len(pred) != expected_rows:
        raise ValueError(f"Bad row count: predictions={len(pred)} expected={expected_rows}")
    values = pd.to_numeric(pred.iloc[:, 0], errors="raise")
    if values.isna().any():
        raise ValueError("Predictions contain NaN values")
    if (values < 0).any() or (values > P_RATED_FARM).any():
        raise ValueError("Predictions are outside physical bounds")
    return {
        "rows": int(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "zeros": int((values == 0).sum()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def train_final_predict(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str],
    verbose: bool,
    final_iterations: dict[str, int] | None = None,
    blend_weights: dict[str, float] | None = None,
    extra_pred_frames: dict[str, pd.DataFrame] | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    print("Training final temporal ensemble on all train rows...", flush=True)
    final_started_at = perf_counter()
    blend_weights = normalize_blend_weights(blend_weights or default_blend_weights())
    print(f"  using blend weights: {format_blend_weights(blend_weights)}", flush=True)
    extra_pred_frames = extra_pred_frames or {}
    if extra_pred_frames:
        train_feature_medians = {
            col: float(pd.to_numeric(train[col], errors="coerce").median())
            for col in feature_cols
            if col in train.columns and pd.to_numeric(train[col], errors="coerce").notna().any()
        }
        prepared_extra_frames = {}
        for name, frame in extra_pred_frames.items():
            prepared = frame.copy()
            for col in feature_cols:
                if col not in prepared.columns:
                    prepared[col] = train_feature_medians.get(col, 0.0)
                elif not pd.to_numeric(prepared[col], errors="coerce").notna().any():
                    prepared[col] = train_feature_medians.get(col, 0.0)
            prepared_extra_frames[name] = prepared.copy()
        extra_pred_frames = prepared_extra_frames
    split_lengths = {"__main__": len(valid)}
    pred_frames = [valid]
    for name, frame in extra_pred_frames.items():
        split_lengths[name] = len(frame)
        pred_frames.append(frame)
    combined_pred = (
        pd.concat(pred_frames, ignore_index=True, sort=False)
        if extra_pred_frames
        else valid
    )
    final_predictions: dict[str, np.ndarray] = {}
    for spec_no, spec in enumerate(MODEL_SPECS, start=1):
        override = (final_iterations or {}).get(spec.name)
        suffix = f" (iters={override})" if override else " (default iters)"
        print(
            f"  Final model {spec_no}/{len(MODEL_SPECS)}: "
            f"{spec.name} ({spec.target_mode}), weight={blend_weights[spec.name]:.3f}{suffix}",
            flush=True,
        )
        pred, _ = fit_predict_spec(
            spec,
            train,
            combined_pred,
            feature_cols,
            verbose=verbose,
            override_iterations=override,
        )
        final_predictions[spec.name] = pred
    print(
        f"Finished final ensemble in {format_duration(perf_counter() - final_started_at)}",
        flush=True,
    )
    blended = blend_model_predictions(final_predictions, blend_weights)
    if not extra_pred_frames:
        return blended
    offset = split_lengths["__main__"]
    main_pred = blended[:offset]
    extra_predictions = {}
    for name in extra_pred_frames:
        length = split_lengths[name]
        extra_predictions[name] = blended[offset:offset + length]
        offset += length
    return main_pred, extra_predictions


def write_predictions(pred_asc: np.ndarray, valid: pd.DataFrame, valid_original_order, path: Path) -> None:
    asc_to_value = dict(zip(valid[DATETIME_COL].values, pred_asc))
    output_values = np.array([asc_to_value[ts] for ts in valid_original_order], dtype=float)
    pd.DataFrame({TARGET: output_values}).to_csv(path, index=False)


def validate_predictions(path: Path, valid_path: Path) -> dict[str, float | int | str]:
    pred = pd.read_csv(path)
    valid = pd.read_csv(valid_path)
    if pred.shape[1] != 1:
        raise ValueError(f"Predictions must have one column, got {pred.shape[1]}")
    if list(pred.columns) != [TARGET]:
        raise ValueError(f"Bad prediction column: {list(pred.columns)}")
    if len(pred) != len(valid):
        raise ValueError(f"Bad row count: predictions={len(pred)} valid={len(valid)}")
    values = pd.to_numeric(pred.iloc[:, 0], errors="raise")
    if values.isna().any():
        raise ValueError("Predictions contain NaN values")
    if (values < 0).any() or (values > P_RATED_FARM).any():
        raise ValueError("Predictions are outside physical bounds")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "rows": int(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "zeros": int((values == 0).sum()),
        "corr_same_order_wind80": float(values.corr(valid["wind_speed_80m"])),
        "sha256": digest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate wind farm predictions.")
    parser.add_argument("--skip-cv", action="store_true", help="Skip Q1 backtest before final fit.")
    parser.add_argument("--verbose-models", action="store_true", help="Print CatBoost training logs.")
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--physics-preset",
        choices=sorted(PHYSICS_PRESETS),
        default="public_best",
        help="Checked-in physics preset for feature generation.",
    )
    parser.add_argument(
        "--model-set",
        choices=sorted(MODEL_SETS),
        default=ACTIVE_MODEL_SET,
        help="Ensemble member set to train.",
    )
    parser.add_argument(
        "--blend-weight-lower",
        type=float,
        default=VALIDATOR_WEIGHT_LOWER,
        help="Lower bound for per-model blend weights during validator calibration.",
    )
    parser.add_argument(
        "--hgb-params-path",
        type=Path,
        default=HGB_FAMILY_PARAMS_PATH,
        help="JSON file with tuned dynamic HGB params for hgb_optuna_family.",
    )
    parser.add_argument(
        "--empirical-curve-config-path",
        type=Path,
        default=EMPIRICAL_CURVE_V4_CONFIG_PATH,
        help="JSON file with formula-only tuning knobs for empirical_curve_v4.",
    )
    parser.add_argument(
        "--catboost-task-type",
        choices=["CPU", "GPU"],
        default="CPU",
        help="Run CatBoost models on CPU or NVIDIA CUDA GPU.",
    )
    parser.add_argument(
        "--catboost-devices",
        default=None,
        help="CatBoost GPU device string, e.g. 0 or 0:1. Used only with --catboost-task-type GPU.",
    )
    parser.add_argument(
        "--skip-external-weather",
        action="store_true",
        help="Disable external Open-Meteo ERA5 features.",
    )
    parser.add_argument(
        "--refresh-external-weather",
        action="store_true",
        help="Refetch open weather caches before training.",
    )
    parser.add_argument(
        "--include-nasa-power",
        action="store_true",
        help="Also fetch NASA POWER features. Disabled by default because it can be unavailable without VPN.",
    )
    parser.add_argument(
        "--external-source-set",
        choices=sorted(EXTERNAL_SOURCE_SETS),
        default="baseline",
        help="Optional extra external adapters beyond the baseline ERA5/Open-Meteo/Meteostat layer.",
    )
    parser.add_argument(
        "--external-source-timeout-sec",
        type=float,
        default=DEFAULT_EXTERNAL_SOURCE_TIMEOUT_SECONDS,
        help="Timeout per optional external source request.",
    )
    parser.add_argument(
        "--external-cache-policy",
        choices=sorted(EXTERNAL_CACHE_POLICIES),
        default="cache_first",
        help="Cache policy for optional external source adapters.",
    )
    parser.add_argument("--require-extra-sources", action="store_true")
    parser.add_argument("--enable-source-interactions", action="store_true")
    parser.add_argument("--enable-lag-residual-features", action="store_true")
    parser.add_argument("--enable-regime-prior-v2", action="store_true")
    parser.add_argument("--enable-regime-models", action="store_true")
    parser.add_argument("--enable-regime-v2", action="store_true")
    parser.add_argument("--enable-windfm-diagnostic", action="store_true")
    parser.add_argument("--enable-hgb-quantile-sisters", action="store_true")
    parser.add_argument("--enable-direction-sector-features", action="store_true")
    parser.add_argument("--enable-weather-analog-residual", action="store_true")
    parser.add_argument("--enable-weather-dynamics-features", action="store_true")
    parser.add_argument("--enable-multi-regime-features", action="store_true")
    parser.add_argument("--enable-multi-regime-experts", action="store_true")
    parser.add_argument(
        "--keep-empty-feature-columns",
        action="store_true",
        help="Debug only: keep all-NaN feature columns instead of pruning them before training.",
    )
    parser.add_argument(
        "--skip-openmeteo-forecast",
        action="store_true",
        help="Disable Open-Meteo Historical Forecast source features.",
    )
    parser.add_argument(
        "--openmeteo-forecast-set",
        choices=sorted(OPENMETEO_FORECAST_SETS),
        default="core_open_forecast",
        help="Open-Meteo Historical Forecast source set.",
    )
    parser.add_argument(
        "--openmeteo-fill-policy",
        choices=sorted(OPENMETEO_FILL_POLICIES),
        default="legacy",
        help="How to fill Open-Meteo forecast gaps. Use strict for leakage-safe source ablations.",
    )
    parser.add_argument(
        "--openmeteo-min-coverage",
        type=float,
        default=OPENMETEO_COVERAGE_MIN_DEFAULT,
        help="Minimum overall/yearly source coverage for strict Open-Meteo forecast guard.",
    )
    parser.add_argument(
        "--include-openmeteo-pressure",
        action="store_true",
        help="Fetch pressure-level wind/temperature/geopotential features for selected forecast models.",
    )
    parser.add_argument(
        "--external-weather-cache-dir",
        type=Path,
        default=EXTERNAL_WEATHER_DIR,
        help="Directory for reproducible external weather CSV caches.",
    )
    parser.add_argument(
        "--meteostat-cache-path",
        type=Path,
        default=METEOSTAT_CACHE_PATH,
        help="Cached Meteostat hourly history CSV.",
    )
    parser.add_argument(
        "--skip-meteostat",
        action="store_true",
        help="Disable Meteostat hourly history features.",
    )
    parser.add_argument(
        "--include-meteostat-derived",
        action="store_true",
        help="Opt into the 10 Meteostat lag/condition/bias features added after the public-best run.",
    )
    parser.add_argument(
        "--include-physics-variants",
        action="store_true",
        help="Add manufacturer/public-best/hub-80 physics variant features.",
    )
    parser.add_argument(
        "--include-so-ups-res",
        action="store_true",
        help="Merge optional official SO UPS RES monthly context cache. 2026 months stay disabled by default.",
    )
    parser.add_argument(
        "--so-ups-res-cache-path",
        type=Path,
        default=SO_UPS_RES_CACHE_PATH,
        help="CSV cache parsed from official SO UPS RES monthly reports.",
    )
    parser.add_argument(
        "--include-so-ups-valid-months",
        action="store_true",
        help="Diagnostic only: allow SO UPS RES monthly context after the train period.",
    )
    parser.add_argument(
        "--so-ups-res-policy",
        choices=sorted(SO_UPS_RES_POLICIES),
        default="legacy_best_gap",
        help="SO UPS monthly cache policy. legacy_best_gap reproduces the current best public candidate.",
    )
    parser.add_argument(
        "--so-ups-res-exclude-months",
        default="",
        help="Comma-separated YYYY-MM months used only with --so-ups-res-policy custom_exclude.",
    )
    meteostat_partial = parser.add_mutually_exclusive_group()
    meteostat_partial.add_argument(
        "--allow-partial-meteostat",
        dest="allow_partial_meteostat",
        action="store_true",
        default=True,
        help="Use Meteostat when it covers at least 90%% of requested hours (default).",
    )
    meteostat_partial.add_argument(
        "--no-partial-meteostat",
        dest="allow_partial_meteostat",
        action="store_false",
        help="Require Meteostat to cover the full requested range.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    set_catboost_runtime(args.catboost_task_type, args.catboost_devices)
    set_model_set(args.model_set)
    configure_v8_experiments(
        source_interactions=args.enable_source_interactions,
        lag_residual_features=args.enable_lag_residual_features,
        regime_prior_v2=args.enable_regime_prior_v2,
        regime_models=args.enable_regime_models,
        regime_v2=args.enable_regime_v2,
        windfm_diagnostic=args.enable_windfm_diagnostic,
        hgb_quantile_sisters=args.enable_hgb_quantile_sisters,
        direction_sector_features=args.enable_direction_sector_features,
        weather_analog_residual=args.enable_weather_analog_residual,
        weather_dynamics_features=args.enable_weather_dynamics_features,
        multi_regime_features=args.enable_multi_regime_features,
        multi_regime_experts=args.enable_multi_regime_experts,
    )
    set_validator_weight_lower(args.blend_weight_lower)
    set_hgb_dynamic_params_path(args.hgb_params_path)
    set_empirical_curve_config_path(args.empirical_curve_config_path)
    physics_cfg = physics_config_from_preset(args.physics_preset)
    set_active_physics_config(physics_cfg)
    include_physics_variants = (
        args.include_physics_variants
        or args.model_set.startswith("hgb_")
        or args.model_set in {
            "physics_first_v2",
            "empirical_curve_v3",
            "empirical_curve_v4",
            "empirical_curve_v4_guarded",
        }
    )
    run_started_at = perf_counter()

    print("Loading and engineering features...", flush=True)
    feature_started_at = perf_counter()
    print(f"  physics preset: {args.physics_preset}", flush=True)
    print(f"  physics config: {asdict(physics_cfg)}", flush=True)
    print(f"  model set: {ACTIVE_MODEL_SET} ({len(MODEL_SPECS)} models)", flush=True)
    print(f"  blend weight lower bound: {VALIDATOR_WEIGHT_LOWER:.4f}", flush=True)
    print(f"  HGB params path: {HGB_DYNAMIC_PARAMS_PATH}", flush=True)
    if ACTIVE_MODEL_SET in {"empirical_curve_v4", "empirical_curve_v4_guarded"}:
        print(f"  empirical curve V4 config path: {EMPIRICAL_CURVE_CONFIG_PATH}", flush=True)
        print(f"  empirical curve V4 config: {asdict(ACTIVE_EMPIRICAL_CURVE_V4_CONFIG)}", flush=True)
    if ACTIVE_MODEL_SET.startswith("hgb_optuna") and not HGB_DYNAMIC_PARAMS_PATH.exists():
        print("  WARNING: tuned HGB params file is missing; hgb_opt_* models will use static fallbacks.", flush=True)
    print(
        "  external weather: "
        + ("disabled" if args.skip_external_weather else f"enabled ({args.external_weather_cache_dir})"),
        flush=True,
    )
    print(
        "  Meteostat weather: "
        + ("disabled" if args.skip_meteostat else f"enabled ({args.meteostat_cache_path})")
        + (" partial-ok" if args.allow_partial_meteostat and not args.skip_meteostat else ""),
        flush=True,
    )
    print(
        "  Open-Meteo historical forecast: "
        + (
            "disabled"
            if args.skip_openmeteo_forecast
            else (
                f"enabled ({args.openmeteo_forecast_set}, "
                f"fill={args.openmeteo_fill_policy}, "
                f"pressure={'yes' if openmeteo_pressure_enabled(args.openmeteo_forecast_set, args.include_openmeteo_pressure) else 'no'})"
            )
        ),
        flush=True,
    )
    print(
        "  SO UPS RES monthly context: "
        + ("enabled" if args.include_so_ups_res else "disabled")
        + (f" ({args.so_ups_res_cache_path})" if args.include_so_ups_res else ""),
        flush=True,
    )
    print(
        "  External source set: "
        + (
            "disabled"
            if args.skip_external_weather
            else f"{args.external_source_set} (cache={args.external_cache_policy}, timeout={args.external_source_timeout_sec:g}s)"
        )
        + (" require-extra" if args.require_extra_sources and not args.skip_external_weather else ""),
        flush=True,
    )
    print(
        "  Wind-tech features: "
        + (
            ", ".join(
                name
                for name, enabled in [
                    ("source_interactions", args.enable_source_interactions),
                    ("lag_residual_priors", args.enable_lag_residual_features),
                    ("regime_prior_v2", args.enable_regime_prior_v2),
                    ("regime_models", args.enable_regime_models),
                    ("regime_v2", args.enable_regime_v2),
                    ("hgb_quantile_sisters", args.enable_hgb_quantile_sisters),
                    ("direction_sector_features", args.enable_direction_sector_features),
                    ("weather_analog_residual", args.enable_weather_analog_residual),
                    ("weather_dynamics_features", args.enable_weather_dynamics_features),
                    ("windfm_diagnostic", args.enable_windfm_diagnostic),
                ]
                if enabled
            )
            or "disabled"
        ),
        flush=True,
    )
    print(
        "  Meteostat derived features: "
        + ("enabled" if args.include_meteostat_derived else "disabled"),
        flush=True,
    )
    print(
        "  Physics variant features: "
        + ("enabled" if include_physics_variants else "disabled"),
        flush=True,
    )
    print(
        f"  CatBoost runtime: {CATBOOST_TASK_TYPE}"
        + (f" devices={CATBOOST_DEVICES}" if CATBOOST_DEVICES else ""),
        flush=True,
    )
    train, valid, valid_original_order, feature_cols = load_and_prepare(
        physics_cfg,
        use_external_weather=not args.skip_external_weather,
        external_cache_dir=args.external_weather_cache_dir,
        refresh_external_weather=args.refresh_external_weather,
        include_nasa_power=args.include_nasa_power,
        external_source_set=args.external_source_set,
        external_source_timeout_sec=args.external_source_timeout_sec,
        external_cache_policy=args.external_cache_policy,
        require_extra_sources=args.require_extra_sources,
        include_openmeteo_forecast=not args.skip_openmeteo_forecast,
        openmeteo_forecast_set=args.openmeteo_forecast_set,
        openmeteo_fill_policy=args.openmeteo_fill_policy,
        openmeteo_min_coverage=args.openmeteo_min_coverage,
        include_openmeteo_pressure=args.include_openmeteo_pressure,
        include_meteostat=not args.skip_meteostat,
        meteostat_cache_path=args.meteostat_cache_path,
        allow_partial_meteostat=args.allow_partial_meteostat,
        include_meteostat_derived=args.include_meteostat_derived,
        include_so_ups_res=args.include_so_ups_res,
        so_ups_res_cache_path=args.so_ups_res_cache_path,
        include_so_ups_valid_months=args.include_so_ups_valid_months,
        so_ups_res_policy=args.so_ups_res_policy,
        so_ups_res_exclude_months=args.so_ups_res_exclude_months,
        include_physics_variants=include_physics_variants,
        keep_empty_feature_columns=args.keep_empty_feature_columns,
    )
    print(
        f"  train rows: {len(train)} | valid rows: {len(valid)} | features: {len(feature_cols)} "
        f"({format_duration(perf_counter() - feature_started_at)})",
        flush=True,
    )

    physics_q1_mae = []
    for year, weight in [(2023, 0.2), (2024, 0.3), (2025, 0.5)]:
        mask = (train[DATETIME_COL] >= pd.Timestamp(f"{year}-01-01")) & (
            train[DATETIME_COL] <= pd.Timestamp(f"{year}-03-31 23:00:00")
        )
        mae = mean_absolute_error(train.loc[mask, TARGET], train.loc[mask, "P_physics_farm"])
        physics_q1_mae.append((year, weight, mae))
    standalone_weighted_mae = sum(w * m for _, w, m in physics_q1_mae)
    print(
        f"  physics-only Q1 MAE: " + ", ".join(f"{y}={m:.3f}" for y, _, m in physics_q1_mae),
        flush=True,
    )
    print(f"  physics-only weighted Q1 MAE: {standalone_weighted_mae:.3f} MW", flush=True)

    final_iterations: dict[str, int] | None = None
    final_blend_weights = default_blend_weights()
    if not args.skip_cv:
        _, best_iters, final_blend_weights = run_q1_cv(train, feature_cols)
        final_iterations = derive_final_iterations(best_iters)
        print(f"  final_iterations: {final_iterations}", flush=True)

    pred_asc = train_final_predict(
        train, valid, feature_cols,
        verbose=args.verbose_models,
        final_iterations=final_iterations,
        blend_weights=final_blend_weights,
    )
    write_predictions(pred_asc, valid, valid_original_order, args.output_path)

    stats = validate_predictions(args.output_path, VALID_PATH)
    print(f"Saved predictions to {args.output_path}", flush=True)
    print(
        "  rows={rows} min={min:.3f} max={max:.3f} mean={mean:.3f} zeros={zeros} "
        "corr_same_wind80={corr_same_order_wind80:.3f} sha256={sha256}".format(**stats),
        flush=True,
    )
    print(f"Total runtime: {format_duration(perf_counter() - run_started_at)}", flush=True)


if __name__ == "__main__":
    main()
