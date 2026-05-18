"""Fold-local empirical power-curve features for strict empirical models.

The calibrator uses only visible training targets plus target-free similarity
to the prediction frame. V3 keeps the original conservative blend of
wind-speed power exponents and a shrunk Q1 residual by potential bin. V4 adds
a PCWG-style monotone empirical curve plus small power-deviation matrices for
turbulence, shear, and veer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "Выработка. Результирующий расчет"
DATETIME_COL = "METEOFORECASTHOUR_OPENM_Datetime"
REPAIR_COL = "Кол-во_ВЭУ_в_ремонте"
N_TOTAL_TURBINES = 26
P_RATED_FARM = 90.09
RHO_STANDARD = 1.225

EXPONENTS = {
    "p15": 1.5,
    "p20": 2.0,
    "p25": 2.5,
    "p30": 3.0,
}
EXPONENT_PRIOR_WEIGHTS = {
    "p15": 0.55,
    "p20": 0.20,
    "p25": 0.15,
    "p30": 0.10,
}
SIMILARITY_COLS = [
    "month",
    "hour_of_day",
    REPAIR_COL,
    "wind_speed_80m",
    "wind_speed_120m",
    "wind_speed_eq",
    "P_physics_farm",
    "shear_120_80",
    "shear_180_80",
    "gust_factor",
    "temperature_80m",
    "pressure_msl",
]


@dataclass(frozen=True)
class EmpiricalCurveCalibrator:
    exponent_weights: dict[str, float] = field(default_factory=lambda: dict(EXPONENT_PRIOR_WEIGHTS))
    bin_edges: list[float] = field(default_factory=lambda: np.linspace(0.0, P_RATED_FARM, 10).tolist())
    bin_corrections: list[float] = field(default_factory=lambda: [0.0] * 9)
    global_correction: float = 0.0


@dataclass(frozen=True)
class EmpiricalCurveV4Config:
    ws_bins: int = 12
    ti_bins: int = 4
    shear_bins: int = 4
    veer_bins: int = 4
    curve_shrink: float = 0.45
    correction_shrink: float = 0.25
    min_cell_rows: int = 60
    min_cell_weight: float = 8.0
    similarity_scale: float = 1.0
    correction_clip: float = 1.15
    guard_delta_strength: float = 0.20
    guard_delta_clip: float = 0.80
    residual_center_strength: float = 0.50


@dataclass(frozen=True)
class EmpiricalCurveV4Calibrator:
    config: EmpiricalCurveV4Config = field(default_factory=EmpiricalCurveV4Config)
    ws_edges: list[float] = field(default_factory=lambda: np.linspace(-1e-6, 1.35 + 1e-6, 13).tolist())
    curve_per_turbine: list[float] = field(default_factory=lambda: [0.0] * 12)
    ti_edges: list[float] = field(default_factory=lambda: np.linspace(0.0, 1.2 + 1e-6, 5).tolist())
    shear_edges: list[float] = field(default_factory=lambda: np.linspace(0.6, 1.8 + 1e-6, 5).tolist())
    veer_edges: list[float] = field(default_factory=lambda: np.linspace(0.0, 90.0 + 1e-6, 5).tolist())
    pdm_ws_ti: list[list[float]] = field(default_factory=lambda: [[0.0] * 4 for _ in range(12)])
    pdm_ws_shear: list[list[float]] = field(default_factory=lambda: [[0.0] * 4 for _ in range(12)])
    pdm_ws_veer: list[list[float]] = field(default_factory=lambda: [[0.0] * 4 for _ in range(12)])
    global_residual: float = 0.0


def default_empirical_curve_v4_config() -> EmpiricalCurveV4Config:
    return EmpiricalCurveV4Config()


def load_empirical_curve_v4_config(path: str | Path | None) -> EmpiricalCurveV4Config:
    if path is None:
        return default_empirical_curve_v4_config()

    path = Path(path)
    if not path.exists():
        return default_empirical_curve_v4_config()

    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and isinstance(payload.get("best"), dict):
        params = payload["best"].get("params", {})
    elif isinstance(payload, dict) and isinstance(payload.get("params"), dict):
        params = payload["params"]
    else:
        params = payload
    if not isinstance(params, dict):
        return default_empirical_curve_v4_config()

    defaults = default_empirical_curve_v4_config()
    field_names = {item.name for item in fields(EmpiricalCurveV4Config)}
    normalized = {item.name: getattr(defaults, item.name) for item in fields(EmpiricalCurveV4Config)}
    normalized.update({key: value for key, value in params.items() if key in field_names})
    normalized["ws_bins"] = int(np.clip(normalized["ws_bins"], 6, 24))
    normalized["ti_bins"] = int(np.clip(normalized["ti_bins"], 2, 8))
    normalized["shear_bins"] = int(np.clip(normalized["shear_bins"], 2, 8))
    normalized["veer_bins"] = int(np.clip(normalized["veer_bins"], 2, 8))
    normalized["curve_shrink"] = float(np.clip(normalized["curve_shrink"], 0.0, 0.95))
    normalized["correction_shrink"] = float(np.clip(normalized["correction_shrink"], 0.0, 0.8))
    normalized["min_cell_rows"] = int(np.clip(normalized["min_cell_rows"], 20, 400))
    normalized["min_cell_weight"] = float(np.clip(normalized["min_cell_weight"], 1.0, 120.0))
    normalized["similarity_scale"] = float(np.clip(normalized["similarity_scale"], 0.35, 3.0))
    normalized["correction_clip"] = float(np.clip(normalized["correction_clip"], 0.15, 3.0))
    normalized["guard_delta_strength"] = float(np.clip(normalized["guard_delta_strength"], 0.0, 0.75))
    normalized["guard_delta_clip"] = float(np.clip(normalized["guard_delta_clip"], 0.05, 3.0))
    normalized["residual_center_strength"] = float(np.clip(normalized["residual_center_strength"], 0.0, 1.0))
    return EmpiricalCurveV4Config(**normalized)


def _empirical_power_curve(v: np.ndarray, cfg, exponent: float) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    p = np.zeros_like(v)
    partial = (v >= cfg.v_cut_in) & (v < cfg.v_rated)
    span = max(float(cfg.v_rated - cfg.v_cut_in), 1e-6)
    p[partial] = cfg.p_rated_per_turbine * ((v[partial] - cfg.v_cut_in) / span) ** exponent
    rated = (v >= cfg.v_rated) & (v < cfg.v_cut_out)
    p[rated] = cfg.p_rated_per_turbine
    return p


def _farm_power_for_exponent(frame: pd.DataFrame, cfg, exponent: float) -> np.ndarray:
    density = (frame["air_density"].values / RHO_STANDARD) ** cfg.density_correction_exp
    p_std = _empirical_power_curve(frame["wind_speed_eq"].values, cfg, exponent)
    p_per_turbine = np.clip(
        p_std * density * cfg.efficiency_factor,
        0.0,
        cfg.p_rated_per_turbine,
    )
    return np.clip(p_per_turbine * frame["n_working_monthly_avg"].values, 0.0, P_RATED_FARM)


def _numeric_series(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


def _density_efficiency(frame: pd.DataFrame, cfg) -> np.ndarray:
    density = _numeric_series(frame, "air_density", RHO_STANDARD).values
    density_corr = (np.clip(density, 0.75, 1.55) / RHO_STANDARD) ** cfg.density_correction_exp
    return np.clip(density_corr * cfg.efficiency_factor, 0.35, 2.2)


def _working_turbines(frame: pd.DataFrame) -> np.ndarray:
    if "n_working_monthly_avg" in frame:
        n_working = _numeric_series(frame, "n_working_monthly_avg", N_TOTAL_TURBINES).values
    elif REPAIR_COL in frame:
        n_working = N_TOTAL_TURBINES - _numeric_series(frame, REPAIR_COL, 0.0).values
    else:
        n_working = np.full(len(frame), N_TOTAL_TURBINES, dtype=float)
    return np.clip(n_working, 0.0, N_TOTAL_TURBINES)


def _physics_std_per_turbine(frame: pd.DataFrame, cfg) -> np.ndarray:
    density_eff = _density_efficiency(frame, cfg)
    if "P_physics_per_turbine" in frame:
        physics = _numeric_series(frame, "P_physics_per_turbine", 0.0).values / np.maximum(density_eff, 1e-6)
        return np.clip(physics, 0.0, cfg.p_rated_per_turbine)
    return _empirical_power_curve(_numeric_series(frame, "wind_speed_eq", 0.0).values, cfg, cfg.cubic_exponent)


def _add_nworking_scenario_features(out: pd.DataFrame) -> pd.DataFrame:
    per_turbine = _numeric_series(out, "P_physics_per_turbine", 0.0).values
    n_working = _working_turbines(out)
    physics_farm = _numeric_series(out, "P_physics_farm", 0.0).values
    out["P_physics_farm_nworking_plus1"] = np.clip(
        per_turbine * np.clip(n_working + 1.0, 0.0, N_TOTAL_TURBINES),
        0.0,
        P_RATED_FARM,
    )
    out["P_physics_farm_nworking_minus1"] = np.clip(
        per_turbine * np.clip(n_working - 1.0, 0.0, N_TOTAL_TURBINES),
        0.0,
        P_RATED_FARM,
    )
    out["P_physics_farm_nworking_plus1_delta"] = out["P_physics_farm_nworking_plus1"] - physics_farm
    out["P_physics_farm_nworking_minus1_delta"] = out["P_physics_farm_nworking_minus1"] - physics_farm
    return out


def _v4_inputs(frame: pd.DataFrame, cfg) -> dict[str, np.ndarray]:
    wind_eq = _numeric_series(frame, "wind_speed_eq", 0.0).values
    span = max(float(cfg.v_rated - cfg.v_cut_in), 1e-6)
    ws_norm = np.clip((wind_eq - cfg.v_cut_in) / span, 0.0, 1.35)

    ws10 = np.maximum(_numeric_series(frame, "wind_speed_10m", 0.0).values, 0.1)
    ws80 = np.maximum(_numeric_series(frame, "wind_speed_80m", 0.0).values, 0.1)
    ws120 = np.maximum(_numeric_series(frame, "wind_speed_120m", 0.0).values, 0.1)
    ws180 = np.maximum(_numeric_series(frame, "wind_speed_180m", 0.0).values, 0.1)

    if "turbulence_intensity" in frame:
        ti_proxy = _numeric_series(frame, "turbulence_intensity", 0.0).values
    elif "gust_factor" in frame:
        ti_proxy = _numeric_series(frame, "gust_factor", 1.0).values - 1.0
    else:
        ti_proxy = (_numeric_series(frame, "wind_gusts_10m", 0.0).values - ws10) / ws10

    rolling_candidates = [
        "wind_speed_80m_roll12_std",
        "wind_speed_120m_roll12_std",
        "wind_gusts_10m_roll12_std",
    ]
    for col in rolling_candidates:
        if col in frame:
            roll_ti = _numeric_series(frame, col, 0.0).values / ws80
            ti_proxy = 0.70 * ti_proxy + 0.30 * roll_ti
            break

    shear_ratio = np.clip(ws120 / ws80, 0.45, 2.25)
    rotor_shear = np.clip((ws180 - ws80) / ws80, -1.00, 2.00)
    if "direction_shear_120_80" in frame:
        veer_abs = np.abs(_numeric_series(frame, "direction_shear_120_80", 0.0).values)
    elif "wind_direction_120m" in frame and "wind_direction_80m" in frame:
        diff = (_numeric_series(frame, "wind_direction_120m", 0.0) - _numeric_series(frame, "wind_direction_80m", 0.0)) % 1.0
        diff = diff.where(diff <= 0.5, diff - 1.0)
        veer_abs = np.abs(diff.values * 360.0)
    else:
        veer_abs = np.zeros(len(frame), dtype=float)

    return {
        "ws_norm": ws_norm,
        "ti_proxy": np.clip(ti_proxy, 0.0, 1.5),
        "shear_ratio": shear_ratio,
        "rotor_shear": rotor_shear,
        "veer_abs": np.clip(veer_abs, 0.0, 180.0),
    }


def initialize_empirical_curve_features(frame: pd.DataFrame, cfg) -> pd.DataFrame:
    """Create deterministic V3 columns before any fold-local target fitting."""

    out = frame.copy()
    wind = pd.to_numeric(out["wind_speed_eq"], errors="coerce").fillna(0.0).clip(lower=0.0)
    for suffix, exponent in EXPONENTS.items():
        out[f"wind_speed_eq_{suffix}"] = wind ** exponent
        out[f"P_empirical_curve_{suffix}"] = _farm_power_for_exponent(out, cfg, exponent)

    raw = np.zeros(len(out), dtype=float)
    for suffix, weight in EXPONENT_PRIOR_WEIGHTS.items():
        raw += weight * out[f"P_empirical_curve_{suffix}"].values
    out["P_empirical_curve_v3_raw"] = np.clip(raw, 0.0, P_RATED_FARM)
    out["q1_curve_residual_bin"] = 0.0
    out["empirical_curve_v3_pbin"] = _bin_indices(
        out["P_physics_farm"].values,
        np.linspace(0.0, P_RATED_FARM, 10),
    ).astype(float)
    out["P_empirical_curve_v3_prior"] = out["P_empirical_curve_v3_raw"]
    out["P_empirical_curve_v3_delta"] = out["P_empirical_curve_v3_prior"] - out["P_physics_farm"]

    per_turbine = pd.to_numeric(out["P_physics_per_turbine"], errors="coerce").fillna(0.0).values
    n_working = pd.to_numeric(out["n_working_monthly_avg"], errors="coerce").fillna(0.0).values
    out["P_physics_farm_nworking_plus1"] = np.clip(
        per_turbine * np.clip(n_working + 1.0, 0.0, N_TOTAL_TURBINES),
        0.0,
        P_RATED_FARM,
    )
    out["P_physics_farm_nworking_minus1"] = np.clip(
        per_turbine * np.clip(n_working - 1.0, 0.0, N_TOTAL_TURBINES),
        0.0,
        P_RATED_FARM,
    )
    out["P_physics_farm_nworking_plus1_delta"] = out["P_physics_farm_nworking_plus1"] - out["P_physics_farm"]
    out["P_physics_farm_nworking_minus1_delta"] = out["P_physics_farm_nworking_minus1"] - out["P_physics_farm"]
    return out


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not bool(mask.any()):
        return 0.0
    return float(np.average(values[mask], weights=weights[mask]))


def _prediction_similarity_weights(train_frame: pd.DataFrame, pred_frame: pd.DataFrame) -> np.ndarray:
    cols = [col for col in SIMILARITY_COLS if col in train_frame.columns and col in pred_frame.columns]
    if not cols:
        return np.ones(len(train_frame), dtype=float)

    train_numeric = train_frame[cols].apply(pd.to_numeric, errors="coerce")
    pred_numeric = pred_frame[cols].apply(pd.to_numeric, errors="coerce")
    center = pred_numeric.median(numeric_only=True)
    scale = pd.concat([train_numeric, pred_numeric], axis=0).std(numeric_only=True).replace(0.0, np.nan)
    scale = scale.fillna(1.0)
    z = ((train_numeric - center) / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dist2 = np.mean(np.square(z.values), axis=1)
    return np.exp(-0.5 * dist2).clip(0.03, 1.0)


def prediction_similarity_weights(train_frame: pd.DataFrame, pred_frame: pd.DataFrame) -> np.ndarray:
    """Public wrapper for target-free similarity weights used by residual guards."""

    return _prediction_similarity_weights(train_frame, pred_frame)


def _safe_bin_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 20 or np.nanmax(values) <= np.nanmin(values):
        return np.linspace(0.0, P_RATED_FARM, 10)
    edges = np.quantile(values, np.linspace(0.0, 1.0, 10))
    edges[0] = min(edges[0], 0.0) - 1e-6
    edges[-1] = max(edges[-1], P_RATED_FARM) + 1e-6
    for idx in range(1, len(edges)):
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = edges[idx - 1] + 1e-6
    return edges


def _bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(values, dtype=float), edges[1:-1], right=True).clip(0, 8)


def _learn_exponent_weights(frame: pd.DataFrame, weights: np.ndarray) -> dict[str, float]:
    target = pd.to_numeric(frame[TARGET], errors="coerce").values
    scores = {}
    for suffix in EXPONENTS:
        pred = pd.to_numeric(frame[f"P_empirical_curve_{suffix}"], errors="coerce").values
        scores[suffix] = _weighted_mean(np.abs(target - pred), weights)

    best = min(scores.values()) if scores else 0.0
    raw = np.array([np.exp(-(scores[suffix] - best) / 0.35) for suffix in EXPONENTS], dtype=float)
    raw = raw / raw.sum() if float(raw.sum()) > 0.0 else np.array(list(EXPONENT_PRIOR_WEIGHTS.values()))
    prior = np.array([EXPONENT_PRIOR_WEIGHTS[suffix] for suffix in EXPONENTS], dtype=float)
    blended = 0.55 * raw + 0.45 * prior
    blended = blended / blended.sum()
    return {suffix: float(weight) for suffix, weight in zip(EXPONENTS, blended)}


def _weighted_curve(frame: pd.DataFrame, exponent_weights: dict[str, float]) -> np.ndarray:
    raw = np.zeros(len(frame), dtype=float)
    for suffix, weight in exponent_weights.items():
        if f"P_empirical_curve_{suffix}" in frame:
            raw += float(weight) * frame[f"P_empirical_curve_{suffix}"].values
    return np.clip(raw, 0.0, P_RATED_FARM)


def fit_empirical_curve_calibrator(
    train_frame: pd.DataFrame,
    pred_frame: pd.DataFrame,
    cfg,
) -> EmpiricalCurveCalibrator:
    """Fit a conservative Q1 empirical curve calibrator from train only."""

    fit = initialize_empirical_curve_features(train_frame, cfg)
    pred = initialize_empirical_curve_features(pred_frame, cfg)
    if TARGET not in fit:
        return EmpiricalCurveCalibrator()

    q1_mask = fit["month"].isin([1, 2, 3]) & pd.to_numeric(fit[TARGET], errors="coerce").notna()
    if int(q1_mask.sum()) < 250:
        q1_mask = pd.to_numeric(fit[TARGET], errors="coerce").notna()
    q1 = fit.loc[q1_mask].copy()
    if q1.empty:
        return EmpiricalCurveCalibrator()

    weights = _prediction_similarity_weights(q1, pred)
    exponent_weights = _learn_exponent_weights(q1, weights)
    raw_curve = _weighted_curve(q1, exponent_weights)
    residual = pd.to_numeric(q1[TARGET], errors="coerce").values - raw_curve

    edges = _safe_bin_edges(pred["P_physics_farm"].values)
    bin_idx = _bin_indices(q1["P_physics_farm"].values, edges)
    global_resid = float(np.clip(_weighted_mean(residual, weights), -1.25, 1.25))

    corrections = []
    for bin_no in range(9):
        mask = bin_idx == bin_no
        n_eff = float(weights[mask].sum())
        if int(mask.sum()) >= 60 and n_eff > 8.0:
            bin_resid = _weighted_mean(residual[mask], weights[mask])
            shrink = 0.35 * n_eff / (n_eff + 300.0)
            correction = bin_resid * shrink
        else:
            correction = 0.10 * global_resid
        corrections.append(float(np.clip(correction, -1.15, 1.15)))

    return EmpiricalCurveCalibrator(
        exponent_weights=exponent_weights,
        bin_edges=edges.tolist(),
        bin_corrections=corrections,
        global_correction=global_resid,
    )


def apply_empirical_curve_features(
    frame: pd.DataFrame,
    calibrator: EmpiricalCurveCalibrator,
    cfg,
) -> pd.DataFrame:
    """Apply fold-local empirical V3 features to train/eval/pred frames."""

    out = initialize_empirical_curve_features(frame, cfg)
    raw = _weighted_curve(out, calibrator.exponent_weights)
    edges = np.asarray(calibrator.bin_edges, dtype=float)
    bin_idx = _bin_indices(out["P_physics_farm"].values, edges)
    corrections = np.asarray(calibrator.bin_corrections, dtype=float)
    correction = corrections[bin_idx]
    correction = np.where(out["month"].isin([1, 2, 3]).values, correction, 0.0)

    out["P_empirical_curve_v3_raw"] = raw
    out["empirical_curve_v3_pbin"] = bin_idx.astype(float)
    out["q1_curve_residual_bin"] = correction
    out["P_empirical_curve_v3_prior"] = np.clip(raw + correction, 0.0, P_RATED_FARM)
    out["P_empirical_curve_v3_delta"] = out["P_empirical_curve_v3_prior"] - out["P_physics_farm"]
    return out


def _safe_feature_edges(values: np.ndarray, n_bins: int, lower: float, upper: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n_bins = int(max(n_bins, 2))
    if len(values) >= n_bins * 4 and np.nanmax(values) > np.nanmin(values):
        clipped = np.clip(values, lower, upper)
        edges = np.quantile(clipped, np.linspace(0.0, 1.0, n_bins + 1))
    else:
        edges = np.linspace(lower, upper, n_bins + 1)
    edges[0] = lower - 1e-6
    edges[-1] = upper + 1e-6
    for idx in range(1, len(edges)):
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = edges[idx - 1] + 1e-6
    return edges


def _bin_indices_general(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    n_bins = max(len(edges) - 1, 1)
    return np.digitize(np.asarray(values, dtype=float), edges[1:-1], right=True).clip(0, n_bins - 1)


def _lookup_1d(values: np.ndarray, edges: np.ndarray, table: np.ndarray) -> np.ndarray:
    idx = _bin_indices_general(values, edges)
    table = np.asarray(table, dtype=float)
    if len(table) == 0:
        return np.zeros(len(idx), dtype=float)
    return table[np.clip(idx, 0, len(table) - 1)]


def _lookup_2d(
    x_values: np.ndarray,
    z_values: np.ndarray,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    table: np.ndarray,
) -> np.ndarray:
    table = np.asarray(table, dtype=float)
    if table.size == 0:
        return np.zeros(len(x_values), dtype=float)
    x_idx = _bin_indices_general(x_values, x_edges).clip(0, table.shape[0] - 1)
    z_idx = _bin_indices_general(z_values, z_edges).clip(0, table.shape[1] - 1)
    return table[x_idx, z_idx]


def initialize_empirical_curve_v4_features(
    frame: pd.DataFrame,
    cfg,
    config: EmpiricalCurveV4Config | None = None,
) -> pd.DataFrame:
    """Create deterministic V4 columns before fold-local target fitting."""

    out = frame.copy()
    inputs = _v4_inputs(out, cfg)
    out["empirical_v4_ws_norm"] = inputs["ws_norm"]
    out["empirical_v4_ti_proxy"] = inputs["ti_proxy"]
    out["empirical_v4_shear_ratio"] = inputs["shear_ratio"]
    out["empirical_v4_rotor_shear"] = inputs["rotor_shear"]
    out["empirical_v4_veer_abs"] = inputs["veer_abs"]

    physics_std = _physics_std_per_turbine(out, cfg)
    out["empirical_v4_curve_per_turbine"] = physics_std
    out["P_empirical_curve_v4_base"] = _numeric_series(out, "P_physics_farm", 0.0).values
    out["pdm_ws_ti_correction"] = 0.0
    out["pdm_ws_shear_correction"] = 0.0
    out["pdm_ws_veer_correction"] = 0.0
    out["P_empirical_curve_v4_prior"] = out["P_empirical_curve_v4_base"]
    out["P_empirical_curve_v4_delta"] = out["P_empirical_curve_v4_prior"] - _numeric_series(
        out, "P_physics_farm", 0.0
    ).values
    out = _add_nworking_scenario_features(out)
    return out


def add_empirical_curve_v4_guarded_features(
    frame: pd.DataFrame,
    config: EmpiricalCurveV4Config | None = None,
) -> pd.DataFrame:
    """Anchor V4's PDM signal to V3 so bad V4 priors cannot dominate shape.

    This keeps the useful PCWG-style columns available to the learner, but the
    standalone guarded prior only moves a clipped fraction away from V3.
    """

    config = config or default_empirical_curve_v4_config()
    out = frame.copy()
    if "P_empirical_curve_v3_prior" in out:
        anchor = _numeric_series(out, "P_empirical_curve_v3_prior", 0.0).values
    else:
        anchor = _numeric_series(out, "P_physics_farm", 0.0).values
    v4_prior = _numeric_series(out, "P_empirical_curve_v4_prior", 0.0).values
    delta = np.clip(v4_prior - anchor, -config.guard_delta_clip, config.guard_delta_clip)
    guarded_delta = config.guard_delta_strength * delta
    out["P_empirical_curve_v4_guarded_prior"] = np.clip(anchor + guarded_delta, 0.0, P_RATED_FARM)
    out["P_empirical_curve_v4_guarded_delta"] = out["P_empirical_curve_v4_guarded_prior"] - anchor
    out["P_empirical_curve_v4_guarded_delta_physics"] = (
        out["P_empirical_curve_v4_guarded_prior"] - _numeric_series(out, "P_physics_farm", 0.0).values
    )
    return out


def _q1_target_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if TARGET not in frame:
        return frame.iloc[0:0].copy()
    target_mask = pd.to_numeric(frame[TARGET], errors="coerce").notna()
    if "month" in frame:
        q1_mask = frame["month"].isin([1, 2, 3]) & target_mask
        if int(q1_mask.sum()) >= 250:
            return frame.loc[q1_mask].copy()
    return frame.loc[target_mask].copy()


def _scaled_similarity_weights(
    train_frame: pd.DataFrame,
    pred_frame: pd.DataFrame,
    config: EmpiricalCurveV4Config,
) -> np.ndarray:
    weights = _prediction_similarity_weights(train_frame, pred_frame)
    exponent = 1.0 / max(float(config.similarity_scale), 0.1)
    weights = np.power(np.clip(weights, 1e-4, 1.0), exponent)
    return weights.clip(0.02, 1.0)


def _physics_curve_for_edges(ws_edges: np.ndarray, cfg) -> np.ndarray:
    centers = (ws_edges[:-1] + ws_edges[1:]) / 2.0
    span = max(float(cfg.v_rated - cfg.v_cut_in), 1e-6)
    wind = cfg.v_cut_in + centers * span
    return np.clip(_empirical_power_curve(wind, cfg, cfg.cubic_exponent), 0.0, cfg.p_rated_per_turbine)


def _fallback_v4_calibrator(
    pred_frame: pd.DataFrame,
    cfg,
    config: EmpiricalCurveV4Config,
) -> EmpiricalCurveV4Calibrator:
    pred = initialize_empirical_curve_v4_features(pred_frame, cfg, config)
    inputs = _v4_inputs(pred, cfg)
    ws_edges = _safe_feature_edges(inputs["ws_norm"], config.ws_bins, 0.0, 1.35)
    ti_edges = _safe_feature_edges(inputs["ti_proxy"], config.ti_bins, 0.0, 1.20)
    shear_edges = _safe_feature_edges(inputs["shear_ratio"], config.shear_bins, 0.60, 1.80)
    veer_edges = _safe_feature_edges(inputs["veer_abs"], config.veer_bins, 0.0, 90.0)
    curve = _physics_curve_for_edges(ws_edges, cfg)
    return EmpiricalCurveV4Calibrator(
        config=config,
        ws_edges=ws_edges.tolist(),
        curve_per_turbine=curve.tolist(),
        ti_edges=ti_edges.tolist(),
        shear_edges=shear_edges.tolist(),
        veer_edges=veer_edges.tolist(),
        pdm_ws_ti=np.zeros((config.ws_bins, config.ti_bins), dtype=float).tolist(),
        pdm_ws_shear=np.zeros((config.ws_bins, config.shear_bins), dtype=float).tolist(),
        pdm_ws_veer=np.zeros((config.ws_bins, config.veer_bins), dtype=float).tolist(),
    )


def _learn_monotone_v4_curve(
    fit: pd.DataFrame,
    pred: pd.DataFrame,
    weights: np.ndarray,
    cfg,
    config: EmpiricalCurveV4Config,
) -> tuple[np.ndarray, np.ndarray]:
    fit_inputs = _v4_inputs(fit, cfg)
    pred_inputs = _v4_inputs(pred, cfg)
    ws_edges = _safe_feature_edges(pred_inputs["ws_norm"], config.ws_bins, 0.0, 1.35)
    ws_bin = _bin_indices_general(fit_inputs["ws_norm"], ws_edges)

    target = pd.to_numeric(fit[TARGET], errors="coerce").values
    n_working = np.maximum(_working_turbines(fit), 1.0)
    density_eff = _density_efficiency(fit, cfg)
    target_std_per_turbine = np.clip(
        target / np.maximum(n_working * density_eff, 1e-6),
        0.0,
        cfg.p_rated_per_turbine,
    )
    physics_std = _physics_std_per_turbine(fit, cfg)
    physics_fallback = _physics_curve_for_edges(ws_edges, cfg)

    curve = np.zeros(config.ws_bins, dtype=float)
    for bin_no in range(config.ws_bins):
        mask = ws_bin == bin_no
        n_eff = float(weights[mask].sum())
        if int(mask.sum()) >= max(12, config.min_cell_rows // 3) and n_eff > 0.0:
            target_mean = _weighted_mean(target_std_per_turbine[mask], weights[mask])
            prior_mean = _weighted_mean(physics_std[mask], weights[mask])
            evidence = n_eff / (n_eff + config.min_cell_weight)
            target_weight = (1.0 - config.curve_shrink) * evidence
            curve[bin_no] = target_weight * target_mean + (1.0 - target_weight) * prior_mean
        else:
            curve[bin_no] = physics_fallback[bin_no]

    curve = np.clip(curve, 0.0, cfg.p_rated_per_turbine)
    centers = (ws_edges[:-1] + ws_edges[1:]) / 2.0
    last = 0.0
    for idx, center in enumerate(centers):
        if center <= 1.05:
            curve[idx] = max(curve[idx], last)
            last = curve[idx]
        else:
            curve[idx] = max(curve[idx], last)
            last = max(last, curve[idx])
    return ws_edges, np.clip(curve, 0.0, cfg.p_rated_per_turbine)


def _v4_base_prior(
    frame: pd.DataFrame,
    cfg,
    ws_edges: np.ndarray,
    curve_per_turbine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inputs = _v4_inputs(frame, cfg)
    curve_std = _lookup_1d(inputs["ws_norm"], ws_edges, curve_per_turbine)
    farm = np.clip(
        curve_std * _density_efficiency(frame, cfg) * _working_turbines(frame),
        0.0,
        P_RATED_FARM,
    )
    return curve_std, farm


def _fit_pdm_matrix(
    residual: np.ndarray,
    weights: np.ndarray,
    x_values: np.ndarray,
    z_values: np.ndarray,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    config: EmpiricalCurveV4Config,
) -> np.ndarray:
    x_bins = _bin_indices_general(x_values, x_edges)
    z_bins = _bin_indices_general(z_values, z_edges)
    matrix = np.zeros((len(x_edges) - 1, len(z_edges) - 1), dtype=float)
    for x_bin in range(matrix.shape[0]):
        for z_bin in range(matrix.shape[1]):
            mask = (x_bins == x_bin) & (z_bins == z_bin)
            n_eff = float(weights[mask].sum())
            if int(mask.sum()) < config.min_cell_rows or n_eff < config.min_cell_weight:
                continue
            mean_residual = _weighted_mean(residual[mask], weights[mask])
            shrink = config.correction_shrink * n_eff / (n_eff + config.min_cell_weight * 12.0)
            matrix[x_bin, z_bin] = float(
                np.clip(mean_residual * shrink, -config.correction_clip, config.correction_clip)
            )
    return matrix


def fit_empirical_curve_v4_calibrator(
    train_frame: pd.DataFrame,
    pred_frame: pd.DataFrame,
    cfg,
    config: EmpiricalCurveV4Config | None = None,
) -> EmpiricalCurveV4Calibrator:
    """Fit a fold-local Q1 monotone curve and PCWG-style PDM corrections."""

    config = config or default_empirical_curve_v4_config()
    fit_all = initialize_empirical_curve_v4_features(train_frame, cfg, config)
    pred = initialize_empirical_curve_v4_features(pred_frame, cfg, config)
    q1 = _q1_target_rows(fit_all)
    if q1.empty:
        return _fallback_v4_calibrator(pred, cfg, config)

    weights = _scaled_similarity_weights(q1, pred, config)
    ws_edges, curve = _learn_monotone_v4_curve(q1, pred, weights, cfg, config)
    _, base_prior = _v4_base_prior(q1, cfg, ws_edges, curve)
    residual = pd.to_numeric(q1[TARGET], errors="coerce").values - base_prior
    residual = np.where(np.isfinite(residual), residual, 0.0)
    global_residual = float(np.clip(_weighted_mean(residual, weights), -config.correction_clip, config.correction_clip))

    pred_inputs = _v4_inputs(pred, cfg)
    fit_inputs = _v4_inputs(q1, cfg)
    ti_edges = _safe_feature_edges(pred_inputs["ti_proxy"], config.ti_bins, 0.0, 1.20)
    shear_edges = _safe_feature_edges(pred_inputs["shear_ratio"], config.shear_bins, 0.60, 1.80)
    veer_edges = _safe_feature_edges(pred_inputs["veer_abs"], config.veer_bins, 0.0, 90.0)
    pdm_ws_ti = _fit_pdm_matrix(
        residual,
        weights,
        fit_inputs["ws_norm"],
        fit_inputs["ti_proxy"],
        ws_edges,
        ti_edges,
        config,
    )
    pdm_ws_shear = _fit_pdm_matrix(
        residual,
        weights,
        fit_inputs["ws_norm"],
        fit_inputs["shear_ratio"],
        ws_edges,
        shear_edges,
        config,
    )
    pdm_ws_veer = _fit_pdm_matrix(
        residual,
        weights,
        fit_inputs["ws_norm"],
        fit_inputs["veer_abs"],
        ws_edges,
        veer_edges,
        config,
    )

    return EmpiricalCurveV4Calibrator(
        config=config,
        ws_edges=ws_edges.tolist(),
        curve_per_turbine=curve.tolist(),
        ti_edges=ti_edges.tolist(),
        shear_edges=shear_edges.tolist(),
        veer_edges=veer_edges.tolist(),
        pdm_ws_ti=pdm_ws_ti.tolist(),
        pdm_ws_shear=pdm_ws_shear.tolist(),
        pdm_ws_veer=pdm_ws_veer.tolist(),
        global_residual=global_residual,
    )


def apply_empirical_curve_v4_features(
    frame: pd.DataFrame,
    calibrator: EmpiricalCurveV4Calibrator,
    cfg,
) -> pd.DataFrame:
    """Apply fold-local empirical V4 features to train/eval/pred frames."""

    out = initialize_empirical_curve_v4_features(frame, cfg, calibrator.config)
    inputs = _v4_inputs(out, cfg)
    ws_edges = np.asarray(calibrator.ws_edges, dtype=float)
    curve = np.asarray(calibrator.curve_per_turbine, dtype=float)
    curve_std, base_prior = _v4_base_prior(out, cfg, ws_edges, curve)

    ti_corr = _lookup_2d(
        inputs["ws_norm"],
        inputs["ti_proxy"],
        ws_edges,
        np.asarray(calibrator.ti_edges, dtype=float),
        np.asarray(calibrator.pdm_ws_ti, dtype=float),
    )
    shear_corr = _lookup_2d(
        inputs["ws_norm"],
        inputs["shear_ratio"],
        ws_edges,
        np.asarray(calibrator.shear_edges, dtype=float),
        np.asarray(calibrator.pdm_ws_shear, dtype=float),
    )
    veer_corr = _lookup_2d(
        inputs["ws_norm"],
        inputs["veer_abs"],
        ws_edges,
        np.asarray(calibrator.veer_edges, dtype=float),
        np.asarray(calibrator.pdm_ws_veer, dtype=float),
    )
    q1_mask = out["month"].isin([1, 2, 3]).values if "month" in out else np.ones(len(out), dtype=bool)
    ti_corr = np.where(q1_mask, ti_corr, 0.0)
    shear_corr = np.where(q1_mask, shear_corr, 0.0)
    veer_corr = np.where(q1_mask, veer_corr, 0.0)
    combined = 0.40 * ti_corr + 0.35 * shear_corr + 0.25 * veer_corr
    combined = np.clip(combined, -calibrator.config.correction_clip, calibrator.config.correction_clip)

    out["empirical_v4_curve_per_turbine"] = curve_std
    out["P_empirical_curve_v4_base"] = base_prior
    out["pdm_ws_ti_correction"] = ti_corr
    out["pdm_ws_shear_correction"] = shear_corr
    out["pdm_ws_veer_correction"] = veer_corr
    out["P_empirical_curve_v4_prior"] = np.clip(base_prior + combined, 0.0, P_RATED_FARM)
    out["P_empirical_curve_v4_delta"] = out["P_empirical_curve_v4_prior"] - _numeric_series(
        out, "P_physics_farm", 0.0
    ).values
    return out
