"""Formula-only Optuna tuner for empirical_curve_v4.

This tunes only the fold-local empirical curve and PDM correction knobs. It
does not train HGB/CatBoost models and does not consume public leaderboard
scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import predict
from empirical_power_curve_model import (
    EmpiricalCurveV4Config,
    apply_empirical_curve_v4_features,
    fit_empirical_curve_v4_calibrator,
)
from phase_d_optimize_weights import resolve_fold_weights


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_EXTERNAL_WEATHER_DIR = PROJECT_ROOT / "dataset" / "external_weather"
DEFAULT_METEOSTAT_CACHE_PATH = DEFAULT_EXTERNAL_WEATHER_DIR / "meteostat_hourly_azov.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune empirical_curve_v4 formula knobs on historical Q1 folds.")
    parser.add_argument("--trials", type=int, default=300, help="Number of Optuna trials.")
    parser.add_argument(
        "--fold-weights",
        default="recency,raw_similarity",
        help="Comma-separated fold weighting schemes to optimize jointly.",
    )
    parser.add_argument(
        "--physics-preset",
        choices=sorted(predict.PHYSICS_PRESETS),
        default="public_best",
        help="Checked-in physics preset used for feature generation.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "empirical_curve_v4_config.json",
        help="Where to write the best V4 config JSON.",
    )
    parser.add_argument("--seed", type=int, default=predict.SEED)
    parser.add_argument(
        "--skip-external-weather",
        action="store_true",
        help="Disable external Open-Meteo ERA5 features.",
    )
    parser.add_argument(
        "--refresh-external-weather",
        action="store_true",
        help="Refetch external weather caches before tuning.",
    )
    parser.add_argument(
        "--include-nasa-power",
        action="store_true",
        help="Also fetch NASA POWER features.",
    )
    parser.add_argument(
        "--skip-openmeteo-forecast",
        action="store_true",
        help="Disable Open-Meteo Historical Forecast source features.",
    )
    parser.add_argument(
        "--openmeteo-forecast-set",
        choices=sorted(predict.OPENMETEO_FORECAST_SETS),
        default="core_open_forecast",
        help="Open-Meteo Historical Forecast source set.",
    )
    parser.add_argument(
        "--openmeteo-fill-policy",
        choices=sorted(predict.OPENMETEO_FILL_POLICIES),
        default="legacy",
        help="How to fill Open-Meteo forecast gaps. Use strict for leakage-safe source tuning.",
    )
    parser.add_argument(
        "--openmeteo-min-coverage",
        type=float,
        default=predict.OPENMETEO_COVERAGE_MIN_DEFAULT,
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
        default=DEFAULT_EXTERNAL_WEATHER_DIR,
        help="Directory for reproducible external weather CSV caches.",
    )
    parser.add_argument(
        "--meteostat-cache-path",
        type=Path,
        default=DEFAULT_METEOSTAT_CACHE_PATH,
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
        help="Opt into Meteostat-derived features.",
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
    parser.add_argument(
        "--allow-missing-meteostat",
        action="store_true",
        help="Continue when requested Meteostat features are unavailable.",
    )
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def suggest_config(trial) -> EmpiricalCurveV4Config:
    return EmpiricalCurveV4Config(
        ws_bins=trial.suggest_int("ws_bins", 8, 18),
        ti_bins=trial.suggest_int("ti_bins", 3, 6),
        shear_bins=trial.suggest_int("shear_bins", 3, 6),
        veer_bins=trial.suggest_int("veer_bins", 2, 5),
        curve_shrink=trial.suggest_float("curve_shrink", 0.15, 0.85),
        correction_shrink=trial.suggest_float("correction_shrink", 0.05, 0.65),
        min_cell_rows=trial.suggest_categorical("min_cell_rows", [30, 45, 60, 80, 120, 160]),
        min_cell_weight=trial.suggest_float("min_cell_weight", 4.0, 35.0, log=True),
        similarity_scale=trial.suggest_float("similarity_scale", 0.55, 2.20),
        correction_clip=trial.suggest_float("correction_clip", 0.35, 2.20),
    )


def evaluate_config(train, physics_cfg, config: EmpiricalCurveV4Config, schemes: list[str]) -> dict:
    fold_scores: dict[int, float] = {}
    for year, train_idx, valid_idx, _ in predict.q1_folds(train):
        fit_df = train.iloc[train_idx].copy()
        fold_df = train.iloc[valid_idx].copy()
        calibrator = fit_empirical_curve_v4_calibrator(fit_df, fold_df, physics_cfg, config)
        scored = apply_empirical_curve_v4_features(fold_df, calibrator, physics_cfg)
        fold_scores[year] = predict.platform_error_percent(
            fold_df[predict.TARGET].values,
            scored["P_empirical_curve_v4_prior"].values,
        )

    losses_by_scheme: dict[str, float] = {}
    for scheme in schemes:
        weights = resolve_fold_weights(scheme)
        losses_by_scheme[scheme] = float(sum(weights[year] * fold_scores[year] for year in fold_scores))
    objective = float(np.mean(list(losses_by_scheme.values())))
    objective += 0.15 * float(max(losses_by_scheme.values()) - min(losses_by_scheme.values()))
    return {
        "objective": objective,
        "losses_by_scheme": losses_by_scheme,
        "fold_scores": {str(year): float(score) for year, score in fold_scores.items()},
    }


def main() -> None:
    args = parse_args()
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is required: pip install optuna") from exc

    schemes = [item.strip() for item in args.fold_weights.split(",") if item.strip()]
    if not schemes:
        raise ValueError("--fold-weights must contain at least one scheme.")
    for scheme in schemes:
        resolve_fold_weights(scheme)

    predict.set_seed(args.seed)
    predict.set_model_set("empirical_curve_v4")
    physics_cfg = predict.physics_config_from_preset(args.physics_preset)
    predict.set_active_physics_config(physics_cfg)

    print("Loading features for formula-only empirical V4 tuning...")
    require_meteostat = not args.skip_external_weather and not args.skip_meteostat and not args.allow_missing_meteostat
    train, _, _, _ = predict.load_and_prepare(
        physics_cfg,
        use_external_weather=not args.skip_external_weather,
        external_cache_dir=_absolute(args.external_weather_cache_dir),
        refresh_external_weather=args.refresh_external_weather,
        include_nasa_power=args.include_nasa_power,
        include_openmeteo_forecast=not args.skip_openmeteo_forecast,
        openmeteo_forecast_set=args.openmeteo_forecast_set,
        openmeteo_fill_policy=args.openmeteo_fill_policy,
        openmeteo_min_coverage=args.openmeteo_min_coverage,
        include_openmeteo_pressure=args.include_openmeteo_pressure,
        include_meteostat=not args.skip_meteostat,
        meteostat_cache_path=_absolute(args.meteostat_cache_path),
        allow_partial_meteostat=args.allow_partial_meteostat,
        require_meteostat=require_meteostat,
        include_meteostat_derived=args.include_meteostat_derived,
        include_physics_variants=True,
    )
    print(f"  train rows: {len(train)}")
    print(f"  fold weight schemes: {', '.join(schemes)}")

    def objective(trial) -> float:
        config = suggest_config(trial)
        report = evaluate_config(train, physics_cfg, config, schemes)
        trial.set_user_attr("params", asdict(config))
        trial.set_user_attr("losses_by_scheme", report["losses_by_scheme"])
        trial.set_user_attr("fold_scores", report["fold_scores"])
        return report["objective"]

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best_config = EmpiricalCurveV4Config(**study.best_trial.user_attrs["params"])
    top_trials = []
    for trial in sorted(study.trials, key=lambda item: item.value if item.value is not None else float("inf"))[:10]:
        top_trials.append(
            {
                "number": trial.number,
                "value": float(trial.value),
                "params": trial.user_attrs.get("params", trial.params),
                "losses_by_scheme": trial.user_attrs.get("losses_by_scheme", {}),
                "fold_scores": trial.user_attrs.get("fold_scores", {}),
            }
        )

    payload = {
        "source": "tune_empirical_curve.py",
        "physics_preset": args.physics_preset,
        "fold_weight_schemes": schemes,
        "trials": args.trials,
        "best": {
            "value": float(study.best_value),
            "params": asdict(best_config),
            "losses_by_scheme": study.best_trial.user_attrs["losses_by_scheme"],
            "fold_scores": study.best_trial.user_attrs["fold_scores"],
        },
        "top_trials": top_trials,
    }
    output_path = _absolute(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Best objective: {study.best_value:.4f}%")
    print(f"Best params: {json.dumps(asdict(best_config), ensure_ascii=False)}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
