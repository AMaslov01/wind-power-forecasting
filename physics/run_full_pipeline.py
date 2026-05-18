"""
Run the current best temporal-physics solution end to end.

This runner keeps the dbc9db6 model stack intact:
  1. optionally tune the physical baseline with Optuna;
  2. validate the tuned config inside the full temporal ensemble and save OOF;
  3. optimize blend weights from OOF predictions;
  4. train the final ensemble on all train rows;
  5. write and validate predictions.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_CANDIDATE = "public_best_hgb_family"
DEFAULT_EXTERNAL_WEATHER_DIR = PROJECT_ROOT / "dataset" / "external_weather"
DEFAULT_METEOSTAT_CACHE_PATH = DEFAULT_EXTERNAL_WEATHER_DIR / "meteostat_hourly_azov.csv"
DEFAULT_SO_UPS_RES_CACHE_PATH = PROJECT_ROOT / "dataset" / "external_energy" / "so_ups_res_monthly.csv"
PHYSICS_PRESET_CHOICES = ["public_best", "legacy_tuned", "manufacturer", "tune_results"]
MODEL_SET_CHOICES = [
    "all_family_probe",
    "empirical_curve_v3",
    "empirical_curve_v4",
    "empirical_curve_v4_guarded",
    "forest_only_probe",
    "forest_probe",
    "hgb_family",
    "hgb_optuna_family",
    "hgb_optuna_only",
    "legacy424",
    "lgb_probe",
]
FOLD_WEIGHT_CHOICES = ["recency", "valid_similarity", "raw_similarity"]
OPENMETEO_FORECAST_SET_CHOICES = [
    "core_open_forecast",
    "core_plus_arpege",
    "core_plus_cma",
    "core_plus_icon_eu",
    "core_pressure_forecast",
    "ecmwf_only",
    "expanded_open_forecast",
    "expanded_pressure_forecast",
    "gfs_only",
    "icon_only",
    "pressure_level_forecast",
]
OPENMETEO_FILL_POLICY_CHOICES = ["legacy", "strict"]
EXTERNAL_SOURCE_SET_CHOICES = [
    "all_available",
    "all_strict",
    "baseline",
    "climatology_probe",
    "diagnostic_forecast",
    "keyed_safe",
    "open_free",
    "strict_copernicus",
    "strict_keyed",
    "strict_open",
    "tech_features",
]
EXTERNAL_CACHE_POLICY_CHOICES = ["cache_first", "refresh", "offline"]
SO_UPS_RES_POLICY_CHOICES = ["legacy_best_gap", "clean_48", "custom_exclude"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Optuna, Q1 OOF CV, blend optimization, and final prediction."
    )
    parser.add_argument("--trials", type=int, default=1200, help="Number of Optuna physics trials.")
    parser.add_argument(
        "--tune-search-space",
        choices=["narrow", "wide"],
        default="narrow",
        help="Physics Optuna search space passed to tune_physics.py.",
    )
    parser.add_argument(
        "--blend-method",
        choices=["slsqp", "optuna", "both", "all"],
        default="all",
        help="Blend optimizer set. 'all' adds non-negative ridge stacking.",
    )
    parser.add_argument(
        "--blend-trials",
        type=int,
        default=1500,
        help="Number of Optuna trials for blend weights.",
    )
    parser.add_argument(
        "--candidate-name",
        default=DEFAULT_CANDIDATE,
        help=f"Name used for OOF/cache/result files (default: {DEFAULT_CANDIDATE}).",
    )
    parser.add_argument(
        "--physics-preset",
        choices=PHYSICS_PRESET_CHOICES,
        default="public_best",
        help="Physics source. Use tune_results to consume physics/tune_results.json.",
    )
    parser.add_argument(
        "--model-set",
        choices=MODEL_SET_CHOICES,
        default="hgb_family",
        help="Ensemble member set to train.",
    )
    parser.add_argument(
        "--fold-weights",
        choices=FOLD_WEIGHT_CHOICES,
        default="recency",
        help="Fold weighting scheme for final blend optimization.",
    )
    parser.add_argument(
        "--blend-weight-lower",
        type=float,
        default=0.001,
        help="Lower bound for per-model blend weights in validator and final blend search.",
    )
    parser.add_argument(
        "--blend-constraint-profile",
        choices=["none", "hgb_conservative", "lgb_conservative", "forest_conservative", "all_family_conservative"],
        default="none",
        help="Optional conservative group constraints/priors for final blend search.",
    )
    parser.add_argument(
        "--stable-hgb-min-weight",
        type=float,
        default=None,
        help="Minimum combined final blend weight for hgb_direct_deep + hgb_residual_smooth.",
    )
    parser.add_argument(
        "--optuna-hgb-max-weight",
        type=float,
        default=None,
        help="Maximum combined final blend weight for hgb_opt_* models.",
    )
    parser.add_argument(
        "--lgb-max-weight",
        type=float,
        default=None,
        help="Maximum combined final blend weight for lgb_* models.",
    )
    parser.add_argument(
        "--treebag-max-weight",
        type=float,
        default=None,
        help="Maximum combined final blend weight for etr_* bagged-tree models.",
    )
    parser.add_argument(
        "--blend-prior-strength",
        type=float,
        default=None,
        help="L2 penalty strength toward model-set starting weights during final blend search.",
    )
    parser.add_argument(
        "--hgb-params-path",
        type=Path,
        default=ROOT / "hgb_family_params.json",
        help="JSON file with tuned dynamic HGB params for hgb_optuna_family.",
    )
    parser.add_argument(
        "--empirical-curve-config-path",
        type=Path,
        default=ROOT / "empirical_curve_v4_config.json",
        help="JSON file with formula-only tuning knobs for empirical_curve_v4.",
    )
    parser.add_argument(
        "--allow-hgb-fallbacks",
        action="store_true",
        help="Allow hgb_opt_* models to use static fallback params when --hgb-params-path is missing.",
    )
    parser.add_argument(
        "--expect-feature-count",
        type=int,
        default=None,
        help="Fail if engineered feature count differs from this value.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "predictions.csv",
        help="Where to write the final predictions.csv.",
    )
    parser.add_argument(
        "--output-post-blank-path",
        type=Path,
        default=None,
        help="Optional one-column output for blank rows in the post-Q1 operational dataset.",
    )
    parser.add_argument(
        "--model-artifact-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for a reviewer artifact bundle: ensemble manifest, "
            "validated prediction snapshots, and run reports. The current pipeline "
            "rebuilds model estimators during --retrain; fast reviewer inference "
            "restores these validated snapshots from the bundle."
        ),
    )
    parser.add_argument(
        "--enable-2026-actual-adapter",
        action="store_true",
        help="Fit a post-Q1 adapter from 2026 April-May rows with observed generation.",
    )
    parser.add_argument(
        "--post-q1-dataset-path",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "3888f9f2-9bda-4b2c-94af-5562668bce86_test_dataset.csv",
        help="Official April-May 2026 operational dataset; observed rows fit the adapter and blank rows are forecast.",
    )
    parser.add_argument(
        "--actual-adapter-correction-cap",
        type=float,
        default=8.0,
        help="Absolute MW cap for raw post-Q1 adapter corrections before shrinkage.",
    )
    parser.add_argument(
        "--q1-actual-adapter-policy",
        choices=["off", "scalar", "conservative", "full"],
        default="off",
        help=(
            "How to transfer the April-May actual adapter back to Q1. "
            "off is safest and uses the adapter only for May18/post-Q1 output; "
            "scalar allows only a tiny guarded global bias/scale transfer; "
            "conservative allows only small affine/analog corrections; full repeats the old high-risk behavior."
        ),
    )
    parser.add_argument(
        "--actual-adapter-report-path",
        type=Path,
        default=ROOT / "actual_adapter_report.json",
        help="Where to write post-Q1 adapter diagnostics.",
    )
    parser.add_argument(
        "--enable-q1-energy-anchor",
        action="store_true",
        help="Rescale Q1 predictions to a robust Azov Q1 energy estimate from weather proxy and EL5 wind-output sanity data.",
    )
    parser.add_argument(
        "--q1-energy-anchor-source",
        choices=["azov_weather_proxy"],
        default="azov_weather_proxy",
        help="Energy-anchor method. azov_weather_proxy uses historical Azov/P_physics ratios.",
    )
    parser.add_argument(
        "--energy-anchor-report-path",
        type=Path,
        default=ROOT / "energy_anchor_report.json",
        help="Where to write Q1 energy-anchor diagnostics.",
    )
    parser.add_argument(
        "--skip-tune",
        action="store_true",
        help="Reuse physics/tune_results.json instead of running Optuna.",
    )
    parser.add_argument("--verbose-models", action="store_true", help="Print CatBoost logs.")
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
        help="Refetch external weather caches before training.",
    )
    parser.add_argument(
        "--include-nasa-power",
        action="store_true",
        help="Also fetch NASA POWER features.",
    )
    parser.add_argument(
        "--external-source-set",
        choices=EXTERNAL_SOURCE_SET_CHOICES,
        default="baseline",
        help="Optional extra external adapters beyond the baseline ERA5/Open-Meteo/Meteostat layer.",
    )
    parser.add_argument(
        "--external-source-timeout-sec",
        type=float,
        default=45.0,
        help="Timeout per optional external source request.",
    )
    parser.add_argument(
        "--external-cache-policy",
        choices=EXTERNAL_CACHE_POLICY_CHOICES,
        default="cache_first",
        help="Cache policy for optional external source adapters.",
    )
    parser.add_argument(
        "--require-extra-sources",
        action="store_true",
        help="Abort non-baseline external-source runs when no optional source columns are available.",
    )
    parser.add_argument(
        "--enable-source-interactions",
        action="store_true",
        help="Add targeted speed/direction/temperature/density interactions for external weather sources.",
    )
    parser.add_argument(
        "--enable-lag-residual-features",
        action="store_true",
        help="Add fold-safe train-only residual priors by month/hour and wind regime.",
    )
    parser.add_argument(
        "--enable-regime-prior-v2",
        action="store_true",
        help="Add shrinked fold-safe priors grouped by wind regime, month, hour, and wind-speed bin.",
    )
    parser.add_argument(
        "--enable-regime-models",
        action="store_true",
        help="Append HGB regime-specific direct/residual models to the active model family.",
    )
    parser.add_argument(
        "--enable-regime-v2",
        action="store_true",
        help="Use specialized per-regime HGB params and soft boundary blending for regime models.",
    )
    parser.add_argument(
        "--enable-regime-calibration",
        choices=["none", "affine", "isotonic"],
        default="none",
        help="OOF-trained post-blend calibration by wind regime.",
    )
    parser.add_argument(
        "--enable-hgb-quantile-sisters",
        action="store_true",
        help="Append HGB median-quantile direct/residual sister models as capped diversity members.",
    )
    parser.add_argument(
        "--quantile-hgb-max-weight",
        type=float,
        default=0.12,
        help="Maximum combined Phase-D weight for hgb_q50_* members when enabled.",
    )
    parser.add_argument(
        "--enable-direction-sector-features",
        action="store_true",
        help="Add 8 direction-sector x 5 wind-speed-bin interaction features.",
    )
    parser.add_argument(
        "--enable-weather-analog-residual",
        action="store_true",
        help="Add fold-safe weather analog residual median features.",
    )
    parser.add_argument(
        "--enable-weather-dynamics-features",
        action="store_true",
        help="Add target-free weather smoothing, trend, and source-disagreement features.",
    )
    parser.add_argument(
        "--enable-multi-regime-features",
        action="store_true",
        help="Add target-free speed x direction x shear/uncertainty regime context plus fold-safe priors.",
    )
    parser.add_argument(
        "--enable-multi-regime-experts",
        action="store_true",
        help="Make hgb_regime_* experts use a softened speed x direction regime split.",
    )
    parser.add_argument(
        "--enable-windfm-diagnostic",
        action="store_true",
        help="Join optional precomputed WindFM OOF/valid predictions as diagnostic features when present.",
    )
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
        choices=OPENMETEO_FORECAST_SET_CHOICES,
        default="core_open_forecast",
        help="Open-Meteo Historical Forecast source set.",
    )
    parser.add_argument(
        "--openmeteo-fill-policy",
        choices=OPENMETEO_FILL_POLICY_CHOICES,
        default="legacy",
        help="How to fill Open-Meteo forecast gaps. Use strict for leakage-safe source ablations.",
    )
    parser.add_argument(
        "--openmeteo-min-coverage",
        type=float,
        default=0.80,
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
        "--allow-missing-meteostat",
        action="store_true",
        help="Continue when requested Meteostat features are unavailable. Disabled by default for safe experiments.",
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
        default=DEFAULT_SO_UPS_RES_CACHE_PATH,
        help="CSV cache parsed from official SO UPS RES monthly reports.",
    )
    parser.add_argument(
        "--include-so-ups-valid-months",
        action="store_true",
        help="Diagnostic only: allow SO UPS RES monthly context after the train period.",
    )
    parser.add_argument(
        "--so-ups-res-policy",
        choices=SO_UPS_RES_POLICY_CHOICES,
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


def run_command(cmd: list[str]) -> None:
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_if_exists(src: Path, dst: Path) -> dict | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "path": str(dst),
        "sha256": _sha256(dst),
        "bytes": dst.stat().st_size,
    }


def write_model_artifact_bundle(
    artifact_dir: Path,
    *,
    args: argparse.Namespace,
    physics_cfg,
    feature_cols: list[str],
    final_iterations: dict[str, int],
    phase_d: dict,
    output_path: Path,
    post_blank_output_path: Path | None,
    stats: dict,
    post_blank_stats: dict | None,
    summary_path: Path,
    environment_path: Path,
) -> Path:
    """Write a compact reviewer bundle for fast, deterministic submission replay."""
    artifact_dir = _absolute(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    q1_copy = _copy_if_exists(output_path, artifact_dir / "predictions_q1.csv")
    may18_copy = (
        _copy_if_exists(post_blank_output_path, artifact_dir / "predictions_may18.csv")
        if post_blank_output_path is not None
        else None
    )
    report_copies = {
        "full_pipeline_results": _copy_if_exists(summary_path, artifact_dir / "reports" / summary_path.name),
        "environment_report": _copy_if_exists(environment_path, artifact_dir / "reports" / environment_path.name),
        "actual_adapter_report": _copy_if_exists(
            _absolute(args.actual_adapter_report_path),
            artifact_dir / "reports" / Path(args.actual_adapter_report_path).name,
        ),
        "energy_anchor_report": _copy_if_exists(
            _absolute(args.energy_anchor_report_path),
            artifact_dir / "reports" / Path(args.energy_anchor_report_path).name,
        ),
    }
    data_files = {
        "train": PROJECT_ROOT / "dataset" / "train_dataset.csv",
        "valid": PROJECT_ROOT / "dataset" / "valid_features.csv",
        "post_q1": _absolute(args.post_q1_dataset_path),
        "so_ups_res": _absolute(args.so_ups_res_cache_path),
    }
    data_manifest = {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "sha256": _sha256(path) if path.exists() else None,
            "bytes": path.stat().st_size if path.exists() else None,
        }
        for name, path in data_files.items()
    }
    manifest = {
        "artifact_version": 1,
        "artifact_kind": "validated_prediction_bundle",
        "note": (
            "This bundle stores the final ensemble configuration, blend weights, "
            "and validated prediction snapshots for one-command reviewer replay. "
            "Use bash run_solution.sh --retrain to rebuild estimators from source data."
        ),
        "candidate": args.candidate_name,
        "physics_config": asdict(physics_cfg),
        "model_set": args.model_set,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "final_iterations": final_iterations,
        "blend_method": phase_d.get("selected_method"),
        "blend_weights": phase_d.get("optimal_weights"),
        "prediction_stats": stats,
        "post_blank_prediction_stats": post_blank_stats,
        "outputs": {
            "q1": q1_copy,
            "may18": may18_copy,
        },
        "data": data_manifest,
        "reports": report_copies,
        "run_command": " ".join([sys.executable, "physics/run_full_pipeline.py", *sys.argv[1:]]),
        "created_at_unix": time.time(),
        "packages": {
            name: _package_version(name)
            for name in ["numpy", "pandas", "scikit-learn", "catboost", "optuna", "scipy", "meteostat", "lightgbm", "pypdf"]
        },
    }
    manifest_path = artifact_dir / "ensemble_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest_path


def environment_report(
    args: argparse.Namespace,
    feature_count: int | None = None,
    prediction_stats: dict | None = None,
) -> dict:
    packages = {
        name: _package_version(name)
        for name in ["numpy", "pandas", "scikit-learn", "catboost", "optuna", "scipy", "meteostat", "lightgbm"]
    }
    hgb_params_path = _absolute(args.hgb_params_path)
    empirical_curve_config_path = _absolute(args.empirical_curve_config_path)
    meteostat_requested = not args.skip_external_weather and not args.skip_meteostat
    return {
        "model_set": args.model_set,
        "physics_preset": args.physics_preset,
        "feature_count": feature_count,
        "expected_feature_count": args.expect_feature_count,
        "meteostat_requested": meteostat_requested,
        "allow_missing_meteostat": args.allow_missing_meteostat,
        "meteostat_cache_path": str(_absolute(args.meteostat_cache_path)),
        "openmeteo_historical_forecast": not args.skip_openmeteo_forecast,
        "openmeteo_forecast_set": args.openmeteo_forecast_set,
        "openmeteo_fill_policy": args.openmeteo_fill_policy,
        "openmeteo_min_coverage": args.openmeteo_min_coverage,
        "include_openmeteo_pressure": args.include_openmeteo_pressure,
        "external_source_set": args.external_source_set,
        "external_source_timeout_sec": args.external_source_timeout_sec,
        "external_cache_policy": args.external_cache_policy,
        "include_so_ups_res": args.include_so_ups_res,
        "so_ups_res_cache_path": str(_absolute(args.so_ups_res_cache_path)),
        "include_so_ups_valid_months": args.include_so_ups_valid_months,
        "so_ups_res_policy": args.so_ups_res_policy,
        "so_ups_res_exclude_months": args.so_ups_res_exclude_months,
        "enable_regime_v2": args.enable_regime_v2,
        "enable_regime_calibration": args.enable_regime_calibration,
        "enable_hgb_quantile_sisters": args.enable_hgb_quantile_sisters,
        "quantile_hgb_max_weight": args.quantile_hgb_max_weight,
        "enable_direction_sector_features": args.enable_direction_sector_features,
        "enable_weather_analog_residual": args.enable_weather_analog_residual,
        "enable_multi_regime_features": args.enable_multi_regime_features,
        "enable_multi_regime_experts": args.enable_multi_regime_experts,
        "enable_2026_actual_adapter": args.enable_2026_actual_adapter,
        "post_q1_dataset_path": str(_absolute(args.post_q1_dataset_path)),
        "actual_adapter_correction_cap": args.actual_adapter_correction_cap,
        "q1_actual_adapter_policy": args.q1_actual_adapter_policy,
        "enable_q1_energy_anchor": args.enable_q1_energy_anchor,
        "q1_energy_anchor_source": args.q1_energy_anchor_source,
        "output_post_blank_path": str(_absolute(args.output_post_blank_path)) if args.output_post_blank_path else None,
        "hgb_params_path": str(hgb_params_path),
        "hgb_params_exists": hgb_params_path.exists(),
        "empirical_curve_config_path": str(empirical_curve_config_path),
        "empirical_curve_config_exists": empirical_curve_config_path.exists(),
        "allow_hgb_fallbacks": args.allow_hgb_fallbacks,
        "packages": packages,
        "prediction_stats": prediction_stats,
    }


def write_environment_report(args: argparse.Namespace, feature_count: int | None = None, prediction_stats: dict | None = None) -> Path:
    report = environment_report(args, feature_count=feature_count, prediction_stats=prediction_stats)
    path = ROOT / "environment_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return path


def load_best_physics_config(preset: str):
    import predict

    if preset != "tune_results":
        return predict.physics_config_from_preset(preset), {
            "source": preset,
            "params": asdict(predict.physics_config_from_preset(preset)),
        }

    tune_path = ROOT / "tune_results.json"
    if not tune_path.exists():
        raise FileNotFoundError(f"Missing {tune_path}; run without --skip-tune first.")

    payload = json.loads(tune_path.read_text())
    params = payload.get("best", {}).get("params")
    if not isinstance(params, dict):
        raise ValueError(f"{tune_path} does not contain best.params")

    field_names = {field.name for field in fields(predict.PhysicsConfig)}
    normalized = asdict(predict.DEFAULT_PHYSICS)
    normalized.update({key: value for key, value in params.items() if key in field_names})
    normalized["hub_height"] = predict.HUB_HEIGHT_M
    cfg = predict.PhysicsConfig(**normalized)

    if params.get("hub_height") != predict.HUB_HEIGHT_M:
        print(
            f"  forcing hub_height={predict.HUB_HEIGHT_M:.1f} m "
            f"(tune file had {params.get('hub_height')})",
            flush=True,
        )
    return cfg, payload


def optimize_blend_weights_slsqp(oof: dict, y_true: dict, fold_weights: dict[int, float]) -> dict:
    import numpy as np
    from phase_d_optimize_weights import (
        MODEL_NAMES,
        default_starting_points,
        optimize_weights,
        weighted_platform_error,
    )
    import predict

    current_w = np.array([spec.weight for spec in predict.MODEL_SPECS], dtype=float)
    baseline_loss = weighted_platform_error(current_w, oof, y_true, fold_weights=fold_weights)
    runs = {}
    for name, x0 in default_starting_points(MODEL_NAMES, current_w).items():
        x0 = x0 / x0.sum()
        run = optimize_weights(oof, y_true, x0, fold_weights=fold_weights, prior_weights=current_w)
        runs[name] = run
        weights_str = ", ".join(f"{n}={w:.3f}" for n, w in zip(MODEL_NAMES, run["weights"]))
        print(
            f"  start={name:18s} loss={run['loss']:.4f}% "
            f"objective={run['objective']:.4f}  weights: {weights_str}"
        )
        if not run["success"]:
            print(f"    ! optimizer reported: {run['message']}")

    best_run = min(runs.values(), key=lambda r: r["objective"])
    consistency = max(abs(r["objective"] - best_run["objective"]) for r in runs.values())
    return {
        "method": "slsqp",
        "baseline_weighted_q1_cv": float(baseline_loss),
        "best_weighted_q1_cv": float(best_run["loss"]),
        "selection_objective": float(best_run["objective"]),
        "regularization_penalty": float(best_run["regularization_penalty"]),
        "improvement_abs": float(baseline_loss - best_run["loss"]),
        "current_weights": dict(zip(MODEL_NAMES, current_w.tolist())),
        "optimal_weights": dict(zip(MODEL_NAMES, best_run["weights"])),
        "constraint_violations": best_run["constraint_violations"],
        "runs_by_start": runs,
        "max_loss_spread": float(consistency),
    }


def optimize_blend_weights_ridge(oof: dict, y_true: dict, fold_weights: dict[int, float]) -> dict:
    import numpy as np
    from phase_d_optimize_weights import MODEL_NAMES, optimize_weights, stack_ridge, weighted_platform_error
    import predict

    current_w = np.array([spec.weight for spec in predict.MODEL_SPECS], dtype=float)
    baseline_loss = weighted_platform_error(current_w, oof, y_true, fold_weights=fold_weights)
    run = stack_ridge(oof, y_true, alpha=1.0, fold_weights=fold_weights, prior_weights=current_w)
    if any(value > 1e-8 for value in run["constraint_violations"].values()):
        refit = optimize_weights(
            oof,
            y_true,
            np.asarray(run["weights"], dtype=float),
            fold_weights=fold_weights,
            prior_weights=current_w,
        )
        run = {
            **run,
            "method": "ridge_constrained_refit",
            "ridge_weights": run["weights"],
            "weights": refit["weights"],
            "loss": refit["loss"],
            "objective": refit["objective"],
            "regularization_penalty": refit["regularization_penalty"],
            "constraint_violations": refit["constraint_violations"],
            "refit": refit,
        }
    weights_str = ", ".join(f"{n}={w:.3f}" for n, w in zip(MODEL_NAMES, run["weights"]))
    print(f"  ridge alpha=1.0  loss={run['loss']:.4f}% objective={run['objective']:.4f}  weights: {weights_str}")
    return {
        "method": run["method"],
        "alpha": run["alpha"],
        "baseline_weighted_q1_cv": float(baseline_loss),
        "best_weighted_q1_cv": float(run["loss"]),
        "selection_objective": float(run["objective"]),
        "regularization_penalty": float(run["regularization_penalty"]),
        "improvement_abs": float(baseline_loss - run["loss"]),
        "current_weights": dict(zip(MODEL_NAMES, current_w.tolist())),
        "optimal_weights": dict(zip(MODEL_NAMES, run["weights"])),
        "constraint_violations": run["constraint_violations"],
        "raw_coef": run["raw_coef"],
    }


def optimize_blend_weights_optuna(oof: dict, y_true: dict, n_trials: int, fold_weights: dict[int, float]) -> dict:
    import numpy as np
    import optuna
    from phase_d_optimize_weights import (
        MODEL_NAMES,
        blend_regularization,
        constraint_violations,
        model_weight_lower_bounds,
        weighted_platform_error,
    )
    import predict

    current_w = np.array([spec.weight for spec in predict.MODEL_SPECS], dtype=float)
    baseline_loss = weighted_platform_error(current_w, oof, y_true, fold_weights=fold_weights)
    lower_bounds = model_weight_lower_bounds()
    free_mass = 1.0 - float(lower_bounds.sum())
    if free_mass <= 0:
        raise ValueError("Blend lower bounds are too large for the number of blend models.")

    def weights_from_raw(raw: np.ndarray) -> np.ndarray:
        raw = np.maximum(raw, 1e-9)
        return lower_bounds + free_mass * raw / raw.sum()

    def objective(trial: optuna.Trial) -> float:
        raw = np.array(
            [trial.suggest_float(f"raw_{name}", 0.0, 1.0) for name in MODEL_NAMES],
            dtype=float,
        )
        weights = weights_from_raw(raw)
        pure_loss = weighted_platform_error(weights, oof, y_true, fold_weights=fold_weights)
        loss = pure_loss + blend_regularization(weights, current_w)
        trial.set_user_attr("weights", dict(zip(MODEL_NAMES, weights.tolist())))
        trial.set_user_attr("pure_loss", float(pure_loss))
        trial.set_user_attr("constraint_violations", constraint_violations(weights))
        return loss

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    optimal_weights = study.best_trial.user_attrs["weights"]
    best_pure_loss = float(study.best_trial.user_attrs["pure_loss"])
    return {
        "method": "optuna",
        "n_trials": int(n_trials),
        "baseline_weighted_q1_cv": float(baseline_loss),
        "best_weighted_q1_cv": best_pure_loss,
        "selection_objective": float(study.best_value),
        "regularization_penalty": float(study.best_value - best_pure_loss),
        "improvement_abs": float(baseline_loss - best_pure_loss),
        "current_weights": dict(zip(MODEL_NAMES, current_w.tolist())),
        "optimal_weights": {key: float(value) for key, value in optimal_weights.items()},
        "constraint_violations": study.best_trial.user_attrs["constraint_violations"],
        "best_params": study.best_params,
    }


def optimize_blend_weights(
    oof: dict,
    y_true: dict,
    method: str,
    blend_trials: int,
    fold_weights: dict[int, float],
) -> dict:
    candidates = []
    if method in {"slsqp", "both", "all"}:
        print("  SLSQP blend search")
        candidates.append(optimize_blend_weights_slsqp(oof, y_true, fold_weights))
    if method in {"optuna", "both", "all"}:
        print(f"  Optuna blend search ({blend_trials} trials)")
        candidates.append(optimize_blend_weights_optuna(oof, y_true, blend_trials, fold_weights))
    if method == "all":
        print("  Ridge stacking (non-negative)")
        candidates.append(optimize_blend_weights_ridge(oof, y_true, fold_weights))

    selected = min(candidates, key=lambda r: r.get("selection_objective", r["best_weighted_q1_cv"]))
    return {
        "selected_method": selected["method"],
        "baseline_weighted_q1_cv": selected["baseline_weighted_q1_cv"],
        "best_weighted_q1_cv": selected["best_weighted_q1_cv"],
        "selection_objective": selected.get("selection_objective", selected["best_weighted_q1_cv"]),
        "regularization_penalty": selected.get("regularization_penalty", 0.0),
        "improvement_abs": selected["improvement_abs"],
        "current_weights": selected["current_weights"],
        "optimal_weights": selected["optimal_weights"],
        "constraint_violations": selected.get("constraint_violations", {}),
        "candidates": candidates,
        "max_loss_spread": selected.get("max_loss_spread", 0.0),
    }


def main() -> None:
    args = parse_args()
    output_path = _absolute(args.output_path)
    started = time.time()

    if args.physics_preset == "tune_results" and not args.skip_tune:
        run_command(
            [
                sys.executable,
                str(ROOT / "tune_physics.py"),
                "--trials",
                str(args.trials),
                "--search-space",
                args.tune_search_space,
            ]
        )
    elif args.physics_preset == "tune_results":
        print("Skipping Optuna; reusing physics/tune_results.json")
    else:
        print(f"Skipping Optuna; using checked-in physics preset {args.physics_preset!r}")

    import numpy as np
    import predict

    predict.set_seed(predict.SEED)
    predict.set_catboost_runtime(args.catboost_task_type, args.catboost_devices)
    auto_source_interactions = args.enable_source_interactions
    auto_lag_residual_features = args.enable_lag_residual_features
    auto_regime_prior_v2 = args.enable_regime_prior_v2
    auto_regime_models = args.enable_regime_models
    auto_regime_v2 = args.enable_regime_v2
    auto_hgb_quantile_sisters = args.enable_hgb_quantile_sisters
    auto_direction_sector_features = args.enable_direction_sector_features
    auto_weather_analog_residual = args.enable_weather_analog_residual
    auto_weather_dynamics_features = args.enable_weather_dynamics_features
    auto_multi_regime_features = args.enable_multi_regime_features or args.enable_multi_regime_experts
    auto_multi_regime_experts = args.enable_multi_regime_experts
    hgb_params_path = _absolute(args.hgb_params_path)
    predict.set_hgb_dynamic_params_path(hgb_params_path)
    empirical_curve_config_path = _absolute(args.empirical_curve_config_path)
    predict.set_empirical_curve_config_path(empirical_curve_config_path)
    include_physics_variants = (
        args.include_physics_variants
        or args.model_set.startswith("hgb_")
        or args.model_set in {
            "empirical_curve_v3",
            "empirical_curve_v4",
            "empirical_curve_v4_guarded",
            "lgb_probe",
            "forest_probe",
            "forest_only_probe",
            "all_family_probe",
        }
    )

    if args.model_set == "hgb_optuna_family" and not hgb_params_path.exists() and not args.allow_hgb_fallbacks:
        raise FileNotFoundError(
            f"{hgb_params_path} is required for --model-set hgb_optuna_family. "
            "Run physics/tune_hgb_family.py first, or pass --allow-hgb-fallbacks for an explicit degraded run."
        )
    if args.model_set == "lgb_probe" and _package_version("lightgbm") is None:
        raise RuntimeError(
            "LightGBM model set selected, but the 'lightgbm' package is not installed. "
            "Install dependencies with: pip install -r physics/requirements.txt"
        )

    import phase_d_optimize_weights as phase_d_module
    from phase_d_optimize_weights import (
        constraint_summary,
        format_constraint_summary,
        resolve_constraint_profile,
        resolve_fold_weights,
        set_blend_constraints,
        set_model_names,
        set_quantile_hgb_max_weight,
        set_weight_lower,
    )

    set_model_names(args.model_set)
    predict.configure_v8_experiments(
        source_interactions=auto_source_interactions,
        lag_residual_features=auto_lag_residual_features,
        regime_prior_v2=auto_regime_prior_v2,
        regime_models=auto_regime_models,
        regime_v2=auto_regime_v2,
        windfm_diagnostic=args.enable_windfm_diagnostic,
        hgb_quantile_sisters=auto_hgb_quantile_sisters,
        direction_sector_features=auto_direction_sector_features,
        weather_analog_residual=auto_weather_analog_residual,
        weather_dynamics_features=auto_weather_dynamics_features,
        multi_regime_features=auto_multi_regime_features,
        multi_regime_experts=auto_multi_regime_experts,
        regime_calibration=args.enable_regime_calibration,
    )
    phase_d_module.MODEL_NAMES = [spec.name for spec in predict.MODEL_SPECS]
    set_weight_lower(args.blend_weight_lower)

    from phase_c_validate import run_q1_cv_with_oof, save_oof

    stable_min, optuna_max, lgb_max, treebag_max, prior_strength = resolve_constraint_profile(
        args.blend_constraint_profile,
        args.stable_hgb_min_weight,
        args.optuna_hgb_max_weight,
        args.lgb_max_weight,
        args.treebag_max_weight,
        args.blend_prior_strength,
    )
    set_blend_constraints(stable_min, optuna_max, lgb_max, treebag_max, prior_strength)
    set_quantile_hgb_max_weight(args.quantile_hgb_max_weight if auto_hgb_quantile_sisters else 1.0)
    fold_weights = resolve_fold_weights(args.fold_weights)
    physics_cfg, tune_payload = load_best_physics_config(args.physics_preset)
    print("\nSelected PhysicsConfig:")
    print(json.dumps(asdict(physics_cfg), indent=2, ensure_ascii=False))
    print(f"Model set: {predict.ACTIVE_MODEL_SET} ({len(predict.MODEL_SPECS)} models)")
    print(f"Fold weights for final blend: {args.fold_weights} {fold_weights}")
    print(f"Blend weight lower bound: {args.blend_weight_lower:.4f}")
    print(f"Blend constraints: {format_constraint_summary()}")
    print(f"HGB params path: {predict.HGB_DYNAMIC_PARAMS_PATH}")
    if args.model_set in {"empirical_curve_v4", "empirical_curve_v4_guarded"}:
        print(f"Empirical curve V4 config path: {predict.EMPIRICAL_CURVE_CONFIG_PATH}")
        print(f"Empirical curve V4 config: {json.dumps(asdict(predict.ACTIVE_EMPIRICAL_CURVE_V4_CONFIG), ensure_ascii=False)}")
    if args.model_set.startswith("hgb_optuna") and not hgb_params_path.exists():
        print("WARNING: tuned HGB params file is missing; hgb_opt_* models will use static fallbacks.")
    print(
        "\nExternal weather: "
        + ("disabled" if args.skip_external_weather else f"enabled ({args.external_weather_cache_dir})")
    )
    print(
        "Meteostat weather: "
        + ("disabled" if args.skip_meteostat else f"enabled ({args.meteostat_cache_path})")
        + (" partial-ok" if args.allow_partial_meteostat and not args.skip_meteostat else "")
    )
    print(
        "Open-Meteo historical forecast: "
        + (
            "disabled"
            if args.skip_openmeteo_forecast
            else (
                f"enabled ({args.openmeteo_forecast_set}, "
                f"fill={args.openmeteo_fill_policy}, "
                f"pressure={'yes' if predict.openmeteo_pressure_enabled(args.openmeteo_forecast_set, args.include_openmeteo_pressure) else 'no'})"
            )
        )
    )
    print(
        "SO UPS RES monthly context: "
        + ("enabled" if args.include_so_ups_res else "disabled")
        + (f" ({args.so_ups_res_cache_path})" if args.include_so_ups_res else "")
        + (f" policy={args.so_ups_res_policy}" if args.include_so_ups_res else "")
    )
    print(
        "External source set: "
        + (
            "disabled"
            if args.skip_external_weather
            else f"{args.external_source_set} (cache={args.external_cache_policy}, timeout={args.external_source_timeout_sec:g}s)"
        )
        + (" require-extra" if args.require_extra_sources and not args.skip_external_weather else "")
    )
    print(
        "Wind-tech features: "
        + ", ".join(
            name
            for name, enabled in [
                ("source_interactions", auto_source_interactions),
                ("lag_residual_priors", auto_lag_residual_features),
                ("regime_prior_v2", auto_regime_prior_v2),
                ("regime_models", auto_regime_models),
                ("regime_v2", auto_regime_v2),
                (f"regime_calibration={args.enable_regime_calibration}", args.enable_regime_calibration != "none"),
                ("hgb_quantile_sisters", auto_hgb_quantile_sisters),
                ("direction_sector_features", auto_direction_sector_features),
                ("weather_analog_residual", auto_weather_analog_residual),
                ("weather_dynamics_features", auto_weather_dynamics_features),
                ("multi_regime_features", auto_multi_regime_features),
                ("multi_regime_experts", auto_multi_regime_experts),
                (f"actual_2026_adapter(q1={args.q1_actual_adapter_policy})", args.enable_2026_actual_adapter),
                (f"q1_energy_anchor={args.q1_energy_anchor_source}", args.enable_q1_energy_anchor),
                ("windfm_diagnostic", args.enable_windfm_diagnostic),
            ]
            if enabled
        )
        if any(
            [
                auto_source_interactions,
                auto_lag_residual_features,
                auto_regime_prior_v2,
                auto_regime_models,
                auto_regime_v2,
                args.enable_regime_calibration != "none",
                auto_hgb_quantile_sisters,
                auto_direction_sector_features,
                auto_weather_analog_residual,
                auto_weather_dynamics_features,
                auto_multi_regime_features,
                auto_multi_regime_experts,
                args.enable_2026_actual_adapter,
                args.enable_q1_energy_anchor,
                args.enable_windfm_diagnostic,
            ]
        )
        else "Wind-tech features: disabled"
    )
    print(
        "Meteostat derived features: "
        + ("enabled" if args.include_meteostat_derived else "disabled")
    )
    print(
        "Physics variant features: "
        + ("enabled" if include_physics_variants else "disabled")
    )
    print(
        f"CatBoost runtime: {predict.CATBOOST_TASK_TYPE}"
        + (f" devices={predict.CATBOOST_DEVICES}" if predict.CATBOOST_DEVICES else "")
    )

    print("\nLoading features with selected physics config...")
    require_meteostat = not args.skip_external_weather and not args.skip_meteostat and not args.allow_missing_meteostat
    train, valid, valid_original_order, feature_cols = predict.load_and_prepare(
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
        require_meteostat=require_meteostat,
        include_meteostat_derived=args.include_meteostat_derived,
        include_so_ups_res=args.include_so_ups_res,
        so_ups_res_cache_path=_absolute(args.so_ups_res_cache_path),
        include_so_ups_valid_months=args.include_so_ups_valid_months,
        so_ups_res_policy=args.so_ups_res_policy,
        so_ups_res_exclude_months=args.so_ups_res_exclude_months,
        include_physics_variants=include_physics_variants,
        keep_empty_feature_columns=args.keep_empty_feature_columns,
    )
    print(f"  train rows: {len(train)} | valid rows: {len(valid)} | features: {len(feature_cols)}")
    if args.expect_feature_count is not None and len(feature_cols) != args.expect_feature_count:
        write_environment_report(args, feature_count=len(feature_cols))
        raise ValueError(
            f"Feature count guard failed: expected {args.expect_feature_count}, got {len(feature_cols)}. "
            "This usually means weather/source features differ from the intended experiment."
        )
    post_q1_bundle = None
    if args.enable_2026_actual_adapter or args.output_post_blank_path is not None:
        post_path = _absolute(args.post_q1_dataset_path)
        if not post_path.exists():
            raise FileNotFoundError(f"Missing post-Q1 operational dataset: {post_path}")
        print(f"\nLoading post-Q1 operational dataset: {post_path}")
        post_q1_bundle = predict.load_post_q1_prediction_frame(
            post_path,
            feature_cols,
            physics_cfg=physics_cfg,
            include_so_ups_res=args.include_so_ups_res,
            so_ups_res_cache_path=_absolute(args.so_ups_res_cache_path),
            include_so_ups_valid_months=args.include_so_ups_valid_months,
            so_ups_res_policy=args.so_ups_res_policy,
            so_ups_res_exclude_months=args.so_ups_res_exclude_months,
            include_physics_variants=include_physics_variants,
        )
        known_rows = int(post_q1_bundle.target.notna().sum())
        blank_rows = int(post_q1_bundle.target.isna().sum())
        print(
            f"  post-Q1 rows: {len(post_q1_bundle.frame)} | "
            f"observed={known_rows} | blank={blank_rows}",
            flush=True,
        )

    print("\nPhase C: full ensemble Q1 CV + OOF cache")
    phase_c_start = time.time()
    scores, oof, y_true, best_iters, validator_weights = run_q1_cv_with_oof(train, feature_cols)
    oof_path = save_oof(args.candidate_name, oof, y_true)
    final_iterations = predict.derive_final_iterations(best_iters)
    phase_c_record = {
        "name": args.candidate_name,
        "config": asdict(physics_cfg),
        "weighted_q1_cv": float(scores["weighted"]),
        "static_weighted_q1_cv": float(scores["static_weighted"]),
        "validator_improvement_abs": float(scores["static_weighted"] - scores["weighted"]),
        "per_year_cv": {str(y): float(scores[y]) for y in (2023, 2024, 2025)},
        "final_iterations": final_iterations,
        "validator_weights": {
            str(key): {model: float(weight) for model, weight in weights.items()}
            for key, weights in validator_weights.items()
        },
        "elapsed_sec": time.time() - phase_c_start,
        "oof_path": str(oof_path.relative_to(ROOT)),
    }
    (ROOT / "phase_c_results.json").write_text(json.dumps([phase_c_record], indent=2, ensure_ascii=False))
    print(f"  saved OOF to {oof_path}")
    print(f"  final iterations: {final_iterations}")

    print("\nPhase D: optimize blend weights")
    phase_d = optimize_blend_weights(oof, y_true, args.blend_method, args.blend_trials, fold_weights)
    phase_d["candidate"] = args.candidate_name
    phase_d["model_set"] = args.model_set
    phase_d["fold_weight_scheme"] = args.fold_weights
    phase_d["fold_weights"] = fold_weights
    phase_d["blend_weight_lower"] = args.blend_weight_lower
    phase_d["blend_constraint_profile"] = args.blend_constraint_profile
    phase_d["blend_constraints"] = constraint_summary()
    phase_d["quantile_hgb_max_weight"] = args.quantile_hgb_max_weight if auto_hgb_quantile_sisters else 1.0
    (ROOT / "phase_d_results.json").write_text(json.dumps(phase_d, indent=2, ensure_ascii=False))
    print(f"  best weighted Q1 CV: {phase_d['best_weighted_q1_cv']:.4f}%")
    print(f"  selected method: {phase_d['selected_method']}")
    print(f"  optimal weights: {phase_d['optimal_weights']}")
    if phase_d["max_loss_spread"] > 0.05:
        print("  WARNING: optimizer starts diverged; inspect phase_d_results.json")

    regime_oof_diagnostics = predict.compute_regime_oof_diagnostics(
        train,
        oof,
        y_true,
        phase_d["optimal_weights"],
        fold_weights,
    )
    regime_calibration = predict.fit_regime_prediction_calibrator(
        train,
        oof,
        y_true,
        phase_d["optimal_weights"],
        fold_weights,
        mode=args.enable_regime_calibration,
    )
    print("\nRegime diagnostics")
    for regime, payload in regime_oof_diagnostics["by_regime"].items():
        print(
            f"  regime={regime} rows={payload['rows']} "
            f"mae={payload['mae_mw']:.3f}MW "
            f"err={payload['platform_error_percent']:.4f}% "
            f"bias={payload['residual_bias_mw']:+.3f}MW"
        )
    if args.enable_regime_calibration != "none":
        print(
            f"  calibration={args.enable_regime_calibration} "
            f"OOF {regime_calibration['base_weighted_q1_cv']:.4f}% -> "
            f"{regime_calibration['calibrated_weighted_q1_cv']:.4f}% "
            f"(delta={regime_calibration['improvement_abs']:+.4f}%)"
        )

    print("\nFinal fit with optimized weights")
    extra_pred_frames = {"post_q1": post_q1_bundle.frame} if post_q1_bundle is not None else None
    final_result = predict.train_final_predict(
        train,
        valid,
        feature_cols,
        verbose=args.verbose_models,
        final_iterations=final_iterations,
        blend_weights=phase_d["optimal_weights"],
        extra_pred_frames=extra_pred_frames,
    )
    if post_q1_bundle is not None:
        final_pred, extra_predictions = final_result
        post_q1_pred = np.asarray(extra_predictions["post_q1"], dtype=float)
    else:
        final_pred = final_result
        post_q1_pred = None
    final_pred = np.asarray(final_pred, dtype=float)
    actual_adapter_report = {"enabled": False}
    actual_adapter_application = None
    post_actual_adapter_application = None
    q1_actual_adapter_report = {"enabled": False, "policy": args.q1_actual_adapter_policy}
    if args.enable_regime_calibration != "none":
        final_pred = predict.apply_regime_prediction_calibrator(final_pred, valid, regime_calibration)
        if post_q1_bundle is not None and post_q1_pred is not None:
            post_q1_pred = predict.apply_regime_prediction_calibrator(post_q1_pred, post_q1_bundle.frame, regime_calibration)
    if args.enable_2026_actual_adapter:
        if post_q1_bundle is None or post_q1_pred is None:
            raise RuntimeError("--enable-2026-actual-adapter requires a post-Q1 prediction frame.")
        post_q1_base_pred = post_q1_pred.copy()
        post_adapter, actual_adapter_report = predict.fit_post_q1_actual_adapter(
            post_q1_bundle.frame,
            post_q1_pred,
            feature_cols,
            correction_cap=args.actual_adapter_correction_cap,
        )
        post_q1_pred, post_actual_adapter_application = predict.apply_post_q1_actual_adapter(
            post_q1_pred,
            post_q1_bundle.frame,
            post_adapter,
        )
        if args.q1_actual_adapter_policy == "full":
            final_pred, actual_adapter_application = predict.apply_post_q1_actual_adapter(final_pred, valid, post_adapter)
            q1_actual_adapter_report = {
                "enabled": True,
                "policy": "full",
                "source": "same adapter selected for April-May post-Q1 validation",
            }
        elif args.q1_actual_adapter_policy == "scalar":
            final_pred, q1_actual_adapter_report = predict.apply_q1_scalar_actual_transfer(
                final_pred,
                valid,
                post_q1_bundle.frame,
                post_q1_base_pred,
            )
            actual_adapter_application = q1_actual_adapter_report.get("application")
        elif args.q1_actual_adapter_policy == "conservative":
            q1_adapter, q1_actual_adapter_report = predict.fit_post_q1_actual_adapter(
                post_q1_bundle.frame,
                post_q1_base_pred,
                feature_cols,
                correction_cap=min(args.actual_adapter_correction_cap, 2.0),
                allowed_kinds=("affine_by_regime", "weather_analog"),
                shrink_values=(0.05, 0.10, 0.15),
                min_improvement_mw=0.50,
            )
            q1_actual_adapter_report["policy"] = "conservative"
            if q1_actual_adapter_report.get("enabled"):
                final_pred, actual_adapter_application = predict.apply_post_q1_actual_adapter(final_pred, valid, q1_adapter)
        else:
            q1_actual_adapter_report = {
                "enabled": False,
                "policy": "off",
                "reason": "April-May adapter is used only for post-Q1/May18 output by default.",
            }
        actual_adapter_report["q1_application"] = actual_adapter_application
        actual_adapter_report["q1_adapter"] = q1_actual_adapter_report
        actual_adapter_report["post_q1_application"] = post_actual_adapter_application
        actual_adapter_report["source_path"] = post_q1_bundle.source_path
        _absolute(args.actual_adapter_report_path).write_text(
            json.dumps(actual_adapter_report, indent=2, ensure_ascii=False)
        )
        print(
            "\n2026 post-Q1 adapter: "
            + (
                f"{actual_adapter_report.get('selected_kind')} "
                f"MAE {actual_adapter_report.get('base_validation_mae_mw', float('nan')):.3f} -> "
                f"{actual_adapter_report.get('selected_validation_mae_mw', float('nan')):.3f} MW "
                f"(shrink={actual_adapter_report.get('selected_shrink')})"
                if actual_adapter_report.get("enabled")
                else f"disabled ({actual_adapter_report.get('reason', 'no improvement')})"
            ),
            flush=True,
        )
        print(
            "Q1 actual-adapter transfer: "
            + (
                f"{args.q1_actual_adapter_policy} applied {actual_adapter_application}"
                if q1_actual_adapter_report.get("enabled")
                else f"{args.q1_actual_adapter_policy} skipped ({q1_actual_adapter_report.get('reason', 'no accepted Q1 transfer')})"
            ),
            flush=True,
        )
    energy_anchor_report = {"enabled": False}
    if args.enable_q1_energy_anchor:
        final_pred, energy_anchor_report = predict.apply_q1_energy_anchor(
            final_pred,
            train,
            valid,
            source=args.q1_energy_anchor_source,
        )
        _absolute(args.energy_anchor_report_path).write_text(
            json.dumps(energy_anchor_report, indent=2, ensure_ascii=False)
        )
        correction = energy_anchor_report["correction"]
        print(
            "\nQ1 energy anchor: "
            f"{correction['before_gwh']:.3f} -> {correction['after_gwh']:.3f} GWh "
            f"(target={correction['target_gwh']:.3f}, "
            f"mean_delta={correction['mean_correction_mw']:+.3f}MW, "
            f"max_abs_delta={correction['max_abs_correction_mw']:.3f}MW)",
            flush=True,
        )
    predict.write_predictions(final_pred, valid, valid_original_order, output_path)
    stats = predict.validate_predictions(output_path, predict.VALID_PATH)
    post_blank_stats = None
    if args.output_post_blank_path is not None:
        if post_q1_bundle is None or post_q1_pred is None:
            raise RuntimeError("--output-post-blank-path requires --post-q1-dataset-path to be loadable.")
        blank_mask = post_q1_bundle.target.isna().to_numpy()
        blank_frame = post_q1_bundle.frame.loc[blank_mask].reset_index(drop=True)
        blank_pred = post_q1_pred[blank_mask]
        post_blank_output_path = _absolute(args.output_post_blank_path)
        predict.write_prediction_values_by_datetime(
            blank_pred,
            blank_frame,
            post_q1_bundle.blank_original_order,
            post_blank_output_path,
        )
        post_blank_stats = predict.validate_prediction_file(post_blank_output_path, expected_rows=len(blank_frame))
        print(
            f"\nMay18 forecast: {post_blank_output_path} "
            f"rows={post_blank_stats['rows']} mean={post_blank_stats['mean']:.3f} "
            f"min={post_blank_stats['min']:.3f} max={post_blank_stats['max']:.3f}",
            flush=True,
        )

    summary = {
        "candidate": args.candidate_name,
        "elapsed_sec": time.time() - started,
        "trials": args.trials if args.physics_preset == "tune_results" and not args.skip_tune else None,
        "tune_search_space": args.tune_search_space,
        "physics_preset": args.physics_preset,
        "model_set": args.model_set,
        "fold_weight_scheme": args.fold_weights,
        "blend_weight_lower": args.blend_weight_lower,
        "blend_constraint_profile": args.blend_constraint_profile,
        "blend_constraints": constraint_summary(),
        "hgb_params_path": str(hgb_params_path),
        "empirical_curve_config_path": str(empirical_curve_config_path),
        "allow_hgb_fallbacks": args.allow_hgb_fallbacks,
        "expect_feature_count": args.expect_feature_count,
        "allow_missing_meteostat": args.allow_missing_meteostat,
        "include_meteostat_derived": args.include_meteostat_derived,
        "openmeteo_historical_forecast": not args.skip_openmeteo_forecast,
        "openmeteo_forecast_set": args.openmeteo_forecast_set,
        "openmeteo_fill_policy": args.openmeteo_fill_policy,
        "openmeteo_min_coverage": args.openmeteo_min_coverage,
        "include_openmeteo_pressure": args.include_openmeteo_pressure,
        "external_source_set": args.external_source_set,
        "external_source_timeout_sec": args.external_source_timeout_sec,
        "external_cache_policy": args.external_cache_policy,
        "include_so_ups_res": args.include_so_ups_res,
        "so_ups_res_cache_path": str(_absolute(args.so_ups_res_cache_path)),
        "include_so_ups_valid_months": args.include_so_ups_valid_months,
        "so_ups_res_policy": args.so_ups_res_policy,
        "so_ups_res_exclude_months": args.so_ups_res_exclude_months,
        "include_physics_variants": include_physics_variants,
        "enable_regime_v2": auto_regime_v2,
        "enable_regime_calibration": args.enable_regime_calibration,
        "enable_hgb_quantile_sisters": auto_hgb_quantile_sisters,
        "quantile_hgb_max_weight": args.quantile_hgb_max_weight if auto_hgb_quantile_sisters else 1.0,
        "enable_direction_sector_features": auto_direction_sector_features,
        "enable_weather_analog_residual": auto_weather_analog_residual,
        "enable_weather_dynamics_features": auto_weather_dynamics_features,
        "enable_multi_regime_features": auto_multi_regime_features,
        "enable_multi_regime_experts": auto_multi_regime_experts,
        "enable_2026_actual_adapter": args.enable_2026_actual_adapter,
        "post_q1_dataset_path": str(_absolute(args.post_q1_dataset_path)),
        "actual_adapter_correction_cap": args.actual_adapter_correction_cap,
        "q1_actual_adapter_policy": args.q1_actual_adapter_policy,
        "enable_q1_energy_anchor": args.enable_q1_energy_anchor,
        "q1_energy_anchor_source": args.q1_energy_anchor_source,
        "output_post_blank_path": str(_absolute(args.output_post_blank_path)) if args.output_post_blank_path else None,
        "model_artifact_dir": str(_absolute(args.model_artifact_dir)) if args.model_artifact_dir else None,
        "blend_method": args.blend_method,
        "blend_trials": args.blend_trials,
        "catboost_task_type": args.catboost_task_type,
        "catboost_devices": args.catboost_devices,
        "allow_partial_meteostat": args.allow_partial_meteostat,
        "output_path": str(output_path),
        "tune_best": tune_payload.get("best"),
        "phase_c": phase_c_record,
        "phase_d": phase_d,
        "regime_oof_diagnostics": regime_oof_diagnostics,
        "regime_calibration": regime_calibration,
        "actual_adapter": actual_adapter_report,
        "energy_anchor": energy_anchor_report,
        "prediction_stats": stats,
        "post_blank_prediction_stats": post_blank_stats,
        "environment_report_path": str(ROOT / "environment_report.json"),
    }
    summary_path = ROOT / "full_pipeline_results.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    env_report_path = write_environment_report(args, feature_count=len(feature_cols), prediction_stats=stats)
    artifact_manifest_path = None
    if args.model_artifact_dir is not None:
        artifact_manifest_path = write_model_artifact_bundle(
            args.model_artifact_dir,
            args=args,
            physics_cfg=physics_cfg,
            feature_cols=feature_cols,
            final_iterations=final_iterations,
            phase_d=phase_d,
            output_path=output_path,
            post_blank_output_path=_absolute(args.output_post_blank_path) if args.output_post_blank_path else None,
            stats=stats,
            post_blank_stats=post_blank_stats,
            summary_path=summary_path,
            environment_path=env_report_path,
        )

    print("\nDone.")
    print(f"  predictions: {output_path}")
    print(
        "  rows={rows} min={min:.3f} max={max:.3f} mean={mean:.3f} zeros={zeros} sha256={sha256}".format(
            **stats
        )
    )
    print(f"  summary: {ROOT / 'full_pipeline_results.json'}")
    print(f"  environment: {env_report_path}")
    if artifact_manifest_path is not None:
        print(f"  artifact bundle: {artifact_manifest_path}")


if __name__ == "__main__":
    main()
