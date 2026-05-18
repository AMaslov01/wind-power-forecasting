"""Run Open-Meteo forecast source ablations through the full pipeline.

The runner is intentionally thin: it delegates training to run_full_pipeline.py
so every ablation uses the same CV, OOF cache, blend optimizer, and final-fit
path as the main candidate.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

import predict


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
VALID_PATH = PROJECT_ROOT / "dataset" / "valid_features.csv"
DATETIME_COL = "METEOFORECASTHOUR_OPENM_Datetime"
TARGET = "Выработка. Результирующий расчет"
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
PHYSICS_PRESET_CHOICES = ["public_best", "legacy_tuned", "manufacturer", "tune_results"]
OPENMETEO_FORECAST_SETS = [
    "gfs_only",
    "icon_only",
    "ecmwf_only",
    "core_open_forecast",
    "core_pressure_forecast",
    "pressure_level_forecast",
    "core_plus_icon_eu",
    "core_plus_cma",
    "core_plus_arpege",
    "expanded_open_forecast",
    "expanded_pressure_forecast",
]
OPENMETEO_PRESSURE_SETS = {
    "core_pressure_forecast",
    "pressure_level_forecast",
    "expanded_pressure_forecast",
}
OPENMETEO_FILL_POLICIES = ["legacy", "strict"]
OPENMETEO_COVERAGE_MIN_DEFAULT = 0.80
DEFAULT_SETS = [
    "gfs_only",
    "icon_only",
    "ecmwf_only",
    "core_open_forecast",
    "core_plus_icon_eu",
    "core_plus_cma",
    "core_plus_arpege",
    "expanded_open_forecast",
    "pressure_level_forecast",
    "core_pressure_forecast",
    "expanded_pressure_forecast",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hgb_family source ablation matrix.")
    parser.add_argument(
        "--sets",
        default=",".join(DEFAULT_SETS),
        help="Comma-separated Open-Meteo forecast sets to run.",
    )
    parser.add_argument("--model-set", default="hgb_family", choices=MODEL_SET_CHOICES)
    parser.add_argument("--physics-preset", default="public_best", choices=PHYSICS_PRESET_CHOICES)
    parser.add_argument("--fold-weights", default="recency", choices=["recency", "valid_similarity", "raw_similarity"])
    parser.add_argument("--blend-method", default="all", choices=["slsqp", "optuna", "both", "all"])
    parser.add_argument("--blend-trials", type=int, default=1500)
    parser.add_argument("--openmeteo-fill-policy", default="legacy", choices=OPENMETEO_FILL_POLICIES)
    parser.add_argument("--openmeteo-min-coverage", type=float, default=OPENMETEO_COVERAGE_MIN_DEFAULT)
    parser.add_argument("--include-openmeteo-pressure", action="store_true")
    parser.add_argument("--refresh-external-weather", action="store_true")
    parser.add_argument("--include-nasa-power", action="store_true")
    parser.add_argument("--external-source-set", default="baseline", choices=sorted(predict.EXTERNAL_SOURCE_SETS))
    parser.add_argument("--external-source-timeout-sec", type=float, default=45.0)
    parser.add_argument("--external-cache-policy", default="cache_first", choices=["cache_first", "refresh", "offline"])
    parser.add_argument("--skip-meteostat", action="store_true")
    parser.add_argument("--allow-missing-meteostat", action="store_true")
    parser.add_argument("--include-meteostat-derived", action="store_true")
    parser.add_argument("--include-so-ups-res", action="store_true")
    parser.add_argument("--include-so-ups-valid-months", action="store_true")
    parser.add_argument("--so-ups-res-policy", default="legacy_best_gap", choices=["legacy_best_gap", "clean_48", "custom_exclude"])
    parser.add_argument("--so-ups-res-exclude-months", default="")
    parser.add_argument("--catboost-task-type", default="CPU", choices=["CPU", "GPU"])
    parser.add_argument("--catboost-devices", default=None)
    parser.add_argument("--candidate-prefix", default="source_ablation")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra raw argument passed to run_full_pipeline.py.")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=ROOT / "source_ablation_results.json",
        help="Where to write the ablation summary JSON.",
    )
    return parser.parse_args()


def _prediction_profile(prediction_path: Path) -> dict:
    pred = pd.read_csv(prediction_path)
    values = pd.to_numeric(pred[TARGET], errors="raise")
    profile: dict[str, object] = {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "zeros": int((values == 0.0).sum()),
    }
    if not VALID_PATH.exists():
        profile["profile_warning"] = f"valid feature file not found: {VALID_PATH}"
        return profile

    valid = pd.read_csv(VALID_PATH, parse_dates=[DATETIME_COL])
    month_frame = pd.DataFrame(
        {
            "month": valid[DATETIME_COL].dt.to_period("M").astype(str),
            "prediction": values,
        }
    )
    profile["month_mean"] = {
        month: float(group["prediction"].mean())
        for month, group in month_frame.groupby("month", sort=True)
    }
    try:
        pbin = pd.qcut(valid["wind_speed_80m"], 9, labels=[f"pbin{i:02d}" for i in range(9)], duplicates="drop")
        pbin_frame = pd.DataFrame({"pbin": pbin.astype(str), "prediction": values})
        profile["pbin_mean"] = {
            pbin_name: float(group["prediction"].mean())
            for pbin_name, group in pbin_frame.groupby("pbin", sort=True)
        }
    except ValueError:
        profile["pbin_mean"] = {}
    return profile


def _copy_if_exists(source: Path, destination: Path) -> str | None:
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def main() -> None:
    args = parse_args()
    source_sets = [item.strip() for item in args.sets.split(",") if item.strip()]
    unknown = [item for item in source_sets if item not in OPENMETEO_FORECAST_SETS]
    if unknown:
        raise ValueError(f"Unknown Open-Meteo forecast sets: {unknown}")

    results = []
    for source_set in source_sets:
        candidate_name = f"{args.candidate_prefix}_{args.model_set}_{source_set}_{args.openmeteo_fill_policy}"
        if args.include_openmeteo_pressure and "pressure" not in source_set:
            candidate_name += "_pressure"
        prediction_path = ROOT / f"predictions_{candidate_name}.csv"
        summary_copy = ROOT / f"full_pipeline_results_{candidate_name}.json"
        if args.skip_completed and summary_copy.exists() and prediction_path.exists():
            print(f"\nSkipping completed {candidate_name}")
            payload = json.loads(summary_copy.read_text())
            results.append(
                {
                    "candidate": candidate_name,
                    "source_set": source_set,
                    "summary_path": str(summary_copy),
                    "prediction_path": str(prediction_path),
                    "phase_d_best_weighted_q1_cv": payload.get("phase_d", {}).get("best_weighted_q1_cv"),
                    "prediction_profile": _prediction_profile(prediction_path),
                }
            )
            continue

        cmd = [
            sys.executable,
            str(ROOT / "run_full_pipeline.py"),
            "--skip-tune",
            "--model-set",
            args.model_set,
            "--physics-preset",
            args.physics_preset,
            "--fold-weights",
            args.fold_weights,
            "--blend-method",
            args.blend_method,
            "--blend-trials",
            str(args.blend_trials),
            "--candidate-name",
            candidate_name,
            "--openmeteo-forecast-set",
            source_set,
            "--openmeteo-fill-policy",
            args.openmeteo_fill_policy,
            "--openmeteo-min-coverage",
            str(args.openmeteo_min_coverage),
            "--external-source-set",
            args.external_source_set,
            "--external-source-timeout-sec",
            str(args.external_source_timeout_sec),
            "--external-cache-policy",
            args.external_cache_policy,
            "--catboost-task-type",
            args.catboost_task_type,
            "--output-path",
            str(prediction_path),
        ]
        if args.catboost_devices:
            cmd.extend(["--catboost-devices", args.catboost_devices])
        if args.include_openmeteo_pressure:
            cmd.append("--include-openmeteo-pressure")
        if args.refresh_external_weather:
            cmd.append("--refresh-external-weather")
        if args.include_nasa_power:
            cmd.append("--include-nasa-power")
        if args.skip_meteostat:
            cmd.append("--skip-meteostat")
        if args.allow_missing_meteostat:
            cmd.append("--allow-missing-meteostat")
        if args.include_meteostat_derived:
            cmd.append("--include-meteostat-derived")
        if args.include_so_ups_res:
            cmd.append("--include-so-ups-res")
            cmd.extend(["--so-ups-res-policy", args.so_ups_res_policy])
            if args.so_ups_res_exclude_months:
                cmd.extend(["--so-ups-res-exclude-months", args.so_ups_res_exclude_months])
        if args.include_so_ups_valid_months:
            cmd.append("--include-so-ups-valid-months")
        cmd.extend(args.extra_arg)

        print("\n$ " + " ".join(cmd), flush=True)
        try:
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
            summary_path = _copy_if_exists(ROOT / "full_pipeline_results.json", summary_copy)
            _copy_if_exists(ROOT / "environment_report.json", ROOT / f"environment_report_{candidate_name}.json")
            _copy_if_exists(ROOT / "phase_c_results.json", ROOT / f"phase_c_results_{candidate_name}.json")
            _copy_if_exists(ROOT / "phase_d_results.json", ROOT / f"phase_d_results_{candidate_name}.json")
            payload = json.loads(summary_copy.read_text()) if summary_path else {}
            record = {
                "candidate": candidate_name,
                "source_set": source_set,
                "fill_policy": args.openmeteo_fill_policy,
                "include_pressure": bool(
                    args.include_openmeteo_pressure
                    or source_set in OPENMETEO_PRESSURE_SETS
                ),
                "summary_path": str(summary_copy),
                "prediction_path": str(prediction_path),
                "phase_c_weighted_q1_cv": payload.get("phase_c", {}).get("weighted_q1_cv"),
                "phase_d_best_weighted_q1_cv": payload.get("phase_d", {}).get("best_weighted_q1_cv"),
                "prediction_stats": payload.get("prediction_stats", {}),
                "prediction_profile": _prediction_profile(prediction_path),
            }
            results.append(record)
        except Exception as exc:
            if not args.keep_going:
                raise
            results.append(
                {
                    "candidate": candidate_name,
                    "source_set": source_set,
                    "status": "failed",
                    "error": repr(exc),
                }
            )

        args.summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    args.summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.summary_path}")


if __name__ == "__main__":
    main()
