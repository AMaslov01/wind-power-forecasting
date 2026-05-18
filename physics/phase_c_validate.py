"""
Phase C: validate top physics candidates against the full ensemble.

For each candidate PhysicsConfig, runs the per-fold per-model loop directly
(equivalent to run_q1_cv) and saves both:
  - weighted Q1 CV platform error  (in phase_c_results.json)
  - per-fold per-model out-of-fold predictions (in oof_<name>.npz)

The OOF cache is consumed later by Phase D to re-optimize blend weights
without an extra CV run.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict import (
    DATETIME_COL,
    HUB_HEIGHT_M,
    MODEL_SPECS,
    PhysicsConfig,
    TARGET,
    blend_model_predictions,
    default_blend_weights,
    derive_final_iterations,
    fit_predict_spec,
    format_blend_weights,
    load_and_prepare,
    optimize_validator_blend_weights,
    platform_error_percent,
    q1_folds,
    score_blend_weights,
    set_seed,
    SEED,
)


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "phase_c_results.json"


CANDIDATES: list[tuple[str, PhysicsConfig]] = [
    (
        "optuna_narrow_top1",
        PhysicsConfig(
            v_cut_in=2.7266,
            v_rated=9.6478,
            v_cut_out=24.5148,
            p_rated_per_turbine=3.4002,
            cubic_exponent=2.3246,
            hub_height=HUB_HEIGHT_M,
            efficiency_factor=1.0586,
            ice_threshold=1.9882,
            density_correction_exp=1.1900,
        ),
    ),
    (
        "optuna_wide_top1",
        PhysicsConfig(
            v_cut_in=2.3557,
            v_rated=10.7345,
            v_cut_out=23.0528,
            p_rated_per_turbine=3.3260,
            cubic_exponent=1.5041,
            hub_height=HUB_HEIGHT_M,
            efficiency_factor=1.0480,
            ice_threshold=0.6342,
            density_correction_exp=0.8658,
        ),
    ),
    (
        "physical_compromise",
        PhysicsConfig(
            v_cut_in=2.8,
            v_rated=10.3,
            v_cut_out=25.0,
            p_rated_per_turbine=3.45,
            cubic_exponent=2.5,
            hub_height=HUB_HEIGHT_M,
            efficiency_factor=1.05,
            ice_threshold=2.0,
            density_correction_exp=1.0,
        ),
    ),
]


def run_q1_cv_with_oof(
    train,
    feature_cols,
) -> tuple[dict, dict, dict, dict, dict]:
    """Run Q1 CV and return per-fold per-model OOF + y_true + best_iters."""
    fold_scores: dict[int | str, float] = {}
    baseline_fold_scores: dict[int, float] = {}
    best_iters: dict[str, list[int]] = {spec.name: [] for spec in MODEL_SPECS}
    oof: dict[int, dict[str, np.ndarray]] = {}
    y_true: dict[int, np.ndarray] = {}
    validator_weights: dict[int | str, dict[str, float]] = {}
    folds = q1_folds(train)
    fold_weights = {year: weight for year, _, _, weight in folds}
    base_weights = default_blend_weights()

    for year, train_idx, valid_idx, _ in folds:
        fold_df = train.iloc[valid_idx].reset_index(drop=True)
        fold_blend_weights = optimize_validator_blend_weights(
            oof,
            y_true,
            fold_weights,
            base_weights=base_weights,
        )
        validator_weights[year] = fold_blend_weights
        if oof:
            calibration_score = score_blend_weights(fold_blend_weights, oof, y_true, fold_weights)
            print(
                f"    validator weights for Q1 {year}: "
                f"{format_blend_weights(fold_blend_weights)} "
                f"(historical calibration={calibration_score:.4f}%)"
            )
        else:
            print(
                f"    validator weights for Q1 {year}: "
                f"{format_blend_weights(fold_blend_weights)} "
                "(no earlier folds)"
            )
        y_true[year] = fold_df[TARGET].values.copy()
        oof[year] = {}

        for spec in MODEL_SPECS:
            pred, best_iter = fit_predict_spec(
                spec,
                train,
                fold_df,
                feature_cols,
                train_idx=train_idx,
                eval_idx=valid_idx,
            )
            oof[year][spec.name] = pred.copy()
            if best_iter is not None:
                best_iters[spec.name].append(best_iter)

        baseline_blended = blend_model_predictions(oof[year], base_weights)
        baseline_score = platform_error_percent(y_true[year], baseline_blended)
        baseline_fold_scores[year] = baseline_score
        blended = blend_model_predictions(oof[year], fold_blend_weights)
        score = platform_error_percent(y_true[year], blended)
        fold_scores[year] = score
        print(
            f"    Q1 {year}: {score:.4f}% platform error "
            f"(static={baseline_score:.4f}%, delta={baseline_score - score:+.4f}%)"
        )

    weighted = sum(
        fold_scores[year] * weight for year, _, _, weight in folds
    )
    baseline_weighted = sum(
        baseline_fold_scores[year] * weight for year, _, _, weight in folds
    )
    fold_scores["weighted"] = weighted
    fold_scores["static_weighted"] = baseline_weighted
    final_weights = optimize_validator_blend_weights(oof, y_true, fold_weights, base_weights=base_weights)
    validator_weights["final"] = final_weights
    print(
        f"    Weighted Q1 proxy: {weighted:.4f}% "
        f"(static={baseline_weighted:.4f}%, delta={baseline_weighted - weighted:+.4f}%)"
    )
    print(f"    Final validator weights: {format_blend_weights(final_weights)}")
    return fold_scores, oof, y_true, best_iters, validator_weights


def save_oof(name: str, oof, y_true) -> Path:
    path = ROOT / f"oof_{name}.npz"
    payload: dict = {}
    for year, by_model in oof.items():
        for model_name, arr in by_model.items():
            payload[f"oof__{year}__{model_name}"] = arr
        payload[f"y_true__{year}"] = y_true[year]
    np.savez(path, **payload)
    return path


def main() -> None:
    set_seed(SEED)
    results = []

    for name, cfg in CANDIDATES:
        print(f"\n{'='*60}")
        print(f"Candidate: {name}")
        print(f"  cfg: {asdict(cfg)}")
        print(f"{'='*60}")

        t0 = time.time()
        train, valid, _, feature_cols = load_and_prepare(cfg)
        print(f"  features: {len(feature_cols)}")
        scores, oof, y_true, best_iters, validator_weights = run_q1_cv_with_oof(train, feature_cols)
        elapsed = time.time() - t0

        oof_path = save_oof(name, oof, y_true)

        record = {
            "name": name,
            "config": asdict(cfg),
            "weighted_q1_cv": float(scores["weighted"]),
            "static_weighted_q1_cv": float(scores["static_weighted"]),
            "validator_improvement_abs": float(scores["static_weighted"] - scores["weighted"]),
            "per_year_cv": {str(y): float(scores[y]) for y in (2023, 2024, 2025)},
            "final_iterations": derive_final_iterations(best_iters),
            "validator_weights": {
                str(key): {model: float(weight) for model, weight in weights.items()}
                for key, weights in validator_weights.items()
            },
            "elapsed_sec": elapsed,
            "oof_path": str(oof_path.relative_to(ROOT)),
        }
        results.append(record)
        print(f"  -> weighted Q1 CV: {scores['weighted']:.4f}%  (took {elapsed/60:.1f} min)")
        print(f"     OOF saved to {oof_path.name}")

        RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print("Final ranking (weighted Q1 CV, lower is better):")
    for r in sorted(results, key=lambda x: x["weighted_q1_cv"]):
        print(f"  {r['name']:30s}  {r['weighted_q1_cv']:.4f}%")


if __name__ == "__main__":
    main()
