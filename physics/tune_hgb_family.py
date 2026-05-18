"""
Tune HGB variants around the current public-best HGB family.

This keeps the public_best physics feature matrix fixed, screens HGB
hyperparameters on a cheap fold subset, then re-scores the top candidates on
the requested final Q1 folds. It writes physics/hgb_family_params.json, which
predict.py consumes through the hgb_optuna_family model set.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))

import predict
from phase_d_optimize_weights import FOLD_WEIGHT_SCHEMES, resolve_fold_weights


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = predict.HGB_FAMILY_PARAMS_PATH
SEED = predict.SEED
ALL_Q1_YEARS = (2023, 2024, 2025)


@dataclass(frozen=True)
class FoldData:
    year: int
    weight: float
    fit_x: np.ndarray
    pred_x: np.ndarray
    y_direct: np.ndarray
    y_residual: np.ndarray
    target: np.ndarray
    pred_physics: np.ndarray


def parse_years(value: str) -> tuple[int, ...]:
    if value.strip().lower() == "all":
        return ALL_Q1_YEARS
    years = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    unknown = [year for year in years if year not in ALL_Q1_YEARS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown Q1 fold year(s): {unknown}")
    if not years:
        raise argparse.ArgumentTypeError("At least one fold year is required.")
    return years


def normalized_fold_weights(base_weights: dict[int, float], years: tuple[int, ...]) -> dict[int, float]:
    selected = {year: float(base_weights[year]) for year in years}
    total = sum(selected.values())
    if total <= 0:
        raise ValueError("Selected fold weights must sum to a positive value.")
    return {year: weight / total for year, weight in selected.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune HGB direct/residual variants.")
    parser.add_argument(
        "--trials-per-mode",
        type=int,
        default=30,
        help="Optuna screen trials for each target mode: direct and residual.",
    )
    parser.add_argument(
        "--top-per-mode",
        type=int,
        default=2,
        help="Number of best direct/residual models to write. The pipeline consumes the first two.",
    )
    parser.add_argument(
        "--candidate-pool-per-mode",
        type=int,
        default=6,
        help="Number of screened candidates per mode to re-score on final folds.",
    )
    parser.add_argument(
        "--search-profile",
        choices=["fast", "full"],
        default="fast",
        help="fast uses cheaper HGB ranges; full opens the wider/slow search.",
    )
    parser.add_argument(
        "--screen-fold-years",
        type=parse_years,
        default=parse_years("2025"),
        help="Comma-separated Q1 years used inside Optuna objective, or 'all'.",
    )
    parser.add_argument(
        "--final-fold-years",
        type=parse_years,
        default=parse_years("all"),
        help="Comma-separated Q1 years used to re-score top candidates, or 'all'.",
    )
    parser.add_argument(
        "--fold-weights",
        choices=sorted(FOLD_WEIGHT_SCHEMES),
        default="recency",
        help="Fold weighting scheme used by the HGB tuning objective.",
    )
    parser.add_argument(
        "--include-meteostat-derived",
        action="store_true",
        help="Tune on the opt-in 10 Meteostat-derived features.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Where to write tuned HGB params (default: {OUTPUT_PATH}).",
    )
    return parser.parse_args()


def suggest_hgb_params(trial: optuna.Trial, target_mode: str, search_profile: str) -> dict:
    if search_profile == "fast":
        if target_mode == "direct":
            max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 24, 56)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 28, 110)
            learning_rate = trial.suggest_float("learning_rate", 0.025, 0.075, log=True)
            l2_regularization = trial.suggest_float("l2_regularization", 0.05, 0.90, log=True)
            validation_fraction = trial.suggest_float("validation_fraction", 0.12, 0.22)
            max_iter = trial.suggest_int("max_iter", 250, 850)
        elif target_mode == "residual":
            max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 12, 32)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 60, 160)
            learning_rate = trial.suggest_float("learning_rate", 0.018, 0.055, log=True)
            l2_regularization = trial.suggest_float("l2_regularization", 0.20, 1.80, log=True)
            validation_fraction = trial.suggest_float("validation_fraction", 0.14, 0.26)
            max_iter = trial.suggest_int("max_iter", 300, 950)
        else:
            raise ValueError(f"Unknown target mode: {target_mode}")
        max_bins = trial.suggest_categorical("max_bins", [63, 127])
    elif search_profile == "full":
        if target_mode == "direct":
            max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 28, 72)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 22, 90)
            learning_rate = trial.suggest_float("learning_rate", 0.018, 0.060, log=True)
            l2_regularization = trial.suggest_float("l2_regularization", 0.03, 0.60, log=True)
            validation_fraction = trial.suggest_float("validation_fraction", 0.10, 0.24)
            max_iter = trial.suggest_int("max_iter", 650, 1500)
        elif target_mode == "residual":
            max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 12, 38)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 55, 150)
            learning_rate = trial.suggest_float("learning_rate", 0.012, 0.045, log=True)
            l2_regularization = trial.suggest_float("l2_regularization", 0.15, 1.50, log=True)
            validation_fraction = trial.suggest_float("validation_fraction", 0.12, 0.28)
            max_iter = trial.suggest_int("max_iter", 750, 1800)
        else:
            raise ValueError(f"Unknown target mode: {target_mode}")
        max_bins = trial.suggest_categorical("max_bins", [63, 127, 255])
    else:
        raise ValueError(f"Unknown search profile: {search_profile}")

    return {
        "loss": "absolute_error",
        "max_iter": int(max_iter),
        "learning_rate": float(learning_rate),
        "max_leaf_nodes": int(max_leaf_nodes),
        "min_samples_leaf": int(min_samples_leaf),
        "l2_regularization": float(l2_regularization),
        "max_bins": int(max_bins),
        "random_state": SEED,
        "early_stopping": True,
        "validation_fraction": float(validation_fraction),
        "n_iter_no_change": int(trial.suggest_int("n_iter_no_change", 8, 18)),
    }


def build_fold_cache(
    train,
    feature_cols: list[str],
    fold_weights: dict[int, float],
    years: tuple[int, ...],
) -> list[FoldData]:
    folds_by_year = {year: (train_idx, valid_idx) for year, train_idx, valid_idx, _ in predict.q1_folds(train)}
    cached = []
    for year in years:
        train_idx, valid_idx = folds_by_year[year]
        fold_df = train.iloc[valid_idx].reset_index(drop=True)
        fit_df, pred_df, _ = predict.prepare_model_frames_with_latent_availability(
            train,
            fold_df,
            train_idx=train_idx,
            eval_idx=None,
        )
        cached.append(
            FoldData(
                year=year,
                weight=float(fold_weights[year]),
                fit_x=np.ascontiguousarray(fit_df[feature_cols].to_numpy(dtype=np.float32)),
                pred_x=np.ascontiguousarray(pred_df[feature_cols].to_numpy(dtype=np.float32)),
                y_direct=fit_df[predict.TARGET].to_numpy(dtype=np.float32),
                y_residual=fit_df["residual"].to_numpy(dtype=np.float32),
                target=fold_df[predict.TARGET].to_numpy(dtype=np.float64),
                pred_physics=pred_df["P_physics_farm"].to_numpy(dtype=np.float64),
            )
        )
        print(
            f"  cached Q1 {year}: train_rows={len(train_idx)} pred_rows={len(valid_idx)} "
            f"weight={fold_weights[year]:.3f}",
            flush=True,
        )
    return cached


def score_prediction(target: np.ndarray, pred: np.ndarray) -> float:
    pred = np.clip(np.asarray(pred, dtype=np.float64), 0.0, predict.P_RATED_FARM)
    return float(np.mean(np.abs(target - pred)) / predict.P_RATED_FARM * 100.0)


def evaluate_hgb_params(
    folds: list[FoldData],
    target_mode: str,
    params: dict,
) -> tuple[float, dict[int, float]]:
    per_year: dict[int, float] = {}
    total = 0.0

    for fold in folds:
        y_fit = fold.y_direct if target_mode == "direct" else fold.y_residual
        model = HistGradientBoostingRegressor(**params)
        model.fit(fold.fit_x, y_fit)
        raw_pred = model.predict(fold.pred_x)
        pred = fold.pred_physics + raw_pred if target_mode == "residual" else raw_pred
        score = score_prediction(fold.target, pred)
        per_year[fold.year] = float(score)
        total += fold.weight * score

    return float(total), per_year


def run_mode_study(
    screen_folds: list[FoldData],
    target_mode: str,
    n_trials: int,
    search_profile: str,
) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        params = suggest_hgb_params(trial, target_mode, search_profile)
        score, per_year = evaluate_hgb_params(screen_folds, target_mode, params)
        trial.set_user_attr("params", params)
        trial.set_user_attr("screen_per_year_cv", per_year)
        return score

    sampler = optuna.samplers.TPESampler(seed=SEED + (0 if target_mode == "direct" else 17))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def top_trials(study: optuna.Study, limit: int) -> list[optuna.trial.FrozenTrial]:
    complete = [trial for trial in study.trials if trial.value is not None]
    return sorted(complete, key=lambda trial: float(trial.value))[:limit]


def rescore_top_trials(
    study: optuna.Study,
    final_folds: list[FoldData],
    target_mode: str,
    candidate_pool: int,
) -> list[dict]:
    records = []
    for trial in top_trials(study, candidate_pool):
        params = trial.user_attrs["params"]
        cv_score, per_year = evaluate_hgb_params(final_folds, target_mode, params)
        records.append(
            {
                "screen_score": float(trial.value),
                "screen_per_year_cv": trial.user_attrs["screen_per_year_cv"],
                "cv_score": float(cv_score),
                "per_year_cv": per_year,
                "params": params,
            }
        )
    return sorted(records, key=lambda record: record["cv_score"])


def main() -> None:
    args = parse_args()
    started = time.time()
    base_fold_weights = resolve_fold_weights(args.fold_weights)
    screen_fold_weights = normalized_fold_weights(base_fold_weights, args.screen_fold_years)
    final_fold_weights = normalized_fold_weights(base_fold_weights, args.final_fold_years)

    print("Loading public_best feature matrix for HGB tuning...")
    train, _, _, feature_cols = predict.load_and_prepare(
        predict.PUBLIC_BEST_PHYSICS,
        use_external_weather=True,
        include_meteostat=True,
        allow_partial_meteostat=True,
        include_meteostat_derived=args.include_meteostat_derived,
        include_physics_variants=True,
    )
    print(f"  train rows: {len(train)} | features: {len(feature_cols)}")
    print(f"  search profile: {args.search_profile}")
    print(f"  screen folds: {args.screen_fold_years} {screen_fold_weights}")
    print(f"  final folds: {args.final_fold_years} {final_fold_weights}")

    print("\nPreparing screen fold matrices...")
    screen_folds = build_fold_cache(train, feature_cols, screen_fold_weights, args.screen_fold_years)
    print("\nPreparing final re-score fold matrices...")
    final_folds = build_fold_cache(train, feature_cols, final_fold_weights, args.final_fold_years)

    studies = {}
    best_by_mode = {}
    models = []
    for target_mode, prefix in [("direct", "hgb_opt_direct"), ("residual", "hgb_opt_residual")]:
        print(f"\nTuning {target_mode} HGB ({args.trials_per_mode} screen trials)...")
        study = run_mode_study(
            screen_folds,
            target_mode,
            args.trials_per_mode,
            args.search_profile,
        )
        studies[target_mode] = study
        if not top_trials(study, 1):
            raise RuntimeError(f"No completed Optuna trials for {target_mode}.")
        print(f"  best screen {target_mode}: {study.best_value:.4f}%")

        pool_size = max(args.top_per_mode, args.candidate_pool_per_mode)
        rescored = rescore_top_trials(study, final_folds, target_mode, pool_size)
        best_by_mode[target_mode] = rescored[0]
        print(f"  best final {target_mode}: {rescored[0]['cv_score']:.4f}%")

        for rank, record in enumerate(rescored[: args.top_per_mode], start=1):
            models.append(
                {
                    "name": f"{prefix}_{rank}",
                    "target_mode": target_mode,
                    "rank": rank,
                    "cv_score": record["cv_score"],
                    "per_year_cv": record["per_year_cv"],
                    "screen_score": record["screen_score"],
                    "screen_per_year_cv": record["screen_per_year_cv"],
                    "params": record["params"],
                }
            )

    payload = {
        "source": "tune_hgb_family.py",
        "seed": SEED,
        "elapsed_sec": time.time() - started,
        "trials_per_mode": args.trials_per_mode,
        "top_per_mode": args.top_per_mode,
        "candidate_pool_per_mode": args.candidate_pool_per_mode,
        "search_profile": args.search_profile,
        "physics_preset": "public_best",
        "feature_count": len(feature_cols),
        "include_meteostat_derived": args.include_meteostat_derived,
        "include_physics_variants": True,
        "fold_weight_scheme": args.fold_weights,
        "base_fold_weights": base_fold_weights,
        "screen_fold_years": list(args.screen_fold_years),
        "screen_fold_weights": screen_fold_weights,
        "final_fold_years": list(args.final_fold_years),
        "final_fold_weights": final_fold_weights,
        "models": models,
        "best": {
            mode: {
                "cv_score": record["cv_score"],
                "params": record["params"],
                "per_year_cv": record["per_year_cv"],
                "screen_score": record["screen_score"],
                "screen_per_year_cv": record["screen_per_year_cv"],
            }
            for mode, record in best_by_mode.items()
        },
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nSaved tuned HGB params to {args.output_path}")
    for model in models:
        print(
            f"  {model['name']}: final={model['cv_score']:.4f}% "
            f"screen={model['screen_score']:.4f}% ({model['target_mode']})"
        )


if __name__ == "__main__":
    main()
