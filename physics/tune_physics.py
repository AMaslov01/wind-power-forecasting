"""
Standalone Optuna tuning of physical constants for the wind farm model.

The objective is the weighted Q1 MAE between target generation and the
physics baseline P_physics_farm. Three folds (Q1 of 2023, 2024, 2025) with
weights 0.2 / 0.3 / 0.5, matching the validation scheme in predict.run_q1_cv.

This script does NOT touch CatBoost; it only re-evaluates the physical
power-curve formula on the training data for each Optuna trial. One trial
takes well under a second.

Outputs:
    physics/tune_results.json — top-10 candidates by standalone MAE.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import (
    DATETIME_COL,
    HUB_HEIGHT_M,
    PhysicsConfig,
    TARGET,
    TRAIN_PATH,
    add_physical_features,
)


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "tune_results.json"
N_TRIALS = 1200
SEED = 42

FOLDS = [(2023, 0.2), (2024, 0.3), (2025, 0.5)]
SEARCH_SPACES = {
    "wide": {
        "v_cut_in": (2.0, 3.3),
        "v_rated": (8.0, 11.5),
        "v_cut_out": (23.0, 27.0),
        "p_rated_per_turbine": (3.20, 3.65),
        "cubic_exponent": (1.5, 3.5),
        "efficiency_factor": (0.85, 1.15),
        "ice_threshold": (-2.0, 5.0),
        "density_correction_exp": (0.5, 1.6),
    },
    "narrow": {
        "v_cut_in": (1.95, 2.45),
        "v_rated": (9.95, 10.90),
        "v_cut_out": (23.0, 26.9),
        "p_rated_per_turbine": (3.25, 3.42),
        "cubic_exponent": (1.45, 1.90),
        "efficiency_factor": (0.95, 1.12),
        "ice_threshold": (2.4, 5.0),
        "density_correction_exp": (0.45, 0.85),
    },
}


def load_train() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_PATH, parse_dates=[DATETIME_COL])
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)
    df["wind_direction_10m"] = df["wind_direction_10m"].fillna(df["wind_direction_80m"])
    df["wind_direction_180m"] = df["wind_direction_180m"].fillna(df["wind_direction_120m"])
    df["wind_speed_180m"] = df["wind_speed_180m"].fillna(df["wind_speed_120m"])
    return df


def q1_masks(df: pd.DataFrame) -> dict[int, np.ndarray]:
    masks = {}
    for year, _ in FOLDS:
        mask = (df[DATETIME_COL] >= pd.Timestamp(f"{year}-01-01")) & (
            df[DATETIME_COL] <= pd.Timestamp(f"{year}-03-31 23:00:00")
        )
        masks[year] = mask.values
    return masks


def evaluate_config(df: pd.DataFrame, cfg: PhysicsConfig, masks: dict[int, np.ndarray]) -> tuple[float, dict[int, float]]:
    feat = add_physical_features(df, cfg)
    per_year = {}
    weighted = 0.0
    for year, weight in FOLDS:
        m = masks[year]
        mae = mean_absolute_error(feat.loc[m, TARGET], feat.loc[m, "P_physics_farm"])
        per_year[year] = float(mae)
        weighted += weight * mae
    return float(weighted), per_year


def make_objective(df: pd.DataFrame, masks: dict[int, np.ndarray], search_space: str):
    space = SEARCH_SPACES[search_space]

    def objective(trial: optuna.Trial) -> float:
        cfg = PhysicsConfig(
            v_cut_in=trial.suggest_float("v_cut_in", *space["v_cut_in"]),
            v_rated=trial.suggest_float("v_rated", *space["v_rated"]),
            v_cut_out=trial.suggest_float("v_cut_out", *space["v_cut_out"]),
            p_rated_per_turbine=trial.suggest_float("p_rated_per_turbine", *space["p_rated_per_turbine"]),
            cubic_exponent=trial.suggest_float("cubic_exponent", *space["cubic_exponent"]),
            hub_height=HUB_HEIGHT_M,
            efficiency_factor=trial.suggest_float("efficiency_factor", *space["efficiency_factor"]),
            ice_threshold=trial.suggest_float("ice_threshold", *space["ice_threshold"]),
            density_correction_exp=trial.suggest_float("density_correction_exp", *space["density_correction_exp"]),
        )
        weighted, per_year = evaluate_config(df, cfg, masks)
        trial.set_user_attr("per_year_mae", per_year)
        trial.set_user_attr("params", asdict(cfg))
        return weighted

    return objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune physical wind-farm baseline parameters.")
    parser.add_argument(
        "--trials",
        type=int,
        default=N_TRIALS,
        help=f"Number of Optuna trials to run (default: {N_TRIALS}).",
    )
    parser.add_argument(
        "--search-space",
        choices=sorted(SEARCH_SPACES),
        default="narrow",
        help="Physics Optuna search space. narrow exploits the current 84m leaderboard cluster.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading train data from {TRAIN_PATH}...")
    df = load_train()
    masks = q1_masks(df)
    for year, mask in masks.items():
        print(f"  Q1 {year}: {int(mask.sum())} hours")

    default_weighted, default_per_year = evaluate_config(df, PhysicsConfig(), masks)
    print(f"Default config: weighted MAE = {default_weighted:.4f}, per-year = {default_per_year}")

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    objective = make_objective(df, masks, args.search_space)

    print(f"Running {args.trials} Optuna trials ({args.search_space} search space)...")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print(f"\nBest weighted MAE: {study.best_value:.4f}")
    best_params = study.best_trial.user_attrs.get("params", {**study.best_params, "hub_height": HUB_HEIGHT_M})
    print(f"Best params: {best_params}")

    trials_sorted = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
    )
    top10 = trials_sorted[:10]

    results = {
        "n_trials": args.trials,
        "seed": SEED,
        "search_space": args.search_space,
        "default": {
            "weighted_mae": default_weighted,
            "per_year_mae": default_per_year,
            "params": asdict(PhysicsConfig()),
        },
        "best": {
            "weighted_mae": float(study.best_value),
            "per_year_mae": study.best_trial.user_attrs.get("per_year_mae"),
            "params": study.best_trial.user_attrs.get("params", {**study.best_params, "hub_height": HUB_HEIGHT_M}),
        },
        "top10": [
            {
                "trial": t.number,
                "weighted_mae": float(t.value),
                "per_year_mae": t.user_attrs.get("per_year_mae"),
                "params": t.user_attrs.get("params", {**t.params, "hub_height": HUB_HEIGHT_M}),
            }
            for t in top10
        ],
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved top-10 to {RESULTS_PATH}")
    print("\nTop 10 candidates by weighted MAE:")
    for i, t in enumerate(top10, 1):
        print(f"  #{i} trial={t.number} mae={t.value:.4f}")
        for k, v in t.params.items():
            print(f"      {k} = {v:.4f}")


if __name__ == "__main__":
    main()
