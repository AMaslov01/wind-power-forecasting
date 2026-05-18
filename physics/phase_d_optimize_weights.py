"""
Phase D: re-optimize ensemble blend weights via SLSQP.

Loads OOF predictions (saved by phase_c_validate.py) for the chosen physics
candidate, then minimizes the weighted Q1 platform error over the active model
weights. Uses several starting points to verify convergence to a single
optimum.

Usage:
    python physics/phase_d_optimize_weights.py [candidate_name]

If candidate_name is omitted, the script picks the winner from
phase_c_results.json (lowest weighted_q1_cv).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))

import predict


ROOT = Path(__file__).resolve().parent
RECENCY_FOLD_WEIGHTS = {2023: 0.2, 2024: 0.3, 2025: 0.5}
VALID_SIMILARITY_FOLD_WEIGHTS = {2023: 0.520, 2024: 0.255, 2025: 0.225}
RAW_SIMILARITY_FOLD_WEIGHTS = {2023: 0.300, 2024: 0.385, 2025: 0.315}
FOLD_WEIGHT_SCHEMES = {
    "recency": RECENCY_FOLD_WEIGHTS,
    "valid_similarity": VALID_SIMILARITY_FOLD_WEIGHTS,
    "raw_similarity": RAW_SIMILARITY_FOLD_WEIGHTS,
}
FOLD_WEIGHTS = RECENCY_FOLD_WEIGHTS
WEIGHT_LOWER = 0.001
MODEL_NAMES = [spec.name for spec in predict.MODEL_SPECS]
STABLE_HGB_CORE_NAMES = ("hgb_direct_deep", "hgb_residual_smooth")
OPTUNA_HGB_PREFIX = "hgb_opt_"
LGB_PREFIX = "lgb_"
TREEBAG_PREFIX = "etr_"
QUANTILE_HGB_PREFIX = "hgb_q50_"
STABLE_HGB_MIN_WEIGHT = 0.0
OPTUNA_HGB_MAX_WEIGHT = 1.0
LGB_MAX_WEIGHT = 1.0
TREEBAG_MAX_WEIGHT = 1.0
QUANTILE_HGB_MAX_WEIGHT = 1.0
BLEND_PRIOR_STRENGTH = 0.0
CONSTRAINT_PENALTY_SCALE = 1000.0


def set_model_names(model_set: str) -> None:
    global MODEL_NAMES
    predict.set_model_set(model_set)
    MODEL_NAMES = [spec.name for spec in predict.MODEL_SPECS]


def set_weight_lower(weight_lower: float) -> None:
    global WEIGHT_LOWER
    if weight_lower < 0.0:
        raise ValueError("Blend weight lower bound must be non-negative.")
    WEIGHT_LOWER = float(weight_lower)
    predict.set_validator_weight_lower(WEIGHT_LOWER)
    if float(model_weight_lower_bounds().sum()) >= 1.0:
        raise ValueError("Blend weight lower bounds are too large for the active model set.")


def model_weight_lower_bounds() -> np.ndarray:
    return np.array(
        [max(WEIGHT_LOWER, predict.blend_weight_lower_bound(name)) for name in MODEL_NAMES],
        dtype=float,
    )


def set_blend_constraints(
    stable_hgb_min_weight: float = 0.0,
    optuna_hgb_max_weight: float = 1.0,
    lgb_max_weight: float = 1.0,
    treebag_max_weight: float = 1.0,
    blend_prior_strength: float = 0.0,
) -> None:
    global STABLE_HGB_MIN_WEIGHT, OPTUNA_HGB_MAX_WEIGHT, LGB_MAX_WEIGHT, TREEBAG_MAX_WEIGHT, BLEND_PRIOR_STRENGTH
    stable_hgb_min_weight = float(stable_hgb_min_weight)
    optuna_hgb_max_weight = float(optuna_hgb_max_weight)
    lgb_max_weight = float(lgb_max_weight)
    treebag_max_weight = float(treebag_max_weight)
    blend_prior_strength = float(blend_prior_strength)
    if not 0.0 <= stable_hgb_min_weight <= 1.0:
        raise ValueError("Stable HGB minimum weight must be between 0 and 1.")
    if not 0.0 <= optuna_hgb_max_weight <= 1.0:
        raise ValueError("Optuna HGB maximum weight must be between 0 and 1.")
    if not 0.0 <= lgb_max_weight <= 1.0:
        raise ValueError("LightGBM maximum weight must be between 0 and 1.")
    if not 0.0 <= treebag_max_weight <= 1.0:
        raise ValueError("Bagged-tree maximum weight must be between 0 and 1.")
    if blend_prior_strength < 0.0:
        raise ValueError("Blend prior strength must be non-negative.")

    lower_bounds = model_weight_lower_bounds()
    stable_idx = _indices_for_names(STABLE_HGB_CORE_NAMES)
    optuna_idx = _indices_for_prefix(OPTUNA_HGB_PREFIX)
    lgb_idx = _indices_for_prefix(LGB_PREFIX)
    treebag_idx = _indices_for_prefix(TREEBAG_PREFIX)
    if stable_hgb_min_weight > 0.0 and not stable_idx:
        raise ValueError("Stable HGB minimum requested, but active model set has no stable HGB core models.")
    if optuna_idx and optuna_hgb_max_weight < float(lower_bounds[optuna_idx].sum()):
        raise ValueError("Optuna HGB cap is below the active lower-bound mass for optuna models.")
    if lgb_idx and lgb_max_weight < float(lower_bounds[lgb_idx].sum()):
        raise ValueError("LightGBM cap is below the active lower-bound mass for LightGBM models.")
    if treebag_idx and treebag_max_weight < float(lower_bounds[treebag_idx].sum()):
        raise ValueError("Bagged-tree cap is below the active lower-bound mass for bagged-tree models.")
    stable_set = set(stable_idx)
    non_stable_floor = float(sum(bound for idx, bound in enumerate(lower_bounds) if idx not in stable_set))
    if stable_idx and stable_hgb_min_weight > 1.0 - non_stable_floor:
        raise ValueError("Stable HGB minimum is infeasible with the active lower bounds.")

    STABLE_HGB_MIN_WEIGHT = stable_hgb_min_weight
    OPTUNA_HGB_MAX_WEIGHT = optuna_hgb_max_weight
    LGB_MAX_WEIGHT = lgb_max_weight
    TREEBAG_MAX_WEIGHT = treebag_max_weight
    BLEND_PRIOR_STRENGTH = blend_prior_strength


def set_quantile_hgb_max_weight(quantile_hgb_max_weight: float = 1.0) -> None:
    global QUANTILE_HGB_MAX_WEIGHT
    quantile_hgb_max_weight = float(quantile_hgb_max_weight)
    if not 0.0 <= quantile_hgb_max_weight <= 1.0:
        raise ValueError("Quantile HGB maximum weight must be between 0 and 1.")
    lower_bounds = model_weight_lower_bounds()
    quantile_idx = _indices_for_prefix(QUANTILE_HGB_PREFIX)
    if quantile_idx and quantile_hgb_max_weight < float(lower_bounds[quantile_idx].sum()):
        raise ValueError("Quantile HGB cap is below the active lower-bound mass for hgb_q50_* models.")
    QUANTILE_HGB_MAX_WEIGHT = quantile_hgb_max_weight


def resolve_fold_weights(scheme: str = "recency") -> dict[int, float]:
    if scheme not in FOLD_WEIGHT_SCHEMES:
        raise ValueError(f"Unknown fold weight scheme: {scheme}")
    weights = FOLD_WEIGHT_SCHEMES[scheme]
    total = sum(weights.values())
    return {year: float(weight / total) for year, weight in weights.items()}


def resolve_constraint_profile(
    profile: str,
    stable_hgb_min_weight: float | None,
    optuna_hgb_max_weight: float | None,
    lgb_max_weight: float | None,
    treebag_max_weight: float | None,
    blend_prior_strength: float | None,
) -> tuple[float, float, float, float, float]:
    if profile == "hgb_conservative":
        return (
            0.70 if stable_hgb_min_weight is None else stable_hgb_min_weight,
            0.25 if optuna_hgb_max_weight is None else optuna_hgb_max_weight,
            1.0 if lgb_max_weight is None else lgb_max_weight,
            1.0 if treebag_max_weight is None else treebag_max_weight,
            0.08 if blend_prior_strength is None else blend_prior_strength,
        )
    if profile == "lgb_conservative":
        return (
            0.70 if stable_hgb_min_weight is None else stable_hgb_min_weight,
            1.0 if optuna_hgb_max_weight is None else optuna_hgb_max_weight,
            0.18 if lgb_max_weight is None else lgb_max_weight,
            1.0 if treebag_max_weight is None else treebag_max_weight,
            0.06 if blend_prior_strength is None else blend_prior_strength,
        )
    if profile == "forest_conservative":
        return (
            0.65 if stable_hgb_min_weight is None else stable_hgb_min_weight,
            1.0 if optuna_hgb_max_weight is None else optuna_hgb_max_weight,
            0.14 if lgb_max_weight is None else lgb_max_weight,
            0.18 if treebag_max_weight is None else treebag_max_weight,
            0.06 if blend_prior_strength is None else blend_prior_strength,
        )
    if profile == "all_family_conservative":
        return (
            0.68 if stable_hgb_min_weight is None else stable_hgb_min_weight,
            1.0 if optuna_hgb_max_weight is None else optuna_hgb_max_weight,
            0.16 if lgb_max_weight is None else lgb_max_weight,
            0.08 if treebag_max_weight is None else treebag_max_weight,
            0.08 if blend_prior_strength is None else blend_prior_strength,
        )
    if profile == "none":
        return (
            0.0 if stable_hgb_min_weight is None else stable_hgb_min_weight,
            1.0 if optuna_hgb_max_weight is None else optuna_hgb_max_weight,
            1.0 if lgb_max_weight is None else lgb_max_weight,
            1.0 if treebag_max_weight is None else treebag_max_weight,
            0.0 if blend_prior_strength is None else blend_prior_strength,
        )
    raise ValueError(f"Unknown blend constraint profile: {profile}")


def _indices_for_names(names: tuple[str, ...]) -> list[int]:
    wanted = set(names)
    return [i for i, name in enumerate(MODEL_NAMES) if name in wanted]


def _indices_for_prefix(prefix: str) -> list[int]:
    return [i for i, name in enumerate(MODEL_NAMES) if name.startswith(prefix)]


def stable_hgb_weight(weights: np.ndarray) -> float:
    idx = _indices_for_names(STABLE_HGB_CORE_NAMES)
    return float(np.sum(weights[idx])) if idx else 0.0


def optuna_hgb_weight(weights: np.ndarray) -> float:
    idx = _indices_for_prefix(OPTUNA_HGB_PREFIX)
    return float(np.sum(weights[idx])) if idx else 0.0


def lgb_weight(weights: np.ndarray) -> float:
    idx = _indices_for_prefix(LGB_PREFIX)
    return float(np.sum(weights[idx])) if idx else 0.0


def treebag_weight(weights: np.ndarray) -> float:
    idx = _indices_for_prefix(TREEBAG_PREFIX)
    return float(np.sum(weights[idx])) if idx else 0.0


def quantile_hgb_weight(weights: np.ndarray) -> float:
    idx = _indices_for_prefix(QUANTILE_HGB_PREFIX)
    return float(np.sum(weights[idx])) if idx else 0.0


def constraint_violations(weights: np.ndarray) -> dict[str, float]:
    lower_bounds = model_weight_lower_bounds()
    return {
        "lower_bounds": float(np.maximum(lower_bounds - weights, 0.0).sum()),
        "stable_hgb_min": max(0.0, STABLE_HGB_MIN_WEIGHT - stable_hgb_weight(weights)),
        "optuna_hgb_max": max(0.0, optuna_hgb_weight(weights) - OPTUNA_HGB_MAX_WEIGHT),
        "lgb_max": max(0.0, lgb_weight(weights) - LGB_MAX_WEIGHT),
        "treebag_max": max(0.0, treebag_weight(weights) - TREEBAG_MAX_WEIGHT),
        "quantile_hgb_max": max(0.0, quantile_hgb_weight(weights) - QUANTILE_HGB_MAX_WEIGHT),
    }


def constraint_penalty(weights: np.ndarray) -> float:
    violations = constraint_violations(weights)
    return CONSTRAINT_PENALTY_SCALE * sum(value * value for value in violations.values())


def prior_penalty(weights: np.ndarray, prior_weights: np.ndarray | None) -> float:
    if BLEND_PRIOR_STRENGTH <= 0.0 or prior_weights is None:
        return 0.0
    return BLEND_PRIOR_STRENGTH * float(np.sum((weights - prior_weights) ** 2))


def blend_regularization(weights: np.ndarray, prior_weights: np.ndarray | None) -> float:
    return prior_penalty(weights, prior_weights) + constraint_penalty(weights)


def constraint_summary() -> dict[str, float]:
    return {
        "stable_hgb_min_weight": float(STABLE_HGB_MIN_WEIGHT),
        "optuna_hgb_max_weight": float(OPTUNA_HGB_MAX_WEIGHT),
        "lgb_max_weight": float(LGB_MAX_WEIGHT),
        "treebag_max_weight": float(TREEBAG_MAX_WEIGHT),
        "quantile_hgb_max_weight": float(QUANTILE_HGB_MAX_WEIGHT),
        "blend_prior_strength": float(BLEND_PRIOR_STRENGTH),
    }


def format_constraint_summary() -> str:
    return (
        f"stable_hgb_min={STABLE_HGB_MIN_WEIGHT:.3f}, "
        f"optuna_hgb_max={OPTUNA_HGB_MAX_WEIGHT:.3f}, "
        f"lgb_max={LGB_MAX_WEIGHT:.3f}, "
        f"treebag_max={TREEBAG_MAX_WEIGHT:.3f}, "
        f"quantile_hgb_max={QUANTILE_HGB_MAX_WEIGHT:.3f}, "
        f"prior_strength={BLEND_PRIOR_STRENGTH:.3f}"
    )


def default_starting_points(model_names: list[str], current_w: np.ndarray) -> dict[str, np.ndarray]:
    """N-aware starting points: uniform, current, and task-shaped priors."""
    n = len(model_names)
    pts: dict[str, np.ndarray] = {
        "equal": np.full(n, 1.0 / n),
        "current": current_w.copy(),
    }

    direct_mask = np.array(["direct" in name for name in model_names], dtype=float)
    residual_mask = np.array(["residual" in name for name in model_names], dtype=float)

    def _norm(v: np.ndarray) -> np.ndarray:
        v = np.maximum(v, 1e-3)
        return v / v.sum()

    if direct_mask.sum() > 0:
        pts["direct_heavy"] = _norm(direct_mask * 4.0 + 1.0)
    if residual_mask.sum() > 0:
        pts["residual_heavy"] = _norm(residual_mask * 4.0 + 1.0)
    return pts


def stack_ridge(
    oof: dict,
    y_true: dict,
    alpha: float = 1.0,
    fold_weights: dict[int, float] | None = None,
    prior_weights: np.ndarray | None = None,
) -> dict:
    """Non-negative ridge stacking over OOF predictions."""
    fold_weights = fold_weights or FOLD_WEIGHTS
    X_parts, y_parts, weight_parts = [], [], []
    for year, fold_weight in fold_weights.items():
        n_rows = len(y_true[year])
        X_parts.append(np.column_stack([oof[year][name] for name in MODEL_NAMES]))
        y_parts.append(y_true[year])
        weight_parts.append(np.full(n_rows, fold_weight / n_rows))

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    sample_weight = np.concatenate(weight_parts, axis=0)

    model = Ridge(alpha=alpha, positive=True, fit_intercept=False)
    model.fit(X_all, y_all, sample_weight=sample_weight)
    raw = np.maximum(model.coef_, 0.0)
    if raw.sum() <= 0:
        raw = np.full(len(MODEL_NAMES), 1.0 / len(MODEL_NAMES))
    weights = raw / raw.sum()
    lower_bounds = model_weight_lower_bounds()
    free_mass = 1.0 - float(lower_bounds.sum())
    if free_mass <= 0.0:
        raise ValueError("Blend lower bounds are too large for ridge stacking.")
    surplus = np.maximum(weights - lower_bounds, 0.0)
    if float(surplus.sum()) > 0.0:
        weights = lower_bounds + free_mass * surplus / surplus.sum()
    else:
        weights = lower_bounds + free_mass / len(MODEL_NAMES)
    loss = weighted_platform_error(weights, oof, y_true, fold_weights=fold_weights)
    objective = loss + blend_regularization(weights, prior_weights)
    return {
        "method": "ridge",
        "alpha": float(alpha),
        "weights": weights.tolist(),
        "loss": float(loss),
        "objective": float(objective),
        "regularization_penalty": float(objective - loss),
        "constraint_violations": constraint_violations(weights),
        "raw_coef": model.coef_.tolist(),
    }


def load_oof(candidate_name: str) -> tuple[dict, dict]:
    path = ROOT / f"oof_{candidate_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"OOF cache not found: {path}")
    payload = np.load(path)
    oof: dict[int, dict[str, np.ndarray]] = {}
    y_true: dict[int, np.ndarray] = {}
    for key in payload.files:
        if key.startswith("y_true__"):
            year = int(key.split("__")[1])
            y_true[year] = payload[key]
        elif key.startswith("oof__"):
            _, year_str, model_name = key.split("__")
            year = int(year_str)
            oof.setdefault(year, {})[model_name] = payload[key]
    return oof, y_true


def weighted_platform_error(
    weights: np.ndarray,
    oof: dict,
    y_true: dict,
    fold_weights: dict[int, float] | None = None,
) -> float:
    fold_weights = fold_weights or FOLD_WEIGHTS
    total = 0.0
    for year, fold_weight in fold_weights.items():
        blended = np.zeros_like(y_true[year])
        for i, name in enumerate(MODEL_NAMES):
            blended += weights[i] * oof[year][name]
        mae = np.mean(np.abs(y_true[year] - blended))
        total += fold_weight * mae
    return float(total / predict.P_RATED_FARM * 100.0)


def optimize_weights(
    oof: dict,
    y_true: dict,
    x0: np.ndarray,
    fold_weights: dict[int, float] | None = None,
    prior_weights: np.ndarray | None = None,
) -> dict:
    def loss(w: np.ndarray) -> float:
        return weighted_platform_error(w, oof, y_true, fold_weights=fold_weights) + blend_regularization(
            w,
            prior_weights,
        )

    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    if STABLE_HGB_MIN_WEIGHT > 0.0:
        constraints.append({"type": "ineq", "fun": lambda w: stable_hgb_weight(w) - STABLE_HGB_MIN_WEIGHT})
    if OPTUNA_HGB_MAX_WEIGHT < 1.0:
        constraints.append({"type": "ineq", "fun": lambda w: OPTUNA_HGB_MAX_WEIGHT - optuna_hgb_weight(w)})
    if LGB_MAX_WEIGHT < 1.0:
        constraints.append({"type": "ineq", "fun": lambda w: LGB_MAX_WEIGHT - lgb_weight(w)})
    if TREEBAG_MAX_WEIGHT < 1.0:
        constraints.append({"type": "ineq", "fun": lambda w: TREEBAG_MAX_WEIGHT - treebag_weight(w)})
    if QUANTILE_HGB_MAX_WEIGHT < 1.0:
        constraints.append({"type": "ineq", "fun": lambda w: QUANTILE_HGB_MAX_WEIGHT - quantile_hgb_weight(w)})
    lower_bounds = model_weight_lower_bounds()
    bounds = [(float(bound), 1.0) for bound in lower_bounds]

    result = minimize(
        loss, x0=x0, method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={"ftol": 1e-7, "maxiter": 500},
    )
    weights = np.asarray(result.x, dtype=float)
    pure_loss = weighted_platform_error(weights, oof, y_true, fold_weights=fold_weights)
    objective = pure_loss + blend_regularization(weights, prior_weights)
    return {
        "x0": x0.tolist(),
        "weights": weights.tolist(),
        "loss": float(pure_loss),
        "objective": float(objective),
        "regularization_penalty": float(objective - pure_loss),
        "constraint_violations": constraint_violations(weights),
        "success": bool(result.success),
        "n_iter": int(result.nit),
        "message": str(result.message),
    }


def pick_winner() -> str:
    results_path = ROOT / "phase_c_results.json"
    if not results_path.exists():
        raise FileNotFoundError("phase_c_results.json missing")
    results = json.loads(results_path.read_text())
    if not results:
        raise ValueError("No candidates in phase_c_results.json")
    best = min(results, key=lambda r: r["weighted_q1_cv"])
    return best["name"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize blend weights from saved OOF predictions.")
    parser.add_argument("candidate", nargs="?", default=None, help="OOF candidate name.")
    parser.add_argument(
        "--model-set",
        choices=sorted(predict.MODEL_SETS),
        default=predict.ACTIVE_MODEL_SET,
        help="Model set used when the OOF cache was generated.",
    )
    parser.add_argument(
        "--fold-weights",
        choices=sorted(FOLD_WEIGHT_SCHEMES),
        default="recency",
        help="Fold weighting scheme for the blend objective.",
    )
    parser.add_argument(
        "--blend-weight-lower",
        type=float,
        default=WEIGHT_LOWER,
        help="Lower bound for each model weight in SLSQP/Optuna blend search.",
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
        help="Minimum combined weight for hgb_direct_deep + hgb_residual_smooth.",
    )
    parser.add_argument(
        "--optuna-hgb-max-weight",
        type=float,
        default=None,
        help="Maximum combined weight for hgb_opt_* models.",
    )
    parser.add_argument(
        "--lgb-max-weight",
        type=float,
        default=None,
        help="Maximum combined weight for lgb_* models.",
    )
    parser.add_argument(
        "--treebag-max-weight",
        type=float,
        default=None,
        help="Maximum combined weight for etr_* bagged-tree models.",
    )
    parser.add_argument(
        "--quantile-hgb-max-weight",
        type=float,
        default=1.0,
        help="Maximum combined weight for hgb_q50_* quantile sister models.",
    )
    parser.add_argument(
        "--blend-prior-strength",
        type=float,
        default=None,
        help="L2 penalty strength toward the model-set starting weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_model_names(args.model_set)
    set_weight_lower(args.blend_weight_lower)
    stable_min, optuna_max, lgb_max, treebag_max, prior_strength = resolve_constraint_profile(
        args.blend_constraint_profile,
        args.stable_hgb_min_weight,
        args.optuna_hgb_max_weight,
        args.lgb_max_weight,
        args.treebag_max_weight,
        args.blend_prior_strength,
    )
    set_blend_constraints(stable_min, optuna_max, lgb_max, treebag_max, prior_strength)
    set_quantile_hgb_max_weight(args.quantile_hgb_max_weight)
    fold_weights = resolve_fold_weights(args.fold_weights)

    if args.candidate:
        candidate = args.candidate
    else:
        candidate = pick_winner()
        print(f"Auto-selected winner from phase_c_results.json: {candidate}")

    print(f"\nLoading OOF for candidate: {candidate}")
    oof, y_true = load_oof(candidate)
    for year in sorted(y_true.keys()):
        n_models = len(oof[year])
        n_rows = len(y_true[year])
        print(f"  Q1 {year}: {n_rows} rows, {n_models} models")
    print(f"  model set: {args.model_set} ({len(MODEL_NAMES)} models)")
    print(f"  fold weights: {args.fold_weights} {fold_weights}")
    print(f"  blend weight lower bound: {WEIGHT_LOWER:.4f}")
    print(f"  blend constraints: {format_constraint_summary()}")

    print("\nBaseline weighted platform error with CURRENT blend weights:")
    current_w = np.array([spec.weight for spec in predict.MODEL_SPECS])
    print(f"  current weights: {dict(zip(MODEL_NAMES, current_w.tolist()))}")
    print(f"  sum: {current_w.sum():.4f}")
    baseline_loss = weighted_platform_error(current_w, oof, y_true, fold_weights=fold_weights)
    print(f"  weighted Q1 CV: {baseline_loss:.4f}%")

    starting_points = default_starting_points(MODEL_NAMES, current_w)

    print("\nRunning SLSQP from multiple starting points:")
    runs = {}
    for name, x0 in starting_points.items():
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
    optimal_weights = np.array(best_run["weights"])
    selected_method = "slsqp"
    best_loss = best_run["loss"]
    best_objective = best_run["objective"]
    print(f"\nBest SLSQP run loss: {best_loss:.4f}% objective={best_objective:.4f}")

    print("\nRidge stacking (non-negative, alpha=1.0):")
    ridge_run = stack_ridge(oof, y_true, alpha=1.0, fold_weights=fold_weights, prior_weights=current_w)
    weights_str = ", ".join(f"{n}={w:.3f}" for n, w in zip(MODEL_NAMES, ridge_run["weights"]))
    print(f"  loss={ridge_run['loss']:.4f}% objective={ridge_run['objective']:.4f}  weights: {weights_str}")
    if ridge_run["objective"] < best_objective:
        selected_method = "ridge"
        best_loss = ridge_run["loss"]
        best_objective = ridge_run["objective"]
        optimal_weights = np.array(ridge_run["weights"])

    print(f"\nSelected method: {selected_method}")
    print(f"Improvement vs current: {baseline_loss - best_loss:.4f}% absolute")
    print(f"Constraint violations: {constraint_violations(optimal_weights)}")

    consistency = max(abs(r["objective"] - best_run["objective"]) for r in runs.values())
    print(f"Max loss spread across starting points: {consistency:.4f}%")
    if consistency > 0.05:
        print("  WARNING: starting points diverge — check OOF or constraints")

    output = {
        "candidate": candidate,
        "model_set": args.model_set,
        "fold_weight_scheme": args.fold_weights,
        "fold_weights": fold_weights,
        "blend_weight_lower": WEIGHT_LOWER,
        "blend_constraint_profile": args.blend_constraint_profile,
        "blend_constraints": constraint_summary(),
        "selected_method": selected_method,
        "baseline_weighted_q1_cv": baseline_loss,
        "best_weighted_q1_cv": best_loss,
        "selection_objective": best_objective,
        "regularization_penalty": blend_regularization(optimal_weights, current_w),
        "constraint_violations": constraint_violations(optimal_weights),
        "improvement_abs": baseline_loss - best_loss,
        "current_weights": dict(zip(MODEL_NAMES, current_w.tolist())),
        "optimal_weights": dict(zip(MODEL_NAMES, optimal_weights.tolist())),
        "slsqp_runs": runs,
        "ridge_run": ridge_run,
    }
    out_path = ROOT / "phase_d_results.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
