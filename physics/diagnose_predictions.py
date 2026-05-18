"""
Small diagnostics for candidate OOF and prediction files.

Reports per-model Q1 OOF errors, model correlations, selected blend weights,
prediction shape, and optional deviation from a reference prediction file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TARGET = "Выработка. Результирующий расчет"
P_RATED_FARM = 90.09


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose OOF and final prediction shape.")
    parser.add_argument("--candidate", required=True, help="Candidate name used in oof_<candidate>.npz.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "predictions.csv",
        help="Prediction CSV to summarize.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional reference prediction CSV for shape/deviation comparison.",
    )
    parser.add_argument(
        "--phase-d-path",
        type=Path,
        default=ROOT / "phase_d_results.json",
        help="Phase D JSON containing selected blend weights.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Where to write diagnostics JSON. Defaults to diagnostics_<candidate>.json.",
    )
    return parser.parse_args()


def platform_error_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)) / P_RATED_FARM * 100.0)


def load_oof(candidate: str) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, np.ndarray]]:
    path = ROOT / f"oof_{candidate}.npz"
    if not path.exists():
        raise FileNotFoundError(f"OOF cache not found: {path}")
    payload = np.load(path)
    oof: dict[int, dict[str, np.ndarray]] = {}
    y_true: dict[int, np.ndarray] = {}
    for key in payload.files:
        if key.startswith("y_true__"):
            y_true[int(key.split("__")[1])] = payload[key]
        elif key.startswith("oof__"):
            _, year_str, model_name = key.split("__")
            oof.setdefault(int(year_str), {})[model_name] = payload[key]
    return oof, y_true


def prediction_stats(path: Path) -> dict:
    frame = pd.read_csv(path)
    if frame.shape[1] != 1:
        raise ValueError(f"{path} must contain exactly one prediction column.")
    values = pd.to_numeric(frame.iloc[:, 0], errors="raise")
    return {
        "path": str(path),
        "rows": int(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "zeros": int((values == 0).sum()),
    }


def reference_delta(predictions: Path, reference: Path) -> dict:
    pred = pd.to_numeric(pd.read_csv(predictions).iloc[:, 0], errors="raise").to_numpy(dtype=float)
    ref = pd.to_numeric(pd.read_csv(reference).iloc[:, 0], errors="raise").to_numpy(dtype=float)
    if len(pred) != len(ref):
        raise ValueError(f"Prediction/reference row mismatch: {len(pred)} vs {len(ref)}")
    delta = pred - ref
    return {
        "reference": str(reference),
        "mean_delta": float(delta.mean()),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "max_abs_delta": float(np.max(np.abs(delta))),
        "corr": float(np.corrcoef(pred, ref)[0, 1]),
        "prediction_mean": float(pred.mean()),
        "reference_mean": float(ref.mean()),
    }


def oof_report(oof: dict[int, dict[str, np.ndarray]], y_true: dict[int, np.ndarray]) -> dict:
    years = sorted(y_true)
    model_names = sorted({name for by_model in oof.values() for name in by_model})
    per_year = {}
    all_preds = {name: [] for name in model_names}
    all_targets = []

    for year in years:
        per_year[str(year)] = {}
        all_targets.append(y_true[year])
        for name in model_names:
            if name not in oof[year]:
                continue
            pred = oof[year][name]
            all_preds[name].append(pred)
            per_year[str(year)][name] = {
                "platform_error": platform_error_percent(y_true[year], pred),
                "mean": float(np.mean(pred)),
                "min": float(np.min(pred)),
                "max": float(np.max(pred)),
            }

    y_all = np.concatenate(all_targets)
    overall = {}
    matrix_cols = []
    present_names = []
    for name in model_names:
        if len(all_preds[name]) != len(years):
            continue
        pred_all = np.concatenate(all_preds[name])
        overall[name] = {
            "platform_error": platform_error_percent(y_all, pred_all),
            "mean": float(np.mean(pred_all)),
            "min": float(np.min(pred_all)),
            "max": float(np.max(pred_all)),
        }
        matrix_cols.append(pred_all)
        present_names.append(name)

    corr = {}
    if matrix_cols:
        corr_matrix = np.corrcoef(np.column_stack(matrix_cols), rowvar=False)
        for i, name_i in enumerate(present_names):
            corr[name_i] = {
                name_j: float(corr_matrix[i, j])
                for j, name_j in enumerate(present_names)
            }

    return {
        "per_year": per_year,
        "overall": overall,
        "correlation": corr,
    }


def load_phase_d(path: Path, candidate: str) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("candidate") not in {None, candidate}:
        payload["warning"] = f"phase_d candidate is {payload.get('candidate')!r}, not {candidate!r}"
    return {
        "path": str(path),
        "candidate": payload.get("candidate"),
        "selected_method": payload.get("selected_method"),
        "best_weighted_q1_cv": payload.get("best_weighted_q1_cv"),
        "optimal_weights": payload.get("optimal_weights"),
        "warning": payload.get("warning"),
    }


def main() -> None:
    args = parse_args()
    output_path = args.output_path or (ROOT / f"diagnostics_{args.candidate}.json")
    oof, y_true = load_oof(args.candidate)
    report = {
        "candidate": args.candidate,
        "oof": oof_report(oof, y_true),
        "phase_d": load_phase_d(args.phase_d_path, args.candidate),
        "predictions": prediction_stats(args.predictions),
        "reference_delta": reference_delta(args.predictions, args.reference) if args.reference else None,
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"Diagnostics saved to {output_path}")
    print("Prediction shape:", report["predictions"])
    if report["reference_delta"]:
        print("Reference delta:", report["reference_delta"])
    best = sorted(
        report["oof"]["overall"].items(),
        key=lambda item: item[1]["platform_error"],
    )[:8]
    print("Top OOF models by unweighted overall Q1 error:")
    for name, stats in best:
        print(f"  {name:24s} {stats['platform_error']:.4f}% mean={stats['mean']:.3f}")


if __name__ == "__main__":
    main()
