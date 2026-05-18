"""Validation diagnostics for public-like Q1 2026 behavior.

This validator does not know the hidden target. It checks whether a candidate
prediction moves the same feature regimes that look public-like: Q1 rows,
low repair counts, and mid/high partial-load potential bins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
TRAIN_PATH = PROJECT_ROOT / "dataset" / "train_dataset.csv"
VALID_PATH = PROJECT_ROOT / "dataset" / "valid_features.csv"
TARGET = "Выработка. Результирующий расчет"
DATETIME_COL = "METEOFORECASTHOUR_OPENM_Datetime"
REPAIR_COL = "Кол-во_ВЭУ_в_ремонте"
P_RATED_FARM = 90.09


RAW_FEATURES = [
    "month",
    "hour_of_day",
    REPAIR_COL,
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_120m",
    "wind_speed_180m",
    "wind_direction_80m",
    "wind_direction_120m",
    "wind_gusts_10m",
    "temperature_80m",
    "temperature_120m",
    "pressure_msl",
    "rain",
    "showers",
    "snowfall",
    "cloud_cover_low",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report valid-like Q1 diagnostics for a submission.")
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--candidate-oof", type=Path, default=None)
    parser.add_argument("--baseline-oof", type=Path, default=None)
    parser.add_argument("--candidate-weights-json", type=Path, default=None)
    parser.add_argument("--baseline-weights-json", type=Path, default=None)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--valid", type=Path, default=VALID_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "valid_like_report.json")
    parser.add_argument("--fail-on-bad-direction", action="store_true")
    parser.add_argument("--fail-on-unsupported-transport", action="store_true")
    return parser.parse_args()


def _fill_known_missing(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["wind_direction_10m"] = frame["wind_direction_10m"].fillna(frame["wind_direction_80m"])
    frame["wind_direction_180m"] = frame["wind_direction_180m"].fillna(frame["wind_direction_120m"])
    frame["wind_speed_180m"] = frame["wind_speed_180m"].fillna(frame["wind_speed_120m"])
    return frame


def _potential_proxy(frame: pd.DataFrame) -> pd.Series:
    ws_eq = (
        0.10 * frame["wind_speed_10m"] ** 3
        + 0.45 * frame["wind_speed_80m"] ** 3
        + 0.30 * frame["wind_speed_120m"] ** 3
        + 0.15 * frame["wind_speed_180m"] ** 3
    ).clip(lower=0.0) ** (1.0 / 3.0)

    v_cut_in = 2.3393477705815173
    v_rated = 10.83096885835045
    p_rated_per_turbine = 3.3260919549786356
    cubic_exponent = 1.498293155285144
    efficiency_factor = 1.119626992504275

    p_per_turbine = np.zeros(len(frame), dtype=float)
    partial = (ws_eq >= v_cut_in) & (ws_eq < v_rated)
    p_per_turbine[partial] = p_rated_per_turbine * (
        (ws_eq[partial] - v_cut_in) / (v_rated - v_cut_in)
    ) ** cubic_exponent
    p_per_turbine[ws_eq >= v_rated] = p_rated_per_turbine
    n_working = 26.0 - pd.to_numeric(frame[REPAIR_COL], errors="coerce")
    return pd.Series(np.clip(p_per_turbine * efficiency_factor * n_working, 0.0, P_RATED_FARM))


def _read_prediction(path: Path, valid_len: int) -> pd.Series:
    frame = pd.read_csv(path)
    if list(frame.columns) != [TARGET]:
        raise ValueError(f"{path} must contain only {TARGET!r}, got {list(frame.columns)}")
    if len(frame) != valid_len:
        raise ValueError(f"{path} row count {len(frame)} != valid rows {valid_len}")
    values = pd.to_numeric(frame[TARGET], errors="raise")
    if values.isna().any() or (values < 0).any() or (values > P_RATED_FARM).any():
        raise ValueError(f"{path} contains NaN or out-of-bound predictions")
    return values.astype(float)


def _similarity_report(train: pd.DataFrame, valid: pd.DataFrame) -> dict:
    train_q1 = train.loc[pd.to_datetime(train[DATETIME_COL]).dt.month <= 3].copy()
    valid_q1 = valid.loc[pd.to_datetime(valid[DATETIME_COL]).dt.month <= 3].copy()
    cols = [c for c in RAW_FEATURES if c in train_q1 and c in valid_q1]
    x = pd.concat([train_q1[cols], valid_q1[cols]], ignore_index=True)
    y = np.r_[np.zeros(len(train_q1)), np.ones(len(valid_q1))]
    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=18,
        min_samples_leaf=60,
        l2_regularization=0.3,
        random_state=42,
    )
    model.fit(x, y)
    proba = model.predict_proba(train_q1[cols])[:, 1].clip(1e-4, 0.999)
    weights = proba / (1.0 - proba)
    auc = float(roc_auc_score(y, model.predict_proba(x)[:, 1]))
    train_q1 = train_q1.assign(_valid_like_weight=weights, _year=pd.to_datetime(train_q1[DATETIME_COL]).dt.year)
    by_year = {}
    for year, group in train_q1.groupby("_year"):
        by_year[str(int(year))] = {
            "rows": int(len(group)),
            "mean_weight": float(group["_valid_like_weight"].mean()),
            "target_mean": float(group[TARGET].mean()),
            "repair_mean": float(group[REPAIR_COL].mean()),
            "wind80_mean": float(group["wind_speed_80m"].mean()),
        }
    return {"auc": auc, "by_year": by_year}


def _prediction_report(valid: pd.DataFrame, pred: pd.Series, baseline: pd.Series | None) -> dict:
    valid = valid.copy()
    valid["potential_proxy"] = _potential_proxy(valid)
    edges = np.unique(np.quantile(valid["potential_proxy"], np.linspace(0.0, 1.0, 10)))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    bins = pd.IntervalIndex.from_breaks(edges, closed="right")
    valid["pbin"] = pd.cut(valid["potential_proxy"], bins=bins, include_lowest=True)
    valid["prediction"] = pred.values
    if baseline is not None:
        valid["baseline"] = baseline.values
        valid["lift"] = valid["prediction"] - valid["baseline"]

    pbin_rows = []
    for idx, interval in enumerate(bins):
        group = valid.loc[valid["pbin"] == interval]
        if group.empty:
            continue
        row = {
            "bin": idx,
            "rows": int(len(group)),
            "prediction_mean": float(group["prediction"].mean()),
            "potential_mean": float(group["potential_proxy"].mean()),
            "repair_mean": float(group[REPAIR_COL].mean()),
        }
        if baseline is not None:
            row["lift_mean"] = float(group["lift"].mean())
        pbin_rows.append(row)

    month_rows = []
    for month, group in valid.groupby("month"):
        row = {
            "month": int(month),
            "rows": int(len(group)),
            "prediction_mean": float(group["prediction"].mean()),
        }
        if baseline is not None:
            row["lift_mean"] = float(group["lift"].mean())
        month_rows.append(row)

    repair_rows = []
    for repair_count, group in valid.groupby(REPAIR_COL):
        row = {
            "repair_count": int(repair_count),
            "rows": int(len(group)),
            "prediction_mean": float(group["prediction"].mean()),
        }
        if baseline is not None:
            row["lift_mean"] = float(group["lift"].mean())
        repair_rows.append(row)

    return {
        "prediction_mean": float(pred.mean()),
        "prediction_min": float(pred.min()),
        "prediction_max": float(pred.max()),
        "pbin": pbin_rows,
        "month": month_rows,
        "repair": repair_rows,
    }


def _load_weight_map(path: Path | None, model_names: list[str]) -> dict[str, float]:
    if path is None:
        return {name: 1.0 / len(model_names) for name in model_names}
    payload = json.loads(path.read_text())
    weights = (
        payload.get("optimal_weights")
        or payload.get("phase_d", {}).get("optimal_weights")
        or payload.get("validator_weights", {}).get("final")
    )
    if not isinstance(weights, dict):
        return {name: 1.0 / len(model_names) for name in model_names}
    selected = {name: float(weights.get(name, 0.0)) for name in model_names}
    total = sum(max(value, 0.0) for value in selected.values())
    if total <= 0.0:
        return {name: 1.0 / len(model_names) for name in model_names}
    return {name: max(value, 0.0) / total for name, value in selected.items()}


def _load_oof_blend(path: Path, weights_json: Path | None) -> dict[int, np.ndarray]:
    payload = np.load(path)
    by_year: dict[int, dict[str, np.ndarray]] = {}
    for key in payload.files:
        if not key.startswith("oof__"):
            continue
        _, year_str, model_name = key.split("__", 2)
        by_year.setdefault(int(year_str), {})[model_name] = payload[key]
    out: dict[int, np.ndarray] = {}
    for year, preds_by_model in by_year.items():
        model_names = sorted(preds_by_model)
        weights = _load_weight_map(weights_json, model_names)
        blended = np.zeros_like(next(iter(preds_by_model.values())), dtype=float)
        for model_name in model_names:
            blended += weights[model_name] * preds_by_model[model_name]
        out[year] = blended
    return out


def _q1_by_year(train: pd.DataFrame) -> dict[int, pd.DataFrame]:
    train = train.sort_values(DATETIME_COL).copy()
    dt = pd.to_datetime(train[DATETIME_COL])
    q1 = train.loc[dt.dt.month <= 3].assign(_year=dt.dt.year).copy()
    return {int(year): group.reset_index(drop=True) for year, group in q1.groupby("_year")}


def _assign_pbin(frame: pd.DataFrame) -> pd.Series:
    potential = _potential_proxy(frame)
    edges = np.unique(np.quantile(potential, np.linspace(0.0, 1.0, 10)))
    if len(edges) < 10:
        low = float(potential.min())
        high = float(potential.max())
        if high <= low:
            high = low + 1.0
        edges = np.linspace(low - 1e-6, high + 1e-6, 10)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return pd.Series(np.digitize(potential, edges[1:-1], right=True).clip(0, 8), index=frame.index)


def _group_delta_means(frame: pd.DataFrame, delta: np.ndarray) -> dict[str, dict[str, dict[str, float]]]:
    work = frame.copy()
    work["_delta"] = np.asarray(delta, dtype=float)
    work["_pbin"] = _assign_pbin(work).astype(int)
    groups = {
        "pbin": "_pbin",
        "month": "month",
        "repair": REPAIR_COL,
    }
    out: dict[str, dict[str, dict[str, float]]] = {}
    for group_name, col in groups.items():
        rows = {}
        for value, group in work.groupby(col):
            rows[str(int(value))] = {
                "rows": int(len(group)),
                "mean_delta": float(group["_delta"].mean()),
                "mean_abs_delta": float(group["_delta"].abs().mean()),
            }
        out[group_name] = rows
    return out


def _transport_guard_report(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    valid_delta: np.ndarray,
    candidate_oof: Path | None,
    baseline_oof: Path | None,
    candidate_weights_json: Path | None,
    baseline_weights_json: Path | None,
) -> dict | None:
    if candidate_oof is None or baseline_oof is None:
        return None
    cand = _load_oof_blend(candidate_oof, candidate_weights_json)
    base = _load_oof_blend(baseline_oof, baseline_weights_json)
    train_years = _q1_by_year(train)
    valid_groups = _group_delta_means(valid, valid_delta)

    historical: dict[str, dict[str, list[dict[str, float]]]] = {"pbin": {}, "month": {}, "repair": {}}
    for year in sorted(set(cand) & set(base) & set(train_years)):
        frame = train_years[year]
        if len(frame) != len(cand[year]) or len(frame) != len(base[year]):
            continue
        yearly = _group_delta_means(frame, cand[year] - base[year])
        for group_name, rows in yearly.items():
            for key, row in rows.items():
                historical[group_name].setdefault(key, []).append(row)

    rows = []
    flags = []
    for group_name, valid_rows in valid_groups.items():
        for key, valid_row in sorted(valid_rows.items(), key=lambda item: int(item[0])):
            hist_rows = historical[group_name].get(key, [])
            if not hist_rows:
                continue
            hist_mean = float(np.average(
                [row["mean_delta"] for row in hist_rows],
                weights=[row["rows"] for row in hist_rows],
            ))
            hist_abs = float(np.average(
                [row["mean_abs_delta"] for row in hist_rows],
                weights=[row["rows"] for row in hist_rows],
            ))
            valid_mean = float(valid_row["mean_delta"])
            allowed_abs = max(0.35, 1.75 * hist_abs + 0.10)
            unsupported = abs(valid_mean) > allowed_abs
            sign_mismatch = abs(valid_mean) > 0.15 and hist_mean * valid_mean < -0.03
            row = {
                "group": group_name,
                "key": int(key),
                "valid_mean_delta": valid_mean,
                "historical_mean_delta": hist_mean,
                "historical_mean_abs_delta": hist_abs,
                "allowed_abs_delta": allowed_abs,
                "unsupported": bool(unsupported),
                "sign_mismatch": bool(sign_mismatch),
            }
            rows.append(row)
            if unsupported or sign_mismatch:
                flags.append(row)

    return {
        "candidate_oof": str(candidate_oof),
        "baseline_oof": str(baseline_oof),
        "rows": rows,
        "flags": flags,
        "passed": not flags,
    }


def main() -> None:
    args = parse_args()
    train = _fill_known_missing(pd.read_csv(args.train, parse_dates=[DATETIME_COL]))
    valid = _fill_known_missing(pd.read_csv(args.valid, parse_dates=[DATETIME_COL]))
    pred = _read_prediction(args.prediction, len(valid))
    baseline = _read_prediction(args.baseline, len(valid)) if args.baseline else None
    transport_guard = None
    if baseline is not None:
        transport_guard = _transport_guard_report(
            train,
            valid,
            (pred - baseline).to_numpy(dtype=float),
            args.candidate_oof,
            args.baseline_oof,
            args.candidate_weights_json,
            args.baseline_weights_json,
        )

    report = {
        "prediction": str(args.prediction),
        "baseline": str(args.baseline) if args.baseline else None,
        "similarity": _similarity_report(train, valid),
        "diagnostics": _prediction_report(valid, pred, baseline),
        "transport_guard": transport_guard,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"valid-like report saved to {args.output}")
    print(f"propensity AUC train-Q1 vs valid-Q1: {report['similarity']['auc']:.3f}")
    diagnostics = report["diagnostics"]
    print(
        f"prediction mean={diagnostics['prediction_mean']:.3f} "
        f"min={diagnostics['prediction_min']:.3f} max={diagnostics['prediction_max']:.3f}"
    )
    for row in diagnostics["pbin"]:
        lift = f" lift={row['lift_mean']:+.3f}" if "lift_mean" in row else ""
        print(
            f"pbin{row['bin']:02d}: rows={row['rows']} "
            f"pred={row['prediction_mean']:.3f} potential={row['potential_mean']:.3f}{lift}"
        )
    for row in diagnostics["repair"]:
        lift = f" lift={row['lift_mean']:+.3f}" if "lift_mean" in row else ""
        print(f"repair{row['repair_count']}: rows={row['rows']} pred={row['prediction_mean']:.3f}{lift}")
    if transport_guard is not None:
        print(
            "transport guard: "
            + ("passed" if transport_guard["passed"] else f"{len(transport_guard['flags'])} unsupported groups")
        )

    if args.fail_on_bad_direction and baseline is not None:
        pbin = {row["bin"]: row.get("lift_mean", 0.0) for row in diagnostics["pbin"]}
        month = {row["month"]: row.get("lift_mean", 0.0) for row in diagnostics["month"]}
        mid_lift = float(np.mean([pbin.get(i, 0.0) for i in (4, 5, 6)]))
        if pbin.get(8, 0.0) > max(0.5, mid_lift):
            raise SystemExit("bad direction: pbin08 lift dominates mid-power lift")
        if pbin.get(6, 0.0) < -0.05:
            raise SystemExit("bad direction: pbin06 is negative")
        if month.get(1, 0.0) < -0.05:
            raise SystemExit("bad direction: month01 is negative")
    if args.fail_on_unsupported_transport and transport_guard is not None and not transport_guard["passed"]:
        first = transport_guard["flags"][0]
        raise SystemExit(
            "unsupported transport: "
            f"{first['group']}={first['key']} valid_delta={first['valid_mean_delta']:+.3f} "
            f"historical={first['historical_mean_delta']:+.3f}"
        )


if __name__ == "__main__":
    main()
