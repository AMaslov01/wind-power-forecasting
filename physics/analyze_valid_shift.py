"""
Compare raw Q1 train folds with the public validation feature distribution.

The public target is hidden, so this does not score candidates directly. It
helps decide whether recency weighting is likely to overfit by reporting which
historical Q1 raw feature distribution looks closest to Q1 2026 validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
TRAIN_PATH = PROJECT_ROOT / "dataset" / "train_dataset.csv"
VALID_PATH = PROJECT_ROOT / "dataset" / "valid_features.csv"
DATETIME_COL = "METEOFORECASTHOUR_OPENM_Datetime"
TARGET = "Выработка. Результирующий расчет"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze raw feature shift between Q1 folds and validation.")
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--valid", type=Path, default=VALID_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "valid_shift_report.json")
    parser.add_argument("--top", type=int, default=12, help="Number of largest shifted columns to print per year.")
    return parser.parse_args()


def q1(frame: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(frame[DATETIME_COL])
    return frame.loc[dt.dt.month <= 3].assign(_year=dt.dt.year).copy()


def numeric_common(train: pd.DataFrame, valid: pd.DataFrame) -> list[str]:
    cols = []
    for col in valid.columns:
        if col in {DATETIME_COL, TARGET} or col not in train.columns:
            continue
        if pd.api.types.is_numeric_dtype(train[col]) and pd.api.types.is_numeric_dtype(valid[col]):
            cols.append(col)
    return cols


def summarize_frame(frame: pd.DataFrame) -> dict:
    out = {"rows": int(len(frame))}
    for col in ["wind_speed_80m", "wind_speed_120m", "temperature_80m", "pressure_msl", "Кол-во_ВЭУ_в_ремонте"]:
        if col in frame:
            values = pd.to_numeric(frame[col], errors="coerce")
            out[f"{col}_mean"] = float(values.mean())
            out[f"{col}_std"] = float(values.std())
    if TARGET in frame:
        out["target_mean"] = float(pd.to_numeric(frame[TARGET], errors="coerce").mean())
        out["target_mae_to_mean"] = float(
            np.mean(np.abs(pd.to_numeric(frame[TARGET], errors="coerce") - out["target_mean"]))
        )
    return out


def compare_year(train_year: pd.DataFrame, valid: pd.DataFrame, cols: list[str], scale: pd.Series, top: int) -> dict:
    train_mean = train_year[cols].mean(numeric_only=True)
    valid_mean = valid[cols].mean(numeric_only=True)
    z = ((train_mean - valid_mean).abs() / scale[cols].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    z = z.dropna().sort_values(ascending=False)

    return {
        "summary": summarize_frame(train_year),
        "mean_abs_z": float(z.mean()),
        "median_abs_z": float(z.median()),
        "p90_abs_z": float(z.quantile(0.90)),
        "top_shifted_means": [
            {
                "column": col,
                "abs_z": float(value),
                "train_mean": float(train_mean[col]),
                "valid_mean": float(valid_mean[col]),
            }
            for col, value in z.head(top).items()
        ],
    }


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train, parse_dates=[DATETIME_COL])
    valid = pd.read_csv(args.valid, parse_dates=[DATETIME_COL])
    train_q1 = q1(train)
    valid_q1 = q1(valid)
    cols = numeric_common(train_q1, valid_q1)
    scale = pd.concat([train_q1[cols], valid_q1[cols]], axis=0).std(numeric_only=True).replace(0.0, np.nan)

    report = {
        "train": str(args.train),
        "valid": str(args.valid),
        "common_numeric_columns": cols,
        "valid_summary": summarize_frame(valid_q1),
        "years": {},
    }
    for year, frame in sorted(train_q1.groupby("_year")):
        report["years"][str(year)] = compare_year(frame, valid_q1, cols, scale, args.top)

    ranked = sorted(report["years"].items(), key=lambda item: item[1]["mean_abs_z"])
    report["closest_by_mean_abs_z"] = [year for year, _ in ranked]
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"Validation shift report saved to {args.output}")
    print("Closest Q1 years by raw mean feature distance:")
    for year, stats in ranked:
        summary = stats["summary"]
        print(
            f"  {year}: mean_abs_z={stats['mean_abs_z']:.3f} "
            f"wind80={summary.get('wind_speed_80m_mean', float('nan')):.3f} "
            f"repair={summary.get('Кол-во_ВЭУ_в_ремонте_mean', float('nan')):.3f}"
        )


if __name__ == "__main__":
    main()
