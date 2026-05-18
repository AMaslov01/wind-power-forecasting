"""Validate final wind-power submission artifacts.

The checker is intentionally strict and user-facing: it reports missing files
and malformed outputs before reviewers hit a deep training traceback.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POST_Q1_NAME = "3888f9f2-9bda-4b2c-94af-5562668bce86_test_dataset.csv"
TARGET_COL = "Выработка. Результирующий расчет"
P_RATED_FARM = 90.09
EXPECTED_ROWS = {
    "train": 32434,
    "valid": 2126,
    "post_q1": 1152,
    "q1_prediction": 2126,
    "may18_prediction": 24,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            text=True,
            timeout=2,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def require_file(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_dataset(path: Path, label: str, expected_rows: int) -> dict[str, Any]:
    info = require_file(path, label)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        try:
            next(reader)
        except StopIteration as exc:
            raise ValueError(f"{label} is empty: {path}") from exc
        rows = sum(1 for _ in reader)
    if rows != expected_rows:
        raise ValueError(f"{label} row count mismatch: expected {expected_rows}, got {rows}")
    info["rows"] = rows
    return info


def validate_prediction(path: Path, label: str, expected_rows: int) -> dict[str, Any]:
    info = require_file(path, label)
    values: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{label} is empty: {path}") from exc
        if len(header) != 1:
            raise ValueError(f"{label} must have one column, got {len(header)}")
        for line_no, row in enumerate(reader, start=2):
            if len(row) != 1:
                raise ValueError(f"{label} row {line_no} must have one value, got {len(row)}")
            raw = row[0].strip()
            if not raw:
                raise ValueError(f"{label} row {line_no} is empty")
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError(f"{label} row {line_no} is not numeric: {raw!r}") from exc
            if value != value:
                raise ValueError(f"{label} row {line_no} contains NaN")
            if value < 0.0 or value > P_RATED_FARM:
                raise ValueError(
                    f"{label} row {line_no} is outside physical bounds [0, {P_RATED_FARM}]: {value}"
                )
            values.append(value)
    if len(values) != expected_rows:
        raise ValueError(f"{label} row count mismatch: expected {expected_rows}, got {len(values)}")
    min_value = min(values)
    max_value = max(values)
    mean_value = sum(values) / len(values)
    info.update(
        {
            "rows": int(len(values)),
            "column": str(header[0]),
            "min": float(min_value),
            "max": float(max_value),
            "mean": float(mean_value),
            "zeros": int(sum(1 for value in values if value == 0.0)),
        }
    )
    return info


def validate_artifact_dir(path: Path) -> dict[str, Any]:
    manifest_path = path / "ensemble_manifest.json"
    info = require_file(manifest_path, "model artifact manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    info["artifact_kind"] = manifest.get("artifact_kind")
    info["candidate"] = manifest.get("candidate")
    for filename in ("predictions_q1.csv", "predictions_may18.csv"):
        require_file(path / filename, f"artifact {filename}")
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate final solution files.")
    parser.add_argument("--q1-path", type=Path, default=PROJECT_ROOT / "outputs" / "predictions_q1.csv")
    parser.add_argument("--may18-path", type=Path, default=PROJECT_ROOT / "outputs" / "predictions_may18.csv")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "model_weights")
    parser.add_argument("--report-path", type=Path, default=PROJECT_ROOT / "outputs" / "RUN_REPORT.json")
    parser.add_argument("--mode", choices=["infer", "retrain", "check"], default="infer")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--check-artifacts", action="store_true")
    parser.add_argument("--skip-output-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root if args.dataset_root.is_absolute() else PROJECT_ROOT / args.dataset_root
    artifact_dir = args.artifact_dir if args.artifact_dir.is_absolute() else PROJECT_ROOT / args.artifact_dir
    report: dict[str, Any] = {
        "mode": args.mode,
        "created_at_unix": time.time(),
        "git_commit": git_commit(),
        "python": sys.version,
        "packages": {
            name: package_version(name)
            for name in ["numpy", "pandas", "scikit-learn", "catboost", "scipy", "optuna", "meteostat", "lightgbm", "pypdf"]
        },
        "inputs": {},
        "outputs": {},
    }
    if args.check_data:
        report["inputs"]["train"] = validate_dataset(dataset_root / "train_dataset.csv", "training dataset", EXPECTED_ROWS["train"])
        report["inputs"]["valid"] = validate_dataset(dataset_root / "valid_features.csv", "Q1 feature dataset", EXPECTED_ROWS["valid"])
        report["inputs"]["post_q1"] = validate_dataset(dataset_root / DEFAULT_POST_Q1_NAME, "post-Q1 operational dataset", EXPECTED_ROWS["post_q1"])
    if args.check_artifacts:
        report["artifact_dir"] = validate_artifact_dir(artifact_dir)
    if not args.skip_output_validation:
        report["outputs"]["q1"] = validate_prediction(args.q1_path, "Q1 predictions", EXPECTED_ROWS["q1_prediction"])
        report["outputs"]["may18"] = validate_prediction(args.may18_path, "May18 predictions", EXPECTED_ROWS["may18_prediction"])

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Validation OK: wrote {args.report_path}")
    if report["outputs"]:
        q1 = report["outputs"]["q1"]
        may18 = report["outputs"]["may18"]
        print(
            "  Q1 rows={rows} mean={mean:.3f} min={min:.3f} max={max:.3f} zeros={zeros} sha256={sha256}".format(
                **q1
            )
        )
        print(
            "  May18 rows={rows} mean={mean:.3f} min={min:.3f} max={max:.3f} zeros={zeros} sha256={sha256}".format(
                **may18
            )
        )


if __name__ == "__main__":
    main()
