"""Restore validated reviewer predictions from the final artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FALLBACK_ARTIFACT_DIR = PROJECT_ROOT / "deliverables" / "final_solution_v14_nocds" / "model_weights"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_artifact_dir(path: Path) -> Path:
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(PROJECT_ROOT / path)
    candidates.append(PROJECT_ROOT / "model_weights")
    candidates.append(DEFAULT_FALLBACK_ARTIFACT_DIR)
    for candidate in candidates:
        if (candidate / "ensemble_manifest.json").exists():
            return candidate
    searched = "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "No artifact bundle found. Expected ensemble_manifest.json in:\n"
        f"  - {searched}\n"
        "Run `bash run_solution.sh --retrain` to rebuild it, or put the bundled model_weights/ directory here."
    )


def copy_checked(src: Path, dst: Path, expected_sha256: str | None) -> dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(f"Missing artifact prediction snapshot: {src}")
    digest = sha256_file(src)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(
            f"Artifact checksum mismatch for {src.name}: expected {expected_sha256}, got {digest}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"path": str(dst), "sha256": digest, "bytes": dst.stat().st_size}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy validated predictions from a final artifact bundle.")
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "model_weights")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--q1-output", type=Path, default=None)
    parser.add_argument("--may18-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = resolve_artifact_dir(args.artifact_dir)
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    q1_output = args.q1_output or (output_dir / "predictions_q1.csv")
    may18_output = args.may18_output or (output_dir / "predictions_may18.csv")
    manifest_path = artifact_dir / "ensemble_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs", {})
    q1_expected = (outputs.get("q1") or {}).get("sha256")
    may18_expected = (outputs.get("may18") or {}).get("sha256")
    report = {
        "mode": "artifact_restore",
        "created_at_unix": time.time(),
        "artifact_dir": str(artifact_dir),
        "manifest_sha256": sha256_file(manifest_path),
        "candidate": manifest.get("candidate"),
        "restored": {
            "q1": copy_checked(artifact_dir / "predictions_q1.csv", q1_output, q1_expected),
            "may18": copy_checked(artifact_dir / "predictions_may18.csv", may18_output, may18_expected),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "ARTIFACT_INFERENCE_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Restored predictions from {artifact_dir}")
    print(f"  Q1: {q1_output}")
    print(f"  May18: {may18_output}")
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
