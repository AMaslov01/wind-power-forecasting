"""Smoke-test optional external source adapters without training.

The script is intentionally fail-soft: every source/date pair reports
available/skipped/failed, and no API key value is printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import predict


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "dataset" / "external_weather"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test optional external source adapters.")
    parser.add_argument("--dates", default="2025-01-15,2026-01-15")
    parser.add_argument(
        "--sources",
        default="",
        help="Comma-separated source names. Defaults to the selected --external-source-set.",
    )
    parser.add_argument(
        "--external-source-set",
        choices=sorted(predict.EXTERNAL_SOURCE_SETS),
        default="strict_open",
        help="Named source set to smoke-test when --sources is not provided.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--external-cache-policy", choices=sorted(predict.EXTERNAL_CACHE_POLICIES), default="cache_first")
    parser.add_argument("--external-source-timeout-sec", type=float, default=predict.DEFAULT_EXTERNAL_SOURCE_TIMEOUT_SECONDS)
    parser.add_argument("--min-coverage", type=float, default=0.80)
    parser.add_argument("--output-path", type=Path, default=ROOT / "external_source_smoke.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    dates = [item.strip() for item in args.dates.split(",") if item.strip()]
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    if not sources:
        sources = list(predict.resolve_external_source_set(args.external_source_set))

    results = {}
    for date in dates:
        results[date] = {}
        for source in sources:
            frame, report = predict._load_optional_external_source(
                source,
                args.cache_dir,
                date,
                date,
                cache_policy=args.external_cache_policy,
                timeout_seconds=args.external_source_timeout_sec,
                min_coverage=args.min_coverage,
            )
            if frame is not None:
                report["preview_columns"] = [col for col in frame.columns if col != predict.DATETIME_COL][:12]
            results[date][source] = report
            status = report.get("status", "skipped")
            reason = report.get("reason", "")
            suffix = f" ({reason})" if reason and status != "available" else ""
            print(f"{date} {source}: {status}{suffix}", flush=True)

    args.output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output_path}")


if __name__ == "__main__":
    main()
