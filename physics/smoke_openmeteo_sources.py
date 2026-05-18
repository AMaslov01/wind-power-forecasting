"""Smoke-test Open-Meteo Historical Forecast model/date coverage.

This is a no-training utility for checking whether a source/date/model returns
usable hourly wind fields before spending Colab GPU time on an ablation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
MODEL_CATALOG = {
    "gfs": "gfs_global",
    "icon": "icon_global",
    "ecmwf": "ecmwf_ifs025",
    "icon_eu": "icon_eu",
    "cma": "cma_grapes_global",
    "arpege": "arpege_world",
}
SURFACE_VARS = [
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_100m",
    "wind_direction_100m",
    "surface_pressure",
    "temperature_2m",
]
PRESSURE_VARS = [
    "wind_speed_1000hPa",
    "wind_direction_1000hPa",
    "temperature_1000hPa",
    "geopotential_height_1000hPa",
    "wind_speed_925hPa",
    "wind_direction_925hPa",
    "temperature_925hPa",
    "geopotential_height_925hPa",
    "wind_speed_900hPa",
    "wind_direction_900hPa",
    "temperature_900hPa",
    "geopotential_height_900hPa",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Open-Meteo Historical Forecast sources.")
    parser.add_argument("--dates", default="2025-01-15,2026-01-15")
    parser.add_argument("--models", default=",".join(MODEL_CATALOG))
    parser.add_argument("--include-pressure", action="store_true")
    parser.add_argument("--latitude", type=float, default=47.107)
    parser.add_argument("--longitude", type=float, default=39.423)
    parser.add_argument("--output-path", type=Path, default=ROOT / "openmeteo_source_smoke.json")
    return parser.parse_args()


def _fetch(model: str, date: str, hourly: list[str], latitude: float, longitude: float) -> dict:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date,
        "end_date": date,
        "hourly": ",".join(hourly),
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        "models": model,
    }
    url = f"{API_URL}?{urlencode(params)}"
    with urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    hourly_payload = payload.get("hourly", {})
    return {
        "status": "ok",
        "hours": len(hourly_payload.get("time", [])),
        "variables": {
            variable: sum(value is not None for value in hourly_payload.get(variable, []))
            for variable in hourly
        },
    }


def main() -> None:
    args = parse_args()
    dates = [item.strip() for item in args.dates.split(",") if item.strip()]
    model_keys = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = [item for item in model_keys if item not in MODEL_CATALOG]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    hourly = list(SURFACE_VARS)
    if args.include_pressure:
        hourly.extend(PRESSURE_VARS)

    results = {}
    for date in dates:
        results[date] = {}
        for key in model_keys:
            try:
                results[date][key] = _fetch(
                    MODEL_CATALOG[key],
                    date,
                    hourly,
                    latitude=args.latitude,
                    longitude=args.longitude,
                )
            except Exception as exc:
                results[date][key] = {"status": "failed", "error": repr(exc)}
            print(f"{date} {key}: {results[date][key]['status']}", flush=True)

    args.output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output_path}")


if __name__ == "__main__":
    main()
