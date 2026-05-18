"""
Export fold-safe WindFM diagnostic predictions for the main V8 pipeline.

This script is intentionally separate from run_full_pipeline.py because WindFM
has heavy optional dependencies and downloads HuggingFace weights. It writes:

  - physics/windfm_oof_predictions.csv
  - physics/windfm_valid_predictions.csv

The main pipeline can then join these files with --enable-windfm-diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import predict


ROOT = Path(__file__).resolve().parent
INPUT_FEATURES = ["wind_speed", "wind_direction", "power", "density", "temperature", "pressure"]


def _absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (ROOT.parent / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-preset", default="public_best", choices=sorted(predict.PHYSICS_PRESETS))
    parser.add_argument("--windfm-repo-path", default="/content/WindFM")
    parser.add_argument("--model-id", default="NeoQuasar/WindFM")
    parser.add_argument("--tokenizer-id", default="NeoQuasar/WindFM-Tokenizer")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--lookback", type=int, default=240)
    parser.add_argument("--pred-len", type=int, default=80)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--stat", choices=["median", "mean", "p10", "p90"], default="median")
    parser.add_argument("--input-source", choices=["auto", "nasa", "openm"], default="auto")
    parser.add_argument("--output-scale", choices=["normalized", "mw", "auto"], default="normalized")
    parser.add_argument("--train-output-path", default=str(ROOT / "windfm_oof_predictions.csv"))
    parser.add_argument("--valid-output-path", default=str(ROOT / "windfm_valid_predictions.csv"))
    parser.add_argument("--verbose-windfm", action="store_true")

    parser.add_argument("--external-weather-cache-dir", type=Path, default=predict.EXTERNAL_WEATHER_DIR)
    parser.add_argument("--external-source-set", default="strict_open", choices=sorted(predict.EXTERNAL_SOURCE_SETS))
    parser.add_argument("--external-cache-policy", default="cache_first", choices=sorted(predict.EXTERNAL_CACHE_POLICIES))
    parser.add_argument("--external-source-timeout-sec", type=float, default=predict.DEFAULT_EXTERNAL_SOURCE_TIMEOUT_SECONDS)
    parser.add_argument("--require-extra-sources", action="store_true")
    parser.add_argument("--openmeteo-forecast-set", default="gfs_only", choices=sorted(predict.OPENMETEO_FORECAST_SETS))
    parser.add_argument("--openmeteo-fill-policy", default="legacy", choices=sorted(predict.OPENMETEO_FILL_POLICIES))
    parser.add_argument("--openmeteo-min-coverage", type=float, default=predict.OPENMETEO_COVERAGE_MIN_DEFAULT)
    parser.add_argument("--include-openmeteo-pressure", action="store_true")
    parser.add_argument("--skip-openmeteo-forecast", action="store_true")
    parser.add_argument("--skip-meteostat", action="store_true")
    parser.add_argument("--meteostat-cache-path", type=Path, default=predict.METEOSTAT_CACHE_PATH)
    parser.add_argument("--allow-partial-meteostat", action="store_true", default=True)
    parser.add_argument("--include-so-ups-res", action="store_true")
    parser.add_argument("--so-ups-res-cache-path", type=Path, default=predict.SO_UPS_RES_CACHE_PATH)
    parser.add_argument("--so-ups-res-policy", default="legacy_best_gap", choices=sorted(predict.SO_UPS_RES_POLICIES))
    parser.add_argument("--so-ups-res-exclude-months", default=None)
    return parser.parse_args()


def _load_predictor(args: argparse.Namespace):
    repo_path = Path(args.windfm_repo_path)
    if not repo_path.exists():
        raise RuntimeError(
            f"WindFM repo not found at {repo_path}. In Colab run:\n"
            f"  !git clone --depth 1 https://github.com/shiyu-coder/WindFM.git {repo_path}\n"
            f"  !pip install -q -r {repo_path}/requirements.txt"
        )
    sys.path.insert(0, str(repo_path.resolve()))
    try:
        from model import WindFM, WindFMTokenizer, WindFMPredictor
    except Exception as exc:
        raise RuntimeError(
            "Could not import WindFM. Install it first:\n"
            f"  !pip install -q -r {repo_path}/requirements.txt"
        ) from exc

    tokenizer = _load_windfm_module(WindFMTokenizer, args.tokenizer_id)
    model = _load_windfm_module(WindFM, args.model_id)
    return WindFMPredictor(
        model,
        tokenizer,
        device=args.device,
        max_context=args.max_context,
        clip=args.clip,
    )


def _load_windfm_module(module_cls, repo_id: str):
    try:
        return module_cls.from_pretrained(repo_id)
    except TypeError as exc:
        # WindFM's README uses from_pretrained(), but some huggingface_hub
        # versions instantiate PyTorchModelHubMixin classes before applying
        # config.json. Fall back to an explicit config + safetensors load.
        if "required positional argument" not in str(exc):
            raise

    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "WindFM manual load fallback needs huggingface_hub and safetensors. "
            "Run: !pip install -q huggingface_hub safetensors"
        ) from exc

    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    config = json.loads(Path(config_path).read_text())
    module = module_cls(**config)
    state_dict = load_file(weights_path)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected WindFM weights for {repo_id}: {unexpected[:10]}")
    if missing:
        print(f"  warning: WindFM {repo_id} missing weights: {missing[:10]}", flush=True)
    return module


def _local_to_utc(series: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(series)
        .dt.tz_localize(predict.LOCAL_TIMEZONE)
        .dt.tz_convert("UTC")
    )


def _direction_degrees(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if not finite.empty and finite.abs().quantile(0.95) <= 1.1:
        numeric = numeric * 360.0
    return numeric % 360.0


def _safe_numeric(frame: pd.DataFrame, col: str, default: float | None = None) -> pd.Series:
    if col in frame.columns:
        values = pd.to_numeric(frame[col], errors="coerce")
    else:
        values = pd.Series(np.nan, index=frame.index, dtype=float)
    if default is not None:
        values = values.fillna(default)
    return values


def _select_source(frame: pd.DataFrame, requested: str) -> str:
    if requested == "nasa":
        return "nasa"
    if requested == "openm":
        return "openm"
    nasa_cols = {"ext_nasa_ws50m", "ext_nasa_wd50m", "ext_nasa_t2m", "ext_nasa_ps"}
    return "nasa" if nasa_cols.issubset(frame.columns) else "openm"


def _to_windfm_frame(frame: pd.DataFrame, source: str, power_mw: pd.Series | np.ndarray | None) -> pd.DataFrame:
    out = pd.DataFrame({"time": _local_to_utc(frame[predict.DATETIME_COL])})
    if source == "nasa":
        out["wind_speed"] = _safe_numeric(frame, "ext_nasa_ws50m")
        out["wind_direction"] = _direction_degrees(_safe_numeric(frame, "ext_nasa_wd50m"))
        out["density"] = _safe_numeric(frame, "ext_nasa_air_density_2m", predict.RHO_STANDARD)
        out["temperature"] = _safe_numeric(frame, "ext_nasa_t2m")
        # NASA PS is kPa. WindFM examples use a generic pressure feature; hPa
        # keeps it on the same scale as Open-Meteo/ERA5 pressure_msl.
        out["pressure"] = _safe_numeric(frame, "ext_nasa_ps") * 10.0
    else:
        out["wind_speed"] = _safe_numeric(frame, "wind_speed_eq")
        out["wind_direction"] = _direction_degrees(_safe_numeric(frame, "wind_direction_80m"))
        out["density"] = _safe_numeric(frame, "air_density", predict.RHO_STANDARD)
        out["temperature"] = _safe_numeric(frame, "temperature_80m")
        out["pressure"] = _safe_numeric(frame, "pressure_msl")

    if power_mw is None:
        out["power"] = np.nan
    else:
        out["power"] = np.clip(np.asarray(power_mw, dtype=float) / predict.P_RATED_FARM, 0.0, 1.2)

    feature_cols = [c for c in INPUT_FEATURES if c != "power"]
    out[feature_cols] = out[feature_cols].interpolate(limit_direction="both").ffill().bfill()
    return out[["time"] + INPUT_FEATURES]


def _sample_stat(samples: pd.DataFrame, stat: str) -> np.ndarray:
    if stat == "mean":
        return samples.mean(axis=1).to_numpy(dtype=float)
    if stat == "p10":
        return samples.quantile(0.10, axis=1).to_numpy(dtype=float)
    if stat == "p90":
        return samples.quantile(0.90, axis=1).to_numpy(dtype=float)
    return samples.quantile(0.50, axis=1).to_numpy(dtype=float)


def _raw_prediction_to_mw(raw: np.ndarray, output_scale: str) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    if output_scale == "mw":
        pred_mw = raw
    elif output_scale == "auto":
        pred_mw = raw * predict.P_RATED_FARM if np.nanquantile(np.abs(raw), 0.95) <= 1.5 else raw
    else:
        pred_mw = raw * predict.P_RATED_FARM
    return np.clip(pred_mw, 0.0, predict.P_RATED_FARM)


def _forecast_block(
    predictor,
    history: pd.DataFrame,
    future: pd.DataFrame,
    args: argparse.Namespace,
) -> np.ndarray:
    if len(history) < args.lookback:
        raise RuntimeError(f"WindFM needs at least lookback={args.lookback} history rows; got {len(history)}")
    context = history.tail(args.lookback).copy().reset_index(drop=True)
    predictions: list[np.ndarray] = []
    for start in range(0, len(future), args.pred_len):
        chunk = future.iloc[start:start + args.pred_len].copy().reset_index(drop=True)
        pred_len = len(chunk)
        x_df = context.tail(args.lookback)[INPUT_FEATURES].reset_index(drop=True)
        x_timestamp = context.tail(args.lookback)["time"].reset_index(drop=True)
        y_timestamp = chunk["time"].reset_index(drop=True)
        samples = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=args.temperature,
            top_p=args.top_p,
            sample_count=args.sample_count,
            verbose=args.verbose_windfm,
        )
        pred_mw = _raw_prediction_to_mw(_sample_stat(samples, args.stat), args.output_scale)
        predictions.append(pred_mw)

        generated = chunk.copy()
        generated["power"] = np.clip(pred_mw / predict.P_RATED_FARM, 0.0, 1.2)
        context = pd.concat([context, generated], ignore_index=True).tail(args.lookback).reset_index(drop=True)
        print(
            f"    WindFM chunk {start:5d}-{start + pred_len:5d}: "
            f"mean={pred_mw.mean():.3f} max={pred_mw.max():.3f}",
            flush=True,
        )
    return np.concatenate(predictions) if predictions else np.array([], dtype=float)


def _prepare_frames(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    predict.set_seed(predict.SEED)
    predict.set_model_set("hgb_family")
    predict.configure_v8_experiments(
        source_interactions=False,
        lag_residual_features=False,
        regime_models=False,
        windfm_diagnostic=False,
    )
    cfg = predict.physics_config_from_preset(args.physics_preset)
    train, valid, _, _ = predict.load_and_prepare(
        cfg,
        use_external_weather=True,
        external_cache_dir=_absolute(args.external_weather_cache_dir),
        refresh_external_weather=False,
        include_nasa_power=False,
        external_source_set=args.external_source_set,
        external_source_timeout_sec=args.external_source_timeout_sec,
        external_cache_policy=args.external_cache_policy,
        require_extra_sources=args.require_extra_sources,
        include_openmeteo_forecast=not args.skip_openmeteo_forecast,
        openmeteo_forecast_set=args.openmeteo_forecast_set,
        openmeteo_fill_policy=args.openmeteo_fill_policy,
        openmeteo_min_coverage=args.openmeteo_min_coverage,
        include_openmeteo_pressure=args.include_openmeteo_pressure,
        include_meteostat=not args.skip_meteostat,
        meteostat_cache_path=_absolute(args.meteostat_cache_path),
        allow_partial_meteostat=args.allow_partial_meteostat,
        require_meteostat=False,
        include_meteostat_derived=False,
        include_so_ups_res=args.include_so_ups_res,
        so_ups_res_cache_path=_absolute(args.so_ups_res_cache_path),
        so_ups_res_policy=args.so_ups_res_policy,
        so_ups_res_exclude_months=args.so_ups_res_exclude_months,
        include_physics_variants=True,
    )
    source = _select_source(train, args.input_source)
    print(f"WindFM input source: {source}", flush=True)
    return train, valid, source


def _export(args: argparse.Namespace) -> None:
    predictor_obj = _load_predictor(args)
    train, valid, source = _prepare_frames(args)
    train_windfm = _to_windfm_frame(train, source, train[predict.TARGET])
    valid_windfm = _to_windfm_frame(valid, source, None)

    train_pred = pd.Series(train["P_physics_farm"].values, index=train.index, dtype=float)
    fold_windows = [
        ("Q1 2023", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-03-31 23:00:00")),
        ("Q1 2024", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-31 23:00:00")),
        ("Q1 2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-31 23:00:00")),
    ]
    for label, start, end in fold_windows:
        target_mask = (train[predict.DATETIME_COL] >= start) & (train[predict.DATETIME_COL] <= end)
        if not target_mask.any():
            continue
        first_idx = int(np.flatnonzero(target_mask.to_numpy())[0])
        history = train_windfm.iloc[:first_idx].dropna(subset=["power"]).copy()
        future = train_windfm.loc[target_mask].drop(columns=["power"]).copy()
        print(f"WindFM OOF {label}: history={len(history)} pred_rows={len(future)}", flush=True)
        train_pred.loc[target_mask] = _forecast_block(predictor_obj, history, future, args)

    print(f"WindFM valid: history={len(train_windfm)} pred_rows={len(valid_windfm)}", flush=True)
    valid_pred = _forecast_block(predictor_obj, train_windfm.dropna(subset=["power"]), valid_windfm.drop(columns=["power"]), args)

    train_out = pd.DataFrame(
        {
            predict.DATETIME_COL: train[predict.DATETIME_COL],
            "prediction": np.clip(train_pred.values, 0.0, predict.P_RATED_FARM),
        }
    )
    valid_out = pd.DataFrame(
        {
            predict.DATETIME_COL: valid[predict.DATETIME_COL],
            "prediction": np.clip(valid_pred, 0.0, predict.P_RATED_FARM),
        }
    )

    train_path = _absolute(args.train_output_path)
    valid_path = _absolute(args.valid_output_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(train_path, index=False)
    valid_out.to_csv(valid_path, index=False)
    print(
        f"Wrote {train_path} rows={len(train_out)} mean={train_out['prediction'].mean():.3f} "
        f"max={train_out['prediction'].max():.3f}",
        flush=True,
    )
    print(
        f"Wrote {valid_path} rows={len(valid_out)} mean={valid_out['prediction'].mean():.3f} "
        f"max={valid_out['prediction'].max():.3f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.lookback > args.max_context:
        raise ValueError("--lookback cannot exceed --max-context")
    _export(args)


if __name__ == "__main__":
    main()
