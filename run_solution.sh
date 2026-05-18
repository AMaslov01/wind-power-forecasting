#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

MODE="infer"
CHECK_ONLY=0
INSTALL_DEPS=1
USE_GPU=0
PYTHON_BIN="${PYTHON:-python3}"
ARTIFACT_DIR="${MODEL_ARTIFACT_DIR:-model_weights}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
POST_Q1_FILE="dataset/3888f9f2-9bda-4b2c-94af-5562668bce86_test_dataset.csv"
SO_UPS_FILE="dataset/external_energy/so_ups_res_monthly.csv"

usage() {
  cat <<'EOF'
Usage:
  bash run_solution.sh [--check-only] [--retrain] [--gpu] [--artifact-dir DIR] [--output-dir DIR] [--no-install]

Default:
  Restores validated Q1 and May18 predictions from the bundled model_weights/ artifact directory.

Options:
  --check-only       Check Python, dependencies, required data files, and artifact bundle only.
  --retrain          Rebuild the stable V14 no-CDS ensemble, write artifacts, then validate outputs.
  --gpu              Use CatBoost GPU during --retrain. Default retrain mode is CPU for portability.
  --artifact-dir DIR Directory containing/writing ensemble_manifest.json and prediction snapshots.
  --output-dir DIR   Directory for predictions_q1.csv, predictions_may18.csv, and RUN_REPORT.json.
  --no-install       Do not install missing Python dependencies automatically.
  --python PATH      Python executable used to create/use .venv when needed.
EOF
}

log() {
  printf '\n[run_solution] %s\n' "$*"
}

die() {
  printf '\n[run_solution] ERROR: %s\n' "$*" >&2
  exit 1
}

on_error() {
  printf '\n[run_solution] ERROR near line %s. See the message above for the failing command.\n' "$1" >&2
}
trap 'on_error "$LINENO"' ERR

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --retrain)
      MODE="retrain"
      shift
      ;;
    --gpu)
      USE_GPU=1
      shift
      ;;
    --artifact-dir)
      [[ $# -ge 2 ]] || die "--artifact-dir requires a value"
      ARTIFACT_DIR="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || die "--output-dir requires a value"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --no-install)
      INSTALL_DEPS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

ensure_python() {
  if [[ "$MODE" != "retrain" ]]; then
    PY="$PYTHON_BIN"
    return
  fi
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PY="$PYTHON_BIN"
    return
  fi
  if [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
    return
  fi
  if "$PYTHON_BIN" -m venv .venv >/dev/null 2>&1; then
    PY=".venv/bin/python"
    return
  fi
  PY="$PYTHON_BIN"
}

ensure_deps() {
  if [[ "$MODE" != "retrain" ]]; then
    return
  fi

  if "$PY" - <<'PY' >/dev/null 2>&1
import catboost, meteostat, numpy, optuna, pandas, scipy, sklearn
import pypdf
PY
  then
    return
  fi
  [[ "$INSTALL_DEPS" -eq 1 ]] || die "Training dependencies are missing and --no-install was used."
  log "Installing Python training dependencies"
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r physics/requirements.txt
  "$PY" - <<'PY' >/dev/null
import catboost, meteostat, numpy, optuna, pandas, scipy, sklearn
import pypdf
PY
}

check_data_files() {
  local missing=()
  for path in \
    "dataset/train_dataset.csv" \
    "dataset/valid_features.csv" \
    "$POST_Q1_FILE"
  do
    [[ -f "$path" ]] || missing+=("$path")
  done
  if [[ "${#missing[@]}" -gt 0 ]]; then
    printf '\nMissing required data files:\n' >&2
    printf '  - %s\n' "${missing[@]}" >&2
    printf '\nPut the official hackathon CSV files into dataset/ and run again.\n' >&2
    exit 1
  fi
}

resolve_artifact_dir() {
  if [[ -f "$ARTIFACT_DIR/ensemble_manifest.json" ]]; then
    return
  fi
  if [[ -f "deliverables/final_solution_v14_nocds/model_weights/ensemble_manifest.json" ]]; then
    ARTIFACT_DIR="deliverables/final_solution_v14_nocds/model_weights"
    return
  fi
  if [[ "$MODE" == "infer" || "$CHECK_ONLY" -eq 1 ]]; then
    die "No model artifact bundle found in $ARTIFACT_DIR. Run: bash run_solution.sh --retrain"
  fi
}

prepare_so_ups_cache() {
  if [[ -f "$SO_UPS_FILE" ]]; then
    return
  fi
  log "Building SO UPS RES monthly cache"
  mkdir -p "$(dirname "$SO_UPS_FILE")"
  "$PY" physics/so_ups_res_parser.py \
    --so-ups-years 2022,2023,2024,2025 \
    --output-path "$SO_UPS_FILE" \
    --dump-json physics/so_ups_res_parser_report.json
}

validate_check_only() {
  "$PY" physics/validate_solution.py \
    --dataset-root dataset \
    --artifact-dir "$ARTIFACT_DIR" \
    --report-path "$OUTPUT_DIR/RUN_REPORT.json" \
    --mode check \
    --check-data \
    --check-artifacts \
    --skip-output-validation
}

run_infer() {
  mkdir -p "$OUTPUT_DIR"
  "$PY" physics/infer_from_artifacts.py \
    --artifact-dir "$ARTIFACT_DIR" \
    --output-dir "$OUTPUT_DIR"
  "$PY" physics/validate_solution.py \
    --q1-path "$OUTPUT_DIR/predictions_q1.csv" \
    --may18-path "$OUTPUT_DIR/predictions_may18.csv" \
    --dataset-root dataset \
    --artifact-dir "$ARTIFACT_DIR" \
    --report-path "$OUTPUT_DIR/RUN_REPORT.json" \
    --mode infer \
    --check-data \
    --check-artifacts
}

run_retrain() {
  mkdir -p "$OUTPUT_DIR" "$ARTIFACT_DIR"
  prepare_so_ups_cache
  local task_type="CPU"
  local gpu_args=()
  if [[ "$USE_GPU" -eq 1 ]]; then
    task_type="GPU"
    gpu_args=(--catboost-devices 0)
  fi
  "$PY" physics/run_full_pipeline.py --skip-tune \
    --model-set hgb_family \
    --physics-preset public_best \
    --fold-weights recency \
    --blend-method all \
    --catboost-task-type "$task_type" "${gpu_args[@]}" \
    --openmeteo-forecast-set gfs_only \
    --include-so-ups-res \
    --so-ups-res-policy legacy_best_gap \
    --external-source-set strict_open \
    --require-extra-sources \
    --enable-regime-models \
    --enable-weather-dynamics-features \
    --enable-2026-actual-adapter \
    --q1-actual-adapter-policy scalar \
    --post-q1-dataset-path "$POST_Q1_FILE" \
    --candidate-name hgb_family_gfs_so_ups_nasa_regime_weather_dyn_q1_scalar_v14_nocds \
    --output-path "$OUTPUT_DIR/predictions_q1.csv" \
    --output-post-blank-path "$OUTPUT_DIR/predictions_may18.csv" \
    --model-artifact-dir "$ARTIFACT_DIR"
  "$PY" physics/validate_solution.py \
    --q1-path "$OUTPUT_DIR/predictions_q1.csv" \
    --may18-path "$OUTPUT_DIR/predictions_may18.csv" \
    --dataset-root dataset \
    --artifact-dir "$ARTIFACT_DIR" \
    --report-path "$OUTPUT_DIR/RUN_REPORT.json" \
    --mode retrain \
    --check-data \
    --check-artifacts
}

ensure_python
log "Using Python: $PY"
ensure_deps
check_data_files
resolve_artifact_dir

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  mkdir -p "$OUTPUT_DIR"
  validate_check_only
  log "Check-only completed successfully"
  exit 0
fi

if [[ "$MODE" == "retrain" ]]; then
  log "Running full stable V14 no-CDS retrain"
  run_retrain
else
  log "Restoring predictions from artifact bundle"
  run_infer
fi

log "Done"
log "Q1 forecast:    $OUTPUT_DIR/predictions_q1.csv"
log "May18 forecast: $OUTPUT_DIR/predictions_may18.csv"
log "Run report:     $OUTPUT_DIR/RUN_REPORT.json"
