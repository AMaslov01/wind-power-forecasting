#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r physics/requirements.txt

# Recommended public-best HGB-family run:
#   bash physics/run_full_pipeline.sh --skip-tune --physics-preset public_best --model-set hgb_family --blend-method all --blend-trials 1500
# Tune HGB params with a quick screen on Q1 2025, then full Q1 re-score:
#   python physics/tune_hgb_family.py --trials-per-mode 30 --candidate-pool-per-mode 6
# Fast HGB-only probe after tuning:
#   bash physics/run_full_pipeline.sh --skip-tune --physics-preset public_best --model-set hgb_optuna_only --blend-method all --blend-trials 1500 --blend-weight-lower 0.001
# Tuned HGB-family run after tuning:
#   bash physics/run_full_pipeline.sh --skip-tune --physics-preset public_best --model-set hgb_optuna_family --blend-method all --blend-trials 1500 --blend-weight-lower 0.001
# Conservative tuned HGB-family run, keeping the stable HGB core dominant:
#   bash physics/run_full_pipeline.sh --skip-tune --physics-preset public_best --model-set hgb_optuna_family --blend-method all --blend-trials 1500 --blend-weight-lower 0.001 --blend-constraint-profile hgb_conservative
# LightGBM diversity probe with explicit feature-count guard:
#   bash physics/run_full_pipeline.sh --skip-tune --physics-preset public_best --model-set lgb_probe --blend-method all --blend-trials 1500 --blend-weight-lower 0.001 --expect-feature-count 430
# Rebuild the exact 5-model/424-feature public-best baseline:
#   bash physics/run_full_pipeline.sh --skip-tune --physics-preset public_best --model-set legacy424 --blend-method slsqp
# GPU run, e.g. Google Colab T4/A100:
#   bash physics/run_full_pipeline.sh --skip-tune --physics-preset public_best --model-set hgb_family --blend-method all --catboost-task-type GPU --catboost-devices 0
python physics/run_full_pipeline.py "$@"
