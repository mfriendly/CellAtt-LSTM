#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
SEEDS="${1:-1}"
CFG=configs/covid_US.yaml
for S in $SEEDS; do
  python train.py --config "$CFG" --model triatt_lstm --seed "$S" --run_all_regions --log_convergence --log_cost
done
for TM in 20 30 45 60 90; do
  for S in $SEEDS; do
    python train.py --config "$CFG" --model triatt_lstm --tmax "$TM" --seed "$S" --run_all_regions
  done
done
python dump_splits.py
