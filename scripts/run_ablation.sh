#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
SEEDS="${1:-1}"
CFG=configs/covid_US.yaml
for LS in 0000 0001 0100 0101 0111 1111; do
  for S in $SEEDS; do
    python train.py --config "$CFG" --model triatt_lstm --layer_string "$LS" --seed "$S" --run_all_regions
  done
done
for FS in "--max_features 20" "--max_features 10" "--max_features 5" "--no_mrmr"; do
  for S in $SEEDS; do
    python train.py --config "$CFG" --model triatt_lstm $FS --seed "$S" --run_all_regions
  done
done
