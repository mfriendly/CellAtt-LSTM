#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
SEEDS="${1:-1}"
MODELS="triatt_lstm lstm gru itransformer patchtst tft dlinear autoformer timesnet timemixer tide timexer card epicolagnn"
for CFG in configs/covid_US.yaml configs/covid_AU.yaml configs/dengue_BR.yaml; do
  for M in $MODELS; do
    for S in $SEEDS; do
      python train.py --config "$CFG" --model "$M" --seed "$S" --run_all_regions
    done
  done
done
