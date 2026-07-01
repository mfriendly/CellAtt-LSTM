# CellAtt-LSTM: Cell-Attentive LSTM with Lag-Aware Feature Selection for Multi-Week Epidemic Forecasting

## Overview

A two-stage framework for multi-week epidemic forecasting:

1. **Lag-aware feature selection** (`data/feature_selection.py`): per-feature optimal-lag discovery via sliding-window Pearson correlation, followed by mRMR redundancy control with separate local/remote candidate pools. Lags and the selected feature set are computed on the training partition only and frozen for validation and test.
2. **CellAtt-LSTM** (`models/triatt_lstm.py`, `layers/lstm_cell.py`, `layers/attention.py`): an encoder-decoder LSTM with input feature attention and cell-level hidden-state attention. The attention configuration is set by a 4-bit `layer_string`; the published model uses `0101` (feature + cell-hidden).

## Environment

```
pip install -r requirements.txt
```

Python 3.10+, PyTorch 2.0+, CUDA GPU recommended.

## Data

Raw surveillance data is not bundled (all sources are public). Download `CellAtt-LSTM-data.zip` and unzip into the repository root so the layout becomes `data/raw/covid/US/`, `data/raw/covid/AU/`, and `data/raw/dengue/BR/`:
```
unzip CellAtt-LSTM-data.zip -d data/raw/
```

- `covid/US` — 50 US states (Google COVID-19 Open Data, https://health.google.com/covid-19/open-data/)
- `covid/AU` — 8 Australian states/territories (same source)
- `dengue/BR` — 27 Brazilian state capitals (InfoDengue API, https://info.dengue.mat.br)

Brazilian dengue data can be regenerated from the API:
```
python data/download_dengue.py
```

## Reproducing the results

```
bash scripts/run_feature_selection.sh   # lag-aligned selected feature matrices
bash scripts/run_main.sh                # CellAtt-LSTM + 13 baselines, all datasets
bash scripts/run_ablation.sh            # attention-component and feature-selection ablations
bash scripts/run_analysis.sh            # convergence, cost, max-lag sensitivity
```

## Configuration

Per-dataset hyperparameters are in `configs/{covid_US,covid_AU,dengue_BR}.yaml`; any setting can be overridden on the command line:
```
python train.py --config configs/covid_US.yaml --model triatt_lstm --layer_string 0101 --run_all_regions
```

## Repository structure

```
train.py                 training and evaluation entry point
evaluate.py              metric computation (NRMSE, NMAE)
dump_splits.py           print train/val/test boundary dates
configs/                 per-dataset YAML configs
data/
  feature_selection.py   lag-aware mRMR feature selection
  prepare_data.py        preprocessing + feature selection driver
  preprocess.py          interpolation, smoothing, differencing, scaling
  dataset.py             windowed dataset / dataloaders
  download_dengue.py     InfoDengue API downloader
  raw/                   raw surveillance data (downloaded separately)
models/                  CellAtt-LSTM and 13 baselines
layers/                  modified LSTM cell, attention block
tslib_modules/           vendored time-series baseline implementations
utils/                   metrics, scaler, early stopping
scripts/                 reproduction scripts
```