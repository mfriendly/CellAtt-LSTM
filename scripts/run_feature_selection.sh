#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python data/prepare_data.py --all
