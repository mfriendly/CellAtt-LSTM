"""Read-only dump of the ACTUAL train/val/test split boundaries used by experiments.

This script does NOT re-implement the split. It imports and calls the very same
data-loading / partitioning function the training pipeline uses
(`data.preprocess.load_and_preprocess`, which `train.py` calls), feeds it the same
project YAML configs, and reports the first/last date actually present in each
partition AFTER the project's own preprocessing (lag-trim / rolling / diff).

It also reports where the national outbreak peak falls relative to those boundaries.

No experiment is re-run, no result file is modified. No try/except: any missing
file or empty partition fails loudly via assertion.

Usage:
  python dump_splits.py
  python dump_splits.py --data_root data/raw --config configs --out splits_actual.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

# Single source of truth: the exact functions the training pipeline uses.
from data.preprocess import load_and_preprocess, get_abbr_list, load_region_data


# Map reporting key -> project YAML the experiments actually ran with.
# (covid_US.yaml is the current split; covid_US_oldsplit.yaml is NOT used here.)
DATASET_CONFIGS = {
    "US": "covid_US.yaml",
    "AU": "covid_AU.yaml",
    "BR": "dengue_BR.yaml",
}

BOUNDARY_FIELDS = [
    "train_start", "train_end",
    "val_start", "val_end",
    "test_start", "test_end",
]


def build_args(config_path, data_root):
    """Load the project's own YAML into a namespace, exactly as train.py would.

    We do NOT inject any date. val_start_date / test_start_date / rolling_window /
    cfs / lag_max etc. all come straight from the project config file.
    """
    assert os.path.isfile(config_path), f"config not found: {config_path}"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    assert cfg is not None, f"empty config: {config_path}"
    # Flags load_and_preprocess may read but that older YAMLs omit.
    cfg.setdefault("skip_lag_alignment", False)
    # cfs may be 0/1 in some configs; load_and_preprocess expects truthy bool.
    if isinstance(cfg.get("cfs"), int):
        cfg["cfs"] = bool(cfg["cfs"])
    args = argparse.Namespace(**cfg)
    # Honor --data_root while keeping the project's dataset/nation layout.
    args.data_dir = os.path.join(data_root, cfg["dataset"], cfg["nation"])
    assert os.path.isdir(args.data_dir), f"data_dir missing: {args.data_dir}"
    return args


def partition_boundaries(df_train, df_val, df_test):
    """First/last actual date in each partition (post-preprocessing)."""
    for name, part in (("train", df_train), ("val", df_val), ("test", df_test)):
        assert len(part) > 0, f"empty {name} partition"
        assert "date" in part.columns, f"{name} partition missing 'date'"
    fmt = lambda d: pd.Timestamp(d).strftime("%Y-%m-%d")
    return {
        "train_start": fmt(df_train["date"].min()),
        "train_end":   fmt(df_train["date"].max()),
        "val_start":   fmt(df_val["date"].min()),
        "val_end":     fmt(df_val["date"].max()),
        "test_start":  fmt(df_test["date"].min()),
        "test_end":    fmt(df_test["date"].max()),
    }


def national_aggregate(args, regions):
    """Sum of RAW target across all regions, aligned by date (no preprocessing)."""
    frames = []
    for abbr in regions:
        df = load_region_data(args, abbr)  # same loader the pipeline uses
        assert "target" in df.columns, f"[{abbr}] no 'target' column"
        frames.append(df[["date", "target"]])
    allrows = pd.concat(frames, ignore_index=True)
    agg = allrows.groupby("date", as_index=False)["target"].sum()
    agg = agg.sort_values("date").reset_index(drop=True)
    assert len(agg) > 0, "empty national aggregate"
    return agg


def assign_partition(date, val_start_date, test_start_date):
    """Partition of a date under the pipeline's own split rule (preprocess.py:97,129)."""
    d = pd.Timestamp(date)
    if d >= pd.Timestamp(test_start_date):
        return "test"
    if d >= pd.Timestamp(val_start_date):
        return "val"
    return "train"


def peak_diagnostic(agg, val_start_date, test_start_date):
    """Global peak location + per-partition max of the national aggregate."""
    imax = int(agg["target"].values.argmax())
    peak_date = pd.Timestamp(agg["date"].iloc[imax]).strftime("%Y-%m-%d")
    peak_val = float(agg["target"].iloc[imax])
    peak_part = assign_partition(agg["date"].iloc[imax], val_start_date, test_start_date)

    part = agg["date"].apply(lambda d: assign_partition(d, val_start_date, test_start_date))
    pmax = {}
    for name in ("train", "val", "test"):
        sub = agg.loc[part == name, "target"]
        assert len(sub) > 0, f"national aggregate has no rows in {name} partition"
        pmax[name] = float(sub.max())

    return ({"date": peak_date, "value": peak_val, "partition": peak_part}, pmax)


def process_dataset(key, config_dir, data_root):
    config_path = os.path.join(config_dir, DATASET_CONFIGS[key])
    args = build_args(config_path, data_root)
    regions = get_abbr_list(args)
    assert len(regions) > 0, f"[{key}] no regions in metadata"

    print(f"\n{'='*64}\n  {key}  ({args.dataset}/{args.nation})  config={config_path}\n"
          f"  split dates from config: val_start={args.val_start_date}  "
          f"test_start={args.test_start_date}\n{'='*64}")

    # 1-3: actual boundaries per region via the real pipeline path.
    per_region = {}
    for abbr in regions:
        df_train, df_val, df_test, _stats, _report = load_and_preprocess(args, abbr)
        per_region[abbr] = partition_boundaries(df_train, df_val, df_test)

    # Consistency check across regions, field by field.
    inconsistent_fields = {}
    for field in BOUNDARY_FIELDS:
        values = {abbr: per_region[abbr][field] for abbr in regions}
        distinct = sorted(set(values.values()))
        if len(distinct) > 1:
            inconsistent_fields[field] = values
    regions_consistent = (len(inconsistent_fields) == 0)

    if regions_consistent:
        boundaries = {f: per_region[regions[0]][f] for f in BOUNDARY_FIELDS}
    else:
        # Keep the consistent fields; mark the varying ones with their distinct set.
        boundaries = {}
        for f in BOUNDARY_FIELDS:
            distinct = sorted({per_region[a][f] for a in regions})
            boundaries[f] = distinct[0] if len(distinct) == 1 else distinct
        print(f"  [!] regions are NOT consistent on: {list(inconsistent_fields)}")
        for field, values in inconsistent_fields.items():
            distinct = sorted(set(values.values()))
            print(f"      {field}: {len(distinct)} distinct -> {distinct}")
            for abbr in regions:
                print(f"        {abbr}: {values[abbr]}")

    # Peak diagnostic on the raw national aggregate.
    agg = national_aggregate(args, regions)
    national_peak, partition_max = peak_diagnostic(
        agg, args.val_start_date, args.test_start_date)

    return {
        "boundaries": boundaries,
        "regions_consistent": regions_consistent,
        "inconsistent_fields": sorted(inconsistent_fields),
        "national_peak": national_peak,
        "partition_max": partition_max,
        "n_regions": len(regions),
    }, regions_consistent


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default="data/raw",
                   help="Root of raw per-region CSVs (default: data/raw)")
    p.add_argument("--config", default="configs",
                   help="Directory holding the project YAML configs (default: configs)")
    p.add_argument("--out", default="splits_actual.json",
                   help="Output JSON path (default: splits_actual.json)")
    args = p.parse_args()

    assert os.path.isdir(args.config), f"--config dir not found: {args.config}"
    assert os.path.isdir(args.data_root), f"--data_root not found: {args.data_root}"

    result = {}
    verdicts = {}
    for key in DATASET_CONFIGS:
        result[key], verdicts[key] = process_dataset(key, args.config, args.data_root)

    # Write artifact FIRST so it survives even if the consistency assertion fires.
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out}")
    print(json.dumps(result, indent=2))

    # One-line summary per dataset.
    print("\n--- summary ---")
    for key in DATASET_CONFIGS:
        b = result[key]["boundaries"]
        pk = result[key]["national_peak"]
        print(f"{key}: test {b['test_start']}..{b['test_end']}, "
              f"national peak {pk['value']:.0f} on {pk['date']} in {pk['partition']}")

    # Fail loudly if any dataset's regions disagree on boundaries (per spec).
    for key in DATASET_CONFIGS:
        assert verdicts[key], (
            f"[{key}] regions do NOT share identical boundary dates "
            f"(varying fields: {result[key]['inconsistent_fields']}) "
            f"-> split is per-region, not global. See printout above.")


if __name__ == "__main__":
    main()
