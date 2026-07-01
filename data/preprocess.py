"""Data loading, feature selection, and preprocessing pipeline.

Unified format after prepare_data.py:
  data/raw/covid/{nation}/{ABBR}.csv   →  date, target [, exog...]
  data/raw/dengue/BR/{UF}.csv          →  date, target, tempmin, ..., umidmax
  data/raw/dengue/BR/metadata.json     →  regions list, date ranges

Pipeline:
  cfs=True:  load → find_optimal_lags → mRMR select → apply lags → preprocess → split
  cfs=False: load → univariate (target only) → preprocess → split
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from utils.scaler import fit_scaler, apply_scaler
from .feature_selection import select_features, find_optimal_lags, apply_lags


# ─────────────────────── Region data loading ───────────────────────
def load_region_data(args, abbr):
    """Load one region's clean CSV from data/raw/{dataset}/{nation}/{abbr}.csv."""
    path = Path(args.data_dir) / f"{abbr}.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_abbr_list(args):
    """Get list of region abbreviations from metadata.json."""
    meta_path = Path(args.data_dir) / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    return meta["regions"]


# ─────────────────────── Feature metadata ───────────────────────
def build_feature_meta(df, abbr):
    """Classify each exog column as local or remote.

    Local: exogenous variables from the same region (weather, search trends, etc.)
    Remote: case data from other regions (prefixed with region_ or cases_)
    """
    meta = {}
    for col in df.columns:
        if col in ("date", "target"):
            continue
        if col.startswith("region_") or col.startswith("cases_"):
            meta[col] = "remote"
        else:
            meta[col] = "local"
    return meta


# ─────────────────────── Preprocessing pipeline ───────────────────────
def preprocess_series(df, lag_info, selected_cols, args):
    """Apply lagging, rolling, diff, scaling to a single-region DataFrame.

    Args:
        df: DataFrame with [date, target, ...exog...]
        lag_info: dict from find_optimal_lags, {col: {"lag": int, "pcc": float}}
        selected_cols: list of original exog column names selected by mRMR
        args: namespace with rolling, rolling_window, diff, scaler,
              replace_zero_to_nan, val_start_date, test_start_date

    Returns:
        (df_train, df_val, df_test, scaler_stats)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # --- apply optimal lags to selected features ---
    max_lag = 0
    lagged_names = []
    for col in selected_cols:
        if col in lag_info and col in df.columns:
            lag = lag_info[col]["lag"]
            max_lag = max(max_lag, lag)
            col_name = f"{col}_lag{lag}"
            df[col_name] = df[col].shift(lag)
            lagged_names.append(col_name)

    # trim rows lost to lagging
    if max_lag > 0:
        df = df.iloc[max_lag:].reset_index(drop=True)

    # keep only date, target, and lagged features (drop everything else)
    keep_cols = ["date", "target"] + lagged_names
    df = df[keep_cols]

    # --- replace zeros ---
    if args.replace_zero_to_nan:
        df["target"] = df["target"].replace(0.0, np.nan).ffill().bfill()

    # --- split by test date (scaling fit only on train+val) ---
    df_trainval = df[df["date"] < args.test_start_date].copy().reset_index(drop=True)
    df_test = df[df["date"] >= args.test_start_date].copy().reset_index(drop=True)

    target_cols = ["target"]

    # --- rolling average ---
    if args.rolling:
        w = args.rolling_window
        for part in [df_trainval, df_test]:
            part["target"] = part["target"].rolling(window=w).mean().ffill().bfill()
        df_trainval = df_trainval.iloc[w - 1:].reset_index(drop=True)
        df_test = df_test.iloc[w - 1:].reset_index(drop=True)

    # --- difference transform ---
    if args.diff:
        for part in [df_trainval, df_test]:
            part["target"] = part["target"].diff().fillna(0.0)

    # --- scaling (fit on train+val only) ---
    scaler, stats = fit_scaler(df_trainval, target_cols, args.scaler)
    df_trainval = apply_scaler(df_trainval, scaler, target_cols)
    df_test = apply_scaler(df_test, scaler, target_cols)

    # scale exogenous columns too
    if lagged_names:
        exog_in_df = [c for c in lagged_names if c in df_trainval.columns]
        if exog_in_df:
            exog_scaler, _ = fit_scaler(df_trainval, exog_in_df, args.scaler)
            df_trainval = apply_scaler(df_trainval, exog_scaler, exog_in_df)
            df_test = apply_scaler(df_test, exog_scaler, exog_in_df)

    # --- split train / val ---
    df_train = df_trainval[df_trainval["date"] < args.val_start_date].reset_index(drop=True)
    df_val = df_trainval[df_trainval["date"] >= args.val_start_date].reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    n_exog = len([c for c in df_train.columns if c not in ("date", "target")])
    print(f"    Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}, Exog: {n_exog}")
    return df_train, df_val, df_test, stats


# ─────────────────────── Entry point ───────────────────────
def load_and_preprocess(args, abbr):
    """Full pipeline: load → feature selection (mRMR) → lag → preprocess → split.

    Returns:
        (df_train, df_val, df_test, scaler_stats, fs_report)
        fs_report is a dict with feature selection metadata (or empty dict).
    """
    df = load_region_data(args, abbr)

    if not args.cfs:
        df = df[["date", "target"]]
        df_train, df_val, df_test, stats = preprocess_series(df, {}, [], args)
        return df_train, df_val, df_test, stats, {}

    if getattr(args, "skip_lag_alignment", False):
        exog_cols = [c for c in df.columns if c not in ("date", "target")]
        if not exog_cols:
            df = df[["date", "target"]]
            df_train, df_val, df_test, stats = preprocess_series(df, {}, [], args)
            return df_train, df_val, df_test, stats, {}
        target = df["target"]
        from scipy.stats import pearsonr as _pearsonr
        lag0_info = {}
        for col in exog_cols:
            x = df[col].values
            y = target.values
            if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                continue
            r, _ = _pearsonr(x, y)
            lag0_info[col] = {"lag": 0, "pcc": r}
        pcc_thr = getattr(args, "pcc_threshold", 0.3)
        kept = [c for c, v in lag0_info.items() if abs(v["pcc"]) >= pcc_thr]
        n_raw = len(exog_cols)
        n_kept = len(kept)
        print(f"    [{abbr}] skip_lag_alignment: {n_raw} → {n_kept} (PCC>{pcc_thr} at lag=0)")
        df_train, df_val, df_test, stats = preprocess_series(df, lag0_info, kept, args)
        fs_report = {"region": abbr, "n_candidates": n_raw,
                     "n_after_pcc_filter": n_kept, "n_selected": n_kept,
                     "mode": "skip_lag_alignment"}
        return df_train, df_val, df_test, stats, fs_report

    if getattr(args, "no_mrmr", False):
        exog_cols = [c for c in df.columns if c not in ("date", "target")]
        target = df["target"]
        lag_info = find_optimal_lags(target, df[exog_cols], lag_max=args.lag_max)
        pcc_thr = getattr(args, "pcc_threshold", 0.3)
        lagged_df, retained = apply_lags(df, lag_info, pcc_threshold=pcc_thr)
        n_raw = len(exog_cols)
        n_kept = len(retained)
        original_cols = list(set(c.rsplit("_lag", 1)[0] for c in retained))
        print(f"    [{abbr}] no_mrmr: {n_raw} → {n_kept} (PCC>{pcc_thr}, lag-aligned)")
        df_train, df_val, df_test, stats = preprocess_series(df, lag_info, original_cols, args)
        fs_report = {"region": abbr, "n_candidates": n_raw,
                     "n_after_pcc_filter": n_kept, "n_selected": n_kept,
                     "mode": "pcc_only"}
        return df_train, df_val, df_test, stats, fs_report

    # multivariate mode: mRMR feature selection
    feature_meta = build_feature_meta(df, abbr)
    selected_cols, lag_info, cfs_score, report = select_features(df, args, abbr, feature_meta)
    df_train, df_val, df_test, stats = preprocess_series(df, lag_info, selected_cols, args)
    return df_train, df_val, df_test, stats, report
