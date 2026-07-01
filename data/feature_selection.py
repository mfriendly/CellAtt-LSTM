"""Feature selection: cross-correlation lag finding, mRMR, CFS robustness check.

Pipeline:
  1. find_optimal_lags()  — sliding window PCC per feature → optimal lag τ*
  2. mrmr_select()        — greedy mRMR (relevance - redundancy)
  3. cfs_merit()          — CFS merit formula for subset validation
  4. select_features()    — local/remote pool split → mRMR each → merge → CFS check
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def find_optimal_lags(target, exog_df, lag_max=50):
    """Find optimal lag for each exogenous feature via sliding window PCC.

    Args:
        target: pd.Series, target variable
        exog_df: pd.DataFrame, exogenous features (columns)
        lag_max: int, maximum lag to test

    Returns:
        dict: {col: {"lag": int, "pcc": float}} for each exog column
    """
    results = {}
    y = target.values
    n = len(y)
    for col in exog_df.columns:
        x = exog_df[col].values
        best_lag, best_pcc = 0, 0.0
        for lag in range(0, min(lag_max + 1, n // 3)):
            if lag == 0:
                x_lagged, y_trimmed = x, y
            else:
                x_lagged = x[:-lag]
                y_trimmed = y[lag:]
            min_len = min(len(x_lagged), len(y_trimmed))
            if min_len < 10:
                continue
            x_lagged, y_trimmed = x_lagged[:min_len], y_trimmed[:min_len]
            if np.std(x_lagged) < 1e-10 or np.std(y_trimmed) < 1e-10:
                continue
            r, p = pearsonr(x_lagged, y_trimmed)
            if abs(r) > abs(best_pcc):
                best_pcc = r
                best_lag = lag
        results[col] = {"lag": best_lag, "pcc": best_pcc}
    return results


def apply_lags(df, lag_info, pcc_threshold=0.3):
    """Create lagged feature DataFrame, filtering by minimum PCC threshold.

    Args:
        df: pd.DataFrame with date, target, and exog columns
        lag_info: dict from find_optimal_lags()
        pcc_threshold: minimum |PCC| to retain feature

    Returns:
        pd.DataFrame with lagged features aligned to target, list of retained feature names
    """
    target = df["target"].copy()
    lagged_cols = {}
    retained = []
    max_lag = max((v["lag"] for v in lag_info.values()), default=0)

    for col, info in lag_info.items():
        if abs(info["pcc"]) < pcc_threshold:
            continue
        lag = info["lag"]
        shifted = df[col].shift(lag)
        lagged_cols[f"{col}_lag{lag}"] = shifted
        retained.append(f"{col}_lag{lag}")

    if not lagged_cols:
        return pd.DataFrame({"target": target}), []

    lagged_df = pd.DataFrame(lagged_cols)
    lagged_df["target"] = target
    lagged_df = lagged_df.iloc[max_lag:].reset_index(drop=True)
    return lagged_df, retained


def mrmr_select(X, y, max_features=None):
    """Greedy mRMR feature selection.

    Relevance: |PCC(feature, target)|
    Redundancy: mean |PCC(feature, already_selected)|
    Score: relevance - redundancy

    Args:
        X: pd.DataFrame, candidate features
        y: pd.Series, target
        max_features: int or None (None = select all with positive score)

    Returns:
        list of selected column names in selection order
    """
    candidates = list(X.columns)
    if not candidates:
        return []

    relevance = {}
    for col in candidates:
        if np.std(X[col].values) < 1e-10:
            relevance[col] = 0.0
            continue
        r, _ = pearsonr(X[col].values, y.values)
        relevance[col] = abs(r)

    selected = []
    remaining = set(candidates)

    if max_features is None:
        max_features = len(candidates)

    for _ in range(max_features):
        if not remaining:
            break
        best_col, best_score = None, -np.inf
        for col in remaining:
            rel = relevance[col]
            if not selected:
                red = 0.0
            else:
                red_vals = []
                x_col = X[col].values
                for s in selected:
                    x_s = X[s].values
                    if np.std(x_col) < 1e-10 or np.std(x_s) < 1e-10:
                        red_vals.append(0.0)
                        continue
                    r, _ = pearsonr(x_col, x_s)
                    red_vals.append(abs(r))
                red = np.mean(red_vals)
            score = rel - red
            if score > best_score:
                best_score = score
                best_col = col
        if best_score <= 0 and len(selected) > 0:
            break
        selected.append(best_col)
        remaining.remove(best_col)

    return selected


def cfs_merit(X, y):
    """Compute CFS merit score for a feature subset.

    Merit_S = k * r_cf / sqrt(k + k*(k-1) * r_ff)

    Args:
        X: pd.DataFrame, selected features
        y: pd.Series, target

    Returns:
        float, CFS merit score
    """
    cols = list(X.columns)
    k = len(cols)
    if k == 0:
        return 0.0

    rcf_vals = []
    for col in cols:
        if np.std(X[col].values) < 1e-10:
            rcf_vals.append(0.0)
            continue
        r, _ = pearsonr(X[col].values, y.values)
        rcf_vals.append(abs(r))
    rcf = np.mean(rcf_vals)

    if k == 1:
        return rcf

    rff_vals = []
    for i in range(k):
        for j in range(i + 1, k):
            xi, xj = X[cols[i]].values, X[cols[j]].values
            if np.std(xi) < 1e-10 or np.std(xj) < 1e-10:
                rff_vals.append(0.0)
                continue
            r, _ = pearsonr(xi, xj)
            rff_vals.append(abs(r))
    rff = np.mean(rff_vals) if rff_vals else 0.0

    denom = np.sqrt(k + k * (k - 1) * rff)
    if denom < 1e-10:
        return 0.0
    return (k * rcf) / denom


def select_features(df, args, abbr, feature_meta=None):
    """Full feature selection pipeline with local/remote separation.

    Args:
        df: pd.DataFrame with date, target, and all candidate exog columns
        args: namespace with lag_max, pcc_threshold, max_features
        abbr: region abbreviation
        feature_meta: dict or None, mapping col -> "local"/"remote"
            If None, all features treated as one pool.

    Returns:
        selected_cols: list of original column names (before lagging)
        lag_info: dict {col: {"lag": int, "pcc": float}}
        cfs_score: float, CFS merit of final selection
        report: dict with selection details
    """
    exog_cols = [c for c in df.columns if c not in ("date", "target")]
    if not exog_cols:
        return [], {}, 0.0, {"n_selected": 0}

    target = df["target"]
    exog_df = df[exog_cols]

    lag_info = find_optimal_lags(target, exog_df, lag_max=args.lag_max)

    lagged_df, retained = apply_lags(df, lag_info, pcc_threshold=args.pcc_threshold)
    if not retained:
        return [], lag_info, 0.0, {"n_selected": 0}

    y = lagged_df["target"]
    X = lagged_df[retained]

    if feature_meta is not None:
        local_cols = [c for c in retained if feature_meta.get(c.rsplit("_lag", 1)[0], "local") == "local"]
        remote_cols = [c for c in retained if feature_meta.get(c.rsplit("_lag", 1)[0], "local") == "remote"]
    else:
        local_cols = retained
        remote_cols = []

    mf = args.max_features
    unlimited = (mf is None or mf <= 0)

    if unlimited:
        max_per_pool = None
    else:
        max_per_pool = mf // 2 if remote_cols else mf

    selected_local = mrmr_select(X[local_cols], y, max_features=max_per_pool) if local_cols else []
    selected_remote = mrmr_select(X[remote_cols], y, max_features=max_per_pool) if remote_cols else []
    selected = selected_local + selected_remote

    if not unlimited and len(selected) > mf:
        all_X = X[selected]
        selected = mrmr_select(all_X, y, max_features=mf)

    score = cfs_merit(X[selected], y) if selected else 0.0

    original_cols = list(set(c.rsplit("_lag", 1)[0] for c in selected))

    report = {
        "region": abbr,
        "n_candidates": len(exog_cols),
        "n_after_pcc_filter": len(retained),
        "n_local_selected": len(selected_local),
        "n_remote_selected": len(selected_remote),
        "n_selected": len(selected),
        "cfs_merit": score,
        "selected_features": selected,
    }
    print(f"    [{abbr}] Features: {len(exog_cols)} → {len(retained)} (PCC>{args.pcc_threshold}) → {len(selected)} (mRMR) | CFS={score:.4f}")

    return original_cols, lag_info, score, report
