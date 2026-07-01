import numpy as np


def RMSE(pred, true):
    return np.sqrt(((pred - true) ** 2).mean())


def MAE(pred, true):
    return np.abs(pred - true).mean()


def MAPE(pred, true):
    mask = np.abs(true) > 1e-8
    return np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100


def sMAPE(pred, true):
    denom = (np.abs(pred) + np.abs(true)) / 2
    mask = denom > 1e-8
    return np.mean(np.abs(pred[mask] - true[mask]) / denom[mask])


def NRMSE(pred, true):
    range_val = true.max() - true.min()
    if range_val < 1e-8:
        return 0.0
    return RMSE(pred, true) / range_val


def NMAE(pred, true):
    s = np.sum(np.abs(true))
    if s < 1e-8:
        return 0.0
    return np.sum(np.abs(true - pred)) / s


def R2(pred, true):
    y_mean = np.mean(true)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - y_mean) ** 2)
    if ss_tot < 1e-8:
        return 0.0
    return 1 - (ss_res / ss_tot)


def CORR(pred, true):
    pred = np.atleast_1d(pred).flatten()
    true = np.atleast_1d(true).flatten()
    if pred.size < 2:
        return 0.0
    p_mean, t_mean = pred.mean(), true.mean()
    num = ((true - t_mean) * (pred - p_mean)).sum()
    den = np.sqrt(((true - t_mean) ** 2).sum() * ((pred - p_mean) ** 2).sum())
    if den < 1e-8:
        return 0.0
    return num / den


def compute_metrics(pred, true):
    """Compute all metrics. pred/true: np arrays of same shape."""
    pred = np.array(pred).flatten()
    true = np.array(true).flatten()
    return {
        "RMSE": float(RMSE(pred, true)),
        "MAE": float(MAE(pred, true)),
        "MAPE": float(MAPE(pred, true)),
        "sMAPE": float(sMAPE(pred, true)),
        "NRMSE": float(NRMSE(pred, true)),
        "NMAE": float(NMAE(pred, true)),
        "R2": float(R2(pred, true)),
        "CORR": float(CORR(pred, true)),
    }


HORIZON_METRICS = ["NRMSE", "NMAE", "RMSE", "MAE", "MAPE"]


def window_evaluation(pred_dict, actual_dict, horizon_steps, ws):
    """Evaluate predictions at specific horizon steps using sliding window.

    Args:
        pred_dict: {start_date_str: [[v1], [v2], ...]}
        actual_dict: same format
        horizon_steps: list of horizon indices to evaluate
        ws: window size for averaging

    Returns:
        per_date_metrics: {date: {metric: [val_per_horizon]}}
        avg_metrics: {metric: [avg_val_per_horizon]}
    """
    metric_funcs = {
        "NRMSE": NRMSE, "NMAE": NMAE, "RMSE": RMSE, "MAE": MAE, "MAPE": MAPE,
    }
    per_date = {}
    for date_str in actual_dict:
        pred = np.array(pred_dict[date_str]).flatten()
        actual = np.array(actual_dict[date_str]).flatten()
        date_metrics = {m: [] for m in HORIZON_METRICS}
        for h in horizon_steps:
            s = slice(max(0, h - ws), h)
            p_slice, a_slice = pred[s], actual[s]
            for m in HORIZON_METRICS:
                date_metrics[m].append(float(metric_funcs[m](p_slice, a_slice)))
        per_date[date_str] = date_metrics

    avg_metrics = {}
    for metric in HORIZON_METRICS:
        vals = [per_date[d][metric] for d in per_date]
        avg_metrics[metric] = list(np.mean(vals, axis=0))
    return per_date, avg_metrics
