import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def fit_scaler(df_train, cols, scaler_type="standard"):
    """Fit scaler on training data, return scaler and stats dict.

    Stats dict contains float scalars (not arrays) for target-only inverse.
    """
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        return None, {}
    scaler.fit(df_train[cols])
    if scaler_type == "standard":
        stats = {"type": "standard",
                 "mean": scaler.mean_.tolist(),
                 "std": scaler.scale_.tolist()}
    else:
        stats = {"type": "minmax",
                 "min": scaler.data_min_.tolist(),
                 "max": scaler.data_max_.tolist()}
    return scaler, stats


def apply_scaler(df, scaler, cols):
    """Apply fitted scaler to dataframe columns. Returns new df."""
    df = df.copy()
    if scaler is not None:
        df[cols] = scaler.transform(df[cols])
    return df


def inverse_scale(values, scaler_stats):
    """Inverse-transform scaled values back to original scale."""
    values = np.array(values)
    if not scaler_stats:
        return values
    if scaler_stats["type"] == "standard":
        return values * scaler_stats["std"] + scaler_stats["mean"]
    elif scaler_stats["type"] == "minmax":
        return values * (scaler_stats["max"] - scaler_stats["min"]) + scaler_stats["min"]
    return values
