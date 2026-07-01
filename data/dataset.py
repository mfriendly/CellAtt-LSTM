"""Windowed time series dataset for training/evaluation."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    """Sliding-window dataset.

    Args:
        df: DataFrame with columns [date, target, ...exog columns...]
        input_window: number of past time steps
        output_window: number of future time steps to predict
        stride: step between consecutive windows
        exog_cols: list of exogenous column names to include as extra features
    """
    def __init__(self, df, input_window, output_window, stride=1,
                 exog_cols=None):
        df = df.reset_index(drop=True)
        target = df["target"].to_numpy().reshape(-1, 1)

        if exog_cols:
            exog = df[exog_cols].to_numpy()
            features = np.concatenate([target, exog], axis=1)
        else:
            features = target

        n_time = len(df)
        n_feat = features.shape[1]
        n_samples = (n_time - input_window - output_window) // stride + 1

        X = np.zeros((n_samples, input_window, n_feat))
        Y = np.zeros((n_samples, output_window, 1))
        self.dates_past = []
        self.dates_future = []

        dates = df["date"].tolist() if "date" in df.columns else list(range(n_time))

        for i in range(n_samples):
            s = stride * i
            X[i] = features[s:s + input_window]
            Y[i] = target[s + input_window:s + input_window + output_window]
            self.dates_past.append(dates[s:s + input_window])
            self.dates_future.append(
                dates[s + input_window:s + input_window + output_window])

        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def create_dataloaders(df_train, df_val, df_test, args, exog_cols=None):
    """Create train/val/test DataLoaders from preprocessed DataFrames."""
    train_ds = TimeSeriesDataset(
        df_train, args.past_steps, args.future_steps, stride=1,
        exog_cols=exog_cols)
    val_ds = TimeSeriesDataset(
        df_val, args.past_steps, args.future_steps, stride=1,
        exog_cols=exog_cols)
    test_ds = TimeSeriesDataset(
        df_test, args.past_steps, args.future_steps,
        stride=args.eval_stride, exog_cols=exog_cols)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=1,
                             shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader
