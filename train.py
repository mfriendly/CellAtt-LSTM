"""Unified training entry point.

Usage:
  python train.py --config configs/dengue_BR.yaml --model triatt_lstm --abbr SP
  python train.py --config configs/covid_US.yaml  --model patchtst     --abbr NY
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import yaml

from data.preprocess import load_and_preprocess, get_abbr_list
from data.dataset import create_dataloaders
from models import build_model
from utils.seed import set_seed
from utils.early_stopping import EarlyStopping
from utils.scaler import inverse_scale
from utils.metrics import compute_metrics, window_evaluation


# ─────────────────────── Config loading ───────────────────────
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_args(config, cli_args):
    """Merge YAML config with CLI overrides into a flat namespace."""
    for k, v in vars(cli_args).items():
        if v is not None:
            config[k] = v
    # cfs: ensure bool (CLI passes int 0/1)
    if "cfs" in config and isinstance(config["cfs"], int):
        config["cfs"] = bool(config["cfs"])
    # tmax overrides lag_max
    if config.get("tmax") is not None:
        config["lag_max"] = config["tmax"]
    # defaults for new flags
    config.setdefault("skip_lag_alignment", False)
    config.setdefault("no_mrmr", False)
    config.setdefault("log_convergence", False)
    config.setdefault("log_cost", False)
    config.setdefault("convergence_dir", None)
    config.setdefault("cost_csv", None)
    config.setdefault("save_dir", None)
    ns = argparse.Namespace(**config)
    return ns


def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--abbr", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--hidden_size", type=int, default=None)
    p.add_argument("--layer_string", type=str, default=None)
    p.add_argument("--run_all_regions", action="store_true", default=False)
    # feature selection
    p.add_argument("--lag_max", type=int, default=None)
    p.add_argument("--pcc_threshold", type=float, default=None)
    p.add_argument("--max_features", type=int, default=None)
    p.add_argument("--cfs", type=int, default=None,
                   help="Override cfs: 0=univariate, 1=mRMR (default: from YAML)")
    # training
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    # architecture
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--d_ff", type=int, default=None)
    p.add_argument("--n_heads", type=int, default=None)
    p.add_argument("--e_layers", type=int, default=None)
    p.add_argument("--patch_len", type=int, default=None)
    p.add_argument("--patch_stride", type=int, default=None)
    p.add_argument("--endo_layers", type=int, default=None)
    p.add_argument("--exo_layers", type=int, default=None)
    p.add_argument("--fusion_layers", type=int, default=None)
    p.add_argument("--sep_exo", action="store_true", default=False)
    p.add_argument("--slim_cell", action="store_true", default=False)
    p.add_argument("--native_enc", action="store_true", default=False)
    p.add_argument("--pre_enc_layers", type=int, default=None)
    p.add_argument("--last_skip", action="store_true", default=None)
    p.add_argument("--last_linear", action="store_true", default=None)
    # logging and cost-measurement options
    p.add_argument("--tmax", type=int, default=None,
                   help="Override lag_max for feature selection")
    p.add_argument("--skip_lag_alignment", action="store_true", default=False,
                   help="Bypass lag-aware feature selection; use raw features")
    p.add_argument("--no_mrmr", action="store_true", default=False,
                   help="Skip mRMR; keep all PCC-filtered features with lag alignment")
    p.add_argument("--log_convergence", action="store_true", default=False,
                   help="Log per-epoch train_loss and val_nrmse to CSV")
    p.add_argument("--log_cost", action="store_true", default=False,
                   help="Log n_params, train_time, inference_time")
    p.add_argument("--convergence_dir", type=str, default=None,
                   help="Directory for convergence CSVs (default: results/logs)")
    p.add_argument("--cost_csv", type=str, default=None,
                   help="Path for computational cost CSV")
    p.add_argument("--save_dir", type=str, default=None,
                   help="Override result save directory")
    # CARD
    p.add_argument("--card_patch_len", type=int, default=None)
    p.add_argument("--card_stride", type=int, default=None)
    p.add_argument("--card_d_model", type=int, default=None)
    p.add_argument("--card_d_ff", type=int, default=None)
    p.add_argument("--card_n_heads", type=int, default=None)
    p.add_argument("--card_e_layers", type=int, default=None)
    p.add_argument("--card_dropout", type=float, default=None)
    p.add_argument("--card_dp_rank", type=int, default=None)
    p.add_argument("--card_merge_size", type=int, default=None)
    p.add_argument("--card_momentum", type=float, default=None)
    p.add_argument("--card_alpha", type=float, default=None)
    p.add_argument("--card_use_statistic", action="store_true", default=False)
    # EpiColaGNN
    p.add_argument("--epicola_nhid", type=int, default=None)
    p.add_argument("--epicola_n_layer", type=int, default=None)
    p.add_argument("--epicola_rnn", type=str, default=None)
    return p.parse_args()


# ─────────────────────── CSV logging ───────────────────────
CSV_COLUMNS = [
    "timestamp", "dataset", "nation", "model", "abbr", "seed",
    "NRMSE", "NMAE", "RMSE", "MAE", "MAPE", "sMAPE", "R2", "CORR",
    "epochs_run", "best_val_loss", "duration_min", "args_string",
]

_ARGS_KEYS = [
    "lr", "batch_size", "patience", "epochs",
    "d_model", "d_ff", "n_heads", "e_layers", "dropout",
    "endo_layers", "exo_layers", "fusion_layers",
    "patch_len", "patch_stride",
    "hidden_size", "past_steps", "future_steps",
    "max_features", "lag_max", "pcc_threshold",
]


def build_args_string(args):
    """Build compact key=value string from args for CSV logging."""
    parts = []
    for k in _ARGS_KEYS:
        v = getattr(args, k, None)
        if v is not None:
            parts.append(f"{k}={v}")
    return "|".join(parts)


def _get_csv_path(dataset, nation):
    return os.path.join("results", f"results_{dataset}_{nation}.csv")


def _append_csv(row_dict):
    """Append a single row to results_{dataset}_{nation}.csv."""
    csv_path = _get_csv_path(row_dict["dataset"], row_dict["nation"])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)


# ─────────────────────── Training loop ───────────────────────
def train_one_region(args, device):
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"  {args.dataset}/{args.nation} | {args.abbr} | {args.model} | seed={args.seed}")
    print(f"{'='*60}")

    # --- data ---
    df_train, df_val, df_test, scaler_stats, fs_report = load_and_preprocess(args, args.abbr)
    exog_cols = [c for c in df_train.columns if c not in ("date", "target")]
    args.in_channels = 1 + len(exog_cols)

    train_loader, val_loader, test_loader = create_dataloaders(
        df_train, df_val, df_test, args, exog_cols=exog_cols or None)

    # --- model ---
    model = build_model(args)

    dtype = torch.float64 if args.dtype == "double" else torch.float32
    model = model.to(device, dtype=dtype)

    # --- computational cost: parameter count ---
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.log_cost:
        print(f"  Trainable parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # --- paths ---
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = os.path.join(
            "results", args.dataset, args.nation, args.model, args.abbr,
            str(args.seed))
    os.makedirs(save_dir, exist_ok=True)
    ckpt_dir = os.path.join(save_dir, "ckpt")
    early_stop = EarlyStopping(patience=args.patience)

    # --- convergence logging setup ---
    convergence_rows = []

    # --- save config ---
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        cfg = {k: str(v) if not isinstance(v, (int, float, bool, str, list))
               else v for k, v in vars(args).items()}
        json.dump(cfg, f, indent=2)

    # --- save feature selection report ---
    if fs_report:
        with open(os.path.join(save_dir, "feature_selection.json"), "w") as f:
            json.dump(fs_report, f, indent=2)

    # --- train ---
    best_val_loss = float("inf")
    best_epoch = 0
    global_step = 0
    final_epoch = 0
    train_start_time = time.time()

    for epoch in tqdm(range(args.epochs), desc="Training"):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device, dtype=dtype)
            y = y.to(device, dtype=dtype)
            if args.model == "triatt_lstm" and args.use_curriculum_learning:
                output = model(x, target=y, global_step=global_step)
            else:
                output = model(x)
            loss = torch.sqrt(criterion(output, y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            global_step += 1
        avg_train = total_loss / len(train_loader)

        # --- validation ---
        model.eval()
        val_loss = 0.0
        val_pred_all, val_actual_all = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, dtype=dtype)
                y = y.to(device, dtype=dtype)
                output = model(x)
                val_loss += torch.sqrt(criterion(output, y)).item()
                if args.log_convergence:
                    val_pred_all.append(inverse_scale(output.cpu().numpy().flatten(), scaler_stats))
                    val_actual_all.append(inverse_scale(y.cpu().numpy().flatten(), scaler_stats))
        avg_val = val_loss / len(val_loader)

        # --- convergence: compute val NRMSE ---
        val_nrmse = 0.0
        if args.log_convergence and val_pred_all:
            vp = np.concatenate(val_pred_all)
            va = np.concatenate(val_actual_all)
            r = va.max() - va.min()
            val_nrmse = np.sqrt(((vp - va) ** 2).mean()) / r if r > 1e-8 else 0.0
            convergence_rows.append({"epoch": epoch, "train_loss": avg_train, "val_nrmse": val_nrmse})

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            torch.save({"epoch": epoch, "model": model.state_dict(),
                         "optimizer": optimizer.state_dict()},
                        os.path.join(ckpt_dir, "best.pth") if os.path.isdir(ckpt_dir)
                        else (os.makedirs(ckpt_dir, exist_ok=True) or
                              os.path.join(ckpt_dir, "best.pth")))

        final_epoch = epoch
        early_stop(avg_val, model, ckpt_dir)
        if early_stop.early_stop:
            print(f"Early stop at epoch {epoch}")
            break

    train_elapsed = time.time() - train_start_time

    # --- save convergence CSV ---
    if args.log_convergence and convergence_rows:
        conv_dir = args.convergence_dir or os.path.join("results", "logs")
        os.makedirs(conv_dir, exist_ok=True)
        conv_path = os.path.join(conv_dir, f"convergence_{args.model}_{args.dataset}_{args.nation}_{args.abbr}_{args.seed}.csv")
        with open(conv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_nrmse"])
            w.writeheader()
            w.writerows(convergence_rows)
        print(f"  Convergence saved: {conv_path}")

    # --- test ---
    ckpt_path = os.path.join(ckpt_dir, "best.pth")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "model" in state:
            model.load_state_dict(state["model"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
    model.eval()

    pred_all, actual_all = {}, {}
    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            x = x.to(device, dtype=dtype)
            output = model(x)
            pred = inverse_scale(output.cpu().numpy().flatten(), scaler_stats)
            actual = inverse_scale(y.numpy().flatten(), scaler_stats)
            date_key = str(idx)
            pred_all[date_key] = pred.tolist()
            actual_all[date_key] = actual.tolist()

    # --- save results ---
    with open(os.path.join(save_dir, "pred_all.json"), "w") as f:
        json.dump(pred_all, f)
    with open(os.path.join(save_dir, "actual_all.json"), "w") as f:
        json.dump(actual_all, f)

    # --- metrics ---
    all_pred = np.concatenate([np.array(v) for v in pred_all.values()])
    all_actual = np.concatenate([np.array(v) for v in actual_all.values()])
    metrics = compute_metrics(all_pred, all_actual)
    print(f"  Test metrics: {metrics}")

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # --- horizon evaluation ---
    if hasattr(args, "horizon_steps") and args.horizon_steps:
        per_date, avg = window_evaluation(
            pred_all, actual_all, args.horizon_steps, args.ws)
        with open(os.path.join(save_dir, "horizon_metrics.json"), "w") as f:
            json.dump({"per_date": per_date, "avg": avg}, f, indent=2)
        print(f"  Horizon NRMSE: {avg.get('NRMSE', [])}")
        print(f"  Horizon NMAE:  {avg.get('NMAE', [])}")
        print(f"  Horizon RMSE:  {avg.get('RMSE', [])}")

    # --- inference time measurement ---
    inference_time = 0.0
    if args.log_cost:
        model.eval()
        n_repeats = 10
        with torch.no_grad():
            # warmup
            for x, y in test_loader:
                x = x.to(device, dtype=dtype)
                _ = model(x)
                break
            torch.cuda.synchronize() if device.type == "cuda" else None
            t_inf_start = time.time()
            for _ in range(n_repeats):
                for x, y in test_loader:
                    x = x.to(device, dtype=dtype)
                    _ = model(x)
            torch.cuda.synchronize() if device.type == "cuda" else None
            inference_time = (time.time() - t_inf_start) / n_repeats

    # --- log to results.csv ---
    duration_min = round((time.time() - t_start) / 60, 2)
    _append_csv({
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "dataset": args.dataset,
        "nation": args.nation,
        "model": args.model,
        "abbr": args.abbr,
        "seed": args.seed,
        **{k: round(v, 6) for k, v in metrics.items()},
        "epochs_run": final_epoch + 1,
        "best_val_loss": round(best_val_loss, 6),
        "duration_min": duration_min,
        "args_string": build_args_string(args),
    })

    # --- log computational cost ---
    if args.log_cost:
        cost_csv = args.cost_csv or os.path.join("results", "logs", "computational_cost.csv")
        os.makedirs(os.path.dirname(cost_csv), exist_ok=True)
        cost_cols = ["model", "dataset", "nation", "abbr", "seed",
                     "n_params", "train_time_sec", "inference_time_sec", "best_epoch"]
        write_hdr = not os.path.exists(cost_csv)
        with open(cost_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cost_cols)
            if write_hdr:
                w.writeheader()
            w.writerow({
                "model": args.model, "dataset": args.dataset,
                "nation": args.nation, "abbr": args.abbr, "seed": args.seed,
                "n_params": n_params,
                "train_time_sec": round(train_elapsed, 2),
                "inference_time_sec": round(inference_time, 4),
                "best_epoch": best_epoch,
            })

    return metrics


# ─────────────────────── Main ───────────────────────
def main():
    cli = parse_cli()
    config = load_config(cli.config)
    args = build_args(config, cli)
    set_seed(args.seed)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    print(f"Device: {device} | CUDA available: {torch.cuda.is_available()}")

    if args.run_all_regions:
        abbr_list = get_abbr_list(args)
        results = {}
        for abbr in abbr_list:
            args.abbr = abbr
            metrics = train_one_region(args, device)
            results[abbr] = metrics
        save_dir = os.path.join("results", args.dataset, args.nation, args.model)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"all_metrics_seed{args.seed}.json"), "w") as f:
            json.dump(results, f, indent=2)
    else:
        train_one_region(args, device)


if __name__ == "__main__":
    main()
