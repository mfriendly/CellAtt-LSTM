"""Standalone evaluation: load saved predictions and compute metrics."""
import argparse
import json
import os
import numpy as np
from utils.metrics import compute_metrics, window_evaluation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", type=str, required=True,
                   help="Path to results dir, e.g. results/dengue/BR/triatt_lstm/SP/71")
    p.add_argument("--horizon_steps", type=int, nargs="+", default=None)
    p.add_argument("--ws", type=int, default=2)
    args = p.parse_args()

    pred_path = os.path.join(args.result_dir, "pred_all.json")
    actual_path = os.path.join(args.result_dir, "actual_all.json")

    with open(pred_path) as f:
        pred_all = json.load(f)
    with open(actual_path) as f:
        actual_all = json.load(f)

    all_pred = np.concatenate([np.array(v) for v in pred_all.values()])
    all_actual = np.concatenate([np.array(v) for v in actual_all.values()])

    metrics = compute_metrics(all_pred, all_actual)
    print("Overall metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    if args.horizon_steps:
        per_date, avg = window_evaluation(
            pred_all, actual_all, args.horizon_steps, args.ws)
        print(f"\nHorizon NRMSE: {avg.get('NRMSE', [])}")
        print(f"Horizon NMAE:  {avg.get('NMAE', [])}")
        print(f"Horizon RMSE:  {avg.get('RMSE', [])}")

        out_path = os.path.join(args.result_dir, "horizon_metrics_reeval.json")
        with open(out_path, "w") as f:
            json.dump({"per_date": per_date, "avg": avg}, f, indent=2)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
