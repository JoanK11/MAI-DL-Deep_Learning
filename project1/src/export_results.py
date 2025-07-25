#!/usr/bin/env python3
import os
import json
import numpy as np
from pathlib import Path

# Directory structure assumes project1/src/export_results.py, so results dir is two levels up
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Hyperparameter grids matching analyze_results.py
DROPOUT_RATES = [0.0, 0.5, 0.95]
EPOCHS_LIST   = [20, 80]
LR_LIST       = [0.001, 0.0001]
MODELS        = ["CustomNet", "GoogLeNet"]


def load_run_data(model, dp, ep, lr, run_num):
    run_dir = RESULTS_DIR / f"{model}_dp{dp}_e{ep}_lr{lr}_run{run_num}"
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return None
    with open(history_path, 'r') as f:
        return json.load(f)


def aggregate_model_results(model):
    records = []
    for dp in DROPOUT_RATES:
        for ep in EPOCHS_LIST:
            for lr in LR_LIST:
                # load per-run data
                runs = []
                for run_num in range(1, 4):
                    data = load_run_data(model, dp, ep, lr, run_num)
                    if data is not None:
                        runs.append(data)
                if not runs:
                    continue
                # collect epoch-wise metrics
                train_loss_list = []
                train_acc_list  = []
                val_loss_list   = []
                val_acc_list    = []
                for d in runs:
                    hist = d["train_epoch_history"]
                    train_loss_list.append([np.mean(e["train_loss"]) for e in hist])
                    train_acc_list.append([np.mean(e["train_acc"])  for e in hist])
                    val_loss_list.append([e["val_loss"] for e in hist])
                    val_acc_list.append([e["val_acc"] for e in hist])
                # pad to equal length
                max_len = max(len(x) for x in train_loss_list)
                def pad(arr): return np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
                train_loss_arr = np.vstack([pad(x) for x in train_loss_list])
                train_acc_arr  = np.vstack([pad(x) for x in train_acc_list])
                val_loss_arr   = np.vstack([pad(x) for x in val_loss_list])
                val_acc_arr    = np.vstack([pad(x) for x in val_acc_list])
                # average across runs ignoring NaNs
                avg_train_loss = np.nanmean(train_loss_arr, axis=0)
                avg_train_acc  = np.nanmean(train_acc_arr,  axis=0)
                avg_val_loss   = np.nanmean(val_loss_arr,   axis=0)
                avg_val_acc    = np.nanmean(val_acc_arr,    axis=0)
                # final epoch values
                final_train_loss = float(avg_train_loss[-1])
                final_train_acc  = float(avg_train_acc[-1])
                final_val_loss   = float(avg_val_loss[-1])
                final_val_acc    = float(avg_val_acc[-1])
                # test data averages
                test_losses = [d["test_data"]["loss"] for d in runs]
                test_accs   = [d["test_data"]["acc"]  for d in runs]
                avg_test_loss = float(np.mean(test_losses))
                avg_test_acc  = float(np.mean(test_accs))
                # record entry
                records.append({
                    "dropout": dp,
                    "epochs": ep,
                    "lr": lr,
                    "train_loss": final_train_loss,
                    "train_acc": final_train_acc,
                    "val_loss": final_val_loss,
                    "val_acc": final_val_acc,
                    "test_loss": avg_test_loss,
                    "test_acc": avg_test_acc
                })
    return records


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for model in MODELS:
        recs = aggregate_model_results(model)
        out_path = RESULTS_DIR / f"{model}.json"
        with open(out_path, 'w') as f:
            json.dump(recs, f, indent=4)
        print(f"Saved summary for {model} to {out_path}")


if __name__ == "__main__":
    main() 