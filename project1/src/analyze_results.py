import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({'font.size': 14})


def load_run_data(model, dropout, epochs, lr, run_num):
    run_dir = f"../results/{model}_dp{dropout}_e{epochs}_lr{lr}_run{run_num}"
    history_path = os.path.join(run_dir, "history.json")
    
    if not os.path.exists(history_path):
        print(f"Warning: {history_path} does not exist")
        return None
    
    with open(history_path, 'r') as f:
        data = json.load(f)
    
    return data


def analyze_model_results(model, dropout, epochs, lr, save_plots=False, show_plots=True, shadow=False, report_epochs=None):
    runs_data = []
    
    for run_num in range(1, 4):
        data = load_run_data(model, dropout, epochs, lr, run_num)
        if data:
            runs_data.append(data)
    
    if not runs_data:
        print(f"No data found for {model} with dropout={dropout}, epochs={epochs}, lr={lr}")
        return

    # Print header for averaged results
    print("\n" + "="*50)
    print(f"Results for {model} (dp={dropout}, e={epochs}, lr={lr}):")
    print("="*50)
    print("\nAverage across runs:")

    # Process per-epoch metrics from run data
    runs_metrics = []
    for data in runs_data:
        epoch_entries = data["train_epoch_history"]
        # Compute per-epoch training metrics (mean over batches) and validation metrics
        train_losses = [np.mean(ep["train_loss"]) for ep in epoch_entries]
        train_accs   = [np.mean(ep["train_acc"])   for ep in epoch_entries]
        val_losses   = [ep["val_loss"] for ep in epoch_entries]
        val_accs     = [ep["val_acc"]  for ep in epoch_entries]
        runs_metrics.append({
            "train_loss": train_losses,
            "train_acc":  train_accs,
            "val_loss":   val_losses,
            "val_acc":    val_accs,
            "test_loss":  data["test_data"]["loss"],
            "test_acc":   data["test_data"]["acc"]
        })

    # Determine number of epochs and initialize accumulators
    max_epochs = max(len(m["train_loss"]) for m in runs_metrics)
    avg_train_loss = np.zeros(max_epochs)
    avg_train_acc  = np.zeros(max_epochs)
    avg_val_loss   = np.zeros(max_epochs)
    avg_val_acc    = np.zeros(max_epochs)
    count_per_epoch = np.zeros(max_epochs)

    # Sum metrics over runs per epoch
    for m in runs_metrics:
        n_ep = len(m["train_loss"])
        avg_train_loss[:n_ep] += np.array(m["train_loss"])
        avg_train_acc[:n_ep]  += np.array(m["train_acc"])
        avg_val_loss[:n_ep]   += np.array(m["val_loss"])
        avg_val_acc[:n_ep]    += np.array(m["val_acc"])
        count_per_epoch[:n_ep] += 1

    # Compute average per epoch
    avg_train_loss /= count_per_epoch
    avg_train_acc  /= count_per_epoch
    avg_val_loss   /= count_per_epoch
    avg_val_acc    /= count_per_epoch

    # Compute final (last epoch) metrics
    avg_final_train_loss = avg_train_loss[-1]
    avg_final_train_acc  = avg_train_acc[-1]
    avg_final_val_loss   = avg_val_loss[-1]
    avg_final_val_acc    = avg_val_acc[-1]
    avg_test_loss = np.mean([m["test_loss"] for m in runs_metrics])
    avg_test_acc  = np.mean([m["test_acc"] for m in runs_metrics])

    # Print averaged final metrics
    print(f"  Final train loss:            {avg_final_train_loss:.4f}")
    print(f"  Final train accuracy:        {avg_final_train_acc:.4f}")
    print(f"  Final validation loss:       {avg_final_val_loss:.4f}")
    print(f"  Final validation accuracy:   {avg_final_val_acc:.4f}")
    print(f"  Test loss:                   {avg_test_loss:.4f}")
    print(f"  Test accuracy:               {avg_test_acc:.4f}")

    # Report specific epochs if requested
    if report_epochs:
        for ep in report_epochs:
            idx = ep - 1
            if idx < 0 or idx >= max_epochs:
                print(f"Epoch {ep}: out of range (1-{max_epochs})")
            else:
                print(f"Epoch {ep}: Training Loss={avg_train_loss[idx]:.4f}, Validation Loss={avg_val_loss[idx]:.4f}, Training Acc={avg_train_acc[idx]:.4f}, Validation Acc={avg_val_acc[idx]:.4f}")

    # If shadow is enabled, compute standard deviation across runs per epoch
    if shadow:
        train_loss_arr = np.vstack([m["train_loss"] for m in runs_metrics])
        val_loss_arr   = np.vstack([m["val_loss"]   for m in runs_metrics])
        train_acc_arr  = np.vstack([m["train_acc"]  for m in runs_metrics])
        val_acc_arr    = np.vstack([m["val_acc"]    for m in runs_metrics])
        std_train_loss = np.std(train_loss_arr, axis=0)
        std_val_loss   = np.std(val_loss_arr,   axis=0)
        std_train_acc  = np.std(train_acc_arr,  axis=0)
        std_val_acc    = np.std(val_acc_arr,    axis=0)

    # Create plots with average training and validation curves
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot average training and validation curves
    epochs_range = range(max_epochs)

    # Loss curves
    axs[0].plot(epochs_range, avg_train_loss, label='Training Loss', linewidth=2, color='blue')
    axs[0].plot(epochs_range, avg_val_loss,   label='Validation Loss', linestyle='--', color='red')
    axs[0].set_title(f"{model} Training and Validation Loss (dp={dropout}, e={epochs}, lr={lr})")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    # Shade standard deviation for loss curves
    if shadow:
        axs[0].fill_between(epochs_range, avg_train_loss - std_train_loss, avg_train_loss + std_train_loss, alpha=0.2, color='blue')
        axs[0].fill_between(epochs_range, avg_val_loss   - std_val_loss,   avg_val_loss   + std_val_loss,   alpha=0.2, color='red')

    # Accuracy curves
    axs[1].plot(epochs_range, avg_train_acc, label='Training Accuracy', linewidth=2, color='blue')
    axs[1].plot(epochs_range, avg_val_acc,   label='Validation Accuracy', linestyle='--', color='red')
    axs[1].set_title(f"{model} Training and Validation Accuracy (dp={dropout}, e={epochs}, lr={lr})")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].grid(True)

    # Shade standard deviation for accuracy curves
    if shadow:
        axs[1].fill_between(epochs_range, avg_train_acc - std_train_acc, avg_train_acc + std_train_acc, alpha=0.2, color='blue')
        axs[1].fill_between(epochs_range, avg_val_acc   - std_val_acc,   avg_val_acc   + std_val_acc,   alpha=0.2, color='red')
    
    plt.tight_layout(h_pad=5.0)
    
    if save_plots:
        plots_dir = f"../plots/{model}"
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, f"{model}_dp{dropout}_e{epochs}_lr{lr}.png")
        plt.savefig(plot_path)
        print(f"\nPlot saved to {plot_path}")
    
    if show_plots:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze model training results")
    parser.add_argument("-m", "--model", type=str, required=True, choices=["CustomNet", "GoogLeNet"],
                        help="Model name (CustomNet, GoogLeNet)")
    parser.add_argument("-dp", "--dropout", type=float, required=False, default=None,
                        help="Dropout rate (0.0, 0.5, 0.95)")
    parser.add_argument("-e", "--epochs", type=int, required=False, default=None,
                        help="Number of epochs (20, 80)")
    parser.add_argument("-lr", "--lr", type=float, required=False, default=None,
                        help="Learning rate (0.001, 0.0001)")
    parser.add_argument("-s", "--save", action="store_true", default=True,
                        help="Save plots to plots/model directory")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Compute plots for all combinations of dropout, epochs, and learning rate")
    parser.add_argument("-n", "--no-show", action="store_true", help="Do not display plots")
    parser.add_argument("-sh", "--shadow", action="store_true", default=False, help="Show shaded standard deviation in plots")
    parser.add_argument("-rep", "--report-epochs", type=int, nargs='+', default=None, help="Epochs to report loss and accuracy for (1-based indexing)")
    
    args = parser.parse_args()
    
    if args.save:
        plot_dir = Path(f"../plots/{args.model}")
        plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Predefined hyperparameter lists for --all option
    dropout_rates = [0.0, 0.5, 0.95]
    epochs_list = [20, 80]
    lr_list = [0.001, 0.0001]

    # Validate required args if not using --all
    if not args.all and (args.dropout is None or args.epochs is None or args.lr is None):
        parser.error("the following arguments are required: -d/--dropout, -e/--epochs, -lr/--lr for single model analysis. Use --all to process all combinations.")

    show_plots = not args.no_show

    # Analyze results
    if args.all:
        for dp in dropout_rates:
            for ep in epochs_list:
                for lr in lr_list:
                    analyze_model_results(args.model, dp, ep, lr, args.save, show_plots, args.shadow, args.report_epochs)
    else:
        analyze_model_results(args.model, args.dropout, args.epochs, args.lr, args.save, show_plots, args.shadow, args.report_epochs)


if __name__ == "__main__":
    main() 