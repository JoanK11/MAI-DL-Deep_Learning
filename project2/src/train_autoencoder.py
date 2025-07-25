import argparse, torch, torch.nn as nn, torch.optim as optim, os
import pandas as pd
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from models import ConvAutoencoder
from utils import epoch_loop_autoencoder, save_checkpoint
import datetime

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', '-d', default='./data')
    p.add_argument('--epochs', '-e', type=int, default=100)
    p.add_argument('--batch_size', '-bs', type=int, default=512)
    p.add_argument('--lr', '-lr', type=float, default=1e-3)
    p.add_argument('--bottleneck', '-b', type=int, default=64)
    p.add_argument('--results', '-r', default='./results')
    p.add_argument('--plots', '-p', default='./plots')
    p.add_argument('--ood', '-o', type=str, default='tinyimagenet', 
                  choices=['tinyimagenet', 'cifar100', 'svhn'], 
                  help='Out-of-distribution dataset used for context, though AE is trained on ID.')
    args = p.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"cifar10_{args.ood}/autoencoder_e{args.epochs}_lr{args.lr}_bs{args.batch_size}_b{args.bottleneck}_{timestamp}"
    
    results_dir = os.path.join(args.results, experiment_name)
    plots_dir = os.path.join(args.plots, experiment_name)
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader, _ = get_dataloaders(args.data,
                                                  batch_size=args.batch_size,
                                                  ood_dataset_name=args.ood)

    model = ConvAutoencoder(bottleneck_dim=args.bottleneck).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val = 1e9
    train_mse_history = []
    val_mse_history = []

    for epoch in range(args.epochs):
        train_loss, _ = epoch_loop_autoencoder(model, train_loader, criterion,
                                   optimizer, device)
        val_loss, _ = epoch_loop_autoencoder(model, val_loader, criterion,
                                 optimizer=None, device=device)
        
        train_mse_history.append(train_loss)
        val_mse_history.append(val_loss)
        
        print(f'Epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} recon={val_loss:.4f}')
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(results_dir,
                                     f'autoencoder.pt')
            save_checkpoint(model, ckpt_path)

    # Save training log to CSV
    epochs_range = range(1, args.epochs + 1)
    df = pd.DataFrame({
        'epoch': list(epochs_range),
        'train_mse': train_mse_history,
        'val_mse': val_mse_history
    })
    csv_filename = os.path.join(results_dir, 'training_log.csv')
    df.to_csv(csv_filename, index=False)
    print(f"Training log saved to {csv_filename}")

    # Plot training and validation MSE
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_mse_history, label='Training MSE')
    plt.plot(epochs_range, val_mse_history, label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(f'Autoencoder Training MSE (CIFAR-10 ID, {args.ood.upper()} OOD context, Bottleneck: {args.bottleneck})')
    plt.legend()
    plt.grid(True)
    plot_filename = os.path.join(plots_dir, 'mse_plot.png')
    plt.savefig(plot_filename)
    print(f"MSE plot saved to {plot_filename}")
    print(f"AUTOENCODER_RESULTS_DIR={results_dir}")

if __name__ == '__main__':
    main()