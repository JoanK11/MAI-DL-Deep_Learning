import argparse, torch, torch.nn as nn, torch.optim as optim, os
import pandas as pd
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from models import get_resnet18
from utils import epoch_loop, save_checkpoint
import datetime

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', '-d', default='./data')
    p.add_argument('--epochs', '-e', type=int, default=130)
    p.add_argument('--batch_size', '-bs', type=int, default=512)
    p.add_argument('--lr', '-lr', type=float, default=0.1)
    p.add_argument('--results', '-r', default='./results')
    p.add_argument('--plots', '-p', default='./plots')
    p.add_argument('--ood', '-o', type=str, default='tinyimagenet', 
                  choices=['tinyimagenet', 'cifar100', 'svhn'], 
                  help='Out-of-distribution dataset')
    p.add_argument('--weights', '-w', type=str, default=None, 
                  help='Path to pretrained model weights (.pth file)')
    args = p.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"cifar10_{args.ood}/resnet18_e{args.epochs}_lr{args.lr}_bs{args.batch_size}_{timestamp}"
    
    results_dir = os.path.join(args.results, experiment_name)
    plots_dir = os.path.join(args.plots, experiment_name)
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader, _ = get_dataloaders(args.data,
                                                  batch_size=args.batch_size,
                                                  ood_dataset_name=args.ood)

    model = get_resnet18(weights_path=args.weights).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 90], gamma=0.1)

    best_acc = 0.0
    epochs_history = []
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(args.epochs):
        train_loss, train_acc = epoch_loop(model, train_loader, criterion,
                                           optimizer, device)
        val_loss, val_acc = epoch_loop(model, val_loader, criterion,
                                       optimizer=None, device=device)
        scheduler.step()
        
        epochs_history.append(epoch + 1)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(f'Epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}')
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(results_dir, 'resnet18.pt')
            save_checkpoint(model, ckpt_path)

    df = pd.DataFrame({
        'epoch': epochs_history,
        'train_loss': train_loss_history,
        'train_accuracy': train_acc_history,
        'validation_loss': val_loss_history,
        'validation_accuracy': val_acc_history
    })
    csv_filename = os.path.join(results_dir, 'training_log.csv')
    df.to_csv(csv_filename, index=False)
    print(f"Training log saved to {csv_filename}")

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_history, train_loss_history, label='Training Loss')
    plt.plot(epochs_history, val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Classifier Training and Validation Loss (CIFAR-10 vs {args.ood.upper()})')
    plt.legend()
    plt.grid(True)
    loss_plot_filename = os.path.join(plots_dir, 'loss_plot.png')
    plt.savefig(loss_plot_filename)
    plt.close()
    print(f"Loss plot saved to {loss_plot_filename}")

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_history, train_acc_history, label='Training Accuracy')
    plt.plot(epochs_history, val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Classifier Training and Validation Accuracy (CIFAR-10 vs {args.ood.upper()})')
    plt.legend()
    plt.grid(True)
    acc_plot_filename = os.path.join(plots_dir, 'accuracy_plot.png')
    plt.savefig(acc_plot_filename)
    plt.close()
    print(f"Accuracy plot saved to {acc_plot_filename}")
    print(f"CLASSIFIER_RESULTS_DIR={results_dir}")

if __name__ == '__main__':
    main()
