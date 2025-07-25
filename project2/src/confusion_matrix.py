import argparse
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from dataset import get_dataloaders
from models import get_resnet18
from utils import load_checkpoint
from tqdm import tqdm
import datetime

def plot_confusion_matrix(cm, classes, output_path, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(12, 10))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")

@torch.no_grad()
def evaluate_model(model, val_loader, device, class_names, output_dir, model_name):
    # Collect predictions and ground truth
    all_preds = []
    all_targets = []
    
    for inputs, targets in tqdm(val_loader, desc=f'Evaluating {model_name}'):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Create model-specific directory
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Plot and save the raw confusion matrix
    raw_cm_path = os.path.join(model_dir, 'confusion_matrix_raw.png')
    plot_confusion_matrix(cm, classes=class_names, output_path=raw_cm_path,
                          title=f'{model_name} Confusion Matrix (Raw Counts)')
    
    # Plot and save the normalized confusion matrix
    norm_cm_path = os.path.join(model_dir, 'confusion_matrix_normalized.png')
    plot_confusion_matrix(cm, classes=class_names, output_path=norm_cm_path,
                          normalize=True, title=f'{model_name} Confusion Matrix (Normalized)')
    
    # Save the confusion matrix as a CSV file
    cm_csv_path = os.path.join(model_dir, 'confusion_matrix.csv')
    np.savetxt(cm_csv_path, cm, delimiter=',', fmt='%d')
    print(f"Confusion matrix data saved to {cm_csv_path}")
    
    # Calculate and print overall accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"{model_name} Overall Accuracy: {accuracy:.4f}")
    
    # Calculate per-class metrics
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1_score = 2 * precision * recall / (precision + recall)
    
    # Print per-class metrics
    print(f"\n{model_name} Per-class metrics:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1-score={f1_score[i]:.4f}")
        
    # Save metrics to CSV
    metrics_path = os.path.join(model_dir, 'metrics.csv')
    with open(metrics_path, 'w') as f:
        f.write('Class,Precision,Recall,F1-Score\n')
        for i, class_name in enumerate(class_names):
            f.write(f'{class_name},{precision[i]:.4f},{recall[i]:.4f},{f1_score[i]:.4f}\n')
        f.write(f'Overall,{np.mean(precision):.4f},{np.mean(recall):.4f},{np.mean(f1_score):.4f}\n')
        f.write(f'Accuracy,{accuracy:.4f},,\n')
    
    return cm, accuracy

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', '-d', default='./data')
    p.add_argument('--batch_size', '-b', type=int, default=512)
    p.add_argument('--finetuned_weights', '-w', type=str, 
                  default='results/cifar10_tinyimagenet/resnet18_e130_lr0.1_bs512_20250521_143704/resnet18.pt', 
                  help='Path to fine-tuned model weights (.pt file)')
    p.add_argument('--pretrained_weights', '-pw', type=str, default='pretrained_models/resnet18-f37072fd.pth', 
                  help='Path to pretrained model weights (.pth file)')
    p.add_argument('--output_dir', '-o', default='./results/confusion_matrix')
    p.add_argument('--ood', type=str, default='tinyimagenet', 
                  choices=['tinyimagenet', 'cifar100', 'svhn'], 
                  help='Out-of-distribution dataset name (for consistency with evaluate.py)')
    args = p.parse_args()
    
    # Create timestamp for the experiment
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.output_dir, f"comparison_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load CIFAR-10 data (we only need the validation set)
    _, val_loader, _ = get_dataloaders(args.data, batch_size=args.batch_size, ood_dataset_name=args.ood)
    
    # Get class names
    class_names = val_loader.dataset.classes
    
    results = {}
    
    # Evaluate the pretrained model
    pretrained_model = get_resnet18(weights_path=args.pretrained_weights).to(device)
    pretrained_model.eval()
    pretrained_cm, pretrained_acc = evaluate_model(
        pretrained_model, val_loader, device, class_names, 
        experiment_dir, "Pretrained_ResNet18"
    )
    results["Pretrained_ResNet18"] = {"cm": pretrained_cm, "accuracy": pretrained_acc}
    
    # Evaluate fine-tuned model
    finetuned_model = get_resnet18(pretrained=False).to(device)
    try:
        load_checkpoint(finetuned_model, args.finetuned_weights, map_location=device)
        finetuned_model.eval()
        
        # Evaluate fine-tuned model
        finetuned_cm, finetuned_acc = evaluate_model(
            finetuned_model, val_loader, device, class_names, 
            experiment_dir, "Finetuned_ResNet18"
        )
        results["Finetuned_ResNet18"] = {"cm": finetuned_cm, "accuracy": finetuned_acc}
        
        acc_improvement = results["Finetuned_ResNet18"]["accuracy"] - results["Pretrained_ResNet18"]["accuracy"]
        
        # Plot accuracy comparison
        plt.figure(figsize=(10, 6))
        plt.bar(["Pretrained ResNet18", "Fine-tuned ResNet18"], 
                [results["Pretrained_ResNet18"]["accuracy"], results["Finetuned_ResNet18"]["accuracy"]])
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Comparison (Improvement: {acc_improvement:.4f})')
        plt.ylim(0, 1)
        
        for i, acc in enumerate([results["Pretrained_ResNet18"]["accuracy"], results["Finetuned_ResNet18"]["accuracy"]]):
            plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
        
        comparison_path = os.path.join(experiment_dir, 'accuracy_comparison.png')
        plt.savefig(comparison_path)
        plt.close()
        print(f"Accuracy comparison saved to {comparison_path}")
        
        # Save comparison results
        comparison_csv = os.path.join(experiment_dir, 'comparison_results.csv')
        with open(comparison_csv, 'w') as f:
            f.write("Model,Accuracy\n")
            f.write(f"Pretrained_ResNet18,{results['Pretrained_ResNet18']['accuracy']:.4f}\n")
            f.write(f"Finetuned_ResNet18,{results['Finetuned_ResNet18']['accuracy']:.4f}\n")
            f.write(f"Improvement,{acc_improvement:.4f}\n")
        print(f"Comparison results saved to {comparison_csv}")
        
    except FileNotFoundError:
        print(f"WARNING: Fine-tuned model weights not found at {args.finetuned_weights}")
        print("Only evaluated the pretrained model.")

if __name__ == '__main__':
    main() 