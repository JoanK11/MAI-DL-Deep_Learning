import argparse, torch, torch.nn as nn, os, csv, datetime
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dataset import get_dataloaders
from models import get_resnet18, ConvAutoencoder
from utils import load_checkpoint
from mahalanobis import compute_class_stats, mahalanobis_scores
from tqdm import tqdm
import random

CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616])

def unnormalize(tensor, mean, std):
    tensor = tensor.clone() 
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def auroc(scores_in, scores_ood):
    y_true = torch.cat([torch.zeros_like(scores_in), torch.ones_like(scores_ood)]).numpy()
    y_score = torch.cat([scores_in, scores_ood]).numpy()
    return roc_auc_score(y_true, y_score)

def plot_roc(y_true, y_score, path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(path, dpi=120)
    plt.close()

def calculate_ood_metrics(y_true, y_score_ood_higher):
    """
    Calculates various OOD detection metrics.
    Args:
        y_true (np.array): True labels (0 for ID, 1 for OOD).
        y_score_ood_higher (np.array): Anomaly scores (higher means more likely OOD).
    Returns:
        dict: A dictionary containing calculated metrics and curve data.
    """
    # AUROC
    auroc_val = roc_auc_score(y_true, y_score_ood_higher)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score_ood_higher)

    # AUPR-Out (OOD is positive class, pos_label=1)
    precision_out, recall_out, _ = precision_recall_curve(y_true, y_score_ood_higher, pos_label=1)
    aupr_out = auc(recall_out, precision_out)


    # AUPR-In (ID is positive class)
    # y_true_in_positive: 1 for ID, 0 for OOD
    y_true_in_positive = 1 - y_true 
    # scores_in_higher: higher means more ID
    scores_in_higher = -y_score_ood_higher
    precision_in, recall_in, _ = precision_recall_curve(y_true_in_positive, scores_in_higher, pos_label=1)
    aupr_in = auc(recall_in, precision_in)
    
    # FPR @ 95% TPR
    # Find the smallest FPR such that TPR >= 0.95
    idx_tpr_ge_95 = np.searchsorted(tpr, 0.95, side='left')
    if idx_tpr_ge_95 < len(tpr):
        fpr_at_95_tpr = fpr[idx_tpr_ge_95]
    else: # 0.95 TPR is not reached
        fpr_at_95_tpr = 1.0 

    # Detection Error (minimum misclassification rate)
    # DE = min_thresh 0.5 * (P(ID classified as OOD) + P(OOD classified as ID))
    # P(ID classified as OOD) = FPR
    # P(OOD classified as ID) = 1 - TPR (for the OOD class)
    detection_error = np.min(0.5 * (fpr + (1 - tpr))) # fpr and tpr from roc_curve(y_true, y_score_ood_higher)
    
    return {
        "auroc": auroc_val,
        "aupr_out": aupr_out,
        "aupr_in": aupr_in,
        "fpr_at_95_tpr": fpr_at_95_tpr,
        "detection_error": detection_error,
        "roc_curve_data": (fpr, tpr), # For direct use if needed
        "pr_curve_out_data": (recall_out, precision_out), # recall_out is x-axis for plot_pr_curve
        "pr_curve_in_data": (recall_in, precision_in)   # recall_in is x-axis for plot_pr_curve
    }

def plot_pr_curve(recall, precision, plot_title, file_path):
    plt.figure()
    plt.plot(recall, precision, marker='.') # Plot recall (x) vs precision (y)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(plot_title)
    plt.grid(True)
    plt.savefig(file_path, dpi=120)
    plt.close()

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', '-d', default='./data')
    p.add_argument('--batch_size', '-b', type=int, default=512)
    p.add_argument('--ood', '-o', type=str, required=True, 
                  choices=['tinyimagenet', 'cifar100', 'svhn'], 
                  help='Out-of-distribution dataset to evaluate against.')
    # Arguments to specify the *directory* containing the checkpoint from a specific training run
    p.add_argument('--classifier_experiment_dir', '-c', type=str, required=True, 
                  help='Path to the specific experiment directory for the classifier (e.g., results/cifar10_tinyimagenet/20231027_153000_lr0.1_bs256)')
    p.add_argument('--autoencoder_experiment_dir', '-a', type=str, required=True, 
                  help='Path to the specific experiment directory for the autoencoder (e.g., results/cifar10_tinyimagenet/20231027_153000_lr0.001_bs256_b64)')
    p.add_argument('--ae_bottleneck_dim', '-z', type=int, required=True, 
                  help="Bottleneck dimension of the autoencoder that was trained and whose checkpoint will be loaded. Must match the autoencoder_experiment_dir.")
    p.add_argument('--results', '-r', default='./results')
    p.add_argument('--plots', '-p', default='./plots')
    args = p.parse_args()

    classifier_ckpt_name = 'resnet18.pt'
    autoencoder_ckpt_name = 'autoencoder.pt'

    args.classifier_ckpt = os.path.join(args.classifier_experiment_dir, classifier_ckpt_name)
    args.autoencoder_ckpt = os.path.join(args.autoencoder_experiment_dir, autoencoder_ckpt_name)

    if not os.path.exists(args.classifier_ckpt):
        raise FileNotFoundError(f"Classifier checkpoint not found at {args.classifier_ckpt}. Please check path and naming convention.")
    if not os.path.exists(args.autoencoder_ckpt):
        raise FileNotFoundError(f"Autoencoder checkpoint not found at {args.autoencoder_ckpt}. Please check path, naming convention, and bottleneck_dim.")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_experiment_name = f"cifar10_{args.ood}/eval_{timestamp}_ae_b{args.ae_bottleneck_dim}"
    
    results_dir = os.path.join(args.results, eval_experiment_name)
    plots_dir = os.path.join(args.plots, eval_experiment_name)
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader, ood_loader = get_dataloaders(args.data,
                                                            batch_size=args.batch_size,
                                                            ood_dataset_name=args.ood)

    clf = get_resnet18(pretrained=False).to(device)
    load_checkpoint(clf, args.classifier_ckpt, map_location=device)
    clf.eval()

    all_y_true = []
    all_y_pred = []
    all_entropies = []
    all_images_for_samples = []
    all_confidences_for_samples = []

    for x, y in tqdm(val_loader, desc=f'Classifier CIFAR-10 (ID) Eval vs {args.ood.upper()} (OOD)'):
        x_cpu = x.cpu()
        x, y_true_batch = x.to(device), y.to(device)
        logits = clf(x)
        probs = torch.softmax(logits, dim=1)
        confidences_batch, y_pred_batch = torch.max(probs, 1)
        
        entropies_batch = -torch.sum(probs * torch.log(probs.clamp_min(1e-9)), dim=1)

        all_y_true.extend(y_true_batch.cpu().numpy())
        all_y_pred.extend(y_pred_batch.cpu().numpy())
        all_entropies.extend(entropies_batch.cpu().numpy())
        all_images_for_samples.extend([img_tensor for img_tensor in x_cpu])
        all_confidences_for_samples.extend(confidences_batch.cpu().numpy())

    all_y_true_np = np.array(all_y_true)
    all_y_pred_np = np.array(all_y_pred)
    all_confidences_np = np.array(all_confidences_for_samples)
    
    # Overall Accuracy for CIFAR-10 validation set
    acc = np.mean(all_y_true_np == all_y_pred_np)
    print(f"CIFAR-10 Validation Accuracy: {acc:.4f}")

    # Per-class precision, recall, F1-score
    class_names = val_loader.dataset.classes
    report_dict = classification_report(all_y_true_np, all_y_pred_np, target_names=class_names, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report_dict).transpose()
    report_csv_path = os.path.join(results_dir, 'classification_report.csv')
    df_report.to_csv(report_csv_path)
    print(f"Classification report saved to {report_csv_path}")

    # Per-class confidence (Entropy Boxplots)
    df_entropy = pd.DataFrame({
        'true_label': [class_names[i] for i in all_y_true_np],
        'entropy': all_entropies
    })

    plt.figure(figsize=(15, 8))
    sns.boxplot(x='true_label', y='entropy', data=df_entropy, order=class_names)
    plt.xlabel('True Class Label (CIFAR-10)')
    plt.ylabel('Prediction Entropy')
    plt.title(f'Entropy of Softmax Predictions per True Class (CIFAR-10 ID vs {args.ood.upper()} OOD)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')
    plt.tight_layout()
    entropy_plot_path = os.path.join(plots_dir, 'entropy_per_class.png')
    plt.savefig(entropy_plot_path)
    plt.close()
    print(f"Entropy per class boxplot saved to {entropy_plot_path}")

    # Select samples for qualitative grid
    correct_indices = np.where(all_y_true_np == all_y_pred_np)[0]
    incorrect_indices = np.where(all_y_true_np != all_y_pred_np)[0]

    selected_correct_indices = random.sample(list(correct_indices), min(3, len(correct_indices)))
    selected_incorrect_indices = random.sample(list(incorrect_indices), min(3, len(incorrect_indices)))
    
    selected_samples_data = []
    
    for i in selected_correct_indices:
        selected_samples_data.append({
            "image": all_images_for_samples[i],
            "true_label_idx": all_y_true_np[i],
            "pred_label_idx": all_y_pred_np[i],
            "confidence": all_confidences_np[i]
        })

    for i in selected_incorrect_indices:
        selected_samples_data.append({
            "image": all_images_for_samples[i],
            "true_label_idx": all_y_true_np[i],
            "pred_label_idx": all_y_pred_np[i],
            "confidence": all_confidences_np[i]
        })

    if len(selected_samples_data) > 0:
        fig, axes = plt.subplots(max(1, (len(selected_samples_data) + 2) // 3), 3, figsize=(9, 3 * max(1, (len(selected_samples_data) + 2) // 3)))
        axes = axes.flatten()

        for i, sample_data in enumerate(selected_samples_data):
            ax = axes[i]
            img_tensor = sample_data["image"]
            unnorm_image = unnormalize(img_tensor, CIFAR10_MEAN, CIFAR10_STD)
            unnorm_image = torch.clamp(unnorm_image, 0, 1) 
            ax.imshow(unnorm_image.permute(1, 2, 0).numpy()) 
            
            true_label_name = class_names[sample_data["true_label_idx"]]
            pred_label_name = class_names[sample_data["pred_label_idx"]]
            confidence = sample_data["confidence"]
            
            ax.set_title(f"T:{true_label_name}\nP:{pred_label_name} ({confidence:.2f})", fontsize=8)
            ax.axis('off')
        
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        sample_grid_path = os.path.join(plots_dir, 'sample_predictions_grid.png')
        plt.savefig(sample_grid_path)
        plt.close()
        print(f"Sample predictions grid saved to {sample_grid_path}")
    else:
        print("Not enough samples to generate the qualitative grid (e.g., no incorrect predictions).")

    # ===== Autoencoder recon error =====
    ae = ConvAutoencoder(bottleneck_dim=args.ae_bottleneck_dim).to(device)
    load_checkpoint(ae, args.autoencoder_ckpt, map_location=device)
    ae.eval()
    mse = nn.MSELoss(reduction='none')

    def recon_scores(loader, desc_suffix):
        scores = []
        for x_ae, _ in tqdm(loader, leave=False, desc=f'Recon Scores {desc_suffix}'):
            x_ae = x_ae.to(device)
            recon, _ = ae(x_ae)
            err = mse(recon, x_ae).mean([1,2,3]).cpu()
            scores.append(err)
        return torch.cat(scores)

    scores_in_recon  = recon_scores(val_loader, "CIFAR-10 (ID)")
    scores_ood_recon = recon_scores(ood_loader, f"{args.ood.upper()} (OOD)")
    # Note: scores_in_recon and scores_ood_recon are abnormality scores (higher is more OOD)
    y_true_recon = torch.cat([torch.zeros_like(scores_in_recon), torch.ones_like(scores_ood_recon)]).numpy()
    y_score_recon = torch.cat([scores_in_recon, scores_ood_recon]).numpy()

    metrics_recon = calculate_ood_metrics(y_true_recon, y_score_recon)
    auroc_recon = metrics_recon["auroc"]

    plot_roc(y_true_recon, y_score_recon, os.path.join(plots_dir, 'roc_recon.png'))
    plot_pr_curve(metrics_recon["pr_curve_out_data"][0], metrics_recon["pr_curve_out_data"][1],
                  f'PR Curve Recon Error - {args.ood.upper()} (OOD)',
                  os.path.join(plots_dir, 'pr_out_recon.png'))
    plot_pr_curve(metrics_recon["pr_curve_in_data"][0], metrics_recon["pr_curve_in_data"][1],
                  f'PR Curve Recon Error - CIFAR-10 (ID)',
                  os.path.join(plots_dir, 'pr_in_recon.png'))

    plt.figure(figsize=(10, 6))
    plt.hist(scores_in_recon.numpy(), bins=50, alpha=0.7, label='CIFAR-10 (In-distribution)')
    plt.hist(scores_ood_recon.numpy(), bins=50, alpha=0.7, label=f'{args.ood.upper()} (OOD)')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title(f'Reconstruction Error (CIFAR-10 ID vs {args.ood.upper()} OOD)')
    plt.legend()
    plt.grid(True)
    recon_dist_path = os.path.join(plots_dir, 'recon_scores_dist.png')
    plt.savefig(recon_dist_path)
    plt.close()
    print(f"Reconstruction scores distribution plot saved to {recon_dist_path}")

    # ===== Softmax score =====
    def sm_scores(loader, desc_suffix):
        scores = []
        for x_sm, _ in tqdm(loader, leave=False, desc=f'Softmax Scores {desc_suffix}'):
            x_sm = x_sm.to(device)
            probs = torch.softmax(clf(x_sm), dim=1)
            conf, _ = torch.max(probs, 1)
            scores.append(-conf.cpu()) # minus to make it a score of abnormality
        return torch.cat(scores)

    scores_in_sm  = sm_scores(val_loader, "CIFAR-10 (ID)")
    scores_ood_sm = sm_scores(ood_loader, f"{args.ood.upper()} (OOD)")
    # Note: sm_scores returns -confidence, so higher is more OOD
    y_true_sm = torch.cat([torch.zeros_like(scores_in_sm), torch.ones_like(scores_ood_sm)]).numpy()
    y_score_sm = torch.cat([scores_in_sm, scores_ood_sm]).numpy()

    metrics_sm = calculate_ood_metrics(y_true_sm, y_score_sm)
    auroc_sm = metrics_sm["auroc"]

    plot_roc(y_true_sm, y_score_sm, os.path.join(plots_dir, 'roc_sm.png'))
    plot_pr_curve(metrics_sm["pr_curve_out_data"][0], metrics_sm["pr_curve_out_data"][1],
                  f'PR Curve Softmax - {args.ood.upper()} (OOD)',
                  os.path.join(plots_dir, 'pr_out_sm.png'))
    plot_pr_curve(metrics_sm["pr_curve_in_data"][0], metrics_sm["pr_curve_in_data"][1],
                  f'PR Curve Softmax - CIFAR-10 (ID)',
                  os.path.join(plots_dir, 'pr_in_sm.png'))

    plt.figure(figsize=(10, 6))
    plt.hist(scores_in_sm.numpy(), bins=50, alpha=0.7, label='CIFAR-10 (In-distribution)')
    plt.hist(scores_ood_sm.numpy(), bins=50, alpha=0.7, label=f'{args.ood.upper()} (OOD)')
    plt.xlabel('Max Softmax Probability Score (-confidence)')
    plt.ylabel('Frequency')
    plt.title(f'Max Softmax Prob Score (CIFAR-10 ID vs {args.ood.upper()} OOD)')
    plt.legend()
    plt.grid(True)
    sm_dist_path = os.path.join(plots_dir, 'sm_scores_dist.png')
    plt.savefig(sm_dist_path)
    plt.close()
    print(f"Softmax scores distribution plot saved to {sm_dist_path}")

    # ===== Mahalanobis distance =====
    print("Computing Mahalanobis stats...")
    
    feature_layer_name = 'layer4_avgpool'
    print(f"Using features from {feature_layer_name} for Mahalanobis distance.")
    
    num_classes_train = len(train_loader.dataset.classes)

    class_means, class_precisions = compute_class_stats(clf, train_loader, num_classes_train,
                                                        device=device, feature_layer_name=feature_layer_name)

    scores_in_maha = mahalanobis_scores(clf, val_loader, class_means, class_precisions, device, feature_layer_name=feature_layer_name, desc_suffix="CIFAR-10 (ID)")
    scores_ood_maha = mahalanobis_scores(clf, ood_loader, class_means, class_precisions, device, feature_layer_name=feature_layer_name, desc_suffix=f"{args.ood.upper()} (OOD)")
    # Mahalanobis scores are abnormality scores (higher is more OOD)
    y_true_maha = torch.cat([torch.zeros_like(scores_in_maha), torch.ones_like(scores_ood_maha)]).numpy()
    y_score_maha = torch.cat([scores_in_maha, scores_ood_maha]).numpy()

    metrics_maha = calculate_ood_metrics(y_true_maha, y_score_maha)
    auroc_maha = metrics_maha["auroc"]

    plot_roc(y_true_maha, y_score_maha, os.path.join(plots_dir, 'roc_mahalanobis.png'))
    plot_pr_curve(metrics_maha["pr_curve_out_data"][0], metrics_maha["pr_curve_out_data"][1],
                  f'PR Curve Mahalanobis - {args.ood.upper()} (OOD)',
                  os.path.join(plots_dir, 'pr_out_mahalanobis.png'))
    plot_pr_curve(metrics_maha["pr_curve_in_data"][0], metrics_maha["pr_curve_in_data"][1],
                  f'PR Curve Mahalanobis - CIFAR-10 (ID)',
                  os.path.join(plots_dir, 'pr_in_mahalanobis.png'))

    plt.figure(figsize=(10, 6))
    plt.hist(scores_in_maha.numpy(), bins=50, alpha=0.7, label='CIFAR-10 (In-distribution)')
    plt.hist(scores_ood_maha.numpy(), bins=50, alpha=0.7, label=f'{args.ood.upper()} (OOD)')
    plt.xlabel('Mahalanobis Distance Score')
    plt.ylabel('Frequency')
    plt.title(f'Mahalanobis Distance (CIFAR-10 ID vs {args.ood.upper()} OOD)')
    plt.legend()
    plt.grid(True)
    maha_dist_path = os.path.join(plots_dir, 'maha_scores_dist.png')
    plt.savefig(maha_dist_path)
    plt.close()
    print(f"Mahalanobis scores distribution plot saved to {maha_dist_path}")

    # ===== Save metrics =====
    metrics_file_path = os.path.join(results_dir, 'metrics.csv')
    
    current_run_metrics = {}
    current_run_metrics['ID_Dataset'] = 'CIFAR-10'
    current_run_metrics['OOD_Dataset'] = args.ood.upper()
    current_run_metrics['AE_Bottleneck'] = args.ae_bottleneck_dim
    current_run_metrics['CIFAR-10 Val Acc'] = f'{acc:.4f}'
    current_run_metrics['AUROC Recon Error'] = f'{auroc_recon:.4f}'
    current_run_metrics['AUPR-Out Recon Error'] = f'{metrics_recon["aupr_out"]:.4f}'
    current_run_metrics['AUPR-In Recon Error'] = f'{metrics_recon["aupr_in"]:.4f}'
    current_run_metrics['FPR@95TPR Recon Error'] = f'{metrics_recon["fpr_at_95_tpr"]:.4f}'
    current_run_metrics['DetectionErr Recon Error'] = f'{metrics_recon["detection_error"]:.4f}'

    current_run_metrics['AUROC Softmax Score'] = f'{auroc_sm:.4f}'
    current_run_metrics['AUPR-Out Softmax Score'] = f'{metrics_sm["aupr_out"]:.4f}'
    current_run_metrics['AUPR-In Softmax Score'] = f'{metrics_sm["aupr_in"]:.4f}'
    current_run_metrics['FPR@95TPR Softmax Score'] = f'{metrics_sm["fpr_at_95_tpr"]:.4f}'
    current_run_metrics['DetectionErr Softmax Score'] = f'{metrics_sm["detection_error"]:.4f}'

    current_run_metrics['AUROC Mahalanobis'] = f'{auroc_maha:.4f}'
    current_run_metrics['AUPR-Out Mahalanobis'] = f'{metrics_maha["aupr_out"]:.4f}'
    current_run_metrics['AUPR-In Mahalanobis'] = f'{metrics_maha["aupr_in"]:.4f}'
    current_run_metrics['FPR@95TPR Mahalanobis'] = f'{metrics_maha["fpr_at_95_tpr"]:.4f}'
    current_run_metrics['DetectionErr Mahalanobis'] = f'{metrics_maha["detection_error"]:.4f}'

    current_run_metrics['Classifier_Train_Dir'] = args.classifier_experiment_dir
    current_run_metrics['Autoencoder_Train_Dir'] = args.autoencoder_experiment_dir

    with open(metrics_file_path, 'w', newline='', encoding='utf-8') as f_write:
        writer = csv.writer(f_write)
        writer.writerow(['Metric', 'Value'])
        for key, value in current_run_metrics.items():
            writer.writerow([key, value])
    print(f"Evaluation metrics for CIFAR-10 vs {args.ood.upper()} (AE b{args.ae_bottleneck_dim}) saved to {metrics_file_path}")

    # ===== OOD metrics =====
    ood_metrics = metrics_sm
    ood_metrics_path = os.path.join(results_dir, 'ood_metrics_summary.csv')
    with open(ood_metrics_path, 'w', newline='', encoding='utf-8') as f_write:
        writer = csv.writer(f_write)
        writer.writerow(['Metric', 'Value'])
        for key, value in ood_metrics.items():
            writer.writerow([key, value])
    print(f"OOD metrics for CIFAR-10 vs {args.ood.upper()} (AE b{args.ae_bottleneck_dim}) saved to {ood_metrics_path}")

    # ===== Plot PR curves =====
    plot_pr_curve(ood_metrics['pr_curve_out_data'][0], ood_metrics['pr_curve_out_data'][1], f'PR Curve for {args.ood.upper()} (OOD) - Softmax based', os.path.join(plots_dir, 'pr_curve_out_summary.png'))
    plot_pr_curve(ood_metrics['pr_curve_in_data'][0], ood_metrics['pr_curve_in_data'][1], f'PR Curve for ID (In-distribution) - Softmax based', os.path.join(plots_dir, 'pr_curve_in_summary.png'))

if __name__ == '__main__':
    main()