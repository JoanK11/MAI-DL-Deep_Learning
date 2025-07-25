# Out-of-Distribution (OOD) Detection Project: CIFAR-10 vs. OOD Datasets

This repository implements a suite of models and evaluation scripts for
out-of-distribution (OOD) detection, comparing CIFAR-10 (in-distribution)
against various OOD datasets (TinyImageNet, CIFAR-100, SVHN).

Original project statement: `docs/statement.pdf`

---

## 📂 Folder Structure

```
project2/
├── docs/statement.pdf          # Original project description and requirements
├── src/                        # Source code
│   ├── dataset.py             # Data loading & preprocessing for CIFAR-10 + OOD sets
│   ├── models.py              # Model definitions: ResNet18 & ConvAutoencoder
│   ├── train_classifier.py    # Train & validate ResNet18 on CIFAR-10
│   ├── train_autoencoder.py   # Train & validate convolutional autoencoder on CIFAR-10
│   ├── evaluate.py            # Evaluate classification & OOD detection methods
│   ├── mahalanobis.py         # Compute Mahalanobis distance scores on classifier features
│   ├── utils.py               # Training loops, checkpointing, helper functions
│   └── __init__.py
├── pretrained_models/         # Local store for pretrained ResNet-18 weights
├── data/                      # Downloaded datasets (auto-downloaded by scripts)
├── results/                   # Experiment subdirectories with checkpoints & logs
├── plots/                     # Generated plots (ROC, PR, histograms, etc.)
└── README.md                  # Project overview & instructions (this file)
```

---

## 🔧 Setup & Dependencies

Install required Python packages:

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn tqdm
```

Ensure your working directory is the project root (`project2/`).

---

## 🚀 Usage

### 1. Train ResNet18 Classifier on CIFAR-10

```bash
python src/train_classifier.py \
  --data ./data \
  --epochs 130 \
  --batch_size 512 \
  --lr 0.1 \
  --ood tinyimagenet
```

**Outputs** (in `results/cifar10_tinyimagenet/...`):
- `resnet18.pt` (model checkpoint)
- `training_log.csv` (epoch, loss, accuracy)
- `loss_plot.png`, `accuracy_plot.png`
- Console: `CLASSIFIER_RESULTS_DIR` path

### 2. Train Convolutional Autoencoder on CIFAR-10

```bash
python src/train_autoencoder.py \
  --data ./data \
  --epochs 100 \
  --batch_size 512 \
  --lr 1e-3 \
  --bottleneck 64 \
  --ood tinyimagenet
```

**Outputs** (in `results/cifar10_tinyimagenet/...`):
- `autoencoder.pt` (model checkpoint)
- `training_log.csv` (epoch, train_mse, val_mse)
- `mse_plot.png`
- Console: `AUTOENCODER_RESULTS_DIR` path

Launches multiple `train_autoencoder.py` runs over bottleneck dims & learning rates.

### 4. Evaluate Classification & OOD Detection

```bash
python src/evaluate.py \
  --data ./data \
  --batch_size 512 \
  --ood tinyimagenet \
  --classifier_experiment_dir results/cifar10_tinyimagenet/<clf_expt> \
  --autoencoder_experiment_dir results/cifar10_tinyimagenet/<ae_expt> \
  --ae_bottleneck_dim 64
```

**Outputs** (in `results/cifar10_tinyimagenet/.../eval_...`):

1. **Classification (CIFAR-10 ID)**
   - `classification_report.csv`: Per-class precision, recall, F1, support
   - `entropy_per_class.png`: Boxplot of prediction entropy by true class
   - `sample_predictions_grid.png`: Example correct/incorrect image predictions

2. **Autoencoder Reconstruction Error OOD**
   - `roc_recon.png`, `pr_out_recon.png`, `pr_in_recon.png`
   - `recon_scores_dist.png`: Histograms of MSE for ID vs OOD

3. **Softmax Confidence OOD**
   - `roc_sm.png`, `pr_out_sm.png`, `pr_in_sm.png`
   - `sm_scores_dist.png`: Histograms of –max_softmax (higher = more OOD)

4. **Mahalanobis Distance OOD**
   - `roc_mahalanobis.png`, `pr_out_mahalanobis.png`, `pr_in_mahalanobis.png`
   - `maha_scores_dist.png`: Histograms of Mahalanobis distances

5. `metrics.csv`: Combined AUROC, AUPR, FPR@95%TPR, detection error for all methods
6. `ood_metrics_summary.csv`: Summary of OOD metrics (softmax by default)

---

## 📊 Results & Report Guidance

To write the report, refer to:

- **Classification Performance**:
  - Table from `classification_report.csv` — per-class accuracy, precision, recall
  - `entropy_per_class.png` — confidence distribution insights
  - `sample_predictions_grid.png` — qualitative examples

- **OOD Detection Comparison**:
  - **Reconstruction Error** vs **Softmax** vs **Mahalanobis**
  - ROC curves: `roc_recon.png`, `roc_sm.png`, `roc_mahalanobis.png`
  - PR curves: `pr_out_*`, `pr_in_*`
  - Histograms: `recon_scores_dist.png`, `sm_scores_dist.png`, `maha_scores_dist.png`
  - Metrics summary (`metrics.csv`): AUROC, AUPR, FPR@95%TPR, detection error — compare strengths & weaknesses

Include figure references in your report, e.g.: _"Figure X shows the ROC curve for OOD detection using reconstruction error (see `plots/roc_recon.png`)."_

Link back to `docs/statement.pdf` for the original problem description and evaluation criteria.

---

## 📚 References & Acknowledgments

- Original project statement: `docs/statement.pdf`
- ResNet18 pretrained weights (`pretrained_models/resnet18-f37072fd.pth`):
  - Official PyTorch model zoo weights (ResNet18_Weights.IMAGENET1K_V1)
  - Pretrained on ImageNet-1K (ILSVRC2012) classification dataset
  - Performance: 69.758% top-1 and 89.078% top-5 accuracy on ImageNet-1K
  - Source: https://download.pytorch.org/models/resnet18-f37072fd.pth
  - Training details available in torchvision GitHub references
- Mahalanobis OOD method inspired by Denouden et al. (2018).

---