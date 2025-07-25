import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist
from scipy import linalg

@torch.no_grad()
def _extract_features(model, x: torch.Tensor, feature_layer_name: str) -> torch.Tensor:
    model.eval()

    out = model.conv1(x)
    out = model.bn1(out)
    out = model.relu(out)

    if hasattr(model, 'maxpool'):
        out = model.maxpool(out)

    if feature_layer_name == 'relu':
        pooled = nn.AdaptiveAvgPool2d((1,1))(out)
        return torch.flatten(pooled, 1)

    out = model.layer1(out)
    if feature_layer_name == 'layer1':
        pooled = nn.AdaptiveAvgPool2d((1,1))(out)
        return torch.flatten(pooled, 1)

    out = model.layer2(out)
    if feature_layer_name == 'layer2':
        pooled = nn.AdaptiveAvgPool2d((1,1))(out)
        return torch.flatten(pooled, 1)

    out = model.layer3(out)
    if feature_layer_name == 'layer3':
        pooled = nn.AdaptiveAvgPool2d((1,1))(out)
        return torch.flatten(pooled, 1)

    out = model.layer4(out)
    if feature_layer_name in ('layer4', 'layer4_avgpool'):
        pooled = nn.AdaptiveAvgPool2d((1,1))(out)
        return torch.flatten(pooled, 1)

    if hasattr(model, 'avgpool'):
        out = model.avgpool(out)
    else:
        out = nn.AdaptiveAvgPool2d((1,1))(out)

    if feature_layer_name == 'avgpool':
        return torch.flatten(out, 1)

    raise ValueError(
        f"Unsupported feature_layer_name: {feature_layer_name}. "
        "Supported: relu, layer1, layer2, layer3, layer4, layer4_avgpool, avgpool."
    )

@torch.no_grad()
def compute_class_stats(model, loader, num_classes: int, device: str = 'cuda',
                        feature_layer_name: str = 'avgpool', reg_magnitude: float = 1e-2):
    """Compute per-class mean and a shared precision matrix."""
    model.eval()
    feats_list = []
    labels_list = []

    for x, y in tqdm(loader, leave=False, desc=f'Feat Extr ({feature_layer_name})'):
        x = x.to(device)
        y = y.to(device)
        f = _extract_features(model, x, feature_layer_name)
        feats_list.append(f.cpu().numpy())
        labels_list.append(y.cpu().numpy())

    if not feats_list:
        raise ValueError("No features extracted. Empty loader?")

    all_feats = np.concatenate(feats_list, axis=0)   # (N, D)
    all_labels = np.concatenate(labels_list, axis=0) # (N,)

    N, D = all_feats.shape
    if N < 2:
        raise ValueError("Need at least 2 samples to compute covariance")

    cov = np.cov(all_feats, rowvar=False, bias=False)  # (D, D), uses N-1
    cov += reg_magnitude * np.eye(D)                   # regularize
    VI = linalg.inv(cov)                               # (D, D)

    class_means = np.zeros((num_classes, D), dtype=all_feats.dtype)
    for c in range(num_classes):
        mask = (all_labels == c)
        if np.any(mask):
            class_means[c] = all_feats[mask].mean(axis=0)
        else:
            pass

    return class_means, VI


@torch.no_grad()
def mahalanobis_scores(model, loader, class_means: np.ndarray, VI: np.ndarray,
                       device: str = 'cuda', feature_layer_name: str = 'avgpool',
                       desc_suffix: str = ""):
    model.eval()
    scores = []

    D = class_means.shape[1]
    tqdm_desc = f'Mahal Scores ({feature_layer_name})'
    if desc_suffix:
        tqdm_desc += f' {desc_suffix}'

    for x, _ in tqdm(loader, leave=False, desc=tqdm_desc):
        x = x.to(device)
        f = _extract_features(model, x, feature_layer_name)
        batch_np = f.cpu().numpy()
        dists = cdist(batch_np, class_means, metric='mahalanobis', VI=VI)
        min_dists = np.min(dists, axis=1)
        scores.append(torch.from_numpy(min_dists))

    if not scores:
        return torch.empty(0)

    return torch.cat(scores)