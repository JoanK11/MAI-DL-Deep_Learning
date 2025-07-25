import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

def get_resnet18(num_classes: int = 10, pretrained: bool = True, weights_path: str | None = None) -> nn.Module:
    if weights_path:
        model = models.resnet18(weights=None)
        model.load_state_dict(torch.load(weights_path))
    elif pretrained:
        model = models.resnet18(weights=None)
        local_weights_path = "pretrained_models/resnet18-f37072fd.pth"
        model.load_state_dict(torch.load(local_weights_path))
    else:
        model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class ConvAutoencoder(nn.Module):
    def __init__(self, bottleneck_dim: int = 64):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 16×16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), # 8×8
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),# 4×4
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(128*4*4, bottleneck_dim),
        )
        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(bottleneck_dim, 128*4*4),
            nn.Unflatten(1, (128,4,4)),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 8×8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16×16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1)   # 32×32
        )

    def forward(self, x):
        latent = self.enc(x)
        recon  = self.dec(latent)
        return recon, latent