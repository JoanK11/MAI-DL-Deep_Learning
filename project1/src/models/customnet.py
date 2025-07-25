import torch
from torch import nn
from torchvision import transforms


class CustomNet (nn.Module):
  def __init__ (self, num_classes, dropout_ratio=0.4):
    super(CustomNet, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=3, stride=1),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(),

      nn.Conv2d(32, 32, kernel_size=3, stride=2),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(),

      nn.Conv2d(32, 64, kernel_size=3, stride=1),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(),

      nn.Conv2d(64, 64, kernel_size=3, stride=2),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(),
    )
    self.classifier = nn.Sequential(
      nn.Linear(64*53*53, 256),
      nn.BatchNorm1d(1),
      nn.LeakyReLU(),
      nn.Dropout(dropout_ratio),
      nn.Linear(256, num_classes)
    )
  
  def forward (self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = x.unsqueeze(1)
    x = self.classifier(x)
    return torch.flatten(x, 1)
  
  def transforms(self):
    return [
      transforms.Resize((224,224)),
      transforms.ToTensor()
    ]