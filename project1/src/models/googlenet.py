import torch
import torch.nn as nn
from torchvision import transforms


class Inception(nn.Module):
  def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3,
                ch5x5red, ch5x5, pool_proj):
    super(Inception, self).__init__()

    self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

    self.branch2 = nn.Sequential(
      nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
      nn.ReLU(True),
      nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
      nn.ReLU(True)
    )

    self.branch3 = nn.Sequential(
      nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
      nn.ReLU(True),
      nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
      nn.ReLU(True)
    )

    self.branch4 = nn.Sequential(
      nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
      nn.Conv2d(in_channels, pool_proj, kernel_size=1),
      nn.ReLU(True)
    )

  def forward(self, x):
    return torch.cat([
      self.branch1(x),
      self.branch2(x),
      self.branch3(x),
      self.branch4(x)
    ], 1)


class GoogLeNet(nn.Module):
  def __init__(self, num_classes, dropout_ratio=0.4):
    super(GoogLeNet, self).__init__()
    self.pre_layers = nn.Sequential(
      # Part 1
      nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
      nn.ReLU(True),
      nn.MaxPool2d(3, stride=2, ceil_mode=True),
      nn.LocalResponseNorm(5),

      # Part 2
      nn.Conv2d(64, 64, kernel_size=1),
      nn.Conv2d(64, 192, kernel_size=3, padding=1),
      nn.ReLU(True),
      nn.LocalResponseNorm(5),
      nn.MaxPool2d(3, stride=2, ceil_mode=True),

      # Part 3
      Inception(192, 64, 96, 128, 16, 32, 32),
      Inception(256, 128, 128, 192, 32, 96, 64),
      nn.MaxPool2d(3, stride=2, ceil_mode=True),

      # Part 4
      Inception(480, 192, 96, 208, 16, 48, 64),
      Inception(512, 160, 112, 224, 24, 64, 64),
      Inception(512, 128, 128, 256, 24, 64, 64),
      Inception(512, 112, 144, 288, 32, 64, 64),
      Inception(528, 256, 160, 320, 32, 128, 128),
      nn.MaxPool2d(3, stride=2, ceil_mode=True),

      # Part 5
      Inception(832, 256, 160, 320, 32, 128, 128),
      Inception(832, 384, 192, 384, 48, 128, 128),
    )

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.dropout = nn.Dropout(dropout_ratio)
    self.linear =  nn.Linear(1024, num_classes)


  def forward(self, x):
    x = self.pre_layers(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.dropout(x)
    x = self.linear(x)
    return x
  

  def transforms(self):
    return [
      transforms.Resize((224,224)),
      transforms.ToTensor()
    ]