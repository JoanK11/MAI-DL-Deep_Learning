import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class ZfNet(nn.Module):
  def __init__(self, num_classes, dropout_ratio=0.4):
    super(ZfNet, self).__init__()
    self.convs = self.get_convs()
    self.fc = self.get_fc(num_classes, dropout_ratio)
  

  def _init_conv2d(self, inc, outc, kernel_size, stride, padding):
    # Create conv2d layer and init params as in original paper
    conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=stride, padding=padding)
    nn.init.normal_(conv.weight, mean=0.0, std=0.02)
    nn.init.constant_(conv.bias, 0.0)
    return conv


  def get_convs(self):
    layers = []
    layers.append(self._init_conv2d(3, 96, kernel_size=7, stride=2, padding=1))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    layers.append(nn.LocalResponseNorm(5))

    layers.append(self._init_conv2d(96, 256, kernel_size=5, stride=2, padding=0))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    layers.append(nn.LocalResponseNorm(5))

    layers.append(self._init_conv2d(256, 384, kernel_size=3, stride=1, padding=1))
    layers.append(nn.ReLU(inplace=True))

    layers.append(self._init_conv2d(384, 384, kernel_size=3, stride=1, padding=1))
    layers.append(nn.ReLU(inplace=True))

    layers.append(self._init_conv2d(384, 256, kernel_size=3, stride=1, padding=1))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
    return nn.Sequential(*layers)
  

  def _init_linear(self, inn, outn):
    # Create linear layer and init params as in original paper
    fc = nn.Linear(inn, outn)
    nn.init.normal_(fc.weight, mean=0.0, std=0.02)
    nn.init.constant_(fc.bias, 0.0)
    return fc


  def get_fc(self, num_classes, dropout_ratio):
    layers = []
    layers.append(self._init_linear(9216, 4096))
    layers.append(nn.Dropout(dropout_ratio))

    layers.append(self._init_linear(4096, num_classes))
    layers.append(nn.Dropout(dropout_ratio))
    return nn.Sequential(*layers)


  def forward(self, x):
    y = self.convs(x)
    y = y.view(-1, 9216)
    y = self.fc(y)
    return y
  
  def transforms(self):
    return [
      transforms.Resize((224,224)),
      transforms.ToTensor()
    ]