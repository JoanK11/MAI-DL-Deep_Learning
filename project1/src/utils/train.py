from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from utils.dataloader import MameDataSplit, MameDataset

import matplotlib.pylab as plt

def print_results_table(per_class_ratios):
  classes = MameDataset.get_instance().classes
  max_len = max([len(c) for c in classes])
  for i, (class_name, ratio) in enumerate(zip(classes, per_class_ratios)):
    print(f"{str(i):>2}: {class_name:>{max_len}} {int(ratio[0]*100/ratio[1]):3d}%")


def _test_internal(model, data_loader, text_header, device, criterion, print_table=True):
  model.eval()
  sum_loss = 0
  sum_acc = 0
  sum_f1 = 0
  per_class_ratios = [[0,0] for _ in range(len(MameDataset.get_instance().classes))]
  for iter, (images, labels) in enumerate(data_loader):
    with torch.no_grad():
      images, labels_dev = images.to(device), labels.to(device)
      outputs = model(images)
      loss = criterion(outputs, labels_dev)
      preds = np.argmax(outputs.detach().cpu(), axis=1)
    class_labs = np.argmax(labels, axis=1)
    for cls, pred in zip(class_labs, preds):
      per_class_ratios[cls][1] += 1
      if cls == pred:
        per_class_ratios[cls][0] += 1
    accuracy = accuracy_score(class_labs, preds)
    f1 = f1_score(class_labs, preds, average='micro')
    sum_loss += loss.item()
    sum_acc += accuracy
    sum_f1 += f1
    print(f"{text_header}: {iter+1}/{len(data_loader)} loss={sum_loss/(iter+1):5.3f}  acc={sum_acc/(iter+1):4.3f} f1={sum_f1/(iter+1):4.3f}", end="\r")
  print()
  loss = sum_loss/len(data_loader)
  acc = sum_acc/len(data_loader)
  f1 = sum_f1/len(data_loader)
  print(f"{text_header} results: loss={loss:5.3f}  acc={acc:4.3f} f1={f1:4.3f}")
  if print_table:
    print_results_table(per_class_ratios)
  return loss, acc, f1


def get_basic_train_cfg():
  return SimpleNamespace(
    batch=128,
    epochs=40,
    lr=0.0001,
    device='cuda',
    transforms=[
      transforms.RandomHorizontalFlip(),
      transforms.Resize((256, 256)),  # resize to 256x256 without distortion
      transforms.RandomCrop(224),     # random crop 224x224
    ]
  )


def train_model(model, train_cfg, ptable_each=1):
  if train_cfg.device == 'cuda' and not torch.cuda.is_available():
    Exception("cuda is not available.")

  train_transform = transforms.Compose(train_cfg.transforms + model.transforms()) # Data aug
  test_transform  = transforms.Compose(model.transforms())  # No data aug

  mame_dset = MameDataset.get_instance()
  train_loader = DataLoader(MameDataSplit(mame_dset.train_data, transforms=train_transform),
                            batch_size=train_cfg.batch,
                            shuffle=True)
  val_loader = DataLoader(MameDataSplit(mame_dset.val_data, transforms=test_transform),
                          batch_size=train_cfg.batch)
  test_loader = DataLoader(MameDataSplit(mame_dset.test_data, transforms=test_transform),
                           batch_size=train_cfg.batch)


  model = model.to(train_cfg.device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)
  history = []

  for epoch in range(train_cfg.epochs):
    epoch_history = {'train_loss': [], 'train_acc': [], 'train_f1': [], 'val_loss': 0, 'val_acc': 0, 'val_f1': 0}
    acc_avg = None
    f1_avg = None
    # Training
    model.train()
    for iter, (images, labels) in enumerate(train_loader):
      images, labels_dev = images.to(train_cfg.device), labels.to(train_cfg.device)
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels_dev)
      loss.backward()
      optimizer.step()
      preds = np.argmax(outputs.detach().cpu(), axis=1)
      class_labs = np.argmax(labels, axis=1)
      accuracy = accuracy_score(class_labs, preds)
      f1 = f1_score(class_labs, preds, average='micro')
      acc_avg = accuracy if acc_avg is None else acc_avg + 0.1*(accuracy - acc_avg)
      f1_avg = f1 if f1_avg is None else f1_avg + 0.1*(f1 - f1_avg)
      epoch_history['train_loss'].append(loss.item())
      epoch_history['train_acc'].append(accuracy)
      epoch_history['train_f1'].append(f1)
      print(f"Train batch {epoch+1}/{train_cfg.epochs}: {iter+1}/{len(train_loader)} loss={loss.item():5.3f}  acc={acc_avg:4.3f} f1={f1_avg:4.3f}", end="\r")
    print()

    # Validation
    loss, acc, f1 = _test_internal(model, val_loader, f"Val batch {epoch+1}/{train_cfg.epochs}",
                                   train_cfg.device, criterion, print_table=(((epoch+1)%ptable_each)==0))
    epoch_history['val_loss'] = loss
    epoch_history['val_acc'] = acc
    epoch_history['val_f1'] = f1
    print()

    history.append(epoch_history)
  
  loss, acc, f1 = _test_internal(model, test_loader, f"Test", train_cfg.device, criterion)
  return {'train_epoch_history': history, 'test_data': {'loss': loss, 'acc': acc, 'f1': f1}}


def test_model(model):
  test_transform = transforms.Compose(model.transforms()) # No data aug
  mame_dset = MameDataset.get_instance()
  test_loader = DataLoader(MameDataSplit(mame_dset.test_data, transforms=test_transform),
                           batch_size=64)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  criterion = nn.CrossEntropyLoss()
  return _test_internal(model, test_loader, f"Test", device, criterion)