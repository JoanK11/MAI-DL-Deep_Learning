import pandas as pd
import os
from os.path import dirname
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class MameDataset:
  _instance = None

  @staticmethod
  def get_default_data_path():
    root_path = dirname(dirname(dirname(dirname(os.path.realpath(__file__)))))
    return os.path.join(root_path,"data", "proj1")
  
  @staticmethod
  def initialize_instance(**kwargs):
    MameDataset._instance = MameDataset(**kwargs)

  @staticmethod
  def get_instance():
    if MameDataset._instance is None:
      Exception("trying to get non-initialized mame static instance.")
    return MameDataset._instance

  def __init__(self, data_path=None, testing=False):
    if testing:
      print("\033[5;31mWARNING\033[0m: loading dataset in testing mode, only 5% of dataset considered.")
    # Select default path
    if data_path is None:
      data_path = MameDataset.get_default_data_path()
    data = pd.read_csv(os.path.join(data_path, "MAMe_dataset.csv"))
    # Only consider relevant columns, discard the rest
    relevant_categories = ["Image file","Medium","Subset"]
    data = data[relevant_categories]
    # Extract labels as one hot
    onehot_enconder = OneHotEncoder()
    one_hot_labels = onehot_enconder.fit_transform(np.array(data['Medium']).reshape(-1,1)).toarray()
    self.classes_ = onehot_enconder.categories_[0].tolist()
    # Split into train, test and val sets
    self.train_data_ = []
    self.test_data_ = []
    self.val_data_ = []
    for i, ((_, row), one_hot) in enumerate(zip(data.iterrows(), one_hot_labels)):
      # Testing mode enables loading only 5% of the dataset for iteration speed
      if testing and i%50 != 0:
        continue
      if i%100 == 0:
        print(f"Loading dataset {int(i*100/len(data)):3d}%", end="\r")
      file = row["Image file"]
      subset = row["Subset"]
      img_data = Image.open(os.path.join(data_path, "images", file))
      sample = (img_data, torch.tensor(one_hot, dtype=torch.float32)) # X, y format (with y as one hot)
      if subset == "train":
        self.train_data_.append(sample)
      elif subset == "val":
        self.val_data_.append(sample)
      elif subset == "test":
        self.test_data_.append(sample)
      else:
        raise Exception(f"Invalid data subset found in dataset: {subset}")
    print(f"Loaded {len(self.train_data_)} train, {len(self.val_data_)} val and {len(self.test_data_)} test images.\n")

  @property
  def classes(self):
    return self.classes_
  
  @property
  def train_data(self):
    return self.train_data_
  
  @property
  def val_data(self):
    return self.val_data_
  
  @property
  def test_data(self):
    return self.test_data_


class MameDataSplit(Dataset):
  def __init__(self, data, transforms):
    self.data = data
    self.transforms = transforms
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, index):
    x,y = self.data[index]
    if self.transforms is not None:
      x = self.transforms(x)
    return x,y