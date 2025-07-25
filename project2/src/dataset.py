import torch
from torchvision import datasets, transforms
from typing import Tuple
import os
import requests
import zipfile
import io
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

def download_and_extract_tiny_imagenet(root_dir: Path, dataset_path: Path):
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_path = root_dir / 'tiny-imagenet-200.zip'

    if dataset_path.exists():
        print(f"Tiny ImageNet dataset found at {dataset_path}. Skipping download and extraction.")
        return

    if not zip_path.exists():
        print(f"Downloading Tiny ImageNet from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status() 
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded to {zip_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading Tiny ImageNet: {e}")
            if zip_path.exists():
                os.remove(zip_path)
            raise
    else:
        print(f"Found {zip_path}. Skipping download.")
        
    print(f"Extracting {zip_path} to {root_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root_dir)
        print(f"Extracted to {dataset_path}")
    except zipfile.BadZipFile as e:
        print(f"Error extracting Tiny ImageNet: {e}. The downloaded file might be corrupted.")
        if zip_path.exists():
            os.remove(zip_path)
        if dataset_path.exists():
            import shutil
            shutil.rmtree(dataset_path)
        raise

class TinyImageNetDataset(Dataset):
    def __init__(self, root: str, split: str = 'train', transform=None, target_transform=None, download: bool = False):
        self.root = Path(root)
        self.dataset_path = self.root / 'tiny-imagenet-200'
        self.split = split.lower()
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []
        self.targets = []

        if download:
            download_and_extract_tiny_imagenet(self.root, self.dataset_path)

        if not self.dataset_path.exists():
            raise RuntimeError(f"Dataset not found at {self.dataset_path}. "
                               f"Please specify the correct 'root' directory or enable 'download=True'.")

        self._load_metadata()
        self._load_samples()

    def _load_metadata(self):
        wnids_path = self.dataset_path / 'wnids.txt'
        with open(wnids_path, 'r') as f:
            self.wnids = [x.strip() for x in f.readlines()]
        
        self.class_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}
        self.idx_to_class = {i: wnid for i, wnid in enumerate(self.wnids)}

    def _load_samples(self):
        if self.split == 'train':
            train_path = self.dataset_path / 'train'
            for wnid in self.wnids:
                class_dir = train_path / wnid / 'images'
                if not class_dir.is_dir():
                    print(f"Warning: Class directory {class_dir} not found for wnid {wnid}")
                    continue
                for img_file in class_dir.glob('*.JPEG'):
                    self.samples.append(str(img_file))
                    self.targets.append(self.class_to_idx[wnid])
        
        elif self.split == 'val':
            val_images_path = self.dataset_path / 'val' / 'images'
            annotations_path = self.dataset_path / 'val' / 'val_annotations.txt'
            
            val_annotations = {}
            with open(annotations_path, 'r') as f:
                for line_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if parts and len(parts) >= 2:
                        img_name, wnid = parts[0], parts[1]
                        val_annotations[img_name] = wnid
                    else:
                        print(f"DEBUG VAL: Malformed line {line_idx + 1} in {annotations_path}: '{line.strip()}'. Parts after split: {parts}. Length: {len(parts)}. Skipping.")
            
            for img_name, wnid in val_annotations.items():
                img_path = val_images_path / img_name
                if img_path.exists():
                    self.samples.append(str(img_path))
                    self.targets.append(self.class_to_idx[wnid])
                else:
                    print(f"Warning: Image {img_path} not found for validation set.")

        elif self.split == 'test':
            # The official test set for Tiny ImageNet usually has no labels.
            # For OOD, we typically use the 'val' set. If a true 'test' set with labels is needed,
            # this part would require modification or a different data source.
            # For this project, we'll use 'val' as the OOD set.
            print("Using 'val' split for 'test' as Tiny ImageNet official test set has no public labels.")
            val_images_path = self.dataset_path / 'val' / 'images'
            annotations_path = self.dataset_path / 'val' / 'val_annotations.txt'
            
            val_annotations = {}
            with open(annotations_path, 'r') as f:
                for line_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if parts and len(parts) >= 2:
                        img_name, wnid = parts[0], parts[1]
                        val_annotations[img_name] = wnid
                    else:
                        print(f"DEBUG TEST (using VAL): Malformed line {line_idx + 1} in {annotations_path}: '{line.strip()}'. Parts after split: {parts}. Length: {len(parts)}. Skipping.")
            
            for img_name, wnid in val_annotations.items():
                img_path = val_images_path / img_name
                if img_path.exists():
                    self.samples.append(str(img_path))
                    self.targets.append(self.class_to_idx[wnid])
                else:
                    print(f"Warning: Image {img_path} not found for validation set (used as test).")
        else:
            raise ValueError(f"Unknown split: {self.split}. Must be 'train', 'val', or 'test'.")
            
    def __getitem__(self, index):
        img_path = self.samples[index]
        target = self.targets[index]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image on error
            img = Image.new('RGB', (64, 64), color = 'red')

        if self.transform:
            img = self.transform(img)
        
        if self.target_transform:
            target = self.target_transform(target)
            
        return img, target

    def __len__(self):
        return len(self.samples)

def get_dataloaders(data_dir: str,
                    batch_size: int = 256,
                    num_workers: int = 4,
                    ood_dataset_name: str = 'tinyimagenet'
                    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    
    # In-distribution: CIFAR-10
    cifar10_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                           (0.2470, 0.2435, 0.2616))

    transform_train_cifar10 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        cifar10_normalize,
    ])
    transform_test_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        cifar10_normalize,
    ])
    
    trainset_cifar10 = datasets.CIFAR10(root=data_dir, train=True,
                                        download=True, transform=transform_train_cifar10)
    valset_cifar10 = datasets.CIFAR10(root=data_dir, train=False,
                                      download=True, transform=transform_test_cifar10)

    # Out-of-distribution dataset
    ood_dataset_name = ood_dataset_name.lower()
    print(f"Loading OOD dataset: {ood_dataset_name}")

    if ood_dataset_name == 'tinyimagenet':
        tiny_imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
        transform_ood = transforms.Compose([
            transforms.Resize(32), # Tiny ImageNet images are 64x64, resize to 32x32
            transforms.ToTensor(),
            tiny_imagenet_normalize,
        ])
        ood_dataset = TinyImageNetDataset(root=data_dir, split='val', # Use 'val' split of TinyImageNet
                                           download=True, transform=transform_ood)
    elif ood_dataset_name == 'cifar100':
        cifar100_normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                std=[0.2675, 0.2565, 0.2761])
        transform_ood = transforms.Compose([
            transforms.ToTensor(),
            cifar100_normalize,
        ])
        ood_dataset = datasets.CIFAR100(root=data_dir, train=False, # Use test set of CIFAR-100
                                         download=True, transform=transform_ood)
    elif ood_dataset_name == 'svhn':
        svhn_normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                            std=[0.1980, 0.2010, 0.1970])
        transform_ood = transforms.Compose([
            transforms.ToTensor(),
            svhn_normalize,
        ])
        ood_dataset = datasets.SVHN(root=data_dir, split='test', # Use test set of SVHN
                                    download=True, transform=transform_ood)
    else:
        raise ValueError(f"Unsupported OOD dataset: {ood_dataset_name}. Choose from 'tinyimagenet', 'cifar100', 'svhn'.")

    train_loader = torch.utils.data.DataLoader(trainset_cifar10, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valset_cifar10, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers, pin_memory=True)
    ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, ood_loader