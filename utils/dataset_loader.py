import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# def get_dataloaders(batch_size=64, data_dir="path_to_cinic10"): # for now no augumentation

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))  # simple cnn expects input in range [-1, 1]
#     ])
    
#     train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
#     test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
#     return train_loader, test_loader

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import random

def get_dataloaders(batch_size=64, data_dir="path_to_cinic10", augmentations=None):
    """
    Loads CINIC-10 dataset with optional data augmentation.

    Parameters:
    - batch_size: Number of samples per batch.
    - data_dir: Path to the dataset directory.
    - augmentations: List of augmentation techniques to apply. Options:
      ["rotation", "translation", "noise"]

    Returns:
    - train_loader: DataLoader for training data.
    - test_loader: DataLoader for testing data.
    """
    
    # Base transformations (applied always)
    transform_list = [transforms.ToTensor()]
    
    # Apply selected augmentations
    if augmentations:
        if "rotation" in augmentations:
            transform_list.append(transforms.RandomRotation(degrees=15))  # ±15 degrees
        if "translation" in augmentations:
            transform_list.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))  # ±10% translation
        if "noise" in augmentations:
            transform_list.append(AddGaussianNoise(mean=0, std=0.05))  # Custom noise function

    # Normalize (applies always)
    transform_list.append(transforms.Normalize((0.5,), (0.5,)))

    # Compose all transformations
    transform = transforms.Compose(transform_list)

    # Load dataset
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]))  # No augmentation for test set

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class AddGaussianNoise(object):
    """Custom transform to add Gaussian noise to images."""
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"
