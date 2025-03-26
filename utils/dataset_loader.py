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
    
    transform_list = [transforms.ToTensor()]
    
    if augmentations:
        if "rotation" in augmentations:
            transform_list.append(transforms.RandomRotation(degrees=15)) 
        if "translation" in augmentations:
            transform_list.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))  
        if "noise" in augmentations:
            transform_list.append(AddGaussianNoise(mean=0, std=0.05))  

    transform_list.append(transforms.Normalize((0.5,), (0.5,)))

    transform = transforms.Compose(transform_list)

    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]))  

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"
