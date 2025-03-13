import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64, data_dir="path_to_cinic10"): # for now no augumentation

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # simple cnn expects input in range [-1, 1]
    ])
    
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader