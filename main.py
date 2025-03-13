import torch
from models.simple_cnn import SimpleCNN
from data.dataset_loader import get_dataloaders
from training.train import train_model

torch.manual_seed(42)

train_loader, test_loader = get_dataloaders(batch_size=64, data_dir="path_to_cinic10")

model = SimpleCNN()

train_model(model, train_loader, lr=0.001, epochs=10)