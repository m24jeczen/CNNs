import torch
from models.simple_cnn import SimpleCNN
from models.cnn_with_fc import CNNWithFC
from data.dataset_loader import get_dataloaders
from training.train import train_model
from evaluation.evaluate import evaluate_model
from models.deep_cnn import DeepCNN

torch.manual_seed(42)

train_loader, test_loader = get_dataloaders(batch_size=128, data_dir="./data")

#model = SimpleCNN()
#model = CNNWithFC()
model = DeepCNN()


train_model(model, train_loader, lr=0.001, epochs=30)
evaluate_model(model, test_loader)
