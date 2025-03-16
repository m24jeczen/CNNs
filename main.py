import torch
from models.simple_cnn import SimpleCNN
from models.cnn_with_fc import CNNWithFC
from utils.dataset_loader import get_dataloaders
from training.train import train_model
from evaluation.evaluate import evaluate_model
from models.deep_cnn import DeepCNN
from models.mlp_mixer import MLPMixer

torch.manual_seed(42)

train_loader, test_loader = get_dataloaders(batch_size=128, data_dir="./data", augmentations="rotation")

# model_simple_cnn = SimpleCNN()
# print('--- Simple CNN ---')
# train_model(model_simple_cnn, train_loader, lr=0.001, epochs=20)
# evaluate_model(model_simple_cnn, test_loader)

# model_cnn_with_fc = CNNWithFC()
# print('--- CNN with FC ---')
# train_model(model_cnn_with_fc, train_loader, lr=0.001, epochs=20)
# evaluate_model(model_cnn_with_fc, test_loader)

model_deep_cnn = DeepCNN()
print('--- Deep CNN ---')
train_model(model_deep_cnn, train_loader, lr=0.001, epochs=35)
evaluate_model(model_deep_cnn, test_loader)

# model_mlp = MLPMixer()
# print('--- MLP Mixer ---')
# train_model(model_mlp, train_loader, lr=0.001, epochs=35)
# evaluate_model(model_mlp, test_loader)
