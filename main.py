import random
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from models.protypical_net import load_cinic10_few_shot, ProtoNet
from models.simple_cnn import SimpleCNN
from models.cnn_with_fc import CNNWithFC
from utils.dataset_loader import get_dataloaders
from training.train import train_model
from evaluation.evaluate import evaluate_model
from models.deep_cnn import DeepCNN
from models.mlp_mixer import MLPMixer

torch.manual_seed(42)
scheduler_type = None # "one_cycle" or "step_decay"
augmentations=["rotation"] #  "translation", "noise", "style", "erasing"
lr = 0.001
l2_reg = 0.0
epochs = 5

def perform_experiment(model, train_loader, test_loader, lr, epochs, l2_reg, augmentations):
    print(f'--- {model.__class__.__name__} ---')
    train_model(model, train_loader, lr=lr, epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type)
    evaluate_model(model, test_loader, num_epochs=epochs, l2_reg=l2_reg, augmentations=augmentations)

few_shot_train, few_shot_test = load_cinic10_few_shot("./data", num_samples_per_class=16, batch_size=16, augmentations=augmentations)

#model = ProtoNet()
# train_model(model, few_shot_train, lr=lr, epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type)
# evaluate_model(model, few_shot_test, num_epochs=epochs, l2_reg=l2_reg, augmentations=augmentations)
#augemnatiion experiment

augmentations_for_experiment = [["noise"], ["style"], ["erasing"], ["rotation"], ["translation"], None]
for augmentation in augmentations_for_experiment:
    print(f'--- Augmentation: {augmentation} ---')
    model_simple_cnn = SimpleCNN(dropout_p=0.3)
    model_deep_cnn = DeepCNN(p_dropout=0.3)
    model_cnn_with_fc = CNNWithFC(dropout_p=0.3)
    model_mlp = MLPMixer()
    model_proto = ProtoNet()

    models = [model_simple_cnn]

    #few_shot_train, few_shot_test = load_cinic10_few_shot("./data", num_samples_per_class=16, batch_size=16,
    train_loader, test_loader = get_dataloaders(batch_size=1024, data_dir="./data", augmentations=augmentation)
    for model in models:
        perform_experiment(model, train_loader, test_loader, lr, epochs, l2_reg, augmentation)


# model_deep_cnn = DeepCNN()
# print('--- Deep CNN ---')
# model, optimizer = train_model(model_deep_cnn, train_loader, lr=lr, epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type)
# evaluate_model(model_deep_cnn, test_loader, num_epochs=epochs, l2_reg=l2_reg, augmentations=augmentations)

# model_simple_cnn = SimpleCNN(dropout_p=0.55)
# print('--- Simple CNN ---')
# train_model(model_simple_cnn, train_loader, lr=0.001, epochs=epochs, l2_reg=0.001)
# evaluate_model(model_simple_cnn, test_loader, 2)

# model_cnn_with_fc = CNNWithFC()
# print('--- CNN with FC ---')
# train_model(model_cnn_with_fc, train_loader, lr=0.001, epochs=20)
# evaluate_model(model_cnn_with_fc, test_loader)

# model_mlp = MLPMixer()
# print('--- MLP Mixer ---')
# train_model(model_mlp, train_loader, lr=0.001, epochs=35)
# evaluate_model(model_mlp, test_loader)

# model_proto_net = protoNet()
# print('--- protoNet ---')
# train_model(model_proto_net, train_loader, lr=0.001, epochs=epochs)
# evaluate_model(model_proto_net, test_loader, 2)