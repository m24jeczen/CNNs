import torch
from models.simple_cnn import SimpleCNN
from models.cnn_with_fc import CNNWithFC
from utils.dataset_loader import get_dataloaders
from training.train import train_model
from evaluation.evaluate import evaluate_model
from models.deep_cnn import DeepCNN
from models.mlp_mixer import MLPMixer
from utils.save_and_load_model import save_model, load_model
from utils.hard_voting import hard_voting_ensemble

torch.manual_seed(42)
scheduler_type = None  # "one_cycle" or "step_decay"
augmentations=["rotation", "translation"] # rotation "translation", "noise"
lr = 0.001
l2_reg = 0.0001
epochs = 30
dropout_p = 0.3
batch_size = 128 # 128, 512

def run_test(batch_size = batch_size, lr=lr, epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type, dropout_p=dropout_p):
    train_loader, test_loader = get_dataloaders(batch_size=batch_size, data_dir="./data", augmentations=augmentations)

    model_deep_cnn = DeepCNN(p_dropout=dropout_p)
    print('--- Deep CNN ---')
    train_model(model_deep_cnn, train_loader, lr=lr, epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type, dropout_p=dropout_p)
    evaluate_model(model_deep_cnn, test_loader, num_epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type, dropout_p=dropout_p)
    save_model(model_deep_cnn, "Deep_CNN_l2_0001", epoch=epochs)

    model_simple_cnn = SimpleCNN(dropout_p=dropout_p)
    print('--- Simple CNN ---')
    train_model(model_simple_cnn, train_loader, lr=lr, epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type, dropout_p=dropout_p)
    evaluate_model(model_simple_cnn, test_loader, num_epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type, dropout_p=dropout_p)
    save_model(model_simple_cnn, "Simple_CNN_l2_0001", epoch=epochs)

    model_cnn_with_fc = CNNWithFC(dropout_p=dropout_p)
    print('--- CNN with FC ---')
    train_model(model_cnn_with_fc, train_loader, lr=lr, epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type, dropout_p=dropout_p)
    evaluate_model(model_cnn_with_fc, test_loader, num_epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type, dropout_p=dropout_p)
    save_model(model_simple_cnn, "CNN_with_fc_l2_0001", epoch=epochs)

    model_mlp = MLPMixer()
    print('--- MLP Mixer ---')
    train_model(model_mlp, train_loader, lr=lr, epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type, dropout_p=dropout_p)
    evaluate_model(model_mlp, test_loader, num_epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type, dropout_p=dropout_p)
    save_model(model_simple_cnn, "MLP_Mixer_l2_0001", epoch=epochs)

    print('-------------- SUCCESS ----------------')


run_test(batch_size = batch_size, lr=lr, epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type, dropout_p=dropout_p)

# One cycle test
# torch.manual_seed(42)
# scheduler_type = "one_cycle"  # "one_cycle" or "step_decay"
# augmentations=["rotation", "translation"] # rotation "translation", "noise"
# lr = 0.001
# l2_reg = 0.0
# epochs = 30

# train_loader, test_loader = get_dataloaders(batch_size=128, data_dir="./data", augmentations=augmentations)

# model_deep_cnn = DeepCNN()
# print('--- Deep CNN ---')
# train_model(model_deep_cnn, train_loader, lr=lr, epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type)
# evaluate_model(model_deep_cnn, test_loader, num_epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type)

# save_model(model_deep_cnn, "Deep_CNN_one_cycle2", epoch=epochs)


# model_simple_cnn = SimpleCNN()
# print('--- Simple CNN ---')
# train_model(model_simple_cnn, train_loader, lr=lr, epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type)
# evaluate_model(model_simple_cnn, test_loader, num_epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type)

# save_model(model_simple_cnn, "Simple_CNN_one_cycle2", epoch=epochs)

# model_cnn_with_fc = CNNWithFC()
# print('--- CNN with FC ---')
# train_model(model_cnn_with_fc, train_loader, lr=lr, epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type)
# evaluate_model(model_cnn_with_fc, test_loader, num_epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type)

# save_model(model_cnn_with_fc, "CNN_with_fc_one_cycle2", epoch=epochs)

# model_mlp = MLPMixer()
# print('--- MLP Mixer ---')
# train_model(model_mlp, train_loader, lr=lr, epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type)
# evaluate_model(model_mlp, test_loader, num_epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type)

# save_model(model_cnn_with_fc, "MLP_Mixer_one_cycle2", epoch=epochs)


# print('-------------- SUCCESS 1----------------')

# One cycle test
# torch.manual_seed(42)
# scheduler_type = "one_cycle"  # "one_cycle" or "step_decay"
# augmentations=["rotation", "translation"] # rotation "translation", "noise"
# lr = 0.001
# l2_reg = 0.0
# epochs = 30
# dropout_p = 0.3

# train_loader, test_loader = get_dataloaders(batch_size=128, data_dir="./data", augmentations=augmentations)

# model_deep_cnn = DeepCNN(p_dropout=dropout_p)
# print('--- Deep CNN ---')
# train_model(model_deep_cnn, train_loader, lr=lr, epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type, dropout_p=dropout_p)
# evaluate_model(model_deep_cnn, test_loader, num_epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type, dropout_p=dropout_p)

# save_model(model_deep_cnn, "Deep_CNN_one_cycle2", epoch=epochs)

# model_simple_cnn = SimpleCNN()
# print('--- Simple CNN ---')
# train_model(model_simple_cnn, train_loader, lr=lr, epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type)
# evaluate_model(model_simple_cnn, test_loader, num_epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type)

# save_model(model_simple_cnn, "Simple_CNN_one_cycle", epoch=epochs)

# model_cnn_with_fc = CNNWithFC()
# print('--- CNN with FC ---')
# train_model(model_cnn_with_fc, train_loader, lr=lr, epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type)
# evaluate_model(model_cnn_with_fc, test_loader, num_epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type)

# save_model(model_cnn_with_fc, "CNN_with_fc_one_cycle", epoch=epochs)

# model_mlp = MLPMixer()
# print('--- MLP Mixer ---')
# train_model(model_mlp, train_loader, lr=lr, epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type)
# evaluate_model(model_mlp, test_loader, num_epochs=epochs, l2_reg=l2_reg, augmentations=augmentations, scheduler_type=scheduler_type)

# save_model(model_cnn_with_fc, "MLP_Mixer_one_cycle", epoch=epochs)

# print('-------------- SUCCESS 2----------------')

# loaded_model = load_model(DeepCNN, "Deep_CNN_test3", epoch=epochs)
# evaluate_model(loaded_model, test_loader, num_epochs=epochs, l2_reg=0.0, augmentations=augmentations)

# model1 = load_model(SimpleCNN, "Simple_CNN_test3", epoch=epochs)
# model2 = load_model(DeepCNN, "Deep_CNN_test3", epoch=epochs)
# model3 = load_model(CNNWithFC, "CNN_with_fc_test3", epoch=epochs)
# hard_voting_ensemble(
#     models=[model1, model2, model3],
#     test_loader=test_loader,
#     num_epochs=epochs,
#     l2_reg=l2_reg,
#     min_lr=0.001,
#     max_lr=0.003,
#     scheduler_type="one_cycle",
#     augmentations=["rotation"]
# )