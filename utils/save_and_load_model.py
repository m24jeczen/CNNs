import torch
import os

def save_model(model, model_name, save_dir="trained_models", epoch=None, optimizer=None):
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"{model_name}"
    if epoch is not None:
        filename += f"_epoch{epoch}"
    filename += ".pt"

    save_path = os.path.join(save_dir, filename)
    
    save_dict = {
        'model_state_dict': model.state_dict()
    }

    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            save_dict['epoch'] = epoch

    torch.save(save_dict, save_path)
    print(f"Model saved to: {save_path}")

def load_model(model_class, model_name, save_dir="trained_models", epoch=None, optimizer=None):
    filename = f"{model_name}"
    if epoch is not None:
        filename += f"_epoch{epoch}"
    filename += ".pt"

    load_path = os.path.join(save_dir, filename)
    
    model = model_class()
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Model loaded from: {load_path}")
    return model

