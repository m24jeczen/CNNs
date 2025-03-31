import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, StepLR
from torchmetrics.classification import Accuracy, Precision, Recall
import os
import csv

def save_training_results(model_name, num_epochs, augmentations_str, batch_size, l2_reg, lr, max_lr, scheduler_type, training_results, dropout_p):

    filename = f"experiments/{model_name}_{num_epochs}_{augmentations_str}_{batch_size}_{l2_reg}_{lr}_{max_lr}_{scheduler_type}_{dropout_p}.csv"
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss", "Accuracy", "Precision", "Recall"])
        for result in training_results:
            writer.writerow(result)
    
    print(f"Training results saved to {filename}")

def train_model(model, train_loader, lr=0.001, epochs=10, scheduler_type=None, step_size=10, gamma=0.7, max_lr=0.003, l2_reg=0.0,augmentations=None, dropout_p=0.5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    
    accuracy = Accuracy(task="multiclass", num_classes=10).to(device)
    precision = Precision(task="multiclass", average="macro", num_classes=10).to(device)
    recall = Recall(task="multiclass", average="macro", num_classes=10).to(device)
    
    if scheduler_type == "one_cycle":
        scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=epochs)
    elif scheduler_type == "step_decay":
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        scheduler = None
    
    training_results = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if scheduler_type == "one_cycle":
                scheduler.step()
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            accuracy.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
        
        if scheduler_type == "step_decay":
            scheduler.step()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy.compute().item()
        epoch_prec = precision.compute().item()
        epoch_recall = recall.compute().item()
        
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, Prec={epoch_prec:.4f}, Recall={epoch_recall:.4f}")
        
        training_results.append((epoch + 1, epoch_loss, epoch_acc, epoch_prec, epoch_recall))
        
        accuracy.reset()
        precision.reset()
        recall.reset()
    
    augmentations_str = "_".join(augmentations) if augmentations else "none"
    save_training_results(model.__class__.__name__, epochs, augmentations_str, train_loader.batch_size, l2_reg, lr, max_lr, scheduler_type, training_results, dropout_p)

    return model,optimizer
