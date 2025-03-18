import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, StepLR
from torchmetrics.classification import Accuracy, Precision, Recall


def train_model(model, train_loader, lr=0.001, epochs=10, 
                scheduler_type=None, step_size=5, gamma=0.1, max_lr=0.003, weight_decay=0.0):
    """
    Trains a PyTorch model with optional learning rate scheduling and L2 regularization.

    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for training.
        lr (float, optional): Initial learning rate. Default is 0.001.
        epochs (int, optional): Number of epochs. Default is 10.
        scheduler_type (str, optional): Learning rate scheduler type ('one_cycle' or 'step_decay').
        step_size (int, optional): Step size for StepLR. Default is 5.
        gamma (float, optional): Decay factor for StepLR. Default is 0.1.
        max_lr (float, optional): Max learning rate for OneCycleLR. Default is 0.003.
        weight_decay (float, optional): L2 regularization strength. Default is 0.0 (disabled).
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Apply L2 Regularization through weight_decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Metric initialization
    accuracy = Accuracy(task="multiclass", num_classes=10).to(device)
    precision = Precision(task="multiclass", average="macro", num_classes=10).to(device)
    recall = Recall(task="multiclass", average="macro", num_classes=10).to(device)

    # Scheduler setup
    if scheduler_type == "one_cycle":
        scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=epochs)
    elif scheduler_type == "step_decay":
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        scheduler = None

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

        # Step the scheduler if using step decay
        if scheduler_type == "step_decay":
            scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy.compute().item()
        epoch_prec = precision.compute().item()
        epoch_recall = recall.compute().item()

        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, Prec={epoch_prec:.4f}, Recall={epoch_recall:.4f}")

        # Reset metrics for next epoch
        accuracy.reset()
        precision.reset()
        recall.reset()
