import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import Accuracy, Precision, Recall


def train_model(model, train_loader, lr=0.001, epochs=10, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    accuracy = Accuracy(task="multiclass", num_classes=10).to(device)
    precision = Precision(task="multiclass", average="macro", num_classes=10).to(device)
    recall = Recall(task="multiclass", average="macro", num_classes=10).to(device)

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
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            accuracy.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy.compute().item()
        epoch_prec = precision.compute().item()
        epoch_recall = recall.compute().item()

        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, Prec={epoch_prec:.4f}, Recall={epoch_recall:.4f}")

        # Reset metrics for next epoch
        accuracy.reset()
        precision.reset()
        recall.reset()
