import torch
import torch.nn as nn
from torchmetrics import Accuracy, Precision, Recall

def evaluate_model(model, test_loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    accuracy = Accuracy(task="multiclass", num_classes=10).to(device)
    precision = Precision(task="multiclass", average="macro", num_classes=10).to(device)
    recall = Recall(task="multiclass", average="macro", num_classes=10).to(device)
    
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            accuracy.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
    
    avg_loss = total_loss / len(test_loader)
    test_acc = accuracy.compute().item()
    test_prec = precision.compute().item()
    test_recall = recall.compute().item()
    
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_recall:.4f}")
    
    return avg_loss, test_acc, test_prec, test_recall