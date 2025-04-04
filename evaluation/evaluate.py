import torch
import torch.nn as nn
from torchmetrics import Accuracy, Precision, Recall
import os

from models.protypical_net import ProtoNet
from training.train import prototypical_loss,pairwise_distances,compute_prototypes


def save_evaluation_results(model_name, num_epochs, augmentations_str, batch_size, l2_reg, lr, max_lr, scheduler_type, test_metrics):

    model_info = f"{model_name}_{num_epochs}_{augmentations_str}_{batch_size}_{l2_reg}_{lr}_{max_lr}_{scheduler_type}"
    
    with open("experiments/test_results.txt", mode='a') as file:
        file.write(f"{model_info}: {test_metrics}\n")
    
    print("Evaluation results appended to test_results.txt")


def evaluate_model(model, test_loader, num_epochs, l2_reg=0, lr=0.0001, max_lr=0.001, scheduler_type="none", device=None, augmentations=None):
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
            images, labels = images.to(device), labels.to(device)

            if isinstance(model, ProtoNet):
                embeddings = model(images)
                loss = prototypical_loss(embeddings, labels, num_classes=10)
                prototypes = compute_prototypes(embeddings, labels, num_classes=10)
                dists = pairwise_distances(embeddings, prototypes)
                preds = torch.argmin(dists, dim=1)

            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)

            total_loss += loss.item()
            accuracy.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)

    avg_loss = total_loss / len(test_loader)
    test_acc = accuracy.compute().item()
    test_prec = precision.compute().item()
    test_recall = recall.compute().item()

    test_metrics = {"loss": avg_loss, "accuracy": test_acc, "precision": test_prec, "recall": test_recall}
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_recall:.4f}")

    augmentations_str = ("_".join(augmentations)) if augmentations else "none"

    save_evaluation_results(model.__class__.__name__, num_epochs, augmentations_str, test_loader.batch_size, l2_reg, lr, max_lr, scheduler_type, test_metrics)


#
# def evaluate_model(model, test_loader, num_epochs, l2_reg=0, min_lr=0.0001, max_lr=0.001, scheduler_type="none", device=None, augmentations=None):
#
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     model.to(device)
#     model.eval()
#
#     criterion = nn.CrossEntropyLoss()
#     accuracy = Accuracy(task="multiclass", num_classes=10).to(device)
#     precision = Precision(task="multiclass", average="macro", num_classes=10).to(device)
#     recall = Recall(task="multiclass", average="macro", num_classes=10).to(device)
#
#     total_loss = 0.0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#
#             preds = torch.argmax(outputs, dim=1)
#             accuracy.update(preds, labels)
#             precision.update(preds, labels)
#             recall.update(preds, labels)
#
#     avg_loss = total_loss / len(test_loader)
#     test_acc = accuracy.compute().item()
#     test_prec = precision.compute().item()
#     test_recall = recall.compute().item()
#
#     test_metrics = {"loss": avg_loss, "accuracy": test_acc, "precision": test_prec, "recall": test_recall}
#     print(f"Test Loss: {avg_loss:.4f}, Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_recall:.4f}")
#
#     augmentations_str = ("_".join(augmentations)) if augmentations else "none"
#
#     save_evaluation_results(model.__class__.__name__, num_epochs, augmentations_str, test_loader.batch_size, l2_reg, min_lr, max_lr, scheduler_type, test_metrics)
#
#     #return avg_loss, test_acc, test_prec, test_recall