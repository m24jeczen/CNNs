import torch
from torchmetrics import Accuracy, Precision, Recall
from torch.nn import CrossEntropyLoss
import os

def hard_voting_ensemble(
    models, 
    test_loader, 
    num_epochs=0, 
    l2_reg=0.0, 
    min_lr=0.001, 
    max_lr=0.003, 
    scheduler_type="none", 
    device=None, 
    augmentations=None, 
    save_dir="experiments"
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model in models:
        model.to(device)
        model.eval()

    criterion = CrossEntropyLoss()
    accuracy = Accuracy(task="multiclass", num_classes=10).to(device)
    precision = Precision(task="multiclass", average="macro", num_classes=10).to(device)
    recall = Recall(task="multiclass", average="macro", num_classes=10).to(device)

    total_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Collect predictions from all models
            batch_preds = []
            batch_losses = []

            for model in models:
                outputs = model(images)
                loss = criterion(outputs, labels)
                batch_losses.append(loss.item())

                preds = torch.argmax(outputs, dim=1)
                batch_preds.append(preds.cpu())

            # Stack predictions and vote (mode)
            stacked_preds = torch.stack(batch_preds)  # Shape: (num_models, batch_size)
            voted_preds = torch.mode(stacked_preds, dim=0).values.to(device)

            total_loss += sum(batch_losses) / len(models)

            # Update metrics
            accuracy.update(voted_preds, labels)
            precision.update(voted_preds, labels)
            recall.update(voted_preds, labels)

    # Final metrics
    avg_loss = total_loss / len(test_loader)
    test_acc = accuracy.compute().item()
    test_prec = precision.compute().item()
    test_recall = recall.compute().item()

    # Output
    print(f"\n[Hard Voting Ensemble]")
    print(f"Loss: {avg_loss:.4f}, Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_recall:.4f}")

    # Save to file
    if augmentations:
        augmentations_str = "_".join(augmentations)
    else:
        augmentations_str = "none"

    model_info = f"HardVotingEnsemble_{num_epochs}_{augmentations_str}_{test_loader.batch_size}_{l2_reg}_{min_lr}_{max_lr}_{scheduler_type}"
    
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "test_results.txt"), mode='a') as file:
        file.write(f"{model_info}: "
                   f"loss={avg_loss:.4f}, acc={test_acc:.4f}, prec={test_prec:.4f}, recall={test_recall:.4f}\n")

    return avg_loss, test_acc, test_prec, test_recall
