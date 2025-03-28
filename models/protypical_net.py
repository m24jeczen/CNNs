import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import random
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import random
import torch.nn.functional as F


# Define a simple CNN model for prototypical learning
class ProtoNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ProtoNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 8 * 8, embedding_dim)  # Adjusted to correct size

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # Output: (batch, 32, 16, 16)
        x = self.pool(self.relu(self.conv2(x)))  # Output: (batch, 64, 8, 8)
        x = x.view(x.size(0), -1)  # Flatten to (batch, 64*8*8)
        x = self.fc1(x)
        return x  # Return embeddings


# Load and preprocess the CINIC-10 dataset
def load_cinic10_few_shot(data_dir, num_samples_per_class=5):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Select few-shot samples per class
    class_indices = {label: [] for label in range(10)}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    selected_indices = []
    for label, indices in class_indices.items():
        selected_indices.extend(random.sample(indices, num_samples_per_class))

    few_shot_dataset = Subset(dataset, selected_indices)
    dataloader = DataLoader(few_shot_dataset, batch_size=10, shuffle=True)
    return dataloader


# Compute prototype loss (Euclidean distance)
def prototypical_loss(embeddings, labels, num_classes):
    prototypes = torch.zeros(num_classes, embeddings.size(1)).to(embeddings.device)
    for i in range(num_classes):
        class_mask = (labels == i)
        if class_mask.sum() > 0:
            prototypes[i] = embeddings[class_mask].mean(dim=0)

    distances = torch.cdist(embeddings, prototypes)
    return F.cross_entropy(-distances, labels)


# Training function for prototypical learning
def train_prototypical_network(model, dataloader, epochs=5, lr=0.001, num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings = model(images)
            loss = prototypical_loss(embeddings, labels, num_classes)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}, Ac")

    print("Training complete!")


# Example usage
data_dir = "../data/train"
few_shot_dataloader = load_cinic10_few_shot(data_dir)
model = ProtoNet()
train_prototypical_network(model, few_shot_dataloader)
