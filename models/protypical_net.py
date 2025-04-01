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

from utils.dataset_loader import AddGaussianNoise


class ProtoNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ProtoNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 8 * 8, embedding_dim)  

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  
        x = self.pool(self.relu(self.conv2(x))) 
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        return x  


def load_cinic10_few_shot(data_dir, num_samples_per_class=5, batch_size=10, augmentations=None):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset_test = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform_test)

    transform_list = [transforms.ToTensor()]
    if augmentations:
        if "rotation" in augmentations:
            transform_list.append(transforms.RandomRotation(degrees=15))
        if "translation" in augmentations:
            transform_list.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
        if "noise" in augmentations:
            transform_list.append(AddGaussianNoise(mean=0, std=0.05))
        if "erasing" in augmentations:
            transform_list.append(
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False))
        if "style" in augmentations:
            transform_list.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))

    transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    transform_train = transforms.Compose(transform_list)
    dataset_train = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform_train)


    class_indices = {label: [] for label in range(10)}
    for idx, (_, label) in enumerate(dataset_train.samples):
        class_indices[label].append(idx)

    selected_indices = []
    for label, indices in class_indices.items():
        selected_indices.extend(random.sample(indices, num_samples_per_class))

    few_shot_dataset_train = Subset(dataset_train, selected_indices)
    dataloader_train = DataLoader(few_shot_dataset_train, batch_size, shuffle=True)

    class_indices = {label: [] for label in range(10)}
    for idx, (_, label) in enumerate(dataset_test.samples):
        class_indices[label].append(idx)

    selected_indices = []
    for label, indices in class_indices.items():
        selected_indices.extend(random.sample(indices, num_samples_per_class))

    few_shot_dataset_test = Subset(dataset_test, selected_indices)
    dataloader_test = DataLoader(few_shot_dataset_test, batch_size, shuffle=True)

    return dataloader_train, dataloader_test


def prototypical_loss(embeddings, labels, num_classes):
    prototypes = torch.zeros(num_classes, embeddings.size(1)).to(embeddings.device)
    for i in range(num_classes):
        class_mask = (labels == i)
        if class_mask.sum() > 0:
            prototypes[i] = embeddings[class_mask].mean(dim=0)

    distances = torch.cdist(embeddings, prototypes)
    return F.cross_entropy(-distances, labels)


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

        #print training loss and accuracy
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
    print("Training complete!")


# Example usage
# data_dir = "../data/train"
# few_shot_dataloader = load_cinic10_few_shot(data_dir)
# model = ProtoNet()
# train_prototypical_network(model, few_shot_dataloader, epochs= 30)

