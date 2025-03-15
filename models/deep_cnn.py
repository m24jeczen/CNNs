import torch
import torch.nn as nn

class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()

        # Convolutional Layers (5 Conv Layers)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # Output: (64, 32, 32)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: (128, 32, 32)
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: (128, 16, 16)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Output: (256, 16, 16)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # Output: (512, 16, 16)
        self.pool2 = nn.MaxPool2d(2, 2)  # Output: (512, 8, 8)
        
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  # Output: (512, 8, 8)
        self.pool3 = nn.MaxPool2d(2, 2)  # Output: (512, 4, 4)

        # Fully Connected Layers
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)  # 1024 neurons
        self.dropout1 = nn.Dropout(0.5)  # Dropout to reduce overfitting
        self.fc2 = nn.Linear(1024, 512)  # 512 neurons
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 10)  # Output Layer (10 classes)

        # Activation Function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Convolutional Layers + Activation + Pooling
        x = self.relu(self.conv1(x))
        x = self.pool1(self.relu(self.conv2(x)))

        x = self.relu(self.conv3(x))
        x = self.pool2(self.relu(self.conv4(x)))

        x = self.pool3(self.relu(self.conv5(x)))

        # Flatten for Fully Connected Layers
        x = torch.flatten(x, 1)  

        # Fully Connected Layers + Dropout
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # No activation for final logits

        return x
