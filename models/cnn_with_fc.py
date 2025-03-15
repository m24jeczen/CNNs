import torch
import torch.nn as nn

class CNNWithFC(nn.Module):
    def __init__(self):
        super(CNNWithFC, self).__init__()
        
        # Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        
        # Fully Connected Layer
        self.fc1 = nn.Linear(32 * 16 * 16, 128)  # 128 neurons
        
        # Dropout Layer (to prevent overfitting)
        self.dropout = nn.Dropout(p=0.5)
        
        # Activation Function
        self.relu = nn.ReLU()
        
        # Output Layer (10 classes for CINIC-10)
        self.fc2 = nn.Linear(128, 10)

        # Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(self.conv1(x))  # Convolution -> MaxPool
        x = torch.flatten(x, 1)  # Flatten before FC
        x = self.relu(self.fc1(x))  # FC -> Activation
        x = self.dropout(x)  # Dropout
        x = self.fc2(x)  # Output layer
        return x
