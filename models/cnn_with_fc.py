import torch
import torch.nn as nn

class CNNWithFC(nn.Module):
    def __init__(self,dropout_p=0.3):
        super(CNNWithFC, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)  # 128 neurons
        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(self.conv1(x))  # Convolution -> MaxPool
        x = torch.flatten(x, 1)  # Flatten before FC
        x = self.relu(self.fc1(x))  # FC -> Activation
        x = self.dropout(x)  # Dropout
        x = self.fc2(x)  # Output layer
        return x
