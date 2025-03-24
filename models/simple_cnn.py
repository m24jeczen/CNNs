import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, dropout_p=0.5): 
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=dropout_p)  
        self.fc1 = nn.Linear(32 * 16 * 16, 10)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(x)  
        x = self.fc1(x)
        return x