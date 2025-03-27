import torch
import torch.nn as nn

class DeepCNN(nn.Module):
    def __init__(self, p_dropout=0.3):
        super(DeepCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2) 
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1) 
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  
        self.pool2 = nn.MaxPool2d(2, 2) 
        
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1) 
        self.pool3 = nn.MaxPool2d(2, 2) 

        self.fc1 = nn.Linear(512 * 4 * 4, 1024) 
        self.dropout1 = nn.Dropout(p=p_dropout)  
        self.fc2 = nn.Linear(1024, 512)  
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(512, 10)  

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.pool2(self.relu(self.conv4(x)))
        x = self.pool3(self.relu(self.conv5(x)))

        x = torch.flatten(x, 1)  

        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  

        return x
