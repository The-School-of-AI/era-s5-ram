import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # Input: 28x28x1
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 28x28x8
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # 28x28x16
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, padding=1) # 28x28x10
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1) 