import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LightweightNet, self).__init__()
        # Efficient feature extraction
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 8 filters
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # 16 filters
        
        # Spatial attention module
        self.attention = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Lightweight classifier
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Feature extraction
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        
        # Apply attention
        att = self.attention(x)
        x = x * att
        
        # Classification
        x = x.view(-1, 16 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)