import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Projection shortcut if channels change
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = F.relu(out)
        
        return out

class MiniResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MiniResNet, self).__init__()
        
        # Initial conv layer with small number of filters
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        
        # Residual blocks with minimal channels
        self.layer1 = BasicBlock(8, 8)
        self.layer2 = BasicBlock(8, 16)
        
        # Global average pooling instead of large FC layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final classifier
        self.fc = nn.Linear(16, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.layer1(x)
        x = F.max_pool2d(x, 2)
        x = self.layer2(x)
        x = F.max_pool2d(x, 2)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)