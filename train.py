import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import LightweightNet

def train_model(epochs=1, batch_size=128, learning_rate=0.01):
    # Data augmentation and normalization
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightweightNet().to(device)
    
    # Print model parameters
    print(f"Total parameters: {model.count_parameters():,}")
    
    # One cycle learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader)
    )
    
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                      f'Loss: {running_loss/(batch_idx+1):.6f}\t'
                      f'Accuracy: {100.*correct/total:.2f}%')
                
    print(f'Final Accuracy: {100.*correct/total:.2f}%')
    return model

if __name__ == '__main__':
    model = train_model() 