import torch
import torchvision
import torchvision.transforms as transforms
from model import MNISTNet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

def save_augmented_samples(dataset, num_samples=5):
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    for i in range(num_samples):
        # Get original image
        original_dataset = torchvision.datasets.MNIST('./data', train=True, download=True,
                                                    transform=transforms.ToTensor())
        orig_img, _ = original_dataset[i]
        
        # Get augmented image
        aug_img, _ = dataset[i]
        
        # Plot original
        axes[0, i].imshow(orig_img.squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')
        
        # Plot augmented
        axes[1, i].imshow(aug_img.squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Augmented')
    
    plt.tight_layout()
    plt.savefig('images/augmented_samples.png')
    plt.close()

def get_transforms():
    return transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTNet().to(device)
    
    # Data loading with augmentation
    dataset = torchvision.datasets.MNIST('./data', train=True, download=True,
                                     transform=get_transforms())
    
    # Save augmented samples before training
    save_augmented_samples(dataset)
    
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters())
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Loss: {loss.item():.6f}')
    
    return model 