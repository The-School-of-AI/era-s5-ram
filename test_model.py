import torch
import pytest
from model import MNISTNet
import torch.nn.functional as F

def test_model_parameters():
    model = MNISTNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"

def test_model_output_shape():
    model = MNISTNet()
    batch_size = 128
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)
    assert output.shape == (batch_size, 10), f"Expected output shape {(batch_size, 10)}, got {output.shape}"

def test_model_forward_pass():
    model = MNISTNet()
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    assert not torch.isnan(output).any(), "Model output contains NaN values"

def test_model_probability_sum():
    model = MNISTNet()
    x = torch.randn(1, 1, 28, 28)
    output = torch.exp(model(x))
    prob_sum = output.sum().item()
    assert abs(prob_sum - 1.0) < 1e-6, f"Probability sum should be 1, got {prob_sum}"

def test_model_augmentation():
    from train import get_transforms
    transforms = get_transforms()
    x = torch.ones(1, 28, 28)
    augmented = transforms(x)
    assert augmented.shape == (1, 28, 28), "Augmentation should preserve image dimensions"

def test_model_learning():
    model = MNISTNet()
    optimizer = torch.optim.Adam(model.parameters())
    x = torch.randn(32, 1, 28, 28)
    target = torch.randint(0, 10, (32,))
    
    # Initial loss
    initial_output = model(x)
    initial_loss = F.nll_loss(initial_output, target)
    
    # One optimization step
    optimizer.zero_grad()
    loss = F.nll_loss(initial_output, target)
    loss.backward()
    optimizer.step()
    
    # New loss
    new_output = model(x)
    new_loss = F.nll_loss(new_output, target)
    
    assert new_loss < initial_loss, "Model should learn and reduce loss" 