# 🔥 MNIST Lightweight Model

[![Python application](https://github.com/The-School-of-AI/era-s5-ram/workflows/Python%20application/badge.svg)](https://github.com/The-School-of-AI/era-s5-ram/actions)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An efficient MNIST classifier that achieves remarkable performance with minimal parameters. Built with PyTorch and optimized for both accuracy and model size.

## 🌟 Key Features

- **Efficient Architecture**: < 25,000 parameters
- **Fast Training**: 95%+ accuracy in just 1 epoch
- **Modern Design**: Uses GAP and extensive BatchNorm
- **Robust**: Includes dropout and data augmentation
- **Well-Tested**: Comprehensive test suite with CI/CD

## 🏗️ Model Architecture

```python
MNISTNet(
  # Input Block
  (convblock1): Sequential(Conv2d(1, 32, k=3, p=1), BN, ReLU, Dropout)  # 28x28x32, RF=3
  
  # Convolution Block 1
  (convblock2): Sequential(Conv2d(32, 32, k=3, p=1), BN, ReLU, Dropout)  # 28x28x32, RF=5
  (convblock3): Sequential(Conv2d(32, 32, k=3, p=1), BN, ReLU, Dropout)  # 28x28x32, RF=7
  
  # Transition Block 1
  (pool1): MaxPool2d(2, 2)                                               # 14x14x32, RF=8
  (convblock4): Sequential(Conv2d(32, 16, k=1), BN, ReLU, Dropout)      # 14x14x16, RF=8
  
  # Convolution Block 2
  (convblock5): Sequential(Conv2d(16, 32, k=3, p=1), BN, ReLU, Dropout)  # 14x14x32, RF=12
  (convblock6): Sequential(Conv2d(32, 32, k=3, p=1), BN, ReLU, Dropout)  # 14x14x32, RF=16
  
  # Transition Block 2
  (pool2): MaxPool2d(2, 2)                                               # 7x7x32, RF=18
  (convblock7): Sequential(Conv2d(32, 32, k=1), BN, ReLU, Dropout)      # 7x7x32, RF=18
  
  # Convolution Block 3
  (convblock8): Sequential(Conv2d(32, 64, k=3, p=1), BN, ReLU, Dropout)  # 7x7x64, RF=22
  (convblock9): Sequential(Conv2d(64, 64, k=3, p=1), BN, ReLU, Dropout)  # 7x7x64, RF=26
  
  # Output Block
  (convblock10): Sequential(Conv2d(64, 32, k=3, p=1), BN, ReLU, Dropout) # 7x7x32, RF=30
  (convblock11): Conv2d(32, 10, k=1)                                      # 7x7x10, RF=30
  (gap): AvgPool2d(7)                                                     # 1x1x10, RF=44
)
```

Total Parameters: ~24,500

## 🔍 Architecture Highlights

1. **Progressive Channel Growth**: 1 → 32 → 16 → 32 → 64 → 32 → 10 channels
2. **Receptive Field**: Carefully designed to reach RF=44
3. **Multi-scale Processing**:
   - Initial processing at 28x28
   - Mid-level features at 14x14
   - Final features at 7x7
4. **Regularization**: 
   - BatchNorm after every conv layer
   - 5% dropout throughout
   - Two MaxPooling layers
5. **Efficiency Features**:
   - 1x1 convolutions for channel manipulation
   - Global Average Pooling for final feature aggregation
   - Bias=False in all conv layers
   - Padding=1 for maintaining spatial dimensions

## 📊 Training Configuration

Our model uses an optimized training setup:
- 🔄 SGD Optimizer with Nesterov momentum (0.9)
- 📈 OneCycleLR Scheduler:
  - max_lr: 0.15
  - div_factor: 10.0
  - final_div_factor: 100.0
  - pct_start: 0.2
  - anneal_strategy: 'cos'
- 📦 Batch Size: 128
- 🎯 Single Epoch Training
- 🔧 Weight Decay: 5e-5
- 📈 Gradient Clipping: 0.1
- 💧 Dropout: 5%

## 📊 Data Augmentation

Carefully tuned augmentation for optimal performance:

- 🔄 Random rotation (±2°)
- ↔️ Random translation (±2%)
- 📊 Normalization (μ=0.1307, σ=0.3081)

### Augmentation Examples
Below are examples showing original images (top) and their augmented versions (bottom):

![Augmented Samples](images/augmented_samples.png)

## 🧪 Testing Suite

Our comprehensive testing ensures model reliability:

| Test | Description |
|------|-------------|
| ✓ Parameter Count | Verifies model stays under 15K parameters |
| ✓ Output Shape | Ensures correct tensor dimensions |
| ✓ Forward Pass | Validates stable forward propagation |
| ✓ Probability | Checks proper probability distribution |
| ✓ Augmentation | Confirms correct image transformations |
| ✓ Learning | Verifies model's ability to learn |

## 🚀 Quick Start

```python
# Install dependencies
pip install -r requirements.txt

# Train the model
from train import train_model
model = train_model()

# Make predictions
model.eval()
with torch.no_grad():
    predictions = model(images)
```

## 📈 Performance

- Training Accuracy: > 95% (1 epoch)
- Parameters: ~13,000
- Training Time: < 5 minutes (CPU)
- Optimized for both CPU and GPU training
- Stable training with BatchNorm and Dropout

## 🛠️ Development

```bash
# Clone the repository
git clone https://github.com/The-School-of-AI/era-s5-ram.git

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest test_model.py -v
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or feedback, please open an issue in the GitHub repository.

---
Made with ❤️ by Ram