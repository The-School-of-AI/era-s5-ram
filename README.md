# 🔥 MNIST Lightweight Model

[![Python application](https://github.com/The-School-of-AI/era-s5-ram/workflows/Python%20application/badge.svg)](https://github.com/The-School-of-AI/era-s5-ram/actions)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A highly efficient MNIST classifier that achieves remarkable performance with minimal parameters. Built with PyTorch and optimized for both accuracy and model size.

## 🌟 Key Features

- **Lightweight Architecture**: < 15,000 parameters
- **Fast Training**: 95%+ accuracy in just 1 epoch
- **Modern Design**: Uses Global Average Pooling
- **Robust**: Includes data augmentation
- **Well-Tested**: Comprehensive test suite with CI/CD

## 🏗️ Model Architecture

```python
MNISTNet(
  (conv1): Conv2d(1, 8, kernel_size=3, padding=1)    # 28x28x8
  (conv2): Conv2d(8, 16, kernel_size=3, padding=1)   # 28x28x16
  (conv3): Conv2d(16, 10, kernel_size=3, padding=1)  # 28x28x10
  (gap): AdaptiveAvgPool2d(1)                        # Global Average Pooling
)
```

Total Parameters: ~14,000

## 📊 Training Configuration

Our model uses an optimized training setup:
- 🔄 SGD Optimizer with momentum (0.9)
- 📈 OneCycleLR Scheduler (max_lr=0.1)
- 📦 Batch Size: 32
- 🎯 Single Epoch Training

## 📊 Data Augmentation

Carefully tuned augmentation for optimal performance:

- 🔄 Random rotation (±5°)
- ↔️ Random translation (±5%)
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
- Parameters: ~14,000
- Training Time: < 5 minutes (CPU)
- Optimized for both CPU and GPU training

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