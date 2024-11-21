# MNIST Lightweight Model

![Build Status](https://github.com/The-School-of-AI/era-s5-ram/workflows/Python%20application/badge.svg)

A lightweight MNIST classifier with the following characteristics:
- Less than 15,000 parameters
- Uses Global Average Pooling
- Achieves 95%+ training accuracy in 1 epoch
- Includes image augmentation
- Comprehensive test suite

## Model Architecture
- 3 Convolutional layers
- Global Average Pooling
- Total parameters: ~14,000

## Features
- Data augmentation (rotation, translation)
- Automated testing
- CI/CD integration

## Data Augmentation Examples
Below are examples of original images (top row) and their augmented versions (bottom row):

![Augmented Samples](images/augmented_samples.png)

The augmentation pipeline includes:
- Random rotation (±10 degrees)
- Random translation (±10% in both directions)
- Normalization

## Tests
The model includes various tests:
- Parameter count verification
- Output shape validation
- Forward pass stability
- Probability distribution checks
- Augmentation verification
- Learning capability verification

## Usage