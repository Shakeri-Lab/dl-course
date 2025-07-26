#!/usr/bin/env python3
"""
Softmax Regression: PyTorch High-Level Implementation

This script demonstrates the same softmax regression using PyTorch's
built-in tools, showing how the framework automates the manual operations
from our scratch implementation.

Key PyTorch Components:
1. nn.Module for model definition
2. nn.CrossEntropyLoss (combines softmax + cross-entropy)
3. Automatic differentiation
4. Built-in optimizers
5. Data loading utilities

Author: PyTorch Softmax Regression Version
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# =====================================================================
# PART 1: Data Loading with PyTorch Utilities
# =====================================================================

def load_fashion_mnist(batch_size=256):
    """Load FashionMNIST using PyTorch's built-in utilities."""
    
    # Transform: Convert PIL to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    print(f"=== Dataset Information ===")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Class names: {train_dataset.classes}")
    print()
    
    return train_loader, test_loader


# =====================================================================
# PART 2: Model Definition using nn.Module
# =====================================================================

class SoftmaxRegressionPyTorch(nn.Module):
    """
    Softmax regression using PyTorch's nn.Module.
    
    Key differences from scratch implementation:
    - Inherits from nn.Module for automatic features
    - Uses nn.Flatten and nn.Linear layers
    - Parameters are registered automatically
    - No manual softmax (handled by loss function)
    """
    
    def __init__(self, input_size, num_classes):
        """
        Initialize the model.
        
        Args:
            input_size: Number of input features (784 for 28x28 images)
            num_classes: Number of output classes (10 for FashionMNIST)
        """
        super(SoftmaxRegressionPyTorch, self).__init__()
        
        # Define the network architecture
        self.flatten = nn.Flatten()  # Flattens input to 1D
        self.linear = nn.Linear(input_size, num_classes)
        
        print(f"=== PyTorch Model Architecture ===")
        print(f"Input size: {input_size}")
        print(f"Number of classes: {num_classes}")
        print(f"Model: Flatten -> Linear({input_size}, {num_classes})")
        
        # Print model parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params}")
        print()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Raw logits (not probabilities) - softmax applied in loss function
        """
        x = self.flatten(x)  # Flatten to (batch_size, input_size)
        logits = self.linear(x)  # Linear transformation
        return logits  # Return raw logits (no softmax here)
    
    def get_probabilities(self, x):
        """Get actual probabilities (for visualization/analysis)."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


# =====================================================================
# PART 3: Training with PyTorch Tools
# =====================================================================

def train_pytorch_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.1):
    """
    Train the model using PyTorch's optimization tools.
    
    Key components:
    - nn.CrossEntropyLoss: Combines softmax + cross-entropy efficiently
    - optim.SGD: Stochastic gradient descent optimizer
    - Automatic gradient computation with loss.backward()
    """
    print(f"=== Training PyTorch Model ===")
    
    # Define loss function and optimizer
    # CrossEntropyLoss expects raw logits (applies softmax internally)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()  # Set model to training mode
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()  # Compute gradients automatically
            optimizer.step()  # Update parameters
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_predictions / total_samples
        
        # Evaluation phase
        test_acc = evaluate_pytorch_model(model, test_loader, criterion)
        
        # Record history
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['test_acc'].append(test_acc)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}: "
              f"Train Loss = {epoch_loss:.4f}, "
              f"Train Acc = {epoch_acc:.4f}, "
              f"Test Acc = {test_acc:.4f}")
    
    return history


def evaluate_pytorch_model(model, test_loader, criterion):
    """Evaluate model on test set."""
    model.eval()  # Set model to evaluation mode
    
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():  # Disable gradient computation for efficiency
        for data, targets in test_loader:
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
    
    return correct_predictions / total_samples


# =====================================================================
# PART 4: Advanced PyTorch Features
# =====================================================================

def demonstrate_pytorch_features(model, test_loader):
    """Showcase additional PyTorch capabilities."""
    print("\n=== Advanced PyTorch Features ===")
    
    # 1. Model information
    print("\n1. Model Architecture:")
    print(model)
    
    # 2. Parameter inspection
    print("\n2. Model Parameters:")
    for name, param in model.named_parameters():
        print(f"   {name}: {param.shape}, requires_grad={param.requires_grad}")
    
    # 3. Gradient inspection
    print("\n3. Gradient Inspection:")
    # Get a small batch
    data_iter = iter(test_loader)
    sample_data, sample_targets = next(data_iter)
    sample_data, sample_targets = sample_data[:5], sample_targets[:5]
    
    # Forward pass and compute loss
    model.train()
    outputs = model(sample_data)
    loss = nn.CrossEntropyLoss()(outputs, sample_targets)
    loss.backward()
    
    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"   {name} gradient norm: {param.grad.norm().item():.6f}")
    
    # 4. Different optimizers comparison
    print("\n4. Available Optimizers in PyTorch:")
    print("   - SGD: Basic stochastic gradient descent")
    print("   - Adam: Adaptive moment estimation")
    print("   - RMSprop: Root mean square propagation")
    print("   - AdaGrad: Adaptive gradient algorithm")
    print("   - LBFGS: Limited-memory BFGS")


# =====================================================================
# PART 5: Visualization and Comparison
# =====================================================================

def plot_pytorch_results(history, model, test_loader):
    """Create visualizations for PyTorch training results."""
    
    # Training curves
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Training Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Training Loss (PyTorch)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Cross-Entropy Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy Comparison
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training', linewidth=2)
    axes[0, 1].plot(epochs, history['test_acc'], 'r-', label='Test', linewidth=2)
    axes[0, 1].set_title('Accuracy (PyTorch)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Weight Visualization
    weights = model.linear.weight.detach().numpy()  # Shape: (10, 784)
    weight_images = weights.reshape(10, 28, 28)
    
    # Show first 6 weight vectors as images
    for i in range(6):
        row, col = (i // 3) + 1, i % 3 if i < 3 else (i % 3)
        if i < 3:
            axes[1, i].imshow(weight_images[i], cmap='coolwarm')
            axes[1, i].set_title(f'Class {i} Weights')
            axes[1, i].axis('off')
    
    # Remove empty subplot
    if len(axes[1]) > 3:
        fig.delaxes(axes[1, 3] if len(axes[1]) > 3 else None)
    
    plt.tight_layout()
    plt.savefig('module-2/pytorch_softmax_results.png', dpi=150)
    print("\nPyTorch results saved to 'pytorch_softmax_results.png'")


def compare_predictions(model, test_loader):
    """Compare model predictions with ground truth."""
    labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Get test samples
    data_iter = iter(test_loader)
    images, true_labels = next(data_iter)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        probabilities = model.get_probabilities(images[:12])
        predicted_labels = probabilities.argmax(dim=1)
        confidences = probabilities.max(dim=1)[0]
    
    # Visualize predictions
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for i in range(12):
        row, col = i // 4, i % 4
        
        # Show image
        axes[row, col].imshow(images[i].squeeze(), cmap='gray')
        
        # Add prediction info
        true_label = labels[true_labels[i]]
        pred_label = labels[predicted_labels[i]]
        confidence = confidences[i].item()
        
        color = 'green' if true_labels[i] == predicted_labels[i] else 'red'
        title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}'
        
        axes[row, col].set_title(title, color=color, fontsize=8)
        axes[row, col].axis('off')
    
    plt.suptitle('PyTorch Model Predictions', fontsize=14)
    plt.tight_layout()
    plt.savefig('module-2/pytorch_predictions.png', dpi=150)
    print("PyTorch predictions saved to 'pytorch_predictions.png'")


# =====================================================================
# PART 6: Main Execution
# =====================================================================

def main():
    """Run the complete PyTorch softmax regression demonstration."""
    
    print("="*60)
    print("SOFTMAX REGRESSION WITH PYTORCH")
    print("="*60)
    
    # Load data
    train_loader, test_loader = load_fashion_mnist(batch_size=256)
    
    # Create model
    model = SoftmaxRegressionPyTorch(
        input_size=28*28,  # 784 features
        num_classes=10     # 10 FashionMNIST classes
    )
    
    # Train model
    history = train_pytorch_model(
        model, train_loader, test_loader,
        num_epochs=10, learning_rate=0.1
    )
    
    # Demonstrate advanced features
    demonstrate_pytorch_features(model, test_loader)
    
    # Visualize results
    plot_pytorch_results(history, model, test_loader)
    compare_predictions(model, test_loader)
    
    # Final evaluation
    final_test_acc = evaluate_pytorch_model(model, test_loader, nn.CrossEntropyLoss())
    print(f"\n=== Final Results ===")
    print(f"Final test accuracy: {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")
    
    print("\n" + "="*60)
    print("KEY DIFFERENCES FROM SCRATCH IMPLEMENTATION:")
    print("1. nn.Module handles parameter registration automatically")
    print("2. nn.CrossEntropyLoss combines softmax + cross-entropy efficiently")
    print("3. Automatic differentiation with loss.backward()")
    print("4. Built-in optimizers handle parameter updates")
    print("5. No manual gradient computation required")
    print("6. Model can be easily extended with more layers")
    print("7. Better numerical stability and performance")
    print("="*60)


# =====================================================================
# PART 7: Comparison Summary
# =====================================================================

"""
Scratch vs PyTorch Implementation Comparison:

Scratch Implementation:
+ Educational: Shows all mathematical details
+ Full control over every operation
+ Understanding of underlying mechanics
- Manual gradient computation (error-prone)
- More code to write and maintain
- Potential numerical instability

PyTorch Implementation:
+ Automatic differentiation
+ Better numerical stability
+ Less code, more readable
+ Easy to extend and modify
+ Production-ready
+ GPU acceleration available
- Less visibility into internal operations
- Requires understanding of PyTorch concepts

When to use each:
- Scratch: Learning fundamentals, research on new algorithms
- PyTorch: Production code, rapid prototyping, complex models
"""

if __name__ == "__main__":
    main() 