#!/usr/bin/env python3
"""
Softmax Regression from Scratch

This script implements softmax regression (multinomial logistic regression)
from scratch, demonstrating the fundamental concepts of multiclass classification.

Key Concepts:
1. Softmax function for probability distribution
2. Cross-entropy loss for classification
3. One-hot encoding and label handling
4. Multiclass gradient computation
5. Classification accuracy metrics

Author: Softmax Regression Teaching Version
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# =====================================================================
# PART 1: Understanding the Softmax Function
# =====================================================================

def softmax(X):
    """
    Compute softmax values for each row of the input tensor X.
    
    The softmax function converts raw scores (logits) into probabilities:
    softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
    
    Args:
        X: Input tensor of shape (batch_size, num_classes)
        
    Returns:
        Tensor of same shape with probabilities (each row sums to 1)
    """
    # Subtract max for numerical stability (prevents overflow)
    X_exp = torch.exp(X - X.max(dim=1, keepdim=True).values)
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


def demonstrate_softmax():
    """Demonstrate how softmax works with examples."""
    print("=== Softmax Function Demonstration ===")
    
    # Example 1: Two samples, three classes
    logits = torch.tensor([[2.0, 1.0, 0.1],
                          [1.0, 3.0, 0.2]])
    
    probabilities = softmax(logits)
    
    print(f"Input logits:\n{logits}")
    print(f"Softmax probabilities:\n{probabilities}")
    print(f"Row sums (should be 1.0): {probabilities.sum(dim=1)}")
    
    # Show that higher logits get higher probabilities
    print(f"\nClass with highest logit in row 0: {logits[0].argmax().item()}")
    print(f"Class with highest probability in row 0: {probabilities[0].argmax().item()}")
    print()


# =====================================================================
# PART 2: Cross-Entropy Loss Function
# =====================================================================

def cross_entropy_loss(y_hat, y):
    """
    Compute cross-entropy loss for multiclass classification.
    
    Cross-entropy measures the distance between predicted probabilities
    and true labels. For true class i: loss = -log(predicted_prob_i)
    
    Args:
        y_hat: Predicted probabilities, shape (batch_size, num_classes)
        y: True labels, shape (batch_size,) with integer class indices
        
    Returns:
        Scalar loss value (averaged over batch)
    """
    # Select the predicted probability for the true class
    # y_hat[range(len(y)), y] gets y_hat[0, y[0]], y_hat[1, y[1]], etc.
    return -torch.log(y_hat[range(len(y_hat)), y]).mean()


def demonstrate_cross_entropy():
    """Demonstrate cross-entropy loss calculation."""
    print("=== Cross-Entropy Loss Demonstration ===")
    
    # Example: 3 samples, 4 classes
    y_hat = torch.tensor([[0.7, 0.1, 0.1, 0.1],  # Confident in class 0
                          [0.1, 0.1, 0.7, 0.1],  # Confident in class 2
                          [0.25, 0.25, 0.25, 0.25]])  # Uncertain
    
    y_true = torch.tensor([0, 2, 1])  # True classes
    
    loss = cross_entropy_loss(y_hat, y_true)
    
    print(f"Predicted probabilities:\n{y_hat}")
    print(f"True labels: {y_true}")
    print(f"Cross-entropy loss: {loss:.4f}")
    
    # Show individual losses
    individual_losses = -torch.log(y_hat[range(len(y_hat)), y_true])
    print(f"Individual losses: {individual_losses}")
    print("Note: Lower loss for confident correct predictions\n")


# =====================================================================
# PART 3: FashionMNIST Dataset Handling
# =====================================================================

def load_fashion_mnist(batch_size=256):
    """
    Load FashionMNIST dataset with appropriate transformations.
    
    FashionMNIST: 70k images (60k train, 10k test) of 28x28 grayscale
    clothing items in 10 categories.
    """
    # Transform: Convert PIL Image to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor, scale to [0,1]
    ])
    
    # Download and load datasets
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
    
    return train_loader, test_loader


def get_fashion_mnist_labels():
    """Return text labels for FashionMNIST classes."""
    return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def visualize_samples(data_loader, num_samples=12):
    """Visualize sample images from the dataset."""
    labels = get_fashion_mnist_labels()
    
    # Get a batch of data
    data_iter = iter(data_loader)
    images, targets = next(data_iter)
    
    # Plot samples
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for i in range(num_samples):
        row, col = i // 4, i % 4
        axes[row, col].imshow(images[i].squeeze(), cmap='gray')
        axes[row, col].set_title(f'{labels[targets[i]]} ({targets[i]})')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('module-2/fashion_mnist_samples.png')
    print("Sample images saved to 'fashion_mnist_samples.png'")


# =====================================================================
# PART 4: Softmax Regression Model from Scratch
# =====================================================================

class SoftmaxRegressionScratch:
    """
    Softmax regression implemented from scratch.
    
    This model flattens input images and applies a linear transformation
    followed by softmax to produce class probabilities.
    
    Architecture: Flatten -> Linear(784, 10) -> Softmax
    """
    
    def __init__(self, num_inputs, num_outputs, learning_rate=0.1):
        """
        Initialize model parameters.
        
        Args:
            num_inputs: Input feature dimension (784 for 28x28 images)
            num_outputs: Number of classes (10 for FashionMNIST)
            learning_rate: Learning rate for gradient descent
        """
        # Initialize weights with small random values
        self.W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
        
        # Initialize biases to zero
        self.b = torch.zeros(num_outputs, requires_grad=True)
        
        self.lr = learning_rate
        
        print(f"Model initialized:")
        print(f"  Weight shape: {self.W.shape}")
        print(f"  Bias shape: {self.b.shape}")
        print(f"  Learning rate: {self.lr}")
    
    def forward(self, X):
        """
        Forward pass: compute predictions.
        
        Args:
            X: Input batch of shape (batch_size, channels, height, width)
            
        Returns:
            Predicted probabilities of shape (batch_size, num_classes)
        """
        # Flatten images: (batch_size, channels, height, width) -> (batch_size, features)
        X_flat = X.reshape(X.shape[0], -1)
        
        # Linear transformation: X @ W + b
        logits = torch.matmul(X_flat, self.W) + self.b
        
        # Apply softmax to get probabilities
        return softmax(logits)
    
    def compute_loss(self, y_hat, y):
        """Compute cross-entropy loss."""
        return cross_entropy_loss(y_hat, y)
    
    def compute_accuracy(self, y_hat, y):
        """
        Compute classification accuracy.
        
        Args:
            y_hat: Predicted probabilities
            y: True labels
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        predicted_classes = y_hat.argmax(dim=1)
        return (predicted_classes == y).float().mean()
    
    def backward_and_update(self, X, y):
        """
        Compute gradients and update parameters.
        
        This implements the gradient computation for softmax regression:
        - dW = (1/batch_size) * X^T @ (y_hat - y_onehot)
        - db = (1/batch_size) * sum(y_hat - y_onehot)
        """
        batch_size = X.shape[0]
        
        # Forward pass
        y_hat = self.forward(X)
        
        # Convert labels to one-hot encoding
        y_onehot = torch.zeros_like(y_hat)
        y_onehot[range(batch_size), y] = 1
        
        # Compute gradients
        X_flat = X.reshape(X.shape[0], -1)
        
        # Gradient w.r.t. weights: X^T @ (y_hat - y_onehot) / batch_size
        dW = torch.matmul(X_flat.T, (y_hat - y_onehot)) / batch_size
        
        # Gradient w.r.t. bias: mean(y_hat - y_onehot)
        db = (y_hat - y_onehot).mean(dim=0)
        
        # Update parameters (gradient descent)
        with torch.no_grad():
            self.W -= self.lr * dW
            self.b -= self.lr * db
        
        return self.compute_loss(y_hat, y), self.compute_accuracy(y_hat, y)


# =====================================================================
# PART 5: Training Loop
# =====================================================================

def train_model(model, train_loader, test_loader, num_epochs=10):
    """
    Train the softmax regression model.
    
    Args:
        model: SoftmaxRegressionScratch instance
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Number of training epochs
        
    Returns:
        Dictionary with training history
    """
    print(f"\n=== Training Softmax Regression for {num_epochs} epochs ===")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.W.requires_grad_(True)
        model.b.requires_grad_(True)
        
        epoch_train_loss = 0
        epoch_train_acc = 0
        num_batches = 0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            # One training step
            loss, acc = model.backward_and_update(X, y)
            
            epoch_train_loss += loss.item()
            epoch_train_acc += acc.item()
            num_batches += 1
        
        # Average metrics
        avg_train_loss = epoch_train_loss / num_batches
        avg_train_acc = epoch_train_acc / num_batches
        
        # Evaluation phase
        test_acc = evaluate_model(model, test_loader)
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['test_acc'].append(test_acc)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}: "
              f"Train Loss = {avg_train_loss:.4f}, "
              f"Train Acc = {avg_train_acc:.4f}, "
              f"Test Acc = {test_acc:.4f}")
    
    return history


def evaluate_model(model, test_loader):
    """Evaluate model on test set."""
    total_correct = 0
    total_samples = 0
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        for X, y in test_loader:
            y_hat = model.forward(X)
            predicted = y_hat.argmax(dim=1)
            total_correct += (predicted == y).sum().item()
            total_samples += y.size(0)
    
    return total_correct / total_samples


# =====================================================================
# PART 6: Visualization and Analysis
# =====================================================================

def plot_training_history(history):
    """Plot training curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['test_acc'], 'r-', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('module-2/softmax_training_curves.png')
    print("\nTraining curves saved to 'softmax_training_curves.png'")


def visualize_weights(model):
    """Visualize learned weight vectors as images."""
    labels = get_fashion_mnist_labels()
    
    # Reshape weights to image format
    weight_images = model.W.T.reshape(10, 28, 28).detach()
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(10):
        row, col = i // 5, i % 5
        axes[row, col].imshow(weight_images[i], cmap='coolwarm')
        axes[row, col].set_title(f'{labels[i]}')
        axes[row, col].axis('off')
    
    plt.suptitle('Learned Weight Vectors (as Images)')
    plt.tight_layout()
    plt.savefig('module-2/weight_visualizations.png')
    print("Weight visualizations saved to 'weight_visualizations.png'")


def analyze_predictions(model, test_loader, num_samples=10):
    """Analyze model predictions on test samples."""
    labels = get_fashion_mnist_labels()
    
    # Get a batch of test data
    data_iter = iter(test_loader)
    X, y_true = next(data_iter)
    
    # Make predictions
    with torch.no_grad():
        y_hat = model.forward(X[:num_samples])
        y_pred = y_hat.argmax(dim=1)
    
    # Plot results
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(num_samples):
        row, col = i // 5, i % 5
        
        axes[row, col].imshow(X[i].squeeze(), cmap='gray')
        
        true_label = labels[y_true[i]]
        pred_label = labels[y_pred[i]]
        confidence = y_hat[i].max().item()
        
        color = 'green' if y_true[i] == y_pred[i] else 'red'
        title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}'
        
        axes[row, col].set_title(title, color=color, fontsize=8)
        axes[row, col].axis('off')
    
    plt.suptitle('Model Predictions on Test Samples')
    plt.tight_layout()
    plt.savefig('module-2/prediction_analysis.png')
    print("Prediction analysis saved to 'prediction_analysis.png'")


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    """Run the complete softmax regression demonstration."""
    
    print("="*60)
    print("SOFTMAX REGRESSION FROM SCRATCH")
    print("="*60)
    
    # Demonstrate core concepts
    demonstrate_softmax()
    demonstrate_cross_entropy()
    
    # Load data
    print("=== Loading FashionMNIST Dataset ===")
    train_loader, test_loader = load_fashion_mnist(batch_size=256)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Batch size: {train_loader.batch_size}")
    print()
    
    # Visualize sample data
    visualize_samples(train_loader)
    
    # Create and train model
    model = SoftmaxRegressionScratch(
        num_inputs=28*28,  # 784 features (flattened 28x28 images)
        num_outputs=10,    # 10 classes
        learning_rate=0.1
    )
    
    # Train the model
    history = train_model(model, train_loader, test_loader, num_epochs=10)
    
    # Visualize results
    plot_training_history(history)
    visualize_weights(model)
    analyze_predictions(model, test_loader)
    
    # Final evaluation
    final_test_acc = evaluate_model(model, test_loader)
    print(f"\n=== Final Results ===")
    print(f"Final test accuracy: {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")
    
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("1. Softmax converts logits to probabilities")
    print("2. Cross-entropy loss penalizes wrong predictions")
    print("3. Linear model learns class templates (visible in weights)")
    print("4. Model achieves ~85% accuracy on FashionMNIST")
    print("5. Training loss decreases, test accuracy increases over epochs")
    print("="*60)


if __name__ == "__main__":
    main() 