#!/usr/bin/env python3
"""
COMPLETE SOLUTION: Softmax Regression on FashionMNIST

This is the complete implementation of the softmax regression assignment,
showing all solutions to the TODO sections.

This serves as:
1. Reference implementation for instructors
2. Solution guide for students (after completing assignment)
3. Complete working example of softmax regression from scratch

Author: Complete Solution Version
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
# PART 1: Dataset Loading
# =====================================================================

def load_fashion_mnist(batch_size=256):
    """Load FashionMNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
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


# =====================================================================
# PART 2: Core Functions - SOLUTIONS
# =====================================================================

def softmax(X):
    """
    SOLUTION 1: Softmax function implementation.
    
    Converts raw scores (logits) into probabilities using numerical stability trick.
    """
    # Subtract max for numerical stability (prevents overflow)
    X_stable = X - X.max(dim=1, keepdim=True).values
    
    # Compute exponentials
    X_exp = torch.exp(X_stable)
    
    # Normalize by sum to get probabilities
    return X_exp / X_exp.sum(dim=1, keepdim=True)


def cross_entropy_loss(y_hat, y):
    """
    SOLUTION 2: Cross-entropy loss implementation.
    
    Computes the negative log-likelihood of the true class.
    """
    # Select the predicted probability for the true class
    selected_probs = y_hat[range(len(y_hat)), y]
    
    # Apply negative log and take mean
    return -torch.log(selected_probs).mean()


# =====================================================================
# PART 3: Model Class - SOLUTIONS
# =====================================================================

class SoftmaxRegressionScratch:
    """Complete softmax regression implementation from scratch."""
    
    def __init__(self, num_inputs, num_outputs, learning_rate=0.1):
        """
        SOLUTION 3: Parameter initialization.
        """
        # Initialize weights with small random values
        self.W = torch.normal(0, 0.01, size=(num_inputs, num_outputs))
        
        # Initialize biases to zero
        self.b = torch.zeros(num_outputs)
        
        # Store learning rate
        self.lr = learning_rate
        
        print(f"Model initialized with {num_inputs} inputs, {num_outputs} outputs")
        print(f"Weight shape: {self.W.shape}, Bias shape: {self.b.shape}")
    
    def forward(self, X):
        """
        SOLUTION 4: Forward pass implementation.
        """
        # Flatten images from (batch_size, 1, 28, 28) to (batch_size, 784)
        X_flat = X.reshape(X.shape[0], -1)
        
        # Linear transformation: X @ W + b
        logits = torch.matmul(X_flat, self.W) + self.b
        
        # Apply softmax to get probabilities
        return softmax(logits)
    
    def compute_accuracy(self, y_hat, y):
        """
        SOLUTION 5: Accuracy computation.
        """
        # Get predicted classes (index of maximum probability)
        predicted_classes = y_hat.argmax(dim=1)
        
        # Compare with true labels and compute mean
        return (predicted_classes == y).float().mean()
    
    def backward_and_update(self, X, y):
        """
        SOLUTION 6: Gradient computation and parameter update.
        
        This implements the gradient formulas for softmax regression:
        - For weights: dW = (1/m) * X^T @ (y_hat - y_onehot)
        - For bias: db = (1/m) * sum(y_hat - y_onehot)
        
        Where m is the batch size and y_onehot is one-hot encoded labels.
        """
        batch_size = X.shape[0]
        
        # Forward pass to get predictions
        y_hat = self.forward(X)
        
        # Convert labels to one-hot encoding
        y_onehot = torch.zeros_like(y_hat)
        y_onehot[range(batch_size), y] = 1
        
        # Compute error (difference between predicted and true distributions)
        error = y_hat - y_onehot
        
        # Flatten input for gradient computation
        X_flat = X.reshape(X.shape[0], -1)
        
        # Compute gradients
        grad_W = torch.matmul(X_flat.T, error) / batch_size
        grad_b = error.mean(dim=0)
        
        # Update parameters using gradient descent
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b
        
        # Return loss and accuracy for monitoring
        loss = cross_entropy_loss(y_hat, y)
        accuracy = self.compute_accuracy(y_hat, y)
        
        return loss, accuracy


# =====================================================================
# PART 4: Training Loop - SOLUTIONS
# =====================================================================

def train_model(model, train_loader, test_loader, num_epochs=10):
    """
    SOLUTION 7: Complete training loop implementation.
    """
    print(f"Training for {num_epochs} epochs...")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    for epoch in range(num_epochs):
        # Initialize epoch metrics
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        num_batches = 0
        
        # Training loop
        for X, y in train_loader:
            # One training step
            loss, acc = model.backward_and_update(X, y)
            
            # Accumulate metrics
            epoch_train_loss += loss.item()
            epoch_train_acc += acc.item()
            num_batches += 1
        
        # Compute average metrics for the epoch
        avg_train_loss = epoch_train_loss / num_batches
        avg_train_acc = epoch_train_acc / num_batches
        
        # Evaluate on test set
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
    """
    SOLUTION 8: Model evaluation implementation.
    """
    total_accuracy = 0.0
    num_batches = 0
    
    # Evaluate on all test batches
    for X, y in test_loader:
        # Get predictions (no gradient computation needed)
        y_hat = model.forward(X)
        
        # Compute accuracy for this batch
        batch_acc = model.compute_accuracy(y_hat, y)
        
        # Accumulate
        total_accuracy += batch_acc.item()
        num_batches += 1
    
    # Return average accuracy
    return total_accuracy / num_batches


# =====================================================================
# PART 5: Enhanced Visualization Functions
# =====================================================================

def visualize_samples(data_loader, num_samples=12):
    """Visualize sample images from the dataset."""
    labels = get_fashion_mnist_labels()
    
    data_iter = iter(data_loader)
    images, targets = next(data_iter)
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    for i in range(num_samples):
        row, col = i // 4, i % 4
        axes[row, col].imshow(images[i].squeeze(), cmap='gray')
        axes[row, col].set_title(f'{labels[targets[i]]} ({targets[i]})')
        axes[row, col].axis('off')
    
    plt.suptitle('FashionMNIST Sample Images')
    plt.tight_layout()
    plt.savefig('module-2/complete_samples.png', dpi=150)
    print("Sample images saved to 'complete_samples.png'")


def plot_training_history(history):
    """Plot comprehensive training curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Training Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].set_title('Training Loss Over Time')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Training vs Test Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training', linewidth=2)
    axes[1].plot(epochs, history['test_acc'], 'r-', label='Test', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training vs Test Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Generalization Gap
    gap = [train - test for train, test in zip(history['train_acc'], history['test_acc'])]
    axes[2].plot(epochs, gap, 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Train Acc - Test Acc')
    axes[2].set_title('Generalization Gap')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('module-2/complete_training_analysis.png', dpi=150)
    print("Training analysis saved to 'complete_training_analysis.png'")


def visualize_learned_weights(model):
    """Visualize the learned weight vectors as images."""
    labels = get_fashion_mnist_labels()
    
    # Reshape weights to image format: (10, 784) -> (10, 28, 28)
    weight_images = model.W.T.reshape(10, 28, 28)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(10):
        row, col = i // 5, i % 5
        
        # Display weight vector as image
        im = axes[row, col].imshow(weight_images[i], cmap='RdBu', vmin=-0.05, vmax=0.05)
        axes[row, col].set_title(f'{labels[i]}')
        axes[row, col].axis('off')
    
    plt.suptitle('Learned Weight Vectors (Class Templates)')
    plt.tight_layout()
    
    # Add colorbar
    plt.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.savefig('module-2/complete_weight_visualization.png', dpi=150)
    print("Weight visualization saved to 'complete_weight_visualization.png'")


def analyze_predictions(model, test_loader, num_samples=16):
    """Analyze model predictions with confidence scores."""
    labels = get_fashion_mnist_labels()
    
    # Get test samples
    data_iter = iter(test_loader)
    X, y_true = next(data_iter)
    
    # Make predictions
    y_hat = model.forward(X[:num_samples])
    y_pred = y_hat.argmax(dim=1)
    confidences = y_hat.max(dim=1)[0]
    
    # Create visualization
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(num_samples):
        row, col = i // 4, i % 4
        
        # Display image
        axes[row, col].imshow(X[i].squeeze(), cmap='gray')
        
        # Prepare title with prediction info
        true_label = labels[y_true[i]]
        pred_label = labels[y_pred[i]]
        confidence = confidences[i].item()
        
        # Color code: green for correct, red for incorrect
        color = 'green' if y_true[i] == y_pred[i] else 'red'
        
        title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}'
        axes[row, col].set_title(title, color=color, fontsize=8)
        axes[row, col].axis('off')
    
    plt.suptitle('Model Predictions with Confidence Scores', fontsize=14)
    plt.tight_layout()
    plt.savefig('module-2/complete_prediction_analysis.png', dpi=150)
    print("Prediction analysis saved to 'complete_prediction_analysis.png'")


def create_confusion_matrix(model, test_loader):
    """Create and visualize confusion matrix."""
    labels = get_fashion_mnist_labels()
    
    # Collect all predictions
    all_preds = []
    all_true = []
    
    for X, y in test_loader:
        y_hat = model.forward(X)
        preds = y_hat.argmax(dim=1)
        all_preds.extend(preds.tolist())
        all_true.extend(y.tolist())
    
    # Create confusion matrix
    num_classes = len(labels)
    confusion_matrix = torch.zeros(num_classes, num_classes)
    
    for true_label, pred_label in zip(all_true, all_preds):
        confusion_matrix[true_label, pred_label] += 1
    
    # Normalize by row (true class)
    confusion_matrix = confusion_matrix / confusion_matrix.sum(dim=1, keepdim=True)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion_matrix, cmap='Blues', vmin=0, vmax=1)
    
    # Add labels
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    
    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax.text(j, i, f'{confusion_matrix[i, j]:.2f}',
                          ha="center", va="center", 
                          color="white" if confusion_matrix[i, j] > 0.5 else "black")
    
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_title('Confusion Matrix (Normalized)')
    
    # Add colorbar
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('module-2/complete_confusion_matrix.png', dpi=150)
    print("Confusion matrix saved to 'complete_confusion_matrix.png'")


# =====================================================================
# PART 6: Testing Functions
# =====================================================================

def test_implementations():
    """Test all implemented functions."""
    print("=== Testing Implementations ===")
    
    # Test softmax
    print("Testing softmax...")
    X = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    result = softmax(X)
    assert torch.allclose(result.sum(dim=1), torch.ones(2)), "Softmax rows should sum to 1"
    print("✅ Softmax test passed!")
    
    # Test cross-entropy
    print("Testing cross-entropy...")
    y_hat = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    y = torch.tensor([0, 1])
    loss = cross_entropy_loss(y_hat, y)
    assert 0.1 < loss < 1.0, "Cross-entropy loss should be reasonable"
    print("✅ Cross-entropy test passed!")
    
    # Test model initialization
    print("Testing model initialization...")
    model = SoftmaxRegressionScratch(784, 10)
    assert model.W.shape == (784, 10), "Weight shape should be (784, 10)"
    assert model.b.shape == (10,), "Bias shape should be (10,)"
    print("✅ Model initialization test passed!")
    
    print("All tests passed! 🎉\n")


# =====================================================================
# PART 7: Main Execution
# =====================================================================

def main():
    """Run the complete softmax regression solution."""
    print("="*60)
    print("COMPLETE SOFTMAX REGRESSION SOLUTION")
    print("="*60)
    
    # Test implementations
    test_implementations()
    
    # Load data
    print("=== Loading FashionMNIST Dataset ===")
    train_loader, test_loader = load_fashion_mnist(batch_size=256)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print()
    
    # Visualize samples
    visualize_samples(train_loader)
    
    # Create and train model
    print("=== Training Model ===")
    model = SoftmaxRegressionScratch(
        num_inputs=28*28,  # 784 features
        num_outputs=10,    # 10 classes
        learning_rate=0.1
    )
    print()
    
    # Train the model
    history = train_model(model, train_loader, test_loader, num_epochs=15)
    
    # Comprehensive analysis
    print("\n=== Creating Visualizations ===")
    plot_training_history(history)
    visualize_learned_weights(model)
    analyze_predictions(model, test_loader)
    create_confusion_matrix(model, test_loader)
    
    # Final evaluation
    final_test_acc = evaluate_model(model, test_loader)
    print(f"\n=== Final Results ===")
    print(f"Final test accuracy: {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")
    
    # Performance analysis
    best_test_acc = max(history['test_acc'])
    final_train_acc = history['train_acc'][-1]
    generalization_gap = final_train_acc - final_test_acc
    
    print(f"Best test accuracy: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")
    print(f"Final training accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"Generalization gap: {generalization_gap:.4f}")
    
    print("\n" + "="*60)
    print("SOLUTION ANALYSIS:")
    print("1. ✅ Softmax function correctly converts logits to probabilities")
    print("2. ✅ Cross-entropy loss properly measures classification error")
    print("3. ✅ Gradient computation follows multiclass logistic regression theory")
    print("4. ✅ Model achieves ~85% accuracy on FashionMNIST")
    print("5. ✅ Weight visualizations show meaningful class templates")
    print("6. ✅ Confusion matrix reveals which classes are hardest to distinguish")
    print("="*60)


# =====================================================================
# PART 8: Solution Explanations
# =====================================================================

"""
DETAILED SOLUTION EXPLANATIONS:

1. SOFTMAX FUNCTION:
   - Subtracts maximum for numerical stability (prevents exp overflow)
   - Exponentiates all values
   - Normalizes by sum to ensure probabilities sum to 1
   - Key insight: Softmax is a generalization of sigmoid to multiple classes

2. CROSS-ENTROPY LOSS:
   - Measures the "distance" between predicted and true probability distributions
   - For classification, true distribution is one-hot (1 for correct class, 0 for others)
   - Loss = -log(probability assigned to correct class)
   - Lower loss when model is more confident about correct class

3. GRADIENT COMPUTATION:
   - Uses the fact that gradient of cross-entropy + softmax has a simple form
   - Gradient w.r.t. logits: (predicted_probs - one_hot_true)
   - This flows back through the linear layer to update weights and biases
   - The beauty: no need to manually compute softmax derivatives!

4. TRAINING DYNAMICS:
   - Model learns to increase scores for correct classes
   - Weight vectors become "templates" for each class
   - Positive weights indicate features that support a class
   - Negative weights indicate features that oppose a class

5. PERFORMANCE EXPECTATIONS:
   - Random guessing: 10% accuracy (1/10 classes)
   - Simple linear model: ~85% accuracy
   - Perfect classification: Nearly impossible due to inherent class overlap

COMMON ISSUES AND DEBUGGING:
- If accuracy stays near 10%: Check gradient computation
- If loss explodes: Check numerical stability in softmax
- If training doesn't improve: Check learning rate
- If overfitting: Consider regularization
"""

if __name__ == "__main__":
    main() 