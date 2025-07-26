#!/usr/bin/env python3
"""
ASSIGNMENT: Softmax Regression on FashionMNIST

Complete the TODO sections to implement softmax regression for multiclass
classification on the FashionMNIST dataset.

Learning Objectives:
1. Implement the softmax function
2. Implement cross-entropy loss
3. Compute gradients for multiclass classification
4. Build a complete training loop
5. Evaluate model performance on FashionMNIST
6. Visualize and interpret results

Student Name: ________________________
Date: ________________________
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
# PART 1: Dataset Loading (PROVIDED)
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
# PART 2: Core Functions to Implement
# =====================================================================

def softmax(X):
    """
    TODO 1: Implement the softmax function.
    
    The softmax function converts raw scores (logits) into probabilities:
    softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
    
    IMPORTANT: Subtract the maximum value for numerical stability!
    
    Args:
        X: Input tensor of shape (batch_size, num_classes)
        
    Returns:
        Tensor of same shape with probabilities (each row sums to 1)
        
    Hints:
    - Use torch.exp() for exponentiation
    - Use X.max(dim=1, keepdim=True) to find max along each row
    - Use .sum(dim=1, keepdim=True) to sum along each row
    """
    # TODO: Implement softmax function here
    pass


def cross_entropy_loss(y_hat, y):
    """
    TODO 2: Implement cross-entropy loss.
    
    Cross-entropy loss: -log(predicted_probability_for_true_class)
    
    Args:
        y_hat: Predicted probabilities, shape (batch_size, num_classes)
        y: True labels, shape (batch_size,) with integer class indices
        
    Returns:
        Scalar loss value (averaged over batch)
        
    Hints:
    - Use y_hat[range(len(y_hat)), y] to select probabilities for true classes
    - Use torch.log() and torch.mean()
    - Don't forget the negative sign!
    """
    # TODO: Implement cross-entropy loss here
    pass


# =====================================================================
# PART 3: Model Class to Complete
# =====================================================================

class SoftmaxRegressionScratch:
    """Softmax regression implemented from scratch."""
    
    def __init__(self, num_inputs, num_outputs, learning_rate=0.1):
        """
        TODO 3: Initialize model parameters.
        
        Args:
            num_inputs: Input feature dimension (784 for 28x28 images)
            num_outputs: Number of classes (10 for FashionMNIST)
            learning_rate: Learning rate for gradient descent
            
        Hints:
        - Initialize weights with small random values: torch.normal(0, 0.01, size=...)
        - Initialize biases to zero: torch.zeros(...)
        - Store learning rate for later use
        """
        # TODO: Initialize self.W, self.b, and self.lr
        pass
    
    def forward(self, X):
        """
        TODO 4: Implement forward pass.
        
        Args:
            X: Input batch of shape (batch_size, channels, height, width)
            
        Returns:
            Predicted probabilities of shape (batch_size, num_classes)
            
        Hints:
        - Flatten images: X.reshape(X.shape[0], -1)
        - Linear transformation: X @ W + b
        - Apply softmax to get probabilities
        """
        # TODO: Implement forward pass here
        pass
    
    def compute_accuracy(self, y_hat, y):
        """
        TODO 5: Compute classification accuracy.
        
        Args:
            y_hat: Predicted probabilities
            y: True labels
            
        Returns:
            Accuracy as a float between 0 and 1
            
        Hints:
        - Use y_hat.argmax(dim=1) to get predicted classes
        - Compare with true labels and compute mean
        """
        # TODO: Implement accuracy computation here
        pass
    
    def backward_and_update(self, X, y):
        """
        TODO 6: Implement gradient computation and parameter update.
        
        This is the most challenging part! You need to:
        1. Compute forward pass
        2. Convert labels to one-hot encoding
        3. Compute gradients using the formulas:
           - dW = (1/batch_size) * X^T @ (y_hat - y_onehot)
           - db = mean(y_hat - y_onehot, dim=0)
        4. Update parameters using gradient descent
        
        Args:
            X: Input batch
            y: True labels
            
        Returns:
            loss, accuracy for this batch
            
        Hints:
        - Create one-hot encoding: y_onehot = torch.zeros_like(y_hat); y_onehot[range(batch_size), y] = 1
        - Use torch.matmul() for matrix multiplication
        - Update with: self.W -= self.lr * dW
        """
        # TODO: Implement backward pass and parameter update here
        pass


# =====================================================================
# PART 4: Training Loop to Complete
# =====================================================================

def train_model(model, train_loader, test_loader, num_epochs=10):
    """
    TODO 7: Implement the training loop.
    
    Args:
        model: SoftmaxRegressionScratch instance
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Number of training epochs
        
    Returns:
        Dictionary with training history
        
    Structure:
    1. For each epoch:
       a. Loop through training batches
       b. Call model.backward_and_update() for each batch
       c. Track training loss and accuracy
       d. Evaluate on test set
       e. Print progress
    2. Return history dictionary
    """
    print(f"Training for {num_epochs} epochs...")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    for epoch in range(num_epochs):
        # TODO: Implement training loop for one epoch
        # Hints:
        # - Initialize epoch_train_loss = 0, epoch_train_acc = 0, num_batches = 0
        # - Loop through train_loader: for X, y in train_loader:
        # - Call loss, acc = model.backward_and_update(X, y)
        # - Accumulate losses and accuracies
        # - Compute averages and append to history
        # - Call evaluate_model() for test accuracy
        # - Print progress
        pass
    
    return history


def evaluate_model(model, test_loader):
    """
    TODO 8: Implement model evaluation.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        
    Returns:
        Test accuracy
        
    Hints:
    - Loop through test_loader
    - Call model.forward() and model.compute_accuracy()
    - Average accuracies across all batches
    """
    # TODO: Implement model evaluation here
    pass


# =====================================================================
# PART 5: Visualization Functions (PROVIDED)
# =====================================================================

def visualize_samples(data_loader, num_samples=12):
    """Visualize sample images from the dataset."""
    labels = get_fashion_mnist_labels()
    
    data_iter = iter(data_loader)
    images, targets = next(data_iter)
    
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for i in range(num_samples):
        row, col = i // 4, i % 4
        axes[row, col].imshow(images[i].squeeze(), cmap='gray')
        axes[row, col].set_title(f'{labels[targets[i]]}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('module-2/assignment_samples.png')
    print("Sample images saved to 'assignment_samples.png'")


def plot_training_history(history):
    """Plot training curves."""
    if not history['train_loss']:  # Check if history is empty
        print("No training history to plot!")
        return
        
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
    plt.savefig('module-2/assignment_training_curves.png')
    print("Training curves saved to 'assignment_training_curves.png'")


# =====================================================================
# PART 6: Testing Functions
# =====================================================================

def test_softmax():
    """Test your softmax implementation."""
    print("Testing softmax function...")
    
    # Test case 1: Simple case
    X = torch.tensor([[1.0, 2.0, 3.0],
                      [1.0, 1.0, 1.0]])
    
    result = softmax(X)
    
    if result is None:
        print("❌ Softmax not implemented yet!")
        return False
    
    # Check if probabilities sum to 1
    sums = result.sum(dim=1)
    if torch.allclose(sums, torch.ones(2)):
        print("✅ Softmax test passed!")
        return True
    else:
        print("❌ Softmax test failed! Rows don't sum to 1.")
        return False


def test_cross_entropy():
    """Test your cross-entropy implementation."""
    print("Testing cross-entropy loss...")
    
    # Test case
    y_hat = torch.tensor([[0.7, 0.2, 0.1],
                          [0.1, 0.8, 0.1]])
    y = torch.tensor([0, 1])
    
    loss = cross_entropy_loss(y_hat, y)
    
    if loss is None:
        print("❌ Cross-entropy not implemented yet!")
        return False
    
    # Expected loss should be reasonable
    if 0.1 < loss < 1.0:
        print("✅ Cross-entropy test passed!")
        return True
    else:
        print("❌ Cross-entropy test failed! Loss value seems incorrect.")
        return False


# =====================================================================
# PART 7: Main Execution
# =====================================================================

def main():
    """Run the assignment."""
    print("="*60)
    print("SOFTMAX REGRESSION ASSIGNMENT")
    print("="*60)
    
    # Test implementations
    print("\n=== Testing Your Implementations ===")
    softmax_ok = test_softmax()
    cross_entropy_ok = test_cross_entropy()
    
    if not (softmax_ok and cross_entropy_ok):
        print("\n❌ Please fix the failed tests before proceeding!")
        return
    
    # Load data
    print("\n=== Loading FashionMNIST Dataset ===")
    train_loader, test_loader = load_fashion_mnist(batch_size=256)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Visualize samples
    visualize_samples(train_loader)
    
    # Create and train model
    print("\n=== Creating and Training Model ===")
    model = SoftmaxRegressionScratch(
        num_inputs=28*28,  # 784 features
        num_outputs=10,    # 10 classes
        learning_rate=0.1
    )
    
    # Check if model was properly initialized
    if model.W is None or model.b is None:
        print("❌ Model not properly initialized! Check TODO 3.")
        return
    
    # Train the model
    history = train_model(model, train_loader, test_loader, num_epochs=10)
    
    if not history['train_loss']:
        print("❌ Training not implemented! Check TODO 7.")
        return
    
    # Visualize results
    plot_training_history(history)
    
    # Final evaluation
    final_test_acc = evaluate_model(model, test_loader)
    if final_test_acc is not None:
        print(f"\n=== Final Results ===")
        print(f"Final test accuracy: {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")
        
        # Performance expectations
        if final_test_acc > 0.80:
            print("🎉 Excellent! Your model performs very well!")
        elif final_test_acc > 0.75:
            print("👍 Good job! Your model performs well!")
        elif final_test_acc > 0.70:
            print("👌 Not bad! Room for improvement.")
        else:
            print("🤔 Your model might need some debugging.")
    
    print("\n" + "="*60)
    print("ASSIGNMENT COMPLETE!")
    print("="*60)


# =====================================================================
# PART 8: Questions for Reflection
# =====================================================================

"""
REFLECTION QUESTIONS (Answer these in comments):

Q1: What does the softmax function do, and why is it useful for classification?
A1: 

Q2: Why do we subtract the maximum value in the softmax function?
A2: 

Q3: What is cross-entropy loss measuring?
A3: 

Q4: How does the gradient computation for softmax regression differ from linear regression?
A4: 

Q5: What accuracy did your model achieve? How does this compare to random guessing?
A5: 

Q6: Looking at the weight visualizations, what patterns do you notice?
A6: 

Q7: What could you do to improve the model's performance?
A7: 

Q8: What are the limitations of using a linear model for image classification?
A8: 
"""

if __name__ == "__main__":
    main() 