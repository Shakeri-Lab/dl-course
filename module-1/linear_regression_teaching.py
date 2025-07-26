#!/usr/bin/env python3
"""
Linear Regression: From Mathematical Concepts to Implementation

This script demonstrates how to build a neural network for linear regression
from scratch, emphasizing the core concepts and mathematical foundations.

Key Learning Objectives:
1. Understand the linear model: y = Xw + b
2. Implement forward propagation (predictions)
3. Compute loss using Mean Squared Error
4. Calculate gradients manually (backpropagation)
5. Update parameters using gradient descent

Author: Teaching Version
"""

import torch
import random
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)


# =====================================================================
# PART 1: Understanding the Problem and Data
# =====================================================================

def generate_synthetic_data(weights, bias, num_samples):
    """
    Generate synthetic linear data with small noise.
    
    Linear relationship: y = X @ w + b + noise
    
    Args:
        weights: True parameters we want to learn
        bias: True bias/intercept we want to learn
        num_samples: Number of data points to generate
        
    Returns:
        X: Input features (num_samples x num_features)
        y: Target values (num_samples x 1)
    """
    num_features = len(weights)
    
    # Generate random input features from standard normal distribution
    X = torch.randn(num_samples, num_features)
    
    # Compute true output using linear relationship
    y_true = X @ weights + bias
    
    # Add small Gaussian noise to simulate real-world data
    noise = 0.01 * torch.randn(num_samples)
    y = y_true + noise
    
    # Reshape y to be column vector for consistency
    return X, y.reshape(-1, 1)


# Define the true parameters we'll try to learn
TRUE_WEIGHTS = torch.tensor([2.0, -3.4])
TRUE_BIAS = 4.2

# Generate training data
print("=== Data Generation ===")
features, labels = generate_synthetic_data(TRUE_WEIGHTS, TRUE_BIAS, 1000)
print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")
print(f"True parameters: w={TRUE_WEIGHTS.tolist()}, b={TRUE_BIAS}")
print()


# =====================================================================
# PART 2: Building Linear Regression from Scratch
# =====================================================================

class LinearRegressionFromScratch:
    """
    A neural network with one linear layer, implemented from scratch.
    
    This class demonstrates the fundamental operations in neural networks:
    - Forward pass: computing predictions
    - Loss calculation: measuring error
    - Backward pass: computing gradients
    - Parameter update: gradient descent
    """
    
    def __init__(self, input_size, learning_rate=0.01):
        """
        Initialize the model parameters.
        
        Args:
            input_size: Number of input features
            learning_rate: Step size for gradient descent
        """
        # Initialize weights randomly (breaks symmetry)
        self.w = torch.randn(input_size, 1) * 0.01
        
        # Initialize bias to zero (common practice)
        self.b = torch.zeros(1)
        
        # Learning rate controls how much we update parameters
        self.lr = learning_rate
        
        print(f"Model initialized with:")
        print(f"  Weights shape: {self.w.shape}")
        print(f"  Bias shape: {self.b.shape}")
        print(f"  Learning rate: {self.lr}")
    
    def forward(self, X):
        """
        Forward propagation: compute predictions.
        
        Linear transformation: y_hat = X @ w + b
        
        Args:
            X: Input features (batch_size x input_size)
            
        Returns:
            y_hat: Predictions (batch_size x 1)
        """
        return X @ self.w + self.b
    
    def compute_loss(self, y_hat, y_true):
        """
        Calculate Mean Squared Error loss.
        
        MSE = (1/n) * Σ(y_hat - y_true)²
        
        Args:
            y_hat: Predictions
            y_true: True labels
            
        Returns:
            loss: Scalar loss value
        """
        squared_errors = (y_hat - y_true) ** 2
        return squared_errors.mean()
    
    def backward(self, X, y_true):
        """
        Backward propagation: compute gradients and update parameters.
        
        This is where the learning happens!
        
        Math behind gradient computation:
        - Loss L = (1/n) * Σ(y_hat - y)²
        - ∂L/∂w = (2/n) * X^T @ (y_hat - y)
        - ∂L/∂b = (2/n) * Σ(y_hat - y)
        
        We can drop the factor of 2 as it's absorbed by learning rate.
        
        Args:
            X: Input features
            y_true: True labels
        """
        # First, compute predictions
        y_hat = self.forward(X)
        
        # Compute error (this is ∂L/∂y_hat)
        error = y_hat - y_true
        
        # Compute gradients using chain rule
        # For weights: gradient flows through X^T
        grad_w = X.T @ error / X.shape[0]
        
        # For bias: gradient is just mean of errors
        grad_b = error.mean()
        
        # Update parameters using gradient descent
        # New_param = Old_param - learning_rate * gradient
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b
        
        return error.abs().mean().item()  # Return MAE for monitoring
    
    def get_parameters(self):
        """Return current model parameters."""
        return self.w.flatten().tolist(), self.b.item()


# =====================================================================
# PART 3: Training Process
# =====================================================================

def create_mini_batches(X, y, batch_size, shuffle=True):
    """
    Create mini-batches for stochastic gradient descent.
    
    Mini-batch training is a compromise between:
    - Batch gradient descent (slow, stable)
    - Stochastic gradient descent (fast, noisy)
    
    Args:
        X: Features
        y: Labels
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data
        
    Yields:
        X_batch, y_batch: Mini-batches of data
    """
    n_samples = X.shape[0]
    indices = torch.randperm(n_samples) if shuffle else torch.arange(n_samples)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_indices = indices[start:end]
        yield X[batch_indices], y[batch_indices]


def train_model(model, features, labels, epochs=10, batch_size=10):
    """
    Train the model using mini-batch gradient descent.
    
    Training loop structure:
    1. Divide data into mini-batches
    2. For each batch: forward → loss → backward → update
    3. Monitor progress every few epochs
    
    Args:
        model: Our LinearRegressionFromScratch instance
        features: Training features
        labels: Training labels
        epochs: Number of passes through the data
        batch_size: Mini-batch size
    """
    print("\n=== Training Process ===")
    
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0
        
        # Process each mini-batch
        for X_batch, y_batch in create_mini_batches(features, labels, batch_size):
            # One step of gradient descent
            model.backward(X_batch, y_batch)
            
            # Track loss for monitoring
            y_hat = model.forward(X_batch)
            batch_loss = model.compute_loss(y_hat, y_batch)
            epoch_loss += batch_loss.item()
            n_batches += 1
        
        # Average loss for this epoch
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)
        
        # Print progress
        if epoch == 0 or (epoch + 1) % 5 == 0:
            w, b = model.get_parameters()
            print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:.6f}, "
                  f"w = [{w[0]:.3f}, {w[1]:.3f}], b = {b:.3f}")
    
    return loss_history


# =====================================================================
# PART 4: Putting It All Together
# =====================================================================

def main():
    """Main execution function demonstrating the complete workflow."""
    
    # Create model
    print("\n=== Model Creation ===")
    model = LinearRegressionFromScratch(input_size=2, learning_rate=0.03)
    
    # Train model
    loss_history = train_model(
        model, features, labels, 
        epochs=20, batch_size=10
    )
    
    # Compare learned vs true parameters
    print("\n=== Final Results ===")
    learned_w, learned_b = model.get_parameters()
    print(f"True parameters:    w = {TRUE_WEIGHTS.tolist()}, b = {TRUE_BIAS}")
    print(f"Learned parameters: w = {learned_w}, b = {learned_b:.3f}")
    
    # Calculate parameter errors
    w_error = torch.tensor(learned_w) - TRUE_WEIGHTS
    b_error = learned_b - TRUE_BIAS
    print(f"\nParameter errors:")
    print(f"  Weight error: {w_error.tolist()}")
    print(f"  Bias error: {b_error:.3f}")
    
    # Visualize training progress
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('chapter_linear-regression/training_loss.png')
    print("\nTraining loss plot saved to 'training_loss.png'")
    
    # Make predictions on new data
    print("\n=== Making Predictions ===")
    test_X = torch.tensor([[1.0, 2.0], [-1.0, 0.5], [0.0, -1.0]])
    predictions = model.forward(test_X)
    true_values = test_X @ TRUE_WEIGHTS + TRUE_BIAS
    
    print("Test predictions:")
    for i in range(len(test_X)):
        print(f"  Input: {test_X[i].tolist()}, "
              f"Predicted: {predictions[i].item():.3f}, "
              f"True: {true_values[i].item():.3f}")


# =====================================================================
# PART 5: Key Takeaways
# =====================================================================

"""
Key Concepts Demonstrated:

1. **The Linear Model**: 
   - Neural networks start with linear transformations
   - y = Xw + b is the foundation of deep learning

2. **Gradient Descent**:
   - We minimize loss by moving parameters in the opposite direction of gradients
   - Learning rate controls the step size

3. **Backpropagation**:
   - Even in this simple case, we compute gradients using the chain rule
   - This scales to deep networks with many layers

4. **Mini-batch Training**:
   - Processing small batches balances efficiency and stability
   - Essential for training on large datasets

5. **From Scratch Implementation**:
   - Understanding the fundamentals helps debug and improve models
   - PyTorch automates these operations but the principles remain the same

Next Steps:
- Try different learning rates and observe convergence
- Add more features to the synthetic data
- Implement regularization (L2 penalty)
- Compare with PyTorch's built-in nn.Linear
"""

if __name__ == "__main__":
    main() 