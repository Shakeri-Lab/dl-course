#!/usr/bin/env python3
"""
Linear Regression: PyTorch's High-Level API

This script demonstrates the same linear regression problem using PyTorch's
built-in tools (nn.Module, autograd, optimizers), showing how the framework
automates the manual operations from our scratch implementation.

Key PyTorch Components Used:
1. nn.Module - Base class for all neural network modules
2. nn.Linear - Built-in linear transformation layer
3. Autograd - Automatic differentiation
4. Optimizers - Pre-built gradient descent algorithms
5. Loss functions - Standard loss implementations

Author: PyTorch API Version
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)


# =====================================================================
# PART 1: Data Generation (Same as Before)
# =====================================================================

def generate_synthetic_data(weights, bias, num_samples):
    """Generate the same synthetic data as in the scratch version."""
    X = torch.randn(num_samples, len(weights))
    y = X @ weights + bias + 0.01 * torch.randn(num_samples)
    return X, y.reshape(-1, 1)


# True parameters to learn
TRUE_WEIGHTS = torch.tensor([2.0, -3.4])
TRUE_BIAS = 4.2

# Generate data
print("=== Data Generation ===")
features, labels = generate_synthetic_data(TRUE_WEIGHTS, TRUE_BIAS, 1000)
print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")
print(f"True parameters: w={TRUE_WEIGHTS.tolist()}, b={TRUE_BIAS}\n")


# =====================================================================
# PART 2: Model Definition Using nn.Module
# =====================================================================

class LinearRegressionModel(nn.Module):
    """
    A linear regression model using PyTorch's nn.Module.
    
    Key differences from scratch:
    - Inherits from nn.Module for automatic features
    - Uses nn.Linear which handles weight initialization
    - Automatic gradient computation via autograd
    - Parameters are registered automatically
    """
    
    def __init__(self, input_size, output_size=1):
        """
        Initialize the model.
        
        The super().__init__() call is crucial - it sets up the nn.Module
        infrastructure that makes everything work.
        """
        super(LinearRegressionModel, self).__init__()
        
        # nn.Linear creates a linear transformation layer
        # It automatically:
        # - Initializes weights from uniform distribution
        # - Initializes bias to zero
        # - Registers parameters for optimization
        self.linear = nn.Linear(input_size, output_size)
        
        print(f"Model created with nn.Linear({input_size}, {output_size})")
        print(f"Initial parameters:")
        print(f"  Weight shape: {self.linear.weight.shape}")
        print(f"  Bias shape: {self.linear.bias.shape}")
    
    def forward(self, x):
        """
        Define the forward pass.
        
        PyTorch automatically handles:
        - Tracking operations for backpropagation
        - Building the computational graph
        - Memory management
        """
        return self.linear(x)
    
    def get_parameters(self):
        """Extract parameters in a readable format."""
        return (
            self.linear.weight.data.flatten().tolist(),
            self.linear.bias.data.item()
        )


# =====================================================================
# PART 3: PyTorch's Data Loading Utilities
# =====================================================================

def create_data_loader(features, labels, batch_size=32, shuffle=True):
    """
    Create a DataLoader using PyTorch's utilities.
    
    Benefits over manual batching:
    - Automatic batching and shuffling
    - Memory-efficient data loading
    - Multi-processing support (not used here)
    - Consistent API across different data types
    """
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# =====================================================================
# PART 4: Training with PyTorch's Tools
# =====================================================================

def train_pytorch_model(model, train_loader, num_epochs=20, learning_rate=0.03):
    """
    Train the model using PyTorch's optimization tools.
    
    Key components:
    - Loss function: nn.MSELoss() computes mean squared error
    - Optimizer: optim.SGD handles parameter updates
    - Autograd: loss.backward() computes all gradients automatically
    """
    print("\n=== Training with PyTorch Tools ===")
    
    # Define loss function (criterion)
    # reduction='mean' averages the loss over batch
    criterion = nn.MSELoss(reduction='mean')
    
    # Define optimizer
    # It knows which parameters to update via model.parameters()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    loss_history = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            # Step 1: Zero gradients from previous iteration
            # This is necessary because gradients accumulate by default
            optimizer.zero_grad()
            
            # Step 2: Forward pass
            predictions = model(batch_X)
            
            # Step 3: Compute loss
            loss = criterion(predictions, batch_y)
            
            # Step 4: Backward pass (compute gradients)
            # This single line replaces all our manual gradient calculations!
            loss.backward()
            
            # Step 5: Update parameters
            # optimizer.step() applies: param = param - lr * param.grad
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            num_batches += 1
        
        # Average loss for epoch
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        # Print progress
        if epoch == 0 or (epoch + 1) % 5 == 0:
            w, b = model.get_parameters()
            print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:.6f}, "
                  f"w = [{w[0]:.3f}, {w[1]:.3f}], b = {b:.3f}")
    
    return loss_history


# =====================================================================
# PART 5: Advanced PyTorch Features
# =====================================================================

def demonstrate_advanced_features(model, features, labels):
    """
    Showcase additional PyTorch capabilities.
    """
    print("\n=== Advanced PyTorch Features ===")
    
    # 1. Gradient inspection
    print("\n1. Gradient Inspection:")
    sample_X = features[:5]
    sample_y = labels[:5]
    
    # Compute loss and gradients
    predictions = model(sample_X)
    loss = nn.MSELoss()(predictions, sample_y)
    loss.backward()
    
    print(f"   Weight gradients shape: {model.linear.weight.grad.shape}")
    print(f"   Bias gradient: {model.linear.bias.grad.item():.6f}")
    
    # 2. Model evaluation mode
    print("\n2. Evaluation Mode:")
    model.eval()  # Switches to evaluation mode
    with torch.no_grad():  # Disables gradient computation
        test_loss = nn.MSELoss()(model(features), labels)
        print(f"   Test loss (no gradients): {test_loss.item():.6f}")
    model.train()  # Switch back to training mode
    
    # 3. Parameter access
    print("\n3. Parameter Access:")
    for name, param in model.named_parameters():
        print(f"   {name}: shape={param.shape}, requires_grad={param.requires_grad}")
    
    # 4. Different optimizers
    print("\n4. Alternative Optimizers:")
    print("   - SGD: Basic gradient descent")
    print("   - Adam: Adaptive learning rates")
    print("   - RMSprop: Root mean square propagation")
    print("   - LBFGS: Quasi-Newton method")


# =====================================================================
# PART 6: Main Execution
# =====================================================================

def main():
    """Complete workflow with PyTorch's high-level API."""
    
    # Create model
    print("\n=== Model Creation ===")
    model = LinearRegressionModel(input_size=2)
    
    # Create data loader
    train_loader = create_data_loader(features, labels, batch_size=10)
    
    # Train model
    loss_history = train_pytorch_model(
        model, train_loader, 
        num_epochs=20, learning_rate=0.03
    )
    
    # Results comparison
    print("\n=== Final Results ===")
    learned_w, learned_b = model.get_parameters()
    print(f"True parameters:    w = {TRUE_WEIGHTS.tolist()}, b = {TRUE_BIAS}")
    print(f"Learned parameters: w = {learned_w}, b = {learned_b:.3f}")
    
    # Demonstrate advanced features
    demonstrate_advanced_features(model, features, labels)
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Training loss
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss (PyTorch)')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Predictions vs True values
    plt.subplot(1, 2, 2)
    with torch.no_grad():
        predictions = model(features).numpy()
    plt.scatter(labels.numpy(), predictions, alpha=0.5, s=10)
    plt.plot([labels.min(), labels.max()], 
             [labels.min(), labels.max()], 'r--', linewidth=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('chapter_linear-regression/pytorch_results.png')
    print("\n\nResults visualization saved to 'pytorch_results.png'")
    
    # Model summary
    print("\n=== Model Summary ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")


# =====================================================================
# PART 7: Comparison with Scratch Implementation
# =====================================================================

"""
Key Differences from Scratch Implementation:

1. **Automatic Differentiation**:
   - Scratch: Manually computed gradients using calculus
   - PyTorch: loss.backward() computes all gradients automatically

2. **Parameter Management**:
   - Scratch: Manually tracked weights and biases
   - PyTorch: nn.Module handles parameter registration

3. **Optimization**:
   - Scratch: Manual parameter updates with learning rate
   - PyTorch: Optimizers handle updates with advanced algorithms

4. **Data Loading**:
   - Scratch: Simple generator function
   - PyTorch: DataLoader with batching, shuffling, and more

5. **Extensibility**:
   - Scratch: Need to modify code for each new feature
   - PyTorch: Modular design allows easy swapping of components

Benefits of PyTorch's Approach:
- Less error-prone (no manual gradient calculations)
- More efficient (optimized C++ backend)
- Easier to extend (add layers, change optimizers, etc.)
- Better debugging tools (gradient checking, hooks, etc.)
- GPU acceleration with minimal code changes

When to Use Each Approach:
- Scratch: Learning fundamentals, custom algorithms
- PyTorch: Production code, research, complex models
"""

if __name__ == "__main__":
    main() 