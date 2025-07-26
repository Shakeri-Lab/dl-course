#!/usr/bin/env python3
"""
Linear Regression with Weight Decay (L2 Regularization)

This script demonstrates the importance of weight decay (L2 regularization)
in high-dimensional linear regression, showing how it prevents overfitting
when we have many features but few training examples.

Key Concepts:
1. Overfitting in high dimensions
2. L2 regularization (weight decay)
3. Effect of regularization strength (lambda)
4. Comparison of regularized vs unregularized models

Author: Weight Decay Teaching Version
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# =====================================================================
# PART 1: High-Dimensional Synthetic Data Generation
# =====================================================================

def generate_high_dim_data(num_features, num_train, num_val, noise_std=0.01):
    """
    Generate high-dimensional synthetic data where overfitting is likely.
    
    True model: y = 0.05 + sum(0.01 * x_i) + noise
    
    Args:
        num_features: Number of input features (dimensionality)
        num_train: Number of training samples
        num_val: Number of validation samples
        noise_std: Standard deviation of Gaussian noise
        
    Returns:
        Dictionary with train/val features and labels
    """
    # Total samples
    n = num_train + num_val
    
    # Generate features from standard normal
    X = torch.randn(n, num_features)
    
    # True parameters: all weights = 0.01, bias = 0.05
    true_w = torch.ones(num_features, 1) * 0.01
    true_b = 0.05
    
    # Generate labels: y = X @ w + b + noise
    noise = torch.randn(n, 1) * noise_std
    y = X @ true_w + true_b + noise
    
    # Split into train and validation
    X_train, y_train = X[:num_train], y[:num_train]
    X_val, y_val = X[num_train:], y[num_train:]
    
    print(f"Data generated:")
    print(f"  Features dimension: {num_features}")
    print(f"  Training samples: {num_train}")
    print(f"  Validation samples: {num_val}")
    print(f"  True weights: all {true_w[0].item()}")
    print(f"  True bias: {true_b}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'true_w': true_w, 'true_b': true_b
    }


# =====================================================================
# PART 2: Model Definition with Optional Weight Decay
# =====================================================================

class LinearRegressionWithDecay(nn.Module):
    """
    Linear regression model that supports weight decay in the loss.
    
    The total loss is: MSE_loss + (lambda/2) * ||w||^2
    """
    
    def __init__(self, input_size, weight_decay=0.0):
        """
        Args:
            input_size: Number of input features
            weight_decay: Lambda parameter for L2 regularization
        """
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        self.weight_decay = weight_decay
        
    def forward(self, x):
        return self.linear(x)
    
    def l2_penalty(self):
        """Compute L2 penalty: (lambda/2) * ||w||^2"""
        return self.weight_decay * torch.sum(self.linear.weight ** 2) / 2
    
    def get_weights_stats(self):
        """Get statistics about the model weights."""
        weights = self.linear.weight.data.flatten()
        return {
            'mean': weights.mean().item(),
            'std': weights.std().item(),
            'norm': torch.norm(weights).item(),
            'max': weights.max().item(),
            'min': weights.min().item()
        }


# =====================================================================
# PART 3: Training with Weight Decay
# =====================================================================

def train_with_weight_decay(model, train_loader, val_loader, 
                           num_epochs=100, learning_rate=0.1):
    """
    Train model with weight decay included in the loss.
    
    Returns:
        Dictionary with training history
    """
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'weight_norm': []
    }
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(X_batch)
            
            # Compute total loss: MSE + L2 penalty
            mse = mse_loss(predictions, y_batch)
            total_loss = mse + model.l2_penalty()
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            train_loss += mse.item()  # Track MSE only
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                val_loss += mse_loss(predictions, y_batch).item()
        
        # Record history
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['weight_norm'].append(torch.norm(model.linear.weight).item())
        
        # Print progress
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: "
                  f"Train Loss = {history['train_loss'][-1]:.4f}, "
                  f"Val Loss = {history['val_loss'][-1]:.4f}, "
                  f"||w|| = {history['weight_norm'][-1]:.2f}")
    
    return history


# =====================================================================
# PART 4: Experiment - Compare Different Weight Decay Values
# =====================================================================

def run_weight_decay_experiment(data, lambdas=[0, 0.001, 0.01, 0.1, 1.0]):
    """
    Train models with different weight decay values and compare results.
    """
    # Create data loaders
    batch_size = min(10, len(data['X_train']))
    train_dataset = TensorDataset(data['X_train'], data['y_train'])
    val_dataset = TensorDataset(data['X_val'], data['y_val'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    results = {}
    
    print("\n=== Training Models with Different Weight Decay Values ===")
    
    for lam in lambdas:
        print(f"\nTraining with lambda = {lam}")
        
        # Create model
        model = LinearRegressionWithDecay(
            input_size=data['X_train'].shape[1],
            weight_decay=lam
        )
        
        # Train
        history = train_with_weight_decay(
            model, train_loader, val_loader,
            num_epochs=100, learning_rate=0.1
        )
        
        # Get final stats
        final_stats = model.get_weights_stats()
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1]
        
        print(f"Final results for lambda = {lam}:")
        print(f"  Train Loss: {final_train_loss:.4f}")
        print(f"  Val Loss: {final_val_loss:.4f}")
        print(f"  Weight norm: {final_stats['norm']:.3f}")
        
        results[lam] = {
            'model': model,
            'history': history,
            'final_stats': final_stats
        }
    
    return results


# =====================================================================
# PART 5: Visualization
# =====================================================================

def plot_results(results, data):
    """Create comprehensive plots showing the effect of weight decay."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Training and Validation Loss Over Time
    ax1 = axes[0, 0]
    for lam, res in results.items():
        epochs = range(1, len(res['history']['train_loss']) + 1)
        ax1.plot(epochs, res['history']['train_loss'], 
                linestyle='--', label=f'λ={lam} (train)')
        ax1.plot(epochs, res['history']['val_loss'], 
                linestyle='-', label=f'λ={lam} (val)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Weight Norm Evolution
    ax2 = axes[0, 1]
    for lam, res in results.items():
        epochs = range(1, len(res['history']['weight_norm']) + 1)
        ax2.plot(epochs, res['history']['weight_norm'], label=f'λ={lam}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('||w||₂')
    ax2.set_title('Weight Norm During Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final Validation Loss vs Lambda
    ax3 = axes[1, 0]
    lambdas = list(results.keys())
    final_val_losses = [res['history']['val_loss'][-1] for res in results.values()]
    ax3.semilogx(lambdas[1:], final_val_losses[1:], 'bo-', markersize=8)  # Skip λ=0
    ax3.axhline(y=final_val_losses[0], color='r', linestyle='--', 
                label=f'No regularization (λ=0): {final_val_losses[0]:.4f}')
    ax3.set_xlabel('Weight Decay (λ)')
    ax3.set_ylabel('Final Validation Loss')
    ax3.set_title('Validation Loss vs Regularization Strength')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Weight Distribution
    ax4 = axes[1, 1]
    selected_lambdas = [0, 0.01, 1.0]  # Show a subset
    for lam in selected_lambdas:
        if lam in results:
            weights = results[lam]['model'].linear.weight.data.flatten().numpy()
            ax4.hist(weights, bins=30, alpha=0.5, label=f'λ={lam}', density=True)
    ax4.axvline(x=0.01, color='k', linestyle='--', label='True weight')
    ax4.set_xlabel('Weight Value')
    ax4.set_ylabel('Density')
    ax4.set_title('Weight Distribution for Different λ')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('chapter_linear-regression/weight_decay_analysis.png', dpi=150)
    print("\nPlots saved to 'weight_decay_analysis.png'")
    
    # Print summary
    print("\n=== Summary of Results ===")
    print(f"Dataset: {data['X_train'].shape[1]} features, "
          f"{len(data['X_train'])} training samples")
    print("\nOptimal λ based on validation loss:")
    best_lambda = min(results.keys(), 
                     key=lambda k: results[k]['history']['val_loss'][-1])
    print(f"  Best λ = {best_lambda}")
    print(f"  Best validation loss = {results[best_lambda]['history']['val_loss'][-1]:.4f}")
    
    # Compare with no regularization
    no_reg_val = results[0]['history']['val_loss'][-1]
    best_val = results[best_lambda]['history']['val_loss'][-1]
    improvement = (no_reg_val - best_val) / no_reg_val * 100
    print(f"  Improvement over no regularization: {improvement:.1f}%")


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    """Run the complete weight decay demonstration."""
    
    print("=== Weight Decay in High-Dimensional Linear Regression ===\n")
    
    # Generate high-dimensional data with limited samples
    # This creates a scenario prone to overfitting
    data = generate_high_dim_data(
        num_features=200,    # High dimension
        num_train=20,        # Very few training samples
        num_val=100,         # More validation samples
        noise_std=0.01
    )
    
    # Run experiment with different weight decay values
    lambdas = [0, 0.001, 0.01, 0.1, 1.0, 10.0]
    results = run_weight_decay_experiment(data, lambdas)
    
    # Visualize results
    plot_results(results, data)
    
    print("\n" + "="*60)
    print("Key Insights:")
    print("1. Without regularization (λ=0), the model overfits badly")
    print("2. Moderate weight decay improves generalization")
    print("3. Too much weight decay underfits (high bias)")
    print("4. The optimal λ depends on the data and noise level")
    print("5. Weight decay shrinks weights toward zero")
    print("="*60)


# =====================================================================
# ASSIGNMENT VERSION
# =====================================================================

def create_assignment_version():
    """Create a version with TODO sections for students."""
    
    assignment_code = '''#!/usr/bin/env python3
"""
ASSIGNMENT: Linear Regression with Weight Decay

Complete the TODO sections to implement weight decay (L2 regularization)
and analyze its effect on high-dimensional linear regression.

Learning Objectives:
1. Implement L2 penalty calculation
2. Modify training loop to include weight decay
3. Analyze the effect of regularization strength
4. Find optimal weight decay value using validation data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)
np.random.seed(42)


def generate_high_dim_data(num_features, num_train, num_val, noise_std=0.01):
    """Generate high-dimensional synthetic data."""
    n = num_train + num_val
    X = torch.randn(n, num_features)
    
    # TODO 1: Create true parameters
    # All weights should be 0.01, bias should be 0.05
    true_w = None  # TODO: Initialize weights
    true_b = None  # TODO: Initialize bias
    
    # TODO 2: Generate labels using linear model + noise
    # y = X @ w + b + noise
    y = None  # TODO: Generate labels
    
    # Split data
    X_train, y_train = X[:num_train], y[:num_train]
    X_val, y_val = X[num_train:], y[num_train:]
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'true_w': true_w, 'true_b': true_b
    }


class LinearRegressionWithDecay(nn.Module):
    def __init__(self, input_size, weight_decay=0.0):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        self.weight_decay = weight_decay
    
    def forward(self, x):
        return self.linear(x)
    
    def l2_penalty(self):
        """
        TODO 3: Implement L2 penalty calculation
        Return (lambda/2) * sum of squared weights
        Note: Only penalize weights, not bias
        """
        # TODO: Calculate and return L2 penalty
        pass


def train_with_weight_decay(model, train_loader, val_loader, 
                           num_epochs=100, learning_rate=0.1):
    """Train model with weight decay."""
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': [], 'weight_norm': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            predictions = model(X_batch)
            mse = mse_loss(predictions, y_batch)
            
            # TODO 4: Compute total loss including L2 penalty
            total_loss = None  # TODO: mse + weight_decay_penalty
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += mse.item()
        
        # TODO 5: Implement validation loop
        # Calculate validation loss (MSE only, no penalty)
        model.eval()
        val_loss = 0
        # TODO: Complete validation loop
        
        # Record history
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['weight_norm'].append(torch.norm(model.linear.weight).item())
    
    return history


def analyze_weight_decay():
    """
    TODO 6: Complete the analysis
    1. Generate high-dimensional data (200 features, 20 train, 100 val)
    2. Train models with different lambda values: [0, 0.001, 0.01, 0.1, 1.0]
    3. Plot validation loss vs lambda
    4. Find and report the optimal lambda value
    """
    # TODO: Implement the complete analysis
    pass


# TODO 7: Answer these questions in comments:
# Q1: Why does the model overfit without weight decay when we have many features but few samples?
# A1: 

# Q2: How does weight decay help prevent overfitting?
# A2: 

# Q3: What happens if lambda is too large?
# A3: 

# Q4: How would you choose the optimal lambda value in practice?
# A4: 


if __name__ == "__main__":
    # TODO 8: Run your analysis
    pass
'''
    
    # Save assignment version
    with open('chapter_linear-regression/linear_regression_weight_decay_assignment.py', 'w') as f:
        f.write(assignment_code)
    
    print("\nAssignment version created: linear_regression_weight_decay_assignment.py")


if __name__ == "__main__":
    main()
    create_assignment_version() 