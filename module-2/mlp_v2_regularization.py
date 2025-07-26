"""
MLP v2: L2 Regularization
========================
Building on v1, we add L2 (weight decay) regularization to prevent overfitting.

New concepts:
1. L2 regularization (weight decay)
2. Regularization strength tuning
3. Impact on gradients and weight updates

L2 regularization adds a penalty term to the loss:
    L_total = L_data + λ/2 * Σ(W²)
    
This encourages smaller weights, which typically leads to:
- Smoother decision boundaries
- Better generalization
- Reduced overfitting
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Literal

# Import the base class from v1
from mlp_v1_basic import create_house_price_data

class MLPv2:
    """
    MLP with L2 regularization.
    
    The key insight: L2 regularization penalizes large weights,
    encouraging the network to find simpler solutions that generalize better.
    """
    
    def __init__(self, 
                 layer_sizes: List[int],
                 learning_rate: float = 0.01,
                 l2_lambda: float = 0.01,  # NEW: Regularization strength
                 initialization: Literal['random', 'he', 'xavier'] = 'he',
                 random_seed: Optional[int] = None):
        """
        Initialize the MLP with L2 regularization.
        
        Args:
            layer_sizes: Network architecture
            learning_rate: Step size for gradient descent
            l2_lambda: L2 regularization strength (0 = no regularization)
                      Typical values: 0.001 to 0.1
            initialization: Weight initialization method
            random_seed: For reproducibility
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda  # NEW: Store regularization strength
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize weights and biases (same as v1)
        self.weights = []
        self.biases = []
        
        for i in range(self.n_layers - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            
            if initialization == 'random':
                W = np.random.randn(n_in, n_out) * 0.01
            elif initialization == 'he':
                W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            elif initialization == 'xavier':
                W = np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)
            
            b = np.zeros((1, n_out))
            
            self.weights.append(W)
            self.biases.append(b)
        
        # Storage for forward pass
        self.activations = []
        self.z_values = []
        
        # Training history - now including regularization loss
        self.train_losses = []
        self.train_losses_unreg = []  # NEW: Track unregularized loss too
        self.val_losses = []
    
    def relu(self, z: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU"""
        return (z > 0).astype(float)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward propagation (same as v1)"""
        self.activations = [X]
        self.z_values = []
        
        a = X
        
        for i in range(self.n_layers - 1):
            z = a @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            
            if i < self.n_layers - 2:  # Hidden layer
                a = self.relu(z)
            else:  # Output layer
                a = z
            
            self.activations.append(a)
        
        return a
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray, 
                     include_regularization: bool = True) -> float:
        """
        Compute loss with optional L2 regularization.
        
        Total loss = MSE + (λ/2) * Σ(W²)
        
        The factor of 1/2 is conventional and makes derivatives cleaner.
        """
        n_samples = y_true.shape[0]
        
        # Data loss (MSE)
        mse_loss = np.mean((y_pred - y_true) ** 2)
        
        if not include_regularization or self.l2_lambda == 0:
            return mse_loss
        
        # L2 regularization term
        # Sum of squared weights across all layers
        l2_penalty = 0
        for W in self.weights:
            l2_penalty += np.sum(W ** 2)
        
        # Total loss
        total_loss = mse_loss + (self.l2_lambda / 2) * l2_penalty
        
        return total_loss
    
    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Backward propagation with L2 regularization.
        
        The gradient of L2 regularization w.r.t. weights is simply λ*W.
        This gets added to the standard gradient.
        """
        n_samples = X.shape[0]
        
        # Forward pass
        y_pred = self.forward(X)
        
        # Output layer gradient (same as v1)
        delta = 2 * (y_pred - y) / n_samples
        
        # Backpropagate through layers
        for i in reversed(range(self.n_layers - 1)):
            # Standard gradient
            grad_W = self.activations[i].T @ delta
            
            # ADD L2 regularization gradient
            # dL/dW = dL_data/dW + λ*W
            grad_W += self.l2_lambda * self.weights[i]
            
            # Bias gradient (no regularization on biases - common practice)
            grad_b = np.sum(delta, axis=0, keepdims=True)
            
            # Update parameters
            self.weights[i] -= self.learning_rate * grad_W
            self.biases[i] -= self.learning_rate * grad_b
            
            # Propagate error to previous layer
            if i > 0:
                delta = delta @ self.weights[i].T
                delta *= self.relu_derivative(self.z_values[i - 1])
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 1000,
            patience: int = 50,
            verbose: bool = True) -> None:
        """
        Train the network with L2 regularization.
        
        Note: We monitor validation loss WITHOUT regularization for early stopping,
        as we care about actual prediction performance, not the regularized objective.
        """
        # Ensure correct shapes
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_val is not None and y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)
        
        # Early stopping setup
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_weights = None
        best_biases = None
        
        for epoch in range(epochs):
            # Training step
            self.backward(X_train, y_train)
            
            # Compute losses
            train_pred = self.forward(X_train)
            
            # Track both regularized and unregularized training loss
            train_loss_reg = self.compute_loss(train_pred, y_train, include_regularization=True)
            train_loss_unreg = self.compute_loss(train_pred, y_train, include_regularization=False)
            
            self.train_losses.append(train_loss_reg)
            self.train_losses_unreg.append(train_loss_unreg)
            
            # Validation loss (without regularization for fair comparison)
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.compute_loss(val_pred, y_val, include_regularization=False)
                self.val_losses.append(val_loss)
                
                # Early stopping based on validation performance
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                else:
                    epochs_without_improvement += 1
                
                if epochs_without_improvement >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    self.weights = best_weights
                    self.biases = best_biases
                    break
            
            # Progress updates
            if verbose and (epoch + 1) % 100 == 0:
                msg = f"Epoch {epoch + 1}/{epochs}"
                msg += f", Train Loss (reg): {train_loss_reg:.4f}"
                msg += f", Train Loss: {train_loss_unreg:.4f}"
                if X_val is not None:
                    msg += f", Val Loss: {val_loss:.4f}"
                print(msg)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.forward(X)
    
    def plot_losses(self) -> None:
        """Enhanced plot showing regularization effect"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Training progress
        ax1.plot(self.train_losses, label='Train Loss (with L2)', linewidth=2)
        ax1.plot(self.train_losses_unreg, label='Train Loss (no L2)', 
                 linewidth=2, linestyle='--')
        if self.val_losses:
            ax1.plot(self.val_losses, label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Regularization gap
        reg_gap = np.array(self.train_losses) - np.array(self.train_losses_unreg)
        ax2.plot(reg_gap, linewidth=2, color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Regularization Penalty')
        ax2.set_title(f'L2 Penalty Over Time (λ={self.l2_lambda})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_weight_statistics(self) -> dict:
        """
        Analyze weight magnitudes to see regularization effect.
        
        L2 regularization should lead to smaller weight magnitudes.
        """
        stats = {}
        for i, W in enumerate(self.weights):
            stats[f'layer_{i}'] = {
                'mean': np.mean(np.abs(W)),
                'std': np.std(W),
                'max': np.max(np.abs(W)),
                'frobenius_norm': np.linalg.norm(W, 'fro')
            }
        return stats


def compare_regularization_strengths(X_train, y_train, X_val, y_val):
    """
    Compare different L2 regularization strengths.
    
    This demonstrates how to tune the regularization hyperparameter.
    """
    l2_values = [0, 0.001, 0.01, 0.1]
    results = {}
    
    for l2 in l2_values:
        print(f"\nTraining with L2 lambda = {l2}")
        print("-" * 40)
        
        model = MLPv2(
            layer_sizes=[13, 64, 32, 1],
            learning_rate=0.01,
            l2_lambda=l2,
            initialization='he',
            random_seed=42
        )
        
        model.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=1000,
            patience=50,
            verbose=False
        )
        
        # Evaluate
        val_pred = model.predict(X_val)
        val_loss = model.compute_loss(val_pred, y_val.reshape(-1, 1), 
                                     include_regularization=False)
        
        # Get weight statistics
        weight_stats = model.get_weight_statistics()
        avg_weight_magnitude = np.mean([stats['mean'] 
                                      for stats in weight_stats.values()])
        
        results[l2] = {
            'model': model,
            'val_loss': val_loss,
            'avg_weight_magnitude': avg_weight_magnitude,
            'n_epochs': len(model.train_losses)
        }
        
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Avg Weight Magnitude: {avg_weight_magnitude:.4f}")
        print(f"Training stopped at epoch: {results[l2]['n_epochs']}")
    
    return results


# Example usage
if __name__ == "__main__":
    # Generate data
    X, y = create_house_price_data(n_samples=1000, noise_level=0.1)
    
    # Split and normalize (same as v1)
    n_train = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / (std + 1e-8)
    X_val = (X_val - mean) / (std + 1e-8)
    
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    
    # Compare regularization strengths
    print("=" * 60)
    print("COMPARING L2 REGULARIZATION STRENGTHS")
    print("=" * 60)
    
    results = compare_regularization_strengths(X_train, y_train, X_val, y_val)
    
    # Visualize comparison
    plt.figure(figsize=(12, 5))
    
    # Plot validation losses
    plt.subplot(1, 2, 1)
    l2_values = list(results.keys())
    val_losses = [results[l2]['val_loss'] for l2 in l2_values]
    plt.bar([str(l2) for l2 in l2_values], val_losses)
    plt.xlabel('L2 Lambda')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs Regularization Strength')
    
    # Plot weight magnitudes
    plt.subplot(1, 2, 2)
    weight_mags = [results[l2]['avg_weight_magnitude'] for l2 in l2_values]
    plt.bar([str(l2) for l2 in l2_values], weight_mags)
    plt.xlabel('L2 Lambda')
    plt.ylabel('Average Weight Magnitude')
    plt.title('Weight Magnitude vs Regularization Strength')
    
    plt.tight_layout()
    plt.show()
    
    # Show detailed training curves for best model
    best_l2 = min(results.keys(), key=lambda x: results[x]['val_loss'])
    print(f"\nBest L2 value: {best_l2}")
    print("Showing training curves for best model:")
    results[best_l2]['model'].plot_losses()
