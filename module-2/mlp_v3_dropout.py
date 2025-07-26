"""
MLP v3: Dropout Regularization
==============================
Building on v2, we add dropout - a powerful regularization technique.

New concepts:
1. Dropout: randomly "dropping" neurons during training
2. Different behavior during training vs testing
3. Scaling activations to maintain expected values
4. Combining dropout with L2 regularization

Key insight: Dropout prevents co-adaptation of neurons by randomly
disabling them during training, forcing the network to be more robust.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Literal, Union

# Import from previous versions
from mlp_v1_basic import create_house_price_data

class MLPv3:
    """
    MLP with both L2 regularization and dropout.
    
    Dropout works by:
    1. During training: randomly set activations to 0 with probability p
    2. During testing: use all neurons but scale by (1-p)
    
    This is equivalent to training an ensemble of networks.
    """
    
    def __init__(self, 
                 layer_sizes: List[int],
                 learning_rate: float = 0.01,
                 l2_lambda: float = 0.01,
                 dropout_rate: Union[float, List[float]] = 0.5,  # NEW: Dropout probability
                 initialization: Literal['random', 'he', 'xavier'] = 'he',
                 random_seed: Optional[int] = None):
        """
        Initialize MLP with dropout.
        
        Args:
            layer_sizes: Network architecture
            learning_rate: Step size for gradient descent
            l2_lambda: L2 regularization strength
            dropout_rate: Probability of dropping a neuron.
                         Can be a single float (same for all layers)
                         or list of floats (one per hidden layer).
                         Common values: 0.2-0.5
            initialization: Weight initialization method
            random_seed: For reproducibility
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        
        # Process dropout rates
        if isinstance(dropout_rate, float):
            # Same dropout for all hidden layers (not input/output)
            self.dropout_rates = [0.0] + [dropout_rate] * (self.n_layers - 2) + [0.0]
        else:
            # Custom dropout per layer
            assert len(dropout_rate) == self.n_layers - 2, \
                "Must provide dropout rate for each hidden layer"
            self.dropout_rates = [0.0] + list(dropout_rate) + [0.0]
        
        # Training mode flag (important for dropout!)
        self.training = True
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(self.n_layers - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            
            if initialization == 'random':
                W = np.random.randn(n_in, n_out) * 0.01
            elif initialization == 'he':
                # With dropout, we might want to adjust initialization
                # Some papers suggest scaling by sqrt(1/(1-p))
                dropout_factor = 1.0 / (1.0 - self.dropout_rates[i] + 1e-8)
                W = np.random.randn(n_in, n_out) * np.sqrt(2.0 * dropout_factor / n_in)
            elif initialization == 'xavier':
                W = np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)
            
            b = np.zeros((1, n_out))
            
            self.weights.append(W)
            self.biases.append(b)
        
        # Storage for forward pass
        self.activations = []
        self.z_values = []
        self.dropout_masks = []  # NEW: Store which neurons were dropped
        
        # Training history
        self.train_losses = []
        self.train_losses_unreg = []
        self.val_losses = []
    
    def set_training_mode(self, training: bool):
        """Switch between training and evaluation mode"""
        self.training = training
    
    def relu(self, z: np.ndarray) -> np.ndarray:
        """ReLU activation"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z: np.ndarray) -> np.ndarray:
        """ReLU derivative"""
        return (z > 0).astype(float)
    
    def apply_dropout(self, activation: np.ndarray, dropout_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply dropout to activations.
        
        During training:
        - Create binary mask (1 = keep, 0 = drop)
        - Apply mask and scale by 1/(1-p) to maintain expected value
        
        During testing:
        - No dropout, activations pass through unchanged
        
        Returns:
            Tuple of (dropped_activation, dropout_mask)
        """
        if not self.training or dropout_rate == 0:
            # No dropout during evaluation or if rate is 0
            return activation, np.ones_like(activation)
        
        # Create dropout mask
        # np.random.binomial returns 1 with probability (1-dropout_rate)
        keep_prob = 1 - dropout_rate
        mask = np.random.binomial(1, keep_prob, size=activation.shape)
        
        # Apply mask and scale
        # The scaling ensures E[dropped_activation] = E[activation]
        dropped_activation = activation * mask / keep_prob
        
        return dropped_activation, mask
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation with dropout.
        
        Dropout is applied AFTER activation function.
        """
        self.activations = [X]
        self.z_values = []
        self.dropout_masks = []
        
        a = X
        
        for i in range(self.n_layers - 1):
            # Linear transformation
            z = a @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            
            # Activation
            if i < self.n_layers - 2:  # Hidden layer
                a = self.relu(z)
                
                # Apply dropout to hidden layer activations
                a, mask = self.apply_dropout(a, self.dropout_rates[i + 1])
                self.dropout_masks.append(mask)
            else:  # Output layer
                a = z
                self.dropout_masks.append(np.ones_like(a))  # No dropout on output
            
            self.activations.append(a)
        
        return a
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray, 
                     include_regularization: bool = True) -> float:
        """Compute loss with L2 regularization (same as v2)"""
        n_samples = y_true.shape[0]
        
        mse_loss = np.mean((y_pred - y_true) ** 2)
        
        if not include_regularization or self.l2_lambda == 0:
            return mse_loss
        
        l2_penalty = 0
        for W in self.weights:
            l2_penalty += np.sum(W ** 2)
        
        total_loss = mse_loss + (self.l2_lambda / 2) * l2_penalty
        
        return total_loss
    
    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Backward propagation with dropout.
        
        Key: We need to apply the same dropout mask during backprop
        that was used during forward prop.
        """
        n_samples = X.shape[0]
        
        # Forward pass (will apply dropout if in training mode)
        y_pred = self.forward(X)
        
        # Output layer gradient
        delta = 2 * (y_pred - y) / n_samples
        
        # Backpropagate through layers
        for i in reversed(range(self.n_layers - 1)):
            # Gradient w.r.t. weights
            grad_W = self.activations[i].T @ delta
            
            # Add L2 regularization gradient
            grad_W += self.l2_lambda * self.weights[i]
            
            # Gradient w.r.t. biases
            grad_b = np.sum(delta, axis=0, keepdims=True)
            
            # Update parameters
            self.weights[i] -= self.learning_rate * grad_W
            self.biases[i] -= self.learning_rate * grad_b
            
            # Propagate error to previous layer
            if i > 0:
                # First propagate through weights
                delta = delta @ self.weights[i].T
                
                # Apply dropout mask from forward pass
                # This is crucial: we drop the same neurons in backward pass
                dropout_rate = self.dropout_rates[i]
                if self.training and dropout_rate > 0:
                    delta *= self.dropout_masks[i - 1] / (1 - dropout_rate)
                
                # Then apply activation derivative
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
        Train the network with dropout and L2 regularization.
        
        Important: Set training mode for dropout during training,
        and evaluation mode for validation loss computation.
        """
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_val is not None and y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_weights = None
        best_biases = None
        
        for epoch in range(epochs):
            # TRAINING MODE for parameter updates
            self.set_training_mode(True)
            
            # Training step
            self.backward(X_train, y_train)
            
            # Compute training loss (with dropout for logging)
            train_pred = self.forward(X_train)
            train_loss_reg = self.compute_loss(train_pred, y_train, include_regularization=True)
            train_loss_unreg = self.compute_loss(train_pred, y_train, include_regularization=False)
            
            self.train_losses.append(train_loss_reg)
            self.train_losses_unreg.append(train_loss_unreg)
            
            # EVALUATION MODE for validation
            self.set_training_mode(False)
            
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.compute_loss(val_pred, y_val, include_regularization=False)
                self.val_losses.append(val_loss)
                
                # Early stopping
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
            
            if verbose and (epoch + 1) % 100 == 0:
                msg = f"Epoch {epoch + 1}/{epochs}"
                msg += f", Train Loss: {train_loss_unreg:.4f}"
                if X_val is not None:
                    msg += f", Val Loss: {val_loss:.4f}"
                print(msg)
        
        # Ensure we're in evaluation mode after training
        self.set_training_mode(False)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions (ensure evaluation mode)"""
        # Store current mode
        was_training = self.training
        
        # Switch to evaluation mode for predictions
        self.set_training_mode(False)
        predictions = self.forward(X)
        
        # Restore previous mode
        self.set_training_mode(was_training)
        
        return predictions
    
    def plot_losses(self) -> None:
        """Plot training curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses_unreg, label='Training Loss', linewidth=2)
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title(f'Training Progress (Dropout rates: {self.dropout_rates[1:-1]})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def monte_carlo_predict(self, X: np.ndarray, n_samples: int = 100) -> dict:
        """
        Monte Carlo Dropout: Make predictions with dropout enabled.
        
        This gives us uncertainty estimates by running multiple
        forward passes with different dropout masks.
        
        Returns:
            Dictionary with 'mean', 'std', and 'samples' of predictions
        """
        # Enable training mode for dropout
        self.set_training_mode(True)
        
        predictions = []
        for _ in range(n_samples):
            pred = self.forward(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Restore evaluation mode
        self.set_training_mode(False)
        
        return {
            'mean': np.mean(predictions, axis=0),
            'std': np.std(predictions, axis=0),
            'samples': predictions
        }


def compare_dropout_rates(X_train, y_train, X_val, y_val):
    """
    Compare different dropout rates to find optimal value.
    
    Too little dropout: minimal regularization effect
    Too much dropout: underfitting, slow convergence
    """
    dropout_rates = [0, 0.2, 0.5, 0.7]
    results = {}
    
    for dropout in dropout_rates:
        print(f"\nTraining with dropout rate = {dropout}")
        print("-" * 40)
        
        model = MLPv3(
            layer_sizes=[13, 64, 32, 1],
            learning_rate=0.01,
            l2_lambda=0.01,
            dropout_rate=dropout,
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
        
        results[dropout] = {
            'model': model,
            'val_loss': val_loss,
            'n_epochs': len(model.train_losses)
        }
        
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Training stopped at epoch: {results[dropout]['n_epochs']}")
    
    return results


def demonstrate_uncertainty_estimation(model, X_test, y_test):
    """
    Show how Monte Carlo Dropout provides uncertainty estimates.
    
    This is particularly useful for:
    1. Identifying when the model is uncertain
    2. Active learning (query points with high uncertainty)
    3. Rejecting predictions when uncertainty is too high
    """
    # Get MC Dropout predictions
    mc_results = model.monte_carlo_predict(X_test[:20], n_samples=100)
    
    # Standard predictions (no dropout)
    standard_pred = model.predict(X_test[:20])
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    x_axis = np.arange(20)
    
    # Plot with uncertainty bands
    plt.errorbar(x_axis, mc_results['mean'].flatten(), 
                 yerr=2*mc_results['std'].flatten(),  # 95% confidence
                 fmt='o', capsize=5, label='MC Dropout (±2σ)')
    
    plt.plot(x_axis, standard_pred.flatten(), 'rs', 
             label='Standard Prediction', markersize=8)
    plt.plot(x_axis, y_test[:20].flatten(), 'g^', 
             label='True Value', markersize=8)
    
    plt.xlabel('Test Sample')
    plt.ylabel('Predicted Value (normalized)')
    plt.title('Prediction Uncertainty using Monte Carlo Dropout')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print uncertainty statistics
    avg_uncertainty = np.mean(mc_results['std'])
    print(f"\nAverage prediction uncertainty (std): {avg_uncertainty:.4f}")
    
    # Identify most/least certain predictions
    uncertainties = mc_results['std'].flatten()
    most_certain_idx = np.argmin(uncertainties)
    least_certain_idx = np.argmax(uncertainties)
    
    print(f"Most certain prediction: Sample {most_certain_idx}, std={uncertainties[most_certain_idx]:.4f}")
    print(f"Least certain prediction: Sample {least_certain_idx}, std={uncertainties[least_certain_idx]:.4f}")


# Example usage
if __name__ == "__main__":
    # Generate data
    X, y = create_house_price_data(n_samples=1000, noise_level=0.1)
    
    # Split and normalize
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
    
    # Compare dropout rates
    print("=" * 60)
    print("COMPARING DROPOUT RATES")
    print("=" * 60)
    
    results = compare_dropout_rates(X_train, y_train, X_val, y_val)
    
    # Visualize comparison
    plt.figure(figsize=(10, 6))
    dropout_values = list(results.keys())
    val_losses = [results[d]['val_loss'] for d in dropout_values]
    plt.bar([str(d) for d in dropout_values], val_losses)
    plt.xlabel('Dropout Rate')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs Dropout Rate')
    plt.show()
    
    # Demonstrate uncertainty estimation with best model
    print("\n" + "=" * 60)
    print("UNCERTAINTY ESTIMATION WITH MONTE CARLO DROPOUT")
    print("=" * 60)
    
    best_dropout = min(results.keys(), key=lambda x: results[x]['val_loss'])
    best_model = results[best_dropout]['model']
    
    demonstrate_uncertainty_estimation(best_model, X_val, y_val)
    
    # Show training curves for best model
    print(f"\nTraining curves for best dropout rate ({best_dropout}):")
    best_model.plot_losses()
