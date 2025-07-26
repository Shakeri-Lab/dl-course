"""
MLP v1: Basic Implementation
============================
This is our foundational MLP implementation with:
- ReLU activation function
- Different weight initialization methods (Random, He, Xavier)
- Early stopping
- Basic forward and backward propagation

Key concepts covered:
1. Neural network architecture
2. Weight initialization strategies
3. Forward propagation
4. Backpropagation with gradient descent
5. Early stopping for preventing overfitting
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Literal

class MLPv1:
    """
    Basic Multi-Layer Perceptron with ReLU activation.
    
    This implementation focuses on the core concepts of neural networks:
    - Layer-wise computation
    - Different initialization strategies
    - Gradient-based learning
    """
    
    def __init__(self, 
                 layer_sizes: List[int],
                 learning_rate: float = 0.01,
                 initialization: Literal['random', 'he', 'xavier'] = 'he',
                 random_seed: Optional[int] = None):
        """
        Initialize the MLP.
        
        Args:
            layer_sizes: List of integers defining the network architecture.
                        E.g., [13, 64, 32, 1] for 13 inputs, two hidden layers, 1 output
            learning_rate: Step size for gradient descent
            initialization: Weight initialization method
                - 'random': Small random values
                - 'he': He initialization (good for ReLU)
                - 'xavier': Xavier/Glorot initialization
            random_seed: For reproducibility
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(self.n_layers - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            
            # Weight initialization based on chosen method
            if initialization == 'random':
                # Small random values centered around 0
                W = np.random.randn(n_in, n_out) * 0.01
            elif initialization == 'he':
                # He initialization: std = sqrt(2/n_in)
                # Best for ReLU activations as it accounts for the fact that
                # ReLU zeros out negative values
                W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            elif initialization == 'xavier':
                # Xavier initialization: std = sqrt(1/n_in)
                # Good for tanh/sigmoid, okay for ReLU
                W = np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)
            else:
                raise ValueError(f"Unknown initialization: {initialization}")
            
            # Biases are typically initialized to zero
            b = np.zeros((1, n_out))
            
            self.weights.append(W)
            self.biases.append(b)
        
        # Storage for activations during forward pass (needed for backprop)
        self.activations = []
        self.z_values = []  # Pre-activation values
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def relu(self, z: np.ndarray) -> np.ndarray:
        """ReLU activation function: max(0, z)"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU: 1 if z > 0, else 0"""
        return (z > 0).astype(float)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through the network.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Output predictions of shape (n_samples, n_outputs)
        """
        # Clear previous activations
        self.activations = [X]
        self.z_values = []
        
        # Current activation starts as input
        a = X
        
        # Propagate through each layer
        for i in range(self.n_layers - 1):
            # Linear transformation: z = a @ W + b
            z = a @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            
            # Apply activation function (ReLU for hidden layers, none for output)
            if i < self.n_layers - 2:  # Hidden layer
                a = self.relu(z)
            else:  # Output layer (regression - no activation)
                a = z
            
            self.activations.append(a)
        
        return a
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute Mean Squared Error loss for regression.
        
        MSE = (1/n) * Σ(y_pred - y_true)²
        """
        n_samples = y_true.shape[0]
        loss = np.mean((y_pred - y_true) ** 2)
        return loss
    
    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Backward propagation to compute gradients.
        
        This implements the chain rule to compute gradients of the loss
        with respect to all weights and biases.
        """
        n_samples = X.shape[0]
        
        # First, get predictions via forward pass
        y_pred = self.forward(X)
        
        # Initialize gradient of loss w.r.t. output
        # For MSE loss: dL/dy_pred = (2/n) * (y_pred - y)
        delta = 2 * (y_pred - y) / n_samples
        
        # Backpropagate through layers in reverse order
        for i in reversed(range(self.n_layers - 1)):
            # Gradient w.r.t. weights: dL/dW = a^T @ delta
            # where a is the activation from the previous layer
            grad_W = self.activations[i].T @ delta
            
            # Gradient w.r.t. biases: dL/db = sum(delta) over samples
            grad_b = np.sum(delta, axis=0, keepdims=True)
            
            # Update weights and biases using gradient descent
            self.weights[i] -= self.learning_rate * grad_W
            self.biases[i] -= self.learning_rate * grad_b
            
            # Propagate error to previous layer (if not input layer)
            if i > 0:
                # Gradient w.r.t. previous activation
                delta = delta @ self.weights[i].T
                # Apply derivative of activation function
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
        Train the network using gradient descent with optional early stopping.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (for early stopping)
            y_val: Validation targets
            epochs: Maximum number of training epochs
            patience: Number of epochs to wait for improvement before stopping
            verbose: Whether to print progress
        """
        # Ensure y has correct shape
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_val is not None and y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)
        
        # Early stopping variables
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_weights = None
        best_biases = None
        
        for epoch in range(epochs):
            # Forward and backward pass
            self.backward(X_train, y_train)
            
            # Compute losses
            train_pred = self.forward(X_train)
            train_loss = self.compute_loss(train_pred, y_train)
            self.train_losses.append(train_loss)
            
            # Validation loss and early stopping
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.compute_loss(val_pred, y_val)
                self.val_losses.append(val_loss)
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    # Save best weights
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                else:
                    epochs_without_improvement += 1
                
                # Early stopping check
                if epochs_without_improvement >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    # Restore best weights
                    self.weights = best_weights
                    self.biases = best_biases
                    break
            
            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                msg = f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}"
                if X_val is not None:
                    msg += f", Val Loss: {val_loss:.4f}"
                print(msg)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        return self.forward(X)
    
    def plot_losses(self) -> None:
        """Plot training and validation losses over epochs."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', linewidth=2)
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def create_house_price_data(n_samples: int = 1000, 
                           noise_level: float = 0.1,
                           random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic house price data for demonstration.
    
    Features include:
    - Square footage
    - Number of bedrooms
    - Number of bathrooms
    - Age of house
    - Garage size
    - Distance to city center
    - Crime rate
    - School rating
    - And more...
    
    The true relationship includes non-linearities and interactions.
    """
    np.random.seed(random_seed)
    
    # Generate features
    sqft = np.random.uniform(500, 5000, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.uniform(1, 4, n_samples)
    age = np.random.uniform(0, 50, n_samples)
    garage = np.random.randint(0, 4, n_samples)
    distance = np.random.exponential(10, n_samples)
    crime_rate = np.random.exponential(0.02, n_samples)
    school_rating = np.random.uniform(1, 10, n_samples)
    
    # Additional features
    pool = np.random.binomial(1, 0.3, n_samples)
    renovated = np.random.binomial(1, 0.2, n_samples)
    corner_lot = np.random.binomial(1, 0.25, n_samples)
    cul_de_sac = np.random.binomial(1, 0.15, n_samples)
    highway_access = np.random.binomial(1, 0.4, n_samples)
    
    # Create feature matrix
    X = np.column_stack([
        sqft, bedrooms, bathrooms, age, garage, distance,
        crime_rate, school_rating, pool, renovated,
        corner_lot, cul_de_sac, highway_access
    ])
    
    # Create non-linear target with interactions
    # Base price influenced by square footage
    price = 50000 + 100 * sqft
    
    # Bedroom/bathroom ratio matters
    bed_bath_ratio = bedrooms / (bathrooms + 0.5)
    price += 10000 * (2 - np.abs(bed_bath_ratio - 1.5))
    
    # Age affects price non-linearly
    price -= 1000 * age + 10 * age**2
    
    # Location effects
    price -= 2000 * distance
    price -= 50000 * crime_rate
    price += 5000 * school_rating
    
    # Amenities
    price += 20000 * pool
    price += 15000 * renovated
    price += 5000 * corner_lot
    price += 8000 * cul_de_sac
    price -= 10000 * highway_access  # Noise discount
    
    # Add some non-linear interactions
    price += 10 * sqft * school_rating
    price -= 5000 * age * (1 - renovated)
    
    # Add noise
    noise = np.random.normal(0, noise_level * np.std(price), n_samples)
    price += noise
    
    # Ensure positive prices
    price = np.maximum(price, 50000)
    
    return X, price


# Example usage
if __name__ == "__main__":
    # Generate data
    X, y = create_house_price_data(n_samples=1000)
    
    # Split into train/validation
    n_train = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # Normalize features (important for neural networks!)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / (std + 1e-8)
    X_val = (X_val - mean) / (std + 1e-8)
    
    # Normalize targets
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    
    # Train model with different initializations
    for init_method in ['random', 'he', 'xavier']:
        print(f"\n{'='*50}")
        print(f"Training with {init_method} initialization")
        print(f"{'='*50}")
        
        model = MLPv1(
            layer_sizes=[13, 64, 32, 1],
            learning_rate=0.01,
            initialization=init_method,
            random_seed=42
        )
        
        model.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=1000,
            patience=50,
            verbose=True
        )
        
        # Make predictions
        val_pred = model.predict(X_val)
        val_mse = np.mean((val_pred - y_val.reshape(-1, 1))**2)
        print(f"\nFinal Validation MSE: {val_mse:.4f}")
        
        # Denormalize for interpretable RMSE
        val_pred_denorm = val_pred * y_std + y_mean
        y_val_denorm = y_val * y_std + y_mean
        rmse = np.sqrt(np.mean((val_pred_denorm - y_val_denorm)**2))
        print(f"RMSE in dollars: ${rmse:,.2f}")
