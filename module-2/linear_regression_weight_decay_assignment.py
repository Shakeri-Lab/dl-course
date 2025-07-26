#!/usr/bin/env python3
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