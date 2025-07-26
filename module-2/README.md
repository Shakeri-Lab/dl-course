# Module 2: Linear Regression and Fundamentals

This module covers the fundamentals of linear regression, implementing it both from scratch and using PyTorch's high-level API. We also explore regularization techniques, specifically weight decay (L2 regularization).

## Learning Objectives

By the end of this module, you will:
1. Understand the mathematical foundations of linear regression
2. Implement linear regression from scratch using manual gradient computation
3. Use PyTorch's `nn.Module` and automatic differentiation
4. Understand overfitting in high-dimensional settings
5. Implement and apply weight decay (L2 regularization)
6. Analyze the bias-variance tradeoff

## Files in This Module

### 1. `linear_regression_teaching.py`
**From-Scratch Implementation**
- Complete implementation of linear regression without using PyTorch's autograd
- Manual gradient computation and parameter updates
- Comprehensive comments explaining each step
- Demonstrates fundamental concepts: forward pass, loss computation, backpropagation

**Key Features:**
- Synthetic data generation
- Mini-batch gradient descent
- Training loop with progress monitoring
- Visualization of training progress

**Run with:**
```bash
python linear_regression_teaching.py
```

### 2. `linear_regression_pytorch.py`
**PyTorch High-Level API Implementation**
- Uses `nn.Module`, `nn.Linear`, and automatic differentiation
- Demonstrates PyTorch's optimization framework
- Shows advanced features like gradient inspection and model evaluation modes

**Key Features:**
- `nn.Module` class definition
- Automatic gradient computation with `loss.backward()`
- PyTorch optimizers and loss functions
- Data loading with `DataLoader`
- Comparison with scratch implementation

**Run with:**
```bash
python linear_regression_pytorch.py
```

### 3. `linear_regression_weight_decay.py`
**Weight Decay and Regularization**
- Demonstrates overfitting in high-dimensional scenarios (200 features, 20 samples)
- Implements L2 regularization (weight decay)
- Compares different regularization strengths
- Comprehensive visualization of regularization effects

**Key Features:**
- High-dimensional synthetic data generation
- Custom model with L2 penalty computation
- Training with multiple λ (lambda) values
- 4-panel visualization showing:
  - Training/validation loss curves
  - Weight norm evolution
  - Optimal λ selection
  - Weight distribution changes

**Run with:**
```bash
python linear_regression_weight_decay.py
```

### 4. `linear_regression_weight_decay_assignment.py`
**Assignment Version**
- Student version with TODO sections
- Hands-on implementation of key concepts
- Conceptual questions about regularization

**Assignment Tasks:**
1. Implement L2 penalty calculation
2. Complete training loop with weight decay
3. Implement validation loop
4. Analyze different λ values
5. Answer conceptual questions

## Key Concepts Covered

### Mathematical Foundations
- Linear model: `y = Xw + b`
- Mean Squared Error loss
- Gradient computation using chain rule
- Stochastic Gradient Descent

### Implementation Details
- Forward propagation
- Backpropagation (manual vs automatic)
- Mini-batch training
- Parameter initialization

### Regularization
- Overfitting in high dimensions
- L2 regularization: `Loss = MSE + (λ/2)||w||²`
- Bias-variance tradeoff
- Hyperparameter selection using validation data

### PyTorch Framework
- `nn.Module` architecture
- Automatic differentiation
- Optimizers (`SGD`, `Adam`, etc.)
- Data handling with `DataLoader`

## Expected Outputs

When you run the scripts, you'll see:
1. **Training progress**: Loss values and parameter estimates
2. **Visualizations**: 
   - Training loss curves
   - Predictions vs true values
   - Weight decay analysis plots
3. **Parameter comparison**: Learned vs true parameters
4. **Regularization insights**: Effect of different λ values

## Prerequisites

- Basic Python programming
- Linear algebra (matrix operations)
- Calculus (gradients, chain rule)
- Basic statistics

## Next Steps

After completing this module:
- Try different learning rates and observe convergence
- Experiment with different data sizes and noise levels
- Implement L1 regularization (Lasso)
- Move to Module 3: Classification and Logistic Regression

## Tips for Success

1. **Understand the math first**: Work through the gradient derivations on paper
2. **Compare implementations**: Run both scratch and PyTorch versions
3. **Experiment**: Try different hyperparameters and observe effects
4. **Visualize**: Always plot your results to gain intuition
5. **Ask questions**: If something doesn't make sense, investigate further

## Common Issues

- **Convergence problems**: Try smaller learning rates
- **Overfitting**: Use more training data or stronger regularization
- **Underfitting**: Reduce regularization or increase model capacity
- **Numerical instability**: Check for proper weight initialization 