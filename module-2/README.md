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

## 📁 Files in This Module

### Linear Regression - PyTorch Implementations
- **`linear_regression_pytorch.py`** - Concise implementation using PyTorch's `nn.Module` and autograd
- **`linear_regression_weight_decay.py`** - Demonstrates L2 regularization with visualization of overfitting
- **`linear_regression_weight_decay_assignment.py`** - Student assignment with TODOs to implement weight decay

### Softmax Regression - PyTorch Implementation  
- **`softmax_regression_pytorch.py`** - Multiclass classification using PyTorch's high-level APIs

### Multi-Layer Perceptron (MLP) - Progressive Implementation Series
**NEW: Complete MLP implementation series showing gradual complexity increase**

- **`mlp_v1_basic.py`** - Basic MLP with ReLU activation and different initialization strategies
  - Weight initialization methods (Random, He, Xavier)
  - Forward and backward propagation
  - Early stopping
  - Synthetic house price prediction dataset

- **`mlp_v2_regularization.py`** - Adds L2 regularization to prevent overfitting
  - L2 penalty implementation
  - Regularization strength comparison
  - Weight magnitude analysis

- **`mlp_v3_dropout.py`** - Adds dropout regularization for robust training
  - Dropout implementation with proper scaling
  - Training vs evaluation mode handling
  - Monte Carlo dropout for uncertainty estimation

- **`mlp_v4_cv.py`** - Adds cross-validation for robust evaluation
  - K-fold cross-validation
  - Grid search for hyperparameter optimization
  - Nested cross-validation for unbiased performance estimation

- **`house_prices_demo.py`** - Comprehensive demonstration script
  - Progressive comparison of all MLP versions
  - Visualization of improvements from each technique
  - Complete experimental pipeline

### Assignment Materials

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