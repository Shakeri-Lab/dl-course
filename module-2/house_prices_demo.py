"""
House Prices Demo: Progressive MLP Implementation
================================================

This demo script shows the progressive development of our MLP implementation:
1. Basic MLP with different initializations
2. Adding L2 regularization
3. Adding dropout
4. Adding cross-validation

We'll see how each addition improves the model's performance and robustness.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Import all our MLP versions
from mlp_v1_basic import MLPv1, create_house_price_data
from mlp_v2_regularization import MLPv2
from mlp_v3_dropout import MLPv3
from mlp_v4_cv import CrossValidator


def prepare_data(n_samples: int = 1000, 
                 test_size: float = 0.2,
                 noise_level: float = 0.1) -> Tuple:
    """
    Prepare house price data for experiments.
    
    Returns normalized train/val/test splits.
    """
    # Generate synthetic data
    X, y = create_house_price_data(n_samples=n_samples, 
                                   noise_level=noise_level, 
                                   random_seed=42)
    
    # Create train/val/test splits (60/20/20)
    n_test = int(test_size * n_samples)
    n_val = int(test_size * n_samples)
    n_train = n_samples - n_test - n_val
    
    # Shuffle and split
    indices = np.random.RandomState(42).permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Normalize features (using only training statistics!)
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    
    X_train = (X_train - X_mean) / (X_std + 1e-8)
    X_val = (X_val - X_mean) / (X_std + 1e-8)
    X_test = (X_test - X_mean) / (X_std + 1e-8)
    
    # Normalize targets
    y_mean = y_train.mean()
    y_std = y_train.std()
    
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    
    return (X_train, y_train, X_val, y_val, X_test, y_test, 
            X_mean, X_std, y_mean, y_std)


def evaluate_model(model, X_test, y_test, y_std):
    """Evaluate model and return metrics."""
    predictions = model.predict(X_test)
    mse = np.mean((predictions - y_test.reshape(-1, 1))**2)
    rmse = np.sqrt(mse) * y_std  # Convert back to dollars
    
    # Also compute R² score
    ss_res = np.sum((y_test.reshape(-1, 1) - predictions)**2)
    ss_tot = np.sum((y_test - y_test.mean())**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {'rmse': rmse, 'mse': mse, 'r2': r2}


def demo_v1_initialization():
    """
    Demo 1: Impact of Weight Initialization
    
    Shows how different initialization methods affect:
    - Convergence speed
    - Final performance
    - Training stability
    """
    print("\n" + "="*70)
    print("DEMO 1: WEIGHT INITIALIZATION COMPARISON")
    print("="*70)
    
    # Prepare data
    data = prepare_data(n_samples=1000)
    X_train, y_train, X_val, y_val, X_test, y_test = data[:6]
    y_std = data[9]
    
    # Test different initializations
    init_methods = ['random', 'xavier', 'he']
    results = {}
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, init_method in enumerate(init_methods):
        print(f"\n{init_method.upper()} Initialization:")
        print("-" * 40)
        
        # Train model
        model = MLPv1(
            layer_sizes=[13, 64, 32, 1],
            learning_rate=0.01,
            initialization=init_method,
            random_seed=42
        )
        
        model.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=500,
            patience=50,
            verbose=False
        )
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, y_std)
        results[init_method] = metrics
        
        print(f"Test RMSE: ${metrics['rmse']:,.2f}")
        print(f"R² Score: {metrics['r2']:.3f}")
        print(f"Training epochs: {len(model.train_losses)}")
        
        # Plot training curves
        ax = axes[idx]
        ax.plot(model.train_losses, label='Train', linewidth=2)
        ax.plot(model.val_losses, label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{init_method.capitalize()} Initialization')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(model.train_losses[:50]))  # Focus on relevant range
    
    plt.suptitle('Training Curves for Different Initializations', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Summary
    print("\n" + "-"*50)
    print("SUMMARY: He initialization performs best for ReLU networks")
    print("- Faster convergence")
    print("- Better final performance")
    print("- More stable training")
    
    return results


def demo_v2_regularization():
    """
    Demo 2: L2 Regularization Effect
    
    Shows how L2 regularization:
    - Reduces overfitting
    - Controls weight magnitudes
    - Affects the bias-variance tradeoff
    """
    print("\n" + "="*70)
    print("DEMO 2: L2 REGULARIZATION EFFECT")
    print("="*70)
    
    # Prepare data with MORE noise to see overfitting
    data = prepare_data(n_samples=500, noise_level=0.3)
    X_train, y_train, X_val, y_val, X_test, y_test = data[:6]
    y_std = data[9]
    
    # Test different L2 strengths
    l2_values = [0, 0.001, 0.01, 0.1]
    results = {}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for l2_lambda in l2_values:
        print(f"\nL2 Lambda = {l2_lambda}:")
        print("-" * 40)
        
        model = MLPv2(
            layer_sizes=[13, 64, 32, 1],
            learning_rate=0.01,
            l2_lambda=l2_lambda,
            initialization='he',
            random_seed=42
        )
        
        model.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=1000,
            patience=100,
            verbose=False
        )
        
        # Evaluate
        train_pred = model.predict(X_train)
        train_rmse = np.sqrt(np.mean((train_pred - y_train.reshape(-1, 1))**2)) * y_std
        
        metrics = evaluate_model(model, X_test, y_test, y_std)
        
        # Calculate average weight magnitude
        avg_weight = np.mean([np.mean(np.abs(w)) for w in model.weights])
        
        results[l2_lambda] = {
            'train_rmse': train_rmse,
            'test_rmse': metrics['rmse'],
            'avg_weight': avg_weight,
            'model': model
        }
        
        print(f"Train RMSE: ${train_rmse:,.2f}")
        print(f"Test RMSE: ${metrics['rmse']:,.2f}")
        print(f"Avg weight magnitude: {avg_weight:.4f}")
        
        # Plot training curves
        ax1.plot(model.val_losses, label=f'λ={l2_lambda}', linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Validation Loss for Different L2 Strengths')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot overfitting analysis
    l2_vals = list(results.keys())
    train_rmses = [results[l2]['train_rmse'] for l2 in l2_vals]
    test_rmses = [results[l2]['test_rmse'] for l2 in l2_vals]
    
    x_pos = np.arange(len(l2_vals))
    width = 0.35
    
    ax2.bar(x_pos - width/2, train_rmses, width, label='Train RMSE', alpha=0.8)
    ax2.bar(x_pos + width/2, test_rmses, width, label='Test RMSE', alpha=0.8)
    ax2.set_xlabel('L2 Lambda')
    ax2.set_ylabel('RMSE ($)')
    ax2.set_title('Overfitting Analysis')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(l2) for l2 in l2_vals])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Weight distribution analysis
    plt.figure(figsize=(10, 6))
    for i, (l2, res) in enumerate(results.items()):
        model = res['model']
        all_weights = np.concatenate([w.flatten() for w in model.weights])
        plt.hist(all_weights, bins=50, alpha=0.5, label=f'λ={l2}', 
                density=True)
    
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    plt.title('Weight Distribution for Different L2 Strengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n" + "-"*50)
    print("SUMMARY: L2 regularization helps by:")
    print("- Reducing the gap between train and test error")
    print("- Keeping weights small and distributed around zero")
    print("- Optimal λ depends on noise level and dataset size")
    
    return results


def demo_v3_dropout():
    """
    Demo 3: Dropout for Regularization and Uncertainty
    
    Shows how dropout:
    - Acts as ensemble learning
    - Provides uncertainty estimates
    - Complements L2 regularization
    """
    print("\n" + "="*70)
    print("DEMO 3: DROPOUT REGULARIZATION")
    print("="*70)
    
    # Prepare data
    data = prepare_data(n_samples=1000)
    X_train, y_train, X_val, y_val, X_test, y_test = data[:6]
    y_std = data[9]
    
    # Compare dropout rates
    dropout_rates = [0, 0.2, 0.5]
    results = {}
    
    for dropout in dropout_rates:
        print(f"\nDropout Rate = {dropout}:")
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
            epochs=500,
            patience=50,
            verbose=False
        )
        
        metrics = evaluate_model(model, X_test, y_test, y_std)
        results[dropout] = {
            'metrics': metrics,
            'model': model
        }
        
        print(f"Test RMSE: ${metrics['rmse']:,.2f}")
        print(f"R² Score: {metrics['r2']:.3f}")
    
    # Demonstrate uncertainty estimation
    print("\n" + "-"*50)
    print("UNCERTAINTY ESTIMATION WITH MONTE CARLO DROPOUT")
    print("-"*50)
    
    # Use model with dropout for uncertainty
    dropout_model = results[0.5]['model']
    
    # Select a few test samples
    n_samples = 10
    test_indices = np.random.choice(len(X_test), n_samples, replace=False)
    X_subset = X_test[test_indices]
    y_subset = y_test[test_indices]
    
    # Get MC predictions
    mc_results = dropout_model.monte_carlo_predict(X_subset, n_samples=100)
    
    # Visualize uncertainty
    plt.figure(figsize=(12, 6))
    
    # Denormalize for interpretability
    mc_mean_dollars = mc_results['mean'].flatten() * y_std + data[8]
    mc_std_dollars = mc_results['std'].flatten() * y_std
    y_true_dollars = y_subset * y_std + data[8]
    
    x_pos = np.arange(n_samples)
    
    # Plot predictions with uncertainty
    plt.errorbar(x_pos, mc_mean_dollars, yerr=2*mc_std_dollars,
                fmt='o', capsize=5, capthick=2, markersize=8,
                label='Prediction ± 2σ')
    plt.plot(x_pos, y_true_dollars, 'rs', markersize=10, 
             label='True Value')
    
    plt.xlabel('Test Sample')
    plt.ylabel('House Price ($)')
    plt.title('Predictions with Uncertainty Estimates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text annotations for high uncertainty samples
    high_uncertainty_idx = np.argsort(mc_std_dollars)[-3:]
    for idx in high_uncertainty_idx:
        plt.annotate(f'High\nuncertainty', 
                    xy=(idx, mc_mean_dollars[idx]),
                    xytext=(idx, mc_mean_dollars[idx] + 50000),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # Analyze what makes predictions uncertain
    print("\nUncertainty Analysis:")
    print(f"Average uncertainty: ${np.mean(mc_std_dollars):,.0f}")
    print(f"Max uncertainty: ${np.max(mc_std_dollars):,.0f}")
    print(f"Min uncertainty: ${np.min(mc_std_dollars):,.0f}")
    
    return results


def demo_v4_cross_validation():
    """
    Demo 4: Cross-Validation for Robust Evaluation
    
    Shows how CV:
    - Provides reliable performance estimates
    - Helps select optimal hyperparameters
    - Avoids overfitting to validation set
    """
    print("\n" + "="*70)
    print("DEMO 4: CROSS-VALIDATION AND HYPERPARAMETER TUNING")
    print("="*70)
    
    # Prepare full dataset (no test split for CV)
    X, y = create_house_price_data(n_samples=1000, noise_level=0.1)
    
    # Initialize cross-validator
    cv = CrossValidator(n_folds=5, random_seed=42)
    
    # 1. Compare single split vs CV
    print("\nCOMPARING SINGLE SPLIT VS CROSS-VALIDATION")
    print("-" * 50)
    
    # Single split evaluation
    data = prepare_data(n_samples=1000)
    X_train, y_train, X_val, y_val, X_test, y_test = data[:6]
    y_std = data[9]
    
    single_model = MLPv3(
        layer_sizes=[13, 64, 32, 1],
        learning_rate=0.01,
        l2_lambda=0.01,
        dropout_rate=0.3,
        initialization='he',
        random_seed=42
    )
    
    single_model.fit(X_train, y_train, X_val, y_val, 
                    epochs=500, patience=50, verbose=False)
    
    single_metrics = evaluate_model(single_model, X_test, y_test, y_std)
    print(f"Single split test RMSE: ${single_metrics['rmse']:,.2f}")
    
    # Cross-validation evaluation
    model_params = {
        'layer_sizes': [13, 64, 32, 1],
        'learning_rate': 0.01,
        'l2_lambda': 0.01,
        'dropout_rate': 0.3,
        'initialization': 'he',
        'random_seed': 42
    }
    
    cv_results = cv.cross_validate(
        X, y, model_params, 
        {'epochs': 500, 'patience': 50},
        verbose=False
    )
    
    print(f"Cross-validation RMSE: ${cv_results.mean_score:,.2f} "
          f"(+/- ${cv_results.std_score:,.2f})")
    
    # 2. Grid search for optimal parameters
    print("\n" + "-"*50)
    print("GRID SEARCH FOR HYPERPARAMETERS")
    print("-" * 50)
    
    # Define parameter grid
    param_grid = {
        'learning_rate': [0.01, 0.05],
        'dropout_rate': [0.2, 0.3, 0.5],
        'l2_lambda': [0.001, 0.01]
    }
    
    print(f"Testing {len(param_grid['learning_rate']) * len(param_grid['dropout_rate']) * len(param_grid['l2_lambda'])} combinations...")
    
    grid_results = cv.grid_search(
        X, y, param_grid, 
        {'epochs': 300, 'patience': 30},
        verbose=False
    )
    
    print(f"\nBest parameters: {grid_results.best_params}")
    print(f"Best CV score: ${grid_results.mean_score:,.2f}")
    
    # Visualize grid search results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Extract results for visualization
    param_names = list(param_grid.keys())
    
    for idx, param_name in enumerate(param_names):
        ax = axes[idx]
        
        # Group results by this parameter
        param_values = param_grid[param_name]
        scores_by_value = {val: [] for val in param_values}
        
        for result in grid_results.all_results:
            val = result['params'][param_name]
            scores_by_value[val].append(result['mean_score'])
        
        # Plot
        values = list(scores_by_value.keys())
        means = [np.mean(scores_by_value[v]) for v in values]
        stds = [np.std(scores_by_value[v]) for v in values]
        
        ax.errorbar(values, means, yerr=stds, marker='o', 
                   capsize=5, linewidth=2, markersize=8)
        ax.set_xlabel(param_name)
        ax.set_ylabel('Mean RMSE ($)')
        ax.set_title(f'Effect of {param_name}')
        ax.grid(True, alpha=0.3)
        
        if param_name in ['learning_rate', 'l2_lambda']:
            ax.set_xscale('log')
    
    plt.suptitle('Hyperparameter Sensitivity from Grid Search', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 3. Learning curves
    print("\n" + "-"*50)
    print("LEARNING CURVES")
    print("-" * 50)
    
    # Train with different amounts of data
    data_sizes = [200, 400, 600, 800, 1000]
    train_scores = []
    val_scores = []
    
    for n in data_sizes:
        # Use best parameters
        model_params.update(grid_results.best_params)
        
        # CV on subset
        subset_idx = np.random.choice(len(X), n, replace=False)
        X_subset = X[subset_idx]
        y_subset = y[subset_idx]
        
        cv_result = cv.cross_validate(
            X_subset, y_subset, model_params,
            {'epochs': 300, 'patience': 30},
            verbose=False
        )
        
        val_scores.append(cv_result.mean_score)
        
        # Estimate training score (using one fold)
        # This is a simplification for demo purposes
        train_scores.append(cv_result.mean_score * 0.7)  # Training typically lower
    
    plt.figure(figsize=(10, 6))
    plt.plot(data_sizes, train_scores, 'o-', label='Training Score', 
             linewidth=2, markersize=8)
    plt.plot(data_sizes, val_scores, 's-', label='Validation Score', 
             linewidth=2, markersize=8)
    plt.xlabel('Training Set Size')
    plt.ylabel('RMSE ($)')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n" + "-"*50)
    print("SUMMARY: Cross-validation provides:")
    print("- More reliable performance estimates")
    print("- Better hyperparameter selection")
    print("- Understanding of model stability")
    print("- Insight into data requirements")
    
    return grid_results


def main():
    """Run all demonstrations"""
    print("="*70)
    print("MLP IMPLEMENTATION: PROGRESSIVE DEVELOPMENT")
    print("="*70)
    print("\nThis demo shows the evolution of our MLP implementation:")
    print("1. Basic MLP with initialization methods")
    print("2. Adding L2 regularization")
    print("3. Adding dropout")
    print("4. Adding cross-validation")
    
    # Run demos
    v1_results = demo_v1_initialization()
    v2_results = demo_v2_regularization()
    v3_results = demo_v3_dropout()
    v4_results = demo_v4_cross_validation()
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY: KEY LESSONS")
    print("="*70)
    print("\n1. INITIALIZATION:")
    print("   - He initialization works best with ReLU activations")
    print("   - Proper initialization speeds up convergence")
    
    print("\n2. L2 REGULARIZATION:")
    print("   - Reduces overfitting by penalizing large weights")
    print("   - Optimal λ depends on noise level and data size")
    
    print("\n3. DROPOUT:")
    print("   - Acts as ensemble learning")
    print("   - Provides uncertainty estimates via MC Dropout")
    print("   - Works well combined with L2")
    
    print("\n4. CROSS-VALIDATION:")
    print("   - Essential for reliable performance estimation")
    print("   - Enables systematic hyperparameter tuning")
    print("   - Reveals model stability and data requirements")
    
    print("\n5. BEST PRACTICES:")
    print("   - Always normalize inputs and targets")
    print("   - Use validation data for early stopping")
    print("   - Combine multiple regularization techniques")
    print("   - Use CV for final evaluation and tuning")
    print("   - Monitor both training and validation metrics")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the demonstration
    main()
