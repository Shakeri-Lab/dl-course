"""
MLP v4: Cross-Validation
========================
Building on v3, we add cross-validation for robust model evaluation
and hyperparameter tuning.

New concepts:
1. K-fold cross-validation
2. Stratified splits for regression
3. Grid search for hyperparameter optimization
4. Nested cross-validation for unbiased performance estimation

Cross-validation helps us:
- Get more reliable performance estimates
- Make better use of limited data
- Select optimal hyperparameters
- Avoid overfitting to a particular train/val split
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import json

# Import from previous versions
from mlp_v1_basic import create_house_price_data
from mlp_v3_dropout import MLPv3

@dataclass
class CVResults:
    """Store cross-validation results"""
    fold_scores: List[float]
    mean_score: float
    std_score: float
    best_params: Optional[Dict[str, Any]] = None
    all_results: Optional[List[Dict]] = None


class CrossValidator:
    """
    Cross-validation utilities for neural networks.
    
    Includes:
    - K-fold CV
    - Stratified regression splits (based on target quantiles)
    - Grid search
    - Nested CV
    """
    
    def __init__(self, n_folds: int = 5, random_seed: Optional[int] = None):
        """
        Initialize cross-validator.
        
        Args:
            n_folds: Number of folds for cross-validation
            random_seed: For reproducible splits
        """
        self.n_folds = n_folds
        self.random_seed = random_seed
    
    def create_folds(self, X: np.ndarray, y: np.ndarray, 
                     stratified: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create cross-validation folds.
        
        For regression, stratification is based on target quantiles
        to ensure each fold has similar target distribution.
        
        Returns:
            List of (train_indices, val_indices) tuples
        """
        n_samples = len(X)
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        if stratified and y is not None:
            # Stratified split for regression
            # Bin targets into quantiles
            n_bins = min(self.n_folds, 10)
            y_binned = np.digitize(y, np.percentile(y, np.linspace(0, 100, n_bins + 1)[1:-1]))
            
            # Create folds ensuring each has similar distribution
            folds = [[] for _ in range(self.n_folds)]
            
            for bin_idx in range(n_bins):
                bin_samples = np.where(y_binned == bin_idx)[0]
                np.random.shuffle(bin_samples)
                
                # Distribute samples from this bin across folds
                for i, idx in enumerate(bin_samples):
                    folds[i % self.n_folds].append(idx)
            
            # Convert to train/val splits
            fold_indices = []
            for i in range(self.n_folds):
                val_idx = np.array(folds[i])
                train_idx = np.array([idx for j, fold in enumerate(folds) 
                                     if j != i for idx in fold])
                fold_indices.append((train_idx, val_idx))
        
        else:
            # Simple random k-fold
            indices = np.random.permutation(n_samples)
            fold_size = n_samples // self.n_folds
            
            fold_indices = []
            for i in range(self.n_folds):
                start = i * fold_size
                end = start + fold_size if i < self.n_folds - 1 else n_samples
                val_idx = indices[start:end]
                train_idx = np.concatenate([indices[:start], indices[end:]])
                fold_indices.append((train_idx, val_idx))
        
        return fold_indices
    
    def cross_validate(self, 
                      X: np.ndarray, 
                      y: np.ndarray,
                      model_params: Dict[str, Any],
                      training_params: Dict[str, Any],
                      stratified: bool = True,
                      verbose: bool = True) -> CVResults:
        """
        Perform k-fold cross-validation with given parameters.
        
        Args:
            X: Features
            y: Targets
            model_params: Parameters for MLPv3 initialization
            training_params: Parameters for fit method
            stratified: Whether to use stratified splits
            verbose: Print progress
            
        Returns:
            CVResults object with scores and statistics
        """
        folds = self.create_folds(X, y, stratified)
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            if verbose:
                print(f"\nFold {fold_idx + 1}/{self.n_folds}")
                print("-" * 30)
            
            # Split data
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            # Normalize using training data statistics
            # IMPORTANT: Compute stats only on training fold!
            X_mean = X_train.mean(axis=0)
            X_std = X_train.std(axis=0)
            X_train = (X_train - X_mean) / (X_std + 1e-8)
            X_val = (X_val - X_mean) / (X_std + 1e-8)
            
            y_mean = y_train.mean()
            y_std = y_train.std()
            y_train = (y_train - y_mean) / y_std
            y_val = (y_val - y_mean) / y_std
            
            # Train model
            model = MLPv3(**model_params)
            model.fit(X_train, y_train, X_val, y_val, 
                     verbose=False, **training_params)
            
            # Evaluate
            val_pred = model.predict(X_val)
            val_mse = np.mean((val_pred - y_val.reshape(-1, 1))**2)
            
            # Convert to RMSE in original scale for interpretability
            val_rmse = np.sqrt(val_mse) * y_std
            fold_scores.append(val_rmse)
            
            if verbose:
                print(f"Fold RMSE: ${val_rmse:,.2f}")
        
        # Compute statistics
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        if verbose:
            print(f"\nCross-validation results:")
            print(f"Mean RMSE: ${mean_score:,.2f} (+/- ${std_score:,.2f})")
        
        return CVResults(
            fold_scores=fold_scores,
            mean_score=mean_score,
            std_score=std_score
        )
    
    def grid_search(self,
                   X: np.ndarray,
                   y: np.ndarray,
                   param_grid: Dict[str, List[Any]],
                   training_params: Dict[str, Any],
                   stratified: bool = True,
                   verbose: bool = True) -> CVResults:
        """
        Grid search with cross-validation for hyperparameter optimization.
        
        Args:
            X: Features
            y: Targets
            param_grid: Dictionary of parameter names and values to try
                       E.g., {'learning_rate': [0.001, 0.01, 0.1],
                              'dropout_rate': [0.2, 0.5],
                              'l2_lambda': [0, 0.01, 0.1]}
            training_params: Fixed parameters for training
            stratified: Whether to use stratified CV
            verbose: Print progress
            
        Returns:
            CVResults with best parameters and all results
        """
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Create all combinations
        param_combinations = []
        
        def generate_combinations(idx, current_combo):
            if idx == len(param_names):
                param_combinations.append(dict(current_combo))
                return
            
            param_name = param_names[idx]
            for value in param_values[idx]:
                current_combo[param_name] = value
                generate_combinations(idx + 1, current_combo.copy())
        
        generate_combinations(0, {})
        
        if verbose:
            print(f"Grid search: testing {len(param_combinations)} parameter combinations")
            print("=" * 60)
        
        # Test each combination
        all_results = []
        best_score = float('inf')
        best_params = None
        
        for i, params in enumerate(param_combinations):
            if verbose:
                print(f"\nCombination {i + 1}/{len(param_combinations)}:")
                print(f"Parameters: {params}")
            
            # Fixed model parameters
            model_params = {
                'layer_sizes': [13, 64, 32, 1],
                'initialization': 'he',
                'random_seed': 42
            }
            model_params.update(params)
            
            # Run CV
            cv_results = self.cross_validate(
                X, y, model_params, training_params,
                stratified=stratified, verbose=False
            )
            
            result = {
                'params': params,
                'mean_score': cv_results.mean_score,
                'std_score': cv_results.std_score,
                'fold_scores': cv_results.fold_scores
            }
            all_results.append(result)
            
            if verbose:
                print(f"Mean RMSE: ${cv_results.mean_score:,.2f} "
                      f"(+/- ${cv_results.std_score:,.2f})")
            
            # Track best
            if cv_results.mean_score < best_score:
                best_score = cv_results.mean_score
                best_params = params
        
        if verbose:
            print("\n" + "=" * 60)
            print(f"Best parameters: {best_params}")
            print(f"Best mean RMSE: ${best_score:,.2f}")
        
        # Get detailed results for best params
        best_result = next(r for r in all_results 
                          if r['params'] == best_params)
        
        return CVResults(
            fold_scores=best_result['fold_scores'],
            mean_score=best_result['mean_score'],
            std_score=best_result['std_score'],
            best_params=best_params,
            all_results=all_results
        )
    
    def nested_cv(self,
                  X: np.ndarray,
                  y: np.ndarray,
                  param_grid: Dict[str, List[Any]],
                  training_params: Dict[str, Any],
                  inner_folds: int = 3,
                  verbose: bool = True) -> CVResults:
        """
        Nested cross-validation for unbiased performance estimation.
        
        Outer loop: Performance estimation
        Inner loop: Hyperparameter selection
        
        This prevents overfitting to the validation set during
        hyperparameter tuning.
        """
        outer_folds = self.create_folds(X, y, stratified=True)
        outer_scores = []
        best_params_per_fold = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_folds):
            if verbose:
                print(f"\n{'='*60}")
                print(f"OUTER FOLD {fold_idx + 1}/{self.n_folds}")
                print(f"{'='*60}")
            
            # Outer split
            X_train_outer, y_train_outer = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Inner CV for hyperparameter selection
            inner_cv = CrossValidator(n_folds=inner_folds, 
                                    random_seed=self.random_seed)
            
            inner_results = inner_cv.grid_search(
                X_train_outer, y_train_outer,
                param_grid, training_params,
                verbose=False
            )
            
            best_params = inner_results.best_params
            best_params_per_fold.append(best_params)
            
            if verbose:
                print(f"Best params from inner CV: {best_params}")
            
            # Train final model on full outer training set
            # Normalize
            X_mean = X_train_outer.mean(axis=0)
            X_std = X_train_outer.std(axis=0)
            X_train_norm = (X_train_outer - X_mean) / (X_std + 1e-8)
            X_test_norm = (X_test - X_mean) / (X_std + 1e-8)
            
            y_mean = y_train_outer.mean()
            y_std = y_train_outer.std()
            y_train_norm = (y_train_outer - y_mean) / y_std
            y_test_norm = (y_test - y_mean) / y_std
            
            # Train model
            model_params = {
                'layer_sizes': [13, 64, 32, 1],
                'initialization': 'he',
                'random_seed': 42
            }
            model_params.update(best_params)
            
            model = MLPv3(**model_params)
            model.fit(X_train_norm, y_train_norm, 
                     verbose=False, **training_params)
            
            # Evaluate on outer test set
            test_pred = model.predict(X_test_norm)
            test_mse = np.mean((test_pred - y_test_norm.reshape(-1, 1))**2)
            test_rmse = np.sqrt(test_mse) * y_std
            
            outer_scores.append(test_rmse)
            
            if verbose:
                print(f"Outer fold test RMSE: ${test_rmse:,.2f}")
        
        # Summary
        mean_score = np.mean(outer_scores)
        std_score = np.std(outer_scores)
        
        if verbose:
            print(f"\n{'='*60}")
            print("NESTED CV RESULTS")
            print(f"{'='*60}")
            print(f"Unbiased performance estimate: ${mean_score:,.2f} "
                  f"(+/- ${std_score:,.2f})")
            print("\nBest parameters per fold:")
            for i, params in enumerate(best_params_per_fold):
                print(f"Fold {i+1}: {params}")
        
        return CVResults(
            fold_scores=outer_scores,
            mean_score=mean_score,
            std_score=std_score,
            best_params=best_params_per_fold
        )


def plot_cv_results(cv_results: CVResults):
    """Visualize cross-validation results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Fold scores
    folds = np.arange(len(cv_results.fold_scores)) + 1
    ax1.bar(folds, cv_results.fold_scores, alpha=0.7)
    ax1.axhline(cv_results.mean_score, color='red', linestyle='--', 
                label=f'Mean: ${cv_results.mean_score:,.0f}')
    ax1.fill_between([0.5, len(folds) + 0.5], 
                     cv_results.mean_score - cv_results.std_score,
                     cv_results.mean_score + cv_results.std_score,
                     alpha=0.2, color='red', label='±1 std')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('RMSE ($)')
    ax1.set_title('Cross-Validation Fold Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameter importance (if grid search results available)
    if cv_results.all_results:
        # Analyze parameter importance
        param_impacts = {}
        
        for result in cv_results.all_results:
            for param, value in result['params'].items():
                if param not in param_impacts:
                    param_impacts[param] = []
                param_impacts[param].append((value, result['mean_score']))
        
        # Calculate variance in scores for each parameter
        param_importance = {}
        for param, values_scores in param_impacts.items():
            values = [v for v, _ in values_scores]
            scores = [s for _, s in values_scores]
            
            # Group by unique values
            unique_values = sorted(set(values))
            avg_scores = []
            for val in unique_values:
                val_scores = [s for v, s in values_scores if v == val]
                avg_scores.append(np.mean(val_scores))
            
            # Importance = variance in average scores
            param_importance[param] = np.std(avg_scores)
        
        # Plot parameter importance
        params = list(param_importance.keys())
        importances = list(param_importance.values())
        
        ax2.bar(params, importances, alpha=0.7)
        ax2.set_xlabel('Hyperparameter')
        ax2.set_ylabel('Impact on Performance (std of scores)')
        ax2.set_title('Hyperparameter Importance')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_hyperparameter_landscape(cv_results: CVResults):
    """
    Plot how each hyperparameter affects performance.
    
    Useful for understanding parameter sensitivity and interactions.
    """
    if not cv_results.all_results:
        print("No grid search results to plot")
        return
    
    # Extract parameter names
    param_names = list(cv_results.all_results[0]['params'].keys())
    n_params = len(param_names)
    
    # Create subplots
    fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 5))
    if n_params == 1:
        axes = [axes]
    
    for idx, param_name in enumerate(param_names):
        ax = axes[idx]
        
        # Collect values and scores for this parameter
        values_scores = []
        for result in cv_results.all_results:
            value = result['params'][param_name]
            score = result['mean_score']
            values_scores.append((value, score))
        
        # Sort by value
        values_scores.sort(key=lambda x: x[0])
        values = [v for v, _ in values_scores]
        scores = [s for _, s in values_scores]
        
        # Plot with error bars if available
        unique_values = sorted(set(values))
        mean_scores = []
        std_scores = []
        
        for val in unique_values:
            val_results = [r for r in cv_results.all_results 
                          if r['params'][param_name] == val]
            val_scores = [r['mean_score'] for r in val_results]
            mean_scores.append(np.mean(val_scores))
            std_scores.append(np.std(val_scores))
        
        # Plot
        ax.errorbar(unique_values, mean_scores, yerr=std_scores, 
                   marker='o', capsize=5, linewidth=2, markersize=8)
        ax.set_xlabel(param_name)
        ax.set_ylabel('Mean RMSE ($)')
        ax.set_title(f'Effect of {param_name}')
        ax.grid(True, alpha=0.3)
        
        # Log scale for certain parameters
        if param_name in ['learning_rate', 'l2_lambda']:
            ax.set_xscale('log')
    
    plt.suptitle('Hyperparameter Sensitivity Analysis', fontsize=16)
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Generate data
    X, y = create_house_price_data(n_samples=1000, noise_level=0.1)
    
    # Initialize cross-validator
    cv = CrossValidator(n_folds=5, random_seed=42)
    
    # 1. Simple cross-validation with fixed parameters
    print("=" * 60)
    print("SIMPLE 5-FOLD CROSS-VALIDATION")
    print("=" * 60)
    
    model_params = {
        'layer_sizes': [13, 64, 32, 1],
        'learning_rate': 0.01,
        'l2_lambda': 0.01,
        'dropout_rate': 0.3,
        'initialization': 'he',
        'random_seed': 42
    }
    
    training_params = {
        'epochs': 500,
        'patience': 30
    }
    
    cv_results = cv.cross_validate(X, y, model_params, training_params)
    plot_cv_results(cv_results)
    
    # 2. Grid search for hyperparameter optimization
    print("\n" + "=" * 60)
    print("GRID SEARCH WITH CROSS-VALIDATION")
    print("=" * 60)
    
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'dropout_rate': [0.2, 0.3, 0.5],
        'l2_lambda': [0, 0.01, 0.1]
    }
    
    grid_results = cv.grid_search(X, y, param_grid, training_params)
    plot_cv_results(grid_results)
    plot_hyperparameter_landscape(grid_results)
    
    # 3. Nested cross-validation for unbiased performance
    print("\n" + "=" * 60)
    print("NESTED CROSS-VALIDATION")
    print("=" * 60)
    
    # Use smaller grid for nested CV (computationally expensive)
    small_param_grid = {
        'learning_rate': [0.01, 0.1],
        'dropout_rate': [0.2, 0.5],
        'l2_lambda': [0.01]
    }
    
    nested_results = cv.nested_cv(X, y, small_param_grid, training_params,
                                  inner_folds=3)
    
    # Train final model with best parameters from grid search
    print("\n" + "=" * 60)
    print("FINAL MODEL WITH BEST PARAMETERS")
    print("=" * 60)
    
    # Use best parameters
    best_params = grid_results.best_params
    final_model_params = {
        'layer_sizes': [13, 64, 32, 1],
        'initialization': 'he',
        'random_seed': 42
    }
    final_model_params.update(best_params)
    
    # Train on full dataset
    # Split for final evaluation
    n_train = int(0.9 * len(X))
    indices = np.random.permutation(len(X))
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Normalize
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train = (X_train - X_mean) / (X_std + 1e-8)
    X_test = (X_test - X_mean) / (X_std + 1e-8)
    
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    
    # Train final model
    final_model = MLPv3(**final_model_params)
    final_model.fit(X_train, y_train, epochs=1000, patience=50)
    
    # Final evaluation
    test_pred = final_model.predict(X_test)
    test_mse = np.mean((test_pred - y_test.reshape(-1, 1))**2)
    test_rmse = np.sqrt(test_mse) * y_std
    
    print(f"\nFinal test RMSE: ${test_rmse:,.2f}")
    print(f"Expected performance from CV: ${grid_results.mean_score:,.2f} "
          f"(+/- ${grid_results.std_score:,.2f})")
