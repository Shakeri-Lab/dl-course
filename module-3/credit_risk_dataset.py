"""
Credit Risk Dataset: A Challenging Tabular Problem
==================================================

This dataset simulates a credit risk prediction problem with:
1. Mixed feature types (numerical, categorical, temporal)
2. Non-linear relationships and complex interactions
3. Class imbalance
4. Missing values
5. Different feature scales
6. Temporal patterns

This complexity justifies the need for advanced techniques like:
- Batch normalization for handling different scales
- Complex architectures for capturing interactions
- Advanced optimizers for difficult loss landscapes
- Learning rate scheduling for convergence
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns


def create_credit_risk_dataset(
    n_samples: int = 10000,
    n_numerical: int = 15,
    n_categorical: int = 8,
    n_temporal: int = 5,
    missing_rate: float = 0.1,
    noise_level: float = 0.1,
    imbalance_ratio: float = 0.15,  # 15% default rate
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Create a complex credit risk dataset.
    
    Features include:
    - Income, debt ratios, credit history
    - Employment, education, location
    - Transaction patterns, seasonality
    - Non-linear interactions between features
    
    Returns:
        X: Feature matrix
        y: Binary labels (0=good, 1=default)
        feature_info: Dictionary with feature names and types
    """
    np.random.seed(random_seed)
    
    # Initialize feature storage
    features = []
    feature_names = []
    feature_types = []
    
    # 1. NUMERICAL FEATURES
    # Income (log-normal distribution)
    income = np.random.lognormal(10.5, 0.7, n_samples)
    features.append(income)
    feature_names.append('annual_income')
    feature_types.append('numerical')
    
    # Age (truncated normal)
    age = np.clip(np.random.normal(40, 12, n_samples), 18, 80)
    features.append(age)
    feature_names.append('age')
    feature_types.append('numerical')
    
    # Credit score (beta distribution scaled)
    credit_score = 300 + 550 * np.random.beta(5, 2, n_samples)
    features.append(credit_score)
    feature_names.append('credit_score')
    feature_types.append('numerical')
    
    # Debt-to-income ratio
    debt_ratio = np.random.beta(2, 5, n_samples)
    features.append(debt_ratio)
    feature_names.append('debt_to_income_ratio')
    feature_types.append('numerical')
    
    # Number of credit accounts
    n_accounts = np.random.poisson(5, n_samples) + 1
    features.append(n_accounts.astype(float))
    feature_names.append('num_credit_accounts')
    feature_types.append('numerical')
    
    # Credit utilization
    credit_util = np.random.beta(2, 3, n_samples)
    features.append(credit_util)
    feature_names.append('credit_utilization')
    feature_types.append('numerical')
    
    # Months since last delinquency (exponential with some having none)
    has_delinquency = np.random.binomial(1, 0.4, n_samples)
    months_since_delinq = np.where(
        has_delinquency,
        np.random.exponential(24, n_samples),
        999  # Special value for "never"
    )
    features.append(months_since_delinq)
    feature_names.append('months_since_last_delinquency')
    feature_types.append('numerical')
    
    # Add more complex numerical features
    for i in range(n_numerical - 7):
        if i % 3 == 0:
            # Some features with heavy tails
            feat = np.random.gamma(2, 2, n_samples)
        elif i % 3 == 1:
            # Some with bimodal distributions
            mode = np.random.binomial(1, 0.5, n_samples)
            feat = np.where(mode, 
                           np.random.normal(10, 2, n_samples),
                           np.random.normal(20, 3, n_samples))
        else:
            # Some with truncated distributions
            feat = np.clip(np.random.normal(0, 1, n_samples), -2, 2)
        
        features.append(feat)
        feature_names.append(f'numerical_feature_{i+8}')
        feature_types.append('numerical')
    
    # 2. CATEGORICAL FEATURES
    # Employment status (ordinal)
    employment_status = np.random.choice(
        [0, 1, 2, 3, 4],  # unemployed, part-time, full-time, self-employed, retired
        n_samples,
        p=[0.05, 0.15, 0.60, 0.15, 0.05]
    )
    features.append(employment_status.astype(float))
    feature_names.append('employment_status')
    feature_types.append('categorical')
    
    # Education level (ordinal)
    education = np.random.choice(
        [0, 1, 2, 3, 4],  # high school, some college, bachelor's, master's, phd
        n_samples,
        p=[0.30, 0.25, 0.30, 0.12, 0.03]
    )
    features.append(education.astype(float))
    feature_names.append('education_level')
    feature_types.append('categorical')
    
    # State (nominal - simplified to regions)
    region = np.random.choice(
        [0, 1, 2, 3],  # NE, SE, MW, W
        n_samples,
        p=[0.20, 0.35, 0.25, 0.20]
    )
    features.append(region.astype(float))
    feature_names.append('region')
    feature_types.append('categorical')
    
    # Home ownership
    home_ownership = np.random.choice(
        [0, 1, 2],  # rent, mortgage, own
        n_samples,
        p=[0.35, 0.45, 0.20]
    )
    features.append(home_ownership.astype(float))
    feature_names.append('home_ownership')
    feature_types.append('categorical')
    
    # Loan purpose
    loan_purpose = np.random.choice(
        range(10),  # various purposes
        n_samples
    )
    features.append(loan_purpose.astype(float))
    feature_names.append('loan_purpose')
    feature_types.append('categorical')
    
    # Add more categorical features
    for i in range(n_categorical - 5):
        n_categories = np.random.randint(3, 8)
        cat_feat = np.random.randint(0, n_categories, n_samples)
        features.append(cat_feat.astype(float))
        feature_names.append(f'categorical_feature_{i+6}')
        feature_types.append('categorical')
    
    # 3. TEMPORAL FEATURES
    # Day of week when applied
    day_of_week = np.random.randint(0, 7, n_samples)
    features.append(day_of_week.astype(float))
    feature_names.append('application_day_of_week')
    feature_types.append('temporal')
    
    # Month of year (seasonality)
    month = np.random.randint(1, 13, n_samples)
    features.append(month.astype(float))
    feature_names.append('application_month')
    feature_types.append('temporal')
    
    # Time since account opened (in months)
    account_age = np.random.gamma(4, 12, n_samples)
    features.append(account_age)
    feature_names.append('months_since_account_opened')
    feature_types.append('temporal')
    
    # Recent inquiry trend (number in last 6 months)
    recent_inquiries = np.random.poisson(1.5, n_samples)
    features.append(recent_inquiries.astype(float))
    feature_names.append('inquiries_last_6_months')
    feature_types.append('temporal')
    
    # Payment trend (improving=1, stable=0, worsening=-1)
    payment_trend = np.random.choice(
        [-1, 0, 1],
        n_samples,
        p=[0.15, 0.70, 0.15]
    )
    features.append(payment_trend.astype(float))
    feature_names.append('payment_trend')
    feature_types.append('temporal')
    
    # Stack all features
    X = np.column_stack(features)
    
    # 4. CREATE TARGET WITH COMPLEX RELATIONSHIPS
    # Base default probability
    default_prob = np.zeros(n_samples)
    
    # Credit score has strong non-linear effect
    credit_effect = 1 / (1 + np.exp((credit_score - 600) / 50))
    default_prob += 0.3 * credit_effect
    
    # Income and debt ratio interaction
    income_debt_effect = debt_ratio * np.exp(-income / 50000)
    default_prob += 0.2 * income_debt_effect
    
    # Age has U-shaped relationship (young and old are riskier)
    age_effect = 0.1 * ((age - 45) / 20) ** 2
    default_prob += 0.1 * age_effect
    
    # Employment and education interaction
    emp_edu_effect = (4 - employment_status) * (4 - education) / 16
    default_prob += 0.1 * emp_edu_effect
    
    # Credit utilization has threshold effect
    util_effect = np.where(credit_util > 0.7, 
                          (credit_util - 0.7) * 2, 
                          0)
    default_prob += 0.15 * util_effect
    
    # Recent inquiries increase risk exponentially
    inquiry_effect = 1 - np.exp(-recent_inquiries / 3)
    default_prob += 0.1 * inquiry_effect
    
    # Seasonal effect (higher defaults in December/January)
    seasonal_effect = 0.05 * np.sin(2 * np.pi * (month - 1) / 12) + \
                     0.05 * (month == 12) + 0.05 * (month == 1)
    default_prob += seasonal_effect
    
    # Account age protective effect
    account_age_effect = 1 - (1 - np.exp(-account_age / 24))
    default_prob *= account_age_effect
    
    # Add some pure noise
    default_prob += noise_level * np.random.randn(n_samples)
    
    # Clip probabilities
    default_prob = np.clip(default_prob, 0.01, 0.99)
    
    # Generate binary outcomes
    # Adjust to achieve desired imbalance ratio
    threshold = np.percentile(default_prob, (1 - imbalance_ratio) * 100)
    y = (default_prob > threshold).astype(int)
    
    # 5. ADD MISSING VALUES
    # Some features are more likely to have missing values
    missing_probs = {
        'months_since_last_delinquency': 0.3,  # Often not applicable
        'inquiries_last_6_months': 0.15,
        'payment_trend': 0.20,
        'employment_status': 0.05
    }
    
    for i, feat_name in enumerate(feature_names):
        if feat_name in missing_probs:
            miss_prob = missing_probs[feat_name]
        else:
            miss_prob = missing_rate
        
        # Create missing mask
        missing_mask = np.random.binomial(1, miss_prob, n_samples).astype(bool)
        X[missing_mask, i] = np.nan
    
    # Create feature info dictionary
    feature_info = {
        'names': feature_names,
        'types': feature_types,
        'n_features': len(feature_names),
        'n_numerical': sum(1 for t in feature_types if t == 'numerical'),
        'n_categorical': sum(1 for t in feature_types if t == 'categorical'),
        'n_temporal': sum(1 for t in feature_types if t == 'temporal')
    }
    
    return X, y, feature_info


def analyze_dataset(X: np.ndarray, y: np.ndarray, feature_info: dict):
    """
    Analyze and visualize the credit risk dataset.
    """
    print("DATASET ANALYSIS")
    print("=" * 60)
    
    # Basic statistics
    print(f"Number of samples: {len(X):,}")
    print(f"Number of features: {X.shape[1]}")
    print(f"  - Numerical: {feature_info['n_numerical']}")
    print(f"  - Categorical: {feature_info['n_categorical']}")
    print(f"  - Temporal: {feature_info['n_temporal']}")
    
    # Class distribution
    n_defaults = y.sum()
    print(f"\nClass distribution:")
    print(f"  - Good (0): {len(y) - n_defaults:,} ({(1 - y.mean())*100:.1f}%)")
    print(f"  - Default (1): {n_defaults:,} ({y.mean()*100:.1f}%)")
    
    # Missing values
    missing_counts = np.isnan(X).sum(axis=0)
    print(f"\nMissing values:")
    for i, (name, count) in enumerate(zip(feature_info['names'], missing_counts)):
        if count > 0:
            print(f"  - {name}: {count} ({count/len(X)*100:.1f}%)")
    
    # Feature correlations with target
    print("\nTop features correlated with default:")
    correlations = []
    for i in range(X.shape[1]):
        # Skip if too many missing values
        valid_mask = ~np.isnan(X[:, i])
        if valid_mask.sum() > len(X) * 0.5:
            corr = np.corrcoef(X[valid_mask, i], y[valid_mask])[0, 1]
            correlations.append((feature_info['names'][i], corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, corr in correlations[:10]:
        print(f"  - {name}: {corr:.3f}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Feature distributions by class
    ax = axes[0, 0]
    feature_idx = feature_info['names'].index('credit_score')
    valid_mask = ~np.isnan(X[:, feature_idx])
    for class_val in [0, 1]:
        mask = (y == class_val) & valid_mask
        ax.hist(X[mask, feature_idx], bins=30, alpha=0.6, 
                label=f'Class {class_val}', density=True)
    ax.set_xlabel('Credit Score')
    ax.set_ylabel('Density')
    ax.set_title('Credit Score Distribution by Class')
    ax.legend()
    
    # 2. Income vs Debt Ratio
    ax = axes[0, 1]
    income_idx = feature_info['names'].index('annual_income')
    debt_idx = feature_info['names'].index('debt_to_income_ratio')
    valid_mask = ~np.isnan(X[:, income_idx]) & ~np.isnan(X[:, debt_idx])
    
    scatter = ax.scatter(X[valid_mask, income_idx], 
                        X[valid_mask, debt_idx],
                        c=y[valid_mask], 
                        alpha=0.5, 
                        cmap='RdYlBu',
                        s=10)
    ax.set_xlabel('Annual Income')
    ax.set_ylabel('Debt-to-Income Ratio')
    ax.set_title('Income vs Debt Ratio')
    ax.set_xscale('log')
    plt.colorbar(scatter, ax=ax)
    
    # 3. Age effect (U-shaped)
    ax = axes[0, 2]
    age_idx = feature_info['names'].index('age')
    valid_mask = ~np.isnan(X[:, age_idx])
    
    age_bins = np.linspace(18, 80, 20)
    age_groups = np.digitize(X[valid_mask, age_idx], age_bins)
    default_rates = []
    age_centers = []
    
    for i in range(1, len(age_bins)):
        mask = age_groups == i
        if mask.sum() > 10:
            default_rates.append(y[valid_mask][mask].mean())
            age_centers.append((age_bins[i-1] + age_bins[i]) / 2)
    
    ax.plot(age_centers, default_rates, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Age')
    ax.set_ylabel('Default Rate')
    ax.set_title('Default Rate by Age (U-shaped)')
    ax.grid(True, alpha=0.3)
    
    # 4. Seasonal pattern
    ax = axes[1, 0]
    month_idx = feature_info['names'].index('application_month')
    valid_mask = ~np.isnan(X[:, month_idx])
    
    monthly_defaults = []
    for month in range(1, 13):
        mask = (X[valid_mask, month_idx] == month)
        if mask.sum() > 0:
            monthly_defaults.append(y[valid_mask][mask].mean())
    
    ax.plot(range(1, 13), monthly_defaults, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Month')
    ax.set_ylabel('Default Rate')
    ax.set_title('Seasonal Pattern in Defaults')
    ax.set_xticks(range(1, 13))
    ax.grid(True, alpha=0.3)
    
    # 5. Employment status effect
    ax = axes[1, 1]
    emp_idx = feature_info['names'].index('employment_status')
    valid_mask = ~np.isnan(X[:, emp_idx])
    
    emp_labels = ['Unemployed', 'Part-time', 'Full-time', 'Self-emp', 'Retired']
    emp_defaults = []
    emp_counts = []
    
    for emp_status in range(5):
        mask = (X[valid_mask, emp_idx] == emp_status)
        if mask.sum() > 0:
            emp_defaults.append(y[valid_mask][mask].mean())
            emp_counts.append(mask.sum())
    
    bars = ax.bar(emp_labels, emp_defaults)
    ax.set_xlabel('Employment Status')
    ax.set_ylabel('Default Rate')
    ax.set_title('Default Rate by Employment Status')
    ax.tick_params(axis='x', rotation=45)
    
    # Add count labels
    for bar, count in zip(bars, emp_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'n={count}', ha='center', va='bottom', fontsize=8)
    
    # 6. Feature importance proxy (mutual information)
    ax = axes[1, 2]
    
    # Simple mutual information approximation
    mi_scores = []
    for i in range(min(15, X.shape[1])):  # Top 15 features
        valid_mask = ~np.isnan(X[:, i])
        if valid_mask.sum() > len(X) * 0.5:
            # Discretize continuous features
            if feature_info['types'][i] == 'numerical':
                x_discrete = np.digitize(X[valid_mask, i], 
                                       np.percentile(X[valid_mask, i], 
                                                   [20, 40, 60, 80]))
            else:
                x_discrete = X[valid_mask, i].astype(int)
            
            # Compute mutual information (simplified)
            mi = 0
            for x_val in np.unique(x_discrete):
                for y_val in [0, 1]:
                    p_xy = ((x_discrete == x_val) & (y[valid_mask] == y_val)).mean()
                    if p_xy > 0:
                        p_x = (x_discrete == x_val).mean()
                        p_y = (y[valid_mask] == y_val).mean()
                        mi += p_xy * np.log(p_xy / (p_x * p_y) + 1e-10)
            
            mi_scores.append((feature_info['names'][i], mi))
    
    mi_scores.sort(key=lambda x: x[1], reverse=True)
    
    names = [x[0] for x in mi_scores[:10]]
    scores = [x[1] for x in mi_scores[:10]]
    
    bars = ax.barh(range(len(names)), scores)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Mutual Information Score')
    ax.set_title('Top 10 Most Informative Features')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'correlations': correlations,
        'mi_scores': mi_scores
    }


def prepare_credit_data(X: np.ndarray, y: np.ndarray, feature_info: dict,
                       handle_missing: str = 'impute') -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare credit data for neural network training.
    
    Args:
        X: Raw features
        y: Labels
        feature_info: Feature metadata
        handle_missing: How to handle missing values
                       'impute': Simple imputation
                       'indicator': Add missing indicators
                       'drop': Drop samples with missing values
    
    Returns:
        X_processed: Processed features
        y_processed: Processed labels
    """
    X_processed = X.copy()
    y_processed = y.copy()
    
    if handle_missing == 'drop':
        # Drop samples with any missing values
        complete_mask = ~np.isnan(X_processed).any(axis=1)
        X_processed = X_processed[complete_mask]
        y_processed = y_processed[complete_mask]
    
    elif handle_missing == 'impute':
        # Simple imputation by feature type
        for i, (name, feat_type) in enumerate(zip(feature_info['names'], 
                                                  feature_info['types'])):
            if np.isnan(X_processed[:, i]).any():
                if feat_type == 'numerical':
                    # Median imputation for numerical
                    median_val = np.nanmedian(X_processed[:, i])
                    X_processed[np.isnan(X_processed[:, i]), i] = median_val
                else:
                    # Mode imputation for categorical/temporal
                    valid_values = X_processed[~np.isnan(X_processed[:, i]), i]
                    if len(valid_values) > 0:
                        mode_val = np.bincount(valid_values.astype(int)).argmax()
                        X_processed[np.isnan(X_processed[:, i]), i] = mode_val
    
    elif handle_missing == 'indicator':
        # Add missing indicators and impute
        missing_indicators = []
        
        for i in range(X.shape[1]):
            if np.isnan(X_processed[:, i]).any():
                # Create indicator
                indicator = np.isnan(X_processed[:, i]).astype(float)
                missing_indicators.append(indicator)
                
                # Impute with median/mode
                if feature_info['types'][i] == 'numerical':
                    fill_val = np.nanmedian(X_processed[:, i])
                else:
                    valid_values = X_processed[~np.isnan(X_processed[:, i]), i]
                    if len(valid_values) > 0:
                        fill_val = np.bincount(valid_values.astype(int)).argmax()
                    else:
                        fill_val = 0
                
                X_processed[np.isnan(X_processed[:, i]), i] = fill_val
        
        # Add indicators to features
        if missing_indicators:
            X_processed = np.hstack([X_processed] + 
                                  [ind.reshape(-1, 1) for ind in missing_indicators])
    
    return X_processed, y_processed


# Example usage
if __name__ == "__main__":
    # Create dataset
    print("Creating credit risk dataset...")
    X, y, feature_info = create_credit_risk_dataset(
        n_samples=10000,
        imbalance_ratio=0.15,
        missing_rate=0.1,
        noise_level=0.1
    )
    
    # Analyze dataset
    analysis_results = analyze_dataset(X, y, feature_info)
    
    # Prepare data
    print("\nPreparing data for neural network...")
    X_processed, y_processed = prepare_credit_data(X, y, feature_info, 
                                                  handle_missing='indicator')
    
    print(f"\nProcessed data shape: {X_processed.shape}")
    print(f"Original features: {feature_info['n_features']}")
    print(f"After missing indicators: {X_processed.shape[1]}")
    
    # Show class weights for handling imbalance
    class_counts = np.bincount(y_processed)
    class_weights = len(y_processed) / (2 * class_counts)
    print(f"\nClass weights for training:")
    print(f"  - Class 0: {class_weights[0]:.3f}")
    print(f"  - Class 1: {class_weights[1]:.3f}")
