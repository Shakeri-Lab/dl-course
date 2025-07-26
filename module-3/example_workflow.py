"""
Module 3 Example Workflow
========================

This script demonstrates the complete Module 3 workflow:
1. Building a model with the modular system
2. Using advanced optimizers and schedulers
3. Training on the challenging credit risk dataset
4. Analyzing results comprehensively

Run this script to see all Module 3 components in action!
"""

import numpy as np
import matplotlib.pyplot as plt

# Import all Module 3 components
from mlp_modules import MLP, Sequential, Linear, BatchNorm1d, ReLU, Dropout, Sigmoid
from advanced_optimizers import Adam, SGD, CosineAnnealingLR, OneCycleLR
from credit_risk_dataset import create_credit_risk_dataset, prepare_credit_data
from advanced_trainer import TrainingConfig, Trainer, DataLoader, analyze_predictions


def main():
    """Run complete Module 3 workflow."""
    
    print("="*70)
    print("MODULE 3: ADVANCED OPTIMIZATION AND TRAINING")
    print("="*70)
    
    # Step 1: Create challenging dataset
    print("\n1. Creating credit risk dataset...")
    X, y, feature_info = create_credit_risk_dataset(
        n_samples=5000,
        imbalance_ratio=0.15,
        missing_rate=0.1
    )
    
    # Prepare data with missing value handling
    X_processed, y_processed = prepare_credit_data(
        X, y, feature_info, 
        handle_missing='indicator'
    )
    
    print(f"   Dataset shape: {X_processed.shape}")
    print(f"   Class distribution: {np.bincount(y_processed)}")
    
    # Step 2: Split data
    print("\n2. Splitting data (70/15/15)...")
    n_train = int(0.7 * len(X_processed))
    n_val = int(0.15 * len(X_processed))
    
    indices = np.random.permutation(len(X_processed))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    X_train, y_train = X_processed[train_idx], y_processed[train_idx]
    X_val, y_val = X_processed[val_idx], y_processed[val_idx]
    X_test, y_test = X_processed[test_idx], y_processed[test_idx]
    
    # Normalize features
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    
    # Reshape targets
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    # Step 3: Build model using modular system
    print("\n3. Building model with modular architecture...")
    
    # Option 1: Using MLP convenience class
    model = MLP(
        input_dim=X_train.shape[1],
        hidden_dims=[128, 64, 32],
        output_dim=1,
        activation='relu',
        use_batchnorm=True,
        dropout=0.3,
        use_residual=False  # Try True for deeper networks!
    )
    
    # Option 2: Building manually with Sequential (commented out)
    # model = Sequential(
    #     Linear(X_train.shape[1], 128),
    #     BatchNorm1d(128),
    #     ReLU(),
    #     Dropout(0.3),
    #     Linear(128, 64),
    #     BatchNorm1d(64),
    #     ReLU(),
    #     Dropout(0.3),
    #     Linear(64, 32),
    #     BatchNorm1d(32),
    #     ReLU(),
    #     Dropout(0.3),
    #     Linear(32, 1),
    #     Sigmoid()
    # )
    
    print(f"   Model parameters: {model.count_parameters():,}")
    print(f"   Architecture: {model}")
    
    # Step 4: Configure training
    print("\n4. Configuring advanced training...")
    
    # Compute class weights for imbalanced data
    class_counts = np.bincount(y_train.flatten())
    class_weights = len(y_train) / (2 * class_counts)
    print(f"   Class weights: {class_weights}")
    
    # Training configuration
    config = TrainingConfig(
        # Model config
        model_config={
            'input_dim': X_train.shape[1],
            'hidden_dims': [128, 64, 32],
            'output_dim': 1
        },
        
        # Optimization
        optimizer='adam',
        learning_rate=0.001,
        weight_decay=0.01,
        gradient_clip=1.0,
        
        # Training
        epochs=50,  # Reduced for demo
        batch_size=64,
        gradient_accumulation_steps=1,
        
        # Regularization
        dropout=0.3,
        label_smoothing=0.1,
        
        # Learning rate schedule
        scheduler='cosine',
        scheduler_config={'T_max': 50},
        
        # Early stopping
        early_stopping=True,
        patience=10,
        
        # Other
        save_path='./checkpoints_demo',
        log_every=5,
        class_weights=class_weights
    )
    
    # Step 5: Create data loaders
    print("\n5. Creating data loaders...")
    train_loader = DataLoader(
        X_train, y_train, 
        batch_size=config.batch_size,
        shuffle=True,
        class_weights=class_weights
    )
    
    val_loader = DataLoader(
        X_val, y_val, 
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # Step 6: Train model
    print("\n6. Training with advanced optimizer and scheduler...")
    trainer = Trainer(model, config)
    
    # Train!
    trainer.fit(train_loader, val_loader)
    
    # Step 7: Visualize training
    print("\n7. Visualizing training history...")
    trainer.plot_history()
    
    # Step 8: Analyze predictions
    print("\n8. Analyzing model predictions on test set...")
    analyze_predictions(model, X_test, y_test.flatten())
    
    # Step 9: Final evaluation
    print("\n9. Final test set evaluation...")
    test_loader = DataLoader(
        X_test, y_test, 
        batch_size=config.batch_size, 
        shuffle=False
    )
    
    test_metrics = trainer.evaluate(test_loader)
    
    print("\nTEST SET PERFORMANCE:")
    print("="*50)
    for metric, value in test_metrics.items():
        print(f"{metric:15s}: {value:.4f}")
    
    # Step 10: Experiment suggestions
    print("\n" + "="*70)
    print("EXPERIMENT SUGGESTIONS:")
    print("="*70)
    print("1. Try different optimizers: SGD vs Adam vs RMSprop")
    print("2. Compare schedulers: Step vs Exponential vs Cosine vs OneCycle")
    print("3. Test architectures: Deeper vs Wider, with/without residual")
    print("4. Adjust regularization: Different dropout rates, L2 strengths")
    print("5. Handle imbalance: Try oversampling or different class weights")
    print("\nModify the config and model architecture to explore!")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the complete workflow
    main() 