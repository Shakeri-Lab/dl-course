"""
Advanced Training Framework with GPU Support
===========================================

This module provides a comprehensive training framework that includes:
1. GPU memory management (simulated for NumPy)
2. Mixed precision training concepts
3. Gradient accumulation
4. Advanced metrics and logging
5. Model checkpointing
6. Early stopping with patience
7. Handling class imbalance

Key features:
- Modular design for easy experimentation
- Comprehensive logging and visualization
- Production-ready training loops

LEARNING PATH:
1. Start by understanding TrainingConfig dataclass
2. Study the Metrics class for evaluation methods
3. Explore the DataLoader for handling batches
4. Deep dive into Trainer class for the main training logic
5. Run the example to see everything in action
"""

import numpy as np
import time
import os
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from mlp_modules import Module, MLP
from advanced_optimizers import Optimizer, Adam, SGD, LRScheduler, clip_grad_norm


@dataclass
class TrainingConfig:
    """Configuration for training.
    
    This dataclass encapsulates all training hyperparameters in one place.
    Using a config object makes experiments reproducible and easy to track.
    
    Pro tip: Save configs with your model checkpoints for full reproducibility!
    """
    # Model
    model_config: Dict[str, Any]
    
    # Optimization
    optimizer: str = 'adam'
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    gradient_accumulation_steps: int = 1  # Simulate larger batches
    
    # Regularization
    dropout: float = 0.0
    label_smoothing: float = 0.0  # Prevents overconfidence
    
    # Learning rate schedule
    scheduler: Optional[str] = None
    scheduler_config: Dict[str, Any] = field(default_factory=dict)
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    
    # Checkpointing
    save_path: str = './checkpoints'
    save_every: int = 10
    keep_best_only: bool = True
    
    # Logging
    log_every: int = 10
    verbose: bool = True
    
    # Device (simulated)
    device: str = 'cuda'
    mixed_precision: bool = False
    
    # Class weights for imbalanced data
    class_weights: Optional[np.ndarray] = None


class Metrics:
    """Compute and track various metrics.
    
    This class provides a comprehensive set of evaluation metrics.
    Different metrics are suitable for different problems:
    - Accuracy: Good for balanced datasets
    - Precision/Recall: Important for imbalanced datasets
    - F1: Balanced measure between precision and recall
    - AUC-ROC: Threshold-independent performance measure
    """
    
    @staticmethod
    def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, 
                           label_smoothing: float = 0.0) -> float:
        """Binary cross-entropy with optional label smoothing."""
        eps = 1e-7
        
        # Apply label smoothing
        if label_smoothing > 0:
            y_true = y_true * (1 - label_smoothing) + 0.5 * label_smoothing
        
        # Clip predictions for numerical stability
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # Compute loss
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        return loss
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        """Binary classification accuracy."""
        y_pred_binary = (y_pred > threshold).astype(float)
        return np.mean(y_true == y_pred_binary)
    
    @staticmethod
    def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, 
                           threshold: float = 0.5) -> Dict[str, float]:
        """Compute precision, recall, and F1 score."""
        y_pred_binary = (y_pred > threshold).astype(float)
        
        # True positives, false positives, false negatives
        tp = np.sum((y_true == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true == 1) & (y_pred_binary == 0))
        
        # Compute metrics
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @staticmethod
    def auc_roc(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 100) -> float:
        """Compute Area Under ROC Curve."""
        # Sort by predicted probability
        sorted_indices = np.argsort(y_pred)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        # Compute TPR and FPR at different thresholds
        tprs = []
        fprs = []
        
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos
        
        tp = 0
        fp = 0
        
        for i in range(len(y_true_sorted)):
            if y_true_sorted[i] == 1:
                tp += 1
            else:
                fp += 1
            
            tpr = tp / n_pos
            fpr = fp / n_neg
            
            tprs.append(tpr)
            fprs.append(fpr)
        
        # Compute AUC using trapezoidal rule
        auc = 0
        for i in range(1, len(fprs)):
            auc += (fprs[i] - fprs[i-1]) * (tprs[i] + tprs[i-1]) / 2
        
        return auc


class GPUManager:
    """
    Simulated GPU manager for NumPy.
    
    In a real implementation, this would handle:
    - Device placement
    - Memory management
    - Mixed precision
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.memory_allocated = 0
        self.peak_memory = 0
        
    def to_device(self, array: np.ndarray) -> np.ndarray:
        """Move array to device (no-op for NumPy)."""
        # Simulate memory tracking
        self.memory_allocated += array.nbytes
        self.peak_memory = max(self.peak_memory, self.memory_allocated)
        return array
    
    def from_device(self, array: np.ndarray) -> np.ndarray:
        """Move array from device (no-op for NumPy)."""
        self.memory_allocated -= array.nbytes
        return array
    
    def synchronize(self):
        """Synchronize device (no-op for NumPy)."""
        pass
    
    def empty_cache(self):
        """Clear cache (reset memory tracking)."""
        self.memory_allocated = 0
    
    def memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        return {
            'allocated': self.memory_allocated,
            'peak': self.peak_memory,
            'device': self.device
        }


class DataLoader:
    """
    Simple data loader with batching and shuffling.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 batch_size: int = 32, shuffle: bool = True,
                 class_weights: Optional[np.ndarray] = None):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.class_weights = class_weights
        
        self.n_samples = len(X)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
        
    def __iter__(self):
        """Iterate over batches."""
        indices = np.arange(self.n_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            X_batch = self.X[batch_indices]
            y_batch = self.y[batch_indices]
            
            # Compute sample weights if class weights provided
            if self.class_weights is not None:
                sample_weights = self.class_weights[y_batch.astype(int).flatten()]
            else:
                sample_weights = np.ones(len(y_batch))
            
            yield X_batch, y_batch, sample_weights
    
    def __len__(self):
        return self.n_batches


class Trainer:
    """
    Advanced trainer for neural networks.
    """
    
    def __init__(self, model: Module, config: TrainingConfig):
        self.model = model
        self.config = config
        
        # Initialize GPU manager
        self.gpu = GPUManager(config.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler() if config.scheduler else None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.history = defaultdict(list)
        
        # Create checkpoint directory
        os.makedirs(config.save_path, exist_ok=True)
    
    def _create_optimizer(self) -> Optimizer:
        """Create optimizer based on config."""
        params = self.model.parameters()
        
        if self.config.optimizer == 'adam':
            return Adam(params, lr=self.config.learning_rate,
                       weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'sgd':
            return SGD(params, lr=self.config.learning_rate,
                      momentum=0.9, weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[LRScheduler]:
        """Create learning rate scheduler based on config."""
        from advanced_optimizers import (StepLR, ExponentialLR, 
                                       CosineAnnealingLR, OneCycleLR)
        
        if self.config.scheduler == 'step':
            return StepLR(self.optimizer, **self.config.scheduler_config)
        elif self.config.scheduler == 'exponential':
            return ExponentialLR(self.optimizer, **self.config.scheduler_config)
        elif self.config.scheduler == 'cosine':
            return CosineAnnealingLR(self.optimizer, **self.config.scheduler_config)
        elif self.config.scheduler == 'onecycle':
            return OneCycleLR(self.optimizer, **self.config.scheduler_config)
        
        return None
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray,
                     sample_weights: np.ndarray) -> float:
        """Compute weighted loss."""
        # Binary cross-entropy
        loss = Metrics.binary_cross_entropy(
            y_true, y_pred, 
            label_smoothing=self.config.label_smoothing
        )
        
        # Apply sample weights
        if sample_weights is not None:
            loss = np.mean(loss * sample_weights)
        
        return loss
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        This method implements a complete training epoch with:
        1. Forward pass through the model
        2. Loss computation with sample weights
        3. Backward pass for gradients
        4. Gradient accumulation for effective larger batches
        5. Gradient clipping to prevent explosion
        6. Parameter updates via optimizer
        
        Key concept: Gradient accumulation allows us to simulate larger
        batch sizes than what fits in memory by accumulating gradients
        over multiple forward/backward passes before updating weights.
        """
        self.model.train()
        
        epoch_loss = 0
        all_predictions = []
        all_targets = []
        
        # Gradient accumulation
        accumulated_steps = 0
        
        for batch_idx, (X_batch, y_batch, sample_weights) in enumerate(train_loader):
            # Move to device
            X_batch = self.gpu.to_device(X_batch)
            y_batch = self.gpu.to_device(y_batch)
            
            # Forward pass
            y_pred = self.model(X_batch)
            
            # Compute loss
            loss = self._compute_loss(y_batch, y_pred, sample_weights)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            eps = 1e-7
            grad_output = -(y_batch / (y_pred + eps) - 
                           (1 - y_batch) / (1 - y_pred + eps)) / len(y_batch)
            
            # Apply sample weights to gradients
            grad_output *= sample_weights.reshape(-1, 1) / self.config.gradient_accumulation_steps
            
            self.model.backward(grad_output)
            
            accumulated_steps += 1
            
            # Update weights
            if accumulated_steps >= self.config.gradient_accumulation_steps:
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    grad_norm = clip_grad_norm(
                        self.model.parameters(), 
                        self.config.gradient_clip
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                accumulated_steps = 0
                self.global_step += 1
            
            # Track metrics
            epoch_loss += loss.item() * self.config.gradient_accumulation_steps
            all_predictions.extend(y_pred.flatten())
            all_targets.extend(y_batch.flatten())
            
            # Clear GPU cache periodically
            if batch_idx % 100 == 0:
                self.gpu.empty_cache()
        
        # Compute epoch metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        metrics = {
            'loss': epoch_loss / len(train_loader),
            'accuracy': Metrics.accuracy(all_targets, all_predictions),
            'auc': Metrics.auc_roc(all_targets, all_predictions)
        }
        
        pr_metrics = Metrics.precision_recall_f1(all_targets, all_predictions)
        metrics.update(pr_metrics)
        
        return metrics
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        
        val_loss = 0
        all_predictions = []
        all_targets = []
        
        for X_batch, y_batch, sample_weights in val_loader:
            # Move to device
            X_batch = self.gpu.to_device(X_batch)
            y_batch = self.gpu.to_device(y_batch)
            
            # Forward pass
            y_pred = self.model(X_batch)
            
            # Compute loss
            loss = self._compute_loss(y_batch, y_pred, sample_weights)
            
            val_loss += loss
            all_predictions.extend(y_pred.flatten())
            all_targets.extend(y_batch.flatten())
        
        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        metrics = {
            'loss': val_loss / len(val_loader),
            'accuracy': Metrics.accuracy(all_targets, all_predictions),
            'auc': Metrics.auc_roc(all_targets, all_predictions)
        }
        
        pr_metrics = Metrics.precision_recall_f1(all_targets, all_predictions)
        metrics.update(pr_metrics)
        
        return metrics
    
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Full training loop.
        
        This is the main training orchestrator that:
        1. Manages epochs and early stopping
        2. Calls train_epoch and evaluate methods
        3. Handles learning rate scheduling
        4. Tracks training history
        5. Saves checkpoints
        6. Implements early stopping logic
        
        Early stopping prevents overfitting by monitoring validation
        performance and stopping when it stops improving.
        """
        print(f"Starting training on {self.config.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                
                # Early stopping check
                if self.config.early_stopping:
                    if val_metrics['loss'] < self.best_metric - self.config.min_delta:
                        self.best_metric = val_metrics['loss']
                        self.patience_counter = 0
                        
                        # Save best model
                        if self.config.keep_best_only:
                            self.save_checkpoint('best_model.pkl', is_best=True)
                    else:
                        self.patience_counter += 1
                        
                        if self.patience_counter >= self.config.patience:
                            print(f"\nEarly stopping at epoch {epoch + 1}")
                            break
            else:
                val_metrics = {}
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Record history
            for key, value in train_metrics.items():
                self.history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                self.history[f'val_{key}'].append(value)
            self.history['lr'].append(self.optimizer.lr)
            
            # Logging
            epoch_time = time.time() - start_time
            
            if self.config.verbose and (epoch + 1) % self.config.log_every == 0:
                self._log_metrics(epoch, train_metrics, val_metrics, epoch_time)
            
            # Checkpointing
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pkl')
        
        # Final cleanup
        self.gpu.empty_cache()
        
        # Print memory usage
        if self.config.verbose:
            memory_info = self.gpu.memory_summary()
            print(f"\nPeak memory usage: {memory_info['peak'] / 1e9:.2f} GB")
    
    def _log_metrics(self, epoch: int, train_metrics: Dict[str, float],
                    val_metrics: Dict[str, float], epoch_time: float):
        """Log metrics during training."""
        log_str = f"Epoch {epoch + 1}/{self.config.epochs} ({epoch_time:.2f}s)"
        log_str += f" - LR: {self.optimizer.lr:.6f}"
        
        # Train metrics
        log_str += " - Train:"
        for key, value in train_metrics.items():
            log_str += f" {key}: {value:.4f}"
        
        # Validation metrics
        if val_metrics:
            log_str += " - Val:"
            for key, value in val_metrics.items():
                log_str += f" {key}: {value:.4f}"
        
        print(log_str)
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'history': dict(self.history),
            'best_metric': self.best_metric
        }
        
        filepath = os.path.join(self.config.save_path, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        if self.config.verbose:
            print(f"{'Best model' if is_best else 'Checkpoint'} saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        if self.scheduler and checkpoint['scheduler_state']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.history = defaultdict(list, checkpoint['history'])
        self.best_metric = checkpoint['best_metric']
        
        print(f"Checkpoint loaded from {filepath}")
    
    def plot_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Loss
        ax = axes[0, 0]
        ax.plot(self.history['train_loss'], label='Train', linewidth=2)
        if 'val_loss' in self.history:
            ax.plot(self.history['val_loss'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy
        ax = axes[0, 1]
        ax.plot(self.history['train_accuracy'], label='Train', linewidth=2)
        if 'val_accuracy' in self.history:
            ax.plot(self.history['val_accuracy'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # AUC
        ax = axes[0, 2]
        ax.plot(self.history['train_auc'], label='Train', linewidth=2)
        if 'val_auc' in self.history:
            ax.plot(self.history['val_auc'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        ax.set_title('AUC History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # F1 Score
        ax = axes[1, 0]
        ax.plot(self.history['train_f1'], label='Train', linewidth=2)
        if 'val_f1' in self.history:
            ax.plot(self.history['val_f1'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning Rate
        ax = axes[1, 1]
        ax.plot(self.history['lr'], linewidth=2, color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Precision vs Recall
        ax = axes[1, 2]
        if 'val_precision' in self.history and 'val_recall' in self.history:
            ax.plot(self.history['val_recall'], self.history['val_precision'], 
                   'o-', linewidth=2, markersize=4)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def analyze_predictions(model: Module, X_test: np.ndarray, y_test: np.ndarray,
                       feature_names: Optional[List[str]] = None):
    """Analyze model predictions in detail.
    
    This function provides comprehensive analysis of model predictions:
    1. Prediction distribution by class
    2. Calibration plot (are predicted probabilities accurate?)
    3. ROC curve and AUC
    4. Confusion matrix at optimal threshold
    
    These visualizations help diagnose model issues:
    - Poor separation in distributions → need better features/model
    - Poor calibration → model is overconfident/underconfident
    - Low AUC → model can't distinguish classes well
    """
    model.eval()
    predictions = model(X_test).flatten()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Prediction distribution
    ax = axes[0, 0]
    ax.hist(predictions[y_test == 0], bins=30, alpha=0.6, 
            label='True Negative', density=True)
    ax.hist(predictions[y_test == 1], bins=30, alpha=0.6, 
            label='True Positive', density=True)
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Distribution by True Label')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Calibration plot
    ax = axes[0, 1]
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    actual_probs = []
    predicted_probs = []
    
    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i+1])
        if mask.sum() > 0:
            actual_probs.append(y_test[mask].mean())
            predicted_probs.append(predictions[mask].mean())
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.plot(predicted_probs, actual_probs, 'o-', linewidth=2, 
            markersize=8, label='Model')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. ROC Curve
    ax = axes[1, 0]
    
    # Compute ROC curve
    thresholds = np.linspace(0, 1, 100)
    tprs = []
    fprs = []
    
    for threshold in thresholds:
        y_pred_binary = (predictions > threshold).astype(float)
        tp = np.sum((y_test == 1) & (y_pred_binary == 1))
        fp = np.sum((y_test == 0) & (y_pred_binary == 1))
        fn = np.sum((y_test == 1) & (y_pred_binary == 0))
        tn = np.sum((y_test == 0) & (y_pred_binary == 0))
        
        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        
        tprs.append(tpr)
        fprs.append(fpr)
    
    auc = Metrics.auc_roc(y_test, predictions)
    
    ax.plot(fprs, tprs, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Confusion matrix at optimal threshold
    ax = axes[1, 1]
    
    # Find optimal threshold (maximize F1)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        metrics = Metrics.precision_recall_f1(y_test, predictions, threshold)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = threshold
    
    # Compute confusion matrix
    y_pred_binary = (predictions > best_threshold).astype(float)
    tp = np.sum((y_test == 1) & (y_pred_binary == 1))
    fp = np.sum((y_test == 0) & (y_pred_binary == 1))
    fn = np.sum((y_test == 1) & (y_pred_binary == 0))
    tn = np.sum((y_test == 0) & (y_pred_binary == 0))
    
    confusion_matrix = np.array([[tn, fp], [fn, tp]])
    
    # Plot
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['True 0', 'True 1'], ax=ax)
    ax.set_title(f'Confusion Matrix (threshold={best_threshold:.3f})')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed metrics
    print("\nDETAILED METRICS:")
    print("=" * 50)
    
    metrics = Metrics.precision_recall_f1(y_test, predictions, best_threshold)
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"Accuracy: {Metrics.accuracy(y_test, predictions, best_threshold):.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"AUC-ROC: {auc:.4f}")


# Example usage
if __name__ == "__main__":
    # Import dataset creation
    from credit_risk_dataset import (create_credit_risk_dataset, 
                                   prepare_credit_data)
    
    # Create dataset
    print("Creating credit risk dataset...")
    X, y, feature_info = create_credit_risk_dataset(
        n_samples=5000,
        imbalance_ratio=0.15
    )
    
    # Prepare data
    X_processed, y_processed = prepare_credit_data(
        X, y, feature_info, 
        handle_missing='indicator'
    )
    
    # Split data
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
    
    # Compute class weights
    class_counts = np.bincount(y_train.flatten())
    class_weights = len(y_train) / (2 * class_counts)
    
    print(f"\nDataset shape: {X_train.shape}")
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights}")
    
    # Create model
    model_config = {
        'input_dim': X_train.shape[1],
        'hidden_dims': [128, 64, 32],
        'output_dim': 1,
        'activation': 'relu',
        'use_batchnorm': True,
        'dropout': 0.3,
        'use_residual': False
    }
    
    model = MLP(**model_config)
    
    # Training configuration
    config = TrainingConfig(
        model_config=model_config,
        optimizer='adam',
        learning_rate=0.001,
        weight_decay=0.01,
        gradient_clip=1.0,
        epochs=100,
        batch_size=64,
        gradient_accumulation_steps=1,
        dropout=0.3,
        label_smoothing=0.1,
        scheduler='cosine',
        scheduler_config={'T_max': 100},
        early_stopping=True,
        patience=15,
        save_path='./checkpoints',
        save_every=20,
        log_every=5,
        class_weights=class_weights
    )
    
    # Create data loaders
    train_loader = DataLoader(X_train, y_train, 
                             batch_size=config.batch_size,
                             shuffle=True,
                             class_weights=class_weights)
    
    val_loader = DataLoader(X_val, y_val, 
                           batch_size=config.batch_size,
                           shuffle=False)
    
    # Initialize trainer
    trainer = Trainer(model, config)
    
    # Train model
    print("\nStarting training...")
    trainer.fit(train_loader, val_loader)
    
    # Plot training history
    trainer.plot_history()
    
    # Load best model
    trainer.load_checkpoint('./checkpoints/best_model.pkl')
    
    # Analyze predictions
    print("\nAnalyzing model predictions...")
    analyze_predictions(model, X_test, y_test.flatten())
    
    # Final test evaluation
    test_loader = DataLoader(X_test, y_test, batch_size=config.batch_size, 
                            shuffle=False)
    test_metrics = trainer.evaluate(test_loader)
    
    print("\nFINAL TEST METRICS:")
    print("=" * 50)
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")
