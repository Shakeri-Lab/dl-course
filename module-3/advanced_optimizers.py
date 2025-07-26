"""
Advanced Optimizers and Learning Rate Scheduling
================================================

This module implements various optimization algorithms and learning rate schedules:
1. SGD with momentum and Nesterov acceleration
2. Adam (Adaptive Moment Estimation)
3. RMSprop
4. AdaGrad
5. Learning rate schedulers
6. Gradient clipping and weight decay

Key concepts:
- Adaptive learning rates
- Momentum for faster convergence
- Learning rate scheduling for better convergence
- Gradient clipping for stability
"""

import numpy as np
from typing import List, Optional, Dict, Any, Callable, Tuple
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from mlp_modules import Module, Parameter


class Optimizer(ABC):
    """Base class for all optimizers."""
    
    def __init__(self, params: List[Parameter], lr: float = 0.001):
        self.params = list(params)
        self.lr = lr
        self.state = {}  # State dictionary for optimizer variables
        self.iteration = 0
        
    @abstractmethod
    def step(self):
        """Perform a single optimization step."""
        pass
    
    def zero_grad(self):
        """Zero all parameter gradients."""
        for param in self.params:
            param.zero_grad()
            
    def state_dict(self) -> Dict[str, Any]:
        """Return optimizer state."""
        return {
            'state': self.state,
            'lr': self.lr,
            'iteration': self.iteration
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state."""
        self.state = state_dict['state']
        self.lr = state_dict['lr']
        self.iteration = state_dict['iteration']


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum and Nesterov acceleration.
    
    Update rules:
    - Vanilla SGD: θ = θ - lr * grad
    - Momentum: v = momentum * v - lr * grad; θ = θ + v
    - Nesterov: v = momentum * v - lr * grad(θ + momentum * v); θ = θ + v
    """
    
    def __init__(self, params: List[Parameter], lr: float = 0.01,
                 momentum: float = 0.0, weight_decay: float = 0.0,
                 nesterov: bool = False):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        
        # Initialize momentum buffers
        if momentum > 0:
            for param in self.params:
                self.state[id(param)] = {'momentum_buffer': np.zeros_like(param.data)}
    
    def step(self):
        """Perform a single optimization step."""
        for param in self.params:
            if not param.requires_grad:
                continue
                
            # Get gradient
            grad = param.grad
            
            # Add weight decay (L2 regularization)
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data
            
            if self.momentum > 0:
                state = self.state[id(param)]
                buf = state['momentum_buffer']
                
                if self.nesterov:
                    # Nesterov momentum
                    buf = self.momentum * buf + grad
                    grad = grad + self.momentum * buf
                else:
                    # Standard momentum
                    buf = self.momentum * buf + grad
                    grad = buf
                
                state['momentum_buffer'] = buf
            
            # Update parameters
            param.data -= self.lr * grad
        
        self.iteration += 1


class Adam(Optimizer):
    """
    Adam optimizer: Adaptive Moment Estimation.
    
    Combines ideas from RMSprop and momentum:
    - Maintains running averages of gradients (first moment)
    - Maintains running averages of squared gradients (second moment)
    - Bias correction for the running averages
    """
    
    def __init__(self, params: List[Parameter], lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0,
                 amsgrad: bool = False):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        
        # Initialize state
        for param in self.params:
            self.state[id(param)] = {
                'exp_avg': np.zeros_like(param.data),      # First moment
                'exp_avg_sq': np.zeros_like(param.data),   # Second moment
                'max_exp_avg_sq': np.zeros_like(param.data) if amsgrad else None
            }
    
    def step(self):
        """Perform a single optimization step."""
        beta1, beta2 = self.betas
        
        for param in self.params:
            if not param.requires_grad:
                continue
            
            grad = param.grad
            
            # Add weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data
            
            state = self.state[id(param)]
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            
            # Update biased first moment estimate
            exp_avg = beta1 * exp_avg + (1 - beta1) * grad
            # Update biased second raw moment estimate
            exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad**2
            
            state['exp_avg'] = exp_avg
            state['exp_avg_sq'] = exp_avg_sq
            
            # Bias correction
            bias_correction1 = 1 - beta1 ** (self.iteration + 1)
            bias_correction2 = 1 - beta2 ** (self.iteration + 1)
            
            # Corrected moments
            exp_avg_corrected = exp_avg / bias_correction1
            exp_avg_sq_corrected = exp_avg_sq / bias_correction2
            
            if self.amsgrad:
                # Maintain max of second moment
                max_exp_avg_sq = state['max_exp_avg_sq']
                max_exp_avg_sq = np.maximum(max_exp_avg_sq, exp_avg_sq_corrected)
                state['max_exp_avg_sq'] = max_exp_avg_sq
                denominator = np.sqrt(max_exp_avg_sq) + self.eps
            else:
                denominator = np.sqrt(exp_avg_sq_corrected) + self.eps
            
            # Update parameters
            param.data -= self.lr * exp_avg_corrected / denominator
        
        self.iteration += 1


class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    
    Divides learning rate by an exponentially decaying average of squared gradients.
    Helps with non-stationary objectives.
    """
    
    def __init__(self, params: List[Parameter], lr: float = 0.01,
                 alpha: float = 0.99, eps: float = 1e-8,
                 weight_decay: float = 0.0, momentum: float = 0.0):
        super().__init__(params, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        
        # Initialize state
        for param in self.params:
            self.state[id(param)] = {
                'square_avg': np.zeros_like(param.data),
                'momentum_buffer': np.zeros_like(param.data) if momentum > 0 else None
            }
    
    def step(self):
        """Perform a single optimization step."""
        for param in self.params:
            if not param.requires_grad:
                continue
                
            grad = param.grad
            
            # Add weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data
            
            state = self.state[id(param)]
            square_avg = state['square_avg']
            
            # Update running average of squared gradients
            square_avg = self.alpha * square_avg + (1 - self.alpha) * grad**2
            state['square_avg'] = square_avg
            
            # Compute update
            avg = np.sqrt(square_avg + self.eps)
            
            if self.momentum > 0:
                buf = state['momentum_buffer']
                buf = self.momentum * buf + grad / avg
                state['momentum_buffer'] = buf
                param.data -= self.lr * buf
            else:
                param.data -= self.lr * grad / avg
        
        self.iteration += 1


class AdaGrad(Optimizer):
    """
    AdaGrad optimizer.
    
    Adapts learning rate based on historical gradients.
    Good for sparse features but learning rate can become too small.
    """
    
    def __init__(self, params: List[Parameter], lr: float = 0.01,
                 eps: float = 1e-10, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize state
        for param in self.params:
            self.state[id(param)] = {
                'sum_of_squares': np.zeros_like(param.data)
            }
    
    def step(self):
        """Perform a single optimization step."""
        for param in self.params:
            if not param.requires_grad:
                continue
                
            grad = param.grad
            
            # Add weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data
            
            state = self.state[id(param)]
            sum_of_squares = state['sum_of_squares']
            
            # Accumulate squared gradients
            sum_of_squares += grad**2
            state['sum_of_squares'] = sum_of_squares
            
            # Update parameters
            param.data -= self.lr * grad / (np.sqrt(sum_of_squares) + self.eps)
        
        self.iteration += 1


class LRScheduler(ABC):
    """Base class for learning rate schedulers."""
    
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.iteration = 0
        
    @abstractmethod
    def get_lr(self) -> float:
        """Get current learning rate."""
        pass
    
    def step(self):
        """Update learning rate."""
        self.iteration += 1
        self.optimizer.lr = self.get_lr()
        
    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state."""
        return {
            'iteration': self.iteration,
            'base_lr': self.base_lr
        }


class StepLR(LRScheduler):
    """
    Step learning rate scheduler.
    
    Decays learning rate by gamma every step_size epochs.
    """
    
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        
    def get_lr(self) -> float:
        return self.base_lr * (self.gamma ** (self.iteration // self.step_size))


class ExponentialLR(LRScheduler):
    """
    Exponential learning rate scheduler.
    
    Decays learning rate exponentially: lr = base_lr * gamma^epoch
    """
    
    def __init__(self, optimizer: Optimizer, gamma: float = 0.95):
        super().__init__(optimizer)
        self.gamma = gamma
        
    def get_lr(self) -> float:
        return self.base_lr * (self.gamma ** self.iteration)


class CosineAnnealingLR(LRScheduler):
    """
    Cosine annealing learning rate scheduler.
    
    Anneals learning rate using cosine schedule.
    """
    
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
        
    def get_lr(self) -> float:
        return self.eta_min + (self.base_lr - self.eta_min) * \
               (1 + np.cos(np.pi * self.iteration / self.T_max)) / 2


class CosineAnnealingWarmRestarts(LRScheduler):
    """
    Cosine annealing with warm restarts.
    
    Resets learning rate periodically using cosine schedule.
    """
    
    def __init__(self, optimizer: Optimizer, T_0: int, T_mult: int = 1,
                 eta_min: float = 0):
        super().__init__(optimizer)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        
    def get_lr(self) -> float:
        return self.eta_min + (self.base_lr - self.eta_min) * \
               (1 + np.cos(np.pi * self.T_cur / self.T_i)) / 2
    
    def step(self):
        """Update learning rate with warm restarts."""
        self.T_cur += 1
        
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult
            
        self.iteration += 1
        self.optimizer.lr = self.get_lr()


class OneCycleLR(LRScheduler):
    """
    One cycle learning rate scheduler.
    
    Implements the 1cycle policy: gradual warmup, then annealing.
    """
    
    def __init__(self, optimizer: Optimizer, max_lr: float, total_steps: int,
                 pct_start: float = 0.3, anneal_strategy: str = 'cos'):
        super().__init__(optimizer)
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.step_size_up = int(pct_start * total_steps)
        
    def get_lr(self) -> float:
        if self.iteration < self.step_size_up:
            # Warmup phase
            return self.base_lr + (self.max_lr - self.base_lr) * \
                   self.iteration / self.step_size_up
        else:
            # Annealing phase
            progress = (self.iteration - self.step_size_up) / \
                      (self.total_steps - self.step_size_up)
            
            if self.anneal_strategy == 'cos':
                return self.base_lr + (self.max_lr - self.base_lr) * \
                       (1 + np.cos(np.pi * progress)) / 2
            else:  # linear
                return self.max_lr - (self.max_lr - self.base_lr) * progress


def clip_grad_norm(parameters: List[Parameter], max_norm: float) -> float:
    """
    Clip gradients by global norm.
    
    Prevents gradient explosion in deep networks.
    """
    total_norm = 0.0
    
    # Calculate total norm
    for param in parameters:
        if param.requires_grad and param.grad is not None:
            param_norm = np.linalg.norm(param.grad)
            total_norm += param_norm ** 2
    
    total_norm = np.sqrt(total_norm)
    
    # Clip if necessary
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1:
        for param in parameters:
            if param.requires_grad and param.grad is not None:
                param.grad *= clip_coef
    
    return total_norm


def visualize_lr_schedules():
    """Visualize different learning rate schedules."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Dummy optimizer
    class DummyOptimizer:
        def __init__(self, lr):
            self.lr = lr
    
    base_lr = 0.1
    epochs = 100
    
    # 1. Step LR
    optimizer = DummyOptimizer(base_lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    lrs = []
    for _ in range(epochs):
        lrs.append(scheduler.get_lr())
        scheduler.step()
    
    axes[0].plot(lrs, linewidth=2)
    axes[0].set_title('Step LR (step=30, γ=0.1)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Learning Rate')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Exponential LR
    optimizer = DummyOptimizer(base_lr)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    lrs = []
    for _ in range(epochs):
        lrs.append(scheduler.get_lr())
        scheduler.step()
    
    axes[1].plot(lrs, linewidth=2)
    axes[1].set_title('Exponential LR (γ=0.95)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Cosine Annealing
    optimizer = DummyOptimizer(base_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    lrs = []
    for _ in range(epochs):
        lrs.append(scheduler.get_lr())
        scheduler.step()
    
    axes[2].plot(lrs, linewidth=2)
    axes[2].set_title('Cosine Annealing')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].grid(True, alpha=0.3)
    
    # 4. Cosine Annealing with Warm Restarts
    optimizer = DummyOptimizer(base_lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    lrs = []
    for _ in range(epochs):
        lrs.append(scheduler.get_lr())
        scheduler.step()
    
    axes[3].plot(lrs, linewidth=2)
    axes[3].set_title('Cosine Annealing with Warm Restarts')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Learning Rate')
    axes[3].grid(True, alpha=0.3)
    
    # 5. One Cycle
    optimizer = DummyOptimizer(base_lr/10)  # Start low
    scheduler = OneCycleLR(optimizer, max_lr=base_lr, total_steps=epochs)
    lrs = []
    for _ in range(epochs):
        lrs.append(scheduler.get_lr())
        scheduler.step()
    
    axes[4].plot(lrs, linewidth=2)
    axes[4].set_title('One Cycle LR')
    axes[4].set_xlabel('Epoch')
    axes[4].set_ylabel('Learning Rate')
    axes[4].grid(True, alpha=0.3)
    
    # 6. Comparison
    axes[5].plot(range(epochs), [base_lr] * epochs, label='Constant', linewidth=2)
    
    # Add a few schedules for comparison
    optimizer1 = DummyOptimizer(base_lr)
    scheduler1 = StepLR(optimizer1, step_size=30, gamma=0.5)
    lrs1 = []
    
    optimizer2 = DummyOptimizer(base_lr)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=epochs)
    lrs2 = []
    
    for _ in range(epochs):
        lrs1.append(scheduler1.get_lr())
        lrs2.append(scheduler2.get_lr())
        scheduler1.step()
        scheduler2.step()
    
    axes[5].plot(lrs1, label='Step LR', linewidth=2)
    axes[5].plot(lrs2, label='Cosine', linewidth=2)
    axes[5].set_title('Schedule Comparison')
    axes[5].set_xlabel('Epoch')
    axes[5].set_ylabel('Learning Rate')
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_optimizers_2d():
    """
    Compare optimizer trajectories on a 2D optimization problem.
    """
    # Define a simple 2D loss landscape
    def rosenbrock(x, y):
        """Rosenbrock function - a classic test for optimizers."""
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    def rosenbrock_grad(x, y):
        """Gradient of Rosenbrock function."""
        dx = -2 * (1 - x) - 400 * x * (y - x**2)
        dy = 200 * (y - x**2)
        return np.array([dx, dy])
    
    # Create meshgrid for visualization
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)
    
    # Starting point
    start = np.array([-1.5, 2.5])
    
    # Create dummy parameter for optimizers
    class DummyParam(Parameter):
        def __init__(self, data):
            super().__init__(data)
            
    # Test different optimizers
    optimizers = {
        'SGD': lambda p: SGD([p], lr=0.001),
        'SGD + Momentum': lambda p: SGD([p], lr=0.001, momentum=0.9),
        'Adam': lambda p: Adam([p], lr=0.01),
        'RMSprop': lambda p: RMSprop([p], lr=0.01)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, opt_fn) in enumerate(optimizers.items()):
        ax = axes[idx]
        
        # Initialize parameter and optimizer
        param = DummyParam(start.copy())
        optimizer = opt_fn(param)
        
        # Track trajectory
        trajectory = [param.data.copy()]
        
        # Optimize
        for _ in range(200):
            # Compute gradient
            grad = rosenbrock_grad(param.data[0], param.data[1])
            param.grad = grad
            
            # Update
            optimizer.step()
            trajectory.append(param.data.copy())
            
            # Reset gradient
            param.zero_grad()
        
        trajectory = np.array(trajectory)
        
        # Plot
        contour = ax.contour(X, Y, Z, levels=50, alpha=0.6)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, alpha=0.8)
        ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
        ax.plot(1, 1, 'r*', markersize=15, label='Optimum')
        ax.set_title(f'{name}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 3)
    
    plt.suptitle('Optimizer Comparison on Rosenbrock Function', fontsize=14)
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Visualize learning rate schedules
    print("Visualizing learning rate schedules...")
    visualize_lr_schedules()
    
    # Compare optimizers on 2D problem
    print("\nComparing optimizer trajectories...")
    compare_optimizers_2d()
    
    # Example: Using optimizers with a model
    print("\n" + "="*50)
    print("Example: Training with different optimizers")
    print("="*50)
    
    from mlp_modules import MLP
    
    # Create a simple model
    model = MLP(
        input_dim=10,
        hidden_dims=[32, 16],
        output_dim=1,
        activation='relu',
        use_batchnorm=True,
        dropout=0.2
    )
    
    # Create synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = (np.sum(X[:, :5], axis=1) > 0).astype(float).reshape(-1, 1)
    
    # Test different optimizers
    optimizers_to_test = {
        'SGD': SGD(model.parameters(), lr=0.01),
        'SGD + Momentum': SGD(model.parameters(), lr=0.01, momentum=0.9),
        'Adam': Adam(model.parameters(), lr=0.001),
        'RMSprop': RMSprop(model.parameters(), lr=0.01)
    }
    
    for opt_name, optimizer in optimizers_to_test.items():
        print(f"\nTraining with {opt_name}:")
        
        # Reset model
        for param in model.parameters():
            param.data = np.random.randn(*param.shape) * 0.1
        
        losses = []
        
        # Mini training loop
        for epoch in range(50):
            # Forward pass
            model.train()
            output = model(X)
            
            # Binary cross-entropy loss
            eps = 1e-7
            loss = -np.mean(y * np.log(output + eps) + 
                          (1 - y) * np.log(1 - output + eps))
            losses.append(loss)
            
            # Backward pass
            grad_output = -(y / (output + eps) - (1 - y) / (1 - output + eps)) / len(y)
            model.zero_grad()
            model.backward(grad_output)
            
            # Gradient clipping example
            grad_norm = clip_grad_norm(model.parameters(), max_norm=1.0)
            
            # Update
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}: Loss = {loss:.4f}, Grad norm = {grad_norm:.4f}")
    
    # Example with learning rate scheduling
    print("\n" + "="*50)
    print("Example: Learning rate scheduling")
    print("="*50)
    
    # Reset model
    for param in model.parameters():
        param.data = np.random.randn(*param.shape) * 0.1
    
    # Create optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    
    lrs = []
    losses = []
    
    for epoch in range(50):
        # Training step (simplified)
        output = model(X)
        loss = -np.mean(y * np.log(output + 1e-7) + 
                       (1 - y) * np.log(1 - output + 1e-7))
        
        grad_output = -(y / (output + 1e-7) - (1 - y) / (1 - output + 1e-7)) / len(y)
        model.zero_grad()
        model.backward(grad_output)
        
        optimizer.step()
        scheduler.step()
        
        lrs.append(optimizer.lr)
        losses.append(loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Loss = {loss:.4f}, LR = {optimizer.lr:.6f}")
    
    # Plot learning rate and loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(lrs, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('Learning Rate Schedule')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(losses, linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
