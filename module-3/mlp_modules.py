"""
MLP v5: Module-based Architecture System
========================================

This implements a PyTorch-like module system for building complex architectures:
1. Base Module class with parameter management
2. Reusable layer components
3. Flexible model composition
4. Easy experimentation with architectures

Key concepts:
- Modules encapsulate parameters and computation
- Composability through container modules
- Automatic parameter tracking
- Forward/backward through module trees
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import OrderedDict
import json
from abc import ABC, abstractmethod


class Parameter:
    """
    A parameter is a trainable tensor with gradient tracking.
    """
    def __init__(self, data: np.ndarray, requires_grad: bool = True):
        self.data = data
        self.grad = np.zeros_like(data)
        self.requires_grad = requires_grad
        self.name = None  # Set by Module
        
    def zero_grad(self):
        """Reset gradients to zero."""
        if self.requires_grad:
            self.grad.fill(0)
    
    @property
    def shape(self):
        return self.data.shape
    
    def __repr__(self):
        return f"Parameter({self.shape})"


class Module(ABC):
    """
    Base class for all neural network modules.
    
    A module can contain parameters and other modules, forming a tree structure.
    This allows building complex architectures from simple components.
    """
    
    def __init__(self):
        self._modules = OrderedDict()
        self._parameters = OrderedDict()
        self._training = True
        self._device = 'cpu'  # For future GPU support
        
    def add_module(self, name: str, module: Optional['Module']):
        """Add a child module."""
        if module is None:
            self._modules[name] = None
        elif not isinstance(module, Module):
            raise TypeError(f"{type(module)} is not a Module subclass")
        else:
            self._modules[name] = module
            
    def add_parameter(self, name: str, param: Optional[Parameter]):
        """Add a parameter."""
        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(f"{type(param)} is not a Parameter")
        else:
            param.name = name
            self._parameters[name] = param
    
    def __setattr__(self, name: str, value: Union[Parameter, 'Module', Any]):
        """Automatically register parameters and modules."""
        if isinstance(value, Parameter):
            self.add_parameter(name, value)
        elif isinstance(value, Module):
            self.add_module(name, value)
        else:
            super().__setattr__(name, value)
    
    def parameters(self, recurse: bool = True) -> List[Parameter]:
        """Return all parameters, optionally recursing through child modules."""
        params = list(self._parameters.values())
        
        if recurse:
            for module in self._modules.values():
                if module is not None:
                    params.extend(module.parameters(recurse=True))
        
        return params
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> List[Tuple[str, Parameter]]:
        """Return all parameters with their names."""
        params = []
        
        for name, param in self._parameters.items():
            if param is not None:
                full_name = f"{prefix}.{name}" if prefix else name
                params.append((full_name, param))
        
        if recurse:
            for name, module in self._modules.items():
                if module is not None:
                    submodule_prefix = f"{prefix}.{name}" if prefix else name
                    params.extend(module.named_parameters(submodule_prefix, recurse=True))
        
        return params
    
    def modules(self) -> List['Module']:
        """Return all modules in the network."""
        modules = [self]
        for module in self._modules.values():
            if module is not None:
                modules.extend(module.modules())
        return modules
    
    def train(self, mode: bool = True):
        """Set training mode for this module and all submodules."""
        self._training = mode
        for module in self._modules.values():
            if module is not None:
                module.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        return self.train(False)
    
    def zero_grad(self):
        """Zero all parameter gradients."""
        for param in self.parameters():
            param.zero_grad()
    
    def state_dict(self) -> Dict[str, np.ndarray]:
        """Return state dictionary with all parameters."""
        state = OrderedDict()
        for name, param in self.named_parameters():
            state[name] = param.data.copy()
        return state
    
    def load_state_dict(self, state_dict: Dict[str, np.ndarray]):
        """Load parameters from state dictionary."""
        for name, param in self.named_parameters():
            if name in state_dict:
                param.data = state_dict[name].copy()
            else:
                print(f"Warning: {name} not found in state_dict")
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass - must be implemented by subclasses."""
        pass
    
    def __call__(self, *args, **kwargs):
        """Make module callable."""
        return self.forward(*args, **kwargs)
    
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return ''
    
    def __repr__(self) -> str:
        """String representation."""
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        
        child_lines = []
        for key, module in self._modules.items():
            if module is None:
                child_lines.append(f'({key}): None')
            else:
                mod_str = repr(module)
                mod_str = self._addindent(mod_str, 2)
                child_lines.append(f'({key}): {mod_str}')
        
        lines = extra_lines + child_lines
        
        main_str = f'{self.__class__.__name__}('
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'
        
        main_str += ')'
        return main_str
    
    @staticmethod
    def _addindent(s: str, numSpaces: int) -> str:
        """Add indentation to string."""
        s = s.split('\n')
        if len(s) == 1:
            return s[0]
        first = s.pop(0)
        s = [(numSpaces * ' ') + line for line in s]
        s = '\n'.join(s)
        s = first + '\n' + s
        return s


class Linear(Module):
    """
    Fully connected layer: y = xW + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Initialize parameters
        # He initialization for ReLU
        std = np.sqrt(2.0 / in_features)
        self.weight = Parameter(np.random.randn(in_features, out_features) * std)
        
        if bias:
            self.bias = Parameter(np.zeros((1, out_features)))
        else:
            self.bias = None
        
        # Cache for backward pass
        self.input_cache = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: y = xW + b"""
        self.input_cache = x
        output = x @ self.weight.data
        
        if self.bias is not None:
            output += self.bias.data
            
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass to compute gradients."""
        # Gradient w.r.t. weight: x^T @ grad_output
        self.weight.grad += self.input_cache.T @ grad_output
        
        # Gradient w.r.t. bias: sum over batch
        if self.bias is not None:
            self.bias.grad += np.sum(grad_output, axis=0, keepdims=True)
        
        # Gradient w.r.t. input: grad_output @ W^T
        grad_input = grad_output @ self.weight.data.T
        
        return grad_input
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}'


class BatchNorm1d(Module):
    """
    Batch Normalization for 1D inputs (batch_size, features).
    
    Normalizes across batch dimension and learns affine transformation.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = Parameter(np.ones((1, num_features)))  # gamma
        self.bias = Parameter(np.zeros((1, num_features)))   # beta
        
        # Running statistics (not trainable)
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        
        # Cache for backward pass
        self.cache = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with different behavior for training/evaluation.
        """
        if self._training:
            # Compute batch statistics
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)
            
            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.eps)
            
            # Scale and shift
            out = self.weight.data * x_norm + self.bias.data
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                               self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                              self.momentum * batch_var
            
            # Cache for backward
            self.cache = (x, x_norm, batch_mean, batch_var)
            
        else:
            # Use running statistics
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.weight.data * x_norm + self.bias.data
            
        return out
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass - complex due to batch statistics dependencies.
        """
        if not self._training or self.cache is None:
            # Simplified backward for eval mode
            x_norm = (grad_output - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self.weight.grad += np.sum(grad_output * x_norm, axis=0, keepdims=True)
            self.bias.grad += np.sum(grad_output, axis=0, keepdims=True)
            return grad_output * self.weight.data / np.sqrt(self.running_var + self.eps)
        
        x, x_norm, batch_mean, batch_var = self.cache
        N = x.shape[0]
        
        # Gradients w.r.t. scale and shift
        self.weight.grad += np.sum(grad_output * x_norm, axis=0, keepdims=True)
        self.bias.grad += np.sum(grad_output, axis=0, keepdims=True)
        
        # Gradient w.r.t. x_norm
        dx_norm = grad_output * self.weight.data
        
        # Gradient w.r.t. variance
        dvar = np.sum(dx_norm * (x - batch_mean) * -0.5 * 
                     np.power(batch_var + self.eps, -1.5), axis=0, keepdims=True)
        
        # Gradient w.r.t. mean
        dmean = np.sum(dx_norm * -1.0 / np.sqrt(batch_var + self.eps), 
                      axis=0, keepdims=True) + \
                dvar * np.mean(-2.0 * (x - batch_mean), axis=0, keepdims=True)
        
        # Gradient w.r.t. x
        dx = dx_norm / np.sqrt(batch_var + self.eps) + \
             dvar * 2.0 * (x - batch_mean) / N + \
             dmean / N
        
        return dx
    
    def extra_repr(self) -> str:
        return f'num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}'


class Dropout(Module):
    """
    Dropout regularization layer.
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.mask = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply dropout with proper scaling."""
        if self._training and self.p > 0:
            # Create and apply dropout mask
            self.mask = np.random.binomial(1, 1 - self.p, size=x.shape) / (1 - self.p)
            return x * self.mask
        else:
            return x
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through dropout."""
        if self._training and self.p > 0 and self.mask is not None:
            return grad_output * self.mask
        else:
            return grad_output
    
    def extra_repr(self) -> str:
        return f'p={self.p}'


class ReLU(Module):
    """ReLU activation function."""
    def __init__(self):
        super().__init__()
        self.mask = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return np.maximum(0, x)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self.mask.astype(float)


class LeakyReLU(Module):
    """Leaky ReLU activation function."""
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
        self.mask = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return np.where(self.mask, x, self.negative_slope * x)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * np.where(self.mask, 1, self.negative_slope)
    
    def extra_repr(self) -> str:
        return f'negative_slope={self.negative_slope}'


class Sigmoid(Module):
    """Sigmoid activation function."""
    def __init__(self):
        super().__init__()
        self.output = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Stable sigmoid computation
        self.output = np.where(x >= 0, 
                              1 / (1 + np.exp(-x)),
                              np.exp(x) / (1 + np.exp(x)))
        return self.output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self.output * (1 - self.output)


class Sequential(Module):
    """
    Sequential container for stacking layers.
    """
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers in sequence."""
        for module in self._modules.values():
            x = module(x)
        return x
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass in reverse order."""
        # Reverse through modules
        modules = list(self._modules.values())
        for module in reversed(modules):
            grad_output = module.backward(grad_output)
        return grad_output


class Residual(Module):
    """
    Residual block: output = x + F(x)
    
    Helps with gradient flow in deep networks.
    """
    def __init__(self, module: Module):
        super().__init__()
        self.module = module
        self.input_cache = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_cache = x
        return x + self.module(x)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Gradient flows through both paths
        grad_residual = self.module.backward(grad_output)
        return grad_output + grad_residual


class MLPBlock(Module):
    """
    A standard MLP block with linear, normalization, activation, and dropout.
    """
    def __init__(self, in_features: int, out_features: int, 
                 activation: str = 'relu',
                 use_batchnorm: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        
        # Build the block
        self.linear = Linear(in_features, out_features)
        
        self.norm = BatchNorm1d(out_features) if use_batchnorm else None
        
        # Activation
        if activation == 'relu':
            self.activation = ReLU()
        elif activation == 'leaky_relu':
            self.activation = LeakyReLU()
        elif activation == 'sigmoid':
            self.activation = Sigmoid()
        else:
            self.activation = None
            
        self.dropout = Dropout(dropout) if dropout > 0 else None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.linear(x)
        
        if self.norm is not None:
            x = self.norm(x)
            
        if self.activation is not None:
            x = self.activation(x)
            
        if self.dropout is not None:
            x = self.dropout(x)
            
        return x
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.dropout is not None:
            grad_output = self.dropout.backward(grad_output)
            
        if self.activation is not None:
            grad_output = self.activation.backward(grad_output)
            
        if self.norm is not None:
            grad_output = self.norm.backward(grad_output)
            
        grad_output = self.linear.backward(grad_output)
        
        return grad_output


class MLP(Module):
    """
    Multi-layer Perceptron using the module system.
    
    Supports:
    - Arbitrary depth and width
    - Different activation functions
    - Batch normalization
    - Dropout
    - Residual connections
    """
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 activation: str = 'relu',
                 use_batchnorm: bool = False,
                 dropout: float = 0.0,
                 use_residual: bool = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = use_residual
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            block = MLPBlock(
                prev_dim, hidden_dim,
                activation=activation,
                use_batchnorm=use_batchnorm,
                dropout=dropout
            )
            
            # Add residual connection if dimensions match
            if use_residual and prev_dim == hidden_dim:
                block = Residual(block)
                
            layers.append(block)
            prev_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(Linear(prev_dim, output_dim))
        
        # If binary classification, add sigmoid
        if output_dim == 1:
            layers.append(Sigmoid())
        
        self.model = Sequential(*layers)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.model(x)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return self.model.backward(grad_output)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions (evaluation mode)."""
        self.eval()
        output = self.forward(x)
        self.train()
        return output
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(np.prod(p.shape) for p in self.parameters())


# Example usage
if __name__ == "__main__":
    # Create a complex MLP architecture
    print("Building complex MLP architecture...")
    
    model = MLP(
        input_dim=50,
        hidden_dims=[128, 64, 32],
        output_dim=1,
        activation='relu',
        use_batchnorm=True,
        dropout=0.3,
        use_residual=False
    )
    
    print("\nModel Architecture:")
    print(model)
    
    print(f"\nTotal parameters: {model.count_parameters():,}")
    
    # List all parameters
    print("\nNamed parameters:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")
    
    # Test forward pass
    batch_size = 32
    x = np.random.randn(batch_size, 50)
    
    # Training mode
    model.train()
    output_train = model(x)
    print(f"\nTraining output shape: {output_train.shape}")
    
    # Evaluation mode
    model.eval()
    output_eval = model(x)
    print(f"Evaluation output shape: {output_eval.shape}")
    
    # Test backward pass
    model.train()
    grad_output = np.random.randn(*output_train.shape)
    grad_input = model.backward(grad_output)
    print(f"\nGradient input shape: {grad_input.shape}")
    
    # Check that gradients were computed
    print("\nGradient norms:")
    for name, param in model.named_parameters()[:5]:  # First 5 parameters
        grad_norm = np.linalg.norm(param.grad)
        print(f"  {name}: {grad_norm:.6f}")
    
    # Test state dict
    state = model.state_dict()
    print(f"\nState dict keys: {len(state)}")
    
    # Create a custom architecture with residual connections
    print("\n" + "="*50)
    print("Custom Architecture with Residual Connections")
    
    class ResidualMLP(Module):
        def __init__(self):
            super().__init__()
            
            # Input projection
            self.input_proj = Linear(50, 64)
            self.bn1 = BatchNorm1d(64)
            self.act1 = ReLU()
            
            # Residual blocks
            self.res_block1 = Residual(
                Sequential(
                    Linear(64, 64),
                    BatchNorm1d(64),
                    ReLU(),
                    Dropout(0.2),
                    Linear(64, 64),
                    BatchNorm1d(64)
                )
            )
            
            self.res_block2 = Residual(
                Sequential(
                    Linear(64, 64),
                    BatchNorm1d(64),
                    ReLU(),
                    Dropout(0.2),
                    Linear(64, 64),
                    BatchNorm1d(64)
                )
            )
            
            # Output projection
            self.output_proj = Sequential(
                ReLU(),
                Linear(64, 1),
                Sigmoid()
            )
            
        def forward(self, x):
            # Input projection
            x = self.input_proj(x)
            x = self.bn1(x)
            x = self.act1(x)
            
            # Residual blocks
            x = self.res_block1(x)
            x = self.res_block2(x)
            
            # Output
            x = self.output_proj(x)
            
            return x
        
        def backward(self, grad_output):
            # Backward through output projection
            grad = self.output_proj.backward(grad_output)
            
            # Backward through residual blocks
            grad = self.res_block2.backward(grad)
            grad = self.res_block1.backward(grad)
            
            # Backward through input projection
            grad = self.act1.backward(grad)
            grad = self.bn1.backward(grad)
            grad = self.input_proj.backward(grad)
            
            return grad
    
    custom_model = ResidualMLP()
    print("\nCustom Architecture:")
    print(custom_model)
    print(f"\nTotal parameters: {sum(np.prod(p.shape) for p in custom_model.parameters()):,}")
