"""
2-layer MLP with quadratic activation for modular arithmetic grokking.

Single-task experts use one output head. The concatenated joint model uses
two output heads (one per task) to avoid gradient conflicts.
"""

import torch
import torch.nn as nn


class QuadraticActivation(nn.Module):
    """x -> x**2. Named module so forward hooks can be registered."""
    def forward(self, x):
        return x ** 2


class GrokMLP(nn.Module):
    """
    2-layer MLP: input -> W1 -> x**2 -> W2 -> output

    For single-task experts: one output head (layer2).
    For joint model: two output heads (head_add, head_mul), selected by task arg.

    Weight shapes (via nn.Linear):
        layer1.weight: (hidden_dim, input_dim)  = W1
        layer2.weight: (output_dim, hidden_dim) = W2 (single-task)
        head_add.weight / head_mul.weight: (output_dim, hidden_dim) (joint)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 joint: bool = False):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = QuadraticActivation()
        self._joint = joint

        if joint:
            self.head_add = nn.Linear(hidden_dim, output_dim)
            self.head_mul = nn.Linear(hidden_dim, output_dim)
            self.layer2 = None  # not used in joint mode
        else:
            self.layer2 = nn.Linear(hidden_dim, output_dim)
            self.head_add = None
            self.head_mul = None

    @property
    def W1(self) -> torch.Tensor:
        return self.layer1.weight

    @property
    def W2(self) -> torch.Tensor:
        """Output weights for single-task model."""
        if self.layer2 is not None:
            return self.layer2.weight
        # For joint model, return add head by default (for decay application)
        return self.head_add.weight

    @property
    def b1(self) -> torch.Tensor:
        return self.layer1.bias

    @property
    def b2(self) -> torch.Tensor:
        if self.layer2 is not None:
            return self.layer2.bias
        return self.head_add.bias

    @property
    def hidden_dim(self) -> int:
        return self.layer1.out_features

    def forward(self, x: torch.Tensor, task: str = None) -> torch.Tensor:
        h = self.layer1(x)
        h = self.activation(h)
        if self._joint:
            if task == 'add':
                return self.head_add(h)
            elif task == 'mul':
                return self.head_mul(h)
            else:
                raise ValueError(f"Joint model requires task='add' or 'mul', got {task!r}")
        return self.layer2(h)
