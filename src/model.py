"""
2-layer MLP with quadratic activation for modular arithmetic grokking.
One-hot input with task token.
"""

import torch
import torch.nn as nn


class QuadraticActivation(nn.Module):
    """x → x². Named module so forward hooks can be registered."""
    def forward(self, x):
        return x ** 2


class GrokMLP(nn.Module):
    """
    2-layer MLP: input → W1 → x² → W2 → output

    Weight shapes (via nn.Linear):
        layer1.weight: (hidden_dim, input_dim)  = W1
        layer2.weight: (output_dim, hidden_dim) = W2
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = QuadraticActivation()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    @property
    def W1(self) -> torch.Tensor:
        """Input weights: (hidden_dim, input_dim)"""
        return self.layer1.weight

    @property
    def W2(self) -> torch.Tensor:
        """Output weights: (output_dim, hidden_dim)"""
        return self.layer2.weight

    @property
    def b1(self) -> torch.Tensor:
        """Hidden bias: (hidden_dim,)"""
        return self.layer1.bias

    @property
    def b2(self) -> torch.Tensor:
        """Output bias: (output_dim,)"""
        return self.layer2.bias

    @property
    def hidden_dim(self) -> int:
        return self.layer1.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x)
        h = self.activation(h)
        return self.layer2(h)
