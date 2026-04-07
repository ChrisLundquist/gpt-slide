"""Tests for GrokMLP model."""

import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import GrokMLP


def test_forward_shape():
    p = 7
    model = GrokMLP(input_dim=2*p+2, hidden_dim=4, output_dim=p)
    x = torch.randn(10, 2*p+2)
    out = model(x)
    assert out.shape == (10, p)


def test_weight_shapes():
    model = GrokMLP(input_dim=16, hidden_dim=8, output_dim=5)
    assert model.W1.shape == (8, 16)   # (hidden, input)
    assert model.W2.shape == (5, 8)    # (output, hidden)
    assert model.b1.shape == (8,)
    assert model.b2.shape == (5,)


def test_quadratic_activation():
    model = GrokMLP(input_dim=4, hidden_dim=3, output_dim=2)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    # Manually compute layer1 output
    h_linear = model.layer1(x)
    h_squared = h_linear ** 2

    # Verify activation is x²
    h_via_model = model.activation(h_linear)
    assert torch.allclose(h_via_model, h_squared)


def test_hidden_dim_property():
    model = GrokMLP(input_dim=16, hidden_dim=32, output_dim=8)
    assert model.hidden_dim == 32


if __name__ == '__main__':
    test_forward_shape()
    test_weight_shapes()
    test_quadratic_activation()
    test_hidden_dim_property()
    print("All model tests passed.")
