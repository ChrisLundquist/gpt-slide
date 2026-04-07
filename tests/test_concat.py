"""Tests for expert concatenation."""

import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import GrokMLP
from src.concat import concatenate_experts


def test_concat_shapes():
    model_a = GrokMLP(input_dim=16, hidden_dim=8, output_dim=5)
    model_b = GrokMLP(input_dim=16, hidden_dim=8, output_dim=5)
    cat = concatenate_experts(model_a, model_b)

    assert cat.W1.shape == (16, 16)   # (8+8, 16)
    assert cat._joint is True
    assert cat.head_add.weight.shape == (5, 16)
    assert cat.head_mul.weight.shape == (5, 16)
    assert cat.hidden_dim == 16


def test_concat_preserves_w1():
    model_a = GrokMLP(input_dim=16, hidden_dim=8, output_dim=5)
    model_b = GrokMLP(input_dim=16, hidden_dim=8, output_dim=5)

    W1_a = model_a.W1.data.clone()
    W1_b = model_b.W1.data.clone()

    cat = concatenate_experts(model_a, model_b)

    assert torch.allclose(cat.W1.data[:8], W1_a)
    assert torch.allclose(cat.W1.data[8:], W1_b)


def test_concat_head_init():
    """Add head reads from left half, mul head from right half."""
    model_a = GrokMLP(input_dim=16, hidden_dim=8, output_dim=5)
    model_b = GrokMLP(input_dim=16, hidden_dim=8, output_dim=5)

    W2_a = model_a.W2.data.clone()
    W2_b = model_b.W2.data.clone()

    cat = concatenate_experts(model_a, model_b)

    # Add head: left=W2_a, right=zeros
    assert torch.allclose(cat.head_add.weight.data[:, :8], W2_a)
    assert (cat.head_add.weight.data[:, 8:] == 0).all()

    # Mul head: left=zeros, right=W2_b
    assert (cat.head_mul.weight.data[:, :8] == 0).all()
    assert torch.allclose(cat.head_mul.weight.data[:, 8:], W2_b)


def test_concat_forward():
    model_a = GrokMLP(input_dim=16, hidden_dim=8, output_dim=5)
    model_b = GrokMLP(input_dim=16, hidden_dim=8, output_dim=5)
    cat = concatenate_experts(model_a, model_b)

    x = torch.randn(4, 16)
    out_add = cat(x, task='add')
    out_mul = cat(x, task='mul')
    assert out_add.shape == (4, 5)
    assert out_mul.shape == (4, 5)


if __name__ == '__main__':
    for fn in [test_concat_shapes, test_concat_preserves_w1,
               test_concat_head_init, test_concat_forward]:
        fn()
    print("All concat tests passed.")
