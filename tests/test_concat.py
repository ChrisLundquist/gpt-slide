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
    assert cat.W2.shape == (5, 16)    # (5, 8+8)
    assert cat.b1.shape == (16,)
    assert cat.hidden_dim == 16


def test_concat_preserves_halves():
    model_a = GrokMLP(input_dim=16, hidden_dim=8, output_dim=5)
    model_b = GrokMLP(input_dim=16, hidden_dim=8, output_dim=5)

    W1_a = model_a.W1.data.clone()
    W1_b = model_b.W1.data.clone()

    cat = concatenate_experts(model_a, model_b)

    # Left half of W1 should match model_a
    assert torch.allclose(cat.W1.data[:8], W1_a)
    # Right half should match model_b
    assert torch.allclose(cat.W1.data[8:], W1_b)


def test_concat_w2_scaling():
    model_a = GrokMLP(input_dim=16, hidden_dim=8, output_dim=5)
    model_b = GrokMLP(input_dim=16, hidden_dim=8, output_dim=5)

    W2_a = model_a.W2.data.clone()
    W2_b = model_b.W2.data.clone()

    cat = concatenate_experts(model_a, model_b)

    # W2 should be scaled by 0.5
    assert torch.allclose(cat.W2.data[:, :8], W2_a * 0.5)
    assert torch.allclose(cat.W2.data[:, 8:], W2_b * 0.5)


if __name__ == '__main__':
    test_concat_shapes()
    test_concat_preserves_halves()
    test_concat_w2_scaling()
    print("All concat tests passed.")
