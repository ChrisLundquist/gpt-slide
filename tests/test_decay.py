"""Tests for AsymmetricDecay."""

import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.decay import AsymmetricDecay


def test_mask_monotonic():
    decay = AsymmetricDecay(lambda_base=1.0, alpha=2.0)
    mask = decay.compute_decay_mask(8)
    for i in range(len(mask) - 1):
        assert mask[i] <= mask[i + 1], "Mask should be monotonically increasing"


def test_mask_reversed():
    decay = AsymmetricDecay(lambda_base=1.0, alpha=2.0, reverse=True)
    mask = decay.compute_decay_mask(8)
    for i in range(len(mask) - 1):
        assert mask[i] >= mask[i + 1], "Reversed mask should be decreasing"


def test_mask_uniform():
    decay = AsymmetricDecay(lambda_base=1.0, alpha=0.0)
    mask = decay.compute_decay_mask(8)
    assert torch.allclose(mask, torch.ones(8)), "alpha=0 should give uniform mask"


def test_shape_assertion():
    decay = AsymmetricDecay(lambda_base=1.0, alpha=2.0)
    W1 = torch.randn(8, 16)   # (hidden=8, input=16)
    W2_wrong = torch.randn(5, 4)  # hidden dim mismatch (4 != 8)
    with pytest.raises(AssertionError):
        decay.step(W1, W2_wrong, lr=1e-3)


def test_decay_reduces_norms():
    decay = AsymmetricDecay(lambda_base=1.0, alpha=2.0)
    W1 = torch.randn(8, 16)
    W2 = torch.randn(5, 8)
    norm_before = W1.norm().item()
    decay.step(W1, W2, lr=1e-3)
    norm_after = W1.norm().item()
    assert norm_after < norm_before, "Decay should reduce norms"


def test_asymmetric_direction():
    """Right-side neurons should be decayed more than left-side."""
    decay = AsymmetricDecay(lambda_base=1.0, alpha=2.0)
    W1 = torch.ones(8, 16)
    W2 = torch.ones(5, 8)
    decay.step(W1, W2, lr=0.1)
    left_mean = W1[:4].norm(dim=1).mean()
    right_mean = W1[4:].norm(dim=1).mean()
    assert left_mean > right_mean, "Left neurons should have higher norms after decay"


if __name__ == '__main__':
    test_mask_monotonic()
    test_mask_reversed()
    test_mask_uniform()
    test_decay_reduces_norms()
    test_asymmetric_direction()
    print("All decay tests passed.")
