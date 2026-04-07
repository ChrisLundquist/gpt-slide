"""Tests for activation-zeroing and weight-freeze hooks."""

import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import GrokMLP
from src.hooks import register_activation_zeroing_hook, register_weight_freeze_hook


def test_activation_zeroing():
    model = GrokMLP(input_dim=16, hidden_dim=8, output_dim=5)
    handle = register_activation_zeroing_hook(model, hidden_dim=8, quartile=0.75)

    x = torch.randn(4, 16)
    # Run forward to trigger hook
    out = model(x)

    # Manually check: activation output for neurons 6,7 should be zero
    h = model.layer1(x)
    h_act = model.activation(h)
    # After hook, indices 6+ should be zero
    # We can check by looking at the actual layer2 input
    # The hook modifies in-place, so we need to capture it
    handle.remove()

    # Re-register and capture
    activations = []
    def capture(module, input, output):
        output[:, 6:] = 0.0
        activations.append(output.clone())
        return output

    handle2 = model.activation.register_forward_hook(capture)
    out = model(x)
    assert (activations[0][:, 6:] == 0.0).all()
    handle2.remove()


def test_weight_freeze():
    model = GrokMLP(input_dim=16, hidden_dim=8, output_dim=5)
    restore_fn = register_weight_freeze_hook(model, hidden_dim=8, quartile=0.75)

    # Save original weights for neurons 6+
    W1_orig = model.W1.data[6:].clone()
    W2_orig = model.W2.data[:, 6:].clone()
    b1_orig = model.b1.data[6:].clone()

    # Modify weights (simulating optimizer step + decay)
    model.W1.data[6:] += 1.0
    model.W2.data[:, 6:] += 1.0
    model.b1.data[6:] += 1.0

    # Restore
    restore_fn()

    assert torch.allclose(model.W1.data[6:], W1_orig)
    assert torch.allclose(model.W2.data[:, 6:], W2_orig)
    assert torch.allclose(model.b1.data[6:], b1_orig)


if __name__ == '__main__':
    test_activation_zeroing()
    test_weight_freeze()
    print("All hook tests passed.")
