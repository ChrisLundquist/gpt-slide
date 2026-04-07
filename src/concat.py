"""
Expert concatenation: stack two trained experts along hidden dimension.

After concatenation, add-expert neurons are indices 0..W_a-1 (left side),
mul-expert neurons are indices W_a..W_a+W_b-1 (right side).
"""

import torch
from .model import GrokMLP


def concatenate_experts(model_add: GrokMLP, model_mul: GrokMLP) -> GrokMLP:
    """
    Concatenate two experts into a wider model.

    W1: stack rows (hidden dim):  (W_a, input) + (W_b, input) → (W_a+W_b, input)
    W2: stack cols, scale by 0.5: (output, W_a) + (output, W_b) → (output, W_a+W_b)
    b1: concatenate
    b2: average
    """
    W1_a, W1_b = model_add.W1.data, model_mul.W1.data
    W2_a, W2_b = model_add.W2.data, model_mul.W2.data
    b1_a, b1_b = model_add.b1.data, model_mul.b1.data
    b2_a, b2_b = model_add.b2.data, model_mul.b2.data

    input_dim = W1_a.shape[1]
    output_dim = W2_a.shape[0]
    hidden_dim = W1_a.shape[0] + W1_b.shape[0]

    cat_model = GrokMLP(input_dim, hidden_dim, output_dim)

    cat_model.layer1.weight.data = torch.cat([W1_a, W1_b], dim=0)
    cat_model.layer1.bias.data = torch.cat([b1_a, b1_b], dim=0)
    cat_model.layer2.weight.data = torch.cat([W2_a * 0.5, W2_b * 0.5], dim=1)
    cat_model.layer2.bias.data = (b2_a + b2_b) * 0.5

    return cat_model


def calibrate_mixing(model: GrokMLP, add_test_X, add_test_y,
                     mul_test_X, mul_test_y, half: int):
    """
    Check logit scale ratio between expert halves.
    If ratio > 2:1, learn per-expert scalars to equalize contributions.

    Args:
        model: concatenated model
        half: boundary index (e.g., 128 for 128+128 concatenation)
    """
    model.eval()
    with torch.no_grad():
        # Get per-half logit contributions
        W2_full = model.W2.data.clone()

        # Left half only (add expert)
        model.W2.data[:, half:] = 0.0
        logits_left_add = model(add_test_X)
        logits_left_mul = model(mul_test_X)

        # Right half only (mul expert)
        model.W2.data.copy_(W2_full)
        model.W2.data[:, :half] = 0.0
        logits_right_add = model(add_test_X)
        logits_right_mul = model(mul_test_X)

        # Restore
        model.W2.data.copy_(W2_full)

        # Logit scale ratio
        left_scale = logits_left_add.abs().mean().item()
        right_scale = logits_right_mul.abs().mean().item()

        if left_scale < 1e-8 or right_scale < 1e-8:
            return 1.0

        ratio = max(left_scale, right_scale) / min(left_scale, right_scale)

    if ratio > 2.0:
        # Rescale the larger half down
        if left_scale > right_scale:
            scale = right_scale / left_scale
            model.W2.data[:, :half] *= scale
            model.b2.data *= (scale + 1.0) / 2.0
        else:
            scale = left_scale / right_scale
            model.W2.data[:, half:] *= scale
            model.b2.data *= (1.0 + scale) / 2.0

    return ratio
