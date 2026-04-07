"""
Forward hooks and weight manipulation for Step 3 conditions.

Activation-zeroing: blocks information channel from dying neurons.
Weight-freezing: preserves dying neurons' forward pass but prevents learning/decay.
"""

import torch


def register_activation_zeroing_hook(model, hidden_dim: int, quartile: float = 0.75):
    """
    Clamp activations to zero for dying neurons (top quartile).

    These neurons still have weights and receive decay, but receive
    zero gradients (loss has no dependence on zeroed activations).
    Their weights can only shrink toward zero.

    Args:
        model: GrokMLP instance
        hidden_dim: total hidden dimension (e.g., 256 for concatenated)
        quartile: fraction below which neurons are kept active

    Returns:
        hook handle (call .remove() to detach)
    """
    cutoff = int(hidden_dim * quartile)

    def zero_activations(module, input, output):
        output = output.clone()
        output[:, cutoff:] = 0.0
        return output

    return model.activation.register_forward_hook(zero_activations)


def register_weight_freeze_hook(model, hidden_dim: int, quartile: float = 0.75):
    """
    Freeze weights for dying neurons (top quartile).

    These neurons participate in the forward pass with their initial weights
    but cannot learn or decay — weights are restored after each step.

    Args:
        model: GrokMLP instance
        hidden_dim: total hidden dimension
        quartile: fraction below which neurons are free to update

    Returns:
        restore_fn: callable, invoke after optimizer.step() and decay.step()
    """
    cutoff = int(hidden_dim * quartile)

    W1_frozen = model.W1.data[cutoff:].clone()
    W2_frozen = model.W2.data[:, cutoff:].clone()
    b1_frozen = model.b1.data[cutoff:].clone()

    def restore_weights():
        model.W1.data[cutoff:] = W1_frozen
        model.W2.data[:, cutoff:] = W2_frozen
        model.b1.data[cutoff:] = b1_frozen

    return restore_weights
