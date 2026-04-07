"""
Expert concatenation: stack two trained experts along hidden dimension.

The joint model has two output heads (one per task) to avoid gradient conflicts.
Add-expert neurons are indices 0..W_a-1 (left side).
Mul-expert neurons are indices W_a..W_a+W_b-1 (right side).
"""

import torch
from .model import GrokMLP


def concatenate_experts(model_add: GrokMLP, model_mul: GrokMLP) -> GrokMLP:
    """
    Concatenate two single-task experts into a joint model with two output heads.

    W1: stack rows (hidden dim)
    head_add: initialized from add expert's W2, zeros for mul half
    head_mul: initialized from mul expert's W2, zeros for add half
    """
    W1_a, W1_b = model_add.W1.data, model_mul.W1.data
    W2_a, W2_b = model_add.W2.data, model_mul.W2.data
    b1_a, b1_b = model_add.b1.data, model_mul.b1.data
    b2_a, b2_b = model_add.b2.data, model_mul.b2.data

    input_dim = W1_a.shape[1]
    output_dim = W2_a.shape[0]
    W_a = W1_a.shape[0]
    W_b = W1_b.shape[0]
    hidden_dim = W_a + W_b

    cat_model = GrokMLP(input_dim, hidden_dim, output_dim, joint=True)

    # Shared hidden layer: stack both experts' W1
    cat_model.layer1.weight.data = torch.cat([W1_a, W1_b], dim=0)
    cat_model.layer1.bias.data = torch.cat([b1_a, b1_b], dim=0)

    # Add head: reads from add neurons (left), zeros for mul neurons (right)
    cat_model.head_add.weight.data = torch.cat(
        [W2_a, torch.zeros(output_dim, W_b, device=W2_a.device)], dim=1)
    cat_model.head_add.bias.data = b2_a.clone()

    # Mul head: zeros for add neurons (left), reads from mul neurons (right)
    cat_model.head_mul.weight.data = torch.cat(
        [torch.zeros(output_dim, W_a, device=W2_b.device), W2_b], dim=1)
    cat_model.head_mul.bias.data = b2_b.clone()

    return cat_model
