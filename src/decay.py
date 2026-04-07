"""
Asymmetric (position-dependent) weight decay.

For hidden neuron j (0-indexed), decay rate is:
    lambda_j = lambda_base * (1 + alpha * j / (W - 1))

j=0 gets lambda_base (lightest decay).
j=W-1 gets lambda_base * (1 + alpha) (heaviest decay).

All decay is applied externally — AdamW weight_decay MUST be 0.
"""

import torch


class AsymmetricDecay:
    def __init__(self, lambda_base: float, alpha: float,
                 reverse: bool = False, apply_to: str = 'both'):
        self.lambda_base = lambda_base
        self.alpha = alpha
        self.reverse = reverse
        self.apply_to = apply_to

    def compute_decay_mask(self, size: int) -> torch.Tensor:
        positions = torch.linspace(0, 1, size)
        if self.reverse:
            positions = 1.0 - positions
        return self.lambda_base * (1.0 + self.alpha * positions)

    def step(self, W1: torch.Tensor, W2: torch.Tensor, lr: float):
        """
        Apply position-dependent decay in-place. Call after optimizer.step().

        W1: (hidden_dim, input_dim)  — decay along axis 0 (rows = neurons)
        W2: (output_dim, hidden_dim) — decay along axis 1 (columns = neurons)
        """
        W = W1.shape[0]
        assert W == W2.shape[1], (
            f"Hidden dim mismatch: W1.shape[0]={W1.shape[0]}, "
            f"W2.shape[1]={W2.shape[1]}. Check weight matrix orientation. "
            f"Expected W1=(hidden, input), W2=(output, hidden)."
        )
        mask = self.compute_decay_mask(W).to(W1.device)

        if self.apply_to in ('both', 'W1'):
            W1.data.mul_(1.0 - lr * mask.unsqueeze(1))

        if self.apply_to in ('both', 'W2'):
            W2.data.mul_(1.0 - lr * mask.unsqueeze(0))
