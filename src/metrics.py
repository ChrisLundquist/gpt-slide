"""
Fourier analysis and migration metrics for grokking MLPs.

All Fourier analysis operates on the first 2*p columns of W1 (ignoring the
2-column task token). Each W1 row is reshaped to (2, p) for per-input DFT.

Effective rank copied from grokking-svd/metrics.py.
"""

import torch
import numpy as np


def neuron_fourier_spectrum(W1_row: torch.Tensor, p: int) -> torch.Tensor:
    """
    Compute normalized Fourier power spectrum for one hidden neuron.

    Args:
        W1_row: (input_dim,) — full input weights including task token
        p: prime modulus

    Returns:
        spectrum: (p,) — normalized power at each frequency
    """
    w = W1_row[:2 * p].reshape(2, p).float()
    W_hat = torch.fft.fft(w, dim=-1)
    power = (W_hat.abs() ** 2).sum(dim=0)  # sum over 2 inputs
    total = power.sum()
    if total < 1e-10:
        return torch.zeros(p)
    return power / total


def compute_ipr(spectrum: torch.Tensor) -> float:
    """
    Inverse Participation Ratio of a Fourier spectrum.
    1.0 = single frequency, ~1/(p-1) = uniform.
    Excludes DC component.
    """
    s = spectrum[1:]
    total = s.sum()
    if total < 1e-10:
        return 0.0
    s = s / total
    return (s ** 2).sum().item()


def compute_column_norms(W1: torch.Tensor) -> torch.Tensor:
    """Per-neuron L2 norms: ||W1[j, :]||_2 for each hidden neuron j."""
    return W1.norm(dim=1)


def frequency_energy_matrix(W1: torch.Tensor, p: int) -> torch.Tensor:
    """
    Full (num_neurons, p) energy matrix for life history tracking.

    Returns:
        energy: (hidden_dim, p) — normalized power at each frequency per neuron
    """
    hidden_dim = W1.shape[0]
    return torch.stack([neuron_fourier_spectrum(W1[j], p) for j in range(hidden_dim)])


def compute_left_energy(W1: torch.Tensor, p: int) -> float:
    """
    Fraction of total Fourier energy in the left half of hidden neurons.
    Excludes DC component.
    """
    hidden_dim = W1.shape[0]
    half = hidden_dim // 2
    total, left = 0.0, 0.0
    for j in range(hidden_dim):
        spec = neuron_fourier_spectrum(W1[j], p)
        e = spec[1:].sum().item()
        total += e
        if j < half:
            left += e
    return left / total if total > 0 else 0.5


def soft_migration_score(energy_before: torch.Tensor, energy_after: torch.Tensor,
                         p: int) -> tuple[float, float]:
    """
    Energy-weighted mean neuron index per frequency, before vs after.
    Positive = leftward migration (toward lower indices).

    Args:
        energy_before: (hidden_dim, p) from frequency_energy_matrix
        energy_after: (hidden_dim, p) from frequency_energy_matrix

    Returns:
        (mean_shift, std_shift)
    """
    W = energy_before.shape[0]
    indices = torch.arange(W, dtype=torch.float32)
    shifts = []
    for f in range(1, p):  # skip DC
        eb, ea = energy_before[:, f], energy_after[:, f]
        if eb.sum() < 1e-8 or ea.sum() < 1e-8:
            continue
        mu_b = (indices * eb).sum() / eb.sum()
        mu_a = (indices * ea).sum() / ea.sum()
        shifts.append((mu_b - mu_a).item())
    if not shifts:
        return 0.0, 0.0
    return float(np.mean(shifts)), float(np.std(shifts))


def accuracy_after_zeroing(model, X: torch.Tensor, y: torch.Tensor,
                           threshold: int) -> float:
    """
    Zero columns j >= threshold in W1 (and corresponding W2 columns),
    evaluate accuracy, then restore weights.
    """
    W1_backup = model.W1.data.clone()
    W2_backup = model.W2.data.clone()
    b1_backup = model.b1.data.clone()

    model.W1.data[threshold:] = 0.0
    model.W2.data[:, threshold:] = 0.0
    model.b1.data[threshold:] = 0.0

    with torch.no_grad():
        logits = model(X)
        acc = (logits.argmax(-1) == y).float().mean().item()

    model.W1.data.copy_(W1_backup)
    model.W2.data.copy_(W2_backup)
    model.b1.data.copy_(b1_backup)

    return acc


def compute_effective_rank(W: torch.Tensor) -> float:
    """
    Effective rank via Shannon entropy of normalized singular values.
    R_eff = exp(H(sigma)), where H = -sum(p_i * log(p_i)).

    Copied from grokking-svd/metrics.py.
    """
    if W.dim() == 3:
        W = W.reshape(-1, W.shape[-1])
    S = torch.linalg.svdvals(W)
    S = S[S > 1e-10]
    if len(S) == 0:
        return 0.0
    p = S / S.sum()
    entropy = -(p * torch.log(p)).sum().item()
    return float(torch.exp(torch.tensor(entropy)).item())
