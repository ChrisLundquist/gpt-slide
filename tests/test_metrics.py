"""Tests for Fourier analysis metrics."""

import torch
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metrics import (
    neuron_fourier_spectrum, compute_ipr, compute_left_energy,
    compute_column_norms, compute_effective_rank,
)


def test_ipr_single_frequency():
    """A pure single-frequency neuron should have IPR close to 1.0."""
    p = 7
    # Create a W1 row that's a pure cosine at frequency 2
    w = torch.zeros(2 * p + 2)
    for i in range(p):
        w[i] = math.cos(2 * math.pi * 2 * i / p)
        w[p + i] = math.cos(2 * math.pi * 2 * i / p)
    spec = neuron_fourier_spectrum(w, p)
    ipr = compute_ipr(spec)
    assert ipr > 0.4, f"Single-frequency IPR should be high, got {ipr}"


def test_ipr_uniform():
    """Uniform weights should have low IPR."""
    p = 11
    w = torch.ones(2 * p + 2)
    spec = neuron_fourier_spectrum(w, p)
    ipr = compute_ipr(spec)
    # Uniform input gives all energy at DC, which is excluded
    # So the remaining spectrum should be ~uniform → low IPR
    assert ipr < 0.5, f"Uniform weights IPR should be low, got {ipr}"


def test_left_energy_placement():
    """If all Fourier energy is in left-half neurons, left_energy should be ~1.0."""
    p = 7
    hidden = 8
    W1 = torch.zeros(hidden, 2 * p + 2)
    # Put cosine structure in left 4 neurons only
    for j in range(4):
        for i in range(p):
            W1[j, i] = math.cos(2 * math.pi * (j + 1) * i / p)
            W1[j, p + i] = math.sin(2 * math.pi * (j + 1) * i / p)

    le = compute_left_energy(W1, p)
    assert le > 0.9, f"Left energy should be ~1.0 when features are in left half, got {le}"


def test_column_norms():
    W1 = torch.tensor([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    norms = compute_column_norms(W1)
    assert abs(norms[0].item() - 5.0) < 1e-5
    assert abs(norms[1].item() - 0.0) < 1e-5
    assert abs(norms[2].item() - 1.0) < 1e-5


def test_effective_rank():
    # Identity-like matrix should have rank = n
    W = torch.eye(5)
    rank = compute_effective_rank(W)
    assert abs(rank - 5.0) < 0.1, f"Identity should have rank ~5, got {rank}"

    # Rank-1 matrix
    W = torch.ones(5, 5)
    rank = compute_effective_rank(W)
    assert abs(rank - 1.0) < 0.1, f"Rank-1 should have rank ~1, got {rank}"


if __name__ == '__main__':
    test_ipr_single_frequency()
    test_ipr_uniform()
    test_left_energy_placement()
    test_column_norms()
    test_effective_rank()
    print("All metrics tests passed.")
