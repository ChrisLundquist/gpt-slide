"""
Dataset generation for modular arithmetic: (a op b) mod p.
One-hot encoding of (a, b). No task token — task routing is via output heads.

Adapted from grokking-svd/data.py — same deterministic shuffle pattern.
"""

import torch


def build_dataset(p: int = 113, train_frac: float = 0.5, data_seed: int = 0,
                  device: str = 'cuda', operation: str = 'add'):
    """
    Build full-batch training and test sets for modular arithmetic.

    Input encoding: one-hot(a) || one-hot(b), shape (N, 2*p)

    Returns:
        X_train, y_train, X_test, y_test
    """
    all_a = torch.arange(p).repeat_interleave(p)
    all_b = torch.arange(p).repeat(p)

    if operation == 'add':
        all_labels = (all_a + all_b) % p
    elif operation == 'mul':
        all_labels = (all_a * all_b) % p
    else:
        raise ValueError(f'Unknown operation: {operation}')

    # One-hot encode (a, b)
    n_total = p * p
    input_dim = 2 * p
    X = torch.zeros(n_total, input_dim)
    X[torch.arange(n_total), all_a] = 1.0
    X[torch.arange(n_total), p + all_b] = 1.0

    # Deterministic split
    n_train = round(n_total * train_frac)
    rng = torch.Generator()
    rng.manual_seed(data_seed)
    perm = torch.randperm(n_total, generator=rng)

    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    return (X[train_idx].to(device), all_labels[train_idx].to(device),
            X[test_idx].to(device), all_labels[test_idx].to(device))


def build_joint_dataset(p: int = 113, train_frac: float = 0.5, data_seed: int = 0,
                        device: str = 'cuda'):
    """
    Build datasets for both tasks. Same (a,b) pairs, different labels.
    """
    return {
        'add': build_dataset(p, train_frac, data_seed, device, 'add'),
        'mul': build_dataset(p, train_frac, data_seed, device, 'mul'),
    }
