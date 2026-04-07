"""
Dataset generation for modular arithmetic: (a op b) mod p.
One-hot encoding with 2-bit task token appended.

Adapted from grokking-svd/data.py — same deterministic shuffle pattern.
"""

import torch


def build_dataset(p: int = 113, train_frac: float = 0.5, data_seed: int = 0,
                  device: str = 'cuda', operation: str = 'add'):
    """
    Build full-batch training and test sets for modular arithmetic.

    Input encoding: one-hot(a) || one-hot(b) || task_token
    - a, b ∈ {0, ..., p-1}
    - task_token: [1, 0] for addition, [0, 1] for multiplication

    Args:
        operation: 'add' for (a+b) mod p, 'mul' for (a*b) mod p

    Returns:
        X_train: (n_train, 2*p + 2) tensor
        y_train: (n_train,) tensor
        X_test: (n_test, 2*p + 2) tensor
        y_test: (n_test,) tensor
    """
    all_a = torch.arange(p).repeat_interleave(p)
    all_b = torch.arange(p).repeat(p)

    if operation == 'add':
        all_labels = (all_a + all_b) % p
        task_token = torch.tensor([1.0, 0.0])
    elif operation == 'mul':
        all_labels = (all_a * all_b) % p
        task_token = torch.tensor([0.0, 1.0])
    else:
        raise ValueError(f'Unknown operation: {operation}')

    # One-hot encode (a, b) and append task token
    n_total = p * p
    input_dim = 2 * p + 2
    X = torch.zeros(n_total, input_dim)
    X[torch.arange(n_total), all_a] = 1.0             # one-hot a
    X[torch.arange(n_total), p + all_b] = 1.0         # one-hot b
    X[:, 2 * p:] = task_token.unsqueeze(0)             # task token

    # Deterministic split
    n_train = round(n_total * train_frac)
    rng = torch.Generator()
    rng.manual_seed(data_seed)
    perm = torch.randperm(n_total, generator=rng)

    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    X_train = X[train_idx].to(device)
    y_train = all_labels[train_idx].to(device)
    X_test = X[test_idx].to(device)
    y_test = all_labels[test_idx].to(device)

    return X_train, y_train, X_test, y_test


def build_joint_dataset(p: int = 113, train_frac: float = 0.5, data_seed: int = 0,
                        device: str = 'cuda'):
    """
    Build datasets for both addition and multiplication.
    Same train/test split (same pairs), different labels and task tokens.

    Returns:
        dict with keys 'add' and 'mul', each containing
        (X_train, y_train, X_test, y_test)
    """
    return {
        'add': build_dataset(p, train_frac, data_seed, device, 'add'),
        'mul': build_dataset(p, train_frac, data_seed, device, 'mul'),
    }
