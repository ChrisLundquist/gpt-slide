"""Tests for data generation."""

import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import build_dataset, build_joint_dataset


def test_shapes():
    p = 7
    X_train, y_train, X_test, y_test = build_dataset(p=p, train_frac=0.5,
                                                       data_seed=0, device='cpu')
    n_total = p * p
    assert X_train.shape[0] + X_test.shape[0] == n_total
    assert X_train.shape[1] == 2 * p  # one-hot a + one-hot b
    assert y_train.max() < p
    assert y_train.min() >= 0


def test_deterministic():
    X1, y1, _, _ = build_dataset(p=7, data_seed=0, device='cpu')
    X2, y2, _, _ = build_dataset(p=7, data_seed=0, device='cpu')
    assert torch.equal(X1, X2)
    assert torch.equal(y1, y2)


def test_one_hot():
    p = 7
    X, _, _, _ = build_dataset(p=p, device='cpu')
    assert (X[:, :p].sum(dim=1) == 1.0).all()
    assert (X[:, p:2*p].sum(dim=1) == 1.0).all()


def test_labels():
    p = 7
    X, y, _, _ = build_dataset(p=p, device='cpu', operation='add')
    a = X[:, :p].argmax(dim=1)
    b = X[:, p:2*p].argmax(dim=1)
    expected = (a + b) % p
    assert torch.equal(y, expected)


def test_joint_dataset():
    p = 7
    data = build_joint_dataset(p=p, device='cpu')
    assert 'add' in data and 'mul' in data
    # Same (a,b) pairs for both tasks
    assert torch.equal(data['add'][0], data['mul'][0])


def test_split_ratio():
    p = 11
    X_train, _, X_test, _ = build_dataset(p=p, train_frac=0.5, device='cpu')
    n_total = p * p
    assert X_train.shape[0] == round(n_total * 0.5)
    assert X_test.shape[0] == n_total - round(n_total * 0.5)


if __name__ == '__main__':
    for fn in [test_shapes, test_deterministic, test_one_hot, test_labels,
               test_joint_dataset, test_split_ratio]:
        fn()
    print("All data tests passed.")
