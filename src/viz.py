"""
Visualization: column norm heatmaps and comparison plots.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_column_norm_heatmap(metrics_log: list[dict], title: str, save_path: str):
    """
    Plot column norms over training as a heatmap.
    x: neuron index, y: training step, color: L2 norm.

    This is the primary diagnostic plot for the experiment.
    """
    steps = [m['step'] for m in metrics_log if 'column_norms' in m]
    norms = [m['column_norms'] for m in metrics_log if 'column_norms' in m]

    if not norms:
        return

    data = np.array(norms)  # (num_steps, hidden_dim)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(data, aspect='auto', cmap='viridis',
                   extent=[0, data.shape[1], steps[-1], steps[0]])
    ax.set_xlabel('Neuron index')
    ax.set_ylabel('Training step')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='L2 norm')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_side_by_side_heatmaps(conditions: dict[str, list[dict]],
                                save_path: str):
    """
    Side-by-side column norm heatmaps for multiple conditions.
    Used for Gate 2 direction comparison.

    Args:
        conditions: dict of condition_name → metrics_log
    """
    n = len(conditions)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (name, metrics_log) in zip(axes, conditions.items()):
        norms = [m['column_norms'] for m in metrics_log if 'column_norms' in m]
        steps = [m['step'] for m in metrics_log if 'column_norms' in m]
        if not norms:
            continue
        data = np.array(norms)
        im = ax.imshow(data, aspect='auto', cmap='viridis',
                       extent=[0, data.shape[1], steps[-1], steps[0]])
        ax.set_xlabel('Neuron index')
        ax.set_ylabel('Training step')
        ax.set_title(name)
        fig.colorbar(im, ax=ax, label='L2 norm')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
