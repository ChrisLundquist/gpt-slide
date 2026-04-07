"""
Experiment B: Stop-Gradient-Assisted Structured Pruning

Sever-and-decay: detach dying neurons from backward pass, apply decay only
to severed neurons. Tests whether this beats standard pruning baselines.
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import dataclasses

torch.set_float32_matmul_precision('high')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config, SEEDS
from src.model import GrokMLP
from src.data import build_dataset
from src.train import set_seed, evaluate
from src.metrics import (
    compute_column_norms, neuron_fourier_spectrum, compute_ipr,
    accuracy_after_zeroing,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_STEPS = 20_000
LOG_EVERY = 100
LAMBDA = 0.1
LR = 1e-3
P = 113
SEVER_FRACTIONS = [16, 32, 48, 64, 80, 96]
BASELINE_FRACTIONS = [32, 64, 96]
PARETO_THRESHOLDS = [16, 32, 48, 64, 80, 96, 112]


def load_expert(seed):
    path = f'outputs/step1/experts/add_seed{seed}/checkpoint.pt'
    ckpt = torch.load(path, weights_only=True)
    model = GrokMLP(2 * P, 128, P).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    return model


def get_data():
    return build_dataset(P, 0.5, 0, DEVICE, 'add')


def compute_pareto(model, X_test, y_test):
    """Accuracy at each threshold."""
    pareto = {}
    for t in PARETO_THRESHOLDS:
        if t <= model.hidden_dim:
            pareto[t] = accuracy_after_zeroing(model, X_test, y_test, t)
    return pareto


def pareto_auc(pareto):
    widths = sorted(pareto.keys())
    accs = [pareto[w] for w in widths]
    return float(np.trapezoid(accs, widths))


def count_surviving(model, threshold=1e-4):
    return int((model.W1.data.norm(dim=1) > threshold).sum().item())


def count_fourier(model, p=P, ipr_thresh=0.2):
    n = 0
    for j in range(model.hidden_dim):
        spec = neuron_fourier_spectrum(model.W1.data[j], p)
        if compute_ipr(spec) > ipr_thresh:
            n += 1
    return n


def binary_decay_step(model, sever_mask, lr=LR, lam=LAMBDA):
    """Apply decay ONLY to severed neurons' W1 rows, b1 entries, and W2 columns."""
    if sever_mask.sum() == 0:
        return
    decay_factor = 1.0 - lr * lam
    model.W1.data[sever_mask] *= decay_factor
    model.b1.data[sever_mask] *= decay_factor
    model.W2.data[:, sever_mask] *= decay_factor


# ============================================================
# Core training functions
# ============================================================

def run_sever_and_decay(model, sever_count, schedule='instant',
                        ramp_steps=10000):
    """
    Sever-and-decay: detach dying neurons from backward pass, decay them.
    Returns result dict.
    """
    X_train, y_train, X_test, y_test = get_data()

    # Rank neurons by norm at step 0, fix ordering
    initial_norms = model.W1.data.norm(dim=1)
    sever_order = initial_norms.argsort()  # lowest norm first

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    # Mutable cutoff for hook closure
    cutoff_ref = [0 if schedule != 'instant' else sever_count]

    # Build the sever mask from the ordering
    def get_sever_mask():
        mask = torch.zeros(128, dtype=torch.bool, device=DEVICE)
        if cutoff_ref[0] > 0:
            mask[sever_order[:cutoff_ref[0]]] = True
        return mask

    def sever_hook(module, input, output):
        mask = get_sever_mask()
        if mask.any():
            # Detach severed neurons' activations
            detached = output.clone()
            detached[:, mask] = output[:, mask].detach()
            return detached
        return output

    handle = model.activation.register_forward_hook(sever_hook)

    metrics_log = []
    t0 = time.time()

    for step in range(MAX_STEPS):
        # Update cutoff for ramp schedule
        if schedule == 'linear_ramp' and step <= ramp_steps:
            cutoff_ref[0] = int(sever_count * min(step / ramp_steps, 1.0))

        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()

        # Decay only severed neurons
        sever_mask = get_sever_mask()
        binary_decay_step(model, sever_mask)

        if step % LOG_EVERY == 0:
            model.eval()
            _, acc = evaluate(model, X_test, y_test)
            metrics_log.append({
                'step': step, 'test_acc': acc,
                'surviving': count_surviving(model),
                'train_loss': loss.item(),
            })

    handle.remove()
    wall_time = time.time() - t0

    model.eval()
    _, final_acc = evaluate(model, X_test, y_test)
    pareto = compute_pareto(model, X_test, y_test)

    return {
        'final_acc': final_acc,
        'surviving': count_surviving(model),
        'fourier': count_fourier(model),
        'pareto': pareto,
        'pareto_auc': pareto_auc(pareto),
        'wall_time': wall_time,
        'metrics_log': metrics_log,
    }


def run_prune_retrain(model, prune_count):
    """Magnitude prune at step 0, retrain."""
    X_train, y_train, X_test, y_test = get_data()

    # Prune lowest-norm neurons
    norms = model.W1.data.norm(dim=1)
    _, prune_idx = norms.topk(prune_count, largest=False)
    model.W1.data[prune_idx] = 0.0
    model.b1.data[prune_idx] = 0.0
    model.W2.data[:, prune_idx] = 0.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    t0 = time.time()
    for step in range(MAX_STEPS):
        model.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        # Keep pruned neurons at zero
        model.W1.data[prune_idx] = 0.0
        model.b1.data[prune_idx] = 0.0
        model.W2.data[:, prune_idx] = 0.0

    wall_time = time.time() - t0
    model.eval()
    _, final_acc = evaluate(model, X_test, y_test)
    pareto = compute_pareto(model, X_test, y_test)

    return {
        'final_acc': final_acc,
        'surviving': count_surviving(model),
        'fourier': count_fourier(model),
        'pareto': pareto,
        'pareto_auc': pareto_auc(pareto),
        'wall_time': wall_time,
    }


def run_scratch(target_width):
    """Train from scratch at target width."""
    X_train, y_train, X_test, y_test = get_data()
    model = GrokMLP(2 * P, target_width, P).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    # Need some decay for grokking
    from src.decay import AsymmetricDecay
    decay = AsymmetricDecay(LAMBDA, alpha=0.0)

    t0 = time.time()
    for step in range(MAX_STEPS):
        model.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        decay.step(model.W1, model.W2, LR)

    wall_time = time.time() - t0
    model.eval()
    _, final_acc = evaluate(model, X_test, y_test)
    pareto = compute_pareto(model, X_test, y_test)

    return {
        'final_acc': final_acc,
        'surviving': count_surviving(model),
        'fourier': count_fourier(model),
        'pareto': pareto,
        'pareto_auc': pareto_auc(pareto),
        'wall_time': wall_time,
    }


def run_distillation(teacher, target_width, temperature=4.0):
    """Knowledge distillation."""
    X_train, y_train, X_test, y_test = get_data()
    student = GrokMLP(2 * P, target_width, P).to(DEVICE)
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.0)

    teacher.eval()
    t0 = time.time()
    for step in range(MAX_STEPS):
        student.train()
        optimizer.zero_grad()

        with torch.no_grad():
            teacher_logits = teacher(X_train) / temperature
        student_logits = student(X_train) / temperature

        loss_kd = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction='batchmean') * (temperature ** 2)
        loss_hard = F.cross_entropy(student(X_train), y_train)
        loss = (loss_kd + loss_hard) / 2
        loss.backward()
        optimizer.step()

    wall_time = time.time() - t0
    student.eval()
    _, final_acc = evaluate(student, X_test, y_test)
    pareto = compute_pareto(student, X_test, y_test)

    return {
        'final_acc': final_acc,
        'surviving': count_surviving(student),
        'fourier': count_fourier(student),
        'pareto': pareto,
        'pareto_auc': pareto_auc(pareto),
        'wall_time': wall_time,
    }


def run_asymmetric_decay(model):
    """Standard asymmetric decay from Experiment A (no severing)."""
    X_train, y_train, X_test, y_test = get_data()
    from src.decay import AsymmetricDecay
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    decay = AsymmetricDecay(LAMBDA, alpha=5.0)

    t0 = time.time()
    for step in range(MAX_STEPS):
        model.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        decay.step(model.W1, model.W2, LR)

    wall_time = time.time() - t0
    model.eval()
    _, final_acc = evaluate(model, X_test, y_test)
    pareto = compute_pareto(model, X_test, y_test)

    return {
        'final_acc': final_acc,
        'surviving': count_surviving(model),
        'fourier': count_fourier(model),
        'pareto': pareto,
        'pareto_auc': pareto_auc(pareto),
        'wall_time': wall_time,
    }


# ============================================================
# Main experiment
# ============================================================

def run_experiment():
    output_dir = 'outputs/experiment_b'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT B: Stop-Gradient-Assisted Structured Pruning")
    print("=" * 70)

    all_results = {}

    # ---- MAIN SWEEP: sever-and-decay at 6 fractions ----
    print("\n--- Sever-and-Decay (main sweep) ---")
    for n_sever in SEVER_FRACTIONS:
        key = f'sever_{n_sever}'
        results = []
        for seed in SEEDS:
            print(f"  sever={n_sever} seed={seed}...", end='', flush=True)
            set_seed(seed)
            model = load_expert(seed)
            r = run_sever_and_decay(model, n_sever)
            print(f" acc={r['final_acc']:.4f} surv={r['surviving']} "
                  f"auc={r['pareto_auc']:.1f} ({r['wall_time']:.0f}s)", flush=True)
            results.append(r)
        all_results[key] = results

    # ---- SCHEDULE: linear ramp at 50% ----
    print("\n--- Schedule: linear ramp at 50% ---")
    key = 'sever_64_ramp'
    results = []
    for seed in SEEDS:
        print(f"  ramp seed={seed}...", end='', flush=True)
        set_seed(seed)
        model = load_expert(seed)
        r = run_sever_and_decay(model, 64, schedule='linear_ramp')
        print(f" acc={r['final_acc']:.4f} surv={r['surviving']} "
              f"auc={r['pareto_auc']:.1f} ({r['wall_time']:.0f}s)", flush=True)
        results.append(r)
    all_results[key] = results

    # ---- BASELINES at 3 fractions ----
    print("\n--- Baselines ---")
    for n_sever in BASELINE_FRACTIONS:
        target_width = 128 - n_sever

        # Prune + retrain
        key = f'prune_retrain_{n_sever}'
        results = []
        for seed in SEEDS:
            print(f"  prune_retrain sever={n_sever} seed={seed}...", end='', flush=True)
            set_seed(seed)
            model = load_expert(seed)
            r = run_prune_retrain(model, n_sever)
            print(f" acc={r['final_acc']:.4f} ({r['wall_time']:.0f}s)", flush=True)
            results.append(r)
        all_results[key] = results

        # Train from scratch
        key = f'scratch_{target_width}'
        results = []
        for seed in SEEDS:
            print(f"  scratch w={target_width} seed={seed}...", end='', flush=True)
            set_seed(seed)
            r = run_scratch(target_width)
            print(f" acc={r['final_acc']:.4f} ({r['wall_time']:.0f}s)", flush=True)
            results.append(r)
        all_results[key] = results

        # Distillation
        key = f'distill_{target_width}'
        results = []
        for seed in SEEDS:
            print(f"  distill w={target_width} seed={seed}...", end='', flush=True)
            set_seed(seed)
            teacher = load_expert(seed)
            r = run_distillation(teacher, target_width)
            print(f" acc={r['final_acc']:.4f} ({r['wall_time']:.0f}s)", flush=True)
            results.append(r)
        all_results[key] = results

        # Asymmetric decay (only at 64 to avoid redundancy)
        if n_sever == 64:
            key = 'asymmetric_decay'
            results = []
            for seed in SEEDS:
                print(f"  asymmetric seed={seed}...", end='', flush=True)
                set_seed(seed)
                model = load_expert(seed)
                r = run_asymmetric_decay(model)
                print(f" acc={r['final_acc']:.4f} ({r['wall_time']:.0f}s)", flush=True)
                results.append(r)
            all_results[key] = results

    # ---- SUMMARY ----
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Condition':<25} {'Accuracy':>10} {'Surviving':>10} "
          f"{'Fourier':>8} {'Pareto AUC':>12}")
    print("-" * 65)
    for name, results in all_results.items():
        accs = [r['final_acc'] for r in results]
        surv = [r['surviving'] for r in results]
        four = [r['fourier'] for r in results]
        aucs = [r['pareto_auc'] for r in results]
        print(f"{name:<25} {np.mean(accs):>9.4f} {np.mean(surv):>9.1f} "
              f"{np.mean(four):>7.1f} {np.mean(aucs):>11.1f}")

    # ---- PARETO COMPARISON at 50% (sever vs baselines) ----
    print("\n--- Pareto AUC: sever-and-decay vs baselines at 50% ---")
    from scipy import stats
    sever_aucs = np.array([r['pareto_auc'] for r in all_results['sever_64']])
    comparisons = [
        ('prune_retrain_64', 'Prune+retrain'),
        ('scratch_64', 'Scratch W=64'),
        ('distill_64', 'Distill W=64'),
        ('asymmetric_decay', 'Asymmetric decay'),
        ('sever_64_ramp', 'Sever ramp'),
    ]
    for key, label in comparisons:
        if key in all_results:
            other = np.array([r['pareto_auc'] for r in all_results[key]])
            t_stat, p_val = stats.ttest_rel(sever_aucs, other)
            diff = np.mean(sever_aucs - other)
            print(f"  vs {label:<20} diff={diff:>+8.1f}  t={t_stat:>6.2f}  p={p_val:.4f}")

    # ---- SAVE ----
    def sanitize(obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    save_data = {}
    for name, results in all_results.items():
        save_data[name] = [{k: sanitize(v) for k, v in r.items()
                            if k != 'metrics_log'} for r in results]

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {output_dir}/results.json")


if __name__ == '__main__':
    run_experiment()
