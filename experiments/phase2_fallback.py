"""
Phase 2 Fallback — Is Position-Based Pruning Useful?

Migration is ruled out. Compare asymmetric decay against:
1. Distillation (W=128 student from MLP_cat teacher)
2. Structured GMP (gradual magnitude pruning by column norm)

5 seeds each. Compare accuracy at matched parameter count.
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
from src.data import build_joint_dataset
from src.train import set_seed, evaluate, train_run, save_result
from src.decay import AsymmetricDecay


FALLBACK_SEEDS = SEEDS[:5]


def run_distillation(teacher, data, seed, temperature, device, max_steps=20000):
    """Train a W=128 student with soft targets from teacher."""
    set_seed(seed)
    config = Config(device=device)
    student = GrokMLP(config.input_dim, 128, config.output_dim, joint=True).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0.0)

    add_X, add_y = data['add'][:2]
    mul_X, mul_y = data['mul'][:2]

    t0 = time.time()
    for step in range(max_steps):
        optimizer.zero_grad()

        # Soft targets from teacher
        with torch.no_grad():
            teacher_logits_add = teacher(add_X, task='add') / temperature
            teacher_logits_mul = teacher(mul_X, task='mul') / temperature

        student_logits_add = student(add_X, task='add') / temperature
        student_logits_mul = student(mul_X, task='mul') / temperature

        # KL divergence loss (soft targets)
        loss_add = F.kl_div(
            F.log_softmax(student_logits_add, dim=-1),
            F.softmax(teacher_logits_add, dim=-1),
            reduction='batchmean') * (temperature ** 2)
        loss_mul = F.kl_div(
            F.log_softmax(student_logits_mul, dim=-1),
            F.softmax(teacher_logits_mul, dim=-1),
            reduction='batchmean') * (temperature ** 2)

        # Also add hard target loss
        hard_add = F.cross_entropy(student(add_X, task='add'), add_y)
        hard_mul = F.cross_entropy(student(mul_X, task='mul'), mul_y)

        loss = (loss_add + loss_mul + hard_add + hard_mul) / 4
        loss.backward()
        optimizer.step()

    wall_time = time.time() - t0

    _, acc_add = evaluate(student, data['add'][2], data['add'][3], task='add')
    _, acc_mul = evaluate(student, data['mul'][2], data['mul'][3], task='mul')

    return {'acc_add': acc_add, 'acc_mul': acc_mul, 'wall_time': wall_time,
            'temperature': temperature, 'hidden_dim': 128}


def run_structured_gmp(converged_state, data, seed, device,
                       target_neurons=128, max_steps=20000):
    """
    Structured gradual magnitude pruning.
    Prune entire neuron columns ranked by L2 norm.
    Cubic schedule: ramp over first 15k of 20k steps.
    """
    set_seed(seed)
    config = Config(device=device)
    model = GrokMLP(config.input_dim, 256, config.output_dim, joint=True).to(device)
    model.load_state_dict(converged_state)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    decay = AsymmetricDecay(0.1, alpha=0.0)  # uniform decay

    add_X, add_y = data['add'][:2]
    mul_X, mul_y = data['mul'][:2]
    batch_size = min(512, len(add_X))

    n_total = 256
    n_prune = n_total - target_neurons
    ramp_steps = 15000
    prune_interval = 100

    t0 = time.time()
    for step in range(max_steps):
        optimizer.zero_grad()

        idx = torch.randperm(len(add_X))[:batch_size]
        loss_add = F.cross_entropy(model(add_X[idx], task='add'), add_y[idx])
        loss_mul = F.cross_entropy(model(mul_X[idx], task='mul'), mul_y[idx])
        loss = (loss_add + loss_mul) / 2
        loss.backward()
        optimizer.step()
        decay.step(model.W1, [model.head_add.weight, model.head_mul.weight], 1e-3)

        # Pruning step
        if step % prune_interval == 0 and step <= ramp_steps:
            # Cubic schedule: sparsity ramps from 0 to n_prune
            progress = min(step / ramp_steps, 1.0)
            current_prune = int(n_prune * (progress ** 3))

            if current_prune > 0:
                norms = model.W1.data.norm(dim=1)
                _, prune_idx = norms.topk(current_prune, largest=False)
                model.W1.data[prune_idx] = 0.0
                model.b1.data[prune_idx] = 0.0
                model.head_add.weight.data[:, prune_idx] = 0.0
                model.head_mul.weight.data[:, prune_idx] = 0.0

    wall_time = time.time() - t0

    # Count surviving neurons
    surviving = (model.W1.data.norm(dim=1) > 1e-6).sum().item()

    _, acc_add = evaluate(model, data['add'][2], data['add'][3], task='add')
    _, acc_mul = evaluate(model, data['mul'][2], data['mul'][3], task='mul')

    return {'acc_add': acc_add, 'acc_mul': acc_mul, 'wall_time': wall_time,
            'surviving_neurons': surviving, 'hidden_dim': 256}


def get_asymmetric_result(step2_dir):
    """Load best asymmetric decay result from Step 2."""
    results = []
    for seed in FALLBACK_SEEDS:
        path = os.path.join(step2_dir, f'asymmetric_seed{seed}', 'metrics.json')
        if os.path.exists(path):
            with open(path) as f:
                results.append(json.load(f))
    return results


def run_fallback():
    output_dir = 'outputs'
    step1_dir = os.path.join(output_dir, 'step1')
    step2_dir = os.path.join(output_dir, 'step2')
    fallback_dir = os.path.join(output_dir, 'fallback')
    os.makedirs(fallback_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = Config(device=device)

    # Load teacher (MLP_cat) and converged state
    cat_ckpt = torch.load(os.path.join(step1_dir, 'mlp_cat.pt'), weights_only=True)
    teacher = GrokMLP(config.input_dim, 256, config.output_dim, joint=True).to(device)
    teacher.load_state_dict(cat_ckpt['model_state_dict'])
    teacher.eval()

    conv_ckpt = torch.load(os.path.join(step1_dir, 'mlp_converged.pt'), weights_only=True)
    converged_state = conv_ckpt['model_state_dict']

    data = build_joint_dataset(config.p, config.train_frac, config.data_seed, device)

    print("=" * 60)
    print("PHASE 2 FALLBACK: Is position-based pruning useful?")
    print("=" * 60)

    all_results = {}

    # 1. Asymmetric decay (from Step 2)
    print("\n--- Asymmetric Decay (from Step 2) ---")
    asym_results = get_asymmetric_result(step2_dir)
    if asym_results:
        for r in asym_results:
            print(f"  add={r.get('final_acc_add', 'N/A'):.4f} "
                  f"mul={r.get('final_acc_mul', 'N/A'):.4f}")
        all_results['asymmetric'] = asym_results

    # 2. Distillation
    print("\n--- Distillation ---")
    for temp in [1, 2, 4]:
        distill_results = []
        for seed in FALLBACK_SEEDS:
            print(f"  T={temp} seed={seed}...", end='', flush=True)
            r = run_distillation(teacher, data, seed, temp, device)
            print(f" add={r['acc_add']:.4f} mul={r['acc_mul']:.4f} "
                  f"({r['wall_time']:.1f}s)", flush=True)
            distill_results.append(r)
        all_results[f'distill_T{temp}'] = distill_results

    # 3. Structured GMP
    print("\n--- Structured GMP ---")
    gmp_results = []
    for seed in FALLBACK_SEEDS:
        print(f"  seed={seed}...", end='', flush=True)
        r = run_structured_gmp(converged_state, data, seed, device)
        print(f" add={r['acc_add']:.4f} mul={r['acc_mul']:.4f} "
              f"surviving={r['surviving_neurons']} ({r['wall_time']:.1f}s)", flush=True)
        gmp_results.append(r)
    all_results['gmp'] = gmp_results

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Method':<25} {'Add Acc':>10} {'Mul Acc':>10} {'Time (s)':>10}")
    print("-" * 55)

    for name, results in all_results.items():
        add_accs = [r.get('acc_add', r.get('final_acc_add', 0)) for r in results]
        mul_accs = [r.get('acc_mul', r.get('final_acc_mul', 0)) for r in results]
        times = [r.get('wall_time', 0) for r in results]
        print(f"{name:<25} {np.mean(add_accs):>9.4f} {np.mean(mul_accs):>9.4f} "
              f"{np.mean(times):>9.1f}")

    # Save
    with open(os.path.join(fallback_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {fallback_dir}/results.json")


if __name__ == '__main__':
    run_fallback()
