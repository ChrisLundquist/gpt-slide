"""
Experiment A: Single-Task Densification

Can asymmetric weight decay densify a grokked addition model?
Uses single output head for W2 interaction between neurons.
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

from src.config import Config
from src.model import GrokMLP
from src.decay import AsymmetricDecay
from src.data import build_dataset
from src.train import set_seed, evaluate
from src.metrics import (
    compute_column_norms, compute_left_energy, frequency_energy_matrix,
    soft_migration_score, accuracy_after_zeroing, neuron_fourier_spectrum,
    compute_ipr,
)

EXPERT_SEEDS = [42, 137, 256, 512, 1024]
PARETO_THRESHOLDS = [32, 48, 64, 80, 96]
MAX_STEPS = 20_000
LOG_EVERY = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_grokked_expert(seed):
    """Load a grokked single-task addition expert."""
    path = f'outputs/step1/experts/add_seed{seed}/checkpoint.pt'
    ckpt = torch.load(path, weights_only=True)
    config = Config()
    model = GrokMLP(config.input_dim, 128, config.output_dim).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    return model


def run_densification(model, condition, alpha=0.0, reverse=False,
                      gmp_target=None, instant_prune=None,
                      sever_gradient=False, max_steps=MAX_STEPS):
    """
    Run one densification experiment from a grokked checkpoint.
    Returns metrics dict.
    """
    config = Config(device=DEVICE)
    X_train, y_train, X_test, y_test = build_dataset(
        config.p, config.train_frac, config.data_seed, DEVICE, 'add')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    decay = AsymmetricDecay(0.1, alpha, reverse)

    # Initial energy for migration tracking
    initial_energy = frequency_energy_matrix(model.W1.data, config.p)

    # Instant prune at step 0
    if instant_prune is not None:
        norms = model.W1.data.norm(dim=1)
        _, prune_idx = norms.topk(instant_prune, largest=False)
        model.W1.data[prune_idx] = 0.0
        model.b1.data[prune_idx] = 0.0
        model.W2.data[:, prune_idx] = 0.0

    # GMP state
    gmp_ramp_steps = 15000
    gmp_interval = 100

    # Gradient-severed hook for Condition 5
    hook_handle = None
    if sever_gradient:
        cutoff = 96  # sever top 25%

        def sever_hook(module, input, output):
            # Detach the dying neurons' activations so no gradient flows through them
            left = output[:, :cutoff]
            right = output[:, cutoff:].detach()  # no gradient through dying neurons
            return torch.cat([left, right], dim=1)

        hook_handle = model.activation.register_forward_hook(sever_hook)

    # Compile if no hooks
    raw_model = model
    if not sever_gradient and hasattr(torch, 'compile'):
        model = torch.compile(model)

    metrics_log = []
    t0 = time.time()

    for step in range(max_steps):
        model.train()
        optimizer.zero_grad()

        logits = model(X_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()

        # External decay
        decay.step(raw_model.W1, raw_model.W2, 1e-3)

        # GMP pruning
        if gmp_target is not None and step % gmp_interval == 0 and step <= gmp_ramp_steps:
            n_prune = 128 - gmp_target
            progress = min(step / gmp_ramp_steps, 1.0)
            current_prune = int(n_prune * (progress ** 3))
            if current_prune > 0:
                norms = raw_model.W1.data.norm(dim=1)
                _, idx = norms.topk(current_prune, largest=False)
                raw_model.W1.data[idx] = 0.0
                raw_model.b1.data[idx] = 0.0
                raw_model.W2.data[:, idx] = 0.0

        # Logging
        if step % LOG_EVERY == 0:
            raw_model.eval()
            _, test_acc = evaluate(raw_model, X_test, y_test)
            norms = compute_column_norms(raw_model.W1.data).cpu().tolist()
            left_e = compute_left_energy(raw_model.W1.data, config.p)
            energy = frequency_energy_matrix(raw_model.W1.data, config.p)
            sms, _ = soft_migration_score(initial_energy, energy, config.p)

            m = {
                'step': step,
                'test_acc': test_acc,
                'train_loss': loss.item(),
                'column_norms': norms,
                'left_energy': left_e,
                'soft_migration_score': sms,
            }
            metrics_log.append(m)

    wall_time = time.time() - t0

    # Cleanup
    if hook_handle:
        hook_handle.remove()

    # Final evaluation
    raw_model.eval()
    final_energy = frequency_energy_matrix(raw_model.W1.data, config.p)
    _, final_acc = evaluate(raw_model, X_test, y_test)

    # Pareto frontier
    pareto = {}
    for t in PARETO_THRESHOLDS:
        pareto[t] = accuracy_after_zeroing(raw_model, X_test, y_test, t)

    # Surviving neurons
    surviving = (raw_model.W1.data.norm(dim=1) > 1e-4).sum().item()

    # Frequency coverage
    n_fourier = 0
    for j in range(128):
        spec = neuron_fourier_spectrum(raw_model.W1.data[j], config.p)
        if compute_ipr(spec) > 0.2:
            n_fourier += 1

    result = {
        'condition': condition,
        'final_acc': final_acc,
        'surviving_neurons': surviving,
        'fourier_neurons': n_fourier,
        'pareto': pareto,
        'wall_time': wall_time,
        'metrics_log': metrics_log,
    }

    # Pareto AUC (trapezoidal)
    widths = sorted(pareto.keys())
    accs = [pareto[w] for w in widths]
    result['pareto_auc'] = float(np.trapz(accs, widths))

    return result


def run_experiment():
    output_dir = 'outputs/experiment_a'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT A: Single-Task Densification")
    print("=" * 60)

    all_results = {}

    # Phase 1: Run conditions 1, 2, 5, 6 (independent of target width)
    for cond_name, cond_kwargs in [
        ('asymmetric',    {'alpha': 5.0}),
        ('uniform',       {'alpha': 0.0}),
        ('reversed',      {'alpha': 5.0, 'reverse': True}),
        ('gradient_severed', {'alpha': 5.0, 'sever_gradient': True}),
    ]:
        cond_results = []
        for seed in EXPERT_SEEDS:
            print(f"  {cond_name} seed={seed}...", end='', flush=True)
            model = load_grokked_expert(seed)
            r = run_densification(model, cond_name, **cond_kwargs)
            print(f" acc={r['final_acc']:.4f} surviving={r['surviving_neurons']} "
                  f"pareto_auc={r['pareto_auc']:.1f} ({r['wall_time']:.1f}s)", flush=True)
            cond_results.append(r)
        all_results[cond_name] = cond_results

    # Determine target width from asymmetric condition
    asym_surviving = [r['surviving_neurons'] for r in all_results['asymmetric']]
    target_width = int(np.median(asym_surviving))
    print(f"\nTarget width from asymmetric: {target_width} "
          f"(range {min(asym_surviving)}-{max(asym_surviving)})")

    # Phase 2: GMP and instant prune (need target width)
    for cond_name, cond_kwargs in [
        ('gmp',           {'alpha': 0.0, 'gmp_target': target_width}),
        ('instant_prune', {'alpha': 0.0, 'instant_prune': 128 - target_width}),
    ]:
        cond_results = []
        for seed in EXPERT_SEEDS:
            print(f"  {cond_name} seed={seed}...", end='', flush=True)
            model = load_grokked_expert(seed)
            r = run_densification(model, cond_name, **cond_kwargs)
            print(f" acc={r['final_acc']:.4f} surviving={r['surviving_neurons']} "
                  f"pareto_auc={r['pareto_auc']:.1f} ({r['wall_time']:.1f}s)", flush=True)
            cond_results.append(r)
        all_results[cond_name] = cond_results

    # Phase 3: Train from scratch
    cond_results = []
    for seed in EXPERT_SEEDS:
        print(f"  scratch_w{target_width} seed={seed}...", end='', flush=True)
        set_seed(seed)
        config = Config(device=DEVICE)
        model = GrokMLP(config.input_dim, target_width, config.output_dim).to(DEVICE)
        # Match steps (same 20k) — smaller model gets same training budget
        r = run_densification(model, 'scratch', alpha=0.0, max_steps=MAX_STEPS)
        print(f" acc={r['final_acc']:.4f} ({r['wall_time']:.1f}s)", flush=True)
        cond_results.append(r)
    all_results['scratch'] = cond_results

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Condition':<20} {'Accuracy':>10} {'Surviving':>10} {'Pareto AUC':>12} {'Time':>8}")
    print("-" * 60)
    for name, results in all_results.items():
        accs = [r['final_acc'] for r in results]
        surv = [r['surviving_neurons'] for r in results]
        aucs = [r['pareto_auc'] for r in results]
        times = [r['wall_time'] for r in results]
        print(f"{name:<20} {np.mean(accs):>9.4f} {np.mean(surv):>9.1f} "
              f"{np.mean(aucs):>11.1f} {np.mean(times):>7.1f}s")

    # Pairwise comparisons (Pareto AUC)
    print("\n--- Pairwise Pareto AUC comparisons vs Asymmetric ---")
    from scipy import stats
    asym_aucs = np.array([r['pareto_auc'] for r in all_results['asymmetric']])
    for name, results in all_results.items():
        if name == 'asymmetric':
            continue
        other_aucs = np.array([r['pareto_auc'] for r in results])
        if len(other_aucs) == len(asym_aucs):
            t_stat, p_val = stats.ttest_rel(asym_aucs, other_aucs)
            diff = np.mean(asym_aucs - other_aucs)
            print(f"  vs {name:<20} diff={diff:>+8.1f}  t={t_stat:>6.2f}  p={p_val:.4f}")

    # Save
    # Convert results to JSON-safe format
    def sanitize(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    save_results = {}
    for name, results in all_results.items():
        save_results[name] = []
        for r in results:
            sr = {k: sanitize(v) for k, v in r.items() if k != 'metrics_log'}
            save_results[name].append(sr)

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to {output_dir}/results.json")
    return all_results


if __name__ == '__main__':
    run_experiment()
