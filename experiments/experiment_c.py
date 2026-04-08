"""
Experiment C: Addressing the Critical Review

Tests whether the negative migration result is robust to:
1. Optimizer choice (AdamW vs SGD — adaptive rates as life support?)
2. Decay strength (0.0001, 0.001, 0.01 per step — actually kill neurons)
3. Batch size (full-batch vs minibatch — stochastic noise helps death?)

Uses DECOUPLED decay: W *= (1 - rate * mask), independent of learning rate.
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats

torch.set_float32_matmul_precision('high')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config, SEEDS
from src.model import GrokMLP
from src.data import build_dataset
from src.train import set_seed, evaluate
from src.metrics import (
    compute_column_norms, neuron_fourier_spectrum, compute_ipr,
    frequency_energy_matrix, soft_migration_score, accuracy_after_zeroing,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
P = 113
PARETO_THRESHOLDS = [16, 32, 48, 64, 80, 96, 112]


def decoupled_asymmetric_decay(W1, W2, alpha, decay_rate):
    """
    Apply position-dependent decay INDEPENDENT of learning rate.
    W1[j] *= (1 - decay_rate * (1 + alpha * j / (W-1)))
    W2[:, j] *= same factor
    Biases are NOT decayed.
    """
    W = W1.shape[0]
    positions = torch.linspace(0, 1, W, device=W1.device)
    mask = decay_rate * (1.0 + alpha * positions)
    # Clamp to prevent negative multipliers at high decay rates
    factors = (1.0 - mask).clamp(min=0.0)
    W1.data.mul_(factors.unsqueeze(1))
    W2.data.mul_(factors.unsqueeze(0))


def load_expert(seed, optimizer_type='adamw'):
    """Load a grokked expert. AdamW experts from Step 1, SGD trained fresh."""
    if optimizer_type == 'adamw':
        path = f'outputs/step1/experts/add_seed{seed}/checkpoint.pt'
        ckpt = torch.load(path, weights_only=True)
        model = GrokMLP(2 * P, 128, P).to(DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        return model
    else:
        path = f'outputs/experiment_c/sgd_experts/sgd_seed{seed}.pt'
        ckpt = torch.load(path, weights_only=True)
        model = GrokMLP(2 * P, 128, P).to(DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        return model


def count_dead(model, initial_norms, threshold_frac=0.01):
    """Count neurons with norm < 1% of their initial norm."""
    current_norms = model.W1.data.norm(dim=1)
    thresholds = initial_norms * threshold_frac
    return int((current_norms < thresholds).sum().item())


def pareto_auc(model, X_test, y_test):
    pareto = {}
    for t in PARETO_THRESHOLDS:
        pareto[t] = accuracy_after_zeroing(model, X_test, y_test, t)
    widths = sorted(pareto.keys())
    accs = [pareto[w] for w in widths]
    return float(np.trapezoid(accs, widths)), pareto


# ============================================================
# Step 0: SGD grokking sweep
# ============================================================

def sgd_grokking_sweep():
    """Find if/how SGD can grok on modular addition."""
    print("\n--- Step 0: SGD Grokking Sweep ---")
    X_train, y_train, X_test, y_test = build_dataset(P, 0.5, 0, DEVICE, 'add')

    best = None
    for lr in [0.01, 0.03, 0.1, 0.3]:
        for mom in [0.0, 0.9]:
            for dr in [0.0001, 0.001]:
                set_seed(42)
                model = GrokMLP(2 * P, 128, P).to(DEVICE)
                opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom)

                grok_step = None
                run = 0
                for step in range(50000):
                    opt.zero_grad()
                    loss = F.cross_entropy(model(X_train), y_train)
                    loss.backward()
                    opt.step()
                    decoupled_asymmetric_decay(model.W1, model.W2, 0.0, dr)

                    if step % 100 == 0:
                        _, acc = evaluate(model, X_test, y_test)
                        if acc >= 1.0:
                            run += 1
                            if run >= 10:
                                grok_step = step - 900
                                break
                        else:
                            run = 0

                tag = f'grok@{grok_step}' if grok_step else 'no_grok'
                print(f'  SGD lr={lr} mom={mom} dr={dr}: {tag}', flush=True)
                if grok_step and (best is None or grok_step < best[0]):
                    best = (grok_step, lr, mom, dr)

    if best:
        print(f'  Best: grok@{best[0]}, lr={best[1]}, mom={best[2]}, dr={best[3]}')
    else:
        print('  WARNING: SGD did not grok at any configuration!')
    return best


def adamw_decoupled_verify():
    """Verify decoupled decay formula works for AdamW grokking."""
    print("\n--- Step 0: AdamW Decoupled Decay Verify ---")
    X_train, y_train, X_test, y_test = build_dataset(P, 0.5, 0, DEVICE, 'add')

    for dr in [0.0001, 0.001, 0.01]:
        results = []
        for seed in [42, 137, 256]:
            set_seed(seed)
            model = GrokMLP(2 * P, 128, P).to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

            grok_step = None
            run = 0
            for step in range(50000):
                opt.zero_grad()
                loss = F.cross_entropy(model(X_train), y_train)
                loss.backward()
                opt.step()
                decoupled_asymmetric_decay(model.W1, model.W2, 0.0, dr)

                if step % 100 == 0:
                    _, acc = evaluate(model, X_test, y_test)
                    if acc >= 1.0:
                        run += 1
                        if run >= 10:
                            grok_step = step - 900
                            break
                    else:
                        run = 0
            results.append(grok_step)

        tag = ', '.join(f'grok@{g}' if g else 'no_grok' for g in results)
        print(f'  AdamW dr={dr}: [{tag}]', flush=True)


def train_sgd_experts(sgd_lr, sgd_mom, sgd_dr):
    """Train 10 grokked SGD experts."""
    print("\n--- Training SGD Experts ---")
    expert_dir = 'outputs/experiment_c/sgd_experts'
    os.makedirs(expert_dir, exist_ok=True)

    X_train, y_train, X_test, y_test = build_dataset(P, 0.5, 0, DEVICE, 'add')

    for seed in SEEDS:
        set_seed(seed)
        model = GrokMLP(2 * P, 128, P).to(DEVICE)
        opt = torch.optim.SGD(model.parameters(), lr=sgd_lr, momentum=sgd_mom)

        grok_step = None
        run = 0
        for step in range(50000):
            opt.zero_grad()
            loss = F.cross_entropy(model(X_train), y_train)
            loss.backward()
            opt.step()
            decoupled_asymmetric_decay(model.W1, model.W2, 0.0, sgd_dr)

            if step % 100 == 0:
                _, acc = evaluate(model, X_test, y_test)
                if acc >= 1.0:
                    run += 1
                    if run >= 10:
                        grok_step = step - 900
                        break
                else:
                    run = 0

        tag = f'grok@{grok_step}' if grok_step else 'no_grok'
        print(f'  seed={seed}: {tag}', flush=True)

        if grok_step:
            torch.save({'model_state_dict': model.state_dict()},
                       os.path.join(expert_dir, f'sgd_seed{seed}.pt'))


# ============================================================
# Phase 1: Optimizer × Decay Rate factorial
# ============================================================

def run_decay_condition(model, optimizer_fn, decay_rate, alpha, max_steps,
                        batch_size=None):
    """Run one decay condition. Returns result dict."""
    X_train, y_train, X_test, y_test = build_dataset(P, 0.5, 0, DEVICE, 'add')
    optimizer = optimizer_fn(model.parameters())

    initial_norms = model.W1.data.norm(dim=1).clone()
    initial_energy = frequency_energy_matrix(model.W1.data, P)

    metrics_log = []
    t0 = time.time()
    prev_W1 = model.W1.data.clone()

    for step in range(max_steps):
        model.train()
        optimizer.zero_grad()

        if batch_size and batch_size < len(X_train):
            idx = torch.randperm(len(X_train), device=DEVICE)[:batch_size]
            loss = F.cross_entropy(model(X_train[idx]), y_train[idx])
        else:
            loss = F.cross_entropy(model(X_train), y_train)

        loss.backward()
        optimizer.step()
        decoupled_asymmetric_decay(model.W1, model.W2, alpha, decay_rate)

        if step % 100 == 0:
            model.eval()
            _, acc = evaluate(model, X_test, y_test)
            dead = count_dead(model, initial_norms)

            # Per-neuron update magnitude (effective restoring force)
            update_mag = (model.W1.data - prev_W1).norm(dim=1).mean().item()
            prev_W1 = model.W1.data.clone()

            metrics_log.append({
                'step': step, 'test_acc': acc, 'dead': dead,
                'train_loss': loss.item(), 'update_magnitude': update_mag,
            })

    wall_time = time.time() - t0
    model.eval()
    _, final_acc = evaluate(model, X_test, y_test)
    dead = count_dead(model, initial_norms)
    auc, pareto = pareto_auc(model, X_test, y_test)

    # Migration score
    final_energy = frequency_energy_matrix(model.W1.data, P)
    sms, _ = soft_migration_score(initial_energy, final_energy, P)

    # Fourier coverage
    n_fourier = sum(1 for j in range(model.hidden_dim)
                    if compute_ipr(neuron_fourier_spectrum(model.W1.data[j], P)) > 0.2)

    return {
        'final_acc': final_acc,
        'dead_neurons': dead,
        'pareto_auc': auc,
        'pareto': pareto,
        'migration_score': sms,
        'fourier_neurons': n_fourier,
        'wall_time': wall_time,
        'metrics_log': metrics_log,
    }


def run_phase1():
    """Phase 1: 2×3 factorial on optimizer × decay_rate."""
    print("\n" + "=" * 70)
    print("PHASE 1: Can we kill neurons?")
    print("=" * 70)

    cells = {}
    optimizer_configs = {
        'adamw': lambda params: torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0),
        'sgd': lambda params: torch.optim.SGD(params, lr=sgd_best_lr,
                                               momentum=sgd_best_mom),
    }
    decay_rates = [0.0001, 0.001, 0.01]

    for opt_name, opt_fn in optimizer_configs.items():
        for dr in decay_rates:
            cell_name = f'{opt_name}_dr{dr}'
            cell_results = []

            for seed in SEEDS:
                print(f'  {cell_name} seed={seed}...', end='', flush=True)
                set_seed(seed)
                try:
                    model = load_expert(seed, opt_name)
                except FileNotFoundError:
                    print(f' SKIP (no expert)', flush=True)
                    continue

                r = run_decay_condition(model, opt_fn, dr, alpha=5.0,
                                        max_steps=20000)
                print(f' acc={r["final_acc"]:.4f} dead={r["dead_neurons"]} '
                      f'auc={r["pareto_auc"]:.1f} ({r["wall_time"]:.0f}s)',
                      flush=True)
                cell_results.append(r)

            cells[cell_name] = cell_results

    return cells


def evaluate_death_gate(cells):
    """Test which cells have significant neuron death."""
    print("\n--- Death Gate ---")
    passing_cells = []
    for name, results in cells.items():
        deaths = [r['dead_neurons'] for r in results]
        if len(deaths) < 3 or np.mean(deaths) == 0:
            t_stat, p_val = 0, 1.0
        else:
            t_stat, p_val = stats.ttest_1samp(deaths, 0, alternative='greater')
        mean_d = np.mean(deaths)
        passed = p_val < 0.05 and mean_d > 0
        status = 'PASS' if passed else 'FAIL'
        print(f'  {name:<25} mean_dead={mean_d:>5.1f}  t={t_stat:>5.2f}  '
              f'p={p_val:.4f}  {status}', flush=True)
        if passed:
            passing_cells.append(name)
    return passing_cells


# ============================================================
# Phase 2: Minibatch test
# ============================================================

def run_phase2(passing_cells, phase1_cells):
    """Test whether minibatch noise helps at death-producing cells."""
    print("\n" + "=" * 70)
    print("PHASE 2: Does minibatch noise help?")
    print("=" * 70)

    if not passing_cells:
        print("No cells passed death gate. Skipping Phase 2.")
        return {}

    optimizer_configs = {
        'adamw': lambda params: torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0),
        'sgd': lambda params: torch.optim.SGD(params, lr=sgd_best_lr,
                                               momentum=sgd_best_mom),
    }

    minibatch_cells = {}
    for cell_name in passing_cells:
        opt_name = cell_name.split('_dr')[0]
        dr = float(cell_name.split('_dr')[1])
        opt_fn = optimizer_configs.get(opt_name)
        if not opt_fn:
            continue

        mb_name = f'{cell_name}_mb512'
        results = []
        for seed in SEEDS:
            print(f'  {mb_name} seed={seed}...', end='', flush=True)
            set_seed(seed)
            try:
                model = load_expert(seed, opt_name)
            except FileNotFoundError:
                print(f' SKIP', flush=True)
                continue

            r = run_decay_condition(model, opt_fn, dr, alpha=5.0,
                                    max_steps=20000, batch_size=512)
            print(f' acc={r["final_acc"]:.4f} dead={r["dead_neurons"]} '
                  f'({r["wall_time"]:.0f}s)', flush=True)
            results.append(r)

        minibatch_cells[mb_name] = results

        # Paired comparison
        fb_deaths = [r['dead_neurons'] for r in phase1_cells[cell_name]]
        mb_deaths = [r['dead_neurons'] for r in results]
        n = min(len(fb_deaths), len(mb_deaths))
        if n >= 3:
            t_stat, p_val = stats.ttest_rel(mb_deaths[:n], fb_deaths[:n])
            print(f'  Minibatch vs full-batch: diff={np.mean(mb_deaths[:n])-np.mean(fb_deaths[:n]):+.1f} '
                  f'p={p_val:.4f}', flush=True)

    return minibatch_cells


# ============================================================
# Phase 3: Migration test + baselines
# ============================================================

def run_phase3(passing_cells, all_cells):
    """Compare asymmetric decay vs prune+retrain at death-producing cells."""
    print("\n" + "=" * 70)
    print("PHASE 3: Migration test at death-producing cells")
    print("=" * 70)

    if not passing_cells:
        print("No cells passed death gate. Skipping Phase 3.")
        return {}

    X_train, y_train, X_test, y_test = build_dataset(P, 0.5, 0, DEVICE, 'add')

    # Find median death count across all passing cells to set prune target
    all_deaths = []
    for name in passing_cells:
        for r in all_cells[name]:
            all_deaths.append(r['dead_neurons'])
    target_prune = int(np.median(all_deaths))
    target_width = 128 - target_prune
    print(f'Median deaths across passing cells: {target_prune}')
    print(f'Target width for baselines: {target_width}')

    baseline_results = {}

    # Prune + retrain
    print("\n--- Prune + Retrain ---")
    pr_results = []
    for seed in SEEDS:
        print(f'  seed={seed}...', end='', flush=True)
        set_seed(seed)
        model = load_expert(seed, 'adamw')

        norms = model.W1.data.norm(dim=1)
        _, prune_idx = norms.topk(target_prune, largest=False)
        model.W1.data[prune_idx] = 0.0
        model.b1.data[prune_idx] = 0.0
        model.W2.data[:, prune_idx] = 0.0

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        for step in range(20000):
            opt.zero_grad()
            loss = F.cross_entropy(model(X_train), y_train)
            loss.backward()
            opt.step()
            model.W1.data[prune_idx] = 0.0
            model.b1.data[prune_idx] = 0.0
            model.W2.data[:, prune_idx] = 0.0

        _, acc = evaluate(model, X_test, y_test)
        auc, pareto = pareto_auc(model, X_test, y_test)
        print(f' acc={acc:.4f} auc={auc:.1f}', flush=True)
        pr_results.append({'final_acc': acc, 'pareto_auc': auc, 'pareto': pareto})
    baseline_results['prune_retrain'] = pr_results

    # Distillation
    print("\n--- Distillation ---")
    dist_results = []
    for seed in SEEDS:
        print(f'  seed={seed}...', end='', flush=True)
        set_seed(seed)
        teacher = load_expert(seed, 'adamw')
        teacher.eval()
        student = GrokMLP(2 * P, target_width, P).to(DEVICE)
        opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0.0)
        T = 4.0

        for step in range(20000):
            student.train()
            opt.zero_grad()
            with torch.no_grad():
                t_logits = teacher(X_train) / T
            s_logits = student(X_train) / T
            loss_kd = F.kl_div(F.log_softmax(s_logits, -1),
                               F.softmax(t_logits, -1),
                               reduction='batchmean') * T * T
            loss_hard = F.cross_entropy(student(X_train), y_train)
            ((loss_kd + loss_hard) / 2).backward()
            opt.step()

        student.eval()
        _, acc = evaluate(student, X_test, y_test)
        auc, pareto = pareto_auc(student, X_test, y_test)
        print(f' acc={acc:.4f} auc={auc:.1f}', flush=True)
        dist_results.append({'final_acc': acc, 'pareto_auc': auc, 'pareto': pareto})
    baseline_results['distillation'] = dist_results

    # Compare vs best passing cell
    best_cell_name = max(passing_cells,
                         key=lambda n: np.mean([r['dead_neurons'] for r in all_cells[n]]))
    best_cell = all_cells[best_cell_name]

    print(f'\n--- Comparison vs {best_cell_name} ---')
    asym_aucs = np.array([r['pareto_auc'] for r in best_cell])
    for bname, bresults in baseline_results.items():
        b_aucs = np.array([r['pareto_auc'] for r in bresults])
        n = min(len(asym_aucs), len(b_aucs))
        t_stat, p_val = stats.ttest_rel(asym_aucs[:n], b_aucs[:n])
        diff = np.mean(asym_aucs[:n] - b_aucs[:n])
        print(f'  vs {bname:<20} diff={diff:>+8.1f}  t={t_stat:>6.2f}  p={p_val:.4f}')

    # Migration scores
    print(f'\n--- Migration Scores ({best_cell_name}) ---')
    sms_values = [r['migration_score'] for r in best_cell]
    print(f'  Mean SMS: {np.mean(sms_values):.2f} +/- {np.std(sms_values):.2f}')
    t_stat, p_val = stats.ttest_1samp(sms_values, 0, alternative='greater')
    print(f'  SMS > 0 test: t={t_stat:.2f}, p={p_val:.4f}')

    return baseline_results


# ============================================================
# Main
# ============================================================

# Global SGD params (set by sweep)
sgd_best_lr = 0.1
sgd_best_mom = 0.9

def run_experiment():
    global sgd_best_lr, sgd_best_mom

    output_dir = 'outputs/experiment_c'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT C: Addressing the Critical Review")
    print("=" * 70)

    # Step 0: SGD sweep
    sgd_result = sgd_grokking_sweep()
    use_sgd = sgd_result is not None
    if use_sgd:
        sgd_best_lr = sgd_result[1]
        sgd_best_mom = sgd_result[2]
        sgd_best_dr = sgd_result[3]
        train_sgd_experts(sgd_best_lr, sgd_best_mom, sgd_best_dr)
    else:
        print("SGD cannot grok. Running AdamW-only factorial.")

    # Step 0: AdamW verify
    adamw_decoupled_verify()

    # Phase 1
    phase1_cells = run_phase1()
    passing_cells = evaluate_death_gate(phase1_cells)

    # Phase 2
    phase2_cells = run_phase2(passing_cells, phase1_cells)

    # Phase 3
    baseline_results = run_phase3(passing_cells,
                                   {**phase1_cells, **phase2_cells})

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT C SUMMARY")
    print("=" * 70)
    print(f"{'Cell':<30} {'Accuracy':>10} {'Dead':>6} {'Pareto AUC':>12} {'SMS':>8}")
    print("-" * 66)
    for name, results in sorted({**phase1_cells, **phase2_cells}.items()):
        if not results:
            continue
        accs = [r['final_acc'] for r in results]
        deads = [r['dead_neurons'] for r in results]
        aucs = [r['pareto_auc'] for r in results]
        sms = [r.get('migration_score', 0) for r in results]
        print(f"{name:<30} {np.mean(accs):>9.4f} {np.mean(deads):>5.1f} "
              f"{np.mean(aucs):>11.1f} {np.mean(sms):>7.2f}")

    # Save
    def sanitize(obj):
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    save_data = {}
    for name, results in {**phase1_cells, **phase2_cells}.items():
        save_data[name] = [{k: sanitize(v) for k, v in r.items()
                            if k != 'metrics_log'} for r in results]
    if baseline_results:
        for name, results in baseline_results.items():
            save_data[f'baseline_{name}'] = [{k: sanitize(v) for k, v in r.items()}
                                              for r in results]

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {output_dir}/results.json")


if __name__ == '__main__':
    run_experiment()
