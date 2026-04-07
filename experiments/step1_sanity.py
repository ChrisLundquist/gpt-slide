"""
Step 1 — Sanity Check

1. Train 10 MLP_add + 10 MLP_mul experts
2. Verify grokking and Fourier structure
3. Select best per task, concatenate → MLP_cat
4. Converge with uniform decay → MLP_converged
"""

import os
import sys
import json
import torch
import dataclasses

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config, SEEDS, LAMBDA_FALLBACK
from src.train import train_run, save_result, set_seed
from src.model import GrokMLP
from src.concat import concatenate_experts, calibrate_mixing
from src.data import build_joint_dataset
from src.train import evaluate
from src.metrics import compute_ipr, neuron_fourier_spectrum


def train_experts(output_dir: str, device: str = 'cuda'):
    """Train 10 addition + 10 multiplication experts."""
    experts_dir = os.path.join(output_dir, 'step1', 'experts')
    os.makedirs(experts_dir, exist_ok=True)

    results = {'add': {}, 'mul': {}}

    for task in ['add', 'mul']:
        for seed in SEEDS:
            print(f"Training {task} expert, seed={seed}...")

            # Try lambda fallback schedule
            for lam in LAMBDA_FALLBACK:
                config = dataclasses.replace(
                    Config(),
                    seed=seed, task=task, condition='expert',
                    lambda_base=lam, alpha=0.0,
                    device=device,
                    output_dir=experts_dir,
                )
                result = train_run(config)
                final_acc = result.get('final_acc', 0.0)

                if final_acc >= 0.99:
                    print(f"  Grokked! lambda={lam}, acc={final_acc:.4f}, "
                          f"steps={result['steps_trained']}")
                    break
                else:
                    print(f"  lambda={lam}: acc={final_acc:.4f}, trying next...")
            else:
                print(f"  WARNING: {task} seed={seed} did not grok at any lambda!")

            save_result(result, experts_dir, f'{task}_seed{seed}')
            results[task][seed] = {
                'final_acc': result.get('final_acc', 0.0),
                'steps_trained': result['steps_trained'],
                'path': os.path.join(experts_dir, f'{task}_seed{seed}'),
            }

    return results


def select_best(expert_results: dict, task: str) -> tuple[int, dict]:
    """Select best seed by lowest test loss (highest accuracy as proxy)."""
    best_seed = max(expert_results[task],
                    key=lambda s: expert_results[task][s]['final_acc'])
    return best_seed, expert_results[task][best_seed]


def verify_fourier_structure(model: GrokMLP, p: int, threshold: float = 0.5) -> float:
    """Check fraction of neurons with IPR > threshold."""
    W1 = model.W1.data
    n_pass = 0
    for j in range(W1.shape[0]):
        spec = neuron_fourier_spectrum(W1[j], p)
        if compute_ipr(spec) > threshold:
            n_pass += 1
    return n_pass / W1.shape[0]


def run_step1(device: str = 'cuda'):
    output_dir = 'outputs'
    step1_dir = os.path.join(output_dir, 'step1')

    # 1. Train experts
    print("=" * 60)
    print("STEP 1: Training experts")
    print("=" * 60)
    expert_results = train_experts(output_dir, device)

    # 2. Select best per task
    best_add_seed, best_add = select_best(expert_results, 'add')
    best_mul_seed, best_mul = select_best(expert_results, 'mul')
    print(f"\nBest add expert: seed={best_add_seed}, acc={best_add['final_acc']:.4f}")
    print(f"Best mul expert: seed={best_mul_seed}, acc={best_mul['final_acc']:.4f}")

    # Load best models
    add_ckpt = torch.load(os.path.join(best_add['path'], 'checkpoint.pt'),
                          weights_only=True)
    mul_ckpt = torch.load(os.path.join(best_mul['path'], 'checkpoint.pt'),
                          weights_only=True)

    config = Config(device=device)
    model_add = GrokMLP(config.input_dim, 128, config.output_dim).to(device)
    model_add.load_state_dict(add_ckpt['model_state_dict'])

    model_mul = GrokMLP(config.input_dim, 128, config.output_dim).to(device)
    model_mul.load_state_dict(mul_ckpt['model_state_dict'])

    # 3. Verify Fourier structure
    frac_add = verify_fourier_structure(model_add, config.p)
    frac_mul = verify_fourier_structure(model_mul, config.p)
    print(f"\nFourier structure: add={frac_add:.2%}, mul={frac_mul:.2%}")
    assert frac_add > 0.8, f"Add expert Fourier fraction {frac_add:.2%} < 80%"
    assert frac_mul > 0.8, f"Mul expert Fourier fraction {frac_mul:.2%} < 80%"

    # 4. Concatenate
    print("\nConcatenating experts...")
    model_cat = concatenate_experts(model_add, model_mul).to(device)

    # Verify concatenation accuracy
    data = build_joint_dataset(config.p, config.train_frac, config.data_seed, device)
    _, acc_add = evaluate(model_cat, data['add'][2], data['add'][3])
    _, acc_mul = evaluate(model_cat, data['mul'][2], data['mul'][3])
    print(f"MLP_cat accuracy: add={acc_add:.4f}, mul={acc_mul:.4f}")

    # Calibrate if needed
    ratio = calibrate_mixing(model_cat, data['add'][2], data['add'][3],
                              data['mul'][2], data['mul'][3], half=128)
    print(f"Logit scale ratio: {ratio:.2f}")

    _, acc_add = evaluate(model_cat, data['add'][2], data['add'][3])
    _, acc_mul = evaluate(model_cat, data['mul'][2], data['mul'][3])
    print(f"After calibration: add={acc_add:.4f}, mul={acc_mul:.4f}")
    assert acc_add > 0.95, f"Add accuracy {acc_add:.4f} < 95%"
    assert acc_mul > 0.95, f"Mul accuracy {acc_mul:.4f} < 95%"

    torch.save({'model_state_dict': model_cat.state_dict()},
               os.path.join(step1_dir, 'mlp_cat.pt'))

    # 5. Converge
    print("\nConverging MLP_cat with uniform decay...")
    converge_config = dataclasses.replace(
        Config(),
        hidden_dim=256, task='joint', condition='converge',
        lambda_base=1.0, alpha=0.0,
        max_steps=10_000,
        seed=42, device=device,
        output_dir=step1_dir,
    )
    converge_result = train_run(converge_config,
                                starting_model=model_cat)

    acc_add_final = converge_result.get('final_acc_add', 0.0)
    acc_mul_final = converge_result.get('final_acc_mul', 0.0)
    print(f"MLP_converged: add={acc_add_final:.4f}, mul={acc_mul_final:.4f}")
    assert acc_add_final > 0.98, f"Add accuracy {acc_add_final:.4f} < 98%"
    assert acc_mul_final > 0.98, f"Mul accuracy {acc_mul_final:.4f} < 98%"

    torch.save({
        'model_state_dict': converge_result['model_state'],
        'config': dataclasses.asdict(converge_config),
    }, os.path.join(step1_dir, 'mlp_converged.pt'))

    save_result(converge_result, step1_dir, 'mlp_converged')

    # Manifest
    manifest = {
        'best_add_seed': best_add_seed,
        'best_mul_seed': best_mul_seed,
        'add_acc': best_add['final_acc'],
        'mul_acc': best_mul['final_acc'],
        'fourier_frac_add': frac_add,
        'fourier_frac_mul': frac_mul,
        'cat_acc_add': acc_add,
        'cat_acc_mul': acc_mul,
        'converged_acc_add': acc_add_final,
        'converged_acc_mul': acc_mul_final,
        'mlp_converged': os.path.join(step1_dir, 'mlp_converged.pt'),
        'all_expert_results': {
            task: {str(s): v for s, v in seeds.items()}
            for task, seeds in expert_results.items()
        },
    }
    with open(os.path.join(step1_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print("=" * 60)
    return manifest


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_step1(device)
