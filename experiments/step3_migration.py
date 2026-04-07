"""
Step 3 — Migration or Relearning?

Run activation-zeroed and weight-frozen conditions from MLP_converged.
Evaluate Gate 3 using difference-in-differences with Step 2 results.
"""

import os
import sys
import json
import torch
import dataclasses

torch.set_float32_matmul_precision('high')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config, PHASE1_SEEDS, SEEDS, STEP3_CONDITIONS
from src.train import train_run, save_result
from src.gates import evaluate_gate3


def load_step2_results(step2_dir: str) -> dict:
    """Load Step 2 metrics for the conditions needed by Gate 3."""
    results = {}
    for cond in ['asymmetric', 'uniform']:
        cond_results = []
        for seed in PHASE1_SEEDS:
            metrics_path = os.path.join(step2_dir, f'{cond}_seed{seed}', 'metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    cond_results.append(json.load(f))
        results[cond] = cond_results
    return results


def run_step3(device: str = 'cuda'):
    output_dir = 'outputs'
    step1_dir = os.path.join(output_dir, 'step1')
    step2_dir = os.path.join(output_dir, 'step2')
    step3_dir = os.path.join(output_dir, 'step3')
    os.makedirs(step3_dir, exist_ok=True)

    # Load MLP_converged
    ckpt = torch.load(os.path.join(step1_dir, 'mlp_converged.pt'),
                      weights_only=True)
    converged_state = ckpt['model_state_dict']

    print("=" * 60)
    print("STEP 3: Migration vs Relearning")
    print("=" * 60)

    step3_results = {}

    for cond_name, cond_overrides in STEP3_CONDITIONS.items():
        cond_results = []

        for seed in PHASE1_SEEDS:
            print(f"  {cond_name} seed={seed}...")

            config = dataclasses.replace(
                Config(),
                seed=seed, hidden_dim=256, device=device,
                output_dir=step3_dir,
                **cond_overrides,
            )
            result = train_run(config, starting_state=converged_state)
            save_result(result, step3_dir, f'{cond_name}_seed{seed}')
            cond_results.append(result)

            acc_add = result.get('final_acc_add', 0.0)
            acc_mul = result.get('final_acc_mul', 0.0)
            print(f"    add={acc_add:.4f}, mul={acc_mul:.4f}")

        step3_results[cond_name] = cond_results

    # Load Step 2 results for comparison
    print("\nLoading Step 2 results for comparison...")
    step2_results = load_step2_results(step2_dir)

    # Evaluate Gate 3
    print("\n" + "=" * 60)
    print("GATE 3 EVALUATION")
    print("=" * 60)

    # Combine step2 and step3 results
    # Gate 3 needs: asymmetric (step2), asym_actzeroed (step3),
    #               uniform (step2), uniform_actzeroed (step3)
    combined_step2 = step2_results  # has 'asymmetric' and 'uniform'
    combined_step3 = step3_results  # has 'asym_actzeroed' and 'uniform_actzeroed'

    g3_result, g3_detail = evaluate_gate3(combined_step2, combined_step3)
    print(f"Gate 3: {g3_result.upper()} — {g3_detail}")

    # Save manifest
    manifest = {
        'gate3': {'result': g3_result, 'detail': g3_detail},
    }

    if g3_result == 'pass':
        print("\n*** Gate 3 PASSED. Migration has a plausible mechanism. ***")
        print("*** Proceed to full protocol (protocol_v2.md). ***")
    elif g3_result == 'ambiguous':
        print("\n*** Gate 3 AMBIGUOUS. Escalate to 10 seeds and re-evaluate. ***")
    else:
        print("\n*** Gate 3 FAILED. Migration ruled out. ***")
        print("*** Reframe as structured soft pruning. Run Phase 2 fallback. ***")

    with open(os.path.join(step3_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    return step3_results, manifest


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_step3(device)
