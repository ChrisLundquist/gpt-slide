"""
Step 2 — Does Anything Happen?

Run asymmetric, uniform, and reversed conditions from MLP_converged.
Evaluate Gates 1, 2, 2.5. Run variance calibration.
"""

import os
import sys
import json
import torch
import dataclasses

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config, PHASE1_SEEDS, STEP2_CONDITIONS
from src.train import train_run, save_result
from src.gates import (evaluate_gate1, evaluate_gate2, evaluate_gate2_5,
                       compute_variance_calibration)
from src.viz import plot_column_norm_heatmap, plot_side_by_side_heatmaps


def run_step2(device: str = 'cuda'):
    output_dir = 'outputs'
    step1_dir = os.path.join(output_dir, 'step1')
    step2_dir = os.path.join(output_dir, 'step2')
    os.makedirs(step2_dir, exist_ok=True)

    # Load MLP_converged
    ckpt = torch.load(os.path.join(step1_dir, 'mlp_converged.pt'),
                      weights_only=True)
    converged_state = ckpt['model_state_dict']

    print("=" * 60)
    print("STEP 2: Asymmetric decay conditions")
    print("=" * 60)

    all_results = {}

    for cond_name, cond_overrides in STEP2_CONDITIONS.items():
        cond_results = []

        for seed in PHASE1_SEEDS:
            print(f"  {cond_name} seed={seed}...")

            config = dataclasses.replace(
                Config(),
                seed=seed, hidden_dim=256, device=device,
                output_dir=step2_dir,
                **cond_overrides,
            )
            result = train_run(config, starting_state=converged_state)
            save_result(result, step2_dir, f'{cond_name}_seed{seed}')
            cond_results.append(result)

            acc_add = result.get('final_acc_add', 0.0)
            acc_mul = result.get('final_acc_mul', 0.0)
            print(f"    add={acc_add:.4f}, mul={acc_mul:.4f}")

        all_results[cond_name] = cond_results

    # Generate heatmaps
    print("\nGenerating heatmaps...")
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    for cond_name, cond_results in all_results.items():
        # Average norms across seeds for the heatmap
        avg_log = []
        for step_idx in range(len(cond_results[0]['metrics_log'])):
            avg_norms = []
            for r in cond_results:
                if step_idx < len(r['metrics_log']):
                    avg_norms.append(r['metrics_log'][step_idx]['column_norms'])
            import numpy as np
            avg_entry = {
                'step': cond_results[0]['metrics_log'][step_idx]['step'],
                'column_norms': np.mean(avg_norms, axis=0).tolist(),
            }
            avg_log.append(avg_entry)
        plot_column_norm_heatmap(
            avg_log, f'{cond_name} (mean over {len(PHASE1_SEEDS)} seeds)',
            os.path.join(figures_dir, f'gate1_{cond_name}.png'))

    # Side-by-side for Gate 2
    plot_side_by_side_heatmaps(
        {name: all_results[name][0]['metrics_log'] for name in all_results},
        os.path.join(figures_dir, 'gate2_direction.png'))

    # Evaluate gates
    print("\n" + "=" * 60)
    print("GATE EVALUATION")
    print("=" * 60)

    g1_pass, g1_detail = evaluate_gate1(all_results)
    print(f"Gate 1: {'PASS' if g1_pass else 'FAIL'} — {g1_detail}")

    g2_pass, g2_detail = evaluate_gate2(all_results)
    print(f"Gate 2: {'PASS' if g2_pass else 'FAIL'} — {g2_detail}")

    g25_pass, g25_detail = evaluate_gate2_5(all_results)
    print(f"Gate 2.5: {'PASS' if g25_pass else 'FAIL'} — {g25_detail}")

    # Variance calibration
    vcal = compute_variance_calibration(all_results, Config().p)
    print(f"\nVariance calibration: std_lhef={vcal['std_lhef']:.4f}, "
          f"n_seeds={vcal['n_seeds']}")

    # Save manifest
    manifest = {
        'gate1': {'pass': g1_pass, 'detail': g1_detail},
        'gate2': {'pass': g2_pass, 'detail': g2_detail},
        'gate2_5': {'pass': g25_pass, 'detail': g25_detail},
        'variance_calibration': vcal,
        'proceed_to_step3': g1_pass and g2_pass and g25_pass,
    }
    with open(os.path.join(step2_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    if not manifest['proceed_to_step3']:
        print("\n*** One or more gates FAILED. See decision matrix. ***")
    else:
        print("\n*** All gates PASSED. Proceed to Step 3. ***")

    return all_results, manifest


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_step2(device)
