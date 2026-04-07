"""
Standalone gate evaluation script.

Loads results from Steps 2 and 3, evaluates all gates, prints decision matrix.
Can be re-run without re-training.
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import PHASE1_SEEDS
from src.gates import (evaluate_gate1, evaluate_gate2, evaluate_gate2_5,
                       compute_variance_calibration, evaluate_gate3)
from src.viz import plot_side_by_side_heatmaps


def load_results(step_dir: str, conditions: list[str]) -> dict:
    """Load metrics.json for each condition and seed."""
    results = {}
    for cond in conditions:
        cond_results = []
        for seed in PHASE1_SEEDS:
            path = os.path.join(step_dir, f'{cond}_seed{seed}', 'metrics.json')
            if os.path.exists(path):
                with open(path) as f:
                    cond_results.append(json.load(f))
        if cond_results:
            results[cond] = cond_results
    return results


def run_evaluation():
    output_dir = 'outputs'
    step2_dir = os.path.join(output_dir, 'step2')
    step3_dir = os.path.join(output_dir, 'step3')
    figures_dir = os.path.join(output_dir, 'figures')
    summary_dir = os.path.join(output_dir, 'summary')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    print("=" * 60)
    print("GATE EVALUATION")
    print("=" * 60)

    # Load Step 2
    step2 = load_results(step2_dir, ['asymmetric', 'uniform', 'reversed'])

    if not step2:
        print("No Step 2 results found. Run step2_asymmetric.py first.")
        return

    # Gate 1
    g1_pass, g1_detail = evaluate_gate1(step2)
    print(f"\nGate 1 (Norm asymmetry):  {'PASS' if g1_pass else 'FAIL'}")
    print(f"  {g1_detail}")

    # Gate 2
    g2_pass, g2_detail = evaluate_gate2(step2)
    print(f"\nGate 2 (Direction test):  {'PASS' if g2_pass else 'FAIL'}")
    print(f"  {g2_detail}")

    # Gate 2.5
    g25_pass, g25_detail = evaluate_gate2_5(step2)
    print(f"\nGate 2.5 (Compaction):   {'PASS' if g25_pass else 'FAIL'}")
    print(f"  {g25_detail}")

    # Variance calibration
    vcal = compute_variance_calibration(step2, 113)
    print(f"\nVariance calibration:    std_lhef={vcal['std_lhef']:.4f}")

    # Gate 3
    step3 = load_results(step3_dir, ['asym_actzeroed', 'asym_frozen',
                                      'uniform_actzeroed'])
    if step3:
        g3_result, g3_detail = evaluate_gate3(step2, step3)
        print(f"\nGate 3 (Migration):      {g3_result.upper()}")
        print(f"  {g3_detail}")
    else:
        g3_result, g3_detail = 'not_run', 'Step 3 results not found'
        print(f"\nGate 3: NOT RUN (run step3_migration.py first)")

    # Decision matrix
    print("\n" + "=" * 60)
    print("DECISION MATRIX")
    print("=" * 60)

    if not g1_pass:
        print("Gate 1 FAILED → Decay gradient has no effect. STOP.")
    elif not g2_pass:
        print("Gate 2 FAILED → No directional control. Reframe or STOP.")
    elif not g25_pass:
        print("Gate 2.5 FAILED → Compaction destroys features. STOP.")
    elif g3_result == 'fail':
        print("Gate 3 FAILED → Mechanism is relearning. Run Phase 2 fallback.")
    elif g3_result == 'ambiguous':
        print("Gate 3 AMBIGUOUS → Escalate to 10 seeds.")
    elif g3_result == 'pass':
        print("ALL GATES PASSED → Proceed to full protocol.")
    else:
        print("Gate 3 not yet evaluated.")

    # Save summary
    summary = {
        'gate1': {'pass': g1_pass, 'detail': g1_detail},
        'gate2': {'pass': g2_pass, 'detail': g2_detail},
        'gate2_5': {'pass': g25_pass, 'detail': g25_detail},
        'variance_calibration': vcal,
        'gate3': {'result': g3_result, 'detail': g3_detail},
    }
    with open(os.path.join(summary_dir, 'gate_results.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {summary_dir}/gate_results.json")


if __name__ == '__main__':
    run_evaluation()
