# Detailed Results

See [README.md](README.md) for the paper-format writeup covering all experiments.

This file contains raw numerical results for reference.

## Phase 1: Kill Gates

### Step 1 — Sanity Check
- 20/20 experts grokked at lambda=0.1 (10 add + 10 mul, seeds 42-8192)
- Grokking times: add 18k-49k steps, mul 10k-44k steps
- Addition Fourier structure: 72.7% neurons with IPR > 0.2 (93/128)
- Multiplication: 0% Fourier-sparse (expected for a*b mod p)
- MLP_cat: 100%/100% immediately (two-head concatenation)
- MLP_converged: 100%/100% after 10k steps uniform decay

### Step 2 — Asymmetric Decay
Settings: alpha=5.0, lambda_base=0.1, 20k steps, 4 seeds, minibatch

| Condition | Add Acc | Mul Acc | Norm Ratio (R/L) |
|-----------|---------|---------|------------------|
| Asymmetric | 99.97% | 97.35% | 0.694 |
| Uniform | 99.99% | 100.0% | ~1.0 |
| Reversed | 92.7% | 99.94% | 1.569 |

Gates: 1 PASS (0.694 < 0.7), 2 PASS (mirror), 2.5 PASS (99.9% after zeroing)

### Step 3 — Migration vs Relearning
Gate 3 FAIL: LHEF effect 0.012 (threshold 0.05), ACC effect -0.012 (threshold 0.05)

## Phase 2 Fallback

| Method | Add Acc | Mul Acc | Width |
|--------|---------|---------|-------|
| Asymmetric decay | 99.97% | 97.35% | 256 (half dead) |
| Distillation T=4 | 99.96% | 99.97% | 128 |
| Structured GMP | 99.97% | 100.0% | 131 |

## Experiment A: Single-Task Densification

| Condition | Accuracy | Surviving | Pareto AUC |
|-----------|----------|-----------|------------|
| Asymmetric | 97.91% | 128 | 12.2 |
| Uniform | 99.99% | 128 | 10.6 |
| Gradient-severed | 98.93% | 116.6 | 18.3 |
| Scratch W=128 | 99.95% | 128 | 36.3 |

## Experiment B: Stop-Gradient Pruning

At 50% compression (sever 64/128 neurons):

| Method | Pareto AUC | p vs sever |
|--------|-----------|------------|
| Prune + retrain | 35.2 | 0.011 |
| Scratch W=64 | 30.1 | 0.199 |
| Sever ramp | 26.4 | 0.111 |
| Sever instant | 21.3 | — |
| Asymmetric decay | 21.4 | 0.993 |
| Distillation | 10.1 | 0.095 |

## Experiment C: Optimizer x Decay Strength

SGD grokking sweep: 0/16 configurations grokked (50k steps each).

AdamW with decoupled decay:

| Decay Rate | Accuracy | Dead Neurons | Pareto AUC |
|-----------|----------|-------------|-----------|
| 0.0001/step | 98.6% | 3.0 | 21.4 |
| 0.001/step | 79.1% | 0.2 | 29.6 |
| 0.01/step | 0.26% | 0.5 | 0.5 |
| 0.01/step + minibatch | 0.71% | 118.7 | 0.7 |

Migration score: -0.07 ± 1.62 (p=0.55, not significant)
