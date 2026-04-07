# Experiment Results

## Phase 1: Kill Gates

### Step 1 — Sanity Check (PASS)
- 20/20 experts grokked at lambda=0.1 (10 add + 10 mul)
- Grokking times: 10k-49k steps (add: 18k-49k, mul: 10k-44k)
- Addition Fourier structure: 72.7% neurons with IPR > 0.2
- Multiplication: no Fourier-sparse structure (expected for a*b mod p)
- MLP_cat: 100%/100% immediately after concatenation (two-head model)
- MLP_converged: 100%/100% after 10k steps uniform decay

### Step 2 — Asymmetric Decay (Gates 1, 2, 2.5 PASS)
Settings: alpha=5.0, lambda_base=0.1, 20k steps, 4 seeds

| Condition | Add Acc | Mul Acc | Norm Ratio (R/L) |
|-----------|---------|---------|------------------|
| Asymmetric | 99.97% | 97.35% | 0.694 |
| Uniform | 99.99% | 100.0% | ~1.0 |
| Reversed | 92.7% | 99.94% | 1.569 |

- **Gate 1 PASS**: norm ratio 0.694 < 0.7
- **Gate 2 PASS**: asymmetric 0.694 vs reversed 1.569 (mirror image)
- **Gate 2.5 PASS**: add 99.9% after zeroing j >= 128
- Variance calibration: std_lhef = 0.011, SNR = 4.5

### Step 3 — Migration vs Relearning (Gate 3 FAIL)
Settings: same as Step 2 + activation-zeroing/weight-freeze hooks

| Condition | Add Acc | Mul Acc |
|-----------|---------|---------|
| Asymmetric (Step 2) | 99.97% | 97.35% |
| Asym + activation-zeroed | 99.81% | 99.54% |
| Asym + weight-frozen | 99.83% | 98.54% |
| Uniform (Step 2) | 99.99% | 100.0% |
| Uniform + activation-zeroed | 99.63% | 96.41% |

**Gate 3 diff-in-diff results:**
- LHEF effect: 0.012 (threshold 0.05) -- **FAIL**
- SMS effect: 3.79 (threshold 0.5) -- PASS
- ACC effect: -0.012 (threshold 0.05) -- **FAIL**

## Interpretation

Per the decision matrix (Gates 1,2,2.5 pass, Gate 3 fail):

> Compaction works, but mechanism is relearning under constraint.
> Reframe as structured soft pruning. Run Phase 2 fallback.

Asymmetric decay successfully creates directional compaction (kills neurons on the
high-decay side) and the effect is controllable (reversed gradient produces mirror
image). However, surviving neurons learn from data, not from dying neurons' activations.
The dying neurons' forward-pass contributions do not help the surviving neurons
acquire features — blocking those contributions (activation-zeroing) makes no
meaningful difference.

## Next: Phase 2 Fallback

Compare asymmetric decay against distillation and structured GMP to determine
whether it's competitive as a compression method even without migration.
