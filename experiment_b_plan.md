# Experiment B: Stop-Gradient-Assisted Structured Pruning (v2)

## Core Insight

The optimizer's gradients through W2 keep dying neurons alive ("life support").
Detaching dying neurons from the backward pass breaks this and enables actual
neuron death while maintaining a smooth training signal through the forward pass.

## Method: Sever-and-Decay

1. Rank neurons by W1 column norm at step 0. Fix this ordering.
2. Select the bottom-K neurons as the "sever set."
3. Register a forward hook that detaches severed neurons' activations from
   the backward pass. They still contribute to the forward pass.
4. Apply decay (lambda=0.1) to severed neurons' W1 rows, b1 entries, AND
   W2 columns. Surviving neurons get no decay — only gradient-driven updates.
5. Train for 20k steps. Severed neurons shrink to zero; surviving neurons
   absorb the function through data-driven gradients.

**Note on gradient flow:** The sever hook detaches activations AFTER the
quadratic activation, so severed neurons' W1 rows and b1 receive zero
gradient. However, W2 columns for severed neurons still receive gradients
(the loss depends on their forward-pass activations, just not differentiably
through W1). Applying decay to W2 columns of severed neurons prevents W2
from amplifying shrinking activations and counteracting the intended death.

## Seeds

10 grokked addition checkpoints (seeds 42, 137, 256, 512, 1024, 1337, 2048,
3141, 4096, 8192) from Step 1. All conditions use the same 10 checkpoints.

## Conditions

### Main Sweep: Sever Fraction (6 fractions × 10 seeds = 60 runs)

Sever {16, 32, 48, 64, 80, 96} neurons out of 128.
Selection: lowest W1 norm at step 0, ordering fixed for entire run.
Maps the full Pareto frontier of accuracy vs surviving width.

### Schedule Comparison at 50% (2 schedules × 10 seeds = 20 runs)

| Schedule | Description |
|----------|-------------|
| Instant | Sever all 64 at step 0 |
| Linear ramp | Grow sever set from 0 to 64 over first 10k steps, fixed ordering from step 0 |

Ordering is frozen at step 0 (rank by initial norm). No dynamic re-ranking
to avoid the feedback confound where severing changes norms which changes
who gets severed.

### Baselines at 3 Fractions {32, 64, 96} (4 baselines × 3 fractions × 10 seeds = 120 runs)

| Baseline | Description |
|----------|-------------|
| Prune + retrain | Zero lowest-norm neurons at step 0, retrain 20k steps, no decay |
| Train from scratch | Fresh model at target width, train 20k steps |
| Distillation | Student at target width, soft targets from W=128 teacher (T=4), 20k steps |
| Asymmetric decay | alpha=5.0, lambda=0.1, no severing (the Experiment A method) |

### Mechanistic Controls at 50% (3 conditions × 10 seeds = 30 runs)

| Condition | Forward pass | Gradient to W1 | What it tests |
|-----------|-------------|----------------|---------------|
| Sever-and-decay | Active | Blocked | THE METHOD |
| Hard-prune + retrain | Zeroed | Blocked | Does the forward pass of dying neurons help? |
| Standard asymmetric | Active | Active | Does severing the gradient enable death? |

Note: "forward-severed with gradient active" is degenerate — zeroing
activations produces zero gradient by chain rule, making it identical to
"both severed." We drop it and compare directly against hard pruning.

If sever-and-decay > hard-prune: forward pass provides useful smoothing signal.
If sever-and-decay > standard asymmetric: severing enables death.
If hard-prune ≈ sever-and-decay: forward pass is irrelevant — just hard-prune.

## Total Runs

| Group | Runs |
|-------|------|
| Main sweep (6 fractions) | 60 |
| Schedules (2 at 50%) | 20 |
| Baselines (4 × 3 fractions) | 120 |
| Mechanistic (3 at 50%) | 30 |
| **Total** | **230** |

Overlaps (run once, count in multiple analyses):
- Main sweep 64-neuron = instant schedule at 50% = mechanistic sever-and-decay (10 runs)
- Baselines "asymmetric decay" at 64 = mechanistic "standard asymmetric" (10 runs)
- Baselines "prune + retrain" at 64 = mechanistic "hard-prune + retrain" (10 runs)

After deduplication: **200 unique runs**, ~100 minutes.

## Metrics

### Per run (every 100 steps)
- Test accuracy
- Surviving neuron count (norm > 1e-4)
- Column norm profile

### At training end
- Final test accuracy
- Frequency coverage (conjugate pairs with IPR > 0.2 in surviving neurons)
- Accuracy recovery trajectory (how fast does accuracy return after severing?)

### Summary statistics
- Pareto AUC (trapezoidal integration of accuracy vs width curve)
- Dominance count (at how many fractions does method A beat method B?)

## Statistical Design

- 10 seeds, paired tests (same checkpoint across conditions)
- **Primary analysis** (Pareto frontier): permutation test on AUC difference
  between sever-and-decay and each baseline. Holm-Bonferroni across 4 baselines.
  Only decompose to per-fraction paired t-tests if overall AUC test is significant.
- **Schedule comparison**: single paired t-test (2 conditions, no correction needed)
- **Mechanistic controls**: paired t-tests of sever-and-decay vs hard-prune and
  vs standard asymmetric. Holm-Bonferroni across 2 comparisons.
- **Family correction**: the three analysis groups (Pareto, schedule, mechanistic)
  are pre-specified. Pareto is primary; schedule and mechanistic are secondary.
  No cross-family correction — each family controls its own alpha at 0.05.

With 10 seeds at alpha=0.05, paired t-test detects d >= 1.0 at ~80% power.
The mechanistic comparisons (Holm-corrected across 2) require d >= 1.2.

## Implementation

Dedicated `experiments/experiment_b.py` — does NOT overload train_run().

Key components:
- `sever_hook(cutoff_ref)`: closure over a mutable list `[cutoff]`, cat + detach
- `binary_decay(W1, b1, sever_mask, lambda_val, lr)`: decay only masked neurons
- `run_condition(model, sever_set, schedule, ...)`: single run with all metrics
- No torch.compile for hooked runs; compile for baselines without hooks
- Disable grok early-stop (model starts at 100%)
