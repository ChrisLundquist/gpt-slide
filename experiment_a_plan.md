# Experiment A: Single-Task Densification (Revised)

## Hypothesis

Asymmetric weight decay can densify a grokked single-task model while preserving
accuracy, and the shared W2 provides an interaction mechanism that makes this
more effective than naive pruning.

## Setup

- Model: GrokMLP W=128, single head, quadratic activation, addition mod 113
- Starting points: 5 different grokked checkpoints (seeds 42, 137, 256, 512, 1024)
  to provide genuine between-run variance. Each seed produces a different grokked
  solution (different frequency-to-neuron assignments).
- Full-batch training (deterministic within each checkpoint — variance comes from
  different starting checkpoints, not stochastic sampling).

## Conditions (7 total)

### 1. Asymmetric Decay (method under test)
- alpha=5.0, lambda_base=0.1, 20k steps, single head
- Continue training on addition data from grokked checkpoint

### 2. Uniform Decay (continued training baseline)
- alpha=0.0, lambda_base=0.1, 20k steps
- Same as Condition 1 but no asymmetry. Separates "continued training helps"
  from "asymmetric decay specifically helps."

### 3. Gradual Magnitude Pruning (GMP)
- Structured: prune entire neuron columns by L2 norm, cubic schedule
- Ramp over first 15k of 20k steps, pruning mask updated every 100 steps
- Target: same final active neuron count as Condition 1
- Uniform decay lambda=0.1 throughout
- Dependency: run Condition 1 first to determine target width

### 4. Instant Prune + Retrain
- At step 0: zero out lowest-norm neurons to match Condition 1's final count
- Retrain for 20k steps with uniform decay
- Tests: does gradual death help vs instant death + same training budget?

### 5. Asymmetric Decay + Gradient Severed
- Same as Condition 1, but detach() the hidden activations for dying neurons
  (j >= 96) before they reach W2 in the backward pass. Dying neurons still
  contribute to the forward pass (their activations flow through W2 normally)
  but no gradient flows back through W2 into dying neurons or through W2
  columns for dying neurons into surviving neurons' loss signal.
- This cleanly isolates the gradient interaction without removing forward-pass
  contributions (unlike zeroing W2 columns, which does both).

### 6. Reversed Asymmetric Decay (direction control)
- alpha=5.0, reverse=True (high decay on LEFT)
- Confirms the effect is directional

### 7. Train From Scratch (capacity baseline)
- Fresh model at W=target_width (from Condition 1's result)
- Both matched FLOPs and matched steps as sub-conditions
- Baseline: can a smaller model learn the task as well from scratch?

## Seeds

5 grokked checkpoints from Step 1 (seeds 42, 137, 256, 512, 1024).
Conditions 1-6: 5 seeds each = 30 runs.
Condition 7: 5 seeds × 2 matchings = 10 runs.
Total: 40 runs.

## Metrics (every 100 steps)

### Primary
- Test accuracy on addition
- Column norm profile (per-neuron L2 norm of W1 rows)
- Pareto frontier: accuracy after zeroing neurons at thresholds T ∈ {32, 48, 64, 80, 96}
- Frequency energy matrix: (128, p) per logging step — for life history tracking

### At training end
- Pareto AUC (area under accuracy-vs-width curve) as scalar summary
- Frequency coverage: how many of the 56 conjugate pairs are represented?
- Per-frequency phase angles for migration/relearning analysis

## Analysis

### Statistical
- Pareto AUC compared across conditions via paired t-test (Holm-Bonferroni corrected)
- At each threshold T: paired t-test of accuracy across conditions
- Temporal overlap score for frequency life histories (correlation of decay/rise curves)
- Phase difference: circular mean and von Mises concentration, test |Δφ| < π/4 vs chance

### Decision criteria
- If Condition 1 > Conditions 2,3,4: asymmetric decay adds value beyond continued training
- If Condition 1 ≈ Condition 5: gradient interaction through W2 doesn't matter
- If Condition 1 > Condition 5: the teaching signal through W2 contributes
- If Condition 1 ≈ Condition 3: asymmetric decay ≈ structured pruning (thin contribution)
- If overlapping ridges + phase continuity: migration
- If gaps + W2 column changes: W2 adaptation (relearning)

## Implementation Notes

- Disable grok early-stopping for all densification conditions (model starts at 100%)
- GMP scheduler: add to train_run via config flag
- Gradient-severed condition: register a backward hook that detaches activations
  for dying neurons, or use torch.no_grad selectively on the hidden→output path
- Condition 7 width determined after Condition 1 runs
- torch.compile for conditions without hooks (1, 2, 3, 4, 6, 7)

## Compute Estimate

- 40 runs × ~30s each = ~20 minutes
- Analysis: ~5 minutes
- Total: ~25 minutes
