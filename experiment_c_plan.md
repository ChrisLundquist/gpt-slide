# Experiment C: Addressing the Critical Review (v2)

## Motivation

The critical review identified confounds that could mask a real migration effect:
1. AdamW's adaptive rates create "life support" for dying neurons
2. Lambda=0.1 was too weak — neurons never actually die
3. Full-batch training eliminates stochastic noise
4. 2-layer MLP with diagonal features is worst case for migration

## Key Design Fix: Decouple Decay from Learning Rate

The current formula `W1 *= (1 - lr * lambda)` couples decay strength to lr.
SGD at lr=0.01 gets 10x stronger effective decay than AdamW at lr=0.001 at
the same lambda. This confounds the optimizer comparison.

**New formula:** `W1 *= (1 - decay_rate)` where decay_rate is specified directly,
independent of lr. Decay rates tested: {0.0001, 0.001, 0.01} per step.
- 0.0001/step: over 20k steps, multiplicative factor = 0.135 (≈ old lambda=0.1 with AdamW)
- 0.001/step: factor = 2e-9 (neurons effectively die)
- 0.01/step: factor = 2e-87 (instant death, aggressive)

## Design: Sequential, Not Full Factorial

The 3-way factorial is underpowered. Instead, run sequentially:

### Phase 1: Can we kill neurons? (optimizer × decay_rate, 6 cells)

| | decay=0.0001 | decay=0.001 | decay=0.01 |
|---|---|---|---|
| **AdamW** | cell 1 | cell 2 | cell 3 |
| **SGD** | cell 4 | cell 5 | cell 6 |

10 seeds per cell = 60 runs. Full-batch (deterministic, cleaner signal).
Asymmetric decay alpha=5.0, 20k steps, single-task addition.

**Primary metric:** neuron death count (neurons with norm < 1% of their
initial norm at step 0).

**Decision gate:** For each cell, test whether mean death count across
10 seeds is significantly > 0 (one-sample t-test, p < 0.05). Any cell
that passes proceeds to Phase 2/3. If no cell passes, the decay is
always too weak relative to the optimizer. Stop.

### Phase 2: Does minibatch noise help? (at death-producing cells only)

For each cell from Phase 1 that passed the death gate:
- Re-run with minibatch (batch_size=512) instead of full-batch
- Use the SAME 10 seeds as Phase 1 (genuinely paired comparison)
- Compare death counts and accuracy

Runs: at most 6 cells × 10 seeds = 60 runs, likely fewer.

### Phase 3: Migration test (at cells with death)

For each cell where neurons actually die (Phase 1 or 2):
1. Compute frequency life histories — do frequencies move between neurons?
2. Compare Pareto AUC against prune+retrain at matched neuron count
3. Compute soft migration score

**Pooled analysis:** if multiple cells show death, pool across cells with
cell as a covariate rather than testing each cell separately. This
concentrates statistical power.

Baselines: prune+retrain, distillation, scratch — 10 seeds each, matched
to the death-producing cell's final neuron count.

### Phase 4: Depth extension (conditional)

Only if Phase 3 shows any sign of migration (positive migration score or
Pareto advantage over prune+retrain):

- 3-layer MLP: input → W1(128) → x² → W2(64) → x² → W3 → output
- Must first verify 3-layer grokking works (quick sweep)
- Apply decay to W1 (outermost layer) using the best cell's parameters
- 10 seeds

## Step 0: SGD Grokking Sweep

Before any factorial runs, verify SGD can grok.

Sweep: SGD lr in {0.01, 0.03, 0.1, 0.3}, momentum in {0.0, 0.9},
uniform decay_rate in {0.0001, 0.001}, 1 seed, addition task, 50k steps.

If no SGD configuration grokks within 50k steps, drop the SGD factor
entirely and run AdamW-only with the 3 decay rates + batch size.

Also verify AdamW grokking with decoupled decay: lr=1e-3,
decay_rate in {0.0001, 0.001, 0.01}, 3 seeds each. Confirm the decoupled
formula produces the same grokking behavior as the old lr-coupled version.
If decay_rate=0.01 prevents grokking (too aggressive), that's expected —
it will be tested only on already-grokked models in Phase 1.

## Metrics

### Per run (every 100 steps)
- Test accuracy
- Per-neuron norm (all 128)
- Death count (neurons with norm < 1% of their step-0 norm)
- Per-neuron effective update magnitude: ||W1[j,t] - W1[j,t-1]||
  (directly measures the optimizer's restoring force)

### At training end
- Pareto AUC (accuracy after zeroing at thresholds {16,32,48,64,80,96,112})
- Frequency life histories (for migration analysis)
- Soft migration score

## Statistical Design

- **Phase 1:** 2-way ANOVA (optimizer × decay_rate) on death count, 10 seeds.
  Main effects have 1 df (optimizer) and 2 df (decay_rate), interaction 2 df.
  With 10 seeds, detects d >= 0.9 for main effects at 80% power.
- **Phase 2:** Paired t-test (full-batch vs minibatch) within each death cell.
- **Phase 3:** Pooled paired t-test (asymmetric vs prune+retrain) across
  death-producing cells, with cell as covariate. If only 1 cell, standard
  paired t-test with 10 seeds.
- Use Wilcoxon signed-rank as sensitivity check given non-normal Pareto AUC.

## Compute Budget

| Phase | Runs | Time/Run | Total |
|-------|------|----------|-------|
| Step 0: SGD sweep | ~16 | 30s | ~8 min |
| Step 0: AdamW verify | 3 | 30s | ~2 min |
| Phase 1: 6 cells × 10 seeds | 60 | 30s | ~30 min |
| Phase 2: ≤6 cells × 10 seeds | ≤60 | 30s | ≤30 min |
| Phase 3: baselines × 10 seeds | ~30 | 30s | ~15 min |
| Phase 4: depth (conditional) | ~20 | 60s | ~20 min |
| **Total** | **~190** | | **~105 min** |

## Implementation Notes

- **Decoupled decay:** New function `decoupled_asymmetric_decay(W1, W2, alpha, rate)`
  that applies `W1[j] *= (1 - rate * (1 + alpha * j/(W-1)))` and same for W2 columns.
  Biases are NOT decayed (bias compensates for shrunken weights; decaying it breaks
  the gradient). Independent of lr. Do not use old AsymmetricDecay class.
- **W2 IS decayed** along the neuron axis (columns), same as W1. A neuron with
  near-zero W1 but large W2 would still contribute via optimizer gradient signal.
- **Phase 1 uses 20k steps** (not 50k). Step 0 SGD sweep uses 50k to find if
  grokking is possible at all. Once grokked experts exist, Phase 1 applies decay
  for 20k steps from the grokked checkpoint — same as Experiments A and B.
- **decay_rate=0.01 with alpha=5.0** gives max per-step factor 0.94 — after ~115
  steps the heaviest neuron is below 0.1% of initial norm. Use log-scale for norm
  plots and clamp to 1e-30 to avoid log(0).
- **SGD:** `torch.optim.SGD(params, lr, momentum)` — simple drop-in.
- **Death threshold:** per-neuron, computed at step 0 as `0.01 * initial_norm[j]`.
  Store initial norms; check against them at each logging step.
- **Per-neuron update tracking:** save W1 snapshot each logging step, compute
  ||W1[j,t] - W1[j,t-100]|| to measure effective restoring force.
- **3-layer MLP:** extend model.py with GrokMLP3 class. Two hidden layers with
  independent activations. Decay targets the first hidden layer only.
