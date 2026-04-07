# Asymmetric Weight Decay Does Not Cause Knowledge Migration

**The shared output pathway acts as life support for dying neurons, not a teaching channel.**

## Abstract

We test whether position-dependent weight decay can cause learned features to
migrate from heavily-regularized neurons to lightly-regularized neurons in neural
networks trained on modular arithmetic. Across three experimental configurations
(multi-task with separate output heads, multi-task with shared output, single-task
with shared output), we find no evidence of knowledge migration. Surviving neurons
relearn features from data regardless of whether dying neurons contribute
activations. In a single-task setting, we discover a "life support" effect: the
optimizer's gradients through shared output weights actively sustain dying neurons
rather than transferring their knowledge, and severing this gradient pathway
*improves* compression efficiency (p=0.028). Asymmetric weight decay is not
competitive with knowledge distillation or structured magnitude pruning as a
compression method. Our activation-zeroing causal test provides the first direct
experimental evidence that feature transfer between neurons does not occur during
regularization-driven pruning.

## 1. Introduction

Neural network pruning typically removes low-magnitude weights or neurons after
training. A natural question is whether regularization can be structured to guide
*where* knowledge ends up during training — applying stronger decay to some neurons
to drive their learned features toward surviving neurons with lighter decay.

We call this **asymmetric weight decay**: for hidden neuron j in a network of width W,
the decay rate is `lambda_j = lambda_base * (1 + alpha * j / (W-1))`, creating a
linear gradient from light decay (j=0) to heavy decay (j=W-1).

The hypothesis is that this creates a "knowledge migration" pressure: as
heavily-decayed neurons lose representational capacity, their features should flow
through the shared computation graph to lightly-decayed neurons, producing a denser
model without retraining from scratch.

We test this on modular arithmetic grokking (Nanda et al. 2023), where learned
features (Fourier modes) are discrete, identifiable, and trackable — an ideal
setting for detecting migration if it exists.

## 2. Experimental Setting

### Architecture
- 2-layer MLP with quadratic activation (x -> x^2)
- Input: one-hot encoding of (a, b) in Z_113 x Z_113, dimension 226
- Hidden: width 128 (single expert) or 256 (concatenated)
- Output: 113 classes (softmax)
- Tasks: (a + b) mod 113 (addition), (a * b) mod 113 (multiplication)

### Training
- AdamW optimizer, lr=1e-3, weight_decay=0 (all decay applied externally)
- External asymmetric decay after each optimizer step
- p=113, 50/50 train-test split, fixed data seed

### Fourier Analysis
For the addition task, grokked networks learn clean Fourier representations: each
hidden neuron's input weights, reshaped as (2, p) and DFT'd, concentrate energy at
a single conjugate frequency pair (k, p-k). We track these frequencies per neuron
over training to detect migration, relearning, or redistribution.

Multiplication mod p does not produce clean Fourier structure with one-hot encoding
(it requires discrete logarithm representations), limiting mechanistic analysis to
the addition task.

## 3. Experiments and Results

### 3.1 Multi-Task Experiment (Two Output Heads)

**Setup:** Train addition and multiplication experts (W=128 each), concatenate along
hidden dimension to W=256, train jointly with asymmetric decay. Two output heads
(one per task) route gradients to the appropriate neuron group.

**Fail-fast protocol** with sequential gates:

| Gate | Test | Result |
|------|------|--------|
| Gate 1 | Norm asymmetry: right/left ratio < 0.7 | **PASS** (0.694) |
| Gate 2 | Direction control: forward mirrors reversed | **PASS** (0.694 vs 1.569) |
| Gate 2.5 | Compaction quality: >80% accuracy after zeroing right half | **PASS** (99.9%) |
| Gate 3 | Migration: dying neurons' activations help survivors | **FAIL** |

**Gate 3 detail** (difference-in-differences, activation-zeroing causal test):

| Metric | Effect | Threshold | Result |
|--------|--------|-----------|--------|
| Left-half energy fraction | 0.012 | 0.05 | FAIL |
| Accuracy after zeroing | -0.012 | 0.05 | FAIL |
| Soft migration score | 3.79 | 0.5 | PASS |

Blocking dying neurons' forward-pass contributions (activation zeroing) did not
impair surviving neurons' feature acquisition. The mechanism is relearning from
data, not migration through the computation graph.

**Post-hoc discovery:** The two-head architecture severs the gradient pathway
between neuron groups. Each head's W2 has zeros in the columns corresponding to the
other expert's neurons, so no gradient flows between groups. The migration test was
architecturally prevented from succeeding. This motivated the single-task experiment.

### 3.2 Compression Comparison (Phase 2 Fallback)

| Method | Add Acc | Mul Acc | Width |
|--------|---------|---------|-------|
| Asymmetric decay | 99.97% | 97.35% | 256 (half dead) |
| Distillation (T=4) | 99.96% | 99.97% | 128 |
| Structured GMP | 99.97% | 100.0% | 131 |

Asymmetric decay is not competitive. Both baselines achieve higher accuracy at
half the parameter count.

### 3.3 Single-Task Densification (Experiment A)

**Setup:** Take a fully grokked addition expert (W=128, 100% accuracy, ~93 Fourier-
sparse neurons). Apply asymmetric decay with a single shared output head, providing
maximum gradient interaction between all neurons. 7 conditions, 5 grokked
checkpoints (different seeds) for between-run variance.

| Condition | Accuracy | Surviving Neurons | Pareto AUC |
|-----------|----------|-------------------|------------|
| Asymmetric (alpha=5) | 97.9% | 128 | 12.2 |
| Uniform decay | 100.0% | 128 | 10.6 |
| Reversed asymmetric | 97.7% | 128 | 6.3 |
| **Gradient-severed** | **98.9%** | **116.6** | **18.3** |
| Structured GMP | 100.0% | 128 | 10.6 |
| Instant prune + retrain | 100.0% | 128 | 10.6 |
| Train from scratch | 100.0% | 128 | 36.3 |

**Finding 1: Asymmetric decay kills zero neurons.** Despite a per-step decay rate
of 6e-4 on the heaviest neuron over 20k steps (theoretical multiplicative factor
of ~6e-6), all 128 neurons survive. The optimizer's gradient signal through the
shared W2 keeps every neuron alive by pushing norms back up as fast as decay
pushes them down.

**Finding 2: Severing the gradient improves compaction.** The gradient-severed
condition — which detaches dying neurons' activations from the backward pass so no
gradient flows through their W2 columns — is the only condition that actually
reduces neuron count (116.6 mean surviving) and achieves the best Pareto AUC
(18.3 vs 12.2 for standard asymmetric, paired t-test p=0.028).

**Finding 3: The shared W2 acts as life support, not a teaching channel.** The
gradient pathway through W2 that we hypothesized would enable knowledge transfer
instead prevents the neuron death that transfer requires. The optimizer treats
every neuron as worth preserving because each contributes marginally to the loss
through W2, even when asymmetric decay is trying to kill it.

## 4. Discussion

### The Life Support Mechanism

When neuron j has nonzero W2 columns, it contributes to the output. Even a small
contribution means the loss depends on neuron j, which means the gradient updates
neuron j's W1 weights to maintain that contribution. Asymmetric decay pushes W1[j]
toward zero, but the gradient pushes it back proportionally to neuron j's usefulness
to the loss. In a grokked network where every Fourier-sparse neuron encodes a unique
frequency, every neuron IS useful — so the gradient always wins.

Severing the gradient (detaching dying neurons from the backward pass) breaks this
equilibrium. The dying neurons still contribute to the forward pass (preserving the
smooth loss landscape), but they receive no gradient to counteract the decay, so
they actually die. Surviving neurons then receive larger gradients (because the
output increasingly depends on them) and consolidate features.

This suggests that **stop-gradient operators between neuron groups** may be a more
principled approach to structured pruning than differential regularization. Rather
than hoping the optimizer will "let go" of dying neurons, explicitly remove their
gradient access while maintaining their forward-pass contribution for a smooth
training signal.

### Connection to Existing Work

Our activation-zeroing causal test provides the first direct evidence that feature
transfer between neurons does not occur during regularization-driven pruning. This
complements the "It's not a Lottery, it's a Race" finding (2026) that neurons
independently race to learn features rather than transferring them.

The gradient-severing result connects to Gradient Routing (Cloud et al. 2024),
which applies data-dependent gradient masks to localize capabilities. Our
contribution is the pruning application: gradient routing can enhance structured
compression by preventing the life-support effect.

The difficulty of compressing grokked models aligns with findings that grokking
produces holographic representations distributed across full-rank weights (Geometry
of Multi-Task Grokking, 2026), which are inherently resistant to structured pruning.

### Limitations

- Tested only on modular arithmetic with 2-layer MLPs and quadratic activation
- Fourier interpretability limited to addition (multiplication lacks clean structure)
- Full-batch training may interact differently with decay than minibatch
- The gradient-severed improvement, while statistically significant (p=0.028),
  is modest in absolute terms

### 3.4 Stop-Gradient-Assisted Pruning (Experiment B)

**Setup:** Systematically test sever-and-decay as a pruning method. Detach dying
neurons from the backward pass (they still contribute to the forward pass), apply
binary decay only to severed neurons. 6 sever fractions (12.5-75%), 2 schedules,
4 baselines at 3 fractions. 10 seeds, 200 unique runs.

**Result: Sever-and-decay loses to simple pruning.**

| Method (at 50% compression) | Pareto AUC | p vs sever |
|------------------------------|-----------|------------|
| Prune + retrain | **35.2** | **0.011** |
| Scratch W=64 | 30.1 | 0.199 |
| Sever-and-decay (ramp) | 26.4 | 0.111 |
| Sever-and-decay (instant) | 21.3 | — |
| Asymmetric decay | 21.4 | 0.993 |
| Distillation W=64 | 10.1 | 0.095 |

Simple magnitude pruning + retraining significantly outperforms stop-gradient-
assisted pruning (p=0.011). Severing the gradient does not improve over standard
asymmetric decay (p=0.99). The forward-pass contribution of dying neurons does
not provide useful smoothing for the surviving neurons.

The Experiment A finding that gradient-severing improved Pareto efficiency
(p=0.028) does not replicate in this broader, properly controlled comparison.
The earlier result was likely confounded by comparing against asymmetric decay
(which itself performs poorly) rather than against the right baseline (prune +
retrain).

## 5. Conclusion

Asymmetric weight decay creates controllable, directional norm reduction in neural
networks but does not cause knowledge migration. Across four experimental
configurations — multi-task two-head, multi-task shared-head, single-task
densification, and stop-gradient-assisted pruning — the method fails to outperform
simple magnitude pruning followed by retraining.

The shared output pathway hypothesized to enable migration acts as a gradient-
mediated life support system that prevents neuron death. Severing this gradient
does enable neuron death but does not improve compression quality — the surviving
neurons learn from data regardless.

The strongest baseline is the simplest: prune the lowest-norm neurons, retrain for
the same number of steps. No gradient manipulation, decay scheduling, or forward-
pass smoothing improves on this. The optimizer's ability to recover from pruning
via standard gradient descent on the task data is sufficient.

## Reproducibility

All code is in this repository. Experiments run on a single NVIDIA RTX 5090 (32GB).

```bash
pip install -r requirements.txt

# Step 1: Train grokking experts (20 models, ~13 min)
python experiments/step1_sanity.py

# Step 2: Asymmetric decay conditions + gate evaluation (~10 min)
python experiments/step2_asymmetric.py

# Step 3: Migration causal test (~10 min)
python experiments/step3_migration.py

# Phase 2 fallback: compression comparison (~15 min)
python experiments/phase2_fallback.py

# Experiment A: single-task densification (~20 min)
python experiments/experiment_a.py

# Experiment B: stop-gradient pruning sweep (~100 min)
python experiments/experiment_b.py

# Run tests
python -m pytest tests/ -v
```

Total compute: ~170 minutes on RTX 5090.

## References

- Nanda et al. (2023). Progress measures for grokking via mechanistic interpretability. ICLR.
- Cloud et al. (2024). Gradient Routing: Masking Gradients to Localize Computation. arXiv:2410.04332.
- Zhang et al. (2018). Three Mechanisms of Weight Decay Regularization. arXiv:1810.12281.
- Zhu & Gupta (2017). To Prune, or Not to Prune: Exploring the Efficacy of Pruning for Model Compression.
- Li et al. (2017). Pruning Filters for Efficient ConvNets.
- Tessier et al. (2022). Rethinking Weight Decay for Efficient Neural Network Pruning. JMLR.
- Yadav et al. (2023). TIES-Merging: Resolving Interference When Merging Models. NeurIPS.
