# Experimental Protocol: Asymmetric Weight Decay — Fail-Fast Version

## Meta

- **Hardware**: Single NVIDIA RTX 5090 (32GB VRAM)
- **Framework**: PyTorch >= 2.1
- **Seeds**: 4 seeds (42, 137, 256, 512) for Phase 1. Scale to 10 only if Phase 1 passes all gates.
- **Logging**: W&B or TensorBoard. Log metrics every 100 steps.
- **Code**: Shared config dataclass. All hyperparameters via config.
- **AdamW**: Set weight_decay=0 everywhere. All decay applied externally via AsymmetricDecay class. This avoids the double-decay problem where AdamW's uniform decay drowns out the asymmetric component.
- **Data split**: 50% train / 50% test of all p² pairs, generated once with fixed seed=0, shared across all runs.
- **p = 113** (following Nanda et al. 2023)

### Related Positioning

This work differs from model merging methods (TIES-Merging, DARE, model soups) in a fundamental way: merging methods combine models by averaging or selecting parameters within a shared architecture. Our approach concatenates models to create a wider architecture, then compacts. This preserves both experts' full representations initially (no information loss from averaging) at the cost of a larger intermediate model.

The hypothesis is that the intermediate wider model provides a search space in which asymmetric decay can find a compact representation that retains both experts' capabilities — something parameter averaging cannot guarantee when experts have learned structurally different features (e.g., different Fourier modes for different arithmetic operations).

Key references to cite and differentiate from:
- Li et al. (2017) — structured filter pruning via magnitude
- Liu et al. (2017) — network slimming via BN sparsity penalties
- Wortsman et al. (2022) — model soups
- Yadav et al. (2023) — TIES-Merging
- Dai et al. (2019) — grow and prune
- Evci et al. (2020) — RigL / sparse-to-sparse training
- Howard & Ruder (2018) — ULMFiT discriminative fine-tuning (different rates per layer, not per neuron — reviewers will conflate)
- Nanda et al. (2023) — grokking progress measures / Fourier vocabulary

---

## Phase 1: Kill Gates (~4 days)

The goal is to falsify the migration hypothesis as cheaply as possible. Sequential gates; failure at any gate terminates the line of work.

### Wall-Clock Estimate

- **Day 1**: Implement training loop, data generation, AsymmetricDecay, metrics pipeline, concatenation logic. Unit test each component.
- **Day 2**: Run Step 1 (sanity check). Debug grokking if needed. Verify Fourier structure. Build and verify concatenation.
- **Day 3**: Run Step 2 (three conditions). Evaluate Gates 1, 2, 2.5. Run variance calibration.
- **Day 4**: Run Step 3 (activation-zeroed + freeze conditions). Evaluate Gate 3. Make go/no-go decision.

### Step 1 — Sanity Check (~3 hours)

Train MLP_add (W=128) and MLP_mul (W=128) on modular addition and multiplication, 4 seeds each. AdamW lr=1e-3, uniform external decay lambda=1.0. Train until 100% test accuracy for 1000 consecutive steps or 50k steps.

**Grokking failure contingency:** If multiplication does not grokk within 50k steps, reduce external decay to lambda ∈ {0.3, 0.1} and retry. If it still fails at lambda=0.1, increase width to W=256 and retry at lambda=1.0. Document all attempts. This is a known-good setup; persistent failure indicates implementation bugs — do not proceed until resolved.

Verify:
- [ ] Both models grok to 100% test accuracy
- [ ] Fourier structure visible (>80% neurons with IPR > 0.5)

If this fails, debug implementation. Do not proceed.

Concatenate into MLP_cat (W=256). Select the checkpoint with the lowest test loss at final step for each task. Document the selection criterion and the discarded seeds' final metrics. Verify >95% on both tasks (calibrate mixing weights if needed).

Train with uniform decay (lambda=1.0) on joint add+mul data until train loss variance over the last 500 steps < 1e-4, or 10,000 steps, whichever comes first → MLP_converged. Verify >98% test accuracy on both tasks.

### Step 2 — Does Anything Happen? (~5 hours)

From MLP_converged, run three conditions, 4 seeds each:

| Condition | Description |
|-----------|-------------|
| Asymmetric | alpha=2.0, lambda_base=1.0, low decay on left |
| Uniform | alpha=0 (control) |
| Reversed | alpha=2.0, high decay on left |

Train 20,000 steps on joint add+mul data.

**Look at the column norm heatmap.** This takes minutes to evaluate visually.

- [ ] **Gate 1**: Does the asymmetric condition show decreasing norms on the high-decay side and preserved/increasing norms on the low-decay side?

If the heatmaps look the same across all three conditions, the decay gradient isn't doing anything. Stop.

- [ ] **Gate 2 (Direction Test)**: Is the asymmetric heatmap a mirror image of the reversed heatmap?

If asymmetric and reversed produce the same compaction pattern (both kill high-decay neurons regardless of which side that is), the gradient determines which neurons die but not where knowledge goes. The thesis reduces to "position-based pruning," which is a thin contribution. Consider reframing or stopping.

- [ ] **Gate 2.5 (Compaction Quality)**: After Step 2's asymmetric condition, zero columns j >= 128 and evaluate both tasks. At least one task retains >80% accuracy.

If both tasks are below 80%, compaction is destroying useful features regardless of mechanism. The decay gradient is too aggressive at alpha=2.0. Try alpha=1.0 before stopping. If alpha=1.0 also fails this gate, stop.

This adds no compute — it's a post-hoc evaluation of Step 2 results.

### Step 2.5 — Variance Calibration (no additional runs)

From Step 2's asymmetric condition (4 seeds), compute:
- `std_lhef`: between-seed standard deviation of left-half energy fraction
- `std_sms`: between-seed standard deviation of soft migration score

These provide the standard error estimates needed for Gate 3's ambiguity zones.

### Step 3 — Migration or Relearning? (~5 hours)

This is the critical experiment. If Gates 1, 2, and 2.5 pass, run three new conditions from MLP_converged, 4 seeds each:

| Condition | Dying neurons' activations | Dying neurons learn? | Source |
|-----------|---------------------------|---------------------|--------|
| Asymmetric (from Step 2) | Active | Yes (but decaying) | Already ran |
| Asymmetric + activation-zeroed | Zeroed | No (zero grad + decay) | **New** |
| Asymmetric + weight-frozen | Active (frozen at MLP_converged values) | No (weights restored each step) | **New** |
| Uniform (from Step 2) | Active | Yes | Already ran |
| Uniform + activation-zeroed | Zeroed | No (zero grad) | **New** |

Only 3 new conditions × 4 seeds = 12 new runs.

**The logic:** Activation-zeroing blocks the information channel from dying neurons to the loss. If dying neurons carry useful information through their activations (migration), removing those activations should hurt surviving neurons' feature acquisition. The uniform + activation-zeroed condition calibrates for the capacity loss itself — zeroing 25% of neurons changes the loss landscape regardless of mechanism.

The migration signal is the interaction term:

```
migration_effect = (asymmetric_with - asymmetric_without) 
                 - (uniform_with - uniform_without)
```

Note: both the asymmetric and uniform "with vs without activations" gaps may be large. The migration signal is the interaction term (difference of differences), not either gap alone. A large gap in the asymmetric condition does not by itself indicate migration — the uniform gap calibrates for the capacity loss.

**Confound acknowledgment:** The activation-zeroing test conflates "information transfer from dying neurons" with "asymmetric gradient structure helps learning when capacity is removed." The weight-frozen condition disambiguates: if surviving neurons acquire features equally well whether dying neurons are decaying (standard asymmetric) or frozen (activations present but static), then the dying neurons' *changing* activations during decay are not carrying information — the surviving neurons learn from data. If they do worse when dying neurons are frozen vs decaying, the decay trajectory itself contributes signal beyond what static activations provide.

**Gate 3 — Pre-specified thresholds with ambiguity zones:**

Compute the standard error (SE) of the migration_effect from the 4 seeds.

| Zone | Criterion | Action |
|------|-----------|--------|
| **Clear pass** | All three metrics exceed threshold by > 1 SE | Proceed to full protocol |
| **Ambiguous** | Any metric within ± 1 SE of threshold | Escalate to 10 seeds, then re-evaluate as pass/fail |
| **Clear fail** | Any metric below threshold by > 1 SE | Gate fails |

Thresholds:
- **Gate 3a**: migration_effect on left-half energy fraction > 0.05
- **Gate 3b**: migration_effect on soft migration score > 0.5 (at least half a neuron index of additional leftward shift)
- **Gate 3c**: migration_effect on accuracy-after-zeroing-columns-j≥128 > 5% on at least one task

If the gate fails: migration is ruled out as a mechanism. The result is "position-based structured pruning." Run Phase 2 fallback.

If the gate passes: dying neurons' activations carry information that specifically helps surviving neurons under asymmetric decay more than under uniform decay. Migration has a plausible mechanism. Proceed to full protocol.

### Phase 1 Metrics

Compute these for all conditions. They're cheap and you need them for the gates.

- Column norm profile: ‖W₁[j, :]‖₂ per neuron, as heatmap over (step, neuron_index)
- Soft migration score: energy-weighted mean neuron index per frequency, before vs. after
- Left-half Fourier energy fraction
- Per-task accuracy (add, mul) at end of training
- Accuracy after zeroing columns j ≥ T for T ∈ {96, 128, 160}
- For activation-zeroed conditions only: weight norms of zeroed neurons (j >= 192) over training. Verify monotonic decrease as sanity check.

Do NOT compute CKA, phase continuity, weight cosine similarity, or effective dimensionality in Phase 1. These are characterization metrics, not kill-gate metrics.

### Phase 1 Decision Matrix

| Gate 1 | Gate 2 | Gate 2.5 | Gate 3 | Interpretation | Action |
|--------|--------|----------|--------|----------------|--------|
| Fail | — | — | — | Decay gradient has no effect on norm distribution | Stop |
| Pass | Fail | — | — | Position determines which neurons die, not where knowledge goes | Reframe as position-based pruning. Run Phase 2 fallback |
| Pass | Pass | Fail | — | Compaction destroys useful features at alpha=2.0 and 1.0 | Stop |
| Pass | Pass | Pass | Clear fail | Compaction works, mechanism is relearning under constraint | Reframe as structured soft pruning. Run Phase 2 fallback |
| Pass | Pass | Pass | Ambiguous | Insufficient power to determine mechanism | Escalate to 10 seeds. Re-evaluate as pass/fail |
| Pass | Pass | Pass | Clear pass | Migration has a plausible mechanism | Run full protocol (protocol_v2.md) |

### Phase 1 Compute Budget

| Step | Runs | Est. Time/Run | Total |
|------|------|---------------|-------|
| Step 1: Training experts + concat + converge | 8 + 4 | 10–15 min | ~3 hr |
| Step 2: Three conditions × 4 seeds | 12 | 25 min | ~5 hr |
| Step 3: Three new conditions × 4 seeds | 12 | 25 min | ~5 hr |
| **Total** | **~36** | | **~13 hr GPU** |

---

## Phase 2: Conditional on Phase 1 Outcome

### If all gates pass → Full Protocol

Run protocol_v2.md as written, with these additions:
- **Control F — Distillation** in Experiment 1E: W=128 student model, randomly initialized, soft targets from MLP_cat (W=256) teacher on joint add+mul data. Temperature ∈ {1, 2, 4}, select best on validation. Train for 20,000 steps (matched to 1E conditions). 10 seeds. Compare at matched parameter count against asymmetric-decay compacted model.
- Report wall-clock training time alongside accuracy for all conditions.

### If Gate 3 fails → Fallback: Is Position-Based Pruning Useful?

Migration is dead, but maybe position-based structured pruning during training is still a useful compression technique. Run a minimal comparison:

| Condition | Description |
|-----------|-------------|
| Asymmetric decay | Best result from Phase 1 |
| Distillation | W=128 student, soft targets from MLP_cat teacher, temperature ∈ {1, 2, 4} |
| Structured GMP | Structured gradual magnitude pruning on MLP_converged to same final width. Prune entire neuron columns (not individual weights) ranked by column L2 norm. Cubic schedule (Zhu & Gupta 2017): initial sparsity 0%, final sparsity matched to asymmetric decay's Pareto-optimal neuron count, ramp over first 15,000 of 20,000 steps, pruning mask updated every 100 steps. |

5 seeds each. Compare accuracy on both tasks at matched parameter count. Report wall-clock training time.

If asymmetric decay loses to distillation and structured GMP on both tasks: stop entirely. Publish negative result with Phase 1 interpretability data if it's interesting.

If asymmetric decay is competitive or better on at least one metric: publishable as a compression method. Run characterization experiments from protocol_v2.md (minus migration-specific metrics).

### Phase 2 Fallback Compute Budget

| Condition | Runs | Est. Time/Run | Total |
|-----------|------|---------------|-------|
| Distillation (3 temps × 5 seeds) | 15 | 20 min | ~5 hr |
| Structured GMP (5 seeds) | 5 | 25 min | ~2 hr |
| **Total** | **20** | | **~7 hr GPU** |

---

## Appendix: Implementation for Phase 1

All code from protocol_v2.md appendix applies. Key additions and corrections below.

### A.1 AsymmetricDecay (with shape assertions)

```python
class AsymmetricDecay:
    def __init__(self, lambda_base: float, alpha: float,
                 reverse: bool = False, apply_to: str = 'both'):
        self.lambda_base = lambda_base
        self.alpha = alpha
        self.reverse = reverse
        self.apply_to = apply_to

    def compute_decay_mask(self, size: int) -> torch.Tensor:
        positions = torch.linspace(0, 1, size)
        if self.reverse:
            positions = 1.0 - positions
        return self.lambda_base * (1.0 + self.alpha * positions)

    def step(self, W1: torch.Tensor, W2: torch.Tensor, lr: float):
        """Call after optimizer.step(). Decays in-place."""
        W = W1.shape[0]  # hidden dim
        assert W == W2.shape[1], (
            f"Hidden dim mismatch: W1.shape[0]={W1.shape[0]}, "
            f"W2.shape[1]={W2.shape[1]}. Check weight matrix orientation. "
            f"Expected W1=(hidden, input), W2=(output, hidden)."
        )
        mask = self.compute_decay_mask(W).to(W1.device)

        if self.apply_to in ('both', 'W1'):
            W1.data.mul_(1.0 - lr * mask.unsqueeze(1))

        if self.apply_to in ('both', 'W2'):
            W2.data.mul_(1.0 - lr * mask.unsqueeze(0))
```

If using `nn.Linear`, access weights as `layer1.weight` (shape: out_features, in_features). For a first-layer `nn.Linear(input_dim, hidden_dim)`, `weight` is `(hidden_dim, input_dim)` — this is W1 with hidden dim on axis 0. For a second-layer `nn.Linear(hidden_dim, output_dim)`, `weight` is `(output_dim, hidden_dim)` — this is W2 with hidden dim on axis 1. Verify shapes in Step 1 by printing before any training.

### A.2 Activation-Zeroing Hook

```python
def register_activation_zeroing_hook(model, W: int, quartile: float = 0.75):
    """
    Clamp activations to zero for dying neurons (top quartile).
    These neurons still have weights and receive decay, but receive
    zero gradients (loss has no dependence on zeroed activations).
    Their weights can only shrink toward zero. This blocks the
    information channel from dying neurons to surviving neurons
    via the loss.

    Use as a forward hook on the hidden layer activation.
    """
    cutoff = int(W * quartile)

    def zero_activations(module, input, output):
        output[:, cutoff:] = 0.0
        return output

    # Attach to the activation layer (after W1 @ x + b1, after squaring)
    model.hidden_activation.register_forward_hook(zero_activations)
```

### A.3 Weight-Freeze Hook

```python
def register_weight_freeze_hook(model, W: int, quartile: float = 0.75):
    """
    Freeze weights for dying neurons (top quartile). These neurons
    participate in the forward pass with their initial weights but
    cannot learn or decay — weights are restored after each step.
    Their activations still flow through the network normally.

    Returns a callable: call restore_weights() after optimizer.step()
    and decay.step() each training step.
    """
    cutoff = int(W * quartile)

    # Snapshot the initial weights at registration time
    W1_frozen = model.W1.data[cutoff:].clone()
    W2_frozen = model.W2.data[:, cutoff:].clone()
    b1_frozen = model.b1.data[cutoff:].clone()

    def restore_weights():
        model.W1.data[cutoff:] = W1_frozen
        model.W2.data[:, cutoff:] = W2_frozen
        model.b1.data[cutoff:] = b1_frozen

    return restore_weights  # call after optimizer.step() and decay.step()
```

### A.4 Key Metrics

```python
# Soft migration score (preferred over hard argmax version)
def soft_migration_score(energy_before, energy_after, p):
    W = energy_before.shape[0]
    indices = torch.arange(W, dtype=torch.float32)
    shifts = []
    for f in range(1, p):
        eb, ea = energy_before[:, f], energy_after[:, f]
        if eb.sum() < 1e-8 or ea.sum() < 1e-8:
            continue
        mu_b = (indices * eb).sum() / eb.sum()
        mu_a = (indices * ea).sum() / ea.sum()
        shifts.append((mu_b - mu_a).item())
    return float(np.mean(shifts)), float(np.std(shifts))

# IPR: 1.0 = single frequency, 1/(p-1) = uniform
def ipr(spectrum):
    s = spectrum[1:]
    s = s / s.sum()
    return (s ** 2).sum().item()

# Left-half energy fraction
def left_energy(W1, p):
    W_h = W1.shape[0]
    half = W_h // 2
    total, left = 0.0, 0.0
    for j in range(W_h):
        w = W1[j].reshape(2, p).float()
        power = (torch.fft.fft(w, dim=-1).abs() ** 2).sum(0)
        e = power[1:].sum().item()
        total += e
        if j < half:
            left += e
    return left / total if total > 0 else 0.5
```
