# Review Feedback Log

All amendments from the three independent reviews have been integrated into
`protocol_failfast.md`. This file documents what changed and why.

## Changes from Original Protocol

| # | Change | Reviewer | Rationale |
|---|--------|----------|-----------|
| 1 | Seeds 2 → 4 for Phase 1 | All three | 2 seeds = 1 degree of freedom for variance. Can't distinguish noise from signal. |
| 2 | Added weight-freeze condition to Step 3 | Design reviewer | Activation-zeroing conflates "info transfer" with "asymmetric gradients help under capacity loss." Freeze disambiguates. |
| 3 | Gate 3 ambiguity zones (clear pass / ambiguous / clear fail) | Practical reviewer | Binary threshold at 5% with 4 seeds invites judgment calls the protocol tries to avoid. |
| 4 | Shape assertions in AsymmetricDecay | Practical reviewer | nn.Linear transposes weight shape. Silent axis bug invalidates all results. |
| 5 | Defined "best-seed pair" = lowest test loss at final step | Practical reviewer | Original was ambiguous. Selecting on grokking speed biases circuit structure. |
| 6 | Grokking failure contingency (lambda fallback) | Practical reviewer | Multiplication grokks later than addition. lambda=1.0 may be too aggressive at W=128. |
| 7 | Structured GMP (not unstructured) in fallback | Initial review | Asymmetric decay kills whole neurons. Comparing against individual-weight pruning is invalid. |
| 8 | Distillation control added to full protocol, not just fallback | Claims reviewer | Reviewers will demand it. Waiting for the fallback path to discover this wastes time. |
| 9 | Related positioning against model merging literature | Claims reviewer | TIES-Merging, model soups are direct competitors to the concatenation step. |
| 10 | Wall-clock estimate 2 days → 4 days | Practical reviewer | Original counted GPU time only. Code development + debugging is at least 2 days. |
| 11 | Variance calibration step between Step 2 and Step 3 | Initial review | Gate 3 thresholds need empirical anchor. SNR < 2 → escalate seed count. |
| 12 | Gate 2.5 (compaction quality) | Initial review | Gates 1-3 test mechanism, not utility. Catches useless compaction early. |
| 13 | Activation-zeroing docstring corrected | Initial review | Zeroed neurons get zero gradients by chain rule, not nonzero gradients. |
| 14 | Convergence criterion explicit (train loss variance < 1e-4) | Initial review | Original said "until converged" without defining convergence. |
| 15 | Report wall-clock time in fallback comparisons | Claims reviewer | If asymmetric decay is slower than distillation at same compression, practical case collapses. |

## Unresolved Concerns

- **Gate 3's activation-zeroing still has a residual confound:** even with the freeze condition, the comparison mixes "information channel" with "optimization dynamics under capacity loss." The freeze condition helps but doesn't fully isolate the mechanism. Acknowledged in protocol, not resolved.
- **Modular arithmetic is maximally favorable:** Fourier features are discrete, canonical, and independent. Migration may look identical to relearning in this setting because there's only one way to represent each frequency. GELU ablation in the full protocol partially addresses this.
- **If only compression survives, the venue drops:** full migration result → ICLR/NeurIPS interpretability. Compression-only → workshops (NeurIPS Efficient ML, WANT) unless GPT-2 results are strong.
