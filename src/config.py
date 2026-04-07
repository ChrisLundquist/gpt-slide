"""
Shared configuration for all experiments.
All hyperparameters via this dataclass — never hardcoded inline.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Config:
    # Data
    p: int = 113
    train_frac: float = 0.5
    data_seed: int = 0

    # Model
    hidden_dim: int = 128
    # input_dim = 2*p + 2 (one-hot a, one-hot b, 2-bit task token)
    # output_dim = p

    # Training
    lr: float = 1e-3
    weight_decay: float = 0.0  # MUST be 0; all decay is external
    max_steps: int = 50_000
    log_every: int = 100

    # Grokking detection
    grok_window: int = 1000  # consecutive steps at 100% test acc

    # Convergence (for MLP_converged)
    converge_max_steps: int = 10_000
    converge_var_window: int = 500
    converge_var_threshold: float = 1e-4

    # Decay
    lambda_base: float = 0.1  # tuned for 2-layer MLP; 1.0 is too aggressive
    alpha: float = 0.0  # 0 = uniform, 5.0 = asymmetric (tuned for 2-layer MLP)
    reverse: bool = False
    apply_to: str = 'both'  # 'both', 'W1', 'W2'

    # Step 3 hooks
    activation_zeroed: bool = False
    weight_frozen: bool = False
    quartile: float = 0.75  # top 25% = dying neurons

    # Experiment
    seed: int = 42
    task: str = 'add'  # 'add', 'mul', 'joint'
    condition: str = 'expert'
    device: str = 'cuda'

    # Gate thresholds
    gate2_5_acc_threshold: float = 0.80
    gate3a_lhef_threshold: float = 0.05
    gate3b_sms_threshold: float = 0.5
    gate3c_acc_threshold: float = 0.05

    # Paths
    output_dir: str = 'outputs'

    @property
    def input_dim(self) -> int:
        return 2 * self.p

    @property
    def output_dim(self) -> int:
        return self.p

    @property
    def run_name(self) -> str:
        parts = [self.condition]
        if self.task != 'joint':
            parts.append(self.task)
        parts.append(f'seed{self.seed}')
        return '_'.join(parts)


SEEDS = (42, 137, 256, 512, 1024, 1337, 2048, 3141, 4096, 8192)

# Phase 1 uses 4 seeds; scale to 10 only if all gates pass
PHASE1_SEEDS = SEEDS[:4]

# Condition presets for Step 2
STEP2_CONDITIONS = {
    'asymmetric': {'alpha': 5.0, 'reverse': False, 'task': 'joint',
                   'max_steps': 20_000, 'condition': 'asymmetric'},
    'uniform':    {'alpha': 0.0, 'reverse': False, 'task': 'joint',
                   'max_steps': 20_000, 'condition': 'uniform'},
    'reversed':   {'alpha': 5.0, 'reverse': True, 'task': 'joint',
                   'max_steps': 20_000, 'condition': 'reversed'},
}

# Condition presets for Step 3
STEP3_CONDITIONS = {
    'asym_actzeroed':    {'alpha': 5.0, 'reverse': False, 'task': 'joint',
                          'max_steps': 20_000, 'condition': 'asym_actzeroed',
                          'activation_zeroed': True},
    'asym_frozen':       {'alpha': 5.0, 'reverse': False, 'task': 'joint',
                          'max_steps': 20_000, 'condition': 'asym_frozen',
                          'weight_frozen': True},
    'uniform_actzeroed': {'alpha': 0.0, 'reverse': False, 'task': 'joint',
                          'max_steps': 20_000, 'condition': 'uniform_actzeroed',
                          'activation_zeroed': True},
}

# Lambda fallback schedule for grokking failure
LAMBDA_FALLBACK = [0.1]
