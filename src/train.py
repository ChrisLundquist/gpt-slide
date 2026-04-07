"""
Core training loop for all experiment steps.

Adapted from grokking-svd/train.py patterns:
- set_seed(), evaluate(), check_grok()
- Full-batch training with external decay
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import asdict

from .model import GrokMLP
from .decay import AsymmetricDecay
from .hooks import register_activation_zeroing_hook, register_weight_freeze_hook
from .metrics import (
    compute_column_norms, compute_left_energy, frequency_energy_matrix,
    soft_migration_score, compute_ipr, neuron_fourier_spectrum,
    accuracy_after_zeroing,
)


def set_seed(seed: int):
    """Deterministic seeding. From grokking-svd/train.py."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, X, y, task=None):
    """Full-batch evaluation. Returns (loss, accuracy)."""
    model.eval()
    with torch.no_grad():
        logits = model(X, task=task) if task else model(X)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(-1) == y).float().mean()
    return loss.item(), acc.item()


def check_grok(acc_history: list[float], threshold: float = 1.0,
               stability: int = 1000) -> int | None:
    """
    First step where accuracy >= threshold and stays for `stability` consecutive steps.
    Returns None if not met. Adapted from grokking-svd/train.py.
    """
    run_length = 0
    for i, acc in enumerate(acc_history):
        if acc >= threshold:
            run_length += 1
            if run_length >= stability:
                return i - stability + 1
        else:
            run_length = 0
    return None


def train_run(config, starting_model=None, starting_state=None):
    """
    Execute one training run parameterized by config.

    Args:
        config: Config dataclass
        starting_model: optional pre-built GrokMLP (for Steps 2-3)
        starting_state: optional state_dict to load into model

    Returns:
        dict with final metrics, model state, and full metrics log
    """
    from .data import build_dataset, build_joint_dataset

    set_seed(config.seed)
    device = config.device

    # Build or load model
    if starting_model is not None:
        model = starting_model.to(device)
    elif starting_state is not None:
        model = GrokMLP(config.input_dim, config.hidden_dim, config.output_dim).to(device)
        model.load_state_dict(starting_state)
    else:
        model = GrokMLP(config.input_dim, config.hidden_dim, config.output_dim).to(device)

    # Keep reference to the raw module for weight access (compile wraps it)
    raw_model = model

    # Compile model if no hooks will be attached (hooks + compile can conflict)
    use_compile = (not config.activation_zeroed and not config.weight_frozen
                   and hasattr(torch, 'compile'))
    if use_compile:
        model = torch.compile(model)

    # Optimizer — weight_decay MUST be 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)

    # External decay
    decay = AsymmetricDecay(config.lambda_base, config.alpha,
                            config.reverse, config.apply_to)

    # Data
    if config.task == 'joint':
        data = build_joint_dataset(config.p, config.train_frac,
                                   config.data_seed, device)
        add_train_X, add_train_y = data['add'][:2]
        add_test_X, add_test_y = data['add'][2:]
        mul_train_X, mul_train_y = data['mul'][:2]
        mul_test_X, mul_test_y = data['mul'][2:]
    else:
        train_X, train_y, test_X, test_y = build_dataset(
            config.p, config.train_frac, config.data_seed, device, config.task)

    # Hooks (Step 3)
    hook_handle = None
    restore_fn = None
    if config.activation_zeroed:
        hook_handle = register_activation_zeroing_hook(
            raw_model, raw_model.hidden_dim, config.quartile)
    if config.weight_frozen:
        restore_fn = register_weight_freeze_hook(
            raw_model, raw_model.hidden_dim, config.quartile)

    # Capture initial energy matrix for migration score computation
    initial_energy = frequency_energy_matrix(raw_model.W1.data, config.p)

    # Training
    metrics_log = []
    test_acc_history = []
    recent_losses = []  # for convergence detection

    for step in range(config.max_steps):
        model.train()
        optimizer.zero_grad()

        if config.task == 'joint':
            logits_add = model(add_train_X, task='add')
            loss_add = F.cross_entropy(logits_add, add_train_y)
            logits_mul = model(mul_train_X, task='mul')
            loss_mul = F.cross_entropy(logits_mul, mul_train_y)
            loss = (loss_add + loss_mul) / 2
        else:
            logits = model(train_X)
            loss = F.cross_entropy(logits, train_y)

        loss.backward()
        optimizer.step()

        # External decay — apply to all W2 heads
        if raw_model._joint:
            W2s = [raw_model.head_add.weight, raw_model.head_mul.weight]
        else:
            W2s = raw_model.W2
        decay.step(raw_model.W1, W2s, config.lr)

        # Weight freeze restore
        if restore_fn:
            restore_fn()

        # Logging
        if step % config.log_every == 0:
            model.eval()
            m = {'step': step, 'train_loss': loss.item()}

            if config.task == 'joint':
                _, acc_add = evaluate(model, add_test_X, add_test_y, task='add')
                _, acc_mul = evaluate(model, mul_test_X, mul_test_y, task='mul')
                m['test_acc_add'] = acc_add
                m['test_acc_mul'] = acc_mul

                _, train_acc_add = evaluate(model, add_train_X, add_train_y, task='add')
                _, train_acc_mul = evaluate(model, mul_train_X, mul_train_y, task='mul')
                m['train_acc_add'] = train_acc_add
                m['train_acc_mul'] = train_acc_mul
            else:
                test_loss, test_acc = evaluate(model, test_X, test_y)
                _, train_acc = evaluate(model, train_X, train_y)
                m['test_loss'] = test_loss
                m['test_acc'] = test_acc
                m['train_acc'] = train_acc
                test_acc_history.append(test_acc)

            # Column norms
            norms = compute_column_norms(raw_model.W1.data).cpu().tolist()
            m['column_norms'] = norms

            # Left-half energy
            m['left_energy'] = compute_left_energy(raw_model.W1.data, config.p)

            # Soft migration score (vs initial state)
            current_energy = frequency_energy_matrix(raw_model.W1.data, config.p)
            sms_mean, sms_std = soft_migration_score(
                initial_energy, current_energy, config.p)
            m['soft_migration_score'] = sms_mean
            m['soft_migration_std'] = sms_std

            metrics_log.append(m)

        # Track recent losses for convergence detection
        recent_losses.append(loss.item())
        if len(recent_losses) > config.converge_var_window:
            recent_losses.pop(0)

        # Grokking early stop (single-task only)
        if config.task != 'joint' and len(test_acc_history) > 0:
            grok_step = check_grok(test_acc_history, 1.0,
                                   config.grok_window // config.log_every)
            if grok_step is not None:
                break

        # Convergence early stop (joint training, for MLP_converged)
        if (config.condition == 'converge'
                and len(recent_losses) >= config.converge_var_window):
            var = np.var(recent_losses[-config.converge_var_window:])
            if var < config.converge_var_threshold:
                break

    # Cleanup hooks
    if hook_handle is not None:
        hook_handle.remove()

    # Final energy matrix for migration analysis
    final_energy = frequency_energy_matrix(raw_model.W1.data, config.p)

    result = {
        'config': asdict(config),
        'metrics_log': metrics_log,
        'final_energy': final_energy.cpu(),
        'model_state': raw_model.state_dict(),
        'steps_trained': step + 1,
    }

    # Final accuracies (use raw_model for weight manipulation in accuracy_after_zeroing)
    raw_model.eval()
    if config.task == 'joint':
        _, result['final_acc_add'] = evaluate(raw_model, add_test_X, add_test_y, task='add')
        _, result['final_acc_mul'] = evaluate(raw_model, mul_test_X, mul_test_y, task='mul')
        for t in [96, 128, 160]:
            result[f'acc_add_zeroed_{t}'] = accuracy_after_zeroing(
                raw_model, add_test_X, add_test_y, t, task='add')
            result[f'acc_mul_zeroed_{t}'] = accuracy_after_zeroing(
                raw_model, mul_test_X, mul_test_y, t, task='mul')
    else:
        _, result['final_acc'] = evaluate(raw_model, test_X, test_y)

    return result


def save_result(result: dict, output_dir: str, run_name: str):
    """Save checkpoint and metrics to disk."""
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Checkpoint (model state + config)
    torch.save({
        'model_state_dict': result['model_state'],
        'config': result['config'],
        'steps_trained': result['steps_trained'],
    }, os.path.join(run_dir, 'checkpoint.pt'))

    # Metrics (JSON-serializable subset)
    metrics = {
        'config': result['config'],
        'steps_trained': result['steps_trained'],
        'metrics_log': result['metrics_log'],
    }
    # Add final scalars
    for k, v in result.items():
        if k.startswith('final_') or k.startswith('acc_'):
            metrics[k] = v

    class _TensorEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            return super().default(obj)

    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, cls=_TensorEncoder)

    # Energy matrix (binary, for migration analysis)
    if 'final_energy' in result:
        torch.save(result['final_energy'], os.path.join(run_dir, 'energy.pt'))
