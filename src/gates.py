"""
Gate evaluation logic for the fail-fast protocol.

Gates 1 and 2: generate heatmaps + compute numeric checks.
Gate 2.5: accuracy after zeroing.
Gate 3: difference-in-differences with ambiguity zones.
"""

import numpy as np


def evaluate_gate1(step2_results: dict) -> tuple[bool, str]:
    """
    Gate 1: Does asymmetric decay show decreasing norms on high-decay side?

    Checks: mean_norm_right / mean_norm_left < 0.7 for asymmetric condition.
    """
    asym_results = step2_results.get('asymmetric', [])
    if not asym_results:
        return False, "No asymmetric results found"

    ratios = []
    for result in asym_results:
        norms = result['metrics_log'][-1]['column_norms']
        half = len(norms) // 2
        left_mean = np.mean(norms[:half])
        right_mean = np.mean(norms[half:])
        if left_mean < 1e-10:
            continue
        ratios.append(right_mean / left_mean)

    mean_ratio = np.mean(ratios)
    passed = mean_ratio < 0.7
    detail = (f"norm_right/norm_left = {mean_ratio:.3f} "
              f"({'< 0.7 PASS' if passed else '>= 0.7 FAIL'})")
    return passed, detail


def evaluate_gate2(step2_results: dict) -> tuple[bool, str]:
    """
    Gate 2 (Direction Test): Is asymmetric a mirror image of reversed?

    Checks: asymmetric has right < left, reversed has left < right.
    """
    def get_ratio(results):
        ratios = []
        for result in results:
            norms = result['metrics_log'][-1]['column_norms']
            half = len(norms) // 2
            left_mean = np.mean(norms[:half])
            right_mean = np.mean(norms[half:])
            if left_mean + right_mean < 1e-10:
                continue
            ratios.append(right_mean / left_mean)
        return np.mean(ratios) if ratios else 1.0

    asym_ratio = get_ratio(step2_results.get('asymmetric', []))
    rev_ratio = get_ratio(step2_results.get('reversed', []))

    passed = asym_ratio < 1.0 and rev_ratio > 1.0
    detail = (f"asymmetric ratio={asym_ratio:.3f}, reversed ratio={rev_ratio:.3f} "
              f"({'mirror PASS' if passed else 'NOT mirrored FAIL'})")
    return passed, detail


def evaluate_gate2_5(step2_results: dict, threshold: int = 128,
                     acc_threshold: float = 0.80) -> tuple[bool, str]:
    """
    Gate 2.5 (Compaction Quality): After zeroing columns j >= threshold,
    at least one task retains > 80% accuracy.
    """
    asym_results = step2_results.get('asymmetric', [])
    if not asym_results:
        return False, "No asymmetric results found"

    add_accs = [r.get(f'acc_add_zeroed_{threshold}', 0.0) for r in asym_results]
    mul_accs = [r.get(f'acc_mul_zeroed_{threshold}', 0.0) for r in asym_results]

    mean_add = np.mean(add_accs)
    mean_mul = np.mean(mul_accs)

    passed = mean_add > acc_threshold or mean_mul > acc_threshold
    detail = (f"add={mean_add:.3f}, mul={mean_mul:.3f} after zeroing j>={threshold} "
              f"({'PASS' if passed else 'FAIL'})")
    return passed, detail


def compute_variance_calibration(step2_results: dict, p: int) -> dict:
    """
    Step 2.5: compute between-seed std of left-half energy fraction
    and soft migration score from asymmetric condition.
    """
    from .metrics import soft_migration_score as sms_fn

    asym_results = step2_results.get('asymmetric', [])
    lhefs = [r['metrics_log'][-1]['left_energy'] for r in asym_results]

    std_lhef = float(np.std(lhefs))

    return {
        'std_lhef': std_lhef,
        'n_seeds': len(asym_results),
        'lhef_values': lhefs,
    }


def evaluate_gate3(step2_results: dict, step3_results: dict,
                   thresholds: dict | None = None) -> tuple[str, str]:
    """
    Gate 3: Difference-in-differences migration test with ambiguity zones.

    migration_effect = (asymmetric_with - asymmetric_without)
                     - (uniform_with - uniform_without)

    Returns: ('pass' | 'fail' | 'ambiguous', detail_string)
    """
    if thresholds is None:
        thresholds = {
            'lhef': 0.05,
            'sms': 0.5,
            'acc': 0.05,
        }

    # Collect left-energy fraction from each condition
    def mean_lhef(results):
        return np.mean([r['metrics_log'][-1]['left_energy'] for r in results])

    def mean_acc_zeroed(results, threshold=128):
        add_accs = [r.get(f'acc_add_zeroed_{threshold}', 0.0) for r in results]
        mul_accs = [r.get(f'acc_mul_zeroed_{threshold}', 0.0) for r in results]
        return max(np.mean(add_accs), np.mean(mul_accs))

    asym_with = step2_results.get('asymmetric', [])
    asym_without = step3_results.get('asym_actzeroed', [])
    uniform_with = step2_results.get('uniform', [])
    uniform_without = step3_results.get('uniform_actzeroed', [])

    if not all([asym_with, asym_without, uniform_with, uniform_without]):
        return 'fail', "Missing condition results"

    # LHEF migration effect
    lhef_effect = ((mean_lhef(asym_with) - mean_lhef(asym_without))
                   - (mean_lhef(uniform_with) - mean_lhef(uniform_without)))

    # Accuracy migration effect
    acc_effect = ((mean_acc_zeroed(asym_with) - mean_acc_zeroed(asym_without))
                  - (mean_acc_zeroed(uniform_with) - mean_acc_zeroed(uniform_without)))

    # SMS migration effect
    def mean_sms(results):
        vals = [r['metrics_log'][-1].get('soft_migration_score', 0.0)
                for r in results]
        return np.mean(vals)

    sms_effect = ((mean_sms(asym_with) - mean_sms(asym_without))
                  - (mean_sms(uniform_with) - mean_sms(uniform_without)))

    # Compute SE from per-seed effects (paired)
    n = min(len(asym_with), len(asym_without),
            len(uniform_with), len(uniform_without))

    lhef_per_seed = []
    acc_per_seed = []
    sms_per_seed = []
    for i in range(n):
        aw_lhef = asym_with[i]['metrics_log'][-1]['left_energy']
        ao_lhef = asym_without[i]['metrics_log'][-1]['left_energy']
        uw_lhef = uniform_with[i]['metrics_log'][-1]['left_energy']
        uo_lhef = uniform_without[i]['metrics_log'][-1]['left_energy']
        lhef_per_seed.append((aw_lhef - ao_lhef) - (uw_lhef - uo_lhef))

        aw_acc = max(asym_with[i].get('acc_add_zeroed_128', 0),
                     asym_with[i].get('acc_mul_zeroed_128', 0))
        ao_acc = max(asym_without[i].get('acc_add_zeroed_128', 0),
                     asym_without[i].get('acc_mul_zeroed_128', 0))
        uw_acc = max(uniform_with[i].get('acc_add_zeroed_128', 0),
                     uniform_with[i].get('acc_mul_zeroed_128', 0))
        uo_acc = max(uniform_without[i].get('acc_add_zeroed_128', 0),
                     uniform_without[i].get('acc_mul_zeroed_128', 0))
        acc_per_seed.append((aw_acc - ao_acc) - (uw_acc - uo_acc))

        aw_sms = asym_with[i]['metrics_log'][-1].get('soft_migration_score', 0.0)
        ao_sms = asym_without[i]['metrics_log'][-1].get('soft_migration_score', 0.0)
        uw_sms = uniform_with[i]['metrics_log'][-1].get('soft_migration_score', 0.0)
        uo_sms = uniform_without[i]['metrics_log'][-1].get('soft_migration_score', 0.0)
        sms_per_seed.append((aw_sms - ao_sms) - (uw_sms - uo_sms))

    se_lhef = float(np.std(lhef_per_seed) / np.sqrt(n)) if n > 1 else float('inf')
    se_acc = float(np.std(acc_per_seed) / np.sqrt(n)) if n > 1 else float('inf')
    se_sms = float(np.std(sms_per_seed) / np.sqrt(n)) if n > 1 else float('inf')

    # Ambiguity zone logic
    checks = {
        'lhef': (lhef_effect, thresholds['lhef'], se_lhef),
        'sms': (sms_effect, thresholds['sms'], se_sms),
        'acc': (acc_effect, thresholds['acc'], se_acc),
    }

    any_fail = False
    any_ambiguous = False
    details = []

    for name, (effect, thresh, se) in checks.items():
        margin = effect - thresh
        if se > 0 and abs(margin) < se:
            any_ambiguous = True
            details.append(f"{name}: effect={effect:.4f}, threshold={thresh}, "
                          f"SE={se:.4f} → AMBIGUOUS")
        elif effect < thresh:
            any_fail = True
            details.append(f"{name}: effect={effect:.4f} < {thresh} → FAIL")
        else:
            details.append(f"{name}: effect={effect:.4f} > {thresh} → PASS")

    detail = "; ".join(details)

    if any_fail:
        return 'fail', detail
    elif any_ambiguous:
        return 'ambiguous', detail
    else:
        return 'pass', detail
