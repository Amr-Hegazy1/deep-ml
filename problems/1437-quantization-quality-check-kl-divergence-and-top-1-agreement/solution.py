import math
import numpy as np


def log_softmax(z):
    # Find the maximum value along each row (axis=1)
    # keepdims=True ensures proper shape broadcasting (Nx1)
    row_max = np.max(z, axis=1, keepdims=True)
    
    # Shift values to prevent overflow
    shifted_z = z - row_max
    
    # Compute the log of the sum of exponentials row-wise
    log_sum_exp = np.log(np.sum(np.exp(shifted_z), axis=1, keepdims=True))
    
    # Return log-softmax
    return shifted_z - log_sum_exp


def kl_per_position(logits_ref, logits_q):
    prob_ref = np.exp(log_softmax(logits_ref))
    return np.sum(prob_ref * (log_softmax(logits_ref) - log_softmax(logits_q)), axis=-1)


def top1_agreement(logits_ref, logits_q):
    # Get the index of the highest logit for each sample (row)
    preds_ref = np.argmax(logits_ref, axis=-1)
    preds_q = np.argmax(logits_q, axis=-1)
    
    # Calculate the percentage of matching predictions
    agreement = np.mean(preds_ref == preds_q)
    return agreement

def p99_kl(kl_values):
    sorted_kl = np.sort(kl_values)
    n = len(sorted_kl)
    
    ceil_index = int(np.ceil(0.99 * n))
    python_index = ceil_index - 1
    
    # Boundary guard for edge cases where n is extremely small
    python_index = max(0, min(python_index, n - 1))
    
    return sorted_kl[python_index]


def quality_report(logits_ref, logits_q):
    # TODO: {'mean_kl', 'max_kl', 'p99_kl', 'top1_agreement'} rounded to 6 decimals
    kl = kl_per_position(logits_ref, logits_q)
    return {
        'mean_kl': float(np.round(np.mean(kl), 6)),
        'max_kl': float(np.round(np.max(kl), 6)),
        'p99_kl': float(np.round(p99_kl(kl), 6)),
        'top1_agreement': float(np.round(top1_agreement(logits_ref, logits_q), 6))
    }


def passes_gate(report, max_mean_kl, min_agreement):
    return report['mean_kl'] <= max_mean_kl and report['top1_agreement'] >= min_agreement