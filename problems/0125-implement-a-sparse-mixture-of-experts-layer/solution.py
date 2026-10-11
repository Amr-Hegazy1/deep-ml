import numpy as np


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z_max = np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)

def moe(x: np.ndarray, We: np.ndarray, Wg: np.ndarray, n_experts: int, top_k: int) -> np.ndarray:
    """
    Args:
        x: Input tensor of shape (n_batch, l_seq, d_model)
        We: Expert weights of shape (n_experts, d_model, d_model)
        Wg: Gating weights of shape (d_model, n_experts)
        n_experts: Number of experts
        top_k: Number of experts to route each token to
    Returns:
        Output tensor of shape (n_batch, l_seq, d_model)
    """
    n_batch, l_seq, d_model = x.shape

    gate_logits = x @ Wg

    expert_outputs = np.empty((n_experts, n_batch, l_seq, d_model))

    for n_expert in range(n_experts):
        expert_outputs[n_expert] = x @ We[n_expert]

    
    top_k_experts_ind = np.argpartition(gate_logits, -top_k, axis=-1)[..., -top_k:]

    top_k_expert_logits = np.take_along_axis(gate_logits, top_k_experts_ind, axis=-1)

    top_k_expert_probs = softmax(top_k_expert_logits)
    batch_idx, seq_idx = np.indices((n_batch, l_seq))

    result = np.zeros_like(x)

    for k in range(top_k):
        expert = top_k_experts_ind[..., k]               # (n_batch, l_seq)
        prob = top_k_expert_probs[..., k, None]          # (n_batch, l_seq, 1)

        # Index expert, batch, and sequence simultaneously
        selected_output = expert_outputs[expert, batch_idx, seq_idx]  # (n_batch, l_seq, d_model)
        result += selected_output * prob



    return result