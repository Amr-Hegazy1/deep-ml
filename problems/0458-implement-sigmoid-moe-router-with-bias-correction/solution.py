import numpy as np


def sigmoid(z):
    return 1/(1 + np.exp(-z))


# Implement your function below.
def sigmoid_moe_router(hidden_states, gate_weight, score_bias, top_k):
    """
    Implement sigmoid-based MoE routing with bias correction.

    Args:
        hidden_states (np.ndarray): Token representations, shape (num_tokens, hidden_dim).
        gate_weight (np.ndarray): Gate projection weights, shape (num_experts, hidden_dim).
        score_bias (np.ndarray): Learned bias for load balancing, shape (num_experts,).
        top_k (int): Number of experts to select per token.

    Returns:
        tuple: (top_k_weights, top_k_indices)
            - top_k_weights: Normalized routing weights, shape (num_tokens, top_k).
            - top_k_indices: Selected expert indices, shape (num_tokens, top_k).
    """
    
    router_logits = hidden_states @ gate_weight.T # (num_tokens, num_experts)
    

    router_logits_sigmoid = sigmoid(router_logits) # (num_tokens, num_experts)

    

    router_logits_sigmoid_biased = router_logits_sigmoid + score_bias # (num_tokens, num_experts)

    num_tokens, num_experts = router_logits_sigmoid_biased.shape
    expert_ids = np.broadcast_to(np.arange(num_experts), (num_tokens, num_experts))
    sorted_indices = np.lexsort((expert_ids, -router_logits_sigmoid_biased), axis=-1)
    
    top_k_indices = sorted_indices[:, :top_k]

    top_k_weights = np.take_along_axis(router_logits_sigmoid, top_k_indices, axis=-1)

    
    normalized_weights = top_k_weights / np.sum(top_k_weights, axis=-1, keepdims=True)

    return normalized_weights, top_k_indices
