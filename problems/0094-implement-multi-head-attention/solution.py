import numpy as np
from typing import Tuple

def softmax(x):
    """Compute softmax values for each set of scores in x (along the last axis)."""
    # Using axis=-1 ensures it normalizes correctly for 1D or 2D inputs
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    return (X @ W_q, X @ W_k, X @ W_v)

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    return softmax( (Q @ K.T) / np.sqrt(Q.shape[0]) ) @ V

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    hidden_dim = Q.shape[1]
    head_dim = hidden_dim // n_heads

    output = None

    for i in range(n_heads):
        start = i * head_dim
        end = (i + 1) * head_dim
        partial_attn = self_attention(Q[:, start:end], K[:, start:end], V[:, start:end])
        #print(partial_attn)
        
        output = np.concatenate([output, partial_attn], axis=1) if output is not None else partial_attn

    return output