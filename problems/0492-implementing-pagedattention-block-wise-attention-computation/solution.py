import numpy as np

def gather(
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    block_table: list,
    context_len: int
) -> tuple:

    num_physical_blocks, block_size, num_heads, head_dim = key_cache.shape

    K = np.empty((context_len, num_heads, head_dim))

    V = np.empty((context_len, num_heads, head_dim))

    for logical_block, physical_block in enumerate(block_table):
        start = logical_block * block_size
        end = min((logical_block + 1) * block_size, context_len)

        n = end - start

        if n <= 0:
            break

        K[ start : end ] = key_cache[ physical_block , : n, :, : ]

        V[ start : end ] = value_cache[ physical_block , : n, :, : ]

    return K, V

def attention_softmax(raw_scores, axis=-1):
    """
    Computes numerically stable softmax over a specific axis using NumPy.
    
    Args:
        raw_scores (np.ndarray): Array of raw attention scores (e.g., shape [num_heads, seq_len])
        axis (int): The axis along which to compute softmax (defaults to the last axis)
    """
    max_scores = np.max(raw_scores, axis=axis, keepdims=True)
    
    exponent_scores = np.exp(raw_scores - max_scores)
    
    exponent_sum = np.sum(exponent_scores, axis=axis, keepdims=True)
    
    attention_weights = exponent_scores / exponent_sum
    
    return attention_weights
    


def paged_attention(
    query: np.ndarray,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    block_table: list,
    context_len: int
) -> np.ndarray:
    """
    Perform scaled dot-product attention with paged KV cache.

    Args:
        query:       (num_heads, head_dim) query for a single token
        key_cache:   (num_physical_blocks, block_size, num_heads, head_dim)
        value_cache: (num_physical_blocks, block_size, num_heads, head_dim)
        block_table: list of physical block indices (logical -> physical)
        context_len: number of valid KV tokens in the sequence

    Returns:
        (num_heads, head_dim) attention output, rounded to 4 decimal places
    """
    
    num_heads, head_dim = query.shape

    K, V = gather(key_cache,
    value_cache,
    block_table,
    context_len)

    out = np.empty((num_heads, head_dim))

    for head in range(num_heads):
        out[head] = np.round(attention_softmax(query[head] @ K[:, head, :].T / np.sqrt(head_dim)) @ V[:, head, :], 4)

    return out


