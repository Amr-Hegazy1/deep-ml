import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    Q = x_new @ W_Q
    K = x_new @ W_K
    V = x_new @ W_V

    d_k = Q.shape[0]

    

    if cache is None:
        # Reshape 1D vector (d_k,) to 2D matrix (1, d_k)
        K_cache = K[np.newaxis, :]
        V_cache = V[np.newaxis, :]
    else:
        K_cache, V_cache = cache
        # Reshape current token to 2D (1, d_k) and stack vertically
        K_cache = np.vstack([K_cache, K])
        V_cache = np.vstack([V_cache, V])

    
    

    scores = (Q @ K_cache.T) / np.sqrt(d_k)


    
    


    return softmax(scores) @ V_cache, (K_cache, V_cache)