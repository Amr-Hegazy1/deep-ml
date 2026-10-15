import numpy as np

def safe_softmax(z, axis=-1):
  # Subtract the max value for numerical stability
  z_shifted = z - np.max(z, axis=axis, keepdims=True)
  exp_z = np.exp(z_shifted)
  return exp_z / np.sum(exp_z, axis=axis, keepdims=True)

def sliding_window_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute sliding window attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
        window_size: Number of positions to the left and right each query can attend to
    
    Returns:
        Output matrix of shape (seq_len, d_v), rounded to 4 decimal places.
    """

    seq_len, d_k = Q.shape

    scores = Q @ K.T / np.sqrt(d_k)

    for i in range(seq_len):
        neighbors = np.arange(max(0, i - window_size), min(i + window_size+1, seq_len))
        mask = np.arange(0, seq_len)
        

        mask = np.setdiff1d(mask, neighbors)
        # print(i, neighbors, mask)
        
        scores[i, mask] = -np.inf

        # print(i, scores)
    

    scores = safe_softmax(scores)


    return np.round(scores @ V, 4)