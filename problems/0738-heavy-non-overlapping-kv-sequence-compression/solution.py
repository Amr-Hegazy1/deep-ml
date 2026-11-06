import numpy as np

def compress_kv(K, w, b, m):
    """
    Compress a key sequence by aggregating every m consecutive entries into one
    using a content-dependent softmax with a positional bias.

    Args:
        K: array-like of shape (S, D), with S divisible by m
        w: array-like of shape (D,)
        b: array-like of shape (m,)
        m: int, block size

    Returns:
        Nested list of shape (S/m, D).
    """
    K = np.asarray(K)
    w = np.asarray(w)
    b = np.asarray(b)

    S, D = K.shape

    s = (K @ w).reshape(-1, m) + b
    exp_s = np.exp(s - np.max(s, axis=1, keepdims=True))
    weights = exp_s / exp_s.sum(axis=1, keepdims=True)

    K_blocks = K.reshape(-1, m, D)

    compressed = np.sum(
        K_blocks * weights[:, :, None],
        axis=1
    )

    return compressed.tolist()

