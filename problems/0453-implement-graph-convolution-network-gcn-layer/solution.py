import numpy as np

def gcn_layer(A: np.ndarray, X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Perform a single GCN layer forward pass.
    
    Args:
        A: Adjacency matrix of shape (N, N)
        X: Node feature matrix of shape (N, F_in)
        W: Weight matrix of shape (F_in, F_out)
        
    Returns:
        Output feature matrix of shape (N, F_out)
    """
    N, F_in = X.shape
    _, F_out = W.shape

    A += np.identity(N)

    deg = A.sum(axis=1)
    inv_sqrt_deg = np.zeros_like(deg)
    mask = deg > 0
    inv_sqrt_deg[mask] = 1.0 / np.sqrt(deg[mask])
    A = inv_sqrt_deg[:, None] * A * inv_sqrt_deg[None, :]

    out = A @ (X @ W)

    return np.maximum(0.0, out)