import numpy as np


def edge_list_to_adjacency(edges, n, undirected=True):
    # TODO: (n, n) float matrix with a 1.0 for every edge (both directions if undirected)
    adj = np.zeros((n, n))

    for edge in edges:
        u, v  = edge
        adj[u, v] = 1
        if undirected:
            adj[v, u] = 1

    return adj


def degrees(A):
    return A.sum(axis = 1)


def adjacency_to_csr(A):
    # Return (indptr, indices) with neighbours of node i at indices[indptr[i]:indptr[i+1]]
    A = np.asarray(A)
    rows, indices = np.nonzero(A)
    
    counts = np.bincount(rows, minlength=A.shape[0])
    indptr = np.empty(A.shape[0] + 1, dtype=indices.dtype)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    
    return indptr, indices



def normalized_adjacency(A, add_self_loops=True, mode="sym"):
    # TODO: D^-1/2 A D^-1/2 or D^-1 A after optional self-loops; zero-degree rows stay zero
    A = np.array(A, dtype=float, copy=True)
    n, _ = A.shape
    
    if add_self_loops:
        np.fill_diagonal(A, 1.0)
        
    d = degrees(A)
    
    if mode == "sym":
        inv_sqrt_d = np.zeros_like(d, dtype=float)
        mask = d > 0
        inv_sqrt_d[mask] = 1.0 / np.sqrt(d[mask])
        return inv_sqrt_d[:, None] * A * inv_sqrt_d[None, :]
        
    elif mode in ("rw", "row"):
        inv_d = np.zeros_like(d, dtype=float)
        mask = d > 0
        inv_d[mask] = 1.0 / d[mask]
        return inv_d[:, None] * A


def num_walks(A, k):
    return np.linalg.matrix_power(A, k)
