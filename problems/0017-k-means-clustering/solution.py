import numpy as np

def k_means_clustering(points, k, initial_centroids, max_iterations):
    X = np.asarray(points, dtype=float)
    C = np.asarray(initial_centroids, dtype=float)
    for _ in range(max_iterations):
        labels = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2).argmin(axis=1)  # (n,)
        new_C = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else C[j]
                          for j in range(k)])                                            # (k, d)
        if np.allclose(new_C, C):   # converged, stop early
            break
        C = new_C
    return [tuple(c) for c in np.round(C, 4)]