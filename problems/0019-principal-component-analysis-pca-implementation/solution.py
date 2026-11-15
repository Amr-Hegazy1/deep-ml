import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Your code here
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    data = (data - mean) / std

    covariance = np.cov(data, rowvar=False)

    e_values, e_vectors = np.linalg.eigh(covariance)
    

    for i in range(len(e_vectors)):
        e_vector = e_vectors[:, i]
        abs_e_vector = np.abs(e_vector)
        flip_sign = -1 if e_vector[abs_e_vector > 1e-10][0] < 0 else 1
        e_vectors[:,i] = e_vector * flip_sign

    top_k_idx = np.argpartition(e_values, -k)[-k:]

    return np.round(e_vectors[:, top_k_idx], 4)


