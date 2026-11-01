import numpy as np

def K(x, z, kernal, sigma):
    if kernal == 'linear':
        return np.dot(x, z)

    return np.exp(-(np.linalg.norm(x - z) ** 2 / (2 * sigma ** 2)))


def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0) -> tuple:
    """
    Train a kernel SVM using the deterministic Pegasos algorithm.
    
    Args:
        data: Training data of shape (n_samples, n_features)
        labels: Labels of shape (n_samples,) with values in {-1, 1}
        kernel: 'linear' or 'rbf'
        lambda_val: Regularization parameter
        iterations: Number of training iterations
        sigma: RBF kernel bandwidth (only used if kernel='rbf')
    
    Returns:
        Tuple of (alphas, bias) where alphas is a list and bias is a float
    """
    # Your code here
    n_samples, n_features = data.shape

    a = np.zeros(n_samples)
    b = 0

    for t in range(1, iterations+1):
        learning_rate = 1 / (lambda_val * t)

        for i, (sample, label, alpha) in enumerate(zip(data, labels, a)):
            decision_value = 0
            for sample2, label2, alpha2 in zip(data, labels, a):
                 decision_value += alpha2 * label2 * K(sample2, sample, kernel, sigma)
            decision_value += b

            a[i] *= (1 - learning_rate * lambda_val)

            if label * decision_value < 1:
                a[i] += learning_rate
                b += learning_rate * label

    a = np.round(a, 4)

    b = round(b, 4)

    return a.tolist(), float(b)


