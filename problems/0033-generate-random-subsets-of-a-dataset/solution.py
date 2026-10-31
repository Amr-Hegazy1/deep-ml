import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True):
    n = len(y)
    subset_size = n if replacements else n // 2

    result = []

    for subset in range(n_subsets):
        indicies = np.random.choice(n, subset_size, replace=replacements)
        result.append((X[indicies], y[indicies]))

    return result