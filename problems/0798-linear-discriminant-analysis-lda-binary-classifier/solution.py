import numpy as np

def lda_predict(X_train, y_train, X_test):
    """
    Fit a binary LDA classifier on (X_train, y_train) and predict labels for X_test.

    Args:
        X_train: array-like of shape (n_samples, n_features)
        y_train: array-like of shape (n_samples,) with values in {0, 1}
        X_test:  array-like of shape (n_test, n_features)

    Returns:
        List[int] of predicted labels (0 or 1) for each row in X_test.
    """

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)

    n_test, _ = X_test.shape

    X0 = X_train[y_train == 0]
    X1 = X_train[y_train == 1]

    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)

    Sw = (X0 - mu0).T @ (X0 - mu0) + (X1 - mu1).T @ (X1 - mu1)

    w = np.linalg.inv(Sw) @ (mu1 - mu0)

    c = w @ ((mu0 + mu1) / 2)


    res = np.zeros((n_test, ))
    
    for i, x in enumerate(X_test):

        res[i] = (w @ x.T) > c

    return res
