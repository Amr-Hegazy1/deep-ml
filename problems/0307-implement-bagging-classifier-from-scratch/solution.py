import numpy as np

def get_potential_thresholds(values: np.ndarray) -> np.ndarray:
    return np.unique(values)

def train_estimator(data: np.ndarray) -> tuple[int, float]:
    X, Y = data[0], data[1]

    


    n_samples, n_features = X.shape

    best_feature_index = None
    best_threshold = None
    best_error = np.inf

    for feature in range(n_features):

        feature_values = X[:, feature]
        potential_thresholds = get_potential_thresholds(feature_values)

        for potential_threshold in potential_thresholds:
            pred = feature_values > potential_threshold

            error = (pred != Y).sum()

            if error < best_error:
                best_feature_index = feature
                best_threshold = potential_threshold
                best_error = error

    return best_feature_index, best_threshold

def h(x, estimator):
    best_feature_index, best_threshold = estimator

    return x[:, best_feature_index] > best_threshold

def bagging_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_estimators: int = 10, seed: int = 42) -> np.ndarray:
    """
    Implement a bagging classifier using decision stumps.
    
    Args:
        X_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,), binary {0, 1}
        X_test: Test features of shape (n_test_samples, n_features)
        n_estimators: Number of bootstrap samples/base estimators
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray: Predicted labels for X_test
    """
    # Your code here
    n = len(y_train)
    subsets = []

    for _ in range(n_estimators):
        indices = np.random.choice(n, n, replace=True)
        subsets.append((X_train[indices], y_train[indices]))

    

    estimators = []

    for i in range(n_estimators):
        estimator = train_estimator(subsets[i])
        estimators.append(estimator)

    predictions = []

    for i in range(n_estimators):
        predictions.append(h(X_test, estimators[i]))

    predictions = np.array(predictions)

    voting = predictions.mean(axis=0)

    return (voting >= 0.5).astype(int)

    

