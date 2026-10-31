import numpy as np
from typing import Tuple

def calculate_potential_thresholds(values: np.ndarray) -> np.ndarray:
    return np.unique(values)[:-1]
    

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    # ✏️ TODO: implement
    
    rows, features = X.shape

    n = len(y)

    best_feature_index = None
    best_threshold = None

    min_gini = np.inf

    for feature in range(features):
        feature_values = X[:, feature]

        feature_values_sorted = np.sort(feature_values)

        potential_thresholds = calculate_potential_thresholds(feature_values_sorted)

        for potential_threshold in potential_thresholds:
            left = y[feature_values <= potential_threshold]
            right = y[feature_values > potential_threshold]

            left_yes = left.sum()
            left_no = len(left) - left_yes

            right_yes = right.sum()
            right_no = len(right) - right_yes

            gini_left = 1 - (left_yes / len(left)) ** 2 - (left_no / len(left)) ** 2

            gini_right = 1 - (right_yes / len(right)) ** 2 - (right_no / len(right)) ** 2

            weighted_gini = len(left) / n * gini_left + len(right) / n * gini_right

            if weighted_gini < min_gini:
                best_feature_index = feature
                best_threshold = potential_threshold
                min_gini = weighted_gini

    return best_feature_index, best_threshold
            



