import numpy as np

def stacking_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    base_classifiers: list,
    meta_classifier,
    n_folds: int = 5
) -> np.ndarray:

    n_samples = len(X_train)
    n_base = len(base_classifiers)

    
    train_predictions = np.zeros((n_samples, n_base))

    
    folds = np.array_split(np.arange(n_samples), n_folds)

    for i in range(n_folds):

        val_indices = folds[i]

        train_indices = np.concatenate([
            folds[j]
            for j in range(n_folds)
            if j != i
        ])

        subset_X_train = X_train[train_indices]
        subset_y_train = y_train[train_indices]

        X_val = X_train[val_indices]

        for j, base_classifier in enumerate(base_classifiers):

            y_pred = base_classifier(
                subset_X_train,
                subset_y_train,
                X_val
            )

            train_predictions[val_indices, j] = y_pred

    
    test_predictions = np.zeros((len(X_test), n_base))

    for j, base_classifier in enumerate(base_classifiers):

        y_pred = base_classifier(
            X_train,
            y_train,
            X_test
        )

        test_predictions[:, j] = y_pred

    return meta_classifier(
        train_predictions,
        y_train,
        test_predictions
    )