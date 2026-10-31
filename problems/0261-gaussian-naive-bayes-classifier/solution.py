import numpy as np

def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
	"""
	Implements Gaussian Naive Bayes classifier.
	
	Args:
		X_train: Training features (shape: N_train x D)
		y_train: Training labels (shape: N_train)
		X_test: Test features (shape: N_test x D)
	
	Returns:
		Predicted class labels for X_test (shape: N_test)
	"""
	classes = np.unique(y_train)                                                     # (C,)
	means  = np.array([X_train[y_train == c].mean(axis=0) for c in classes])         # (C, D)
	vars_  = np.array([X_train[y_train == c].var(axis=0) for c in classes]) + 1e-9   # (C, D)
	priors = np.array([np.mean(y_train == c) for c in classes])                      # (C,)
	
	diff = X_test[:, None, :] - means[None, :, :]                                    # (N_test, C, D)
	log_lik = -0.5 * (np.log(2 * np.pi * vars_)[None] + diff ** 2 / vars_[None]).sum(axis=2)  # (N_test, C)
	
	return classes[np.argmax(log_lik + np.log(priors)[None], axis=1)]                # (N_test,)