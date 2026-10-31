import numpy as np

def sigma(z: np.ndarray) -> np.ndarray:
	return 1 / (1 + np.exp(-z))

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	# Your code here
	m, n = X.shape # N_train, D
	y = y.reshape(-1, 1)
	bias = np.ones((m, 1))
	

	X = np.column_stack([X, bias]) # N_train, D+1

	# print(X)

	theta = np.zeros((n+1, 1)) # D + 1, 1

	losses = []

	for _ in range(iterations):

		pred = sigma(X @ theta)

		loss = -1 * (y * np.log(pred) + (1 - y) * np.log(1 - pred)).sum()

		residuals = (pred - y)

		theta -= learning_rate * (X.T @ residuals)

		losses.append(round(loss, 4))

	theta = np.round(theta, 4).flatten()

	theta = np.roll(theta, 1, axis=None)


	return theta.tolist(), losses







