import numpy as np


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):

	updated_weights = initial_weights
	updated_bias = initial_bias
	mse_values = []

	rows, cols = features.shape

	for i in range(epochs):

		scores = features @ updated_weights + updated_bias

		probabilities = sigmoid(scores)

		diff = (labels - probabilities)

		sq_diff = diff ** 2

		mse = sq_diff.mean()
		mse_values.append(round(mse, 4))

		# print(diff)
		# print(features)

		d_score = -2 * diff * (probabilities * (1 - probabilities))

		partial_diff_weights = (features.T @ d_score) / rows
		partial_diff_bias = d_score.mean()

		

		updated_weights -= learning_rate * partial_diff_weights
		updated_bias -= learning_rate * partial_diff_bias

		# print(updated_weights)
		# print(updated_bias)

	updated_weights = np.round(updated_weights, 4)
	updated_bias = np.round(updated_bias, 4)
	

	return updated_weights, updated_bias, mse_values