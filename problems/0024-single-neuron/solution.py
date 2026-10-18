import math
import numpy as np

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	
	features = np.array(features)
	labels = np.array(labels)
	weights = np.array(weights)

	probabilities = sigmoid(features @ weights + bias)

	sq_diff = (labels - probabilities) ** 2

	mse = sq_diff.mean()

	return probabilities, mse