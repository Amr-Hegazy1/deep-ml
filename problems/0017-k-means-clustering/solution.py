import numpy as np

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	
	final_centroids = np.array(initial_centroids)

	points = np.array(points)
	
	# print(points)

	# print(final_centroids)

	for _ in range(1):
	
		distances = np.linalg.norm(points[:, None, :] - final_centroids[None, :, :], axis=2)

		# print(distances)

		min_index_each_point = np.argsort(distances, axis=1)[:, 0]

		# print(min_index_each_point)

		clusters = []
		for i in range(k):

			cluster = points[min_index_each_point == i]
			# print(cluster)
			

			clusters.append(cluster)

		clusters = np.array(clusters)

		final_centroids = np.mean(clusters, axis=1)
		# print(final_centroids)

	final_centroids = np.round(final_centroids, decimals=4)


	return list(map(tuple, final_centroids))