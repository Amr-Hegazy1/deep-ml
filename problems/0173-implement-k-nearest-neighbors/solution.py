import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    
    points_np = np.array(points)
    query_point = np.array([query_point[0], query_point[1]])

    distances = [(i, np.linalg.norm(point - query_point)) for i, point in enumerate(points_np)]

    distances_sorted = sorted(distances, key=lambda x: (x[1], x[0]))


    top_k = distances_sorted[:k]

    result = [points[e[0]] for e in top_k]

    return result