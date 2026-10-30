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
    
    P = np.asarray(points, dtype=float)
    q = np.asarray(query_point, dtype=float)
    d = ((P - q) ** 2).sum(axis=1)              
    idx = np.argsort(d, kind='stable')[:k]      
    return [points[i] for i in idx]