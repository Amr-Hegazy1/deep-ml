import numpy as np


def block_diagonal(mats):
    width = 0
    height = 0

    for mat in mats:
        n, m = mat.shape
        width += m
        height += n

    adj = np.zeros((height, width))

    i = 0
    j = 0
    for mat in mats:
        n, m = mat.shape

        adj[i : i + n, j : j + m] = mat

        i += n
        j += m

    return adj


def batch_graphs(adjs, feats):
    # TODO: return (A_block_diagonal, X_stacked, batch_vector)
    A_block_diagonal = block_diagonal(adjs)
    n, _ = A_block_diagonal.shape
    _, d = feats[0].shape
    X_stacked = np.empty((0, d))
    for feat in feats:
        X_stacked = np.vstack((X_stacked, feat))

    batch_vector = []

    for i, feat in enumerate(feats):
        n, _ = feat.shape
        batch_vector.extend([i] * n)

    batch_vector = np.array(batch_vector)

    return A_block_diagonal, X_stacked, batch_vector

    


def readout(H, batch, mode="sum"):
    # TODO: (num_graphs, d) with sum / mean / max per graph
    n, d = H.shape
    G = batch.max() + 1

    out = np.zeros((G, d))

    for i in range(G):
        nodes = H[batch == i]

        if mode == "sum":
            out[i] = nodes.sum(axis=0)

        elif mode == "mean":
            out[i] = nodes.mean(axis=0)

        elif mode == "max":
            out[i] = nodes.max(axis=0)

        else:
            raise ValueError("Invalid readout mode")

    return out


def unbatch(H, batch):
    G = batch.max() + 1
    graphs = []

    for i in range(G):
        nodes = H[batch == i]
        graphs.append(nodes)

    return graphs
