import numpy as np


def leaky_relu(x, slope=0.2):
    return np.where(x > 0, x, x * slope)


def attention_logits(H, a_src, a_dst, slope=0.2):
    # TODO: (n, n) matrix of leaky_relu(H[i] @ a_src + H[j] @ a_dst)
    n, d = H.shape

    out = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            out[i][j] = leaky_relu(H[i] @ a_src + H[j] @ a_dst, slope)

    return out


def masked_softmax(E, mask):
    # Row-wise softmax over masked entries only; zeros elsewhere
    E = np.asarray(E, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    masked_E = np.where(mask, E, -np.inf)

    row_max = np.max(masked_E, axis=1, keepdims=True)
   
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)

    exp_E = np.where(mask, np.exp(masked_E - row_max), 0.0)

    row_sums = exp_E.sum(axis=1, keepdims=True)
    return np.divide(exp_E, row_sums, out=np.zeros_like(exp_E), where=row_sums > 0)


def gat_layer(A, X, W, a_src, a_dst, slope=0.2):
    # TODO: H = X @ W; mask = A + I > 0; alpha = masked softmax of the logits; return (alpha, alpha @ H)
    H = X @ W

    n, d = H.shape

    mask = (A + np.identity(n)) > 0
    
    logits = attention_logits(H, a_src, a_dst, slope)

    alpha = masked_softmax(logits, mask)

    return alpha, alpha @ H


def multi_head_gat(A, X, Ws, a_srcs, a_dsts,
                   concat=True, slope=0.2):

    outputs = []

    for W, a_src, a_dst in zip(Ws, a_srcs, a_dsts):

        alpha, H = gat_layer(
            A, X, W, a_src, a_dst, slope
        )

        outputs.append(H)

    if concat:
        return np.concatenate(outputs, axis=1)
    else:
        return np.mean(outputs, axis=0)
