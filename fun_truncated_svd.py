import numpy as np
from numpy.linalg import svd


def truncated_svd(mat, epsilon, return_rank=True):
    U, s, Vh = svd(mat)
    V = Vh.T
    s[s < epsilon * s[0]] = 0
    S_ = np.diag(s)
    S = np.zeros_like(mat)
    mat_rank = (s[s != 0]).size
    S[:min(mat.shape), :min(mat.shape)] = S_
    if return_rank:
        return U, S, V, mat_rank
    else:
        return U, S, V
