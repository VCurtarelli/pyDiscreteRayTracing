import numpy as np


def munk(eps, height, y_vec, x_vec=None):
    if x_vec is None:
        x_vec = np.zeros_like(y_vec)
    Z = 2 * (height * y_vec - 1300) / 1300
    vY = 1500 * (1 + eps * (Z - 1 + (1+0.1*x_vec) * np.exp(-Z)))
    return vY
