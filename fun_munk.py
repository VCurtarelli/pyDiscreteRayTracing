import numpy as np


def munk(eps, height, y_vec):
    Z = 2 * (height * y_vec - 1300) / 1300
    vY = 1500 * (1 + eps * (Z - 1 + np.exp(-Z)))
    return vY
