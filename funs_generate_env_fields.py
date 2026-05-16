import numpy as np


def generate_temp_field(x_mat, y_mat, tmin=4, tmax=31, dpth=600, wdth=4000, x_fac=0.):
    return tmin + (tmax - tmin) * np.exp(-y_mat / dpth) * (1 - x_fac * (x_mat / wdth))


def generate_sal_field(x_mat, y_mat, smed=34.5, dpth=500, wdth=2000, x_fac=0.):
    return smed + (1 + x_fac * (1 - x_mat / wdth)) / (1 + np.exp((y_mat - dpth) / 200))