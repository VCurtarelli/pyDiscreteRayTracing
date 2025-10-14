from py_libs import *
from fun_ray_tracing import ray_tracing


def calc_travel_time(vf, ray):
    field = vf.field

    thetas, lengths, positions, cells = ray_tracing(vf,ray)
    Lengths = np.zeros_like(field)
    for jdx, cell in enumerate(cells):
        Lengths[cell[0], cell[1]] = lengths[jdx]
    time = (Lengths.reshape(-1, 1).T @ (1 / field.reshape(-1, 1))).item()

    return time, Lengths
