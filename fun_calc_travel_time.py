from py_libs import *
from fun_ray_tracing import ray_tracing

def calc_travel_time(angle, velocity_field, num_cells_x, num_cells_y, width, height, pos_source,):
    thetas, lengths, positions, cells = ray_tracing(angle, velocity_field, num_cells_x, num_cells_y, width, height,
                                                    pos_source, )
    Lengths = np.zeros_like(velocity_field)
    for jdx, cell in enumerate(cells):
        Lengths[cell[0], cell[1]] = lengths[jdx]
    time = (Lengths.reshape(-1, 1).T @ (1 / velocity_field.reshape(-1, 1))).item()

    return time, Lengths
