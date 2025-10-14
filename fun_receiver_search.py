from py_libs import *
from fun_ray_tracing import ray_tracing


def receiver_search(vf, ray,
                    stop_param=0.005, iteration_step=0.05, iterations_max=100):
    theta_0 = ray.angle
    pos_source = ray.source
    pos_receiver = ray.receiver
    angles = [theta_0]
    paths = []
    pos_receiver = np.array(pos_receiver)
    field = vf.field
    num_cells_x = vf.cells_nx
    num_cells_y = vf.cells_ny
    width = vf.width
    height = vf.height

    best_theta = 0
    best_ang_dist = np.inf
    best_path = None
    while True:
        theta = angles[-1]
        _, _, positions, _ = ray_tracing(theta, field, num_cells_x, num_cells_y, width, height, pos_source)
        position = np.array(positions[-1])
        lin_dist = float(np.linalg.norm((position - pos_receiver) * np.array([1/width, 1/height]) * np.sqrt(2)))
        ang_dist = np.angle((position.T @ np.array([1, 1j])).item()) - np.angle((pos_receiver.T @ np.array([1, 1j])).item())
        if np.abs(ang_dist) < np.abs(best_ang_dist):
            best_ang_dist = ang_dist
            best_theta = theta
            best_path = positions
        if lin_dist < stop_param:
            # print("Solution found: {} iterations - {:.2f} / {:.2f}".format(len(angles), lin_dist, stop_param))
            break
        new_theta = (theta - iteration_step * lin_dist * np.sign(ang_dist)).item()
        if len(angles) == iterations_max:
            # print("Iteration forced break achieved: {}".format(len(angles)))
            break
        if len(angles) % 100 == 0:
            iteration_step *= 0.9
        angles.append(new_theta)
        paths.append(positions)

    angles.append(best_theta)
    paths.append(best_path)
    ray.angle = best_theta
    ray.path = best_path
    return angles, paths