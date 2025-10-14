import numpy as np

from py_libs import *
from py_params import *
from fun_calc_grad import calc_grad


def ray_tracing(theta_0, velocity_field, num_cells_x, num_cells_y, width, height, pos_source=(0, 0)):
    thetas = []
    lengths = []
    positions = []
    cells = []
    cell_width = width / num_cells_x
    cell_height = height / num_cells_y

    idx_x = int(pos_source[0] // cell_width)
    idx_y = int(pos_source[1] // cell_height)

    thetas.append(theta_0)
    positions.append(pos_source)
    cells.append((int(pos_source[0] // cell_width), int(pos_source[1] // cell_height)))
    cells.append((int(pos_source[0] // cell_width), int(pos_source[1] // cell_height)))
    while True:
        pos_a_x, pos_a_y = positions[-1]  # crossing position from cell a (previous cell) to cell b (current cell)
        idx_y, idx_x = cells[-1]
        idx_y_, idx_x_ = cells[-2]
        if not (0 <= idx_x < num_cells_x and 0 <= idx_y < num_cells_y):
            break
        # print(idx_x, idx_y)
        # print(idx_x_, idx_y_)
        # print()
        # print(velocity_field.shape)
        Va = velocity_field[idx_y, idx_x]
        Vb = velocity_field[idx_y_, idx_x_]
        mag_grad_V, direc_grad_V = calc_grad(velocity_field, cells[-1], cells[-2], pos_a_x, pos_a_y, cell_width, cell_height, num_cells_x, num_cells_y)
        # print(np.rad2deg(direc_grad_V), mag_grad_V)
        # gradient_factor = 0
        discrete_gradient = 100*mag_grad_V
        gradient_factor = 1/(1+np.exp(-discrete_gradient))
        Va, Vb = gradient_factor*Va+(1-gradient_factor)*Vb, (1-gradient_factor)*Va + gradient_factor*Vb

        phi = direc_grad_V  #TODO: IF GRADIENT OF VELOCITY AND RAY DIRECTIONS ARE REVERTED, CANCEL IT
        theta = thetas[-1]
        if np.cos(theta-phi) < 0:
            phi = (phi+pi+2*pi) % (2*pi)
        alpha = theta - phi
        sin_beta = np.sin(alpha) * Vb / Va
        if abs(sin_beta) > 1.01:
            beta = pi-alpha
        else:
            if np.abs(sin_beta) > 1:
                sin_beta = np.sign(sin_beta)
            beta = np.asin(sin_beta)
        gamma = beta + phi

        theta_b = (gamma+2*pi) % (2*pi)
        sin_theta_b = np.sin(theta_b)
        cos_theta_b = np.cos(theta_b)

        x0 = (idx_x-1)*cell_width
        x1 = (idx_x+1)*cell_width
        y0 = (idx_y-1)*cell_height
        y1 = (idx_y+1)*cell_height
        px = pos_a_x
        py = pos_a_y

        if sin_theta_b == 0:
            t_T = np.inf
            t_B = np.inf
        else:
            t_T = (y1 - py) / sin_theta_b
            t_B = (y0 - py) / sin_theta_b
        if cos_theta_b == 0:
            t_R = np.inf
            t_L = np.inf
        else:
            t_R = (x1 - px) / cos_theta_b
            t_L = (x0 - px) / cos_theta_b
        ts = [t_T, t_B, t_R, t_L]
        for i in range(len(ts)):
            if ts[i] < 0:
                ts[i] = np.inf
        t_min_idx = np.argmin(ts)
        t = ts[t_min_idx]
        if t_min_idx == 0:
            idx_y += 1
        if t_min_idx == 1:
            idx_y -= 1
        if t_min_idx == 2:
            idx_x += 1
        if t_min_idx == 3:
            idx_x -= 1
        pos_b_x = px + t*cos_theta_b
        pos_b_y = py + t*sin_theta_b

        if not ((0 <= pos_b_x <= width)
            and (0 <= pos_b_y <= height)):
            break
        travel_dist = np.sqrt((pos_b_x - pos_a_x) ** 2 + (pos_b_y - pos_a_y) ** 2)
        thetas.append(float(theta_b))
        lengths.append(travel_dist)
        positions.append((pos_b_x, pos_b_y))
        cells.append((idx_y, idx_x))

        if sin_theta_b > 1 or np.isnan(sin_theta_b):
            break

    cells = cells[1:-1]
    thetas = thetas[:-1]
    return thetas,lengths,positions,cells


if __name__ == '__main__':
    angles = np.linspace(0, pi/2, 40)
    paths = []
    for angle in angles:
        thetas, lengths, path, cells = ray_tracing(angle, velocity_field, num_cells_x, num_cells_y, width, height,
                                              pos_source, )
        path = np.array(path)
        paths.append(path)
    ## --------------------
    # RESULT PLOTTING
    plt.imshow(np.flipud(velocity_field), extent=(0, width, 0, height),
               alpha=0.15)  # plots velocity field in background

    color_a = np.array([0, 0, 255]) / 255
    color_b = np.array([255, 255, 0]) / 255
    for idx in range(len(paths)):  # plots each iteration on receiver search
        path = np.array(paths[idx])
        fac = (idx / len(paths)) ** 2
        plt.plot(path[:, 0], path[:, 1], label=np.around(np.rad2deg(angles[idx]), 2),
                 color=fac * color_a + (1 - fac) * color_b)

    # plots source and receiver
    plt.plot(pos_source[0], pos_source[1], marker='+', markerfacecolor='red', markersize=8, markeredgecolor='red',
             markeredgewidth=5)
    # plt.plot(pos_receiver[0], pos_receiver[1], marker='x', markerfacecolor='darkgreen', markersize=10,
    #          markeredgecolor='darkgreen', markeredgewidth=5)

    # final plotting setup
    plt.xlim((-.05 * width, width + .05 * width))
    plt.ylim((-.05 * height, height + .05 * height))
    # plt.legend(loc='upper left')
    plt.show()

