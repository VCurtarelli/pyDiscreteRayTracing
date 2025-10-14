from py_libs import *
from fun_receiver_search import receiver_search
from fun_ray_tracing import ray_tracing
# from py_params import velocity_field


def plot_curves(velocity_field, width, height, num_cells_x, num_cells_y, n_receivers, pos_source, pos_receivers,
                ax=None, show_field=None, show_search=False, color=None):
    if ax is None:
        fig, ax = plt.subplots()
    if show_field is None:
        show_field = velocity_field
    if color is None:
        color = np.array([0, 180, 0]) / 255

    if not (show_field is False):
        ax.imshow(show_field, extent=(0, width, 0, height),alpha=0.2)
        ax.vlines(x=np.linspace(0, width, num_cells_x + 1), ymin=0, ymax=height, color='gray', linestyle='dashed',
                          linewidth=1)
        ax.hlines(y=np.linspace(0, height, num_cells_y + 1), xmin=0, xmax=height, color='gray', linestyle='dashed',
                          linewidth=1)
    ray_angles = []
    for idx in range(n_receivers):
        pos_receiver = pos_receivers[idx]
        theta_0 = np.angle(complex(pos_receiver[0], pos_receiver[1]))  # initial guess for source-to-receiver ray's emission angle
        angles, paths = receiver_search(theta_0, velocity_field, num_cells_x, num_cells_y, width, height,
                                    pos_source,
                                    pos_receiver)
        angle = angles[-1]
        ray_angles.append(angle)
        if show_search:
            for path in paths:
                path = np.array(path)
                ax.plot(path[:, 0], path[:, 1], color=[0.5,0.5,0.5], alpha=0.2)

    for idx in range(n_receivers):
        angle = ray_angles[idx]
        pos_receiver = pos_receivers[idx]
        _, _, path, _ = ray_tracing(angle, velocity_field, num_cells_x, num_cells_y, width, height,
                                              pos_source, )
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], label=np.around(np.rad2deg(angle), 2),
                    color=color)
        if not (show_field is False):
            ax.plot(pos_receiver[0], pos_receiver[1], marker='x', markerfacecolor='darkgreen', markersize=10,
                    markeredgecolor='darkgreen', markeredgewidth=5)
    if not (show_field is False):
        ax.plot(pos_source[0], pos_source[1], marker='+', markerfacecolor='red', markersize=8,
                    markeredgecolor='red',
                    markeredgewidth=5)

    # final plotting setup
    ax.set_xlim((-.05 * width, width + .05 * width))
    ax.set_ylim((-.05 * height, height + .05 * height))
    # ax.legend(loc='upper left')