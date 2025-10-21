from py_params import *
from class_Ray import Ray
from class_VelocityField import Environment, EstEnvironment
import sys


np.set_printoptions(legacy='1.25',precision=2,linewidth=600,threshold=sys.maxsize)
decimal.getcontext().prec = 2


def main(num_cells_x, num_cells_y, width, height, n_receivers, n_sources, show_iterations=False):
    # num_cells = num_cells_x * num_cells_y
    # num_cols = 1
    pos_receivers_x = np.array([0.95*width for _ in range(n_receivers)])
    pos_receivers_y = np.linspace(0.95*height, 0.15*height, n_receivers)

    # pos_sources_x = np.linspace(0.05*width, 0.85*width, n_sources)
    # pos_sources_y = np.array([0.05*height for _ in range(n_sources)])
    pos_sources_x = np.array([0.05*width for _ in range(n_sources)])
    pos_sources_y = np.linspace(0.95*height, 0.05*height, n_sources)

    show_path = False
    mirrored = True
    if mirrored:
        pos_receivers_x = width - pos_receivers_x
        pos_sources_x = width - pos_sources_x
    pos_receivers = list(zip(pos_receivers_x, pos_receivers_y))
    pos_sources = list(zip(pos_sources_x, pos_sources_y))

    # pos_sources = [pos_sources[0]]
    # pos_receivers = [pos_receivers[0]]

    obs_rays = []
    for source in pos_sources:
        for receiver in pos_receivers:
            obs_rays.append(Ray(source, receiver, 'observed'))

    obs_environment = Environment(num_cells_x, num_cells_y, width, height, mirrored=True)

    # rays = [rays[3]]
    for ray in obs_rays:
        ray.calc_path(obs_environment)
        ray.calc_time(obs_environment)
    # obs_environment.plot_curves(obs_rays, pos_sources, pos_receivers, show_path=show_path)
    # plt.show()

    obs_times = [ray.time for ray in obs_rays if ray.converged]
    n_rays = len(obs_times)

    est_environment_lit = EstEnvironment(num_cells_x, num_cells_y, width, height, field_name='Literature')
    est_environment_prp = EstEnvironment(num_cells_x, num_cells_y, width, height, field_name='Proposed')
    est_rays_lit = []
    est_rays_prp = []
    for idx, source in enumerate(pos_sources):
        for jdx, receiver in enumerate(pos_receivers):
            if obs_rays[len(pos_receivers) * idx + jdx].converged:
                ray = Ray(source, receiver, 'Literature', color=(180, 180, 0))
                est_rays_lit.append(ray)
                ray.calc_path(est_environment_lit)
                ray.calc_time(est_environment_lit)
                ray = Ray(source, receiver, 'Proposed', color=(180, 0, 0))
                est_rays_prp.append(ray)
                ray.calc_path(est_environment_prp)
                ray.calc_time(est_environment_prp)

    idx = 0
    while True:
        est_environment_lit.iterate_field(est_rays_lit, n_rays, obs_times, method='literature', alpha=0.1)
        est_environment_prp.iterate_field(est_rays_prp, n_rays, obs_times, method='proposed', epsilon=0.1)
        print("FIELD ITERATED")
        new_show_figure([(obs_environment, obs_rays),
                         (est_environment_lit, est_rays_lit),
                         (est_environment_prp, est_rays_prp)], pos_receivers, pos_sources, title='Field Iterated',
                        show_path=show_path)

        for ray in est_rays_lit:
            ray.calc_path(est_environment_lit)
            ray.calc_time(est_environment_lit)
        for ray in est_rays_prp:
            ray.calc_path(est_environment_prp)
            ray.calc_time(est_environment_prp)

        print("RAYS ITERATED")
        est_environment_prp.field_to_csv(idx)
        est_environment_lit.field_to_csv(idx)
        obs_environment.field_to_csv(idx, export_params=True)
        est_times_lit = [ray.time for ray in est_rays_lit]
        est_times_prp = [ray.time for ray in est_rays_prp]
        err_time_lit = est_environment_lit.cost_function(est_rays_lit, obs_times)
        err_time_prp = est_environment_prp.cost_function(est_rays_prp, obs_times)
        print('\tEst. times - Lit.: ' + ' '.join(['{:.4f}'.format(time) for time in est_times_lit]))
        print('\tEst. times - Prp.: ' + ' '.join(['{:.4f}'.format(time) for time in est_times_prp]))
        print('\tObs. times:        ' + ' '.join(['{:.4f}'.format(time) for time in obs_times]))
        print('\tError - Lit. - {:.4f}'.format(np.sum(err_time_lit)))
        print('\tError - Prp. - {:.4f}'.format(np.sum(err_time_prp)))
        # print(idx, est_environment_lit.cost_function(est_rays_lit, obs_times), 0.01*float(np.mean(obs_times)))
        new_show_figure([(obs_environment, obs_rays),
                         (est_environment_lit,est_rays_lit),
                         (est_environment_prp,est_rays_prp)], pos_receivers, pos_sources, title='Rays Iterated', show_path=show_path)
        if all([ray.converged for ray in est_rays_lit]) and est_environment_lit.cost_function(est_rays_lit, obs_times) < 0.01*float(np.mean(obs_times)):
            print("ALL CONVERGED")
        if idx == 100:
            print("LIMIT REACHED")
            break
        idx += 1


def show_figure(est_rays: list[Ray], est_velocity_field: EstEnvironment,
                pos_receivers: list[tuple[float, float]],
                pos_sources: list[tuple[float, float]], rays: list[Ray], velocity_field: Environment,
                show_path: bool = False, legend: bool = False, title: str = None):
    fig, axs = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(12, 9))
    vmin = np.amin(velocity_field.field) - 0.1*np.std(velocity_field.field)
    vmax = np.amax(velocity_field.field) + 0.1*np.std(velocity_field.field)
    est_velocity_field.plot_curves(est_rays, pos_sources, pos_receivers, axs[0], show_path=show_path, legend=legend, vs=(vmin, vmax))
    velocity_field.plot_curves(rays, pos_sources, pos_receivers, axs[1], show_path=show_path, legend=legend, vs=(vmin, vmax))
    velocity_field.plot_curves(rays + est_rays, pos_sources, pos_receivers, axs[2], show_field=np.abs(velocity_field.field - est_velocity_field.field), cmap='magma')
    for ax in axs:
        ax.set_box_aspect(1)
    axs[0].set_xlabel('Estimate')
    axs[1].set_xlabel('Real')
    axs[2].set_xlabel('Comparison')
    if title is not None:
        fig.suptitle(title)

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()

def new_show_figure(pairs: list[tuple],
                pos_receivers: list[tuple[float, float]],
                pos_sources: list[tuple[float, float]],
                show_path: bool = False,
                legend: bool = False,
                title: str = None):

    fig, axs = plt.subplots(1, len(pairs)+1, sharey=True, sharex=True, figsize=(12, 9))
    standard_field = pairs[0][0].field
    vmin = np.amin(standard_field) - 0.1*np.std(standard_field)
    vmax = np.amax(standard_field) + 0.1*np.std(standard_field)
    all_rays = []
    for idx, pair in enumerate(pairs):
        environment, rays = pair
        environment.plot_curves(rays, pos_sources, pos_receivers, axs[idx], show_path=show_path, legend=legend, vs=(vmin, vmax))
        axs[idx].set_xlabel(environment.field_name)
        all_rays += rays

    pairs[0][0].plot_curves(all_rays, pos_sources, pos_receivers, axs[-1], cmap='magma')
    axs[-1].set_xlabel('Comparison')
    if title is not None:
        fig.suptitle(title)

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()


if __name__ == '__main__':
    nx = 10
    ny = 15
    w = 5000
    h = 5000
    ns = 3
    nr = 8
    np.set_printoptions(legacy='1.25', precision=6, linewidth=320)
    decimal.getcontext().prec = 2
    main(nx, ny, w, h, nr, ns, show_iterations=True)
