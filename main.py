from py_params import *
from class_Ray import Ray
from class_VelocityField import Environment, EstEnvironment
import sys
from fun_encode64 import encode64, mhash


np.set_printoptions(legacy='1.25',precision=2,linewidth=600,threshold=sys.maxsize)
decimal.getcontext().prec = 2


def main(num_cells_x, num_cells_y, width, height, pos_receivers_x, pos_receivers_y, pos_sources_x, pos_sources_y,mirrored,alpha=0.05,epsilon=0.1,show_iterations=False):
    # num_cells = num_cells_x * num_cells_y
    # num_cols = 1
    # pos_receivers_x = np.array([0.95*width for _ in range(n_receivers)])
    # pos_receivers_y = np.linspace(0.95*height, 0.15*height, n_receivers)
    #
    # # pos_sources_x = np.linspace(0.05*width, 0.85*width, n_sources)
    # # pos_sources_y = np.array([0.05*height for _ in range(n_sources)])
    # pos_sources_x = np.array([0.05*width for _ in range(n_sources)])
    # pos_sources_y = np.linspace(0.95*height, 0.05*height, n_sources)
    parameters = {
        'num_cells_x': num_cells_x,
        'num_cells_y': num_cells_y,
        'width': width,
        'height': height,
        'pos_receivers_x': pos_receivers_x,
        'pos_receivers_y': pos_receivers_y,
        'pos_sources_x': pos_sources_x,
        'pos_sources_y': pos_sources_y,
        'mirrored': mirrored,
        'alpha': alpha,
        'epsilon': epsilon
    }

    hash_val = mhash(parameters.values())
    code = encode64(hash_val)
    direc = 'Results/'
    os.makedirs(direc, exist_ok=True)
    os.makedirs(direc + code, exist_ok=True)
    parameters['code'] = code
    parameters_text = '\n'.join([key + ': ' + str(parameters[key]) for key in parameters.keys()])
    with open(direc + code + '/simulation parameters.txt', 'w') as f:
        f.write(parameters_text)
    print(code)

    show_path = False
    if mirrored:
        pos_receivers_x = width - pos_receivers_x
        pos_sources_x = width - pos_sources_x
    pos_receivers = list(zip(pos_receivers_x, pos_receivers_y))
    pos_sources = list(zip(pos_sources_x, pos_sources_y))

    export_pos_devices(pos_sources, pos_receivers, direc + code + '/')
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

    idx = 1
    vmin = np.inf
    vmax = -1
    obs_environment.field_to_csv(0, export_params=True, code=code)
    while True:
        est_environment_lit.iterate_field(est_rays_lit, n_rays, obs_times, method='literature', alpha=alpha)
        est_environment_prp.iterate_field(est_rays_prp, n_rays, obs_times, method='proposed', epsilon=epsilon)
        print("FIELD ITERATED")
        # show_figure([(obs_environment, obs_rays),
        #              (est_environment_lit, est_rays_lit),
        #              (est_environment_prp, est_rays_prp)], pos_receivers, pos_sources, title='Field Iterated',
        #             show_path=show_path)

        for ray in est_rays_lit:
            ray.calc_path(est_environment_lit)
            ray.calc_time(est_environment_lit)
        for ray in est_rays_prp:
            ray.calc_path(est_environment_prp)
            ray.calc_time(est_environment_prp)

        print("RAYS ITERATED")
        est_environment_prp.field_to_csv(idx, code=code)
        est_environment_lit.field_to_csv(idx, code=code)
        est_environment_prp.field_to_csv(idx,comp=obs_environment.field, code=code)
        est_environment_lit.field_to_csv(idx,comp=obs_environment.field, code=code)
        vmin = min(vmin,
                   np.amin(np.abs(est_environment_prp.field - obs_environment.field)),
                   np.amin(np.abs(est_environment_lit.field - obs_environment.field)))
        vmax = max(vmax,
                   np.amax(np.abs(est_environment_prp.field - obs_environment.field)),
                   np.amax(np.abs(est_environment_lit.field - obs_environment.field)))
        obs_environment.field_to_csv(0, export_params=True, vmin=vmin, vmax=vmax, code=code)
        est_times_lit = [ray.time for ray in est_rays_lit]
        est_times_prp = [ray.time for ray in est_rays_prp]
        err_time_lit = est_environment_lit.cost_function(est_rays_lit, obs_times)
        err_time_prp = est_environment_prp.cost_function(est_rays_prp, obs_times)
        print('\tEst. times - Lit.: ' + ' '.join(['{:.4f}'.format(time) for time in est_times_lit]))
        print('\tEst. times - Prp.: ' + ' '.join(['{:.4f}'.format(time) for time in est_times_prp]))
        print('\tObs. times:        ' + ' '.join(['{:.4f}'.format(time) for time in obs_times]))
        print('\tError - Lit. - {:.4f}'.format(np.sum(err_time_lit)))
        print('\tError - Prp. - {:.4f}'.format(np.sum(err_time_prp)))
        print(idx, est_environment_lit.cost_function(est_rays_lit, obs_times), 0.01*float(np.mean(obs_times)))
        show_figure([(obs_environment, obs_rays),
                     (est_environment_lit,est_rays_lit),
                     (est_environment_prp,est_rays_prp)], pos_receivers, pos_sources, title='Rays Iterated', show_path=show_path)
        if all([ray.converged for ray in est_rays_lit]) and est_environment_lit.cost_function(est_rays_lit, obs_times) < 0.01*float(np.mean(obs_times)):
            print("ALL CONVERGED")
        if idx == 6:
            print("LIMIT REACHED")
            break
        idx += 1


def show_figure(pairs: list[tuple],
                pos_receivers: list[tuple[float, float]],
                pos_sources: list[tuple[float, float]],
                show_path: bool = False,
                legend: bool = False,
                title: str = None):

    fig, axs = plt.subplots(1, len(pairs)+1, sharey=True, sharex=True)
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


def export_pos_devices(pos_sources, pos_receivers, direc):
    header = 'x_pos,y_pos'

    txt_sources = '\n'.join([header] + ['{},{}'.format(*pos) for pos in pos_sources])
    txt_receivers = '\n'.join([header] + ['{},{}'.format(*pos) for pos in pos_receivers])

    with open(direc + '/pos_sources.csv', 'w') as f:
        f.write(txt_sources)
        f.close()
    with open(direc + '/pos_receivers.csv', 'w') as f:
        f.write(txt_receivers)
        f.close()
    pass


if __name__ == '__main__':
    nx = 10
    ny = 15
    w = 5000
    h = 5000
    ns = 3
    nr = 8

    pos_rx = np.linspace(0.95*w, 0.15*w, nr)
    pos_ry = np.array([0.95*h for _ in range(nr)])

    pos_sx = np.linspace(0.95*w, 0.05*w, ns)
    pos_sy = np.array([0.05*h for _ in range(ns)])
    alpha = 0.05
    epsilon = 0.1

    np.set_printoptions(legacy='1.25', precision=6, linewidth=320)
    decimal.getcontext().prec = 2
    main(nx, ny, w, h, pos_rx, pos_ry, pos_sx, pos_sy, True, alpha, epsilon, show_iterations=True)
