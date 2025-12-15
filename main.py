from py_libs import *
from class_Ray import Ray
from class_Environment import Environment, EstEnvironment
import sys
from fun_encode64 import encode64, mhash, encode16
import time


class Timer:
    def __init__(self):
        self.times = {'start': time.time()}
        self.delta_times = {}
        self.key_max_length = 5

    def time(self, name=None):
        if name is None:
            name = str(len(self.times.keys()))
        if name in self.times.keys():
            name = '{} '.format(str(len(self.times.keys()))) + name
        if not isinstance(name, str):
            name = str(name)
        new_time = time.time()
        prev_time = list(self.times.values())[-1]
        self.times[name] = new_time
        self.delta_times[name] = new_time - prev_time
        self.key_max_length = max(len(name), self.key_max_length)

    def show_times(self):
        for key in self.times.keys():
            print('{}: {}'.format(key.ljust(self.key_max_length), self.times[key]))

    def show_delta_times(self):
        for key in self.delta_times.keys():
            print('{}: {}ms'.format(key.ljust(self.key_max_length+1), '{:.2f}'.format(1000*self.delta_times[key]).rjust(8)))


np.set_printoptions(legacy='1.25',precision=2,linewidth=600,threshold=sys.maxsize)
decimal.getcontext().prec = 2

rms = lambda x: np.sqrt(np.mean(x**2))
def main(num_cells_x, num_cells_y, width, height, pos_receivers_x, pos_receivers_y, pos_sources_x, pos_sources_y, mirrored,
         alpha=0.05, epsilon=0.1, sigma=0.0, show_path=False, percent_rays=100):
    timer = Timer()

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
    code = encode16(hash_val)
    direc = 'Results/'
    os.makedirs(direc, exist_ok=True)
    os.makedirs(direc + code, exist_ok=True)

    # timer.time('Folder and hash generation')

    show_path = show_path
    if mirrored:
        pos_receivers_x = width - pos_receivers_x
        pos_sources_x = width - pos_sources_x
    pos_receivers = list(set(zip(pos_receivers_x, pos_receivers_y)))
    pos_sources = list(set(zip(pos_sources_x, pos_sources_y)))
    # pos_receivers = pos_receivers + pos_sources
    # pos_sources = pos_receivers

    export_pos_devices(pos_sources, pos_receivers, direc + code + '/')

    # timer.time('Devices positioning')
    obs_rays = []
    for idx_s, source in enumerate(pos_sources):
        for idx_r, receiver in enumerate(pos_receivers):
            if source == receiver:
                continue
            obs_rays.append(Ray(source, receiver, 'observed', marker='x'))

    obs_environment = Environment(num_cells_x, num_cells_y, width, height, obs_rays, mirrored=False)
    obs_environment.update_rays()

    # timer.time('Create obs_environment')

    obs_times = [ray.time for ray in obs_rays]
    mean_time = np.mean(obs_times)
    rng = np.random.default_rng(0)
    noise_times = list(sigma*mean_time*rng.random((len(obs_rays),)))
    new_obs_times = [obs_times[i] + noise_times[i] for i in range(len(obs_rays))]
    obs_times = new_obs_times
    n_rays = len(obs_times)
    print(n_rays)

    parameters['code'] = code
    parameters['n_rays'] = n_rays
    parameters_text = '\n'.join([key + ': ' + str(parameters[key]) for key in parameters.keys()])
    with open(direc + code + '/simulation parameters.txt', 'w') as f:
        f.write(parameters_text)
    print(code)

    # timer.time('Create obs_times')

    est_rays_lit = []
    est_rays_prp = []
    for idx, source in enumerate(pos_sources):
        for jdx, receiver in enumerate(pos_receivers):
            if source == receiver:
                continue
            ray = Ray(source, receiver, 'Literature', color=(180, 180, 0), marker='1')
            est_rays_lit.append(ray)
            ray = Ray(source, receiver, 'Proposed', color=(0, 180, 180), marker='2')
            est_rays_prp.append(ray)
    est_environment_lit = EstEnvironment(num_cells_x, num_cells_y, width, height, est_rays_lit, field_name='Literature')
    est_environment_prp = EstEnvironment(num_cells_x, num_cells_y, width, height, est_rays_prp, field_name='Proposed')
    est_environment_lit.update_rays()
    est_environment_prp.update_rays()

    # timer.time("Create est_environment's")

    # timer.show_delta_times()

    # show_figure([(obs_environment, obs_rays),
    #              (est_environment_lit, est_rays_lit),
    #              (est_environment_prp, est_rays_prp)], pos_receivers, pos_sources, title='Rays Iterated',
    #             show_path=show_path)

    idx = 1
    vmin = np.inf
    vmax = -1
    obs_environment.field_to_csv(0, export_params=True, code=code)
    print('\n'*10)
    param=1
    while True:
        print("Iteration:", idx, "\tEps.:", np.around(epsilon*(param**idx), 4))
        est_environment_lit.iterate_field(est_rays_lit, n_rays, obs_times, method='literature', alpha=alpha,
                                          model=np.ones(obs_environment.field.size) / np.mean(obs_environment.field))
        est_environment_prp.iterate_field(est_rays_prp, n_rays, obs_times, method='proposed', epsilon=epsilon*(param**idx),
                                          model=np.ones(obs_environment.field.size) / np.mean(obs_environment.field))
        # print("FIELD ITERATED")
        # show_figure([(obs_environment, obs_rays),
        #              (est_environment_lit, est_rays_lit),
        #              (est_environment_prp, est_rays_prp)], pos_receivers, pos_sources, title='Field Iterated',
        #             show_path=show_path)

        est_environment_lit.update_rays()
        est_environment_prp.update_rays()

        # print("RAYS ITERATED")
        est_environment_lit.calc_metrics(obs_times, obs_environment.field)
        est_environment_prp.calc_metrics(obs_times, obs_environment.field)


        vmin = min(vmin,
                   np.amin(np.abs(est_environment_prp.field - obs_environment.field)),
                   np.amin(np.abs(est_environment_lit.field - obs_environment.field)))
        vmax = max(vmax,
                   np.amax(np.abs(est_environment_prp.field - obs_environment.field)),
                   np.amax(np.abs(est_environment_lit.field - obs_environment.field)))

        est_environment_prp.field_to_csv(idx, code=code)
        est_environment_lit.field_to_csv(idx, code=code)
        est_environment_prp.field_to_csv(idx,comp=obs_environment.field, code=code)
        est_environment_lit.field_to_csv(idx,comp=obs_environment.field, code=code)
        obs_environment.field_to_csv(0, export_params=True, vmin=vmin, vmax=vmax, code=code)

        est_environment_prp.export_metrics(code=code)
        est_environment_lit.export_metrics(code=code)

        err_time_lit = est_environment_lit.cost_function(est_rays_lit, obs_times)
        err_time_prp = est_environment_prp.cost_function(est_rays_prp, obs_times)
        # print('\tEst. times - Lit.: ' + ' '.join(['{:.4f}'.format(time) for time in est_times_lit]))
        # print('\tEst. times - Prp.: ' + ' '.join(['{:.4f}'.format(time) for time in est_times_prp]))
        # print('\tObs. times:        ' + ' '.join(['{:.4f}'.format(time) for time in obs_times]))
        print('Travel-time error:')
        print('\tLit. - {:.4f}ms'.format(1000*err_time_lit), 'out of {:.4f}s'.format(np.mean(obs_times)))
        print('\tPrp. - {:.4f}ms'.format(1000*err_time_prp), 'out of {:.4f}s'.format(np.mean(obs_times)))
        print('Field RMS error:')
        print('\tLit. - {:.4f}m/s'.format(rms(est_environment_lit.field - obs_environment.field)), 'with σ = {:.4f}m/s'.format(np.std(obs_environment.field)))
        print('\tPrp. - {:.4f}m/s'.format(rms(est_environment_prp.field - obs_environment.field)), 'with σ = {:.4f}m/s'.format(np.std(obs_environment.field)))
        print()
        # print(idx, est_environment_lit.cost_function(est_rays_lit, obs_times), 0.01*float(np.mean(obs_times)))


        if idx == 20:
            print("LIMIT REACHED - Code " + code)
            break
        idx += 1

    show_figure([(obs_environment, obs_rays),
                 (est_environment_lit,est_rays_lit),
                 (est_environment_prp,est_rays_prp)], pos_receivers, pos_sources, title='Rays Iterated', show_path=show_path)


def show_figure(pairs: list[tuple],
                pos_receivers: list[tuple[float, float]],
                pos_sources: list[tuple[float, float]],
                show_path: bool = False,
                legend: bool = False,
                title: str = None):

    fig, axs = plt.subplots(2, 3, sharey=True, sharex=True, figsize=(10,8))
    axs = axs.reshape(-1,)
    standard_field = pairs[0][0].field
    vmin = np.amin(standard_field)
    vmax = np.amax(standard_field)
    all_rays = []
    for idx, pair in enumerate(pairs):
        environment, rays = pair
        environment.plot_curves(rays, pos_sources, pos_receivers, axs[idx], show_path=show_path, legend=legend,
                                vs=(vmin, vmax)
                                )
        axs[idx].set_xlabel(environment.field_name)
        all_rays += rays
    # pairs[0][0].plot_curves(all_rays, pos_sources, pos_receivers, axs[-1], cmap='magma')
    vmin = min([np.amin(np.abs(env.field - standard_field)) for env, _ in pairs[1:]])
    vmax = max([np.amax(np.abs(env.field - standard_field)) for env, _ in pairs[1:]])
    for idx, pair in enumerate(pairs[1:]):
        idx = idx+4
        environment, rays = pair
        environment.plot_curves(rays,pos_sources,pos_receivers,axs[idx], show_path=show_path,legend=legend,
                                vs=(vmin, vmax), show_field=np.abs(environment.field - standard_field),
                                cmap='magma')
        axs[idx].set_xlabel('Comp. '+environment.field_name)
    # axs[-1].set_xlabel('Comparison')
    if title is not None:
        fig.suptitle(title)
    fig.delaxes(axs[3])
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("1000x700+50+50")
    # manager.full_screen_toggle()
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
    ny = 30
    w = 2000
    h = 2000
    ns = 10
    nr = 10
    nr_ = 5

    sources = receivers = 'mixed'

    pos_sx = []
    pos_sy = []
    pos_rx = []
    pos_ry = []


    sr_method = 'new'
    if sr_method == 'new':
        ofs = 0
        pos_sx.append(np.linspace(ofs, nx - 1 - ofs, ns))
        pos_sy.append(np.array([ofs for _ in range(ns)]))

        pos_rx.append(np.linspace(ofs, nx - 1 - ofs, nr))
        pos_ry.append(np.array([ny - 1 - ofs for _ in range(nr)]))
        pos_rx.append(np.array([nx-1 - ofs for _ in range(nr_+2)])[1:-1])
        pos_ry.append(np.linspace(ofs, ny - 1 - ofs, nr_+2)[1:-1])
        pos_rx.append(np.array([ofs for _ in range(nr_+2)])[1:-1])
        pos_ry.append(np.linspace(ofs, ny - 1 - ofs, nr_+2)[1:-1])
        # pos_rx.append(np.linspace(0, nx - 1, nr)[1:-1])
        # pos_ry.append(np.array([0.5*ny - 1 for _ in range(nr)])[1:-1])


        pos_sx = np.concatenate(pos_sx)
        pos_sy = np.concatenate(pos_sy)
        pos_rx = np.concatenate(pos_rx)
        pos_ry = np.concatenate(pos_ry)

        rng = np.random.default_rng(2)
        cw = w / nx
        ch = h / ny
        pos_sx = cw * (0.5 + np.around(pos_sx))
        pos_sy = ch * (0.5 + np.around(pos_sy))
        pos_rx = cw * (0.5 + np.around(pos_rx))
        pos_ry = ch * (0.5 + np.around(pos_ry))
    else:
        if sources == 'mixed':
            ns = int(np.ceil(ns/2))
        if receivers == 'mixed':
            nr = int(np.ceil(nr/2))
        if sources in ('horizontal', 'mixed'):
            pos_sx.append(np.linspace(0.15*w, 0.95*w, ns))
            pos_sy.append(np.array([0.05*h for _ in range(ns)]))
        if sources in ('vertical', 'mixed'):
            pos_sx.append(np.array([0.95*w for _ in range(ns)]))
            pos_sy.append(np.linspace(0.15*h, 0.95*h, ns))

        if receivers in ('horizontal', 'mixed'):
            pos_rx.append(np.linspace(0.05*w, 0.8*w, nr))
            pos_ry.append(np.array([0.95*h for _ in range(nr)]))
        if receivers in ('vertical', 'mixed'):
            pos_rx.append(np.array([0.05*w for _ in range(nr)]))
            pos_ry.append(np.linspace(0.05*h, 0.8*h, nr))

        ns = len(pos_sx)
        nr = len(pos_rx)

        pos_sx = np.concatenate(pos_sx)
        pos_sy = np.concatenate(pos_sy)
        pos_rx = np.concatenate(pos_rx)
        pos_ry = np.concatenate(pos_ry)

        rng = np.random.default_rng(2)
        cw = w/nx
        ch = h/ny
        pos_sx = cw/2 + cw*np.floor((pos_sx + rng.normal(0, 1, pos_sx.shape) * 0.005 * w)/cw)
        pos_sy = ch/2 + ch*np.floor((pos_sy + rng.normal(0, 1, pos_sy.shape) * 0.005 * h)/ch)
        pos_rx = cw/2 + cw*np.floor((pos_rx + rng.normal(0, 1, pos_rx.shape) * 0.005 * w)/cw)
        pos_ry = ch/2 + ch*np.floor((pos_ry + rng.normal(0, 1, pos_ry.shape) * 0.005 * h)/ch)
    pos_sx, pos_sy = tuple(np.array(list(set(zip(list(pos_sx), list(pos_sy))))).T)
    pos_rx, pos_ry = tuple(np.array(list(set(zip(list(pos_rx), list(pos_ry))))).T)
    alp = 0.98
    eps = 0.2
    sig = 0.000

    np.set_printoptions(legacy='1.25', precision=6, linewidth=320)
    decimal.getcontext().prec = 2
    main(nx, ny, w, h, pos_rx, pos_ry, pos_sx, pos_sy, False, alp, eps, sig, show_path=False)
