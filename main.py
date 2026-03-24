from fun_device_positioning import position_devices
from fun_export_pos_devices import export_pos_devices
from fun_show_figure import show_figure
from py_libs import *
from class_Environment import Environment, EstEnvironment
import sys
from fun_encode64 import mhash, encode16
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
def main(num_cells_x, num_cells_y, width, height, dev_positioning, mirrored,
         param_bas=(0.05,), param_trd=(0.05, 0.05, 0), param_spl=(0.05, 0), sigma=0.0, show_path=False, percent_rays=100):
    pos_sources_x, pos_sources_y, pos_receivers_x, pos_receivers_y, pos_name = dev_positioning

    ## ---------------------------
    ## Hash and folder creation
    ## ---------------------------

    parameters = {
        'pos_name': pos_name,
        'num_cells_x': num_cells_x,
        'num_cells_y': num_cells_y,
        'width': width,
        'height': height,
        'pos_receivers_x': pos_receivers_x,
        'pos_receivers_y': pos_receivers_y,
        'pos_sources_x': pos_sources_x,
        'pos_sources_y': pos_sources_y,
        'mirrored': mirrored,
        'param_bas': param_bas,
        'param_trd': param_trd,
        'param_spl': param_spl,
    }

    hash_val = mhash(parameters.values())
    code = encode16(hash_val)
    direc = 'Results/'
    os.makedirs(direc, exist_ok=True)
    os.makedirs(direc + code, exist_ok=True)

    ## ---------------------------
    ## Sources, receivers, and environments
    ## ---------------------------

    pos_receivers = list(set(zip(pos_receivers_x, pos_receivers_y)))
    pos_sources = list(set(zip(pos_sources_x, pos_sources_y)))
    export_pos_devices(pos_sources, pos_receivers, direc + code + '/', width, height)

    obs_env = Environment(num_cells_x, num_cells_y, width, height, None, mirrored=False)
    obs_env.generate_rays(pos_sources, pos_receivers)

    est_envs = []
    for field_name in ['Basic', 'Trade', 'Split']:
        est_env = EstEnvironment(num_cells_x, num_cells_y, width, height, None, field_name=field_name)
        est_env.generate_rays(pos_sources, pos_receivers)
        est_envs.append(est_env)

    ## ---------------------------
    ## Other simulation parameters
    ## ---------------------------

    obs_times = [ray.time for ray in obs_env.rays]
    mean_time = np.mean(obs_times)
    rng = np.random.default_rng(0)
    noise_times = list(sigma*mean_time*rng.random((len(obs_env.rays),)))
    new_obs_times = [obs_times[i] + noise_times[i] for i in range(len(obs_env.rays))]
    obs_times = new_obs_times
    n_rays = len(obs_times)
    model_slowness = 1/obs_env.vy.reshape(-1, 1)

    parameters['code'] = code
    parameters['n_rays'] = n_rays
    parameters_text = '\n'.join([key + ': ' + str(parameters[key]) for key in parameters.keys()])
    with open(direc + code + '/simulation parameters.txt', 'w') as f:
        f.write(parameters_text)

    # obs_env.plot_curves(obs_env.rays, pos_sources, pos_receivers)
    # plt.show()

    it_idx = 1
    vmin = np.inf
    vmax = -1
    obs_env.field_to_csv(0, export_params=True, code=code)
    print('\n'*10)
    while True:
        print("Iteration {}".format(it_idx))
        ## ---------------------------
        ## Iterate estimated fields
        ## ---------------------------
        est_envs[0].iterate_field(n_rays, alpha=param_bas[0], obs_times=obs_times,
                                  mode='classic')
        est_envs[1].iterate_field(n_rays, alpha=param_trd[0], beta=param_trd[1], obs_times=obs_times,
                                  model_slowness=model_slowness, masking_depth=param_trd[2],
                                  mode='trade')
        est_envs[2].iterate_field(n_rays, alpha=param_spl[0], obs_times=obs_times,
                                  model_slowness=model_slowness, masking_depth=param_spl[1],
                                  mode='split')
        # print("FIELD ITERATED")

        ## ---------------------------
        ## Calc metrics #todo: reimplement: field_to_csv generation; vmin, vmax calculation
        ## ---------------------------
        for est_env in est_envs:
            est_env.calc_metrics(obs_times, obs_env.field)

        err_times = []
        for est_env in est_envs:
            est_env.field_to_csv(it_idx, code=code)
            est_env.field_to_csv(it_idx,comp=obs_env.field, code=code,vmin=-20,vmax=20, export_params=True)
            est_env.export_metrics(code=code)

            err_times.append(est_env.cost_function(est_env.rays, obs_times))


        if it_idx == 5:
            print("LIMIT REACHED - Code " + code)
            break
        it_idx += 1

    # err_time_bas = est_env_bas.cost_function(est_env_bas.rays, obs_times)
    print('Travel-time error:')
    for env_idx, est_env in enumerate(est_envs):
        print('\t{}. - {:.4f}ms'.format(est_env.field_name[:3], 1000 * est_env.est_time_mse[-1]),
              'out of {:.4f}s'.format(np.mean(obs_times)))
    print('Field RMS error:')
    for env_idx, est_env in enumerate(est_envs):
        print('\t{}. - {:.4f}m/s'.format(est_env.field_name[:3], est_env.est_ssf_rms[-1]),
              'with σ = {:.4f}m/s'.format(np.std(obs_env.field)))
    print()

    # show_figure([obs_env] + est_envs, pos_receivers, pos_sources, title='Field Iterated',
    #             show_path=show_path, comp_field=True)


if __name__ == '__main__':
    nx = 43
    ny = 50
    w = 2000
    h = 2000
    ns = 15
    nr = 15
    n_travels = 5
    ofs = 0
    params = nx, ny, w, h, ns, nr, ofs, n_travels

    for device_pos in range(4):
    # for device_pos in [2]:
        dev_positioning = position_devices(device_pos, params)

        p_bas = [0.95]
        p_trd = [0.85, 0.45, 1000]
        p_spl = [0.85, 1000]
        sig = 0.000

        np.set_printoptions(legacy='1.25', precision=6, linewidth=320)
        decimal.getcontext().prec = 2
        main(nx, ny, w, h, dev_positioning, False, p_bas, p_trd, p_spl, sig, show_path=False)
