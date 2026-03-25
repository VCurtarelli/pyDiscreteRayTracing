from fun_device_positioning import position_devices
from fun_export_pos_devices import export_pos_devices
from fun_show_figure import show_figure
from py_libs import *
from class_Environment import Environment, EstEnvironment
import sys
from fun_encode64 import mhash, encode16
import time
from dataclasses import dataclass
from typing import Any
from numpy import ndarray


np.set_printoptions(legacy='1.25',precision=2,linewidth=600,threshold=sys.maxsize)
decimal.getcontext().prec = 2

rms = lambda x: np.sqrt(np.mean(x**2))
def main(env_params, dev_positioning,
         param_bas, param_trd, param_spl, sigma=0.0, show_path=False):

    ## Extract parameters from input
    num_cells_x = env_params.num_cells_x
    num_cells_y = env_params.num_cells_y
    width = env_params.width
    height = env_params.height
    pos_sources_x, pos_sources_y, pos_receivers_x, pos_receivers_y, pos_name = dev_positioning

    ## Extract sources and receivers, generate environments, and related variables
    pos_receivers = list(set(zip(pos_receivers_x, pos_receivers_y)))
    pos_sources = list(set(zip(pos_sources_x, pos_sources_y)))

    est_envs, obs_env = generate_environments(height, num_cells_x, num_cells_y, pos_receivers, pos_sources, width)
    model_slowness = 1/obs_env.vy.reshape(-1, 1)
    obs_times, n_rays = generate_observed_times(obs_env, sigma)

    ## Export device positions and simulation parameters
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
        'param_bas': param_bas,
        'param_trd': param_trd,
        'param_spl': param_spl,
    }

    code, direc = make_folders_and_hash(n_rays, parameters)
    export_pos_devices(pos_sources, pos_receivers, direc + code + '/', width, height)

    ## Main iteration loop
    iteration_loop(code, est_envs, model_slowness, n_rays, obs_env, obs_times, param_bas, param_spl, param_trd)

    ## Print metrics for on-execution assessment
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


def generate_environments(height, num_cells_x, num_cells_y, pos_receivers: list[tuple[Any, Any]],
                          pos_sources: list[tuple[Any, Any]], width) -> tuple[list[EstEnvironment], Environment]:
    obs_env = Environment(num_cells_x, num_cells_y, width, height)
    obs_env.generate_rays(pos_sources, pos_receivers)

    est_envs = []
    for field_name in ['Basic', 'Trade', 'Split']:
        est_env = EstEnvironment(num_cells_x, num_cells_y, width, height, None, field_name=field_name)
        est_env.generate_rays(pos_sources, pos_receivers)
        est_envs.append(est_env)
    return est_envs, obs_env


def iteration_loop(code: str, est_envs: list[Any], model_slowness: ndarray,
                   n_rays: int, obs_env: Environment, obs_times: list[Any], param_bas, param_spl, param_trd):
    it_idx = 1

    ## Export observed environment
    obs_env.field_to_csv(0, export_params=True, code=code)
    print('\n' * 10)
    while True:
        print("Iteration {}".format(it_idx))

        ## Iterate estimated fields
        est_envs[0].iterate_field(n_rays, alpha=param_bas.alpha, obs_times=obs_times,
                                  mode='classic')
        est_envs[1].iterate_field(n_rays, alpha=param_trd.alpha, beta=param_trd.beta, obs_times=obs_times,
                                  model_slowness=model_slowness, masking_depth=param_trd.masking_depth,
                                  mode='trade')
        est_envs[2].iterate_field(n_rays, alpha=param_spl.alpha, obs_times=obs_times,
                                  model_slowness=model_slowness, masking_depth=param_spl.masking_depth,
                                  mode='split')

        ## Calculate and export metrics, export estimated environment
        for est_env in est_envs:
            est_env.calc_metrics(obs_times, obs_env.field)

        err_times = []
        for est_env in est_envs:
            est_env.field_to_csv(it_idx, code=code)
            est_env.field_to_csv(it_idx, comp=obs_env.field, code=code, vmin=-20, vmax=20, export_params=True)
            est_env.export_metrics(code=code)

            err_times.append(est_env.cost_function(est_env.rays, obs_times))

        ## Loop end-condition
        if it_idx == 5:
            print("LIMIT REACHED - Code " + code)
            break
        it_idx += 1


def make_folders_and_hash(n_rays: int, parameters: dict) -> tuple[str, str]:
    hash_val = mhash(parameters.values())
    code = encode16(hash_val)
    direc = 'Results/'
    os.makedirs(direc, exist_ok=True)
    os.makedirs(direc + code, exist_ok=True)

    parameters['code'] = code
    parameters['n_rays'] = n_rays
    parameters_text = '\n'.join([key + ': ' + str(parameters[key]) for key in parameters.keys()])
    with open(direc + code + '/simulation parameters.txt', 'w') as f:
        f.write(parameters_text)
    return code, direc


def generate_observed_times(obs_env: Environment, sigma: float) -> tuple[list[Any], int]:
    obs_times = [ray.time for ray in obs_env.rays]
    mean_time = np.mean(obs_times)
    rng = np.random.default_rng(0)
    noise_times = list(sigma * mean_time * rng.random((len(obs_env.rays),)))
    new_obs_times = [obs_times[i] + noise_times[i] for i in range(len(obs_env.rays))]
    obs_times = new_obs_times
    n_rays = len(obs_times)
    return obs_times, n_rays


@dataclass
class EnvParameters:
    num_cells_x: int
    num_cells_y: int
    width: int
    height: int

@dataclass
class DeviceParameters:
    n_sources: int
    n_receivers: int
    n_travels: int
    offset: int

@dataclass
class MethodParameters:
    alpha: float
    beta: float = 0
    masking_depth: int = 1000



if __name__ == '__main__':
    env_params = EnvParameters(num_cells_x=15,#43,
                               num_cells_y=25,#50,
                               width=2000,
                               height=2000)
    dev_params = DeviceParameters(n_sources=8,#15,
                                  n_receivers=7,#15,
                                  n_travels=5,
                                  offset=0)

    for device_pos_mode_idx in range(4):
    # for device_pos in [2]:
        dev_positioning = position_devices(device_pos_mode_idx, env_params, dev_params)

        p_bas = MethodParameters(alpha=0.95)
        p_trd = MethodParameters(alpha=0.85,
                                 beta=0.45,
                                 masking_depth=1000)
        p_spl = MethodParameters(alpha=0.85,
                                 masking_depth=1000)
        sig = 0.000

        np.set_printoptions(legacy='1.25', precision=6, linewidth=320)
        decimal.getcontext().prec = 2
        main(env_params, dev_positioning, p_bas, p_trd, p_spl, sig, show_path=False)
