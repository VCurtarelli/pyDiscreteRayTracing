from threading import settrace

from pandas.io.formats.format import get_format_datetime64

from funs_device_positioning import position_devices, export_pos_devices
from py_libs import *
from class_Environment import Environment, EstEnvironment
import class_Environment as clsEnv
import sys
from funs_encode_and_hash import make_folders_and_hash
from dataclasses import dataclass
from typing import Any
from numpy import ndarray
import copy
import seaborn as sns
import datetime

np.set_printoptions(legacy='1.25',precision=2,linewidth=600,threshold=sys.maxsize)
decimal.getcontext().prec = 2

rms = lambda x: np.sqrt(np.mean(x**2))
getdatetime = lambda: str(datetime.datetime.now().strftime('%y%m%d-%Hh%Mm%Ss'))
def simulation_loop(env_params, dev_positioning, method_params, sigma=0.0, show_path=False, name=None, env_mode=None):

    ## Extract parameters from input
    num_cells_x = env_params.num_cells_x
    num_cells_y = env_params.num_cells_y
    width = env_params.width
    height = env_params.height
    pos_sources_x, pos_sources_y, pos_receivers_x, pos_receivers_y, pos_name = dev_positioning

    ## Extract sources and receivers, generate environments, and related variables
    pos_receivers = list(set(zip(pos_receivers_x, pos_receivers_y)))
    pos_sources = list(set(zip(pos_sources_x, pos_sources_y)))

    est_envs, obs_env = generate_environments(height, num_cells_x, num_cells_y, pos_receivers, pos_sources, width, method_params.keys(), env_mode)
    model_slowness = 1/obs_env.vy.reshape(-1, 1)
    obs_times, n_rays = generate_observed_times(obs_env, sigma)

    # obs_env.plot_curves()

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
        'method_params': method_params,
        # 'method_names': method_names,
    }
    if name is not None:
        # name += '-' + getdatetime()
        name = name.replace('_', '--').replace(' ', '--').replace('|', ';')
    code, direc, parameters = make_folders_and_hash(n_rays, parameters, name)
    if name is None:
        name = code
    export_pos_devices(pos_sources, pos_receivers, direc + name + '/', width, height)

    ## Main iteration loop
    est_envs = iteration_loop(name, est_envs, model_slowness, n_rays, obs_env, obs_times, method_params, env_params)

    ## Print metrics for on-execution assessment
    print('Travel-time error:')
    for env_idx, est_env in enumerate(est_envs):
        print('\t{}. - {:.4f}ms'.format(est_env.field_name[:3], 1000 * est_env.est_time_mse[-1]),
              'out of {:.4f}s'.format(np.mean(obs_times)))
    print('Field RMS error:')
    for env_idx, est_env in enumerate(est_envs):
        print('\t{}. - {:.4f}m/s'.format(est_env.field_name[:3], est_env.est_ssf_rms[-1]),
              'with σ = {:.4f}m/s'.format(np.mean(np.std(obs_env.field, axis=1))))
    print('Roughness:')
    for env_idx, est_env in enumerate(est_envs):
        print('\t{}. - {:.4f} s⁻¹'.format(est_env.field_name[:3], est_env.est_rho_rms[-1]),
              'against {:.4f} s⁻¹'.format(obs_env.calc_rho(obs_env.field)))
    if name.startswith('search'):
        print('Optimal parameters:')
        for env_idx, est_env in enumerate(est_envs):
            print('\t{}. - {}'.format(est_env.field_name[:3], est_env.format_params()))
        print()

        opt_parameters = []
        for env_idx, est_env in enumerate(est_envs):
            opt_parameters.append('\t{}. - {}'.format(est_env.field_name[:3], est_env.format_params()))
        opt_parameters = '\n'.join(opt_parameters)
        with open(direc+name+'/optimal_parameters.txt', 'w') as f:
            f.write(opt_parameters)

    return est_envs
    show_estimated_profiles(env_params, est_envs)


def optimal_metrics(env_params, dev_positioning, env_mode):
    num_cells_x = env_params.num_cells_x
    num_cells_y = env_params.num_cells_y
    width = env_params.width
    height = env_params.height
    pos_sources_x, pos_sources_y, pos_receivers_x, pos_receivers_y, pos_name = dev_positioning

    ## Extract sources and receivers, generate environments, and related variables
    pos_receivers = list(set(zip(pos_receivers_x, pos_receivers_y)))
    pos_sources = list(set(zip(pos_sources_x, pos_sources_y)))

    _, obs_env = generate_environments(height, num_cells_x, num_cells_y, pos_receivers, pos_sources, width, [], env_mode)
    metrics = obs_env.calculate_optimal_metrics()

    return obs_env, metrics


def generate_environments(height, num_cells_x, num_cells_y, pos_receivers: list[tuple[Any, Any]],
                          pos_sources: list[tuple[Any, Any]], width: int, env_names: list[str],
                          env_mode: str) -> tuple[list[EstEnvironment], Environment]:

    if env_mode is None:
        env_mode = 'import'
    obs_env = Environment(num_cells_x, num_cells_y, width, height, mode=env_mode)
    obs_env.generate_rays(pos_sources, pos_receivers)
    # obs_env.plot_curves()

    est_envs = []
    for field_name in env_names:
        est_env = EstEnvironment(num_cells_x, num_cells_y, width, height, None, field_name=field_name, mode=env_mode)
        est_env.generate_rays(pos_sources, pos_receivers)
        est_envs.append(est_env)
    return est_envs, obs_env


def generate_observed_times(obs_env: Environment, sigma: float) -> tuple[list[Any], int]:
    obs_times = [ray.time for ray in obs_env.rays]
    mean_time = np.mean(obs_times)
    rng = np.random.default_rng(0)
    noise_times = list(sigma * mean_time * rng.random((len(obs_env.rays),)))
    new_obs_times = [obs_times[i] + noise_times[i] for i in range(len(obs_env.rays))]
    obs_times = new_obs_times
    n_rays = len(obs_times)
    return obs_times, n_rays


def iteration_loop(code: str, est_envs: list[Any], model_slowness: ndarray,
                   n_rays: int, obs_env: Environment, obs_times: list[Any], method_params: dict[str, Any], env_params):

    ## Export observed environment
    obs_env.field_to_csv(0, export_params=True, code=code)
    obs_env.field_to_csv(0, export_params=True, code=code, mode='yx')
    print('\n' * 10)
    mode_list = list(method_params.keys())
    param_list = [method_params[mode] for mode in mode_list]
    out_est_envs = []
    for idx in range(len(est_envs)):
        est_env = est_envs[idx]
        param = param_list[idx]
        mode = mode_list[idx]
        best_est_env = None
        print(est_env.field_name)
        if isinstance(param, MethodParameters):
            params = [param]
        else:
            params = param
        for param_idx, param in enumerate(params):
            print(str(param_idx+1) + '/' + str(len(params)))
            it_idx = 0
            est_env_it = copy.deepcopy(est_env)  # est_env for iterating
            est_env_it.set_parameters(param)
            while True:
                # print("\tIteration {}".format(it_idx))
                # it_fld_fun(est_env, param)
                try:
                    est_env_it.iterate_field(n_rays, obs_times=obs_times, model_slowness=model_slowness, mode=mode)
                    est_env_it.update_rays()
                    est_env_it.calc_metrics(obs_times, obs_env.field)
                except (RuntimeError, np.linalg.LinAlgError):
                    est_env_it.est_time_mse.append(np.inf)
                    est_env_it.est_ssf_rms.append(np.inf)
                    break
                # est_env_it.calc_metrics(obs_times, obs_env.field)
                it_idx += 1

                if it_idx >= est_env_it.n_its:
                    break
            # show_estimated_profiles(env_params, [est_env_it])
            if best_est_env is not None:
                print('\t'+est_env_it.format_params())
                print('\t{:.4f} - {:.4f}'.format(est_env_it.est_ssf_rms[-1], best_est_env.est_ssf_rms[-1]))
            if best_est_env is None:
                best_est_env = est_env_it
            elif est_env_it.est_ssf_rms[-1] < best_est_env.est_ssf_rms[-1]:
                best_est_env = est_env_it
            # if len(params) == 1:
            #     best_est_env.field_to_csv(it_idx, code=code)
            #     best_est_env.field_to_csv(it_idx, comp=obs_env.field, code=code, vmin=-20, vmax=20, export_params=True)
            #     best_est_env.field_to_csv(it_idx, mode='yx',code=code)
            #     best_est_env.field_to_csv(it_idx, comp=obs_env.field, code=code, mode='yx')
        out_est_envs.append(best_est_env)

    for est_env in out_est_envs:
        est_env.field_to_csv('fin', code=code)
        est_env.field_to_csv('fin', comp=obs_env.field, code=code, vmin=-20, vmax=20, export_params=True)
        est_env.field_to_csv('fin', code=code, mode='yx')
        est_env.field_to_csv('fin', comp=obs_env.field, code=code, mode='yx')
        est_env.export_metrics(code=code)
    return out_est_envs


def show_profiles(env_params, show=False):
    obs_env_0 = Environment(num_cells_x=env_params.num_cells_x,
                            num_cells_y=env_params.num_cells_y,
                            width=env_params.width, height=env_params.height,
                            field_name='munk', mode='munk')
    obs_env_1 = Environment(num_cells_x=env_params.num_cells_x,
                            num_cells_y=env_params.num_cells_y,
                            width=env_params.width, height=env_params.height,
                            field_name='wilson', mode='wilson')
    obs_env_2 = Environment(num_cells_x=env_params.num_cells_x,
                            num_cells_y=env_params.num_cells_y,
                            width=env_params.width, height=env_params.height,
                            field_name='mackenzie', mode='mackenzie')
    obs_env_3 = Environment(num_cells_x=env_params.num_cells_x,
                            num_cells_y=env_params.num_cells_y,
                            width=env_params.width, height=env_params.height,
                            field_name='delgrosso', mode='delgrosso')

    fig, axs = plt.subplots(2, 2, figsize=(18, 5))
    ax1, ax2, ax3, ax4 = axs.flatten()

    # Plot heatmaps on the respective axes
    for col_idx in range(env_params.num_cells_x):
        ax1.plot(obs_env_1.true_field[:, col_idx], np.linspace(0, env_params.height, env_params.num_cells_y))
        ax2.plot(obs_env_2.true_field[:, col_idx], np.linspace(0, env_params.height, env_params.num_cells_y))
        ax3.plot(obs_env_3.true_field[:, col_idx], np.linspace(0, env_params.height, env_params.num_cells_y))
        ax4.plot(obs_env_1.true_field[:, col_idx] - obs_env_0.true_field[:, col_idx],
                 np.linspace(0, 1, env_params.num_cells_y), color='red',
                 label='Wilson' if col_idx == 0 else None, alpha=0.3)
        ax4.plot(obs_env_2.true_field[:, col_idx] - obs_env_0.true_field[:, col_idx],
                 np.linspace(0, 1, env_params.num_cells_y), color='blue',
                 label='Mackenzie' if col_idx == 0 else None, alpha=0.6)
        ax4.plot(obs_env_3.true_field[:, col_idx] - obs_env_0.true_field[:, col_idx],
                 np.linspace(0, 1, env_params.num_cells_y), color='green',
                 label='DelGrosso' if col_idx == 0 else None)
    ax1.plot(obs_env_0.true_field[:, 0], np.linspace(0, env_params.height, env_params.num_cells_y), color='black',)
    ax2.plot(obs_env_0.true_field[:, 0], np.linspace(0, env_params.height, env_params.num_cells_y), color='black',)
    ax3.plot(obs_env_0.true_field[:, 0], np.linspace(0, env_params.height, env_params.num_cells_y), color='black',)

    for ax in axs.flatten():
        ax.yaxis.set_inverted(True)
    # for ax in [ax1, ax2, ax3]:
    #     ax.set_xlim([1520, 1555])

    ax1.set_title(obs_env_1.field_name)
    ax2.set_title(obs_env_2.field_name)
    ax3.set_title(obs_env_3.field_name)

    plt.tight_layout()
    plt.legend()

    if show:
        plt.show()


def show_estimated_profiles(env_params, obs_envs):

    n_envs = len(obs_envs)
    fig, axs = plt.subplots(2, n_envs, figsize=(10,6))
    axs = axs.flatten()

    v_min = np.inf
    v_max = -np.inf
    cv_min = np.inf
    cv_max = -np.inf
    depths = np.linspace(0, env_params.height, env_params.num_cells_y)
    for env_idx in range(len(obs_envs)):
        ax = axs[env_idx]
        cax = axs[env_idx+n_envs]
        env = obs_envs[env_idx]
        v_min = min(v_min, np.amin(env.field), np.amin(env.true_field))
        v_max = max(v_max, np.amax(env.field), np.amax(env.true_field))
        cv_min = min(cv_min, np.amin(env.field - env.true_field))
        cv_max = max(cv_max, np.amax(env.field - env.true_field))
        for col_idx in range(env_params.num_cells_x):
            angle = 2*pi*col_idx/env_params.num_cells_x
            ax.plot(env.field[:, col_idx], depths, color=[0.5+0.5*np.cos(angle),0.5+0.5*np.sin(angle),0])
            cax.plot(env.field[:, col_idx] - env.true_field[:, col_idx], depths, color=[0.5+0.5*np.cos(angle),0.5+0.5*np.sin(angle),0])
            ax.set_title(env.field_name)
            cax.set_title('Comp '+env.field_name)
        cax.plot(np.mean(env.field - env.true_field, axis=1), depths, color='black')
        ax.plot(np.mean(env.true_field, axis=1), depths, color='black')
    for env_idx in range(len(obs_envs)):
        for col_idx in range(env_params.num_cells_x):
            ax.plot(env.true_field[:, col_idx], depths, color='black', alpha=0.3)
        ax = axs[env_idx]
        cax = axs[env_idx+n_envs]
        ax.set_xlim([v_min, v_max])
        cax.set_xlim([cv_min, cv_max])

    for ax in axs.flatten():
        ax.yaxis.set_inverted(True)

    plt.tight_layout()
    # plt.legend()
    fig.canvas.manager.window.wm_geometry("+100+100")
    plt.show()


class Parameters:
    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

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
    offset: int
    n_travels: int = 3
    travels_dpth: int = 500

@dataclass
class MethodParameters:
    alpha: float | tuple[float | int, ...]
    beta: float = None
    masking_depth: int = None


def get_simulation_mode(simulation_mode, environment_parameters, device_parameters, env_mode):
    masking_depth = 0

    match simulation_mode:

        case 'sweep':
            p_bas = [MethodParameters(alpha=alpha) for alpha in np.linspace(0, 0.98, 10)]
            p_trd = [MethodParameters(alpha=alpha, beta=beta, masking_depth=masking_depth)
                     for alpha in np.linspace(0, 0.9, 10)
                     for beta in np.linspace(0, 0.3, 10)]
            p_spl = [MethodParameters(alpha=alpha, masking_depth=masking_depth)
                     for alpha in np.linspace(0, 0.98, 10)]
            p_prm = [MethodParameters(alpha=(alpha1, alpha2, alpha3, alpha4))
                     for alpha1 in np.logspace(-4, -1, 7)
                     for alpha2 in np.linspace(0.6, 0.9, 4)
                     for alpha3 in np.logspace(-4, -1, 7)
                     for alpha4 in np.logspace(-4, -1, 7)
                     ]
            p_svd = [MethodParameters(alpha=alpha)
                     for alpha in (0.5,)]
            methods_parameters = {
                'classic': p_bas,
                'trade': p_trd,
                'split': p_spl,
                'munk': p_prm,
                'svd': p_svd
            }
        case 'optimal-import_bas-vs-svd_surround':
            p_bas = [MethodParameters(alpha=9.618e-01)]
            p_svd = [MethodParameters(alpha=2.622e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['surround']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
        case 'optimal-import_bas-vs-svd_layer':
            p_bas = [MethodParameters(alpha=9.900e-01)]
            p_svd = [MethodParameters(alpha=6.735e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)

        case 'search-import_metrics_layer':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.99, 50)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.05, 0.7, 50)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import_metrics_surround':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.99, 50)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.05, 0.7, 50)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['surround']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import_metrics_layer-blob':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.9, 0.9999, 50)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.05, 0.7, 50)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/blob']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=3, offset=0)
        case 'search-import_metrics_layer-column':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.9, 0.9999, 50)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.4, 0.7, 50)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import_metrics_layer-diag':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.9999, 50)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.05, 0.7, 50)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+diagonal']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)

        case 'optimal-import_metrics_layer':
            p_bas = [MethodParameters(alpha=9.999e-01)]
            p_svd = [MethodParameters(alpha=6.735e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import_metrics_surround':
            p_bas = [MethodParameters(alpha=9.618e-01)]
            p_svd = [MethodParameters(alpha=2.622e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['surround']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import_metrics_layer-blob':
            p_bas = [MethodParameters(alpha=9.991e-01)]
            p_svd = [MethodParameters(alpha=6.867e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/blob']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=3, offset=0)
        case 'optimal-import_metrics_layer-column':
            p_bas = [MethodParameters(alpha=9.856e-01)]
            p_svd = [MethodParameters(alpha=5.959e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import_metrics_layer-diag':
            p_bas = [MethodParameters(alpha=9.999e-01)]
            p_svd = [MethodParameters(alpha=6.602e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+diagonal']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)

        case 'optimal-import_sweep_surround':
            p_bas = [MethodParameters(alpha=9.618e-01)]
            p_svd = [MethodParameters(alpha=2.622e-01)]
            p_trd = [MethodParameters(alpha=9.000e-01, beta=2.000e-01, masking_depth=0)]
            p_spl = [MethodParameters(alpha=9.800e-01, masking_depth=1000)]
            p_prm = [MethodParameters(alpha=(1.000e-3, 9.000e-1, 1.000e-2, 1.000e-4))]
            p_eof1 = [MethodParameters(alpha=6.063e-01)]
            p_eof2 = [MethodParameters(alpha=9.516e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd,
                'trade': p_trd,
                'split': p_spl,
                'param': p_prm,
                'eof1': p_eof1,
                'eof2': p_eof2,
                'mean': [None]
            }
            device_pos_modes = ['surround']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import_sweep_layer':
            p_bas = [MethodParameters(alpha=9.900e-01)]
            p_svd = [MethodParameters(alpha=6.735e-01)]
            p_trd = [MethodParameters(alpha=9.000e-01, beta=3.000e-01, masking_depth=0)]
            p_spl = [MethodParameters(alpha=9.800e-01, masking_depth=1000)]
            p_prm = [MethodParameters(alpha=(3.162e-04, 8.000e-01, 1.000e-01, 1.000e-04))]
            p_eof1 = [MethodParameters(alpha=3.000e-01)]
            p_eof2 = [MethodParameters(alpha=9.900e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd,
                'trade': p_trd,
                'split': p_spl,
                'param': p_prm,
                'eof1': p_eof1,
                'eof2': p_eof2,
                'mean': [None]
            }
            device_pos_modes = ['layer/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)

        case 'search-import_sweep_surround':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.99, 50)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.05, 0.7, 50)]
            p_trd = [MethodParameters(alpha=alpha, beta=beta, masking_depth=0)
                     for alpha in np.linspace(0, 0.9, 7)
                     for beta in np.linspace(0, 0.3, 7)]
            p_spl = [MethodParameters(alpha=alpha, masking_depth=1000)
                     for alpha in np.linspace(0, 0.98, 50)]
            p_prm = [MethodParameters(alpha=(alpha1, alpha2, alpha3, alpha4))
                     for alpha1 in np.logspace(-4, -1, 7)
                     for alpha2 in np.linspace(0.6, 0.9, 4)
                     for alpha3 in np.logspace(-4, -1, 7)
                     for alpha4 in np.logspace(-4, -1, 7)
                     ]
            p_eof1 = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.99, 50)]
            p_eof2 = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.99, 50)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd,
                'trade': p_trd,
                'split': p_spl,
                'param': p_prm,
                'eof1': p_eof1,
                'eof2': p_eof2,
            }
            device_pos_modes = ['surround']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import_sweep_layer':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.99, 50)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.05, 0.7, 50)]
            p_trd = [MethodParameters(alpha=alpha, beta=beta, masking_depth=0)
                     for alpha in np.linspace(0, 0.9, 7)
                     for beta in np.linspace(0, 0.3, 7)]
            p_spl = [MethodParameters(alpha=alpha, masking_depth=1000)
                     for alpha in np.linspace(0, 0.98, 50)]
            p_prm = [MethodParameters(alpha=(alpha1, alpha2, alpha3, alpha4))
                     for alpha1 in np.logspace(-4, -1, 7)
                     for alpha2 in np.linspace(0.6, 0.9, 4)
                     for alpha3 in np.logspace(-4, -1, 7)
                     for alpha4 in np.logspace(-4, -1, 7)
                     ]
            p_eof1 = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.99, 50)]
            p_eof2 = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.99, 50)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd,
                'trade': p_trd,
                'split': p_spl,
                'param': p_prm,
                'eof1': p_eof1,
                'eof2': p_eof2,
            }
            device_pos_modes = ['layer/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)

        case 'search_bas-vs-svd_layer':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.99, 50)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.05, 0.7, 50)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer']

        case 'search-import_petro_layer':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.99, 50)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.05, 0.7, 50)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import_petro_surround':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.99, 50)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.05, 0.7, 50)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['surround']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import_petro_layer|blob':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.99, 50)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.05, 0.7, 50)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/blob']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=3, offset=0)
        case 'search-import_petro_layer|layer-column':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.99, 50)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.05, 0.7, 50)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import_petro_layer|layer-diag':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.99, 50)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.05, 0.7, 50)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+diagonal']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import_petro_layer|layer-scolumn':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.99, 50)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.05, 0.7, 50)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+short-column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import_petro_layer-scolumn|layer':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.99, 50)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.05, 0.7, 50)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+short-column/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import_petro_semi-surround':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.99, 50)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.05, 0.7, 50)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+short-column/layer+short-column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import_petro_layer|bracket':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.99, 50)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.05, 0.7, 50)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/bracket']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import_petro_layer|cap':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.99, 50)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.05, 0.7, 50)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/cap']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)

        case 'optimal-import_petro_layer':
            p_bas = [MethodParameters(alpha=9.900e-01)]
            p_svd = [MethodParameters(alpha=6.204e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import_petro_surround':
            p_bas = [MethodParameters(alpha=9.759e-01)]
            p_svd = [MethodParameters(alpha=3.286e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['surround']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import_petro_layer|blob':
            p_bas = [MethodParameters(alpha=9.900e-01)]
            p_svd = [MethodParameters(alpha=7.000e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/blob']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=3, offset=0)
        case 'optimal-import_petro_layer|layer-column':
            p_bas = [MethodParameters(alpha=9.478e-01)]
            p_svd = [MethodParameters(alpha=5.408e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import_petro_layer|layer-diag':
            p_bas = [MethodParameters(alpha=4.127e-01)]
            p_svd = [MethodParameters(alpha=5.673e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+diagonal']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import_petro_layer|layer-scolumn':
            p_bas = [MethodParameters(alpha=9.900e-01)]
            p_svd = [MethodParameters(alpha=3.684e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+short-column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import_petro_layer-scolumn|layer':
            p_bas = [MethodParameters(alpha=9.759e-01)]
            p_svd = [MethodParameters(alpha=2.224e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+short-column/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import_petro_semi-surround':
            p_bas = [MethodParameters(alpha=9.759e-01)]
            p_svd = [MethodParameters(alpha=3.153e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+short-column/layer+short-column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import_petro_layer|bracket':
            p_bas = [MethodParameters(alpha=8.069e-01)]
            p_svd = [MethodParameters(alpha=5.010e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/bracket']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import_petro_layer|cap':
            p_bas = [MethodParameters(alpha=9.900e-01)]
            p_svd = [MethodParameters(alpha=3.153e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/cap']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)

        case 'meanpar-import_petro_layer':
            p_bas = [MethodParameters(alpha=9.055e-01)]
            p_svd = [MethodParameters(alpha=4.479e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import_petro_surround':
            p_bas = [MethodParameters(alpha=9.055e-01)]
            p_svd = [MethodParameters(alpha=4.479e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['surround']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import_petro_layer|blob':
            p_bas = [MethodParameters(alpha=9.055e-01)]
            p_svd = [MethodParameters(alpha=4.479e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/blob']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=3, offset=0)
        case 'meanpar-import_petro_layer|layer-column':
            p_bas = [MethodParameters(alpha=9.055e-01)]
            p_svd = [MethodParameters(alpha=4.479e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import_petro_layer|layer-diag':
            p_bas = [MethodParameters(alpha=9.055e-01)]
            p_svd = [MethodParameters(alpha=4.479e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+diagonal']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import_petro_layer|layer-scolumn':
            p_bas = [MethodParameters(alpha=9.055e-01)]
            p_svd = [MethodParameters(alpha=4.479e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+short-column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import_petro_layer-scolumn|layer':
            p_bas = [MethodParameters(alpha=9.055e-01)]
            p_svd = [MethodParameters(alpha=4.479e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+short-column/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import_petro_semi-surround':
            p_bas = [MethodParameters(alpha=9.055e-01)]
            p_svd = [MethodParameters(alpha=4.479e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+short-column/layer+short-column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import_petro_layer|bracket':
            p_bas = [MethodParameters(alpha=9.055e-01)]
            p_svd = [MethodParameters(alpha=4.479e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/bracket']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import_petro_layer|cap':
            p_bas = [MethodParameters(alpha=9.055e-01)]
            p_svd = [MethodParameters(alpha=4.479e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/cap']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)

        case 'search-import-deep_petro_layer':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.999, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import-deep_petro_surround':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.999, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['surround']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import-deep_petro_layer|blob':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.999, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/blob']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=3, offset=0)
        case 'search-import-deep_petro_layer|layer-column':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.999, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+column']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import-deep_petro_layer|layer-diag':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.999, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+diagonal']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import-deep_petro_layer|layer-scolumn':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.999, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+short-column']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import-deep_petro_layer-scolumn|layer':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.999, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+short-column/layer']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import-deep_petro_semi-surround':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.999, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+short-column/layer+short-column']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import-deep_petro_layer|bracket':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.999, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/bracket']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import-deep_petro_layer|cap':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.3, 0.999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.999, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/cap']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)

        case 'optimal-import-deep_petro_layer':
            p_bas = [MethodParameters(alpha=9.919e-01)]
            p_svd = [MethodParameters(alpha=2.090e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import-deep_petro_surround':
            p_bas = [MethodParameters(alpha=8.648e-01)]
            p_svd = [MethodParameters(alpha=3.270e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['surround']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import-deep_petro_layer|blob':
            p_bas = [MethodParameters(alpha=6.813e-01)]
            p_svd = [MethodParameters(alpha=1.726e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/blob']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=3, offset=0)
        case 'optimal-import-deep_petro_layer|layer-column':
            p_bas = [MethodParameters(alpha=3.000e-01)]
            p_svd = [MethodParameters(alpha=5.086e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+column']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import-deep_petro_layer|layer-diag':
            p_bas = [MethodParameters(alpha=6.177e-01)]
            p_svd = [MethodParameters(alpha=1.908e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+diagonal']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import-deep_petro_layer|layer-scolumn':
            p_bas = [MethodParameters(alpha=9.566e-01)]
            p_svd = [MethodParameters(alpha=2.181e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+short-column']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import-deep_petro_layer-scolumn|layer':
            p_bas = [MethodParameters(alpha=9.919e-01)]
            p_svd = [MethodParameters(alpha=2.181e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+short-column/layer']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import-deep_petro_semi-surround':
            p_bas = [MethodParameters(alpha=6.107e-01)]
            p_svd = [MethodParameters(alpha=2.635e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+short-column/layer+short-column']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import-deep_petro_layer|bracket':
            p_bas = [MethodParameters(alpha=5.189e-01)]
            p_svd = [MethodParameters(alpha=1.636e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/bracket']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import-deep_petro_layer|cap':
            p_bas = [MethodParameters(alpha=7.942e-01)]
            p_svd = [MethodParameters(alpha=1.726e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/cap']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)

        case 'meanpar-import-deep_petro_layer':
            p_bas = [MethodParameters(alpha=7.323e-01)]
            p_svd = [MethodParameters(alpha=1.667e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import-deep_petro_surround':
            p_bas = [MethodParameters(alpha=7.323e-01)]
            p_svd = [MethodParameters(alpha=1.667e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['surround']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import-deep_petro_layer|blob':
            p_bas = [MethodParameters(alpha=7.323e-01)]
            p_svd = [MethodParameters(alpha=1.667e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/blob']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=3, offset=0)
        case 'meanpar-import-deep_petro_layer|layer-column':
            p_bas = [MethodParameters(alpha=7.323e-01)]
            p_svd = [MethodParameters(alpha=1.667e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+column']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import-deep_petro_layer|layer-diag':
            p_bas = [MethodParameters(alpha=7.323e-01)]
            p_svd = [MethodParameters(alpha=1.667e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+diagonal']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import-deep_petro_layer|layer-scolumn':
            p_bas = [MethodParameters(alpha=7.323e-01)]
            p_svd = [MethodParameters(alpha=1.667e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+short-column']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import-deep_petro_layer-scolumn|layer':
            p_bas = [MethodParameters(alpha=7.323e-01)]
            p_svd = [MethodParameters(alpha=1.667e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+short-column/layer']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import-deep_petro_semi-surround':
            p_bas = [MethodParameters(alpha=7.323e-01)]
            p_svd = [MethodParameters(alpha=1.667e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+short-column/layer+short-column']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import-deep_petro_layer|bracket':
            p_bas = [MethodParameters(alpha=7.323e-01)]
            p_svd = [MethodParameters(alpha=1.667e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/bracket']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import-deep_petro_layer|cap':
            p_bas = [MethodParameters(alpha=7.323e-01)]
            p_svd = [MethodParameters(alpha=1.667e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/cap']
            env_mode = 'import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)

    return environment_parameters, device_parameters, methods_parameters, device_pos_modes[0], env_mode


def main():
    # show_profiles(environment_parameters, True)
    env_mode = 'small'
    match env_mode:
        case 'small':
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=25, width=2000, height=2000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=3, offset=0)
        case 'mid':
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=25, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=3, offset=0)
        case _:
            environment_parameters = EnvParameters(num_cells_x=43, num_cells_y=50, width=2000, height=2000)
            device_parameters = DeviceParameters(n_sources=15, n_receivers=15, n_travels=3, offset=0)
    device_pos_modes = {
        'od1': 'layer/layer',
        'od3': 'layer/blob',
        'od5': 'layer/layer+diagonal',
        'od4': 'layer/layer+column',
        'od9': 'layer/bracket',
        'od2': 'surround',
        'od7': 'layer+short-column/layer',
        'od6': 'layer/layer+short-column',
        'od8': 'layer+short-column/layer+short-column',
        'odA': 'layer/cap',
    }
    metric_estimation = True

    simulation_modes = {'ops': ['optimal-import', 'bas-vs-svd', 'surround'],    # optimum - paper - surround
                        'opl': ['optimal-import', 'bas-vs-svd', 'layer'],       # optimum - paper - layered

                        'sss': ['search-import', 'sweep', 'surround'],          # search - sweep - surround
                        'ssl': ['search-import', 'sweep', 'layer'],             # search - sweep - layer
                        'oss': ['optimal-import', 'sweep', 'surround'],         # optimum - sweep - surround
                        'osl': ['optimal-import', 'sweep', 'layer'],            # optimum - sweep - layered

                        'sml': ['search-import', 'metrics', 'layer'],           # search - metrics - layer
                        'sms': ['search-import', 'metrics', 'surround'],        # search - metrics - surround
                        'smb': ['search-import', 'metrics', 'layer-blob'],      # search - metrics - layer-blob
                        'smc': ['search-import', 'metrics', 'layer-column'],    # search - metrics - layer-column
                        'smd': ['search-import', 'metrics', 'layer-diag'],      # search - metrics - layer-diag
                        'oml': ['optimal-import', 'metrics', 'layer'],          # optimal - metrics - layer
                        'oms': ['optimal-import', 'metrics', 'surround'],       # optimal - metrics - surround
                        'omb': ['optimal-import', 'metrics', 'layer-blob'],     # optimal - metrics - layer-blob
                        'omc': ['optimal-import', 'metrics', 'layer-column'],   # optimal - metrics - layer-column
                        'omd': ['optimal-import', 'metrics', 'layer-diag'],     # optimal - metrics - layer-diag

                        'sp1': ['search-import', 'petro', 'layer'],                 # search - petro - layer
                        'sp2': ['search-import', 'petro', 'surround'],              # search - petro - surround
                        'sp3': ['search-import', 'petro', 'layer|blob'],            # search - petro - layer-blob
                        'sp4': ['search-import', 'petro', 'layer|layer-column'],    # search - petro - layer-column
                        'sp5': ['search-import', 'petro', 'layer|layer-diag'],      # search - petro - layer-diag
                        'sp6': ['search-import', 'petro', 'layer|layer-scolumn'],   # search - petro - layer-scolumn
                        'sp7': ['search-import', 'petro', 'layer-scolumn|layer'],   # search - petro - scolumn-layer
                        'sp8': ['search-import', 'petro', 'semi-surround'],         # search - petro - semi-surround
                        'sp9': ['search-import', 'petro', 'layer|bracket'],         # search - petro - layer-bracket
                        'spA': ['search-import', 'petro', 'layer|cap'],             # search - petro - layer-cap

                        'op1': ['optimal-import', 'petro', 'layer'],                # optimal - petro - layer
                        'op2': ['optimal-import', 'petro', 'surround'],             # optimal - petro - surround
                        'op3': ['optimal-import', 'petro', 'layer|blob'],           # optimal - petro - layer-blob
                        'op4': ['optimal-import', 'petro', 'layer|layer-column'],   # optimal - petro - layer-column
                        'op5': ['optimal-import', 'petro', 'layer|layer-diag'],     # optimal - petro - layer-diag
                        'op6': ['optimal-import', 'petro', 'layer|layer-scolumn'],  # optimal - petro - layer-scolumn
                        'op7': ['optimal-import', 'petro', 'layer-scolumn|layer'],  # optimal - petro - scolumn-layer
                        'op8': ['optimal-import', 'petro', 'semi-surround'],        # optimal - petro - semi-surround
                        'op9': ['optimal-import', 'petro', 'layer|bracket'],        # optimal - petro - layer-bracket
                        'opA': ['optimal-import', 'petro', 'layer|cap'],            # optimal - petro - layer-cap

                        'mp1': ['meanpar-import', 'petro', 'layer'],                # mean-parameter - petro - layer
                        'mp2': ['meanpar-import', 'petro', 'surround'],             # mean-parameter - petro - surround
                        'mp3': ['meanpar-import', 'petro', 'layer|blob'],           # mean-parameter - petro - layer-blob
                        'mp4': ['meanpar-import', 'petro', 'layer|layer-column'],   # mean-parameter - petro - layer-column
                        'mp5': ['meanpar-import', 'petro', 'layer|layer-diag'],     # mean-parameter - petro - layer-diag
                        'mp6': ['meanpar-import', 'petro', 'layer|layer-scolumn'],  # mean-parameter - petro - layer-scolumn
                        'mp7': ['meanpar-import', 'petro', 'layer-scolumn|layer'],  # mean-parameter - petro - scolumn-layer
                        'mp8': ['meanpar-import', 'petro', 'semi-surround'],        # mean-parameter - petro - semi-surround
                        'mp9': ['meanpar-import', 'petro', 'layer|bracket'],        # mean-parameter - petro - layer-bracket
                        'mpA': ['meanpar-import', 'petro', 'layer|cap'],            # mean-parameter - petro - layer-cap

                        'sd1': ['search-import-deep', 'petro', 'layer'],                # search - petro - layer
                        'sd2': ['search-import-deep', 'petro', 'surround'],             # search - petro - surround
                        'sd3': ['search-import-deep', 'petro', 'layer|blob'],           # search - petro - layer-blob
                        'sd4': ['search-import-deep', 'petro', 'layer|layer-column'],   # search - petro - layer-column
                        'sd5': ['search-import-deep', 'petro', 'layer|layer-diag'],     # search - petro - layer-diag
                        'sd6': ['search-import-deep', 'petro', 'layer|layer-scolumn'],  # search - petro - layer-scolumn
                        'sd7': ['search-import-deep', 'petro', 'layer-scolumn|layer'],  # search - petro - scolumn-layer
                        'sd8': ['search-import-deep', 'petro', 'semi-surround'],        # search - petro - semi-surround
                        'sd9': ['search-import-deep', 'petro', 'layer|bracket'],        # search - petro - layer-bracket
                        'sdA': ['search-import-deep', 'petro', 'layer|cap'],            # search - petro - layer-cap

                        'od1': ['optimal-import-deep', 'petro', 'layer'],               # optimal - petro - layer
                        'od2': ['optimal-import-deep', 'petro', 'surround'],            # optimal - petro - surround
                        'od3': ['optimal-import-deep', 'petro', 'layer|blob'],          # optimal - petro - layer-blob
                        'od4': ['optimal-import-deep', 'petro', 'layer|layer-column'],  # optimal - petro - layer-column
                        'od5': ['optimal-import-deep', 'petro', 'layer|layer-diag'],    # optimal - petro - layer-diag
                        'od6': ['optimal-import-deep', 'petro', 'layer|layer-scolumn'], # optimal - petro - layer-scolumn
                        'od7': ['optimal-import-deep', 'petro', 'layer-scolumn|layer'], # optimal - petro - scolumn-layer
                        'od8': ['optimal-import-deep', 'petro', 'semi-surround'],       # optimal - petro - semi-surround
                        'od9': ['optimal-import-deep', 'petro', 'layer|bracket'],       # optimal - petro - layer-bracket
                        'odA': ['optimal-import-deep', 'petro', 'layer|cap'],           # optimal - petro - layer-cap

                        'md1': ['meanpar-import-deep', 'petro', 'layer'],               # mean-parameter - petro - layer
                        'md2': ['meanpar-import-deep', 'petro', 'surround'],            # mean-parameter - petro - surround
                        'md3': ['meanpar-import-deep', 'petro', 'layer|blob'],          # mean-parameter - petro - layer-blob
                        'md4': ['meanpar-import-deep', 'petro', 'layer|layer-column'],  # mean-parameter - petro - layer-column
                        'md5': ['meanpar-import-deep', 'petro', 'layer|layer-diag'],    # mean-parameter - petro - layer-diag
                        'md6': ['meanpar-import-deep', 'petro', 'layer|layer-scolumn'], # mean-parameter - petro - layer-scolumn
                        'md7': ['meanpar-import-deep', 'petro', 'layer-scolumn|layer'], # mean-parameter - petro - scolumn-layer
                        'md8': ['meanpar-import-deep', 'petro', 'semi-surround'],       # mean-parameter - petro - semi-surround
                        'md9': ['meanpar-import-deep', 'petro', 'layer|bracket'],       # mean-parameter - petro - layer-bracket
                        'mdA': ['meanpar-import-deep', 'petro', 'layer|cap'],           # mean-parameter - petro - layer-cap
                        }

    classic_params = [get_simulation_mode('_'.join(simulation_modes[sim_code]), environment_parameters, device_parameters, env_mode)[2]['classic'][0].alpha for sim_code in device_pos_modes.keys()]
    svd_params = [get_simulation_mode('_'.join(simulation_modes[sim_code]), environment_parameters, device_parameters, env_mode)[2]['svd'][0].alpha for sim_code in device_pos_modes.keys()]
    mean_alpha = np.mean(classic_params)  # 7.323e-01
    mean_beta = np.mean(svd_params)  # 1.667e-01
    mean_alpha_1 = np.mean(classic_params[:5])
    mean_beta_1 = np.mean(svd_params[:5])
    mean_alpha_2 = np.mean(classic_params[5:])
    mean_beta_2 = np.mean(svd_params[5:])
    # ------
    # Main estimation loop
    ssf_errors = np.zeros([len(device_pos_modes.keys()), 2])
    for simulation_code_idx, simulation_code in enumerate(device_pos_modes.keys()):
        simulation_mode_list = simulation_modes[simulation_code].copy()
        simulation_mode_list.insert(2, getdatetime())
        simulation_mode = '_'.join(simulation_modes[simulation_code])

        environment_parameters, device_parameters, methods_parameters, device_pos_mode, env_mode\
            = get_simulation_mode(simulation_mode, environment_parameters, device_parameters, env_mode)

        device_positioning = position_devices(device_pos_mode, environment_parameters, device_parameters)
        sig = 0.000

        np.set_printoptions(legacy='1.25', precision=6, linewidth=320)
        decimal.getcontext().prec = 2
        est_envs = simulation_loop(environment_parameters, device_positioning, methods_parameters, sig, show_path=False,
                        name='_'.join(simulation_mode_list), env_mode=env_mode)
        for env_idx, est_env in enumerate(est_envs):
            ssf_errors[simulation_code_idx, env_idx] = est_env.est_ssf_rms[-1]

    if metric_estimation and len(set([tuple(simulation_modes[code][:2]) for code in device_pos_modes.keys()])) == 1:
        simulation_mode_list = simulation_modes[list(device_pos_modes.keys())[0]].copy()
        simulation_mode_list.insert(2, getdatetime())
        simulation_mode_list[-1] = 'metrics-all'
        simulation_mode = '_'.join(simulation_mode_list).replace('_', '--').replace(' ', '--').replace('|', ';')
        # ------
        # Calculates optimal metrics
        cross_mode_metrics = np.zeros([len(device_pos_modes.keys()), 7])
        metrics_keys = []
        for simulation_code_idx, simulation_code in enumerate(device_pos_modes.keys()):
            simulation_mode = '_'.join(simulation_modes[simulation_code])

            environment_parameters, device_parameters, methods_parameters, device_pos_mode, env_mode \
                = get_simulation_mode(simulation_mode, environment_parameters, device_parameters, env_mode)
            device_positioning = position_devices(device_pos_mode, environment_parameters, device_parameters)

            env, metrics = optimal_metrics(environment_parameters, device_positioning, env_mode)
            metrics_keys = metrics.keys()
            # env.plot_curves()
            pos_name = simulation_modes[simulation_code][-1]
            for metric_idx, metric_key in enumerate(metrics.keys()):
                cross_mode_metrics[simulation_code_idx, metric_idx] = metrics[metric_key]
            if simulation_code_idx == 0:
                line = 'Distribution'.rjust(20) + ' | '.join(['{}'.format(metric_key).rjust(12) for metric_key in metrics.keys()])
                print(line)
            line = pos_name.rjust(20) + ' | '.join(['{:.4f}'.format(metrics[metric]).rjust(12) for metric in metrics.keys()])
            print(line)
        # ------
        # Export metrics (RMSE-ssf vs metric) for each method
        filename = 'Results/optimal-import--petro--' + getdatetime() + '--metrics-all/'
        os.makedirs(filename, exist_ok=True)
        for metric_idx, metric_key in enumerate(metrics_keys):
            text = ['ssfe-ttt,ssfe-svd,{}'.format(metric_key)]
            for simulation_code_idx, simulation_code in enumerate(device_pos_modes.keys()):
                text.append('{},{},{}'.format(ssf_errors[simulation_code_idx,0], ssf_errors[simulation_code_idx,1], cross_mode_metrics[simulation_code_idx, metric_idx]))
            text = '\n'.join(text)
            with open(filename + metric_key+ '.csv', 'w') as f:
                f.write(text)

if __name__ == '__main__':
    main()
