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
        name = name.replace('_', '--').replace(' ', '--').replace('|', ';')
    code, direc, parameters = make_folders_and_hash(n_rays, parameters, name)
    if name is None:
        name = code
    export_pos_devices(pos_sources, pos_receivers, direc + name + '/', width, height)

    # if name.startswith('search'):
    #     obs_env.plot_curves(name)

    ## Main iteration loop
    est_envs = iteration_loop(name, est_envs, model_slowness, n_rays, obs_env, obs_times, method_params, env_params)

    ## Print metrics for on-execution assessment
    print(name)
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
        case 'optimal-import_sweep_surround':
            p_bas = [MethodParameters(alpha=9.759e-01)]
            p_svd = [MethodParameters(alpha=3.286e-01)]
            p_trd = [MethodParameters(alpha=9.000e-01, beta=1.000e-01, masking_depth=0)]
            p_spl = [MethodParameters(alpha=9.800e-01, masking_depth=1000)]
            p_prm = [MethodParameters(alpha=(1.000e-04, 9.000e-01, 1.000e-04, 1.000e-01))]
            p_eof1 = [MethodParameters(alpha=5.535e-01)]
            p_eof2 = [MethodParameters(alpha=9.618e-01)]
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
            p_svd = [MethodParameters(alpha=6.204e-01)]
            p_trd = [MethodParameters(alpha=9.000e-01, beta=3.000e-01, masking_depth=0)]
            p_spl = [MethodParameters(alpha=9.800e-01, masking_depth=1000)]
            p_prm = [MethodParameters(alpha=(1.000e-04, 6.000e-01, 1.000e-04, 1.000e-04))]
            p_eof1 = [MethodParameters(alpha=4.971e-01)]
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
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
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
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
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

        case 'search-import_petro_layer':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import_petro_layer|blob':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/blob']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=3, offset=0)
        case 'search-import_petro_layer|layer+diag':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+diagonal']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import_petro_layer|layer+appendix':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+appendix']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import_petro_layer|bracket':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/bracket']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import_petro_layer+column|layer+column':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer+column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'search-import_petro_layer|cap':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/cap']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'search-import_petro_layer+column|layer':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'search-import_petro_layer|layer+column':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'search-import_petro_layer+vertical|layer+vertical':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer+column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=2, offset=0)
        case 'search-import_petro_layer+vertical|layer':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=2, offset=0)
        case 'search-import_petro_layer|layer+vertical':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=2, offset=0)
        case 'search-import_petro_layer+sparse|layer':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+sparse/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=2, offset=0)
        case 'search-import_petro_layer|layer+sparse':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+sparse']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=2, offset=0)
        case 'search-import_petro_cup|layer':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['cup/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=2, offset=0)

        case 'optimal-import_petro_layer':
            p_bas = [MethodParameters(alpha=9.990e-01)]
            p_svd = [MethodParameters(alpha=6.176e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import_petro_layer|blob':
            p_bas = [MethodParameters(alpha=9.990e-01)]
            p_svd = [MethodParameters(alpha=6.993e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/blob']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=3, offset=0)
        case 'optimal-import_petro_layer|layer+diag':
            p_bas = [MethodParameters(alpha=9.990e-01)]
            p_svd = [MethodParameters(alpha=7.447e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+diagonal']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import_petro_layer|layer+appendix':
            p_bas = [MethodParameters(alpha=9.990e-01)]
            p_svd = [MethodParameters(alpha=7.447e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+appendix']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import_petro_layer|bracket':
            p_bas = [MethodParameters(alpha=9.990e-01)]
            p_svd = [MethodParameters(alpha=4.269e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/bracket']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import_petro_layer+column|layer+column':
            p_bas = [MethodParameters(9.708e-01)]
            p_svd = [MethodParameters(5.904e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer+column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'optimal-import_petro_layer|cap':
            p_bas = [MethodParameters(9.849e-01)]
            p_svd = [MethodParameters(3.089e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/cap']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'optimal-import_petro_layer+column|layer':
            p_bas = [MethodParameters(alpha=9.072e-01)]
            p_svd = [MethodParameters(alpha=1.908e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'optimal-import_petro_layer|layer+column':
            p_bas = [MethodParameters(alpha=9.849e-01)]
            p_svd = [MethodParameters(alpha=2.453e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'optimal-import_petro_layer+vertical|layer+vertical':
            p_bas = [MethodParameters(alpha=9.778e-01)]
            p_svd = [MethodParameters(alpha=3.179e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer+column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=2, offset=0)
        case 'optimal-import_petro_layer+vertical|layer':
            p_bas = [MethodParameters(alpha=9.919e-01)]
            p_svd = [MethodParameters(alpha=2.453e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=2, offset=0)
        case 'optimal-import_petro_layer|layer+vertical':
            p_bas = [MethodParameters(alpha=9.849e-01)]
            p_svd = [MethodParameters(alpha=2.635e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=2, offset=0)
        case 'optimal-import_petro_layer+sparse|layer':
            p_bas = [MethodParameters(alpha=9.919e-01)]
            p_svd = [MethodParameters(alpha=3.543e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+sparse/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=2, offset=0)
        case 'optimal-import_petro_layer|layer+sparse':
            p_bas = [MethodParameters(alpha=9.849e-01)]
            p_svd = [MethodParameters(alpha=2.816e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+sparse']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=2, offset=0)
        case 'optimal-import_petro_cup|layer':
            p_bas = [MethodParameters(alpha=9.849e-01)]
            p_svd = [MethodParameters(alpha=2.181e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['cup/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=2, offset=0)

        case 'meanpar-import_petro_layer':
            p_bas = [MethodParameters(alpha=9.990e-01)]
            p_svd = [MethodParameters(alpha=6.647e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import_petro_layer|blob':
            p_bas = [MethodParameters(alpha=9.990e-01)]
            p_svd = [MethodParameters(alpha=6.647e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/blob']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=3, offset=0)
        case 'meanpar-import_petro_layer|layer+diag':
            p_bas = [MethodParameters(alpha=9.990e-01)]
            p_svd = [MethodParameters(alpha=6.647e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+diagonal']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import_petro_layer|layer+appendix':
            p_bas = [MethodParameters(alpha=9.990e-01)]
            p_svd = [MethodParameters(alpha=6.647e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+appendix']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import_petro_layer|bracket':
            p_bas = [MethodParameters(alpha=9.990e-01)]
            p_svd = [MethodParameters(alpha=6.647e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/bracket']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import_petro_layer+column|layer+column':
            p_bas = [MethodParameters(alpha=9.619e-01)]
            p_svd = [MethodParameters(alpha=3.338e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer+column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'meanpar-import_petro_layer|cap':
            p_bas = [MethodParameters(alpha=9.619e-01)]
            p_svd = [MethodParameters(alpha=3.338e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/cap']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'meanpar-import_petro_layer+column|layer':
            p_bas = [MethodParameters(alpha=9.619e-01)]
            p_svd = [MethodParameters(alpha=3.338e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'meanpar-import_petro_layer|layer+column':
            p_bas = [MethodParameters(alpha=9.619e-01)]
            p_svd = [MethodParameters(alpha=3.338e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'meanpar-import_petro_layer+vertical|layer+vertical':
            p_bas = [MethodParameters(alpha=9.861e-01)]
            p_svd = [MethodParameters(alpha=2.801e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer+column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=2, offset=0)
        case 'meanpar-import_petro_layer+vertical|layer':
            p_bas = [MethodParameters(alpha=9.861e-01)]
            p_svd = [MethodParameters(alpha=2.801e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=2, offset=0)
        case 'meanpar-import_petro_layer|layer+vertical':
            p_bas = [MethodParameters(alpha=9.861e-01)]
            p_svd = [MethodParameters(alpha=2.801e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+column']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=2, offset=0)
        case 'meanpar-import_petro_layer+sparse|layer':
            p_bas = [MethodParameters(alpha=9.861e-01)]
            p_svd = [MethodParameters(alpha=2.801e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+sparse/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=2, offset=0)
        case 'meanpar-import_petro_layer|layer+sparse':
            p_bas = [MethodParameters(alpha=9.861e-01)]
            p_svd = [MethodParameters(alpha=2.801e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+sparse']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=2, offset=0)
        case 'meanpar-import_petro_cup|layer':
            p_bas = [MethodParameters(alpha=9.861e-01)]
            p_svd = [MethodParameters(alpha=2.801e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['cup/layer']
            env_mode = 'import'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=31, width=2000, height=1500)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=2, offset=0)

        case 'search-import_petro-deep_layer':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import_petro-deep_layer|blob':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/blob']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=3, offset=0)
        case 'search-import_petro-deep_layer|layer+diag':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+diagonal']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import_petro-deep_layer|layer+appendix':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+appendix']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import_petro-deep_layer|bracket':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/bracket']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'search-import_petro-deep_layer+column|layer+column':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer+column']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'search-import_petro-deep_layer|cap':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/cap']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'search-import_petro-deep_layer+column|layer':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'search-import_petro-deep_layer|layer+column':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+column']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'search-import_petro-deep_layer+vertical|layer+vertical':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer+column']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=4, offset=0)
        case 'search-import_petro-deep_layer+vertical|layer':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=4, offset=0)
        case 'search-import_petro-deep_layer|layer+vertical':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+column']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=4, offset=0)
        case 'search-import_petro-deep_layer+sparse|layer':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+sparse/layer']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=4, offset=0)
        case 'search-import_petro-deep_layer|layer+sparse':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+sparse']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=4, offset=0)
        case 'search-import_petro-deep_cup|layer':
            p_bas = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.6, 0.9999, 100)]
            p_svd = [MethodParameters(alpha=alpha1) for alpha1 in np.linspace(0.1, 0.8, 100)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['cup/layer']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=4, offset=0)

        case 'optimal-import_petro-deep_layer':
            p_bas = [MethodParameters(alpha=9.918e-01)]
            p_svd = [MethodParameters(alpha=2.061e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import_petro-deep_layer|blob':
            p_bas = [MethodParameters(alpha=6.808e-01)]
            p_svd = [MethodParameters(alpha=1.848e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/blob']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=3, offset=0)
        case 'optimal-import_petro-deep_layer|layer+diag':
            p_bas = [MethodParameters(alpha=6.162e-01)]
            p_svd = [MethodParameters(alpha=1.919e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+diagonal']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import_petro-deep_layer|layer+appendix':
            p_bas = [MethodParameters(alpha=6.000e-01)]
            p_svd = [MethodParameters(alpha=5.101e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+appendix']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import_petro-deep_layer|bracket':
            p_bas = [MethodParameters(alpha=6.000e-01)]
            p_svd = [MethodParameters(alpha=1.636e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/bracket']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'optimal-import_petro-deep_layer+column|layer+column':
            p_bas = [MethodParameters(8.666e-01)]
            p_svd = [MethodParameters(2.909e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer+column']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'optimal-import_petro-deep_layer|cap':
            p_bas = [MethodParameters(7.939e-01)]
            p_svd = [MethodParameters(1.848e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/cap']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'optimal-import_petro-deep_layer+column|layer':
            p_bas = [MethodParameters(alpha=6.808e-01)]
            p_svd = [MethodParameters(alpha=1.848e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'optimal-import_petro-deep_layer|layer+column':
            p_bas = [MethodParameters(alpha=6.000e-01)]
            p_svd = [MethodParameters(alpha=1.495e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+column']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'optimal-import_petro-deep_layer+vertical|layer+vertical':
            p_bas = [MethodParameters(alpha=7.212e-01)]
            p_svd = [MethodParameters(alpha=2.626e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer+column']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=4, offset=0)
        case 'optimal-import_petro-deep_layer+vertical|layer':
            p_bas = [MethodParameters(alpha=9.070e-01)]
            p_svd = [MethodParameters(alpha=2.202e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=4, offset=0)
        case 'optimal-import_petro-deep_layer|layer+vertical':
            p_bas = [MethodParameters(alpha=6.000e-01)]
            p_svd = [MethodParameters(alpha=1.990e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+column']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=4, offset=0)
        case 'optimal-import_petro-deep_layer+sparse|layer':
            p_bas = [MethodParameters(alpha=9.474e-01)]
            p_svd = [MethodParameters(alpha=2.131e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+sparse/layer']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=4, offset=0)
        case 'optimal-import_petro-deep_layer|layer+sparse':
            p_bas = [MethodParameters(alpha=8.504e-01)]
            p_svd = [MethodParameters(alpha=1.990e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+sparse']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=4, offset=0)
        case 'optimal-import_petro-deep_cup|layer':
            p_bas = [MethodParameters(alpha=9.676e-01)]
            p_svd = [MethodParameters(alpha=3.899e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['cup/layer']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=4, offset=0)

        case 'meanpar-import_petro-deep_layer':
            p_bas = [MethodParameters(alpha=6.978e-01)]
            p_svd = [MethodParameters(alpha=2.513e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import_petro-deep_layer|blob':
            p_bas = [MethodParameters(alpha=6.978e-01)]
            p_svd = [MethodParameters(alpha=2.513e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/blob']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=3, offset=0)
        case 'meanpar-import_petro-deep_layer|layer+diag':
            p_bas = [MethodParameters(alpha=6.978e-01)]
            p_svd = [MethodParameters(alpha=2.513e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+diagonal']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import_petro-deep_layer|layer+appendix':
            p_bas = [MethodParameters(alpha=6.978e-01)]
            p_svd = [MethodParameters(alpha=2.513e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+appendix']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import_petro-deep_layer|bracket':
            p_bas = [MethodParameters(alpha=6.978e-01)]
            p_svd = [MethodParameters(alpha=2.513e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/bracket']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=1, offset=0)
        case 'meanpar-import_petro-deep_layer+column|layer+column':
            p_bas = [MethodParameters(alpha=7.353e-01)]
            p_svd = [MethodParameters(alpha=2.025e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer+column']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'meanpar-import_petro-deep_layer|cap':
            p_bas = [MethodParameters(alpha=7.353e-01)]
            p_svd = [MethodParameters(alpha=2.025e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/cap']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'meanpar-import_petro-deep_layer+column|layer':
            p_bas = [MethodParameters(alpha=7.353e-01)]
            p_svd = [MethodParameters(alpha=2.025e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'meanpar-import_petro-deep_layer|layer+column':
            p_bas = [MethodParameters(alpha=7.353e-01)]
            p_svd = [MethodParameters(alpha=2.025e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+column']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=9, offset=0)
        case 'meanpar-import_petro-deep_layer+vertical|layer+vertical':
            p_bas = [MethodParameters(alpha=8.323e-01)]
            p_svd = [MethodParameters(alpha=2.247e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer+column']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=4, offset=0)
        case 'meanpar-import_petro-deep_layer+vertical|layer':
            p_bas = [MethodParameters(alpha=8.323e-01)]
            p_svd = [MethodParameters(alpha=2.247e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+column/layer']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=4, offset=0)
        case 'meanpar-import_petro-deep_layer|layer+vertical':
            p_bas = [MethodParameters(alpha=8.323e-01)]
            p_svd = [MethodParameters(alpha=2.247e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+column']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=4, offset=0)
        case 'meanpar-import_petro-deep_layer+sparse|layer':
            p_bas = [MethodParameters(alpha=8.323e-01)]
            p_svd = [MethodParameters(alpha=2.247e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer+sparse/layer']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=4, offset=0)
        case 'meanpar-import_petro-deep_layer|layer+sparse':
            p_bas = [MethodParameters(alpha=8.323e-01)]
            p_svd = [MethodParameters(alpha=2.247e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['layer/layer+sparse']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=4, offset=0)
        case 'meanpar-import_petro-deep_cup|layer':
            p_bas = [MethodParameters(alpha=8.323e-01)]
            p_svd = [MethodParameters(alpha=2.247e-01)]
            methods_parameters = {
                'classic': p_bas,
                'svd': p_svd
            }
            device_pos_modes = ['cup/layer']
            env_mode='import-deep'
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=50, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=4, offset=0)

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
        'sdA': 'layer',
        'sdB': 'layer|blob',
        'sdC': 'layer|layer+diag',
        'sdD': 'layer|layer+appendix',
        'sdE': 'layer|bracket',
        'sdF': 'layer+column|layer+column',
        'sdG': 'layer|cap',
        'sdH': 'layer+column|layer',
        'sdI': 'layer|layer+column',
        'odJ': 'layer+vertical|layer+vertical',
        'odK': 'layer+vertical|layer',
        'odL': 'layer|layer+vertical',
        'odM': 'layer+sparse|layer',
        'odN': 'layer|layer+sparse',
        'odO': 'cup|layer',
    }
    metric_estimation = False

    simulation_modes = {
        'spA': ['search-import', 'petro', 'layer'],                         # search - petro - layer
        'spB': ['search-import', 'petro', 'layer|blob'],                    # search - petro - layer-blob
        'spC': ['search-import', 'petro', 'layer|layer+diag'],              # search - petro - layer-diag
        'spD': ['search-import', 'petro', 'layer|layer+appendix'],          # search - petro - layer-column
        'spE': ['search-import', 'petro', 'layer|bracket'],                 # search - petro - layer-bracket
        'spF': ['search-import', 'petro', 'layer+column|layer+column'],     # search - petro - surround
        'spG': ['search-import', 'petro', 'layer|cap'],                     # search - petro - layer-cap
        'spH': ['search-import', 'petro', 'layer+column|layer'],            # search - petro - T-layer
        'spI': ['search-import', 'petro', 'layer|layer+column'],            # search - petro - layer-T
        'spJ': ['search-import', 'petro', 'layer+vertical|layer+vertical'], # search - petro - layer+vertical-layer+vertical
        'spK': ['search-import', 'petro', 'layer+vertical|layer'],          # search - petro - layer+vertical-layer
        'spL': ['search-import', 'petro', 'layer|layer+vertical'],          # search - petro - layer-layer+vertical
        'spM': ['search-import', 'petro', 'layer+sparse|layer'],            # search - petro - layer+sparse-layer
        'spN': ['search-import', 'petro', 'layer|layer+sparse'],            # search - petro - layer-layer+sparse
        'spO': ['search-import', 'petro', 'cup|layer'],                     # search - petro - cup-layer

        'opA': ['optimal-import', 'petro', 'layer'],                        # optimal - petro - layer
        'opB': ['optimal-import', 'petro', 'layer|blob'],                   # optimal - petro - surround
        'opC': ['optimal-import', 'petro', 'layer|layer+diag'],             # optimal - petro - layer-blob
        'opD': ['optimal-import', 'petro', 'layer|layer+appendix'],         # optimal - petro - layer-column
        'opE': ['optimal-import', 'petro', 'layer|bracket'],                # optimal - petro - layer-diag
        'opF': ['optimal-import', 'petro', 'layer+column|layer+column'],    # optimal - petro - layer-scolumn
        'opG': ['optimal-import', 'petro', 'layer|cap'],                    # optimal - petro - scolumn-layer
        'opH': ['optimal-import', 'petro', 'layer+column|layer'],           # optimal - petro - semi-surround
        'opI': ['optimal-import', 'petro', 'layer|layer+column'],           # optimal - petro - layer-bracket
        'opJ': ['optimal-import', 'petro', 'layer+vertical|layer+vertical'],# optimal - petro - layer-cap
        'opK': ['optimal-import', 'petro', 'layer+vertical|layer'],         # optimal - petro - T-layer
        'opL': ['optimal-import', 'petro', 'layer|layer+vertical'],         # optimal - petro - layer-T
        'opM': ['optimal-import', 'petro', 'layer+sparse|layer'],           # optimal - petro - layer+sparse-layer
        'opN': ['optimal-import', 'petro', 'layer|layer+sparse'],           # optimal - petro - layer-layer+sparse
        'opO': ['optimal-import', 'petro', 'cup|layer'],                    # optimal - petro - cup-layer

        'mpA': ['meanpar-import', 'petro', 'layer'],                        # optimal - petro - layer
        'mpB': ['meanpar-import', 'petro', 'layer|blob'],                   # optimal - petro - surround
        'mpC': ['meanpar-import', 'petro', 'layer|layer+diag'],             # optimal - petro - layer-blob
        'mpD': ['meanpar-import', 'petro', 'layer|layer+appendix'],         # optimal - petro - layer-column
        'mpE': ['meanpar-import', 'petro', 'layer|bracket'],                # optimal - petro - layer-diag
        'mpF': ['meanpar-import', 'petro', 'layer+column|layer+column'],    # optimal - petro - layer-scolumn
        'mpG': ['meanpar-import', 'petro', 'layer|cap'],                    # optimal - petro - scolumn-layer
        'mpH': ['meanpar-import', 'petro', 'layer+column|layer'],           # optimal - petro - semi-surround
        'mpI': ['meanpar-import', 'petro', 'layer|layer+column'],           # optimal - petro - layer-bracket
        'mpJ': ['meanpar-import', 'petro', 'layer+vertical|layer+vertical'],# optimal - petro - layer-cap
        'mpK': ['meanpar-import', 'petro', 'layer+vertical|layer'],         # optimal - petro - T-layer
        'mpL': ['meanpar-import', 'petro', 'layer|layer+vertical'],         # optimal - petro - layer-T
        'mpM': ['meanpar-import', 'petro', 'layer+sparse|layer'],           # optimal - petro - layer+sparse-layer
        'mpN': ['meanpar-import', 'petro', 'layer|layer+sparse'],           # optimal - petro - layer-layer+sparse
        'mpO': ['meanpar-import', 'petro', 'cup|layer'],                    # optimal - petro - cup-layer

        'sdA': ['search-import', 'petro-deep', 'layer'],                            # search - petro-deep - layer
        'sdB': ['search-import', 'petro-deep', 'layer|blob'],                       # search - petro-deep - layer-blob
        'sdC': ['search-import', 'petro-deep', 'layer|layer+diag'],                 # search - petro-deep - layer-diag
        'sdD': ['search-import', 'petro-deep', 'layer|layer+appendix'],             # search - petro-deep - layer-column
        'sdE': ['search-import', 'petro-deep', 'layer|bracket'],                    # search - petro-deep - layer-bracket
        'sdF': ['search-import', 'petro-deep', 'layer+column|layer+column'],        # search - petro-deep - surround
        'sdG': ['search-import', 'petro-deep', 'layer|cap'],                        # search - petro-deep - layer-cap
        'sdH': ['search-import', 'petro-deep', 'layer+column|layer'],               # search - petro-deep - T-layer
        'sdI': ['search-import', 'petro-deep', 'layer|layer+column'],               # search - petro-deep - layer-T
        'sdJ': ['search-import', 'petro-deep', 'layer+vertical|layer+vertical'],    # search - petro-deep - layer+vertical-layer+vertical
        'sdK': ['search-import', 'petro-deep', 'layer+vertical|layer'],             # search - petro-deep - layer+vertical-layer
        'sdL': ['search-import', 'petro-deep', 'layer|layer+vertical'],             # search - petro-deep - layer-layer+vertical
        'sdM': ['search-import', 'petro-deep', 'layer+sparse|layer'],               # search - petro-deep - layer+sparse-layer
        'sdN': ['search-import', 'petro-deep', 'layer|layer+sparse'],               # search - petro-deep - layer-layer+sparse
        'sdO': ['search-import', 'petro-deep', 'cup|layer'],                        # search - petro-deep - cup-layer

        'odA': ['optimal-import', 'petro-deep', 'layer'],                           # optimal - petro-deep - layer
        'odB': ['optimal-import', 'petro-deep', 'layer|blob'],                      # optimal - petro-deep - layer-blob
        'odC': ['optimal-import', 'petro-deep', 'layer|layer+diag'],                # optimal - petro-deep - layer-diag
        'odD': ['optimal-import', 'petro-deep', 'layer|layer+appendix'],            # optimal - petro-deep - layer-column
        'odE': ['optimal-import', 'petro-deep', 'layer|bracket'],                   # optimal - petro-deep - layer-bracket
        'odF': ['optimal-import', 'petro-deep', 'layer+column|layer+column'],       # optimal - petro-deep - surround
        'odG': ['optimal-import', 'petro-deep', 'layer|cap'],                       # optimal - petro-deep - layer-cap
        'odH': ['optimal-import', 'petro-deep', 'layer+column|layer'],              # optimal - petro-deep - T-layer
        'odI': ['optimal-import', 'petro-deep', 'layer|layer+column'],              # optimal - petro-deep - layer-T
        'odJ': ['optimal-import', 'petro-deep', 'layer+vertical|layer+vertical'],   # optimal - petro-deep - layer+vertical-layer+vertical
        'odK': ['optimal-import', 'petro-deep', 'layer+vertical|layer'],            # optimal - petro-deep - layer+vertical-layer
        'odL': ['optimal-import', 'petro-deep', 'layer|layer+vertical'],            # optimal - petro-deep - layer-layer+vertical
        'odM': ['optimal-import', 'petro-deep', 'layer+sparse|layer'],              # optimal - petro-deep - layer+sparse-layer
        'odN': ['optimal-import', 'petro-deep', 'layer|layer+sparse'],              # optimal - petro-deep - layer-layer+sparse
        'odO': ['optimal-import', 'petro-deep', 'cup|layer'],                       # optimal - petro-deep - cup-layer

        'mdA': ['meanpar-import', 'petro-deep', 'layer'],                           # meanpar - petro-deep - layer
        'mdB': ['meanpar-import', 'petro-deep', 'layer|blob'],                      # meanpar - petro-deep - layer-blob
        'mdC': ['meanpar-import', 'petro-deep', 'layer|layer+diag'],                # meanpar - petro-deep - layer-diag
        'mdD': ['meanpar-import', 'petro-deep', 'layer|layer+appendix'],            # meanpar - petro-deep - layer-column
        'mdE': ['meanpar-import', 'petro-deep', 'layer|bracket'],                   # meanpar - petro-deep - layer-bracket
        'mdF': ['meanpar-import', 'petro-deep', 'layer+column|layer+column'],       # meanpar - petro-deep - surround
        'mdG': ['meanpar-import', 'petro-deep', 'layer|cap'],                       # meanpar - petro-deep - layer-cap
        'mdH': ['meanpar-import', 'petro-deep', 'layer+column|layer'],              # meanpar - petro-deep - T-layer
        'mdI': ['meanpar-import', 'petro-deep', 'layer|layer+column'],              # meanpar - petro-deep - layer-T
        'mdJ': ['meanpar-import', 'petro-deep', 'layer+vertical|layer+vertical'],   # meanpar - petro-deep - layer+vertical-layer+vertical
        'mdK': ['meanpar-import', 'petro-deep', 'layer+vertical|layer'],            # meanpar - petro-deep - layer+vertical-layer
        'mdL': ['meanpar-import', 'petro-deep', 'layer|layer+vertical'],            # meanpar - petro-deep - layer-layer+vertical
        'mdM': ['meanpar-import', 'petro-deep', 'layer+sparse|layer'],              # meanpar - petro-deep - layer+sparse-layer
        'mdN': ['meanpar-import', 'petro-deep', 'layer|layer+sparse'],              # meanpar - petro-deep - layer-layer+sparse
        'mdO': ['meanpar-import', 'petro-deep', 'cup|layer'],                       # meanpar - petro-deep - cup-layer
    }

    classic_params = [get_simulation_mode('_'.join(simulation_modes[sim_code]), environment_parameters, device_parameters, env_mode)[2]['classic'][0].alpha for sim_code in device_pos_modes.keys()]
    svd_params = [get_simulation_mode('_'.join(simulation_modes[sim_code]), environment_parameters, device_parameters, env_mode)[2]['svd'][0].alpha for sim_code in device_pos_modes.keys()]
    mean_alpha = np.mean(classic_params)    # 9.839e-01
    mean_beta = np.mean(svd_params)         # 4.166e-01
    median_alpha = np.median(classic_params)  # 9.849e-01
    median_beta = np.median(svd_params)  # 3.179e-01
    min_alpha = np.amin(classic_params)
    min_beta = np.amin(svd_params)
    max_alpha = np.amax(classic_params)
    max_beta = np.amax(svd_params)
    mean_alpha_1 = np.mean(classic_params[:5])  # 6.978e-01
    mean_beta_1 = np.mean(svd_params[:5])       # 2.513e-01
    mean_alpha_2 = np.mean(classic_params[5:9]) # 7.353e-01
    mean_beta_2 = np.mean(svd_params[5:9])      # 2.025e-01
    mean_alpha_3 = np.mean(classic_params[9:])  # 8.323e-01
    mean_beta_3 = np.mean(svd_params[9:])       # 2.247e-01
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
