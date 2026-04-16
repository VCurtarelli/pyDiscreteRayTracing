from threading import settrace

from funs_device_positioning import position_devices, export_pos_devices
from py_libs import *
from class_Environment import Environment, EstEnvironment
import class_Environment as clsenv
import sys
from funs_encode_and_hash import make_folders_and_hash
from dataclasses import dataclass
from typing import Any
from numpy import ndarray
import copy
import seaborn as sns

np.set_printoptions(legacy='1.25',precision=2,linewidth=600,threshold=sys.maxsize)
decimal.getcontext().prec = 2

rms = lambda x: np.sqrt(np.mean(x**2))
def main(env_params, dev_positioning,
         param_bas, param_trd, param_spl, param_prm, sigma=0.0, show_path=False):

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
        'param_prm': param_prm,
    }

    code, direc = make_folders_and_hash(n_rays, parameters)
    export_pos_devices(pos_sources, pos_receivers, direc + code + '/', width, height)

    ## Main iteration loop
    est_envs = iteration_loop(code, est_envs, model_slowness, n_rays, obs_env, obs_times, param_bas, param_spl, param_trd, param_prm)

    ## Print metrics for on-execution assessment
    print('Travel-time error:')
    for env_idx, est_env in enumerate(est_envs):
        print('\t{}. - {:.4f}ms'.format(est_env.field_name[:3], 1000 * est_env.est_time_mse[-1]),
              'out of {:.4f}s'.format(np.mean(obs_times)))
    print('Field RMS error:')
    for env_idx, est_env in enumerate(est_envs):
        print('\t{}. - {:.4f}m/s'.format(est_env.field_name[:3], est_env.est_ssf_rms[-1]),
              'with σ = {:.4f}m/s'.format(np.std(obs_env.field)))
    print('Optimal parameters:')
    for env_idx, est_env in enumerate(est_envs):
        print('\t{}. - {}'.format(est_env.field_name[:3], est_env.format_params()))
    print()

    show_estimated_profiles(env_params, est_envs)


def generate_environments(height, num_cells_x, num_cells_y, pos_receivers: list[tuple[Any, Any]],
                          pos_sources: list[tuple[Any, Any]], width) -> tuple[list[EstEnvironment], Environment]:
    obs_env = Environment(num_cells_x, num_cells_y, width, height)
    obs_env.generate_rays(pos_sources, pos_receivers)
    # obs_env.plot_curves()

    est_envs = []
    for field_name in ['Basic', 'Trade', 'Split', 'Munk']:
        est_env = EstEnvironment(num_cells_x, num_cells_y, width, height, None, field_name=field_name)
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
                   n_rays: int, obs_env: Environment, obs_times: list[Any], param_bas, param_spl, param_trd, param_prm):
    it_idx = 1

    ## Export observed environment
    obs_env.field_to_csv(0, export_params=True, code=code)
    print('\n' * 10)
    param_list = [param_bas, param_trd, param_spl, param_prm]
    mode_list = ['classic', 'trade', 'split', 'munk']
    out_est_envs = []
    for idx in range(len(est_envs)):
        it_idx = 0
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
                    est_env_it.calc_metrics(obs_times, obs_env.field)
                except (RuntimeError, np.linalg.LinAlgError):
                    est_env_it.est_time_mse.append(np.inf)
                    est_env_it.est_ssf_rms.append(np.inf)
                    break
                est_env_it.calc_metrics(obs_times, obs_env.field)
                it_idx += 1
                if it_idx >= 5:
                    break

                est_env_it.field_to_csv(it_idx, code=code)
                est_env_it.field_to_csv(it_idx, comp=obs_env.field, code=code, vmin=-20, vmax=20, export_params=True)
                est_env_it.export_metrics(code=code)
            # print(param, est_env_cp.est_ssf_rms[-1])
            if best_est_env is not None:
                print('\t'+est_env_it.format_params())
                print('\t{:.4f} - {:.4f}'.format(est_env_it.est_ssf_rms[-1], best_est_env.est_ssf_rms[-1]))
            if best_est_env is None:
                best_est_env = est_env_it
            elif est_env_it.est_ssf_rms[-1] < best_est_env.est_ssf_rms[-1]:
                best_est_env = est_env_it
        out_est_envs.append(best_est_env)
    return out_est_envs

    # while True:
    #     print("Iteration {}".format(it_idx))
    #
    #     ## Iterate estimated fields
    #     est_envs[0].iterate_field(n_rays, alpha=param_bas.alpha, obs_times=obs_times,
    #                               mode='classic')
    #     est_envs[1].iterate_field(n_rays, alpha=param_trd.alpha, beta=param_trd.beta, obs_times=obs_times,
    #                               model_slowness=model_slowness, masking_depth=param_trd.masking_depth,
    #                               mode='trade')
    #     est_envs[2].iterate_field(n_rays, alpha=param_spl.alpha, obs_times=obs_times,
    #                               model_slowness=model_slowness, masking_depth=param_spl.masking_depth,
    #                               mode='split')
    #     est_envs[3].iterate_field(n_rays, alpha=param_prm.alpha, obs_times=obs_times,
    #                               mode='munk')
    #
    #     ## Calculate and export metrics, export estimated environment
    #     for est_env in est_envs:
    #         est_env.calc_metrics(obs_times, obs_env.field)
    #
    #     err_times = []
    #     for est_env in est_envs:
    #         est_env.field_to_csv(it_idx, code=code)
    #         est_env.field_to_csv(it_idx, comp=obs_env.field, code=code, vmin=-20, vmax=20, export_params=True)
    #         est_env.export_metrics(code=code)
    #
    #         err_times.append(est_env.cost_function(est_env.rays, obs_times))
    #
    #     ## Loop end-condition
    #     if it_idx == 5:
    #         print("LIMIT REACHED - Code " + code)
    #         break
    #     it_idx += 1


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
        v_min = min(v_min, np.amin(env.field))
        v_max = max(v_max, np.amax(env.field))
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


def show_split_munk():
    dph = 1300 * np.linspace(0, 2, 30)
    mv1, mv2, mv3 = clsenv.mod_munk_profile(0.00737, 1500, 1300, 1, dph, return_separated=True)
    plt.plot(mv1, dph, label='cst')
    plt.plot(mv2, dph, label='lin')
    plt.plot(mv3, dph, label='exp')
    plt.plot(mv1+mv2+mv3,dph,label='sum')
    plt.ylim(np.amax(dph), np.amin(dph))
    plt.legend()
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
    n_travels: int
    offset: int

@dataclass
class MethodParameters:
    alpha: float | tuple[float | int, ...]
    beta: float = None
    masking_depth: int = None



if __name__ == '__main__':
    simulation_mode = 'small'
    match simulation_mode:
        case 'small':
            environment_parameters = EnvParameters(num_cells_x=9, num_cells_y=25, width=2000, height=5000)
            device_parameters = DeviceParameters(n_sources=9, n_receivers=9, n_travels=5, offset=0)
        case 'mid':
            environment_parameters = EnvParameters(num_cells_x=25, num_cells_y=35, width=2000, height=2000)
            device_parameters = DeviceParameters(n_sources=8, n_receivers=7, n_travels=5, offset=0)
        case _:
            environment_parameters = EnvParameters(num_cells_x=43, num_cells_y=50, width=2000, height=2000)
            device_parameters = DeviceParameters(n_sources=15, n_receivers=15, n_travels=5, offset=0)

    # show_profiles(environment_parameters, True)
    device_pos_modes = [
        # 'layer/layer',
        # 'layer/blob',
        'layer/column+blob',
        # 'layer+shallowcolumn/blob',
    ]

    parameter_sweep_mode = 'single'
    masking_depth = 0
    if parameter_sweep_mode == 'sweep':
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
        # print(p_bas[6])
        # print(p_trd[65])
        # print(p_spl[1])
        # print(p_prm[736])
        # input()
    elif parameter_sweep_mode == 'test':
        p_bas = [MethodParameters(alpha=0.65333)]
        p_trd = [MethodParameters(alpha=0.6, beta=0.1666, masking_depth=masking_depth)]
        p_spl = [MethodParameters(alpha=0.10888, masking_depth=masking_depth)]
        p_prm = [MethodParameters(alpha=(0.0031, 0.9, 0.0001, 0.00031))]
    else:
        p_bas = [MethodParameters(alpha=0.762)]
        p_trd = [MethodParameters(alpha=0.7,beta=0.067, masking_depth=masking_depth)]
        p_spl = [MethodParameters(alpha=0.762, masking_depth=masking_depth)]
        p_prm = [MethodParameters(alpha=(0.005, 0.9995, 0.00005, 0.00008))]

    for device_pos_mode in device_pos_modes:
        device_positioning = position_devices(device_pos_mode, environment_parameters, device_parameters)
        sig = 0.000

        np.set_printoptions(legacy='1.25', precision=6, linewidth=320)
        decimal.getcontext().prec = 2
        main(environment_parameters, device_positioning, p_bas, p_trd, p_spl, p_prm, sig, show_path=False)
