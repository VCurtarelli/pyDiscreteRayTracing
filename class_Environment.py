from matplotlib.colors import hsv_to_rgb

from class_Ray import Ray
from py_libs import *
import pickle
from fun_truncated_svd import truncated_svd
from funs_generate_env_fields import generate_temp_field, generate_sal_field

rms = lambda x: np.sqrt(np.mean(x ** 2))
alpha_rp = lambda x: x/np.sqrt(1-x**2)

class Environment:
    def __init__(self, num_cells_x, num_cells_y, width, height, field_name='Observed', mode=None):
        # Initiate input parameters
        self.cells_nx = num_cells_x
        self.cells_ny = num_cells_y
        self.width = width
        self.height = height

        self.cells_n = num_cells_x * num_cells_y
        self.cell_width = width / num_cells_x
        self.cell_height = height / num_cells_y

        # Initiate grids, velocities, and rays
        self.range_x = self.cell_width * (0.5 + np.arange(num_cells_x))
        self.range_y = self.cell_height * (0.5 + np.arange(num_cells_y))
        self.vx = np.zeros([num_cells_y, num_cells_x])
        self.vy = np.zeros([num_cells_y, num_cells_x])

        self.traveled_dists = None
        self.field = self.generate_field(mode=mode)
        self.true_field = np.copy(self.field)

        self.field_name = field_name
        self.rays = None
        self.sources = None
        self.receivers = None
        self.n_its = 10

    def generate_rays(self, pos_sources, pos_receivers, color=(180, 180, 0), marker='1'):
        # Generates rays from source to receiver as Ray class
        rays = []
        for idx, source in enumerate(pos_sources):
            for jdx, receiver in enumerate(pos_receivers):
                if source == receiver:
                    continue
                ray = Ray(source, receiver, self.field_name, color=color, marker=marker)
                rays.append(ray)
        n_rays = len(rays)
        for idx, ray in enumerate(rays):
            ray.color = hsv_to_rgb((idx / n_rays, 0.7, 0.7))
        self.rays = rays
        self.sources = pos_sources
        self.receivers = pos_receivers
        self.update_rays()
        return rays

    def generate_field(self, mode=None, temperature_field=None, salinity_field=None):
        width = self.width
        height = self.height

        # Generate mesh grid for X and Y directions
        x = self.range_x
        y = self.range_y
        X, Y = np.meshgrid(x, y)  # X-Y plane grid

        # temperature_field = 4 + (32-5) * np.exp(-Y / 600) * (1 - 0.2 * (X / 4000))
        # salinity_field = 34.5 + (1 + 0.5 * (X / 4000)) / (1 + np.exp((Y - 500)/200))

        if temperature_field is None:
            temperature_field = generate_temp_field(X, Y, x_fac=0.2)
        if salinity_field is None:
            salinity_field = generate_sal_field(X, Y, x_fac=0.5)
        # salinity_field = 34.5 + (1.5- 0.5 * (X / 2000)) / (1 + np.exp((Y - 500)/200))
        depth_field = Y
        if mode is None:
            mode = 'import'
        match mode:
            case 'munk':
                velocity_field = mod_munk_profile(0.00737, 1500, 1300, 1300, 0,1, Y, x_mat=X)
            case 'mackenzie':
                velocity_field = mackenzie_profile(temperature_field, salinity_field, depth_field, amp=0)
            case 'delgrosso':
                velocity_field = delgrosso_profile(temperature_field, salinity_field, depth_field, amp=0)
            case 'wilson':
                velocity_field = wilson_profile(temperature_field, salinity_field, depth_field, amp=0)
            case 'import':
                with open('input/ssf---36.375---50.375--50m--1500m.dat', 'rb') as f:
                    data = pkl.load(f)
                    velocity_field = np.array(data['velocity_yearly'])
                    depth = data['depth']
            case 'import-deep':
                with open('input/ssf---36.375---38.375--100m--4900m.dat', 'rb') as f:
                    data = pkl.load(f)
                    velocity_field = np.array(data['velocity'])
                    depth = data['depth']
            case _:
                raise NotImplementedError("Needs to implement other types of field")
        if mode == 'import':
            with open('midput/ssf---36.375---50.375--50m--1500m--eofs.dat', 'rb') as f:
                data = pkl.load(f)
        if mode == 'import-deep':
            with open('midput/ssf---36.375---38.375--100m--4900m--eofs.dat', 'rb') as f:
                data = pkl.load(f)
        mean_slowness = data['mean']
        mean_velocity = 1/mean_slowness
        mean_velocity = np.concatenate(self.cells_nx*[mean_velocity.reshape(-1, 1)], axis=1)
        self.vy = mean_velocity
        # self.vy = mod_munk_profile(0.00737, 1500, 1300, 1300, 0, 1, Y)
        self.vx = velocity_field - self.vy
        return velocity_field

    def plot_curves(self, name=None, rays=None, sources=None, receivers=None,
                    ax=None, show_field=None, show_path=False, legend=False, vs=None, cmap=None):
        if rays is None:
            rays = self.rays
        if sources is None:
            sources = self.sources
        if receivers is None:
            receivers = self.receivers
        # Function for plotting the heatmaps
        width = self.width
        height = self.height
        field = self.field

        if ax is None:
            fig, ax = plt.subplots()
        if show_field is None:
            show_field = field

        if vs is None:
            vmin = np.amin(show_field)
            vmax = np.amax(show_field)
            vs = (vmin, vmax)
        vmin, vmax = vs
        if cmap is None:
            cmap = 'viridis'

        if not (show_field is False):

            alpha = np.ones_like(show_field)
            if (show_field == field).all():
                alpha[show_field <= 1] = 0
                background = np.ones_like(show_field)
                background[alpha == 0] = 0
                ax.imshow(background, extent=(0, width, 0, height), cmap='gray', vmin=0, vmax=1)
            pcm = ax.imshow(show_field[::-1], extent=(0, width, 0, height), alpha=alpha,
                            vmin=vmin, vmax=vmax,
                            cmap=cmap)
            plt.colorbar(pcm, ax=ax)

        for ray in rays:
            path = np.array(ray.path)
            ax.plot(path[:, 0], path[:, 1],
                    color=ray.color * (1 if ray.converged else 0.5), label=np.around(ray.time, 4),  # marker=ray.marker,
                    alpha=0.4)

        for receiver in receivers:
            ax.plot(receiver[0], receiver[1], marker='x', markerfacecolor='blue', markersize=6,
                    markeredgecolor='blue', markeredgewidth=3)
        for source in sources:
            ax.plot(source[0], source[1], marker='+', markerfacecolor='red', markersize=6,
                    markeredgecolor='red', markeredgewidth=2)

        # final plotting setup
        ax.set_xlim((-.05 * width, width + .05 * width))
        ax.set_ylim((-.05 * height, height + .05 * height))

        ax.plot([0, width], [500, 500], 'k')
        ax.plot([0, width], [1000, 1000], 'k')
        ax.yaxis.set_inverted(True)

        if legend:
            ax.legend(loc='upper left')
        plt.title(name)
        plt.show()

    def field_to_csv(self, idx, direc='Results/', export_params=False, comp=None, vmin=None, vmax=None, code='', mode='xyv'):
        # Export field to a csv (ordered to follow own LaTeX formatting for heatmaps)
        field = self.field
        grid_x = self.range_x - self.range_x[0]
        grid_y = self.range_y - self.range_y[0]
        if comp is not None:
            field = field - comp
        idx_name = '0_' if idx == 0 else ('it' + str(idx) + '_') if isinstance(idx, int) else str(idx) + '_'
        if mode == 'xyv':
            txt = ['y,x,val']
            for y_idx in range(self.cells_ny + 1):
                if y_idx == self.cells_ny:
                    y_coord = self.height
                    y_idx -= 1
                else:
                    y_coord = grid_y[y_idx]
                for x_idx in range(self.cells_nx + 1):
                    if x_idx == self.cells_nx:
                        x_coord = self.width
                        x_idx -= 1
                    else:
                        x_coord = grid_x[x_idx]
                    velocity = field[y_idx, x_idx]
                    txt.append('{:.4f},{:.4f},{:.4f}'.format(y_coord, x_coord, velocity))
            txt = '\n'.join(txt)
            filename = code + '/' + idx_name + 'field_' + self.field_name.lower()
            if comp is not None:
                filename += '_comp'
            with open(direc + filename + '.csv', 'w') as f:
                f.write(txt)
                f.close()

        elif mode == 'yx':
            txt = [','.join(['y'] + [str(col) for col in range(self.cells_nx)])]
            for y_idx in range(self.cells_ny):
                # if y_idx == self.cells_ny:
                #     y_coord = self.height
                #     y_idx -= 1
                # else:
                y_coord = grid_y[y_idx] + self.cell_height/2
                x_velocity = list(np.around(field[y_idx, :], 4))
                x_velocity = ','.join([str(v) for v in x_velocity])
                txt.append(str(y_coord) + ',' + x_velocity)
            txt = '\n'.join(txt)
            filename = code + '/' + idx_name + 'profiles_' + self.field_name.lower()
            if comp is not None:
                filename += '_comp'
            with open(direc + filename + '.csv', 'w') as f:
                f.write(txt)
                f.close()

        elif mode == 'mean':
            txt = ['y,val']
            for y_idx in range(self.cells_ny + 1):
                if y_idx == self.cells_ny:
                    y_coord = self.height
                    y_idx -= 1
                else:
                    y_coord = grid_y[y_idx]
                velocity = np.mean(field[y_idx,:])
                txt.append('{:.4f},{:.4f}'.format(y_coord, velocity))
            txt = '\n'.join(txt)
            filename = code + '/' + idx_name + 'mean_profile_' + self.field_name.lower()
            if comp is not None:
                filename += '_comp'
            with open(direc + filename + '.csv', 'w') as f:
                f.write(txt)
                f.close()

        if export_params:
            filename = code + '/' + 'params'
            if comp is not None:
                filename += '_comp'
            if vmin is None:
                vmin = 5 * np.floor((np.amin(field) - 0.2 * np.std(field)) / 5)
            if vmax is None:
                vmax = 5 * np.ceil((np.amax(field) + 0.2 * np.std(field)) / 5)
            nrows = self.cells_ny + 1
            ncols = self.cells_nx + 1
            txt = [
                r'\def\zmin{' + str(vmin) + r'}',
                r'\def\zmax{' + str(vmax) + r'}',
                r'\def\nrows{' + str(nrows) + r'}',
                r'\def\ncols{' + str(ncols) + r'}',
                r'\def\pyNx{' + str(ncols) + r'}',
                r'\def\pyNz{' + str(nrows) + r'}',
                r'\def\pyN{' + str(nrows * ncols) + r'}',
                r'\def\pyNs{' + str(len(self.sources)) + r'}',
                r'\def\pyNr{' + str(len(self.receivers)) + r'}',
                r'\def\pyNits{' + str(self.n_its) + r'}',
            ]
            txt = '\n'.join(txt)
            with open(direc + filename + '.tex', 'w') as f:
                f.write(txt)
                f.close()

    def update_rays(self):
        # Updates the rays path and travel-time given the current SSF
        for ray in self.rays:
            ray.calc_path(self)
            ray.calc_time(self)

    def update_traveled_dists(self, rays, n_rays):
        # Updates traveled distances matrix (R)
        num_cells = self.cells_n
        traveled_dists = np.zeros([n_rays, num_cells])
        for idx, ray in enumerate(rays):
            ray.calc_path(self)
            _, Lengths = ray.calc_time(self)
            traveled_dists[idx, :] = Lengths.reshape(-1, )
        self.traveled_dists = traveled_dists

        return traveled_dists

    def calc_rho(self, _field, _dx=None, _dy=None): # calculate roughness
        if _dx is None:
            _dx = self.cell_width
        if _dy is None:
            _dy = self.cell_height
        rho_x = (_field[:, :-2] - 2 * _field[:, 1:-1] + _field[:, 2:]) ** 2
        rho_y = (_field[:-2, :] - 2 * _field[1:-1, :] + _field[2:, :]) ** 2
        rho = np.sum(rho_x) / _dx ** 2 + np.sum(rho_y) / _dy ** 2
        return rho

    def calculate_optimal_metrics(self):
        self.update_traveled_dists(self.rays, len(self.rays))

        R = self.traveled_dists
        # condition reciprocal
        _, S, _ = svd(R)
        s1 = np.amax(S)
        sR = np.amin(S[S > 1e-16])
        K = sR/s1

        # rank
        rank = np.linalg.matrix_rank(R)

        # stable rank
        s_rank = np.sum(S**2) / s1**2

        # entropic rank
        P = S / np.sum(np.abs(S))
        p = np.pow(P, -P)
        e_rank = np.prod(p)

        # resolution
        curlR = np.linalg.pinv(R) @ R
        D = norm(np.diag(curlR))**2 / norm(curlR, ord='fro')

        # column-coherence
        alpha = 0
        N = R.shape[1]
        for i_idx in range(R.shape[1]):
            rho_i = R[:, i_idx].reshape(-1, 1)
            for j_idx in range(i_idx, R.shape[1]):
                rho_j = R[:, j_idx].reshape(-1, 1)
                alpha_ij = (rho_i.T @ rho_j / (norm(rho_i) * norm(rho_j) + 1e-16)).item()
                alpha_ij = 2 / (N*(N-1)) * alpha_ij
                # print(alpha_ij)
                alpha += alpha_ij

        # row-coherence
        A = 0
        M = R.shape[0]
        for i_idx in range(R.shape[0]):
            rho_i = R[i_idx, :].reshape(-1, 1)
            for j_idx in range(i_idx, R.shape[0]):
                rho_j = R[j_idx, :].reshape(-1, 1)
                a_ij = (rho_i.T @ rho_j / (norm(rho_i) * norm(rho_j) + 1e-16)).item()
                A_ij = 2 / (M*(M-1)) * a_ij * (-np.e*np.log(a_ij + 1e-16))
                A += A_ij

        metrics = {'rcond': K,
                   'rank': rank,
                   's_rank': s_rank,
                   'e_rank': e_rank,
                   'resolution': D,
                   'c_coherence': alpha,
                   'r_coherence': A}
        return metrics


class EstEnvironment(Environment):
    def __init__(self, num_cells_x, num_cells_y, width, height, rays, initial_value=1100, field_name='Estimate', mode=None):
        # Initiate superclass
        super().__init__(num_cells_x, num_cells_y, width, height, rays, mode=mode)
        self.field_name = field_name

        # Update field to be uniform, generate laplacian (D) and blur (B) matrices
        self.slowness_vector = self.update_field((1 / initial_value) * np.ones([self.field.size, 1]))
        self.optimization_vector = None
        self.depth_mask = None
        self.laplacian_mtx = self.generate_laplacian()
        self.blur_mtx = self.generate_blur(sigma=0.00)
        self.est_time_mse = []
        self.est_ssf_rms = []
        self.est_rho_rms = []
        self.params = None


    def set_parameters(self, params):
        self.params = params

    def generate_laplacian(self):
        num_cells_x = self.cells_nx
        num_cells_y = self.cells_ny
        num_cells = self.cells_n
        ## --------------------
        # LAPLACIAN MATRIX CONSTRUCTION
        D_laplacian = np.zeros([num_cells, num_cells])
        laplacian_block = np.zeros([num_cells_x, num_cells_x])
        for i in range(num_cells_x):
            u = np.clip(i + 1, 0, num_cells_x - 1)
            d = np.clip(i - 1, 0, num_cells_x - 1)
            laplacian_block[i, u] = 1
            laplacian_block[i, d] = 1
            laplacian_block[i, i] = -4

        for i in range(num_cells_y):
            if i > 0:
                D_laplacian[(i - 1) * num_cells_x:i * num_cells_x, i * num_cells_x:(i + 1) * num_cells_x] = np.eye(
                    num_cells_x)
                D_laplacian[i * num_cells_x:(i + 1) * num_cells_x, (i - 1) * num_cells_x:i * num_cells_x] = np.eye(
                    num_cells_x)
            D_laplacian[i * num_cells_x:(i + 1) * num_cells_x, i * num_cells_x:(i + 1) * num_cells_x] = laplacian_block
        for i in range(num_cells):
            D_laplacian[i, i] = -(np.sum(D_laplacian[i, :]) - D_laplacian[i, i])
            D_laplacian[i, :] = -D_laplacian[i, :] / np.abs(D_laplacian[i, i])
            pass
        self.laplacian_mtx = D_laplacian
        return D_laplacian

    def generate_blur(self, sigma=1.):
        num_cells_x = self.cells_nx
        num_cells_y = self.cells_ny
        num_cells = self.cells_n
        blur = np.zeros([num_cells, num_cells])
        block_A = np.zeros([num_cells_x, num_cells_x])
        block_B = np.zeros([num_cells_x, num_cells_x])
        for i in range(num_cells_x):
            u = np.clip(i + 1, 0, num_cells_x - 1)
            d = np.clip(i - 1, 0, num_cells_x - 1)
            if sigma == 0:
                val = 0
            else:
                val = np.exp(-1 / (2 * sigma ** 2))

            block_A[i, u] = val
            block_A[i, d] = val
            block_A[i, i] = 1
            block_B[i, u] = val ** 2
            block_B[i, d] = val ** 2
            block_B[i, i] = val

        for i in range(num_cells_y):
            if i > 0:
                blur[(i - 1) * num_cells_x:i * num_cells_x, i * num_cells_x:(i + 1) * num_cells_x] = block_B
                blur[i * num_cells_x:(i + 1) * num_cells_x, (i - 1) * num_cells_x:i * num_cells_x] = block_B
            blur[i * num_cells_x:(i + 1) * num_cells_x, i * num_cells_x:(i + 1) * num_cells_x] = block_A
        self.blur_mtx = blur
        return blur

    def generate_mask(self, depth=1000):
        if depth is None:
            self.depth_mask = np.zeros([self.field.size, self.field.size])
            return None
        depth_mask_mtx = np.zeros_like(self.field)
        depth_mask_mtx[np.arange(self.cells_ny) > depth / self.cell_height, :] = 1
        # J[0, :] = 1
        depth_mask_mtx = depth_mask_mtx.reshape(-1, )
        depth_mask_mtx = np.diagflat(depth_mask_mtx)
        self.depth_mask = depth_mask_mtx

        return None

    def update_field(self, z):
        z[z <= 0] = np.median(z[z > 0])
        self.slowness_vector = z
        self.field = 1 / z.reshape(self.cells_ny, -1)
        return z

    def iterate_field(self, n_rays, alpha=None, beta=None, obs_times: np.ndarray = 0,
                      model_slowness=None, masking_depth=None, mode='classic'):
        # Iterate field (based on which method is to be used)
        # TODO: Here is where other iteration methods are to be implemented

        def normalize(mat):
            norm_fac = np.linalg.norm(mat, ord=2)
            norm_mat = mat / norm_fac
            return norm_mat, norm_fac

        def iterate_field_classic(_obs_times, _alpha):
            _R = self.traveled_dists
            _D = self.laplacian_mtx
            _n_alpha = alpha_rp(_alpha)
            _R, _facR = normalize(_R)
            _D, _FacD = normalize(_D)

            _kernel = _R.T @ _R + _n_alpha ** 2 * _D.T @ _D
            _ikernel = inv(_kernel)

            _z = (1 / _facR) * (_ikernel @ _R.T @ _obs_times)
            return _z

        def iterate_field_svd(_obs_times, _alpha):
            _R = self.traveled_dists
            _D = self.laplacian_mtx

            _U, _S, _V, _rank = truncated_svd(_R, _alpha, True)
            _V2 = _V[:, _rank:]

            s0 = pinv(_R) @ _obs_times
            _z = (np.eye(_D.shape[0]) - _V2 @ pinv(_D @ _V2) @ _D) @ s0
            return _z


        def iterate_field_trade(_obs_times, _model_slowness, _alpha, _beta, _masking_depth):
            if _model_slowness is None:
                _model_slowness = np.ones_like(self.slowness_vector) * 1 / 1500
            _R = self.traveled_dists
            _D = self.laplacian_mtx
            self.generate_mask(depth=_masking_depth)
            _J = self.depth_mask
            _z0 = model_slowness

            _n_alpha = alpha_rp(_alpha)
            _n_beta = alpha_rp(_beta)
            _R, _facR = normalize(_R)
            _D, _FacD = normalize(_D)
            _J, _FacJ = normalize(_J)

            _kernel = _R.T @ _R + _n_alpha ** 2 * _D.T @ _D + _n_beta ** 2 * _J.T @ _J
            _ikernel = inv(_kernel)
            _z = _ikernel @ ((1 / _facR) * _R.T @ _obs_times
                             + _n_beta ** 2 * _J.T @ _J @ _z0
                             )
            return _z

        def iterate_field_split(_obs_times, _model_slowness, _alpha, _masking_depth):
            if _model_slowness is None:
                _model_slowness = np.ones_like(self.slowness_vector) * 1 / 1500
            _R = self.traveled_dists
            _D = self.laplacian_mtx
            _n_alpha = alpha_rp(_alpha)
            _R, _facR = normalize(_R)
            _D, _FacD = normalize(_D)

            _masking_idx = int(_masking_depth / self.height * self.cells_n)

            _R1 = _R[:, :_masking_idx]
            _R2 = _R[:, _masking_idx:]
            _D1 = _D[:, :_masking_idx]
            _D2 = _D[:, _masking_idx:]
            _z2 = _model_slowness[_masking_idx:].reshape(-1, 1)

            _kernel = _R1.T @ _R1 + _n_alpha ** 2 * (_D1.T @ _D1)
            _ikernel = inv(_kernel)

            _z1 = _ikernel @ ((1 / _facR) * _R1.T @ _obs_times
                              - _R1.T @ _R2 @ _z2
                              - _n_alpha ** 2 * _D1.T @ _D2 @ _z2)

            _z = np.concatenate((_z1, _z2), axis=0)
            return _z

        def iterate_field_param(_obs_times, _alpha):
            _opt_param = self.optimization_vector
            if _opt_param is None:
                _opt_param = np.ones([4 * self.cells_nx, 1], dtype=float)
                _opt_param[:self.cells_nx] = 1100
                _opt_param[self.cells_nx:2 * self.cells_nx] = 0.05
                _opt_param[2 * self.cells_nx:3 * self.cells_nx] = 1000
                _opt_param[3 * self.cells_nx:] = 1000
            pass
            _R = self.traveled_dists
            # _R, _facR = normalize(_R)
            _opt_param, _velocity_field = parameterized_optimize(_opt_param, self.range_x, self.range_y, _obs_times, _R, norm_alphas=_alpha)
            self.optimization_vector = _opt_param
            _z = 1/_velocity_field
            return _z

        def iterate_field_eof_r1(_obs_times, _alpha, _dir_eof_file):
            _nx = self.cells_nx
            _nz = self.cells_ny
            _R = self.traveled_dists
            _R, _facR = normalize(_R)
            with open(_dir_eof_file, 'rb') as f:
                _data = pickle.load(f)
                _mean_vector = _data['mean']
                _eof_matrix = _data['eigenbase']
                # _corr_matrix = _data['corr_matrix']
            _n_eigenvectors = _eof_matrix.shape[1]
            _G = _eof_matrix.shape[1]
            _eof_matrix_kp = np.kron(_eof_matrix, np.eye(_nx))
            _mean_vector_ssf = np.concatenate(_nx*[_mean_vector.reshape(-1, 1)], axis=1).reshape(-1, 1)
            _n_alpha = alpha_rp(_alpha)
            _V = _eof_matrix_kp
            _kernel = _V.T @ _R.T @ _R @ _V + _n_alpha**2 * np.eye(_G*_nx)
            _ikernel = inv(_kernel)
            _a = _ikernel @ _V.T @ _R.T @ ( (1/_facR) * _obs_times - _R @ _mean_vector_ssf)
            _z = _mean_vector_ssf + _V @ _a
            return _z

        def iterate_field_eof_r2(_obs_times, _alpha, _dir_eof_file):
            _nx = self.cells_nx
            _nz = self.cells_ny
            _R = self.traveled_dists
            _R, _facR = normalize(_R)
            _D = self.laplacian_mtx
            _D, _FacD = normalize(_D)
            with open(_dir_eof_file, 'rb') as f:
                _data = pickle.load(f)
                _mean_vector = _data['mean']
                _eof_matrix = _data['eigenbase']
                # _corr_matrix = _data['corr_matrix']
            _V = np.kron(_eof_matrix, np.eye(_nx))
            _s = np.concatenate(_nx * [_mean_vector.reshape(-1, 1)], axis=1).reshape(-1, 1)
            _n_alpha = alpha_rp(_alpha)
            _kernel = _V.T @ (_R.T @ _R + _n_alpha**2 * _D.T @ _D) @ _V
            _ikernel = inv(_kernel)
            _a = _ikernel @ _V.T @ ( _R.T @ ((1 / _facR) * _obs_times - _R @ _s)  - _n_alpha**2 * _D.T @ _D @ _s)
            _z = _s + _V @ _a
            return _z

        rays = self.rays
        self.update_traveled_dists(rays, n_rays)

        z = self.slowness_vector

        obs_times = np.array(obs_times).reshape(-1, 1)
        if mode is None:
            mode = self.field_name.lower()
        match mode:
            case 'classic' | 'basic' | 'standard':
                z = iterate_field_classic(obs_times, self.params.alpha)
            case 'trade' | 'tradeoff' | 'dual':
                z = iterate_field_trade(obs_times, model_slowness, self.params.alpha, self.params.beta, self.params.masking_depth)
            case 'split':
                z = iterate_field_split(obs_times, model_slowness, self.params.alpha, self.params.masking_depth)
            case 'munk' | 'param' | 'parameterized':
                z = iterate_field_param(obs_times, self.params.alpha)
            case 'svd':
                z = iterate_field_svd(obs_times, self.params.alpha)
            case 'eof1':
                z = iterate_field_eof_r1(obs_times, self.params.alpha, 'midput/slowness_field_eofs.dat')
            case 'eof2':
                z = iterate_field_eof_r2(obs_times, self.params.alpha, 'midput/slowness_field_eofs.dat')
            case 'mean':
                with open('midput/slowness_field_eofs.dat', 'rb') as f:
                    data = pickle.load(f)
                z = np.concatenate(self.cells_nx*[data['mean'].reshape(-1, 1)], axis=1).reshape(-1, 1)
            case _:
                raise NotImplementedError("The iteration mode {} is not implemented.".format(mode))

        self.update_field(z)

        return z

    def cost_function(self, rays, t):
        # Cost function to be minimized
        t_est = np.array([ray.calc_time(self)[0] for ray in rays])
        t = np.array(t)
        return norm(t_est - t) ** 2

    def gradient(self, t):
        # Gradient of cost function
        return self.traveled_dists.T @ (self.traveled_dists @ self.slowness_vector - t)

    def calc_metrics(self, times, field):
        # Calculates metrics
        # TODO: Implement other metrics maybe?
        self.est_time_mse.append(1000 * np.sqrt(1/len(times) * self.cost_function(self.rays, times)))
        self.est_ssf_rms.append(rms(self.field - field))
        self.est_rho_rms.append(self.calc_rho(self.field))

    def export_metrics(self, direc='Results/', code=''):
        # Exports metrics as .csv file
        est_time_txt = ['x,y']
        est_ssf_txt = ['x,y']
        est_rho_txt = ['x,y']
        time_vs_ssf_txt = ['x,y']

        for idx in range(len(self.est_time_mse)):
            iteration = idx + 1
            est_time = self.est_time_mse[idx]
            est_ssf = self.est_ssf_rms[idx]
            est_rho = self.est_rho_rms[idx]

            est_time_txt.append('{},{:.8f}'.format(iteration, est_time))
            est_ssf_txt.append('{},{:.4f}'.format(iteration, est_ssf))
            est_rho_txt.append('{},{:.4f}'.format(iteration, est_rho))
            if idx == len(self.est_time_mse) - 1:  # Remove condition if want to plot evolution of time-vs-ssf error
                time_vs_ssf_txt.append('{:.8f},{:.4f}'.format(est_time,est_ssf))

        measurements_vs_ssf_txt = 'x,y\n{:.4f},{:.4f}'.format(len(self.sources)*len(self.receivers), self.est_ssf_rms[-1])
        devices_vs_ssf_txt = 'x,y\n{:.4f},{:.4f}'.format(len(self.sources)+len(self.receivers), self.est_ssf_rms[-1])
        est_time_txt = '\n'.join(est_time_txt)
        est_ssf_txt = '\n'.join(est_ssf_txt)
        est_rho_txt = '\n'.join(est_rho_txt)
        time_vs_ssf_txt = '\n'.join(time_vs_ssf_txt)

        est_time_filename = code + '/' + 'metric_time_' + self.field_name.lower()
        est_ssf_filename = code + '/' + 'metric_ssf_' + self.field_name.lower()
        est_rho_filename = code + '/' + 'metric_rho_' + self.field_name.lower()
        time_vs_ssf_filename = code + '/' + 'metric_time_vs_ssf_' + self.field_name.lower()
        measurements_vs_ssf_filename = code + '/' + 'metric_msmnts_vs_ssf_' + self.field_name.lower()
        devices_vs_ssf_filename = code + '/' + 'metric_devices_vs_ssf_' + self.field_name.lower()

        with open(direc + est_time_filename + '.csv', 'w') as f:
            f.write(est_time_txt)
            f.close()
        with open(direc + est_ssf_filename + '.csv', 'w') as f:
            f.write(est_ssf_txt)
            f.close()
        # with open(direc + est_rho_filename + '.csv', 'w') as f:  # Enable block if want roughness metric
        #     f.write(est_rho_txt)
        #     f.close()
        with open(direc + time_vs_ssf_filename + '.csv', 'w') as f:
            f.write(time_vs_ssf_txt)
            f.close()
        with open(direc + measurements_vs_ssf_filename + '.csv', 'w') as f:
            f.write(measurements_vs_ssf_txt)
            f.close()
        with open(direc + devices_vs_ssf_filename + '.csv', 'w') as f:
            f.write(devices_vs_ssf_txt)
            f.close()

    def format_params(self):
        rounded = lambda x, v: '{:.{}e}'.format(x, v) if isinstance(x, (int, float))\
            else tuple(['{:.{}e}'.format(x_, v) for x_ in x]) if isinstance(x, (tuple, list))\
            else '' if x is None\
            else str(x)
        param_formatted = '  |  '.join(['{}:{}'.format(param_name, rounded(getattr(self.params, param_name), 3)).ljust(32) for param_name in
                      vars(self.params) if rounded(getattr(self.params, param_name), 3) is not ''])
        return param_formatted


def mod_munk_profile(par_e, par_v, par_z, par_h, par_r, height, y_mat, x_mat=None, return_separated=False):
    y_mat = height*y_mat
    eta = 2*(y_mat - par_z) / par_h
    vY = par_v*(1-par_e) + par_v*par_e*eta + par_v*par_e*np.exp(-eta) + par_r*(y_mat-par_z)

    return vY

def mackenzie_profile(t, s, d, seed=0, amp=1):
    np.random.seed(seed)
    s_ = s - 35
    field_velocity = (  1448.96
                      + 4.5910e0 * t
                      - 5.304e-2 * (t ** 2)
                      + 2.374e-4 * (t ** 3)
                      + 1.340e0  * s_
                      + 1.630e-2 * d
                      + 1.675e-7 *(d ** 2)
                      - 1.025e-2 * t * s_
                      - 7.139e-13 * t * (d ** 3)
                        )
    field_velocity += np.random.randn(*field_velocity.shape) * amp
    return field_velocity


def delgrosso_profile(t, s, d, seed=0, amp=1):
    np.random.seed(seed)
    p = 1.033*d/10 #TODO: FIX THIS
    C0 = 1402.392
    dCt =(0.501109398873e1 *t
        - 0.550946843172e-1*t**2
        + 0.221535969240e-3*t**3)
    dCs =(0.132952290781e1*s
        + 0.128955756844e-3*s**2)
    dCp =(0.156059257041e0*p
        + 0.244998688441e-4*p**2
        - 0.883392332513e-8*p**3)
    dCstp =(-0.127562783426e-1*t*s
           + 0.635191613386e-2*t*p
           + 0.265484716608e-7*t**2*p**2
           - 0.159349479045e-5*t*p**2
           + 0.522116437235e-9*t*p**3
           - 0.438031096213e-6*t**3*p
           - 0.161674495909e-8*s**2*p**2
           + 0.968403156410e-4*t**2*s
           + 0.485639620015e-5*t*s**2*p
           - 0.340597039004e-3*t*s*p
            )
    field_velocity = C0 + dCt + dCs + dCp + dCstp

    return field_velocity

def wilson_profile(t, s, d, seed=0, amp=1):
    p = 1.033*d/10 #TODO: FIX THIS
    C0 = 1449.14
    Ct = (  4.5721*t
          - 4.4532e-2*t**2
          - 2.6045e-4*t**3
          + 7.9851e-6*t**4)
    Cp = (1.60272e-1*p
          + 1.0268e-5*p**2
          + 3.5216e-9*p**3
          - 3.3603e-12*p**4)
    Cs = (1.39799*(s-35)
          + 1.69202e-3*(s-35)**2)
    Cstp = (s-35) * (
        - 1.1244e-2*t
        + 7.7711e-7*t**2
        + 7.7016e-5*p
        - 1.2945e-7*p**2
        + 3.1580e-8*p*t
        + 1.5790e-9*p*t**2
    )
    field_velocity = C0 + Ct + Cs + Cp + Cstp

    return field_velocity


def parameterized_munk_profile(vector_beta, range_y):
    nx = int(vector_beta.size / 4)
    vY = np.zeros([range_y.size, nx])
    for c_idx in range(nx):
        vel = vector_beta[c_idx]
        eps = vector_beta[nx+c_idx]
        dph = vector_beta[2*nx+c_idx]
        fac = vector_beta[3*nx+c_idx]
        vY[:, c_idx] = mod_munk_profile(eps, vel, dph, fac,
                                        0, 1, range_y)
    return vY


def parameterized_optimize(vector_beta, x_vec, y_vec, obs_times, mat_distances, norm_refs=(1500, 0.00737, 1400, 1300),
                           norm_alphas=(0, 0, 0), mode='global'):
    nx = int(vector_beta.size / 4)
    max_iterations = 100
    register_velocity_fields = []
    it_counter = 0
    while True:
        vector_vel = vector_beta[:nx]
        vector_eps = vector_beta[nx:2 * nx]
        vector_depth = vector_beta[2 * nx:3 * nx]
        vector_bfac = vector_beta[3 * nx:]
        velocity_field = parameterized_munk_profile(vector_beta, y_vec)
        slowness_field = 1 / velocity_field
        slowness_field = slowness_field.reshape(-1, 1)
        residuals = mat_distances @ slowness_field - obs_times
        _, y_mat = np.meshgrid(x_vec, y_vec)


        gna_jacobian = np.zeros([obs_times.size, vector_beta.size])
        mode=None
        for m_idx in range(obs_times.size):
            sum_weight = (-1 * (1/velocity_field**2) * mat_distances[m_idx, :].reshape(-1,nx))
            for k_idx in range(nx):
                if mode == 'global':
                    f_z = sum_weight
                    vv = vector_vel[k_idx].item()
                    ve = vector_eps[k_idx].item()
                    vz = vector_depth[k_idx].item()
                    vb = vector_bfac[k_idx].item()
                    vy = y_mat
                else:
                    f_z = sum_weight[:, k_idx]
                    vv = vector_vel[k_idx].item()
                    ve = vector_eps[k_idx].item()
                    vz = vector_depth[k_idx].item()
                    vb = vector_bfac[k_idx].item()
                    vy = y_mat[:, k_idx]

                # rel_depth = y_mat / vector_depth[k_idx]
                eta_z = 2*(vy-vz) / vb
                g_z = np.exp(-eta_z) + eta_z - 1
                vec_gna_jac_vel = f_z * (1 + ve * g_z)
                vec_gna_jac_eps = f_z * vv * g_z
                vec_gna_jac_dph = f_z * vv * ve * 2/vb * (np.exp(-eta_z)-1)
                vec_gna_jac_fac = f_z * vv * ve * eta_z / vb * (np.exp(-eta_z)-1)
                gna_jac_vel = np.sum(vec_gna_jac_vel)
                gna_jac_eps = np.sum(vec_gna_jac_eps)
                gna_jac_dph = np.sum(vec_gna_jac_dph)
                gna_jac_fac = np.sum(vec_gna_jac_fac)
                gna_jacobian[m_idx, k_idx] = gna_jac_vel
                gna_jacobian[m_idx, nx + k_idx] = gna_jac_eps
                gna_jacobian[m_idx, 2 * nx + k_idx] = gna_jac_dph
                gna_jacobian[m_idx, 3 * nx + k_idx] = gna_jac_fac
                del gna_jac_vel, gna_jac_eps, gna_jac_dph, gna_jac_fac
            del sum_weight

        ref_vector_beta = np.ones_like(vector_beta)
        ref_vector_beta[0 * nx:1 * nx] = norm_refs[0]
        ref_vector_beta[1 * nx:2 * nx] = norm_refs[1]
        ref_vector_beta[2 * nx:3 * nx] = norm_refs[2]
        ref_vector_beta[3 * nx:]       = norm_refs[3]

        reg_jacobian = np.ones_like(vector_beta)
        reg_jacobian[0 * nx:1 * nx] = alpha_rp(norm_alphas[0])
        reg_jacobian[1 * nx:2 * nx] = alpha_rp(norm_alphas[1])
        reg_jacobian[2 * nx:3 * nx] = alpha_rp(norm_alphas[2])
        reg_jacobian[3 * nx:]       = alpha_rp(norm_alphas[3])
        reg_jacobian = np.diag(reg_jacobian.reshape(-1,))
        reg_residuals = reg_jacobian @ (vector_beta - ref_vector_beta)
        # gaussnewton_step = pinv(gna_jacobian.T @ gna_jacobian + reg_jacobian.T @ reg_jacobian) @ (gna_jacobian.T @ residuals + reg_jacobian.T @ reg_residuals)
        gaussnewton_step = np.linalg.solve(gna_jacobian.T @ gna_jacobian + reg_jacobian.T @ reg_jacobian, gna_jacobian.T @ residuals + reg_jacobian.T @ reg_residuals)
        new_vector_beta = vector_beta - gaussnewton_step

        new_velocity_field = parameterized_munk_profile(new_vector_beta, y_vec)
        vector_beta = new_vector_beta
        reduced_vector_beta = vector_beta[::nx]
        # print(np.around(reduced_vector_beta.reshape(-1,), decimals=10), )
        prev_velocity_field = velocity_field
        velocity_field = new_velocity_field
        register_velocity_fields.append(velocity_field)
        if it_counter %10 == 0:
            pass
        if it_counter == max_iterations:
            break

        if norm(new_velocity_field - prev_velocity_field) / prev_velocity_field.size < 0.01:
            break
        it_counter += 1
    register_velocity_fields = np.array(register_velocity_fields)
    return vector_beta, velocity_field
