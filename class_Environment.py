from fun_munk import munk
from py_libs import *
from scipy.interpolate import RegularGridInterpolator
from class_Ray import Ray
from matplotlib.colors import hsv_to_rgb
import random

rms = lambda x: np.sqrt(np.mean(x**2))
class Environment:
    def __init__(self, num_cells_x, num_cells_y, width, height, field_name='Observed'):
        # Initiate input parameters
        self.cells_nx = num_cells_x
        self.cells_ny = num_cells_y
        self.width = width
        self.height = height

        self.cells_n = num_cells_x * num_cells_y
        self.cell_width = width/num_cells_x
        self.cell_height = height/num_cells_y

        # Initiate grids, velocities, and rays
        self.grid_x = self.cell_width*(0.5 + np.arange(num_cells_x))
        self.grid_y = self.cell_height*(0.5 + np.arange(num_cells_y))
        self.vx = np.zeros([num_cells_y, num_cells_x])
        self.vy = np.zeros([num_cells_y, num_cells_x])

        self.traveled_dists = None
        self.field = self.generate_field()
        self.true_field = np.copy(self.field)

        self.field_name = field_name
        self.rays = None

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
            ray.color = hsv_to_rgb((idx/n_rays, 0.7, 0.7))
        self.rays = rays
        self.update_rays()
        return rays


    def generate_field(self, mode='munk'):
        width = self.width
        height = self.height

        # Generate mesh grid for X and Y directions
        x = self.grid_x
        y = self.grid_y
        X, Y = np.meshgrid(x, y)  # X-Y plane grid
        X = (X - np.amin(X)) / (np.amax(X) - np.amin(X))
        Y = (Y - np.amin(Y)) / (np.amax(Y) - np.amin(Y))

        if mode == 'munk':
            epsilon=0.00737
            velocity_field = munk(epsilon, height, Y, X)

            self.vy = munk(epsilon, height, Y)
            self.vx = velocity_field - self.vy
        else:
            raise NotImplementedError("Needs to implement other types of field")

        return velocity_field

    def plot_curves(self, rays, sources, receivers,
                ax=None, show_field=None, show_path=False, legend=False, vs=None, cmap=None):
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
                ax.imshow(background, extent=(0, width, 0, height), cmap='gray', vmin=0,vmax=1)
            pcm = ax.imshow(show_field[::-1], extent=(0, width, 0, height), alpha=alpha,
                            vmin=vmin, vmax=vmax,
                            cmap=cmap)
            plt.colorbar(pcm, ax=ax)

        for ray in rays:
            path = np.array(ray.path)
            ax.plot(path[:, 0], self.height - path[:, 1],
                    color=ray.color*(1 if ray.converged else 0.5),label=np.around(ray.time, 4), #marker=ray.marker,
                    alpha=0.4)

        for receiver in receivers:
            ax.plot(receiver[0], self.height - receiver[1], marker='x', markerfacecolor='blue', markersize=6,
                        markeredgecolor='blue', markeredgewidth=3)
        for source in sources:
            ax.plot(source[0], self.height - source[1], marker='+', markerfacecolor='red', markersize=6,
                        markeredgecolor='red', markeredgewidth=2)


        # final plotting setup
        ax.set_xlim((-.05 * width, width + .05 * width))
        ax.set_ylim((-.05 * height, height + .05 * height))

        ax.plot([0, width], [500, 500], 'k')
        ax.plot([0, width], [1000, 1000], 'k')

        if legend:
            ax.legend(loc='upper left')

    def field_to_csv(self, idx, direc='Results/', export_params=False, comp=None, vmin=None, vmax=None, code=''):
        # Export field to a csv (ordered to follow own LaTeX formatting for heatmaps)
        field = self.field
        grid_x = self.grid_x - self.grid_x[0]
        grid_y = self.grid_y - self.grid_y[0]
        if comp is not None:
            field = field - comp
        txt = ['y,x,val']
        for y_idx in range(self.cells_ny+1):
            if y_idx == self.cells_ny:
                y_coord = self.height
                y_idx -= 1
            else:
                y_coord = grid_y[y_idx]
            for x_idx in range(self.cells_nx+1):
                if x_idx == self.cells_nx:
                    x_coord = self.width
                    x_idx -= 1
                else:
                    x_coord = grid_x[x_idx]
                velocity = field[y_idx, x_idx]
                txt.append('{:.4f},{:.4f},{:.4f}'.format(y_coord,x_coord,velocity))
        txt = '\n'.join(txt)
        filename = code + '/' + ('it' + str(idx) + '_' if idx != 0 else '') + 'field_' + self.field_name.lower()
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
                vmin = 5*np.floor((np.amin(field) - 0.2*np.std(field))/5)
            if vmax is None:
                vmax = 5*np.ceil((np.amax(field) + 0.2*np.std(field))/5)
            nrows = self.cells_ny+1
            ncols = self.cells_nx+1
            txt = [
                r'\def\ymin{'+str(vmin)+r'}',
                r'\def\ymax{'+str(vmax)+r'}',
                r'\def\nrows{'+str(nrows)+r'}',
                r'\def\ncols{'+str(ncols)+r'}',
                r'\def\pyNx{'+str(ncols)+r'}',
                r'\def\pyNz{'+str(nrows)+r'}',
                r'\def\pyN{'+str(nrows*ncols)+r'}',
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


class EstEnvironment(Environment):
    def __init__(self, num_cells_x, num_cells_y, width, height, rays, initial_value=1100, field_name='Estimate'):
        # Initiate superclass
        super().__init__(num_cells_x, num_cells_y, width, height, rays)
        self.field_name = field_name

        # Update field to be uniform, generate laplacian (D) and blur (B) matrices
        self.slowness_vector = self.update_field((1 / initial_value) * np.ones([self.field.size, 1]))
        self.depth_mask = None
        self.laplacian_mtx = self.generate_laplacian()
        self.blur_mtx = self.generate_blur(sigma=0.00)
        self.est_time_mse = []
        self.est_ssf_rms = []

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
            D_laplacian[i, :] = -D_laplacian[i, :] / np.abs(D_laplacian[i,i])
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
            block_B[i, u] = val**2
            block_B[i, d] = val**2
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
            self.depth_mask = np.zeros([self.field.size,self.field.size])
            return None
        depth_mask_mtx = np.zeros_like(self.field)
        depth_mask_mtx[np.arange(self.cells_ny)>depth/self.cell_height, :] = 1
        # J[0, :] = 1
        depth_mask_mtx = depth_mask_mtx.reshape(-1,)
        depth_mask_mtx = np.diagflat(depth_mask_mtx)
        self.depth_mask = depth_mask_mtx

        return None

    def update_traveled_dists(self, rays, n_rays):
        # Updates traveled distances matrix (R)
        num_cells = self.cells_n
        traveled_dists = np.zeros([n_rays, num_cells])
        for idx, ray in enumerate(rays):
            ray.calc_path(self)
            _, Lengths = ray.calc_time(self)
            traveled_dists[idx, :] = Lengths.reshape(-1,)
        self.traveled_dists = traveled_dists

        return traveled_dists

    def update_field(self, z):
        z[z <= 0] = np.median(z[z > 0])
        self.slowness_vector = z
        self.field = 1/z.reshape(self.cells_ny, -1)
        return z

    def iterate_field(self, n_rays, alpha=0.01, beta=0.01, obs_times=0,
                      model_slowness=None, masking_depth=None, mode='classic'):
        # Iterate field (based on which method is to be used)
        # TODO: Here is where other iteration methods are to be implemented

        def normalize(mat):
            norm_fac = np.linalg.norm(mat, ord=2)
            norm_mat = mat/norm_fac
            return norm_mat, norm_fac

        def iterate_field_classic(_obs_times, _alpha):
            _R = self.traveled_dists
            _D = self.laplacian_mtx
            _obs_times = np.array(_obs_times).reshape(-1, 1)
            _n_alpha = _alpha / np.sqrt(1 - _alpha ** 2)
            _R, _facR = normalize(_R)
            _D, _FacD = normalize(_D)

            _kernel = _R.T @ _R + _n_alpha ** 2 * _D.T @ _D
            _ikernel = inv(_kernel)

            _z = (1/_facR) * (_ikernel @ _R.T @ _obs_times)
            return _z

        def iterate_field_trade(_obs_times, _model_slowness, _alpha, _beta, _masking_depth):
            if _model_slowness is None:
                _model_slowness = np.ones_like(self.slowness_vector) * 1 / 1500
            _R = self.traveled_dists
            _D = self.laplacian_mtx
            self.generate_mask(depth=_masking_depth)
            _J = self.depth_mask
            _z0 = model_slowness
            _obs_times = np.array(_obs_times).reshape(-1, 1)

            _n_alpha = _alpha / np.sqrt(1 - _alpha ** 2)
            _n_beta = _beta / np.sqrt(1 - _beta ** 2)
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
            _obs_times = np.array(_obs_times).reshape(-1, 1)
            _n_alpha = _alpha / np.sqrt(1 - _alpha ** 2)
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

            _z1 = _ikernel @ ((1/_facR) * _R1.T @ _obs_times
                              - _R1.T @ _R2 @ _z2
                              - _n_alpha ** 2 * _D1.T @ _D2 @ _z2)

            _z = np.concatenate((_z1, _z2), axis=0)
            return _z
        rays = self.rays
        self.update_traveled_dists(rays, n_rays)

        z = self.slowness_vector

        if mode is None:
            mode = self.field_name.lower()
        match mode:
            case 'classic' | 'basic' | 'standard':
                z = iterate_field_classic(obs_times, alpha)
            case 'trade' | 'tradeoff' | 'dual':
                z = iterate_field_trade(obs_times, model_slowness, alpha, beta, masking_depth)
            case 'split':
                z = iterate_field_split(obs_times, model_slowness, alpha, masking_depth)
            case _:
                raise NotImplementedError("The iteration mode {} is not implemented.".format(mode))

        self.update_field(z)

        return z

    def cost_function(self, rays, t):
        # Cost function to be minimized
        t_est = np.array([ray.calc_time(self)[0] for ray in rays])
        t = np.array(t)
        return norm(t_est - t)**2

    def gradient(self, t):
        # Gradient of cost function
        return self.traveled_dists.T @ (self.traveled_dists @ self.slowness_vector - t)

    def calc_metrics(self, t, s):
        # Calculates metrics
        # TODO: Implement other metrics maybe?
        self.est_time_mse.append(1000*self.cost_function(self.rays, t))
        self.est_ssf_rms.append(rms(self.field - s))

    def export_metrics(self, direc='Results/', code=''):
        # Exports metrics as .csv file
        est_time_txt = ['x,y']
        est_ssf_txt = ['x,y']

        for idx in range(len(self.est_time_mse)):
            iteration = idx+1
            est_time = self.est_time_mse[idx]
            est_ssf = self.est_ssf_rms[idx]
            est_time_txt.append('{},{:.8f}'.format(iteration,est_time))
            est_ssf_txt.append('{},{:.4f}'.format(iteration,est_ssf))

        est_time_txt = '\n'.join(est_time_txt)
        est_ssf_txt = '\n'.join(est_ssf_txt)

        est_time_filename = code + '/' + 'metric_time_' + self.field_name.lower()
        est_ssf_filename = code + '/' + 'metric_ssf_' + self.field_name.lower()

        with open(direc + est_time_filename + '.csv', 'w') as f:
            f.write(est_time_txt)
            f.close()
        with open(direc + est_ssf_filename + '.csv', 'w') as f:
            f.write(est_ssf_txt)
            f.close()
