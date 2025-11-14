from fun_truncated_svd import truncated_svd
from py_libs import *
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve2d


class Environment:
    def __init__(self, num_cells_x, num_cells_y, width, height, rays, field_name='Observed', mirrored=False):
        self.cells_nx = num_cells_x
        self.cells_ny = num_cells_y
        self.width = width
        self.height = height

        self.cells_n = num_cells_x * num_cells_y
        self.cell_width = width/num_cells_x
        self.cell_height = height/num_cells_y

        self.grid_x = self.cell_width*(0.5 + np.arange(num_cells_x))
        self.grid_y = self.cell_height*(0.5 + np.arange(num_cells_y))

        self.field = self.generate_field()
        if mirrored:
            self.field = np.fliplr(self.field)
        self.interp = RegularGridInterpolator((self.grid_y, self.grid_x), self.field, method='cubic')
        self.field_name = field_name
        self.rays = rays


    def generate_field(self):
        width = self.width
        height = self.height

        x = self.grid_x
        y = self.grid_y
        X, Y = np.meshgrid(x, y)  # X-Y plane grid
        field_type = 'munk'
        X = X / width
        Y = Y / height
        match field_type:
            case 's_shape':
                fX = (2 * X - 1)  # model for velocity variation on x-axis
                Y = 1-Y
                fY = 2*Y**2 - Y**3 - Y**5
                fY = fY - np.amin(fY)
                fY = fY / np.amax(fY)
                vX = 0*fX
                vY = 30*fY
                velocity_field = 1500 + vY + vX  # velocity field
            case 'circle':
                fX = (2 * X - 1)
                fY = (2 * Y - 1)
                velocity_field = 1600 - 100 * np.sqrt(fX**2 + fY**2)
            case 'munk':
                epsilon=0.00737
                Z = 2*(height*Y-1300)/1300
                vY = 1500 * (1+epsilon*(Z-1+np.exp(-Z)))
                vX = 5*(2*X-1)
                velocity_field = vY + vX
            case _:
                velocity_field = 1600 * np.ones_like(X)

        return velocity_field

    def plot_curves(self, rays, sources, receivers,
                ax=None, show_field=None, show_path=False, legend=False, vs=None, cmap=None):
        width = self.width
        height = self.height
        field = self.field
        num_cells_x = self.cells_nx
        num_cells_y = self.cells_ny

        if ax is None:
            fig, ax = plt.subplots()
        if show_field is None:
            show_field = field
        if vs is None:
            # vmin = np.sort(show_field.reshape(-1,))[int(0.1*show_field.size)]
            # vmax = np.sort(show_field.reshape(-1,))[int(0.9*show_field.size)]
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
            pcm = ax.imshow(show_field, extent=(0, width, 0, height), alpha=alpha,
                            vmin=vmin, vmax=vmax,
                            cmap=cmap)
            plt.colorbar(pcm, ax=ax)
            # ax.imshow(np.ones_like(show_field), alpha=alpha, extent=(0, width, 0, height), cmap='gray')
            # ax.vlines(x=np.linspace(0, width, num_cells_x+1), ymin=0, ymax=height, color='gray', linestyle='dashed',
            #           linewidth=1)
            # ax.hlines(y=np.linspace(0, height, num_cells_y+1), xmin=0, xmax=width, color='gray', linestyle='dashed',
            #           linewidth=1)
        ray_angles = []

        # for ray in rays:
        #     x_pos = [ray.source[0], ray.receiver[0]]
        #     y_pos = [ray.source[1], ray.receiver[1]]
        #     ax.plot(x_pos, y_pos, color=ray.color * 0.3, marker='o')

        if show_path:
            for ray in rays:
                if ray.method == 'old':
                    for path in ray.paths:
                        path = np.array(path)
                        ax.plot(path[:, 0], path[:, 1], color=(0.5,0.5,0.5), alpha=0.5)

        for ray in rays:
            path = np.array(ray.path)
            ax.plot(path[:, 0], path[:, 1],
                    color=ray.color*(1 if ray.converged else 0.5),label=np.around(ray.time, 4), #marker=ray.marker,
                    alpha=0.1)

        for receiver in receivers:
            ax.plot(receiver[0], receiver[1], marker='x', markerfacecolor='blue', markersize=10,
                        markeredgecolor='blue', markeredgewidth=5)
        for source in sources:
            ax.plot(source[0], source[1], marker='+', markerfacecolor='red', markersize=10,
                        markeredgecolor='red', markeredgewidth=4)


        # final plotting setup
        ax.set_xlim((-.05 * width, width + .05 * width))
        ax.set_ylim((-.05 * height, height + .05 * height))
        if legend:
            ax.legend(loc='upper left')

    def field_to_csv(self, idx, direc='Results/', export_params=False, comp=None, vmin=None, vmax=None, code=''):
        field = self.field
        grid_x = self.grid_x - self.grid_x[0]
        grid_y = self.grid_y - self.grid_y[0]
        if comp is not None:
            field = np.abs(field - comp)
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
            if vmin is not None or vmax is not None:
                filename += '_comp'
            if vmin is None:
                vmin = np.amin(field) - 0.2*np.std(field)
            if vmax is None:
                vmax = np.amax(field) + 0.2*np.std(field)
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

    def calc_grad(self, pos_a_x, pos_a_y, angle):
        num_cells_x = self.cells_nx
        num_cells_y = self.cells_ny
        cell_width = self.cell_width
        cell_height = self.cell_height
        x_interp = cell_width*(0.5 + np.arange(num_cells_x))
        y_interp = cell_height*(0.5 + np.arange(num_cells_y))
        # self.interp = RegularGridInterpolator((self.grid_y, self.grid_x), self.field, method='cubic')
        x_p = np.clip(pos_a_x, x_interp[0], x_interp[-1])
        y_p = np.clip(pos_a_y, y_interp[0], y_interp[-1])
        x_l = np.clip(x_p - 0.001*min(cell_width,cell_height), x_interp[0], x_interp[-1])
        x_r = np.clip(x_p + 0.001*min(cell_width,cell_height), x_interp[0], x_interp[-1])
        y_u = np.clip(y_p - 0.001*min(cell_width,cell_height), y_interp[0], y_interp[-1])
        y_d = np.clip(y_p + 0.001*min(cell_width,cell_height), y_interp[0], y_interp[-1])
        positions = np.array([[y_p, x_r],
                              [y_p, x_l],
                              [y_u, x_p],
                              [y_d, x_p]])
        try:
            vels = self.interp(positions)
        except ValueError:
            vels = np.zeros(positions.shape[0])
        grad_x = ((vels[0] - vels[1]) / norm(positions[0] - positions[1])).item()
        grad_y = ((vels[2] - vels[3]) / norm(positions[2] - positions[3])).item()
        dir_grad = np.angle(grad_x + 1j*grad_y)
        if np.cos(angle - dir_grad) < 0:
            dir_grad = (dir_grad + pi + 2 * pi) % (2 * pi)
        # mag_grad = 5*np.sqrt(grad_x ** 2 + grad_y ** 2)
        mag_grad = 0.25

        dx_a = np.clip(x_p - mag_grad*np.cos(angle)*min(cell_width,cell_height), x_interp[0], x_interp[-1])
        dx_b = np.clip(x_p + mag_grad*np.cos(angle)*min(cell_width,cell_height), x_interp[0], x_interp[-1])
        dy_a = np.clip(y_p - mag_grad*np.sin(angle)*min(cell_width,cell_height), y_interp[0], y_interp[-1])
        dy_b = np.clip(y_p + mag_grad*np.sin(angle)*min(cell_width,cell_height), y_interp[0], y_interp[-1])
        positions = np.array([[dy_a, dx_a],
                              [dy_b, dx_b]])

        vels = self.interp(positions)
        Va = vels[0].item()
        Vb = vels[1].item()

        return mag_grad, dir_grad, Va, Vb

    def update_rays(self):
        for ray in self.rays:
            ray.calc_path(self)
            ray.calc_time(self)


class EstEnvironment(Environment):
    def __init__(self, num_cells_x, num_cells_y, width, height, rays, initial_value=1100, field_name='Estimate'):
        super().__init__(num_cells_x, num_cells_y, width, height, rays)
        self.update_field((1/initial_value)*np.ones_like(self.field))
        self.J = None
        self.D = None
        self.B = None
        self.generate_D()
        self.generate_B(sigma=0.05)
        self.field_name = field_name
        self.est_time_mse = []
        self.est_ssf_mse = []

    def generate_D(self):
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
            # laplacian_block[u, i] = 1
            # laplacian_block[d, i] = 1
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
        self.D = D_laplacian
        np.savetxt('mat_D_laplacian.csv', D_laplacian, delimiter=',', fmt='%d')

    def generate_B(self, sigma=1.):
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
        self.B = blur

    def update_J(self, rays, n_rays):
        num_cells = self.cells_n
        J = np.zeros([n_rays, num_cells])
        for idx, ray in enumerate(rays):
            ray.calc_path(self)
            _, Lengths = ray.calc_time(self)
            J[idx, :] = Lengths.reshape(-1,)
        self.J = J

        return J

    def update_field(self, z):
        z[z <= 0] = np.median(z[z > 0])
        self.z = z
        self.field = 1/z.reshape(self.cells_ny, -1)
        self.interp = RegularGridInterpolator((self.grid_y, self.grid_x), self.field, method='cubic')

    def iterate_field(self, _rays, _n_rays, _obs_times, method='prop', **kwargs):
        if method.endswith('*'):
            self.D = np.eye(self.D.shape[0])
        def iterate_field_prop(epsilon=0.05, obs_times=0):
            U, S, V, rank_J = truncated_svd(self.J @ self.B, epsilon)
            V2 = V[:, rank_J:]

            G = self.B @ (np.eye(num_cells) - V2 @ pinv(self.D @ V2) @ self.D) @ V @ pinv(S) @ U.T
            z = G @ obs_times
            if method.endswith('*'):
                s0 = kwargs['model']
                z0 = V2 @ pinv(V2) @ s0
            else:
                z0 = np.zeros_like(z)
            return z + z0

        def iterate_field_lit(alpha=0.01, obs_times=0):
            J = self.J
            facJ = np.amax(svd(J)[1])
            D = self.D
            J = J / facJ
            D = D / np.amax(svd(D)[1])
            B = self.B
            kernel = B.T @ J.T @ J @ B + (alpha**2/(1-alpha**2)) * D.T @ D
            ikernel = inv(kernel)
            G = (1/facJ) * B @ ikernel @ B.T @ J.T
            z = G @ obs_times
            if method.endswith('*'):
                s0 = kwargs['model']
                z0 = alpha**2 * ikernel @ s0
            else:
                z0 = np.zeros_like(z)
            return z + z0


        num_cells = self.cells_n
        self.update_J(_rays, _n_rays)
        match method:
            case 'proposed' | 'proposed*':
                if not 'epsilon' in kwargs.keys():
                    _epsilon=0.0
                else:
                    _epsilon=kwargs['epsilon']
                z = iterate_field_prop(_epsilon, _obs_times)
            case 'literature' | 'literature*':
                if not 'alpha' in kwargs.keys():
                    _alpha=0.0
                else:
                    _alpha = kwargs['alpha']
                if not 'epsilon' in kwargs.keys():
                    _epsilon=0.1
                else:
                    _epsilon = kwargs['epsilon']
                z = iterate_field_lit(_alpha,_obs_times)
            case _:
                z = np.zeros_like(self.J.T)


        self.update_field(z)


    def cost_function(self, rays, t):
        t_est = np.array([ray.calc_time(self)[0] for ray in rays])
        t = np.array(t)
        return norm(t_est - t)**2

    def gradient(self, t):
        return self.J.T @ (self.J @ self.z - t)

    def calc_metrics(self, t, s):
        self.est_time_mse.append(self.cost_function(self.rays, t))
        self.est_ssf_mse.append(1/self.cells_n * norm(s - 1/self.z))

    def export_metrics(self, direc='Results/', code=''):
        est_time_txt = ['x,y']
        est_ssf_txt = ['x,y']

        for idx in range(len(self.est_time_mse)):
            iteration = idx+1
            est_time = self.est_time_mse[idx]
            est_ssf = self.est_ssf_mse[idx]
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
