from py_libs import *
from scipy.interpolate import RegularGridInterpolator


class Environment:
    def __init__(self, num_cells_x, num_cells_y, width, height, field_name='Observed', mirrored=False):
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


    def generate_field(self):
        width = self.width
        height = self.height
        num_cells_x = self.cells_nx
        num_cells_y = self.cells_ny
        cell_width = self.cell_width
        cell_height = self.cell_height

        x = self.grid_x
        y = self.grid_y
        X, Y = np.meshgrid(x, y)  # X-Y plane grid
        field_type = 's_shape'
        X = X / width
        Y = Y / height
        match field_type:
            case 's_shape':
                # fX = (2 * X - 1)  # model for velocity variation on x-axis
                # fY = np.exp((0.748 * Y) ** 2) - 0.748 * Y  # model for velocity variation on y-axis
                # fY = fY - np.amin(fY)
                # fY = fY / np.amax(fY)
                # vX = 10*fX
                # vY = 100*(1 - fY)
                fX = np.exp((0.748 * X) ** 2) - 0.748 * X  # model for velocity variation on y-axis
                fY = (2 * Y - 1)  # model for velocity variation on x-axis
                fX = (fX - np.amin(fX)) / (np.amax(fX) - np.amin(fX))
                vX = 100*(1-fX)
                vY = 10*fY
                velocity_field = 1500 + vY + vX  # velocity field
            case 'circle':
                fX = (2 * X - 1)
                fY = (2 * Y - 1)
                velocity_field = 1600 - 150 * np.sqrt((1.1*fX)**2 + fY**2)
            case _:
                velocity_field = 1500 * np.ones_like(X)

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
            vmin = np.amin(show_field)
            vmax = np.amax(show_field)
            vs = (vmin, vmax)
        vmin, vmax = vs
        if cmap is None:
            cmap = 'viridis'

        if not (show_field is False):

            pcm = ax.imshow(show_field, extent=(0, width, 0, height), alpha=0.2, vmin=vmin, vmax=vmax, cmap=cmap)
            ax.vlines(x=np.linspace(0, width, num_cells_x+1), ymin=0, ymax=height, color='gray', linestyle='dashed',
                      linewidth=1)
            ax.hlines(y=np.linspace(0, height, num_cells_y+1), xmin=0, xmax=width, color='gray', linestyle='dashed',
                      linewidth=1)
        ray_angles = []

        if show_path:
            for ray in rays:
                if not ray.converged:
                    for path in ray.paths:
                        path = np.array(path)
                        ax.plot(path[:, 0], path[:, 1], color=(0.5,0.5,0.5), alpha=0.1)

        # for ray in rays:
        #     x_pos = [ray.source[0], ray.receiver[0]]
        #     y_pos = [ray.source[1], ray.receiver[1]]
        #     ax.plot(x_pos, y_pos, color=ray.color * 0.3, marker='o')
        for ray in rays:
            path = np.array(ray.path)
            ax.plot(path[:, 0], path[:, 1], color=ray.color*(1 if ray.converged else 0.5),label=np.around(ray.time, 4), marker=ray.marker)

        for receiver in receivers:
            ax.plot(receiver[0], receiver[1], marker='x', markerfacecolor='darkgreen', markersize=10,
                        markeredgecolor='darkgreen', markeredgewidth=5)
        for source in sources:
            ax.plot(source[0], source[1], marker='+', markerfacecolor='red', markersize=8,
                        markeredgecolor='red', markeredgewidth=5)


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
        filename = code + '/' + ('it' + str(idx) + '_' if idx != 0 else '') + self.field_name.lower()
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
        mag_grad = 0.3
        # print('{:.2f}, {:.2f}, {:.2f}'.format(5000*grad_x, 5000*grad_y, 5000*mag_grad))

        dx_a = np.clip(x_p - mag_grad*np.cos(dir_grad)*min(cell_width,cell_height), x_interp[0], x_interp[-1])
        dx_b = np.clip(x_p + mag_grad*np.cos(dir_grad)*min(cell_width,cell_height), x_interp[0], x_interp[-1])
        dy_a = np.clip(y_p - mag_grad*np.sin(dir_grad)*min(cell_width,cell_height), y_interp[0], y_interp[-1])
        dy_b = np.clip(y_p + mag_grad*np.sin(dir_grad)*min(cell_width,cell_height), y_interp[0], y_interp[-1])
        positions = np.array([[dy_a, dx_a],
                              [dy_b, dx_b]])

        # print(positions)
        # print()
        vels = self.interp(positions)
        Va = vels[0].item()
        Vb = vels[1].item()

        return mag_grad, dir_grad, Va, Vb


class EstEnvironment(Environment):
    def __init__(self, num_cells_x, num_cells_y, width, height, initial_value=1400, field_name='Estimate'):
        super().__init__(num_cells_x, num_cells_y, width, height)
        self.field = initial_value*np.ones_like(self.field)
        self.J = None
        self.z = None
        self.D = None
        self.generate_D()
        self.field_name = field_name

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
            laplacian_block[u, i] = 1
            laplacian_block[d, i] = 1
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
            # D_laplacian[i, :] = -D_laplacian[i, :] / np.abs(D_laplacian[i,i])
            pass
        self.D = D_laplacian
        np.savetxt('mat_D_laplacian.csv', D_laplacian, delimiter=',', fmt='%d')

    def update_J(self, rays, n_rays):
        num_cells = self.cells_n
        J = np.zeros([n_rays, num_cells])
        for idx, ray in enumerate(rays):
            ray.calc_path(self)
            _, Lengths = ray.calc_time(self)
            if ray.converged:
                J[idx, :] = Lengths.reshape(-1,)
        self.J = J

        return J

    def iterate_field(self, _rays, _n_rays, _obs_times, method='prop', **kwargs):
        def iterate_field_prop(n_rays, epsilon=0.05):
            U, s, Vh = svd(self.J)
            V = Vh.T
            s[s < epsilon * s[0]] = 0
            S_ = np.diag(s)
            S = np.zeros_like(self.J)
            rank_J = (s[s != 0]).size
            V2 = V[:, rank_J:]
            S[:min(n_rays, num_cells), :min(n_rays, num_cells)] = S_
            G = (np.eye(num_cells) - V2 @ pinv(self.D @ V2) @ self.D) @ V @ pinv(S) @ U.T
            return G

        def iterate_field_lit(alpha=0.01):
            J = self.J
            D = self.D
            kernel = J.T @ J + (alpha**2 / (1-alpha**2)) * D.T @ D
            G = pinv(kernel) @ J.T
            return G

        num_cells = self.cells_n
        self.update_J(_rays, _n_rays)
        match method:
            case 'proposed':
                if not 'epsilon' in kwargs.keys():
                    _epsilon=0.0
                else:
                    _epsilon=kwargs['epsilon']
                G = iterate_field_prop(_n_rays, _epsilon)
            case 'literature':
                if not 'alpha' in kwargs.keys():
                    _alpha=0.0
                else:
                    _alpha = kwargs['alpha']
                G = iterate_field_lit(_alpha)
            case _:
                G = np.zeros_like(self.J.T)

        z = G @ _obs_times
        z[z <= 0] = 1e12
        self.z = z
        self.field = 1/z.reshape(self.cells_ny, -1)
        self.interp = RegularGridInterpolator((self.grid_y, self.grid_x), self.field, method='cubic')


    def cost_function(self, rays, t):
        t_est = np.array([ray.calc_time(self)[0] for ray in rays])
        t = np.array(t)
        return norm(t_est - t)**2

    def gradient(self, t):
        return self.J.T @ (self.J @ self.z - t)
