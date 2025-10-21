from py_libs import *


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
        match field_type:
            case 's_shape':
                fX = (2 * X / width - 1)  # model for velocity variation on x-axis
                vX = 10*fX
                fY = np.exp((0.748 * Y / height) ** 2) - 0.748 * Y / height  # model for velocity variation on y-axis
                # fY = fY - np.amin(fY)
                # fY = fY / np.amax(fY)
                print(np.amin(fY))
                print(np.amax(fY))
                input()
                vY = 100*(1 - fY)
                velocity_field = 1500 + vY + vX  # velocity field
            case 'circle':
                fX = (2 * X / width - 1)
                fY = (2 * Y / height - 1)
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
                for path in ray.paths:
                    path = np.array(path)
                    ax.plot(path[:, 0], path[:, 1], color=(0.5,0.5,0.5), alpha=0.1)

        # for ray in rays:
        #     x_pos = [ray.source[0], ray.receiver[0]]
        #     y_pos = [ray.source[1], ray.receiver[1]]
        #     ax.plot(x_pos, y_pos, color=ray.color * 0.3, marker='o')
        for ray in rays:
            path = np.array(ray.path)
            ax.plot(path[:, 0], path[:, 1], color=ray.color*(1 if ray.converged else 0.5),label=np.around(ray.time, 4), marker='x')

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

    def field_to_csv(self, idx, direc='Results/', export_params=False):
        field = self.field
        grid_x = self.grid_x
        grid_y = self.grid_y

        txt = ['y,x,val']
        for y_idx in range(self.cells_ny):
            for x_idx in range(self.cells_nx):
                y_coord = grid_y[y_idx]
                x_coord = grid_x[x_idx]
                velocity = field[y_idx, x_idx]
                txt.append('{:.4f},{:.4f},{:.4f}'.format(y_coord,x_coord,velocity))
        txt = '\n'.join(txt)
        filename = str(idx) + ' ' + self.field_name + '.csv'
        with open(direc + filename, 'w') as f:
            f.write(txt)
            f.close()
        if export_params:
            vmin = np.amin(field) - 0.2*np.std(field)
            vmax = np.amax(field) + 0.2*np.std(field)
            nrows = self.cells_ny
            ncols = self.cells_nx
            txt = [
                r'\def\ymin{'+str(vmin)+r'}',
                r'\def\ymax{'+str(vmax)+r'}',
                r'\def\nrows{'+str(nrows)+r'}',
                r'\def\ncols{'+str(ncols)+r'}',
            ]
            txt = '\n'.join(txt)
            with open(direc + 'params.tex', 'w') as f:
                f.write(txt)
                f.close()



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
            # print('Rank: {} - Shape: {}'.format(rank(self.D @ V2), (self.D @ V2).shape))
            G = (np.eye(num_cells) - V2 @ pinv(self.D @ V2) @ self.D) @ V @ pinv(S) @ U.T
            return G

        def iterate_field_lit(alpha=0.01):
            J = self.J
            D = self.D
            kernel = (1-alpha**2) * J.T @ J + alpha**2 * D.T @ D
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


    def cost_function(self, rays, t):
        t_est = np.array([ray.calc_time(self)[0] for ray in rays])
        t = np.array(t)
        return norm(t_est - t)**2

    def gradient(self, t):
        return self.J.T @ (self.J @ self.z - t)
