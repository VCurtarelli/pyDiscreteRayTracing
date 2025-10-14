from py_libs import *


class VelocityField:
    def __init__(self, num_cells_x, num_cells_y, width, height):
        self.cells_nx = num_cells_x
        self.cells_ny = num_cells_y
        self.width = width
        self.height = height

        self.cells_n = num_cells_x * num_cells_y
        self.cell_width = width/num_cells_x
        self.cell_height = height/num_cells_y

        self.field = self.generate_field()
        self.D = None

    def generate_D(self, type='laplacian'):
        num_cells_x = self.cells_nx
        num_cells_y = self.cells_ny
        num_cells = self.cells_n
        ## --------------------
        # LAPLACIAN MATRIX CONSTRUCTION
        match type:
            case 'bilaplacian':
                laplacian_block = np.zeros([num_cells_x, num_cells_x])
                for i in range(num_cells_x):
                    uu = np.clip(i + 2, 0, num_cells_x - 1)
                    u = np.clip(i + 1, 0, num_cells_x - 1)
                    d = np.clip(i - 1, 0, num_cells_x - 1)
                    dd = np.clip(i - 2, 0, num_cells_x - 1)
                    # laplacian_block[uu, i] = 1
                    # laplacian_block[u, i] = -4
                    # laplacian_block[d, i] = -4
                    # laplacian_block[dd, i] = 1
                    laplacian_block[i, uu] = 1
                    laplacian_block[i, dd] = 1
                    laplacian_block[i, u] = -4
                    laplacian_block[i, d] = -4
                    laplacian_block[i, i] = 0
                D_laplacian = np.zeros([num_cells, num_cells])

                for i in range(num_cells_y):
                    if i > 0:
                        D_laplacian[(i - 1) * num_cells_x:i * num_cells_x, i * num_cells_x:(i + 1) * num_cells_x] = -4*np.eye(num_cells_x)
                        D_laplacian[i * num_cells_x:(i + 1) * num_cells_x, (i - 1) * num_cells_x:i * num_cells_x] = -4*np.eye(num_cells_x)
                    if i > 1:
                        D_laplacian[(i - 2) * num_cells_x:(i-1) * num_cells_x, i * num_cells_x:(i + 1) * num_cells_x] = np.eye(num_cells_x)
                        D_laplacian[i * num_cells_x:(i + 1) * num_cells_x, (i - 2) * num_cells_x:(i-1) * num_cells_x] = np.eye(num_cells_x)
                    D_laplacian[i * num_cells_x:(i + 1) * num_cells_x, i * num_cells_x:(i + 1) * num_cells_x] = laplacian_block
                for i in range(num_cells):
                    D_laplacian[i, i] = -1 * np.sum(D_laplacian[i, :])
                    D_laplacian[i, :] = -D_laplacian[i, :] / D_laplacian[i, i]
            case 'laplacian' | _:
                laplacian_block = np.zeros([num_cells_x, num_cells_x])
                for i in range(num_cells_x):
                    u = np.clip(i + 1, 0, num_cells_x - 1)
                    d = np.clip(i - 1, 0, num_cells_x - 1)
                    laplacian_block[u, i] = 1
                    laplacian_block[d, i] = 1
                    laplacian_block[i, u] = 1
                    laplacian_block[i, d] = 1
                    laplacian_block[i, i] = 0

                D_laplacian = np.zeros([num_cells, num_cells])
                for i in range(num_cells_y):
                    if i > 0:
                        D_laplacian[(i - 1) * num_cells_x:i * num_cells_x, i * num_cells_x:(i + 1) * num_cells_x] = np.eye(
                            num_cells_x)
                        D_laplacian[i * num_cells_x:(i + 1) * num_cells_x, (i - 1) * num_cells_x:i * num_cells_x] = np.eye(
                            num_cells_x)
                    D_laplacian[i * num_cells_x:(i + 1) * num_cells_x, i * num_cells_x:(i + 1) * num_cells_x] = laplacian_block
                for i in range(num_cells):
                    D_laplacian[i, i] = -1 * np.sum(D_laplacian[i, :])
                    D_laplacian[i, :] = -D_laplacian[i, :] / D_laplacian[i,i]

        # D_laplacian = D_laplacian.astype(int)
        print(rank(D_laplacian))
        s = svd(D_laplacian)[1]
        s = s / s[0]
        print(s)
        self.D = D_laplacian

    def generate_field(self):
        width = self.width
        height = self.height
        num_cells_x = self.cells_nx
        num_cells_y = self.cells_ny

        x = np.linspace(0, width, num_cells_x)  # x-axis vector
        y = np.linspace(0, height, num_cells_y)  # y-axis vector
        X, Y = np.meshgrid(x, y)  # X-Y plane grid
        fX = (2 * X / width - 1)  # model for velocity variation on x-axis
        vX = 25 * fX
        fY = np.exp((0.748 * Y / height) ** 2) - 0.748 * Y / height  # model for velocity variation on y-axis
        vY = 300 * ((fY - 1) * 5 + 1)
        velocity_field = 1200 + (250 - vY) + vX  # velocity field
        return velocity_field

    def plot_curves(self, rays, sources, receivers,
                ax=None, show_field=None, show_path=False):
        width = self.width
        height = self.height
        field = self.field
        num_cells_x = self.cells_nx
        num_cells_y = self.cells_ny

        if ax is None:
            fig, ax = plt.subplots()
        if show_field is None:
            show_field = field

        if not (show_field is False):
            ax.imshow(show_field, extent=(0, width, 0, height), alpha=0.2)
            ax.vlines(x=np.linspace(0, width, num_cells_x + 1), ymin=0, ymax=height, color='gray', linestyle='dashed',
                      linewidth=1)
            ax.hlines(y=np.linspace(0, height, num_cells_y + 1), xmin=0, xmax=height, color='gray', linestyle='dashed',
                      linewidth=1)
        ray_angles = []

        if show_path:
            for ray in rays:
                for path in ray.paths:
                    path = np.array(path)
                    ax.plot(path[:, 0], path[:, 1], color=(0.5,0.5,0.5), alpha=0.1)


        for ray in rays:
            path = np.array(ray.path)
            ax.plot(path[:, 0], path[:, 1], color=ray.color*(1 if ray.converged else 0.5),label=np.around(ray.angle, 2),)

        for receiver in receivers:
            ax.plot(receiver[0], receiver[1], marker='x', markerfacecolor='darkgreen', markersize=10,
                        markeredgecolor='darkgreen', markeredgewidth=5)
        for source in sources:
            ax.plot(source[0], source[1], marker='+', markerfacecolor='red', markersize=8,
                        markeredgecolor='red', markeredgewidth=5)


        # final plotting setup
        ax.set_xlim((-.05 * width, width + .05 * width))
        ax.set_ylim((-.05 * height, height + .05 * height))
        # ax.legend(loc='upper left')

class EstVelocityField(VelocityField):
    def __init__(self, num_cells_x, num_cells_y, width, height, initial_value=1000):
        super().__init__(num_cells_x, num_cells_y, width, height)
        self.field = initial_value*np.ones_like(self.field)
        self.J = None
        self.z = None
        self.generate_D()

    def iterate_field(self, rays, n_rays, obs_times):
        num_cells = self.cells_n
        J = np.zeros([n_rays, num_cells])
        for idx, ray in enumerate(rays):
            ray.calc_path(self)
            _, Lengths = ray.calc_time(self)
            if ray.converged:
                J[idx, :] = Lengths.reshape(-1,)
            # else:
            #     del_rows.append(idx)
        # J = np.delete(J, del_rows, axis=0)
        # times = np.delete(obs_times, del_rows, axis=0)
        n_rays = J.shape[0]
        self.J = J
        U, s, Vh = svd(J)
        V = Vh.T
        s[s < 0.1 * s[0]] = 0
        S_ = np.diag(s)
        S = np.zeros_like(J)
        rank_J = (s[s != 0]).size
        V2 = V[:, rank_J:]
        S[:min(n_rays, num_cells), :min(n_rays, num_cells)] = S_
        G = (np.eye(num_cells) - V2 @ pinv(self.D @ V2) @ self.D) @ V @ pinv(S) @ U.T

        z = G @ obs_times
        self.z = z
        self.field = 1/z.reshape(self.cells_ny, -1)

    def cost_function(self, t):
        return norm(self.J @ self.z - t)**2

    def gradient(self, t):
        return self.J.T @ (self.J @ self.z - t)
