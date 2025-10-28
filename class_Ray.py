import scipy.interpolate

from py_libs import *
from fun_receiver_search import receiver_search
from fun_calc_dist import calc_dist, calc_closest_point
import eikonalfm as ekf


class Ray:
    def __init__(self, source, receiver, ray_type, color=(0, 180, 0), marker='x'):
        self.source = source
        self.receiver = receiver
        self.path = None
        self.angle = None
        self.time = None
        self.cells = None
        self.thetas = None
        self.lengths = None
        self.color = np.array(color)/255
        self.marker = marker
        self.converged = False
        self.paths = None
        self.ray_type = ray_type


    def new_calc_path(self, velocity_field):
        field = velocity_field.field
        cell_width = velocity_field.cell_width
        cell_height = velocity_field.cell_height
        ds = (cell_height, cell_width)
        xs = (self.source[1] // cell_height, self.source[0] // cell_width)
        fm_tau = ekf.fast_marching(field,ds,xs,2)
        ffm_tau1 = ekf.factored_fast_marching(field,ds,xs,2)
        curr_cell = np.array((self.receiver[1] // cell_height, self.receiver[0] // cell_width))
        visited_cells = [curr_cell]
        while True:
            xp = curr_cell[1]
            xr = curr_cell[1] + 1
            xl = curr_cell[1] - 1
            yp = curr_cell[0]
            yu = curr_cell[0] + 1
            yd = curr_cell[0] - 1
            tr = np.inf
            tl = np.inf
            tu = np.inf
            td = np.inf
            if xr < velocity_field.cells_nx:
                tr = fm_tau[yp,xr]
            if 0 <= xl:
                tl = fm_tau[yp,xl]
            if yu < velocity_field.cells_ny:
                tu = fm_tau[yu, xp]
            if 0 <= yd:
                td = fm_tau[yd, yp]
            times = [tr, tl, tu, td]
            direc = times.index(min(times))
            match direc:
                case 0:
                    next_cell = curr_cell + np.array([ 0, 1])
                case 1:
                    next_cell = curr_cell + np.array([ 0,-1])
                case 2:
                    next_cell = curr_cell + np.array([ 1, 0])
                case 3:
                    next_cell = curr_cell + np.array([-1, 0])
            pass


    def calc_path(self, velocity_field, stop_param=0.01, iteration_step=0.05, iterations_max=100):
        self.angle = np.angle((np.array(self.receiver) - np.array(self.source)) @ np.array([1, 1j]).T)

        theta_0 = self.angle
        pos_receiver = self.receiver
        pos_source = self.source
        angles = [theta_0]
        paths = []
        pos_receiver = np.array(pos_receiver)
        width = velocity_field.width
        height = velocity_field.height

        best_theta = 0
        best_ang_dist = np.inf
        best_path = None
        while True:
            theta = angles[-1]
            _, _, positions, cells = self.ray_tracing(velocity_field)
            lin_dist, ang_dist = calc_dist(pos_receiver, pos_source, positions)
            lin_dist /= np.sqrt(width**2 + height**2) / 2
            if np.abs(ang_dist) < np.abs(best_ang_dist):
                best_ang_dist = ang_dist
                best_theta = theta
                best_path = positions
            if lin_dist < stop_param:
                # print("Solution found: {} iterations - {:.2f} / {:.2f}".format(len(angles), lin_dist, stop_param))
                self.converged = True
                if (int(pos_receiver[1]//velocity_field.cell_height), int(pos_receiver[0]//velocity_field.cell_width)) in cells:
                    stop = cells.index((int(pos_receiver[1]//velocity_field.cell_height), int(pos_receiver[0]//velocity_field.cell_width)))
                    best_path = cells[:stop]
                break
            new_theta = (theta - iteration_step * lin_dist * np.sign(ang_dist)).item()
            if len(angles) == iterations_max:
                # print("Iteration forced break achieved: {} iterations - {:.2f} / {:.2f}".format(len(angles), lin_dist, stop_param))
                self.converged = False
                break
            if len(angles) % 10 == 0:
                iteration_step *= 0.7
            angles.append(new_theta)
            paths.append(positions)
            self.angle = new_theta

        angles.append(best_theta)
        paths.append(best_path)


        self.angle = best_theta
        self.paths = paths
        return angles, paths

    def calc_time(self, velocity_field):
        field = velocity_field.field
        if not self.converged:
            self.time = 0
            return 0, np.array([])
        Lengths = np.zeros_like(field)
        for jdx, cell in enumerate(self.cells):
            Lengths[cell[0], cell[1]] = self.lengths[jdx]

        time = (Lengths.reshape(-1, 1).T @ (1 / field.reshape(-1, 1))).item()
        self.time = time
        Lengths = Lengths.reshape(-1,1)
        return time, Lengths

    def ray_tracing(self, velocity_field):
        pos_source = self.source

        theta_0 = self.angle
        width = velocity_field.width
        height = velocity_field.height
        num_cells_x = velocity_field.cells_nx
        num_cells_y = velocity_field.cells_ny
        field = velocity_field.field
        thetas = []
        lengths = []
        positions = []
        cells = []
        cell_width = width / num_cells_x
        cell_height = height / num_cells_y
    
        idx_x = int(pos_source[0] // cell_width)
        idx_y = int(pos_source[1] // cell_height)
    
        thetas.append(theta_0)
        positions.append(pos_source)
        cells.append((idx_y, idx_x))
        cells.append((idx_y, idx_x))
        while True:
            pos_a_x, pos_a_y = positions[-1]  # crossing position from cell a (previous cell) to cell b (current cell)
            idx_y, idx_x = cells[-1]
            theta = thetas[-1]
            mag_grad_V, direc_grad_V, Va, Vb = velocity_field.calc_grad(pos_a_x, pos_a_y, theta)

            phi = direc_grad_V  #TODO: IF GRADIENT OF VELOCITY AND RAY DIRECTIONS ARE REVERTED, CANCEL IT
            if np.cos(theta-phi) < 0:
                phi = (phi+pi+2*pi) % (2*pi)
            alpha = theta - phi
            sin_beta = np.sin(alpha) * Vb / Va
            if abs(sin_beta) > 1.01:
                beta = pi-alpha
            else:
                if np.abs(sin_beta) > 1:
                    sin_beta = np.sign(sin_beta)
                beta = np.asin(sin_beta)
            gamma = beta + phi
    
            theta_b = (gamma+2*pi) % (2*pi)
            sin_theta_b = np.sin(theta_b)
            cos_theta_b = np.cos(theta_b)
            epsilon = 1e-6
            px = pos_a_x
            py = pos_a_y
            x0 = cell_width * np.floor(px/cell_width + epsilon)
            x1 = cell_width * np.ceil(px/cell_width + epsilon)
            y0 = cell_height * np.floor(py/cell_height + epsilon)
            y1 = cell_height * np.ceil(py/cell_height + epsilon)
    
            if sin_theta_b == 0:
                t_T = np.inf
                t_B = np.inf
            else:
                t_T = (y1 - py) / sin_theta_b
                t_B = (y0 - py) / sin_theta_b
            if cos_theta_b == 0:
                t_R = np.inf
                t_L = np.inf
            else:
                t_R = (x1 - px) / cos_theta_b
                t_L = (x0 - px) / cos_theta_b
            ts = [t_T, t_B, t_R, t_L]
            for i in range(len(ts)):
                if ts[i] <= 0:
                    ts[i] = np.inf
            t_min_idx = np.argmin(ts)

            t = ts[t_min_idx]
            if t_min_idx == 0:
                idx_y += 1
            if t_min_idx == 1:
                idx_y -= 1
            if t_min_idx == 2:
                idx_x += 1
            if t_min_idx == 3:
                idx_x -= 1
            pos_b_x = px + t*cos_theta_b
            pos_b_y = py + t*sin_theta_b

            if all([t_ == np.inf for t_ in ts]):
                break
            thetas.append(float(theta_b))
            positions.append((pos_b_x, pos_b_y))
            cells.append((idx_y, idx_x))
            if (not 0<=pos_b_x<=width) or (not 0<=pos_b_y<=height):
                break
            if sin_theta_b > 1 or np.isnan(sin_theta_b):
                break

        cells = cells[1:]

        new_cells = []
        new_thetas = []
        new_positions = []
        for idx, position in enumerate(positions):
            if idx == 0:
                new_positions.append(position)
                continue
            pos_x, pos_y = position
            pos, c0, c1, t = calc_closest_point(self.receiver, positions[:idx+1], return_params=True)
            if 0 <= t <= 1 or idx == len(positions)-1:
                pos = tuple(pos.reshape(-1,))
            else:
                pos = position
            new_positions.append(pos)
            new_cells.append(cells[idx-1])
            new_thetas.append(thetas[idx-1])

            if 0 <= t <= 1:
                break
            if (not 0 <= pos_x <= width) or (not 0 <= pos_y <= height):
                break
            if (not 0 <= cells[idx][0] < num_cells_y) or (not 0 <= cells[idx][1] < num_cells_x):
                break

        cells = new_cells
        thetas = new_thetas
        positions = new_positions
        lengths = [norm(np.array(positions[i]) - np.array(positions[i-1])) for i in range(1, len(positions))]
        self.thetas = thetas
        self.cells = cells
        self.lengths = lengths
        self.path = positions
        return thetas,lengths,positions,cells

    # def calc_grad(self, vf, pos_a_x, pos_a_y, angle):
    #     num_cells_x = vf.cells_nx
    #     num_cells_y = vf.cells_ny
    #     cell_width = vf.cell_width
    #     cell_height = vf.cell_height
    #     field = vf.field
    #     x_interp = cell_width*(0.5 + np.arange(num_cells_x))
    #     y_interp = cell_height*(0.5 + np.arange(num_cells_y))
    #     interp = scipy.interpolate.RegularGridInterpolator((y_interp, x_interp), field, method='cubic')
    #     x_p = np.clip(pos_a_x, x_interp[0], x_interp[-1])
    #     y_p = np.clip(pos_a_y, y_interp[0], y_interp[-1])
    #     x_l = np.clip(x_p - 0.01*cell_width, x_interp[0], x_interp[-1])
    #     x_r = np.clip(x_p + 0.01*cell_width, x_interp[0], x_interp[-1])
    #     y_u = np.clip(y_p - 0.01*cell_height, y_interp[0], y_interp[-1])
    #     y_d = np.clip(y_p + 0.01*cell_height, y_interp[0], y_interp[-1])
    #     positions = np.array([[y_p, x_r],
    #                           [y_p, x_l],
    #                           [y_u, x_p],
    #                           [y_d, x_p]])
    #     try:
    #         vels = interp(positions)
    #     except ValueError:
    #         vels = np.zeros(positions.shape[0])
    #     grad_x = ((vels[0] - vels[1]) / norm(positions[0] - positions[1])).item()
    #     grad_y = ((vels[2] - vels[3]) / norm(positions[2] - positions[3])).item()
    #     dir_grad = (float(np.angle(grad_x + 1j * grad_y)) + 2 * pi) % (2 * pi)
    #     mag_grad = 5*np.sqrt(grad_x ** 2 + grad_y ** 2)
    #     # print('{:.2f}, {:.2f}, {:.2f}'.format(5000*grad_x, 5000*grad_y, 5000*mag_grad))
    #
    #     dx_a = np.clip(x_p - mag_grad*np.cos(angle)*cell_width, x_interp[0], x_interp[-1])
    #     dx_b = np.clip(x_p + mag_grad*np.cos(angle)*cell_width, x_interp[0], x_interp[-1])
    #     dy_a = np.clip(y_p - mag_grad*np.sin(angle)*cell_height, y_interp[0], y_interp[-1])
    #     dy_b = np.clip(y_p + mag_grad*np.sin(angle)*cell_height, y_interp[0], y_interp[-1])
    #     positions = np.array([[dy_a, dx_a],
    #                           [dy_b, dx_b]])
    #
    #     # print(positions)
    #     # print()
    #     vels = interp(positions)
    #     Va = vels[0].item()
    #     Vb = vels[1].item()
    #
    #     return mag_grad, dir_grad, Va, Vb