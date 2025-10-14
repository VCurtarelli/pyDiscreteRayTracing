from py_libs import *
from fun_receiver_search import receiver_search


class Ray:
    def __init__(self, source, receiver, color=(0, 180, 0)):
        self.source = source
        self.receiver = receiver
        self.path = None
        self.angle = None
        self.time = None
        self.cells = None
        self.thetas = None
        self.lengths = None
        self.color = np.array(color)/255
        self.converged = True
        self.paths = None

    def calc_path(self, velocity_field, stop_param=0.005, iteration_step=0.1, iterations_max=50):
        self.angle = np.angle((np.array(self.receiver) - np.array(self.source)) @ np.array([1, 1j]).T)

        theta_0 = self.angle
        pos_receiver = self.receiver
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
            _, _, positions, _ = self.ray_tracing(velocity_field)
            position = np.array(positions[-1])
            lin_dist = float(np.linalg.norm((position - pos_receiver) * np.array([1/width, 1/height]) * np.sqrt(2)))
            ang_dist = np.angle((position.T @ np.array([1, 1j])).item()) - np.angle((pos_receiver.T @ np.array([1, 1j])).item())
            if np.abs(ang_dist) < np.abs(best_ang_dist):
                best_ang_dist = ang_dist
                best_theta = theta
                best_path = positions
            if lin_dist < stop_param:
                # print("Solution found: {} iterations - {:.2f} / {:.2f}".format(len(angles), lin_dist, stop_param))
                self.converged = True
                break
            new_theta = (theta - iteration_step * lin_dist * np.sign(ang_dist)).item()
            if len(angles) == iterations_max:
                # print("Iteration forced break achieved: {} iterations - {:.2f} / {:.2f}".format(len(angles), lin_dist, stop_param))
                self.converged = False
                break
            if len(angles) % 100 == 0:
                iteration_step *= 0.8
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

        thetas, lengths, positions, cells = self.ray_tracing(velocity_field)
        Lengths = np.zeros_like(field)
        for jdx, cell in enumerate(cells):
            Lengths[cell[0], cell[1]] = lengths[jdx]
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
            idx_y_, idx_x_ = cells[-2]
            if not (0 <= idx_x < num_cells_x and 0 <= idx_y < num_cells_y):
                break
            Va = field[idx_y, idx_x]
            Vb = field[idx_y_, idx_x_]
            mag_grad_V, direc_grad_V = self.calc_grad(velocity_field, cells[-1], cells[-2], pos_a_x, pos_a_y)
            discrete_gradient = 100*mag_grad_V
            gradient_factor = 1/(1+np.exp(-discrete_gradient))
            Va, Vb = gradient_factor*Va+(1-gradient_factor)*Vb, (1-gradient_factor)*Va + gradient_factor*Vb
    
            phi = direc_grad_V  #TODO: IF GRADIENT OF VELOCITY AND RAY DIRECTIONS ARE REVERTED, CANCEL IT
            theta = thetas[-1]
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
    
            x0 = (idx_x-1)*cell_width
            x1 = (idx_x+1)*cell_width
            y0 = (idx_y-1)*cell_height
            y1 = (idx_y+1)*cell_height
            px = pos_a_x
            py = pos_a_y
    
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
                if ts[i] < 0:
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
    
            if not ((0 <= pos_b_x <= width)
                and (0 <= pos_b_y <= height)):
                break
            travel_dist = np.sqrt((pos_b_x - pos_a_x) ** 2 + (pos_b_y - pos_a_y) ** 2)
            thetas.append(float(theta_b))
            lengths.append(travel_dist)
            positions.append((pos_b_x, pos_b_y))
            cells.append((idx_y, idx_x))
    
            if sin_theta_b > 1 or np.isnan(sin_theta_b):
                break
    
        cells = cells[1:-1]
        thetas = thetas[:-1]
        self.thetas = thetas
        self.cells = cells
        self.lengths = lengths
        self.path = positions
        return thetas,lengths,positions,cells

    def calc_grad(self, vf, cell, prev_cell, pos_a_x, pos_a_y):
        num_cells_x = vf.cells_nx
        num_cells_y = vf.cells_ny
        cell_height = vf.cell_height
        cell_width = vf.cell_width
        field = vf.field

        cell_y = cell[0]
        cell_x = cell[1]
        cell_y_ = prev_cell[0]
        cell_x_ = prev_cell[1]
        cell_l = np.clip(cell_x-1, 0, num_cells_x-1)
        cell_r = np.clip(cell_x+1, 0, num_cells_x-1)
        cell_d = np.clip(cell_y-1, 0, num_cells_y-1)
        cell_u = np.clip(cell_y+1, 0, num_cells_y-1)
        case = ''
        if cell == prev_cell:
            grad_x = 1
            grad_y = 0
            case += 'p'
        elif cell_x == cell_x_:
            grad_x = (field[cell_y, cell_x] - field[cell_y_, cell_x])
            a = (pos_a_y/cell_height) % 1
            b = 1-a
            case += 'H'
            if cell_u == cell_x:
                grad_y = (field[cell_y, cell_x] - field[cell_d, cell_x])
                case += 'd'
            elif cell_d == cell_x:
                grad_y = (field[cell_u, cell_x] - field[cell_y, cell_x])
                case += 'u'
            else:
                grad_y = (a * field[cell_u, cell_x] + (b-a) * field[cell_y, cell_x] - b * field[cell_d, cell_x]) / 2
                case += 'v'
        else:
            grad_y = (field[cell_y,cell_x] - field[cell_y,cell_x_])
            a = (pos_a_x/cell_width) % 1
            b = 1-a
            if cell_r == cell_x:
                grad_x = (field[cell_y, cell_x] - field[cell_y, cell_l])
                case += 'l'
            elif cell_l == cell_x:
                grad_x = (field[cell_y, cell_r] - field[cell_y, cell_x])
                case += 'r'
            else:
                grad_x = (a * field[cell_y, cell_r] + (b-a) * field[cell_y, cell_x] - b * field[cell_y, cell_l]) / 2
                case += 'h'
            case += 'V'
        if np.abs(grad_x) < 1e-6 and np.abs(grad_y) < 1e-6:
            dir_grad = 0
            mag_grad = 0
        else:
            dir_grad = (np.angle(grad_x + 1j*grad_y)+2*pi)%(2*pi)
            mag_grad = np.sqrt(grad_x**2 + grad_y**2)
        return mag_grad, dir_grad
