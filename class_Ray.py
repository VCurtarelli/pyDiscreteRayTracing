import scipy.interpolate

from py_libs import *
# from fun_receiver_search import receiver_search
# from fun_calc_dist import calc_dist, calc_closest_point
import eikonalfm as ekf


class Ray:
    def __init__(self, source, receiver, ray_type, color=(0, 180, 0), marker='x'):
        # Initializes variables
        self.source = source
        self.receiver = receiver
        self.path = None
        self.time = None
        self.cells = None
        self.lengths = None
        self.color = np.array(color)/255
        self.marker = marker
        self.converged = True
        self.ray_type = ray_type


    def calc_path(self, velocity_field):
        # Calculates path for the given ray (for its source and receiver) through the environment
        # Currently uses the fast marching method implemented via the EikonalFM package
        field = velocity_field.field
        cell_width = velocity_field.cell_width
        cell_height = velocity_field.cell_height
        ds = (cell_height, cell_width)
        xs = (self.source[1] // cell_height, self.source[0] // cell_width)

        fm_tau = ekf.fast_marching(field,xs,ds,2)  # Eikonal solution (arrival-time at each cell given the source)
        receiver_cell = np.array((self.receiver[1] // cell_height, self.receiver[0] // cell_width),dtype=int)
        visited_cells = [receiver_cell]  # List of visited cells (starting at receiver)
        dists = [0.]  # Distance traversed at each cell
        slowness = 1/field
        while True:
            # Determines next cell based on time minimization (clipping at the edges)
            curr_cell = visited_cells[-1]
            xp = curr_cell[1]
            xr = curr_cell[1] + 1
            xl = curr_cell[1] - 1
            yp = curr_cell[0]
            yu = curr_cell[0] + 1
            yd = curr_cell[0] - 1

            tr = fm_tau[yp,xr] if xr < velocity_field.cells_nx else np.inf
            tl = fm_tau[yp,xl] if 0 <= xl else np.inf
            tu = fm_tau[yu,xp] if yu < velocity_field.cells_ny else np.inf
            td = fm_tau[yd,xp] if 0 <= yd else np.inf
            times = [tr, tl, tu, td]

            direction_next_cell = times.index(min(times))
            next_cell = [-1, -1]
            match direction_next_cell:
                case 0:
                    next_cell = curr_cell + np.array([ 0, 1])
                case 1:
                    next_cell = curr_cell + np.array([ 0,-1])
                case 2:
                    next_cell = curr_cell + np.array([ 1, 0])
                case 3:
                    next_cell = curr_cell + np.array([-1, 0])
            visited_cells.append(next_cell)
            xn = next_cell[1]
            yn = next_cell[0]
            half_dist = (fm_tau[yp,xp] - fm_tau[yn,xn]) / (slowness[yp,xp] + slowness[yn,xn])
            dists[-1] += half_dist
            dists.append(half_dist)

            if fm_tau[yn,xn] == 0:
                break
        self.cells = visited_cells
        self.path = [((cell[1]+0.5)*cell_width,(cell[0]+0.5)*cell_height) for cell in visited_cells]
        self.lengths = dists


    def calc_time(self, velocity_field):
        field = velocity_field.field
        Lengths = np.zeros_like(field)
        for jdx, cell in enumerate(self.cells):
            Lengths[cell[0], cell[1]] += self.lengths[jdx]

        time = (Lengths.reshape(-1, 1).T @ (1 / field.reshape(-1, 1))).item()
        self.time = time
        Lengths = Lengths.reshape(-1,1)
        return time, Lengths