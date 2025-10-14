from py_libs import *


def calc_velocity_field(num_cells_x, num_cells_y, width, height):
    x = np.linspace(0, width, num_cells_x)  # x-axis vector
    y = np.linspace(0, height, num_cells_y) # y-axis vector
    cell_width = x[1] - x[0]    # width of each cell
    cell_height = y[1] - y[0]   # height of each cell
    X, Y = np.meshgrid(x, y)    # X-Y plane grid
    fX = (2*X/width-1) # model for velocity variation on x-axis
    vX = 25*fX
    fY = np.exp((0.748 * Y / height) ** 2) - 0.748 * Y / height # model for velocity variation on y-axis
    vY = 300*((fY-1)*5+1)
    velocity_field = 1200 + (250-vY) + vX # velocity field
    return velocity_field
