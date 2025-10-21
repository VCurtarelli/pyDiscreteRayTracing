import numpy as np
from numpy import pi


def calc_grad(vf, cell, prev_cell, pos_a_x, pos_a_y):
    # num_cells_x = vf.cells_nx
    # num_cells_y = vf.cells_ny
    # cell_height = vf.cell_height
    # cell_width = vf.cell_width
    # field = vf.field
    #
    # cell_y = cell[0]
    # cell_x = cell[1]
    # cell_y_ = prev_cell[0]
    # cell_x_ = prev_cell[1]
    # cell_l = np.clip(cell_x-1, 0, num_cells_x-1)
    # cell_r = np.clip(cell_x+1, 0, num_cells_x-1)
    # cell_d = np.clip(cell_y-1, 0, num_cells_y-1)
    # cell_u = np.clip(cell_y+1, 0, num_cells_y-1)
    # case = ''
    # if cell == prev_cell:
    #     grad_x = 1
    #     grad_y = 0
    #     case += 'p'
    # elif cell_x == cell_x_:
    #     a = (pos_a_y/cell_height) % 1
    #     b = 1-a
    #     if b < 0.5:
    #         c = 0.5+b
    #         grad_x = c*(field[cell_y, cell_x] - field[cell_y_, cell_x]) + (1-c)*(field[cell_y, cell_r] - field[cell_y_, cell_r])
    #     else:
    #         c = 0.5+a
    #         grad_x = c*(field[cell_y, cell_x] - field[cell_y_, cell_x]) + (1-c)*(field[cell_y, cell_l] - field[cell_y_, cell_l])
    #     # grad_x = (field[cell_y, cell_x] - field[cell_y_, cell_x])
    #     case += 'H'
    #     if cell_u == cell_x:
    #         grad_y = (field[cell_y, cell_x] - field[cell_d, cell_x])
    #         case += 'd'
    #     elif cell_d == cell_x:
    #         grad_y = (field[cell_u, cell_x] - field[cell_y, cell_x])
    #         case += 'u'
    #     else:
    #         grad_y = (a * field[cell_u, cell_x] + (b-a) * field[cell_y, cell_x] - b * field[cell_d, cell_x]) / 2
    #         case += 'v'
    # else:
    #     a = (pos_a_x/cell_width) % 1
    #     b = 1-a
    #     if b < 0.5:
    #         c = 0.5+b
    #         grad_y = c*(field[cell_y, cell_x] - field[cell_y, cell_x_]) + (1-c)*(field[cell_u, cell_x] - field[cell_u, cell_x_])
    #     else:
    #         c = 0.5+a
    #         grad_y = c*(field[cell_y, cell_x] - field[cell_y, cell_x_]) + (1-c)*(field[cell_d, cell_x] - field[cell_d, cell_x_])
    #     # grad_y = (field[cell_y,cell_x] - field[cell_y,cell_x_])
    #     if cell_r == cell_x:
    #         grad_x = (field[cell_y, cell_x] - field[cell_y, cell_l])
    #         case += 'l'
    #     elif cell_l == cell_x:
    #         grad_x = (field[cell_y, cell_r] - field[cell_y, cell_x])
    #         case += 'r'
    #     else:
    #         grad_x = (a * field[cell_y, cell_r] + (b-a) * field[cell_y, cell_x] - b * field[cell_y, cell_l]) / 2
    #         case += 'h'
    #     case += 'V'
    # if np.abs(grad_x) < 1e-6 and np.abs(grad_y) < 1e-6:
    #     dir_grad = 0
    #     mag_grad = 0
    #     # print(grad_x, grad_y)
    # else:
    #     dir_grad = (np.angle(grad_x + 1j*grad_y)+2*pi)%(2*pi)
    #     mag_grad = np.sqrt(grad_x**2 + grad_y**2)
    # return mag_grad, dir_grad
    return 1, 0
