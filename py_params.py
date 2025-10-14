import fun_calc_velocity_field
from py_libs import *


# ## --------------------
# # ENVIRONMENT VARIABLES
# num_cells_x = 6
# num_cells_y = 6  # number of cells in each direction
# num_cells = num_cells_x*num_cells_y
# width = 5000   # width of environment
# height = 5000  # height of environment
# pos_sources = [(0,0)]  # position of source
#
# ## --------------------
# # ENVIRONMENT VARIABLES
# n_receivers = 8
# pos_receivers = [(width, (i+1)/n_receivers * height) for i in range(n_receivers)]
#
# np.set_printoptions(legacy='1.25',precision=2,linewidth=320,threshold=sys.maxsize)
# decimal.getcontext().prec = 2
#
# ## --------------------
# # LAPLACIAN MATRIX CONSTRUCTION
# laplacian_block = np.zeros([num_cells_x,num_cells_x])
# for i in range(num_cells_x):
#     u = np.clip(i+1, 0, num_cells_x-1)
#     d = np.clip(i-1, 0, num_cells_x-1)
#     laplacian_block[u,i] = 1
#     laplacian_block[d,i] = 1
#     laplacian_block[i,u] = 1
#     laplacian_block[i,d] = 1
#     laplacian_block[i,i] = 0
#
# D_laplacian = np.zeros([num_cells, num_cells])
# for i in range(num_cells_y):
#     if i > 0:
#         D_laplacian[(i-1)*num_cells_x:i*num_cells_x, i*num_cells_x:(i+1)*num_cells_x] = np.eye(num_cells_x)
#         D_laplacian[i*num_cells_x:(i+1)*num_cells_x, (i-1)*num_cells_x:i*num_cells_x] = np.eye(num_cells_x)
#     D_laplacian[i*num_cells_x:(i+1)*num_cells_x, i*num_cells_x:(i+1)*num_cells_x] = laplacian_block
# for i in range(num_cells):
#     D_laplacian[i,i] = -1*np.sum(D_laplacian[i, :])
    # D_laplacian[i,:] = -D_laplacian[i,:] / D_laplacian[i,i]