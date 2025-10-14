from fun_calc_travel_time import calc_travel_time
from fun_plot_curves import plot_curves
from py_params import *
from fun_receiver_search import receiver_search
from fun_ray_tracing import ray_tracing
# from py_params import D_laplacian
from fun_calc_velocity_field import calc_velocity_field
from fun_cost_function import cost_function, gradient
from class_Ray import Ray
from class_VelocityField import VelocityField, EstVelocityField
import sys


np.set_printoptions(legacy='1.25',precision=2,linewidth=320,threshold=sys.maxsize)
decimal.getcontext().prec = 2

## --------------------
# ENVIRONMENT VARIABLES
num_cells_x = 6
num_cells_y = 8  # number of cells in each direction
num_cells = num_cells_x*num_cells_y
width = 5000   # width of environment
height = 5000  # height of environment

## --------------------
# ENVIRONMENT VARIABLES
n_receivers = 8
pos_receivers = [(width, (i+1)/n_receivers * height) for i in range(n_receivers)]
pos_sources = [(0,0), (0,0.2*height), (0, 0.4*height)]  # position of source
n_rays = len(pos_receivers) * len(pos_sources)

np.set_printoptions(legacy='1.25',precision=2,linewidth=320)
decimal.getcontext().prec = 2


## --------------------
# SOURCE-TO-RECEIVERS ANGLES ESTIMATION
ray_angles = []
times = []
J = np.zeros([n_rays, num_cells])

rays = []
for source in pos_sources:
    for receiver in pos_receivers:
        rays.append(Ray(source, receiver))

velocity_field = VelocityField(num_cells_x, num_cells_y, width, height)

# rays = [rays[3]]
for ray in rays:
    _, paths = ray.calc_path(velocity_field)
    for path in paths:
        path = np.array(path)
        # print(path.shape)
        # plt.plot(path[:, 0], path[:, 1])
    ray.calc_time(velocity_field)

# velocity_field.plot_curves(rays, pos_sources, pos_receivers)
plt.show()
obs_times = [ray.time for ray in rays]

est_velocity_field = EstVelocityField(num_cells_x, num_cells_y, width, height)
est_rays = []
for source in pos_sources:
    for receiver in pos_receivers:
        est_rays.append(Ray(source, receiver, color=(180,180,0)))

idx = 0
while True:
    est_velocity_field.iterate_field(est_rays, n_rays, obs_times)
    for ray in est_rays:
        ray.calc_path(est_velocity_field)
        ray.calc_time(est_velocity_field)
    est_times = [ray.time for ray in est_rays]
    times = np.array([obs_times, est_times])

    if all([ray.converged for ray in est_rays]):
        print("ALL CONVERGED")
        break
    # if est_velocity_field.cost_function(obs_times) < 1e-5:
    #     print("MINIMUM ERROR ACHIEVED")
    #     break
    if idx == 100:
        print("LIMIT REACHED")
        break
    idx += 1
    print(idx, est_velocity_field.cost_function(obs_times))
    # fig, axs = plt.subplots(1, 3, sharey=True, figsize=(12, 9))
    # est_velocity_field.plot_curves(est_rays, pos_sources, pos_receivers, axs[0], show_path=True)
    # velocity_field.plot_curves(rays, pos_sources, pos_receivers, axs[1], show_path=True)
    # velocity_field.plot_curves(rays + est_rays, pos_sources, pos_receivers, axs[2],
    #                            show_field=np.abs(velocity_field.field - est_velocity_field.field))
    # axs[0].set_xlabel('Estimate')
    # axs[1].set_xlabel('Real')
    # axs[2].set_xlabel('Comparison')
    #
    # manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    # plt.show()

# list_of_cells = [ray.cells for ray in est_rays]
visited_cells = {cell for ray in est_rays for cell in ray.cells }
# for ray in est_rays:
#     visited_cells.update(ray.cells)
#     print(ray.cells)
print(visited_cells)
row_coords, col_coords = zip(*visited_cells)
est_visited_velocities = est_velocity_field.field[row_coords, col_coords]
visited_velocities = velocity_field.field[row_coords, col_coords]
mean_est_velocity = np.mean(est_visited_velocities)
std_est_velocity = np.std(est_visited_velocities)
mean_velocity = np.mean(visited_velocities)
std_velocity = np.std(visited_velocities)
print("      Est\t\t| True")
print("Mean: {:.2f} \t| {:.2f}".format(mean_est_velocity,mean_velocity))
print("Std:  {:.2f} \t| {:.2f}".format(std_est_velocity,std_velocity))

fig, axs = plt.subplots(1, 3, sharey=True, figsize=(12,9))
est_velocity_field.plot_curves(est_rays, pos_sources, pos_receivers, axs[0])
velocity_field.plot_curves(rays, pos_sources, pos_receivers, axs[1])
velocity_field.plot_curves(rays + est_rays, pos_sources, pos_receivers, axs[2], show_field=np.abs(velocity_field.field - est_velocity_field.field))
axs[0].set_xlabel('Estimate')
axs[1].set_xlabel('Real')
axs[2].set_xlabel('Comparison')

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.show()
