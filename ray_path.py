from py_libs import *
from py_params import *
from fun_receiver_search import receiver_search


## --------------------
# RECEIVER SEARCH
pos_receiver = (width, height)  # position of receiver
theta_0 = np.angle(complex(pos_receiver[0],pos_receiver[1]))    # initial guess for source-to-receiver ray's emission angle
angles, paths = receiver_search(theta_0, velocity_field, num_cells_x, num_cells_y, width, height, pos_source, pos_receiver, iterations_max=1000)


## --------------------
# RESULT PLOTTING
plt.imshow(np.flipud(velocity_field), extent=(0, width, 0, height), alpha=0.15) # plots velocity field in background


color_a = np.array([0,0,255]) / 255
color_b = np.array([255,255,0]) / 255
for idx in range(len(paths)):   # plots each iteration on receiver search
    path = np.array(paths[idx])
    fac = (idx/len(paths))**2
    plt.plot(path[:, 0], path[:, 1], label=np.around(np.rad2deg(angles[idx]), 2), color=fac * color_a+(1-fac)*color_b)

# plots source and receiver
plt.plot(pos_source[0], pos_source[1], marker='+',markerfacecolor='red', markersize=8, markeredgecolor='red',markeredgewidth=5)
plt.plot(pos_receiver[0],pos_receiver[1], marker='x',markerfacecolor='darkgreen', markersize=10, markeredgecolor='darkgreen',markeredgewidth=5)

# final plotting setup
plt.xlim((-.05*width, width+.05*width))
plt.ylim((-.05*height, height+.05*height))
# plt.legend(loc='upper left')
print(len(angles))
plt.show()






