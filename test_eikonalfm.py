import numpy as np
import eikonalfm
import matplotlib.pyplot as plt


nx = 100
ny = 100
dx = 0.1
dy = 0.1
x = np.arange(nx) * dx
y = np.arange(ny) * dy
X, Y = np.meshgrid(x, y)
c = 10 + 20*(0.1*X-0.5)**2
print(np.amin(c))
print(np.amax(c))
# c = np.ones((100, 100))
ds = (dx, dy)
x_s = ((49, 49))
order = 2

tau_fm = eikonalfm.fast_marching(c, x_s, ds, order)
tau1_ffm = eikonalfm.factored_fast_marching(c, x_s, ds, order)

# for the distance-function 'x_s' also describes an index-vector
fig, axs = plt.subplots(1, 3)
tau0 = eikonalfm.distance(tau1_ffm.shape, ds, x_s, indexing="ij")
cf0 = axs[0].contourf(Y, X, tau0 * tau1_ffm)
cf1 = axs[1].contourf(Y, X, tau_fm)
cf2 = axs[2].contourf(Y, X, tau_fm - tau0*tau1_ffm)
axs[0].set_aspect('equal')
axs[1].set_aspect('equal')
axs[2].set_aspect('equal')
fig.colorbar(cf0, ax=axs[0])
fig.colorbar(cf1, ax=axs[1])
fig.colorbar(cf2, ax=axs[2])
# plt.colorbar(cf)
# axs[0].hlines(y, xmin=0, xmax=x[-1], color='gray')
# axs[0].vlines(x, ymin=0, ymax=y[-1], color='gray')
plt.show()