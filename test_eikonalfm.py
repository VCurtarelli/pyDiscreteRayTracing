import numpy as np
import eikonalfm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes3d.mouserotationstyle'] = 'azel'  # 'azel', 'trackball', 'sphere', or 'arcball'
x = np.linspace(0, 10, 20)
y = x
X, Y = np.meshgrid(x, y)
c = np.ones_like(X)
# c = 1+np.sqrt(X) + np.sqrt(Y)
x_s = (5, 0)
dx = (1.0, 1.0)
order = 2

tau_fm = eikonalfm.fast_marching(c, x_s, dx, order)
tau1_ffm = eikonalfm.factored_fast_marching(c, x_s, dx, order)
tau0 = eikonalfm.distance(tau1_ffm.shape, dx, x_s, indexing="ij")

# for the distance-function 'x_s' also describes an index-vector
plt.contourf(tau0 * tau1_ffm)
ax = plt.gca()
ax.set_aspect('equal')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, tau0, label='tau0', color='r')
# ax.plot_surface(X, Y, tau_fm / tau1_ffm, label='fm', color='g')
ax.plot_surface(X, Y, tau1_ffm, label='ffm', color='b')
print(tau1_ffm)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# plt.legend()
plt.show()