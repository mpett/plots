import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print "hello"

def lorenz(x, y, z, s=10, r=28, b=2.667):
	x_dot = s*(y-x)
	y_dot = r*x - y - x*z
	z_dot = x*y - b*z
	return x_dot, y_dot, z_dot

dt = 0.01
stepCounter = 10000

xs = np.empty((stepCounter + 1,))
ys = np.empty((stepCounter + 1,))
zs = np.empty((stepCounter + 1,))

xs[0], ys[0], zs[0] = (0., 1., 1.05)

for i in range(stepCounter):
	x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
	xs[i+1] = xs[i] + (x_dot * dt)
	ys[i+1] = ys[i] + (y_dot * dt)
	zs[i+1] = zs[i] + (z_dot * dt)

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(xs, ys, zs, lw = 0.5)
ax.set_xlabel("x axis")
ax.set_ylabel("y axis")
ax.set_zlabel("z axis")
# ax.set_title("Lorenz Attractor")
ax.set_axis_off()
plt.show()
	
