from __future__ import print_function
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import time

def generate(x, y, phi):
	r = 1 - np.sqrt(x**2 + y**2)
	return np.cos(2*np.pi * x + phi)

figure = plt.figure()
ax = figure.add_subplot(111, projection='3d')
xs = np.linspace(-1,1,50)
ys = np.linspace(-1,1,50)
x, y = np.meshgrid(xs,ys)

ax.set_zlim(-1,1)

wframe = None
tstart = time.time()
for phi in np.linspace(0, 180. / np.pi, 100):
	if wframe:
		ax.collections.remove(wframe)
	z = generate(x, y, phi)
	wframe = ax.plot_wireframe(x,y,z,rstride=2,cstride=2)
	plt.pause(0.001)



