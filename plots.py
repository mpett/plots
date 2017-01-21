from __future__ import print_function
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import numpy as np
import matplotlib.pyplot as plt
import time

def lorenz(x, y, z, s=10, r=28, b=2.667):
	x_dot = s*(y-x)
	y_dot = r*x - y - x*z
	z_dot = x*y - b*z
	return x_dot, y_dot, z_dot

def generate(x, y, phi):
	r = 1 - np.sqrt(x**2 + y**2)
	return np.cos(2*np.pi * x + phi)

def step_lorenz():
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
	
	# ax.set_title("Lorenz Attractor")
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot(xs, ys, zs, lw = 0.5)
	ax.set_xlabel("x axis")
	ax.set_ylabel("y axis")
	ax.set_zlabel("z axis")
	ax.set_axis_off()
	plt.show()

def simple_animation():
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

def surface():
	filename = cbook.get_sample_data('jacksboro_fault_dem.npz', 
		asfileobj=False)
	with np.load(filename) as dem:
		z = dem['elevation']
		nrows, ncols = z.shape
		x = np.linspace(dem['xmin'], dem['xmax'], ncols)
		y = np.linspace(dem['ymin'], dem['ymax'], nrows)
		x,y = np.meshgrid(x,y)
	region = np.s_[5:50, 5:50]
	x,y,z = x[region], y[region], z[region]
	fig, ax, = plt.subplots(subplot_kw=dict(projection='3d'))
	ls = LightSource(270,45)
	rgb=ls.shade(z,cmap=cm.gist_earth,vert_exag=0.1,blend_mode='soft')
	surf=ax.plot_surface(x,y,z,rstride=1,cstride=1,facecolors=rgb,
		linewidth=0, antialiased=False,shade=False)
	plt.show()

step_lorenz()
simple_animation()
surface()




