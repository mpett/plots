from __future__ import print_function
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d  import axes3d
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import time

def lorenz(x, y, z, s=10, r=28, b=2.667):
	x_dot = s*(y-x)
	y_dot = r*x - y - x*z
	z_dot = x*y - b*z
	return x_dot, y_dot, z_dot

def generate(x, y, phi):
	r = 1 - np.sqrt(x**2 + y**2)
	return np.cos(2*np.pi * x + phi)

def rotate():
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x,y,z = axes3d.get_test_data(0.1)
	ax.plot_wireframe(x,y,z,rstride=5,cstride=5)
	for angle in range(0,360):
		ax.view_init(30,angle)
		plt.draw()
		plt.pause(.001)

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

def hinton(matrix, max_weight=None, ax=None):
	ax = ax if ax is not None else plt.gca()
	if not max_weight:
		max_weight=2**np.ceil(np.log(np.abs(matrix).max()) / np.log(2))
	ax.patch.set_facecolor('gray')
	ax.set_aspect('equal', 'box')
	ax.xaxis.set_major_locator(plt.NullLocator())
	ax.yaxis.set_major_locator(plt.NullLocator())
	for (x,y), w in np.ndenumerate(matrix):
		color = 'white' if w > 0 else 'black'
		size = np.sqrt(np.abs(w) / max_weight)
		rect = plt.Rectangle([x-size / 2, y-size/2], size, 
			size, facecolor=color, edgecolor=color)
		ax.add_patch(rect)
	ax.autoscale_view()
	ax.invert_yaxis()		

def show_hinton():
	hinton(np.random.rand(20,20)-0.5)
	plt.show()

def tricontour():
	n_angles = 48
	n_radii = 8
	min_radius = 0.25
	radii = np.linspace(min_radius,0.95,n_radii)
	angles = np.linspace(0,2*np.pi,n_angles,endpoint=False)
	angles = np.repeat(angles[...,np.newaxis], n_radii, axis=1)
	angles[:, 1::2] += np.pi / n_angles
	x = (radii*np.cos(angles)).flatten()
	y = (radii*np.sin(angles)).flatten()
	z = (np.cos(radii)*np.cos(angles*3.0)).flatten()
	triang = tri.Triangulation(x,y)
	xmid = x[triang.triangles].mean(axis=1)
	ymid = y[triang.triangles].mean(axis=1)
	mask = np.where(xmid*xmid + ymid*ymid < min_radius*min_radius,1,0)
	triang.set_mask(mask)
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.tricontour(triang,z,cmap=plt.cm.CMRmap)
	plt.show()
	

#step_lorenz()
#simple_animation()
#surface()
#rotate()
#show_hinton()
tricontour()




