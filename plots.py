from __future__ import print_function
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d  import axes3d
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource,Normalize
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from scipy.integrate import odeint
from math import sqrt
from scipy.stats import norm
from pylab import plot,show,grid,xlabel,ylabel
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.cm as cm
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
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

def strip_contour():
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	x,y,z=axes3d.get_test_data(0.05)
	cset=ax.contour(x,y,z,extend3d=True,cmap=cm.coolwarm)
	ax.clabel(cset,fontsize=9,inline=1)
	plt.show()

def contour():
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	x,y,z=axes3d.get_test_data(0.05)
	ax.plot_surface(x,y,z,rstride=8,cstride=8,alpha=0.3)
	cset=ax.contour(x,y,z,zdir='z',offset=-100,cmap=cm.coolwarm)
	cset=ax.contour(x,y,z,zdir='x',offset=-40,cmap=cm.coolwarm)
	cset=ax.contour(x,y,z,zdir='y',offset=40,cmap=cm.coolwarm)
	ax.set_xlabel('x')
	ax.set_xlim(-40,40)
	ax.set_ylabel('y')
	ax.set_ylim(-40,40)
	ax.set_zlabel('z')
	ax.set_zlim(-100,100)
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
	

def contourf_hatching():
	x = np.linspace(-3,5,120).reshape(1,-1)
	y=np.linspace(-3,5,120).reshape(-1,1)
	z=np.cos(x)+np.sin(y)
	x,y=x.flatten(),y.flatten()
	fig=plt.figure()
	cs=plt.contourf(x,y,z,hatches=['-','/','\\','//'],
		cmap=plt.get_cmap('gray'),extend='both',alhpa=0.5)
	plt.colorbar()
	plt.figure()
	n_levels=6
	plt.contourf(x,y,z,n_levels,colors='none',
		hatches=['.','/','\\', None,'\\\\','*'],extend='lower')	
	artists,labels=cs.legend_elements()
	plt.legend(artists,labels,handleheight=2)
	plt.show()

def triangular():
	n_radii=8
	n_angles=36
	radii=np.linspace(0.125,1.0,n_radii)
	angles=np.linspace(0,2*np.pi,n_angles,endpoint=False)
	angles=np.repeat(angles[...,np.newaxis],n_radii,axis=1)
	x=np.append(0,(radii*np.cos(angles)).flatten())	
	y=np.append(0,(radii*np.sin(angles)).flatten())
	z=np.sin(-x*y)
	fig=plt.figure()
	ax=fig.gca(projection='3d')
	ax.plot_trisurf(x,y,z,linewidth=0.2,antialiased=True)
	plt.show()

def mandelbrot_set(xmin,xmax,ymin,ymax,xn,yn,maxiter,horizon=2.0):
	x=np.linspace(xmin,xmax,xn,dtype=np.float32)
	y=np.linspace(ymin,ymax,yn,dtype=np.float32)
	c=x+y[:,None]*1j
	n_=np.zeros(c.shape,dtype=int)
	z=np.zeros(c.shape,np.complex64)
	for n in range(maxiter):
		i=np.less(abs(z),horizon)
		n_[i]=n
		z[i]=z[i]**2+c[i]
	n_[n_==maxiter-1]=0
	return z,n_

def mandelbrot_main():
	xmin,xmax,xn=-2.25,+0.75,3000/2
	ymin,ymax,yn=-1.25,+1.25,2500/2
	maxiter=200
	horizon=2.0**40
	log_horizon=np.log(np.log(horizon))/np.log(2)
	z,n=mandelbrot_set(xmin,xmax,ymin,ymax,xn,yn,maxiter,horizon)
	with(np.errstate(invalid='ignore')):
		m=np.nan_to_num(n+1-np.log(np.log(abs(z)))/np.log(2)+log_horizon)
	dpi=72
	width=10
	height=10*yn/xn
	fig=plt.figure(figsize=(width,height),dpi=dpi)
	ax=fig.add_axes([0.0,0.0,1.0,1.0],frameon=False,aspect=1)
	light=colors.LightSource(azdeg=315,altdeg=10)
	m=light.shade(m,cmap=plt.cm.hot,vert_exag=1.5,norm=colors.PowerNorm(0.3),blend_mode='hsv')
	plt.imshow(m,extent=[xmin,xmax,ymin,ymax],interpolation='bicubic')
	ax.set_xticks([])
	ax.set_yticks([])
	plt.show()

def randrange(n,vmin,vmax):
	return(vmax-vmin)*np.random.rand(n)+vmin

def scatterplot():
	fig=plt.figure()
	ax=fig.add_subplot(111,projection='3d')
	n=100
	for c,m,zlow,zhigh in [('r','o',-50,-25),('b','^',-30,-5)]:
		xs=randrange(n,23,32)
		ys=randrange(n,0,100)
		zs=randrange(n,zlow,zhigh)
		ax.scatter(xs,ys,zs,c=c,marker=m)
	plt.show()
	
def collections():
	fig=plt.figure()
	ax=fig.gca(projection='3d')
	x=np.linspace(0,1,100)
	y=np.sin(x*2*np.pi)/2+0.5
	ax.plot(x,y,zs=0,zdir='z',label='curve in (x,y)')
	colors=('r','g','b','k')
	x=np.random.sample(20*len(colors))
	y=np.random.sample(20*len(colors))
	c_list=[]
	for c in colors:
		c_list.append([c]*20)
	ax.scatter(x,y,zs=0,zdir='y',c=c_list,label='points in (x,z)')
	ax.legend()
	ax.set_xlim(0,1)
	ax.set_ylim(0,1)
	ax.set_zlim(0,1)
	ax.view_init(elev=20.,azim=-35)
	plt.show()

def offset():
	fig = plt.figure()
	ax=fig.gca(projection='3d')
	x,y=np.mgrid[0:6*np.pi:0.25,0:4*np.pi:0.25]
	z=np.sqrt(np.abs(np.cos(x)+np.cos(y)))
	surf=ax.plot_surface(x+1e5,y+1e5,z,cmap='autumn',cstride=2,rstride=2)
	ax.set_zlim(0,2)
	plt.show()

def surface():
	fig =plt.figure()
	ax=fig.add_subplot(111,projection='3d')
	u=np.linspace(0,2*np.pi,100)
	v=np.linspace(0,np.pi,100)
	x=10*np.outer(np.cos(u),np.sin(v))
	y=10*np.outer(np.sin(u),np.sin(v))
	z=10*np.outer(np.ones(np.size(u)),np.cos(v))
	ax.plot_surface(x,y,z,color='b')
	plt.show()

def streamplot():
	y,x=np.mgrid[-3:3:100j, -3:3:100j]
	u=-1-x**2+y
	v=1+x-y**2
	speed=np.sqrt(u*u+v*v)
	fig0,ax0=plt.subplots()
	strm=ax0.streamplot(x,y,u,v,color=u,linewidth=2,cmap=plt.cm.autumn)
	fig0.colorbar(strm.lines)
	fig1, (ax1,ax2)=plt.subplots(ncols=2)
	ax1.streamplot(x,y,u,v,density=[0.5,1])
	lw=5*speed/speed.max()
	ax2.streamplot(x,y,u,v,density=0.6,color='k',linewidth=lw)
	plt.show()

def quiver():
	fig=plt.figure()
	ax=fig.gca(projection='3d')
	x,y,z=np.meshgrid(np.arange(-0.8,1,0.2),np.arange(-0.8,1,0.2),np.arange(-0.8,1,0.8))
	u=np.sin(np.pi*x)*np.cos(np.pi*y)*np.cos(np.pi*z)
	v=-np.cos(np.pi*x)*np.sin(np.pi*y)*np.cos(np.pi*z)
	w=(np.sqrt(2.0/3.0)*np.cos(np.pi*x)*np.cos(np.pi*y)*np.sin(np.pi*z))
	ax.quiver(x,y,z,u,v,w,length=0.1)
	plt.show()

def trisurf():
	fig=plt.figure(figsize=plt.figaspect(0.5))
	u=np.linspace(0,2.0*np.pi,endpoint=True,num=50)
	v=np.linspace(-0.5,0.5,endpoint=True,num=10)
	u,v=np.meshgrid(u,v)
	u,v=u.flatten(),v.flatten()
	x=(1+0.5*v*np.cos(u/2.0))*np.cos(u)
	y=(1+0.5*v*np.cos(u/2.0))*np.sin(u)
	z=0.5*v*np.sin(u/2.0)
	mtri=tri.Triangulation(u,v)
	ax=fig.add_subplot(1,2,1,projection='3d')
	ax.plot_trisurf(x,y,z,triangles=mtri.triangles,cmap=plt.cm.Spectral)
	ax.set_zlim(-1,1)
	n_angles=36
	n_radii=8
	min_radius=0.25
	radii=np.linspace(min_radius,0.95,n_radii)
	angles=np.linspace(0,2*np.pi,n_angles,endpoint=False)
	angles=np.repeat(angles[...,np.newaxis],n_radii,axis=1)
	angles[:,1::2]+=np.pi/n_angles
	x=(radii*np.cos(angles)).flatten()
	y=(radii*np.sin(angles)).flatten()
	z=(np.cos(radii)*np.cos(angles*3.0)).flatten()
	triang=tri.Triangulation(x,y)
	xmid=x[triang.triangles].mean(axis=1)
	ymid=y[triang.triangles].mean(axis=1)
	mask=np.where(xmid**2+ymid**2<min_radius**2,1,0)
	triang.set_mask(mask)
	ax=fig.add_subplot(1,2,2,projection='3d')
	ax.plot_trisurf(triang,z,cmap=plt.cm.CMRmap)
	plt.show()

def display_colorbar():
	y,x=np.mgrid[-4:2:200j,-4:2:200j]
	z=10*np.cos(x**2+y**2)
	cmap=plt.cm.copper
	ls=LightSource(315,45)
	rgb=ls.shade(z,cmap)
	fig,ax=plt.subplots()
	ax.imshow(rgb,interpolation='bilinear')
	im=ax.imshow(z,cmap=cmap)
	im.remove()
	fig.colorbar(im)
	ax.set_title('Using a colorbar with a shaded plot',size='x-large')

def avoid_outliers():
	y,x=np.mgrid[-4:2:200j,-4:2:200j]
	z=10*np.cos(x**2+y**2)
	z[100,105]=2000
	z[120,110]=-9000
	ls=LightSource(315,45)
	fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(8,4.5))
	rgb=ls.shade(z,plt.cm.copper)
	ax1.imshow(rgb,interpolation='bilinear')
	ax1.set_title('Fullrange of data')
	rgb=ls.shade(z,plt.cm.copper,vmin=-10,vmax=10)
	ax2.imshow(rgb,interpolation='bilinear')
	ax2.set_title('Manually set range')
	fig.suptitle('Avoiding outliers in shaded plots',size='x-large')

def shade_other_data():
	y,x=np.mgrid[-4:2:200j,-4:2:200j]
	z1=np.sin(x**2)
	z2=np.cos(x**2+y**2)
	norm=Normalize(z2.min(),z2.max())
	cmap=plt.cm.RdBu
	ls=LightSource(315,45)
	rgb=ls.shade_rgb(cmap(norm(z2)),z1)
	fig,ax=plt.subplots()
	ax.imshow(rgb,interpolation='bilinear')
	ax.set_title('Shade by one variable, color by another', size='x-large')

def hillshading():
	display_colorbar()
	avoid_outliers()
	shade_other_data()
	plt.show()

def lorenz2(state,t):
	x=state[0]
	y=state[1]
	z=state[2]
	sigma=10.0
	rho=28.0
	beta=8.0/3.0
	xd=sigma*(y-x)
	yd=(rho-z)*x-y
	zd=x*y-beta*x
	return[xd,yd,zd]

def plot_lorenz2():
	state0=[1.0,1.0,1.0]
	t=np.arange(0.0,30.0,0.01)
	state=odeint(lorenz2,state0,t)
	fig=plt.figure()
	ax=fig.gca(projection='3d')
	ax.plot(state[:,0],state[:,1],state[:,2])
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	plt.show()

def another_plot():
	fig=plt.figure()
	ax=fig.gca(projection='3d')
	x=np.arange(-5,5,0.25)
	y=np.arange(-5,5,0.25)
	x,y=np.meshgrid(x,y)
	r=np.sqrt(x**2+y**2)
	z=np.sin(r)
	surf=ax.plot_surface(x,y,z,rstride=1,cstride=1,cmap=cm.coolwarm)
	plt.show()

def rgb_cube():
	bits=5
	fig=plt.figure()
	fig.subplots_adjust(left=0,bottom=0,right=1,top=1)
	ax=fig.add_subplot(111,projection='3d')
	ax.set_axis_bgcolor((0.5,0.5,0.5))
	gradient=np.linspace(0,1,2**bits)
	x,y,z=np.meshgrid(gradient,gradient,gradient)
	colors=np.stack((x.flatten(),y.flatten(),z.flatten()),axis=1)
	ax.scatter(x,y,z,alpha=1.0,s=100./2**bits,c=colors,marker='o',linewidth='0')
	def update(i):
		ax.view_init(elev=20.,azim=i)
		return fig,ax
	plt.axis('off')
	fig.set_size_inches(5,5)
	anim=FuncAnimation(fig,update,frames=np.arange(0,360,2),repeat=True)
	anim.save('%d-bit.color.gif' %(3*bits),dpi=80,writer='imagemagick',fps=24)

def rosenbrock_function():
	fig =plt.figure()
	fig.clf()
	ax=Axes3D(fig,azim=-128.0,elev=43.0)
	s=0.05
	x=np.arange(-2.0,2.0+s,s)
	y=np.arange(-1.0,3.0+s,s)
	x,y=np.meshgrid(x,y)
	z=(1.0-x)**2+100.0*(y-x*x)**2
	ax.plot_surface(x,y,z,rstride=1,cstride=1,norm=LogNorm(),cmap=colormap,linewidth=0,edgecolor='none')
	ax.set_xlim([-2,2])
	ax.set_ylim([-1,3])
	ax.set_zlim([0,2500])
	plt.show()

def another_graphic():
	fig=plt.figure()
	ax=Axes3D(fig)
	x=np.arange(-4,4,0.25)
	y=np.arange(-4,4,0.25)
	x,y=np.meshgrid(x,y)
	r=np.sqrt(x**2+y**2)
	z=np.sin(r)
	ax.plot_surface(x,y,z,rstride=1,cstride=1,cmap='hot')
	plt.show()

def basic_animation():	
	def data_gen(t=0):
		cnt=0
		while cnt<1000:
			cnt+=1
			t+=0.1
			yield t,np.sin(2*np.pi*t)*np.exp(-1/10)

	def init():
		ax.set_ylim(-1.1,1.1)
		ax.set_xlim(0,10)
		del xdata[:]
		del ydata[:]
		line.set_data(xdata,ydata)
		return line,

	fig,ax=plt.subplots()
	line,=ax.plot([],[],lw=2)
	ax.grid()
	xdata,ydata=[],[]

	def run(data):
		t,y=data
		xdata.append(t)
		ydata.append(y)
		xmin,xmax=ax.get_xlim()
		if t>=xmax:
			ax.set_xlim(xmin,2*xmax)
			ax.figure.canvas.draw()
		line.set_data(xdata,ydata)
		return line,

	ani=animation.FuncAnimation(fig,run,data_gen,blit=False,interval=10,repeat=False,init_func=init)
	plt.show()

def simple_3d_animation():
	def gen_randline(length,dims=2):
		line_data=np.empty((dims,length))
		line_data[:,0]=np.random.rand(dims)
		for index in range(1,length):
			step=((np.random.rand(dims)-0.5)*0.1)
			line_data[:,index]=line_data[:,index-1]+step
		return line_data
	
	def update_lines(num,data_lines,lines):
		for line,data in zip(lines,data_lines):
			line.set_data(data[0:2,:num])
			line.set_3d_properties(data[2,:num])
		return lines
	
	fig=plt.figure()
	ax=p3.Axes3D(fig)
	data=[gen_randline(25,3) for index in range(50)]
	lines=[ax.plot(dat[0,0:1],dat[1,0:1],dat[2,0:1])[0] for dat in data]
	ax.set_xlim3d([0.0,1.0])
	ax.set_xlabel('X')
	ax.set_ylim3d([0.0,1.0])
	ax.set_ylabel('Y')
	ax.set_zlim3d([0.0,1.0])
	ax.set_zlabel("Z")
	line_ani=animation.FuncAnimation(fig,update_lines,25,fargs=(data,lines),interval=50,blit=False)
	plt.show()

def random_data_animation():
	fig,ax=plt.subplots()
	line,=ax.plot(np.random.rand(10))
	ax.set_ylim(0,1)
	def update(data):
		line.set_ydata(data)
		return line,
	def data_gen():
		while True: yield np.random.rand(10)
	ani = animation.FuncAnimation(fig,update,data_gen,interval=100)
	plt.show()
	
def lines3d():
	mpl.rcParams['legend.fontsize']=10
	fig=plt.figure()
	ax=fig.gca(projection='3d')
	theta=np.linspace(-4*np.pi,4*np.pi,100)
	z=np.linspace(-2,2,100)
	r=z**2+1
	x=r*np.sin(theta)
	y=r*np.cos(theta)
	ax.plot(x,y,z,label='parametric curve')
	ax.legend()
	plt.show()

def brownian_motion():
	def brownian(x0,n,dt,delta,out=None):
		x0=np.asarray(x0)
		r=norm.rvs(size=x0.shape+(n,),scale=delta*sqrt(dt))
		if out is None:
			out=np.empty(r.shape)
		np.cumsum(r,axis=-1,out=out)
		out+=np.expand_dims(x0,axis=-1)
		return out
	import numpy
	delta=2
	T=10.0
	N=500
	dt=T/N
	m=20
	x=numpy.empty((m,N+1))
	x[:,0]=50
	brownian(x[:,0],N,dt,delta,out=x[:,1:])
	t=numpy.linspace(0.0,N*dt,N+1)
	for k in range(m):
		plot(t,x[k])
	xlabel('t',fontsize=16)
	ylabel('x',fontsize=16)
	grid(True)
	show()

#brownian_motion()
#lines3d()
#another_graphic()		
#mandelbrot_main()
step_lorenz()
#simple_animation()
#surface()
#rotate()
#show_hinton()
#tricontour()
#contour()
#strip_contour()
#contourf_hatching()
#triangular()
#scatterplot()
#collections()
#offset()
#surface()
#streamplot()
#quiver()
#trisurf()
#hillshading()
#plot_lorenz2()
#another_plot()
#rgb_cube()
#rosenbrock_function()
#basic_animation()
#simple_3d_animation()
#random_data_animation()
