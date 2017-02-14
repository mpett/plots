from numpy import *

def pulsar():
	import numpy as np
	import matplotlib.pyplot as plt
	import matplotlib.animation as animation
	fig=plt.figure(figsize=(8,8),facecolor='black')
	ax=plt.subplot(111,frameon=False)
	data=np.random.uniform(0,1,(64,75))
	X=np.linspace(-1,1,data.shape[-1])
	G=1.5*np.exp(-4*X*X)
	lines=[]
	for i in range(len(data)):
		xscale=1-i/200.
		lw=1.5-i/100.0
		line,=ax.plot(xscale*X,i+G*data[i],color="w",lw=lw)
		lines.append(line)
	ax.set_ylim(-1,70)
	ax.set_xticks([])
	ax.set_yticks([])
	def update(*args):
		data[:,1:]=data[:,:-1]
		data[:,0]=np.random.uniform(0,1,len(data))
		for i in range(len(data)):
			lines[i].set_ydata(i+G+data[i])
		return lines
	anim=animation.FuncAnimation(fig,update,interval=10)
	plt.show()

def lorenz_attractor():
	import numpy as np
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	def lorenz(x, y, z, s=10, r=28, b=2.667):
		x_dot= s*(y - x)
		y_dot= r*x - y - x*z
		z_dot = x*y - b*z
		return x_dot, y_dot, z_dot
	dt = 0.01
	stepCnt = 10000
	xs=np.empty((stepCnt+1,))
	ys=np.empty((stepCnt+1,))
	zs=np.empty((stepCnt+1,))
	xs[0], ys[0], zs[0] = (0., 1., 1.05)
	for i in range(stepCnt):
		x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
		xs[i + 1] = xs[i] + (x_dot * dt)
		ys[i + 1] = ys[i] + (y_dot * dt)
		zs[i + 1] = zs[i] + (z_dot * dt)
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot(xs, ys, zs, lw=0.5)
	ax.set_xlabel("X axis")
	ax.set_ylabel("Y axis")
	ax.set_zlabel("Z axis")
	plt.show()

def interpolation_b_splines():
	from numpy import r_,sin,cos
	from scipy.signal import cspline1d,cspline1d_eval
	#%pylab inline
	x=r_[0:10]
	dx=x[1]-x[0]
	newx=r_[-3:13:0.1]
	y=cos(x)
	cj=cspline1d(y)
	newy=cspline1d_eval(cj,newx,dx=dx,x0=x[0])
	from pylab import plot,show
	plot(newx,newy,x,y,'o')
	show()
	
def wireframe():
	import pylab as p
	import mpl_toolkits.mplot3d.axes3d as p3
	u=r_[0:2*pi:100j]
	v=r_[0:pi:100j]
	x=10*outer(cos(u),sin(v))
	y=10*outer(sin(u),sin(v))
	z=10*outer(ones(size(u)),cos(v))
	fig=p.figure()
	ax=p3.Axes3D(fig)
	ax.plot_wireframe(x,y,z)
	p.show()	

def mandelbrot():
#	from numpy import *
	def mandel(n,m,itermax,xmin,xmax,ymin,ymax):
		ix,iy=mgrid[0:n,0:m]
		x=linspace(xmin,xmax,n)[ix]
		y=linspace(ymin,ymax,m)[iy]
		c=x+complex(0,1)*y
		del x,y
		img=zeros(c.shape,dtype=int)
		ix.shape=n*m
		iy.shape=n*m
		c.shape=n*m
		z=copy(c)
		for i in xrange(itermax):
			if not len(z): break
			multiply(z,z,z)
			add(z,c,z)
			rem=abs(z)>2.0
			img[ix[rem],iy[rem]]=i+1
			rem=-rem
			z=z[rem]
			ix,iy=ix[rem],iy[rem]
			c=c[rem]
		return img
		
def main():
#	pulsar()
	#lorenz_attractor()
	#interpolation_b_splines()
#	wireframe()
	mandelbrot()
	
main()
