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

def main():
	#pulsar()
	lorenz_attractor()
main()
