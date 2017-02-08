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

def main():
	pulsar()

main()
