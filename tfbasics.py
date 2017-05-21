from __future__ import print_function

import tensorflow as tf

a=tf.placeholder("float")
b=tf.placeholder("float")
y=tf.multiply(a,b)

def multiplication_basics():
	with tf.Session() as sess:
		print("%f should equal 2.0" % sess.run(y,feed_dict={a: 1,b:2}))
		print("%f should equal 9.0" % sess.run(y,feed_dict={a:3,b:3}))


def hello_world():
	hello=tf.constant("Hello,Tensorflow!")
	sess=tf.Session()
	print(sess.run(hello))

def matrix_multiplication():
	matrix1=tf.constant([[3.,3.]])
	matrix2=tf.constant([[2.],[2.]])
	product=tf.matmul(matrix1,matrix2)
	with tf.Session() as sess:
		result=sess.run(product)
		print(result)

def nearest_neighbor():
	import numpy as np
	from tensorflow.examples.tutorials.mnist import input_data
	mnist=input_data.read_data_sets("tmp/data",one_hot=True)
	Xtr,Ytr=mnist.train.next_batch(5000)
	Xte,Yte=mnist.test.next_batch(200)
	xtr=tf.placeholder("float", [None,784])
	xte=tf.placeholder("float",[784])
	distance=tf.reduce_sum(tf.abs(tf.add(xtr,tf.negative(xte))),reduction_indices=1)
	pred=tf.arg_min(distance,0)
	accuracy=0.
	init=tf.global_variables_initializer()	
	with tf.Session() as sess:
		sess.run(init)
		for i in range(len(Xte)):
			nn_index=sess.run(pred,feed_dict={xtr:Xtr,xte:Xte[i,:]})
			print("Test",i,"Prediction:",np.argmax(Ytr[nn_index]), \
				"True Class:",np.argmax(Yte[i]))
			if np.argmax(Ytr[nn_index])==np.argmax(Yte[i]):
				accuracy+=1./len(Xte)
		print("Done!")
		print("Accuracy:",accuracy)

def linear_regression():
	import numpy
	import matplotlib.pyplot as plt
	rng=numpy.random
	learning_rate=0.01
	training_epochs=1000
	display_step=50
	train_X=numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
	train_Y=numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])	
	n_samples=train_X.shape[0]
	X=tf.placeholder("float")
	Y=tf.placeholder("float")
	W=tf.Variable(rng.randn(),name="weight")
	b=tf.Variable(rng.randn(),name="bias")
	pred=tf.add(tf.multiply(X,W),b)
	cost=tf.reduce_sum(tf.pow(pred-Y,2))/(2*n_samples)
	optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
	init=tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(training_epochs):
			for(x,y) in zip(train_X,train_Y):
				sess.run(optimizer,feed_dict={X:x,Y:y})
			if(epoch+1)%display_step==0:
				c=sess.run(cost,feed_dict={X:train_X,Y:train_Y})
				print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
			                "W=", sess.run(W), "b=", sess.run(b))
		print("Optimization finished.")
		training_cost=sess.run(cost,feed_dict={X:train_X,Y:train_Y})
		print("Training cost=",training_cost,"W=",sess.run(W),"b=",sess.run(b),'\n')
		plt.plot(train_X,train_Y,'ro',label='Original data')
		plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label='Fitted line')
		plt.legend()
		plt.show()
		test_X=numpy.asarray([6.83,4.668,8.9,7.91,5.7,8.7,3.1,2.1])
		test_Y=numpy.asarray([1.84,2.273,3.2,2.831,2.92,3.24,1.35,1.03])
		print("Testing Mean Square Loss Comparison")
		testing_cost=sess.run(
			tf.reduce_sum(tf.pow(pred-Y,2))/(2*test_X.shape[0]),
			feed_dict={X:test_X,Y:test_Y})
		print("Testing cost=",testing_cost)
		print("Absolute mean square loss difference:",abs(training_cost-testing_cost))
		plt.plot(test_X,test_Y,'bo',label='Testing data')
		plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label='Fitted line')
		plt.legend()
		plt.show()

def logistic_regression():
	from tensorflow.examples.tutorials.mnist import input_data
	mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)
	learning_rate=0.01
	training_epochs=25
	batch_size=100
	display_step=1
	x=tf.placeholder(tf.float32,[None,784])
	y=tf.placeholder(tf.float32,[None,10])
	W=tf.Variable(tf.zeros([784,10]))
	b=tf.Variable(tf.zeros([10]))
	pred=tf.nn.softmax(tf.matmul(x,W)+b)
	cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))
	optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
	init=tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(training_epochs):
			avg_cost=0.
			total_batch=int(mnist.train.num_examples/batch_size)
			for i in range(total_batch):
				batch_xs,batch_ys=mnist.train.next_batch(batch_size)
				_,c=sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
				avg_cost+=c/total_batch
			if (epoch+1) % display_step == 0:
				print("Epoch:",'%04d'%(epoch+1),"cost=","{:.9f}".format(avg_cost))
		print("Optimization Finished.")
		correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
		accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
		print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))

def multilayer_perceptron():
	from tensorflow.examples.tutorials.mnist import input_data
	mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)
	learning_rate=0.001
	training_epochs=15
	batch_size=100
	display_step=1
	n_hidden_1=256
	n_hidden_2=256
	n_input=784
	n_classes=10
	x=tf.placeholder("float",[None,n_input])
	y=tf.placeholder("float",[None,n_classes])
	
	def perceptron(x,weights,biases):
		layer_1=tf.add(tf.matmul(x,weights['h1']),biases['b1'])
		layer_1=tf.nn.relu(layer_1)
		layer_2=tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
		layer_2=tf.nn.relu(layer_2)
		out_layer=tf.matmul(layer_2,weights['out'])+biases['out']
		return out_layer

	weights = {
			'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
			'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
			'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
		  }
	biases = {
			'b1':tf.Variable(tf.random_normal([n_hidden_1])),
			'b2':tf.Variable(tf.random_normal([n_hidden_2])),
			'out':tf.Variable(tf.random_normal([n_classes]))
		 }

	pred=perceptron(x,weights,biases)
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
	optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	init=tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(training_epochs):
			avg_cost=0.
			total_batch=int(mnist.train.num_examples/batch_size)
			for i in range(total_batch):
				batch_x,batch_y=mnist.train.next_batch(batch_size)
				_,c=sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
				avg_cost+=c/total_batch
			if epoch % display_step == 0:
				print("Epoch:",'%04d'%(epoch+1),"cost=","{:.9f}".format(avg_cost))
		print("Optimization Finished.")
		correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
		accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
		print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))

def convolutional_network():
	from tensorflow.examples.tutorials.mnist import input_data
	mnist=input_data.read_data_sets("tmp/data/",one_hot=True)
	learning_rate=0.001
	training_iters=200000
	batch_size=128
	display_step=10
	n_input=784
	n_classes=10
	dropout=0.75
	x=tf.placeholder(tf.float32,[None,n_input])
	y=tf.placeholder(tf.float32,[None,n_classes])
	keep_prob=tf.placeholder(tf.float32)
	
	def conv2d(x,W,b,strides=1):
		x=tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
		x=tf.nn.bias_add(x,b)
		return tf.nn.relu(x)

	def maxpool2d(x,k=2):
		return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

	def conv_net(x,weights,biases,dropout):
		x=tf.reshape(x,shape=[-1,28,28,1])
		conv1=conv2d(x,weights['wc1'],biases['bc1'])
		conv1=maxpool2d(conv1,k=2)
		conv2=conv2d(conv1,weights['wc2'],biases['bc2'])
		conv2=maxpool2d(conv2,k=2)
		fc1=tf.reshape(conv2,[-1,weights['wd1'].get_shape().as_list()[0]])
		fc1=tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
		fc1=tf.nn.relu(fc1)
		fc1=tf.nn.dropout(fc1,dropout)
		out=tf.add(tf.matmul(fc1,weights['out']),biases['out'])
		return out

	weights = {
	    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
	    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
	    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
	    'out': tf.Variable(tf.random_normal([1024, n_classes]))
	}

	biases = {
		'bc1':tf.Variable(tf.random_normal([32])),
		'bc2':tf.Variable(tf.random_normal([64])),
		'bd1':tf.Variable(tf.random_normal([1024])),
		'out':tf.Variable(tf.random_normal([n_classes]))
	}

	pred=conv_net(x,weights,biases,keep_prob)
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
	optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
	accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
	init=tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		step=1
		while step * batch_size < training_iters:
			batch_x,batch_y=mnist.train.next_batch(batch_size)
			sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})
			if step % display_step == 0:
				loss,acc=sess.run([cost,accuracy],feed_dict={x:batch_x,y:batch_y,keep_prob:.1})
				print("Iter "+str(step*batch_size)+", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy=" + \
					"{:.5f}".format(acc))
			step+=1
		print("Optimization Finished")
		print("Testing Accuracy", \
			sess.run(accuracy,feed_dict={x:mnist.test.images[:256],y:mnist.test.labels[:256],keep_prob:1.}))


def second_convnet():
	import random
	import numpy as np
	import matplotlib.pyplot as plt
	import datetime
	from tensorflow.examples.tutorials.mnist import input_data
	mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
	tf.reset_default_graph()
	sess=tf.InteractiveSession()
	x=tf.placeholder("float",shape=[None,28,28,1])
	y_=tf.placeholder("float",shape=[None,10])
	W_conv1=tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
	b_conv1=tf.Variable(tf.constant(.1,shape=[32]))
	print(x)
	print(W_conv1)	
	h_conv1=tf.nn.conv2d(input=x,filter=W_conv1,strides=[1,1,1,1],padding='SAME')+b_conv1
	h_conv1=tf.nn.relu(h_conv1)
	h_pool1=tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

	def conv2d(x,W):
		return tf.nn.conv2d(input=x,filter=W,strides=[1,1,1,1],padding='SAME')

	def max_pool_2x2(x):
		return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

	W_conv2=tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
	b_conv2=tf.Variable(tf.constant(.1,shape=[64]))
	h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
	h_pool2=max_pool_2x2(h_conv2)
	W_fcl=tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
	b_fcl=tf.Variable(tf.constant(.1,shape=[1024]))
	h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
	h_fcl=tf.nn.relu(tf.matmul(h_pool2_flat,W_fcl)+b_fcl)
	keep_prob=tf.placeholder("float")
	h_fcl_drop=tf.nn.dropout(h_fcl,keep_prob)
	W_fc2=tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
	b_fc2=tf.Variable(tf.constant(.1,shape=[10]))
	y=tf.matmul(h_fcl_drop,W_fc2)+b_fc2
	crossEntropyLoss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
	trainStep=tf.train.AdamOptimizer().minimize(crossEntropyLoss)
	
		
def main():
#	multiplication_basics()
#	hello_world()
#	matrix_multiplication()
#	nearest_neighbor()
#	logistic_regression()
#	multilayer_perceptron()
#	linear_regression()
#	convolutional_network()
	second_convnet()

main()
