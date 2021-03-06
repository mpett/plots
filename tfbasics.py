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
	correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
	sess.run(tf.global_variables_initializer())
	tf.summary.scalar('Cross Entropy Loss',crossEntropyLoss)
	tf.summary.scalar('Accuracy',accuracy)
	merged=tf.summary.merge_all()
	logdir="tensorboard/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/"
	writer=tf.summary.FileWriter(logdir,sess.graph)
	b=mnist.train.next_batch(1)
	print(b[0].shape)
	image=tf.reshape(b[0],[-1,28,28,1])
	print(image)
	my_img=image.eval()
	my_i=my_img.squeeze()
	plt.imshow(my_i,cmap='gray_r')
	plt.show()
	batchSize=50

	for i in range(1000):
		batch=mnist.train.next_batch(batchSize)
		trainingInputs=batch[0].reshape([batchSize,28,28,1])
		trainingLabels=batch[1]
		if i%10==0:
			summary=sess.run(merged,{x:trainingInputs,y_:trainingLabels,keep_prob:1.0})
			writer.add_summary(summary,i)
		if i%100==0:
			trainAccuracy=accuracy.eval(session=sess,feed_dict={x:trainingInputs,y_:trainingLabels,keep_prob:1.0})
			print("step %d, training accuracy %g"%(i,trainAccuracy))
		trainStep.run(session=sess,feed_dict={x:trainingInputs,y_:trainingLabels,keep_prob:0.5})

	testInputs=mnist.test.images.reshape([-1,28,28,1])
	testLabels=mnist.test.labels
	acc=accuracy.eval(feed_dict={x:testInputs,y_:testLabels,keep_prob:1.0})
	print("test accuracy: {}".format(acc))

def recurrent_neural_network_for_spam_detection():
	import os
	import re
	import io
	import requests
	import numpy as np
	import matplotlib.pyplot as plt
	from zipfile import ZipFile
	from tensorflow.python.framework import ops
	ops.reset_default_graph()
	sess = tf.Session()
	epochs=20
	batch_size=250
	max_sequence_length=25
	rnn_size=10
	embedding_size=50
	min_word_frequency=10
	learning_rate=0.0005
	dropout_keep_prob=tf.placeholder(tf.float32)
	data_dir='temp'

def gaussian_mixture_models():
	import numpy as np
	import matplotlib.pyplot as plt
	from scipy import stats
	import seaborn as sns; sns.set()

def modern_multilayer_perceptron():
	import numpy as np
	import tensorflow as tf
	import matplotlib.pyplot as plt
	from tensorflow.examples.tutorials.mnist import input_data
	mnist=input_data.read_data_sets('data',one_hot=True)
	X_train=mnist.train.images
	Y_train=mnist.train.labels
	X_test=mnist.test.images
	Y_test=mnist.test.labels
	dimX=X_train.shape[1]
	dimY=Y_train.shape[1]
	nTrain=X_train.shape[0]
	nTest=X_test.shape[0]
	print("Shape of (X_train, X_test, Y_train, Y_test)")
	print(X_train.shape, X_test.shape,Y_train.shape,Y_test.shape)
	
	def xavier_init(n_inputs,n_outputs,uniform=True):
		if uniform:
			init_range=tf.sqrt(6.0/(n_inputs+n_outputs))
			return tf.random_uniform_initializer(-init_range,init_range)
		else:
			stddev=tf.sqrt(3.0,(n_inputs+n_outputs))
		return tf.truncated_normal_initializer(stddev=stddev)

	nLayer0=dimX
	nLayer1=256
	nLayer2=256
	nLayer3=dimY
	sigma_init=0.1

	W = {
		'W1': tf.Variable(tf.random_normal([nLayer0,nLayer1],stddev=sigma_init)),
		'W2': tf.Variable(tf.random_normal([nLayer1,nLayer2],stddev=sigma_init)),
		'W3': tf.Variable(tf.random_normal([nLayer2,nLayer3],stddev=sigma_init))
	    }

	b = {
		'b1': tf.Variable(tf.random_normal([nLayer1])),
		'b2': tf.Variable(tf.random_normal([nLayer2])),
		'b3': tf.Variable(tf.random_normal([nLayer3]))
	    }

	def model_myNN(_X,_W,_b):
		Layer1=tf.nn.sigmoid(tf.add(tf.matmul(_X,_W['W1']),_b['b1']))
		Layer2=tf.nn.sigmoid(tf.add(tf.matmul(Layer1,_W['W2']),_b['b2']))
		Layer3=tf.add(tf.matmul(Layer2,_W['W3']),_b['b3'])
		return Layer3

	X=tf.placeholder(tf.float32,[None,dimX],name="input")
	Y=tf.placeholder(tf.float32,[None,dimY],name="output")
	Y_pred=model_myNN(X,W,b)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=Y))	
	learning_rate=0.001
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
	training_epochs=30
	display_epoch=5
	batch_size=100
	correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

def chi_squared_feature_selection():
	import numpy as np
	import pandas as pd
	from sklearn.preprocessing import LabelBinarizer
	from sklearn.feature_selection import chi2, SelectKBest
	from sklearn.feature_extraction.text import CountVectorizer
	X = np.array(['call you tonight', 'Call me a cab', 'please call me... please', 'he will call me'])
	y = [1,1,2,0]
	vect = CountVectorizer()
	X_dtm = vect.fit_transform(X)
	X_dtm = X_dtm.toarray()
	print(pd.DataFrame(X_dtm, columns = vect.get_feature_names()))
	y_binarized = LabelBinarizer().fit_transform(y)
	print(y_binarized)
	print()
	observed = np.dot(y_binarized.T, X_dtm)
	print(observed)
	class_prob = y_binarized.mean(axis = 0).reshape(1, -1)
	feature_count = X_dtm.sum(axis = 0).reshape(1, -1)
	expected = np.dot(class_prob.T, feature_count)
	print(expected)
	chisq = (observed - expected) ** 2 / expected
	chisq_score = chisq.sum(axis = 0)
	print(chisq_score)
	chi2score = chi2(X_dtm, y)
	print(chi2score)
	kbest = SelectKBest(score_func = chi2, k = 4)
	X_dtm_kbest = kbest.fit_transform(X_dtm, y)
	print(X_dtm_kbest)

def genetic():
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	chromo_size = 5
	low = 0
	high = 100
	lol = np.random.randint(low, high + 1, chromo_size)
	print(lol)
	pop_size = 6
	pop = np.random.randint(low, high + 1, (pop_size, chromo_size))
	print(pop)
	target = 200
	cost = np.abs(np.sum(pop, axis = 1) - target)
	graded = [(c, list(p)) for p, c in zip(pop, cost)]

	for cost, chromo in graded:
		print("chromo {}'s cost is {}".format(chromo, cost))

def main():
	genetic()
	chi_squared_feature_selection()
	multiplication_basics()
	hello_world()
	matrix_multiplication()
	nearest_neighbor()
	logistic_regression()
	multilayer_perceptron()
	linear_regression()
	convolutional_network()
	second_convnet()
	recurrent_neural_network_for_spam_detection()
	gaussian_mixture_models()
	modern_multilayer_perceptron()
	
main()
