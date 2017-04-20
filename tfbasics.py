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

def main():
	#multiplication_basics()
	#hello_world()
	#matrix_multiplication()
	nearest_neighbor()

main()
