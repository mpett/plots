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

def main():
	#multiplication_basics()
	#hello_world()
	matrix_multiplication()

main()
