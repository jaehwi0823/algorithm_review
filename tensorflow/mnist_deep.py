# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# 텐서플로 튜토리얼 리뷰

# Dependencies 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
FLAGS = None


# 아키텍쳐 생성 함수
def deepnn(x):
	"""
	Args:
	  x: MNIST image input tensor of shape (N_examples, 784)

	Returns:
	  A tuple (y, keep_prob).
	  y: 0~9에 대한 확률값을 갖는 텐서이며, shape은 (N_examples, 10)
	  keep_prob: dropout 확률
	"""
	# CNN 구조에 맞게 Reshape!
	# tf.reshape()에서 -1은 특수값으로 전체 차원을 유지하는 수가 자동으로 할당되며, tf.reshape[-1]은 1-D flattening 효과가 있음.
	x_image =tf.reshape(x, [-1, 28, 28, 1])

	# Shape Parameter대로 각 tf변수 및 tf상수를 초기화&생성 해준다.
	# 첫번째 Conv Layer에서 feature map 수는 32개로 증가시킨다.
	# 첫번째 Conv Layer는 Relu를 Activation Func으로 사용한다.
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

	# relu를 통과한 직후 max pooling
	# shape: [-1, 14, 14, 32]
	h_pool1 = max_pool_2x2(h_conv1)

	# 두번째 필터는 feature map을 32개에서 64개로 증폭
	W_conv2 =weight_variable([5, 5, 32, 64])
	b_conv2 =bias_variable([64])
	h_conv2 =tf.nn.relu(conv2d(h_pool1, W_conv2) +b_conv2)

	# relu를 통과한 직후 max pooling
	# shape: [-1, 7, 7, 64]
	h_pool2 = max_pool_2x2(h_conv2)

	# FC로 연결하기 위해서 7*7*64 -> 1024 를 연결해주는 변수생성
	W_fc1 = weight_variable([7*7*64, 1024])
	b_fc1 = bias_variable([1024])

	# 두번째 pooling layer을 통과한 후의 shape: [-1, 7, 7, 64] -> -[1, 7*7*64])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	# relu를 지나며 shape: [-1, 1024]
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	#Dropout을 통해 Regularization
	#Dropout을 할 때 keep_prop의 확률만큼만 노드들을 살리고 나머지는 죽인다.
	#대신 살아남은 노드들은 1/keep_prop만큼 scale-up시킨다.
	#왜냐하면 drop된 노드들 때문에 원래 모형이 예측하려는 값보다 전체 결과값이 줄어들었기 때문
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	# 두번째 FC를 이용해서 1024차원에서 10차원 (최종결과)로 mapping 해준다.
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	y_conv =tf.matmul(h_fc1_drop, W_fc2) +b_fc2


	return y_conv, keep_prob


def conv2d(x, W):
	"""conv2d returns a 2d convolution layer with full stride."""
	return  tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
	"""weight_variable generates a weight variable of a given shape."""
	initial =tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	"""bias_variable generates a bias variable of a given shape."""
	initial =tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def main(_):
	# Import data
	mnist =input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

	# Create the model
	x =tf.placeholder(tf.float32, [None, 784])

	# Define loss and optimizer
	y_ =tf.placeholder(tf.float32, [None, 10])

	# Build the graph for the deep net
	y_conv, keep_prob =deepnn(x)

	cross_entropy =tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	train_step =tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction =tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	accuracy =tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(20000):
			batch =mnist.train.next_batch(50)
			if i %100==0:
				train_accuracy =accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
			print('step %d, training accuracy %g'%(i, train_accuracy))
			train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

		print('test accuracy %g'%accuracy.eval(feed_dict={
		x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

	if __name__ == '__main__':
		parser =argparse.ArgumentParser()
		parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
							help='Directory for storing input data')
		FLAGS, unparsed =parser.parse_known_args()
		tf.app.run(main=main, argv=[sys.argv[0]] +unparsed)
