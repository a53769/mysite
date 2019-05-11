import tensorflow as tf
import numpy as np

def print_acrivations(t):
    """
    显示每一层的名称(t.op.name)和tensor的尺寸(t.get_shape.as_list())
    param :
    t -- Tensor类型的输入
    """
    print(t.op.name," ",t.get_shape().as_list())

class AlexNet():
	#实现AlexNet
	def __init__(self, x, keep_prob, num_classes, training):
		'''crate the graph of the AlexNet model
		Args:
		x: Placeholder for the input tensor
		keep_prob: Dropout probability
		num_classes: Number of classes in the dataset
		'''
		self.x = x
		self.keep_prob = keep_prob
		self.num_classes = num_classes

		self.create(training)

	def create(self,training):
		'''Create the network graph.'''

		# 1 layer:conv  lrn  pool
		with tf.variable_scope("conv1"):
			#卷积核大小为11*11*3，个数为96个
			conv1_weights = tf.get_variable(name="weight_conv1",shape=[11,11,3,96],
											dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01))
			#初始化biases全部为0
			conv1_biases = tf.get_variable(name="biases_conv1",shape=[96],
											dtype=tf.float32, initializer=tf.constant_initializer(0.0))
			#卷积操作，步长=4，padding=‘VALID’
			conv1 = tf.nn.conv2d(input=self.x, filter=conv1_weights, strides=[1,4,4,1],padding='VALID')
			#将conv1和biases相加，然后进行tanh激活
			conv1 = tf.nn.tanh(tf.nn.bias_add(conv1,conv1_biases), name='conv1_layer')
			#卷积层后添加LRN层（除AlexNet其他经典网络基本无LRN层）
			lrn1 = tf.nn.lrn(conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0, name='lrn1')
			 # 最大池化层(尺寸为 3*3，步长s=2*2)
			pool1 = tf.nn.max_pool(lrn1,ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='poo1')
			pool1 = tf.layers.batch_normalization(inputs=pool1,training=training)
			# print_acrivations(pool1)
		with tf.variable_scope("conv2"):
			#卷积核大小为5*5*96，个数为256个
			conv2_weights = tf.get_variable(name="weight_conv2",shape=[5,5,96,256],
											dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01))
			#初始化biases全部为0
			conv2_biases = tf.get_variable(name="biases_conv2",shape=[256],
											dtype=tf.float32, initializer=tf.constant_initializer(0.0))
			#卷积操作，步长=1，padding=‘VALID’
			conv2 = tf.nn.conv2d(input=pool1, filter=conv2_weights, strides=[1,1,1,1],padding='SAME')
			#将conv2和biases相加，然后进行tanh激活
			conv2 = tf.nn.tanh(tf.nn.bias_add(conv2,conv2_biases), name='conv2_layer')
			#卷积层后添加LRN层（除AlexNet其他经典网络基本无LRN层）
			lrn2 = tf.nn.lrn(conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0, name='lrn2')
			 # 最大池化层(尺寸为 3*3，步长s=2*2)
			pool2 = tf.nn.max_pool(lrn2,ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool2')
			pool2 = tf.layers.batch_normalization(inputs=pool2,training=training)
			# print_acrivations(pool2)
		with tf.variable_scope("conv3"):
			#卷积核大小为3*3*256，个数为384
			conv3_weights = tf.get_variable(name="weight_conv3",shape=[3,3,256,384],
											dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01))
			#初始化biases全部为0
			conv3_biases = tf.get_variable(name="biases_conv3",shape=[384],
											dtype=tf.float32, initializer=tf.constant_initializer(0.0))
			#卷积操作，步长=1，padding=‘VALID’
			conv3 = tf.nn.conv2d(input=pool2, filter=conv3_weights, strides=[1,1,1,1], padding='SAME')
			#conv3和biases相加，然后进行tanh激活
			conv3 = tf.nn.tanh(tf.nn.bias_add(conv3,conv3_biases),name='conv3_layer')
			conv3 = tf.layers.batch_normalization(inputs=conv3,training=training)
			# print_acrivations(conv3)
		with tf.variable_scope("conv4"):
			#卷积核大小为3*3*384，个数为384
			conv4_weights = tf.get_variable(name="weight_conv4", shape=[3,3,384,384],
											dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01))
			#初始化biases全部为0
			conv4_biases = tf.get_variable(name="biases_conv4",shape=[384],
											dtype=tf.float32,initializer=tf.constant_initializer(0.0))
			#卷积操作，步长为1，padding='VALID'
			conv4 = tf.nn.conv2d(input=conv3, filter=conv4_weights, strides=[1,1,1,1], padding='SAME')
			#conv4和biases相加，然后进行tanh激活
			conv4 = tf.nn.tanh(tf.nn.bias_add(conv4,conv4_biases), name='conv4_layer')
			conv4 = tf.layers.batch_normalization(inputs=conv4,training=training)
			# print_acrivations(conv4)
		with tf.variable_scope("conv5"):
			#卷积核大小为3*3*384，个数为256
			conv5_weights = tf.get_variable(name="weight_conv5", shape=[3,3,384,256],
											dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01))
			#初始化biases全部为0
			conv5_biases = tf.get_variable(name="biases_conv5", shape=[256],
											dtype=tf.float32, initializer=tf.constant_initializer(0.0))
			#卷积操作，步长为1，padding=‘VALID’
			conv5 = tf.nn.conv2d(input=conv4, filter=conv5_weights, strides=[1,1,1,1], padding='SAME')
			#conv4和biases相加，然后进行tanh激活
			conv5 = tf.nn.tanh(tf.nn.bias_add(conv5,conv5_biases), name='conv5_layer')
			# 最大池化层(尺寸为 3*3，步长s=2*2)
			pool5 = tf.nn.max_pool(conv5,ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool5')
			pool5 = tf.layers.batch_normalization(inputs=pool5, training=training)
			# print_acrivations(pool5)
		#第六层 ----全连接层及dropout
		pool5_flat = tf.reshape(pool5, [-1, 6*6*256])

		with tf.variable_scope("fc6"):
			fc6_weights = tf.get_variable(name="weight_fc6", shape=[6*6*256,4096],
											dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
			fc6_biases = tf.get_variable(name="biases_fc6", shape=[4096],
											dtype=tf.float32, initializer=tf.constant_initializer(0.0))
			fc6 = tf.nn.xw_plus_b(pool5_flat,fc6_weights,fc6_biases,name='plus_fc6')
			fc6 = tf.nn.tanh(fc6)
			#dropout
			fc6 = tf.nn.dropout(fc6, keep_prob=self.keep_prob)

		with tf.variable_scope("fc7"):
			fc7_weights = tf.get_variable(name="weight_fc7",shape=[4096,4096],
											dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
			fc7_biases = tf.get_variable(name="biases_fc7",shape=[4096],
											dtype=tf.float32,initializer=tf.constant_initializer(0.0))
			fc7 = tf.nn.xw_plus_b(fc6,fc7_weights,fc7_biases,name='plus_fc7')
			fc7 = tf.nn.tanh(fc7)
			#dropout
			fc7 = tf.nn.dropout(fc7,keep_prob=self.keep_prob)

		with tf.variable_scope("fc8"):
			fc8_weights = tf.get_variable(name="weight_fc8",shape=[4096,self.num_classes],
											dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
			fc8_biases = tf.get_variable(name="biases_fc8", shape=[self.num_classes],
											dtype=tf.float32, initializer=tf.constant_initializer(0.0))
			self.fc8 = tf.nn.xw_plus_b(fc7,fc8_weights,fc8_biases,name='plus_fc8')
