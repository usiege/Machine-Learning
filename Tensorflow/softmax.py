
w = tf.Variable(tf.zeros([4, 3]), name='weights')
b = tf.Variable(tf.zeros([3], name='bias'))

def inference(X):
	return tf.nn.softmax(combine_inputs(X))

def loss(X, Y):
	return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(combine_inputs(X, Y)))	

def inputs():

	sepal_length, sepal_width, petal_length, petal_width, label = \
	read_csv(100, 'iris.data', [[0.0], [0.0], [0.0], [0.0], ['']])

	#将类名称转换为从0开始计的类别索引
	label_number = tf.to_int32(tf.argmax(tf.to_int32(tf.pack([
		tf.equal(label, ['Iris-setosa']),
		tf.equal(label, ['Iris-versicolor']),
		tf.equal(label, ['Iris-virginica'])
		])), 0))

	#将所关心的所有特征装入单个矩阵中，然后对该矩阵转置，使其每行对应一个样本，而每列对应一个特征
	features = tf.transpose(tf.pack([sepal_length, sepal_width, petal_length, petal_width]))

	return features, label_number

def evaluate(sess, X, Y):
	predicted = tf.cast(tf.arg_max(inference(X), 1), tf.int32)

	print sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32)))



'''
cast(x, dtype, name=None) 
将x的数据格式转化成dtype.例如，原来x的数据格式是bool， 
那么将其转化成float以后，就能够将其转化成0和1的序列。反之也可以

a = tf.Variable([1,0,0,1,1])
b = tf.cast(a,dtype=tf.bool)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(sess.run(b))
#[ True False False  True  True]

---------------------
'''


'''
reduce_mean

x = tf.constant([[1., 1.], [2., 2.]])
tf.reduce_mean(x)  # 1.5
tf.reduce_mean(x, 0)  # [1.5, 1.5]
tf.reduce_mean(x, 1)  # [1.,  2.]

# 如果不设置axis，所有维度上的元素都会被求平均值，并且只会返回一个只有一个元素的张量。

'''
