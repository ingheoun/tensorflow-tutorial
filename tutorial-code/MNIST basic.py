# Import the MNIST-set from tensorflow tutorial file.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Import tf
import tensorflow as tf
sess = tf.InteractiveSession()

# Make a placeholder for training batch
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Make W (weight) and b(bias) for regression 
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Initialize the variables
sess.run(tf.initialize_all_variables())

# Calculate the output of regression and softmax
y = tf.nn.softmax(tf.matmul(x,W) + b)

# Design a cost function model with cross-entropy
# reduction_indices = [1] means sum through the dimension 1 (col-dimension)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices = [1]))

## Training phase
# Use gradient descent w/ lr = 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# assign batch by feed_dict
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# calculate accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
