import numpy as np
import tensorflow as tf

def weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.random.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.compat.v1.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.compat.v1.summary.scalar('stddev', stddev)
    tf.compat.v1.summary.scalar('max', tf.reduce_max(var))
    tf.compat.v1.summary.scalar('min', tf.reduce_min(var))
    tf.compat.v1.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.
  It does a matrix multiply, bias add, and then uses an activation function
  (ReLu defauly) to nonlinearize. It also sets up name scoping so that the
  resultant graph is easy to read, and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.compat.v1.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.compat.v1.summary.histogram('activations', activations)
    return activations


def cross_entropy_training(y, y_, learning_rate=0.001):
  """Reusable code to add a cross entropy loss function and
  optimser to a model"""

  with tf.name_scope('cross_entropy'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.losses.sparse_softmax_cross_entropy on the
    # raw logit outputs of the nn_layer above, and then average across
    # the batch.
    with tf.name_scope('total'):
      cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(
          labels=y_, logits=y)
  tf.compat.v1.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).\
      minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.compat.v1.summary.scalar('accuracy', accuracy)

  return [train_step, accuracy]
