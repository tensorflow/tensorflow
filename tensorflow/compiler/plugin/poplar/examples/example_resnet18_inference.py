# 2016 Graphcore
# carlo@graphcore.ai
# ===============================================
"""Convolutional Network for MNIST classification."""

import tensorflow as tf
import numpy as np

datatype = tf.float16

def _get_variable(name, shape, init):
  return tf.get_variable(name, shape, initializer=init, dtype=datatype)

def inference(x):

  with tf.variable_scope('all', use_resource=True):
    x = conv(x, 7, 2, 64)
    x = tf.nn.relu(x)
    x = max_pool(x, ksize=3, stride=2)
    x = block("b1", 64, 1, 2, x)
    x = block("b2", 128, 2, 2, x)
    x = block("b3", 256, 2, 2, x)
    x = block("b4", 512, 2, 2, x)
    x = tf.reduce_mean(x, reduction_indices=[1, 2])
    x = fc("fc1", x, 1000)

  return x


def block(name, out_filters, first_stride, count, x):

  for i in range(count):
    shortcut = x
    shape_in = x.get_shape()
    stride = (first_stride if (i==0) else 1)

    with tf.variable_scope(name + "/" + str(i) + "/1", use_resource=True):
      x = conv(x, 3, stride, out_filters)
      x = tf.nn.relu(x)

    with tf.variable_scope(name + "/" + str(i) + "/2", use_resource=True):
      x = conv(x, 3, 1, out_filters)

      # shortcut
      if (stride != 1):
        shortcut = tf.strided_slice(shortcut, [0,0,0,0], shortcut.get_shape(),
                                    strides=[1, stride, stride, 1])
      pad = int(x.get_shape()[3] - shape_in[3])
      if (pad != 0):
        shortcut = tf.pad(shortcut, paddings=[[0,0],[0,0],[0,0],[0,pad]])

      x = tf.nn.relu(x + shortcut)

  return x


def fc(name, x, num_units_out):
  num_units_in = x.get_shape()[1]
  weights_initializer = tf.truncated_normal_initializer(stddev=0.01)

  with tf.variable_scope(name, use_resource=True):
    weights = _get_variable('weights', shape=[num_units_in, num_units_out],
                            init=weights_initializer)
    biases = _get_variable('biases', shape=[num_units_out],
                           init=tf.constant_initializer(0.0))

    x = tf.nn.xw_plus_b(x, weights, biases)

  return x

def conv(x, ksize, stride, filters_out):

  filters_in = x.get_shape()[-1]
  shape = [ksize, ksize, filters_in, filters_out]
  initializer = tf.truncated_normal_initializer(stddev=0.1)

  weights = _get_variable('weights', shape=shape, init=initializer)
  return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def max_pool(x, ksize=3, stride=2):
  return tf.nn.max_pool(x,
                        ksize=[1, ksize, ksize, 1],
                        strides=[1, stride, stride, 1],
                        padding='SAME')


#
# Main code
#

# Inputs
x = tf.placeholder(datatype, shape=[1, 224, 224, 4])

with tf.device("/device:XLA_IPU:0"):
  # Inference
  logits = inference(x)

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

training_data = np.zeros([1, 224, 224, 4]);

sess.run(logits, feed_dict={x: training_data})

sess.close()
