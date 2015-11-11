"""Timing benchmark for AlexNet inference.

To run, use:
  bazel run -c opt --config=cuda \
      third_party/tensorflow/models/image/alexnet:alexnet_benchmark

Across 100 steps on batch size = 128.

Forward pass:
Run on Tesla K40c: 145 +/- 1.5 ms / batch
Run on Titan X:     70 +/- 0.1 ms / batch

Forward-backward pass:
Run on Tesla K40c: 480 +/- 48 ms / batch
Run on Titan X:    244 +/- 30 ms / batch

Note that this benchmark does not include the local response normalization
(LRN) layers of AlexNet.
"""
from __future__ import print_function
from datetime import datetime
import math
import numpy as np
import time

import tensorflow.python.platform
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_batches', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('l2_loss', False,
                            """Use L2 loss, rather than softmax.""")

NUM_LABELS = 1000

def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())


def inference(images, train=True):
  """Build the AlexNet model.

  Args:
    images: Images Tensor

  Returns:
    pool5: the last Tensor in the convolutional component of AlexNet.
    parameters: a list of Tensors corresponding to the weights and biases of the
        AlexNet model.
  """
  parameters = []
  # conv1
  with tf.name_scope('conv1') as scope:
    kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='VALID')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    conv1 = tf.nn.relu(bias, name=scope)
    print_activations(conv1)
    parameters += [kernel, biases]

  # lrn1
  # TODO(shlens, jiayq): Add a GPU version of local response normalization.

  # pool1
  pool1 = tf.nn.max_pool(conv1,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool1')
  print_activations(pool1)

  # conv2
  with tf.name_scope('conv2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    conv2 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
  print_activations(conv2)

  # pool2
  pool2 = tf.nn.max_pool(conv2,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool2')
  print_activations(pool2)

  # conv3
  with tf.name_scope('conv3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    conv3 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv3)

  # conv4
  with tf.name_scope('conv4') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    conv4 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv4)

  # conv5
  with tf.name_scope('conv5') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    conv5 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv5)

  # pool5
  pool5 = tf.nn.max_pool(conv5,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool5')
  print_activations(pool5)

  pool5_shape = pool5.get_shape().as_list()
  pool5_dim = np.prod(pool5_shape[1:])
  reshape = tf.reshape(pool5, [pool5_shape[0], pool5_dim])

  # fc6
  with tf.name_scope('fc6') as scope:
    weights = tf.Variable(tf.truncated_normal([pool5_dim, 4096],
                                              dtype=tf.float32,
                                              stddev=1e-1), name='weights')
    acts = tf.matmul(reshape, weights)
    biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(acts, biases)
    fc6 = tf.nn.relu(bias, name=scope)
    if train:
      fc6 = tf.nn.dropout(fc6, 0.5, name=scope)
    parameters += [weights, biases]
    print_activations(fc6)

  # fc7
  with tf.name_scope('fc7') as scope:
    weights = tf.Variable(tf.truncated_normal([4096, 4096],
                                              dtype=tf.float32,
                                              stddev=1e-1), name='weights')
    acts = tf.matmul(fc6, weights)
    biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(acts, biases)
    fc7 = tf.nn.relu(bias, name=scope)
    if train:
      fc7 = tf.nn.dropout(fc7, 0.5, name=scope)
    parameters += [weights, biases]
    print_activations(fc7)

  # fc8
  with tf.name_scope('fc8') as scope:
    weights = tf.Variable(tf.truncated_normal([4096, NUM_LABELS],
                                              dtype=tf.float32,
                                              stddev=1e-1), name='weights')
    acts = tf.matmul(fc7, weights)
    biases = tf.Variable(tf.constant(0.0, shape=[NUM_LABELS], dtype=tf.float32),
                         trainable=True, name='biases')
    fc8 = tf.nn.bias_add(acts, biases, name=scope)
    parameters += [weights, biases]
    print_activations(fc8)

  return fc8, parameters


def time_tensorflow_run(session, target, info_string):
  """Run the computation to obtain the target tensor and print timing stats.

  Args:
    session: the TensorFlow session to run the computation under.
    target: the targe Tensor that is passed to the session's run() function.
    info_string: a string summarizing this run, to be printed with the stats.

  Returns:
    None
  """
  num_steps_burn_in = 10
  total_duration = 0.0
  total_duration_squared = 0.0
  for i in xrange(FLAGS.num_batches + num_steps_burn_in):
    start_time = time.time()
    _ = session.run(target)
    duration = time.time() - start_time
    if i > num_steps_burn_in:
      if not i % 10:
        print ('%s: step %d, duration = %.3f' %
               (datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration
      total_duration_squared += duration * duration
  mn = total_duration / FLAGS.num_batches
  vr = total_duration_squared / FLAGS.num_batches - mn * mn
  sd = math.sqrt(vr)
  print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
         (datetime.now(), info_string, FLAGS.num_batches, mn, sd))



def run_benchmark():
  """Run the benchmark on AlexNet."""
  with tf.Graph().as_default():
    # Generate some dummy images.
    image_size = 224
    # Note that our padding definition is slightly different the cuda-convnet.
    # In order to force the model to start with the same activations sizes,
    # we add 3 to the image_size and employ VALID padding above.
    images = tf.Variable(tf.random_normal([FLAGS.batch_size,
                                           image_size + 3,
                                           image_size + 3, 3],
                                          dtype=tf.float32,
                                          stddev=1e-1))
    label_indices = np.random.randint(0, NUM_LABELS, FLAGS.batch_size)
    one_hot_labels = np.zeros((FLAGS.batch_size, NUM_LABELS), dtype=np.float32)
    one_hot_labels[np.arange(FLAGS.batch_size), label_indices] = 1
    labels = tf.Variable(one_hot_labels)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, parameters = inference(images)

    # Build an initialization operation.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session('')
    sess.run(init)

    # Run the forward benchmark.
    time_tensorflow_run(sess, logits, "Forward")

    # Add the objective so we can calculate the backward pass.
    if FLAGS.l2_loss:
      objective = tf.nn.l2_loss(logits)
    else:
      objective = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
          logits, labels))
    # Compute the gradient with respect to all the parameters.
    grad = tf.gradients(objective, parameters)
    sink = tf.group(*grad)
    # Run the backward benchmark.
    time_tensorflow_run(sess, sink, "Forward-backward")


def main(_):
  run_benchmark()


if __name__ == '__main__':
  tf.app.run()
