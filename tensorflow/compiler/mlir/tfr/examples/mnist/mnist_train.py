# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MNIST model float training script with TensorFlow graph execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.compiler.mlir.tfr.examples.mnist import gen_mnist_ops
from tensorflow.compiler.mlir.tfr.examples.mnist import ops_defs  # pylint: disable=unused-import
from tensorflow.python.framework import load_library

flags.DEFINE_integer('train_steps', 200, 'Number of steps in training.')

_lib_dir = os.path.dirname(gen_mnist_ops.__file__)
_lib_name = os.path.basename(gen_mnist_ops.__file__)[4:].replace('.py', '.so')
load_library.load_op_library(os.path.join(_lib_dir, _lib_name))

# MNIST dataset parameters.
num_classes = 10  # total classes (0-9 digits).
num_features = 784  # data features (img shape: 28*28).
num_channels = 1

# Training parameters.
learning_rate = 0.01
display_step = 10
batch_size = 128

# Network parameters.
n_hidden_1 = 32  # 1st conv layer number of neurons.
n_hidden_2 = 64  # 2nd conv layer number of neurons.
n_hidden_3 = 1024  # 1st fully connected layer of neurons.
flatten_size = num_features // 16 * n_hidden_2

seed = 66478


class FloatModel(tf.Module):
  """Float inference for mnist model."""

  def __init__(self):
    self.weights = {
        'f1':
            tf.Variable(
                tf.random.truncated_normal([5, 5, num_channels, n_hidden_1],
                                           stddev=0.1,
                                           seed=seed)),
        'f2':
            tf.Variable(
                tf.random.truncated_normal([5, 5, n_hidden_1, n_hidden_2],
                                           stddev=0.1,
                                           seed=seed)),
        'f3':
            tf.Variable(
                tf.random.truncated_normal([n_hidden_3, flatten_size],
                                           stddev=0.1,
                                           seed=seed)),
        'f4':
            tf.Variable(
                tf.random.truncated_normal([num_classes, n_hidden_3],
                                           stddev=0.1,
                                           seed=seed)),
    }

    self.biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_1])),
        'b2': tf.Variable(tf.zeros([n_hidden_2])),
        'b3': tf.Variable(tf.zeros([n_hidden_3])),
        'b4': tf.Variable(tf.zeros([num_classes])),
    }

  @tf.function
  def __call__(self, data):
    """The Model definition."""
    x = tf.reshape(data, [-1, 28, 28, 1])

    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input).

    # NOTE: The data/x/input is always specified in floating point precision.
    # output shape: [-1, 28, 28, 32]
    conv1 = gen_mnist_ops.new_conv2d(x, self.weights['f1'], self.biases['b1'],
                                     1, 1, 1, 1, 'SAME', 'RELU')

    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    # output shape: [-1, 14, 14, 32]
    max_pool1 = gen_mnist_ops.new_max_pool(conv1, 2, 2, 2, 2, 'SAME')

    # output shape: [-1, 14, 14, 64]
    conv2 = gen_mnist_ops.new_conv2d(max_pool1, self.weights['f2'],
                                     self.biases['b2'], 1, 1, 1, 1, 'SAME',
                                     'RELU')

    # output shape: [-1, 7, 7, 64]
    max_pool2 = gen_mnist_ops.new_max_pool(conv2, 2, 2, 2, 2, 'SAME')

    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    # output shape: [-1, 7*7*64]
    reshape = tf.reshape(max_pool2, [-1, flatten_size])

    # output shape: [-1, 1024]
    fc1 = gen_mnist_ops.new_fully_connected(reshape, self.weights['f3'],
                                            self.biases['b3'], 'RELU')
    # output shape: [-1, 10]
    return gen_mnist_ops.new_fully_connected(fc1, self.weights['f4'],
                                             self.biases['b4'])


def main(strategy):
  """Trains an MNIST model using the given tf.distribute.Strategy."""
  # TODO(fengliuai): put this in some automatically generated code.
  os.environ[
      'TF_MLIR_TFR_LIB_DIR'] = 'tensorflow/compiler/mlir/tfr/examples/mnist'

  ds_train = tfds.load('mnist', split='train', shuffle_files=True)
  ds_train = ds_train.shuffle(1024).batch(batch_size).prefetch(64)
  ds_train = strategy.experimental_distribute_dataset(ds_train)

  with strategy.scope():
    # Create an mnist float model with the specified float state.
    model = FloatModel()
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

  def train_step(features):
    inputs = tf.image.convert_image_dtype(
        features['image'], dtype=tf.float32, saturate=False)
    labels = tf.one_hot(features['label'], num_classes)

    with tf.GradientTape() as tape:
      logits = model(inputs)
      loss_value = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels, logits))

    grads = tape.gradient(loss_value, model.trainable_variables)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return accuracy, loss_value

  @tf.function
  def distributed_train_step(dist_inputs):
    per_replica_accuracy, per_replica_losses = strategy.run(
        train_step, args=(dist_inputs,))
    accuracy = strategy.reduce(
        tf.distribute.ReduceOp.MEAN, per_replica_accuracy, axis=None)
    loss_value = strategy.reduce(
        tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
    return accuracy, loss_value

  iterator = iter(ds_train)
  accuracy = 0.0
  for step in range(flags.FLAGS.train_steps):
    accuracy, loss_value = distributed_train_step(next(iterator))
    if step % display_step == 0:
      tf.print('Step %d:' % step)
      tf.print('    Loss = %f' % loss_value)
      tf.print('    Batch accuracy = %f' % accuracy)

  return accuracy
