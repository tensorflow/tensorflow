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
from absl import app
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

weights = {
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

biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'b3': tf.Variable(tf.zeros([n_hidden_3])),
    'b4': tf.Variable(tf.zeros([num_classes])),
}


class FloatModel(tf.Module):
  """Float inference for mnist model."""

  @tf.function
  def __call__(self, data):
    """The Model definition."""
    x = tf.reshape(data, [-1, 28, 28, 1])

    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input).

    # NOTE: The data/x/input is always specified in floating point precision.
    # output shape: [-1, 28, 28, 32]
    conv1 = gen_mnist_ops.new_conv2d(x, weights['f1'], biases['b1'], 1, 1, 1, 1,
                                     'SAME', 'RELU')

    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    # output shape: [-1, 14, 14, 32]
    max_pool1 = gen_mnist_ops.new_max_pool(conv1, 2, 2, 2, 2, 'SAME')

    # output shape: [-1, 14, 14, 64]
    conv2 = gen_mnist_ops.new_conv2d(max_pool1, weights['f2'], biases['b2'], 1,
                                     1, 1, 1, 'SAME', 'RELU')

    # output shape: [-1, 7, 7, 64]
    max_pool2 = gen_mnist_ops.new_max_pool(conv2, 2, 2, 2, 2, 'SAME')

    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    # output shape: [-1, 7*7*64]
    reshape = tf.reshape(max_pool2, [-1, flatten_size])

    # output shape: [-1, 1024]
    fc1 = gen_mnist_ops.new_fully_connected(reshape, weights['f3'],
                                            biases['b3'], 'RELU')
    # output shape: [-1, 10]
    return gen_mnist_ops.new_fully_connected(fc1, weights['f4'], biases['b4'])


def grad(model, inputs, labels, trainable_variables):
  with tf.GradientTape() as tape:
    logits = model(inputs)
    loss_value = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels, logits))
    grads = tape.gradient(loss_value, trainable_variables)
  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return accuracy, loss_value, grads


def training_step(model, inputs, labels, optimizer, step):
  trainable_variables = list(weights.values()) + list(biases.values())
  accuracy, loss_value, grads = grad(model, inputs, labels, trainable_variables)
  if step % display_step == 0:
    print('Step %d:' % step)
    print('    Loss = %f' % loss_value)
    print('    Batch accuracy: %f' % accuracy)
  optimizer.apply_gradients(zip(grads, trainable_variables))


def get_next_batch(iter_):
  features = next(iter_)
  images, labels = features['image'], features['label']
  return (mnist_preprocess(images), tf.one_hot(labels, num_classes))


def mnist_preprocess(x):
  x_float = tf.cast(x, tf.float32)
  return x_float / 255.0


def train(model, dataset, optimizer):
  iter_ = iter(dataset)
  for step in range(flags.FLAGS.train_steps):
    inputs, labels = get_next_batch(iter_)
    training_step(model, inputs, labels, optimizer, step)


def main(_):
  # TODO(fengliuai): put this in some automatically generated code.
  os.environ[
      'TF_MLIR_TFR_LIB_DIR'] = 'tensorflow/compiler/mlir/tfr/examples/mnist'
  # Create an mnist float model with the specified float state.
  model = FloatModel()
  optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

  ds_train = tfds.load('mnist', split='train', shuffle_files=True)
  ds_train = ds_train.shuffle(1024).batch(batch_size).prefetch(64)

  train(model, ds_train, optimizer)


if __name__ == '__main__':
  app.run(main)
