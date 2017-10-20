# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
# pylint: disable=line-too-long
r"""TensorFlow Eager Execution Example: Linear Regression.

This example shows how to use TensorFlow Eager Execution to fit a simple linear
regression model using some synthesized data. Specifically, it illustrates how
to define the forward path of the linear model and the loss function, as well
as how to obtain the gradients of the loss function with respect to the
variables and update the variables with the gradients.
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# TODO(cais): Use tf.contrib.eager namespace when ready.
from tensorflow.contrib.eager.python import tfe


class DataGenerator(object):
  """Generates synthetic data for linear regression."""

  def __init__(self, w, b, noise_level, batch_size):
    self._w = w
    self._b = b
    self._noise_level = noise_level
    self._batch_size = batch_size
    self._ndims = w.shape[0]

  def next_batch(self):
    """Generate a synthetic batch of xs and ys."""
    xs = tf.random_normal([self._batch_size, self._ndims])
    ys = (tf.matmul(xs, self._w) + self._b +
          self._noise_level * tf.random_normal([self._batch_size, 1]))
    return xs, ys


class LinearModel(object):
  """A TensorFlow linear regression model.

  Uses TensorFlow's eager execution.

  For those familiar with TensorFlow graphs, notice the absence of
  `tf.Session`. The `forward()` method here immediately executes and
  returns output values. The `loss()` method immediately compares the
  output of `forward()` with the target adn returns the MSE loss value.
  The `fit()` performs gradient-descent training on the model's weights
  and bias.
  """

  def __init__(self):
    """Constructs a LinearModel object."""
    self._hidden_layer = tf.layers.Dense(1)

    # loss_value_and_grad_fn is a function that when invoked, will return the
    # loss value and the gradients of loss with respect to the variables. It has
    # the same input arguments as `self.loss()`.
    self._loss_value_and_grad_fn = tfe.implicit_value_and_gradients(self.loss)

  @property
  def weights(self):
    """Get values of weights as a numpy array."""
    return self._hidden_layer.variables[0].read_value().numpy()

  @property
  def biases(self):
    """Get values of biases as a numpy array."""
    return self._hidden_layer.variables[1].read_value().numpy()

  def forward(self, xs):
    """Invoke the linear model.

    Args:
      xs: input features, as a tensor of size [batch_size, ndims].

    Returns:
      ys: the predictions of the linear mode, as a tensor of size [batch_size]
    """
    # Note: Unlike classic TensorFlow, operations such as self._hidden_layer
    # will execute the underlying computation immediately.
    return self._hidden_layer(xs)

  def loss(self, xs, ys):
    """Loss of the linear model.

    Args:
      xs: input features, as a tensor of size [batch_size, ndims].
      ys: the target values of y, as a tensor of size [batch_size].

    Returns:
      The mean square error loss value.
    """
    return tf.reduce_mean(tf.square(self.forward(xs) - ys))

  def fit(self,
          batch_fn,
          optimizer,
          num_iters,
          verbose=False,
          logdir=None):
    """Fit the linear-regression model.

    Args:
      batch_fn: A function, which when called without any arguments, returns a
        batch of xs and ys for training.
      optimizer: The TensorFlow Optimizer object to be used.
      num_iters: Number of training iterations to perform.
      verbose: If true, will print out loss values at every iteration.
      logdir: The directory in which summaries will be written for TensorBoard
        (optional).
    """
    if logdir:
      # Support for TensorBoard summaries. Once training has started, use:
      #   tensorboard --logdir=<logdir>
      summary_writer = tfe.SummaryWriter(logdir)

    # Training loop.
    for i in xrange(num_iters):
      # Generate a (mini-)batch of data for training.
      xs, ys = batch_fn()

      # Call the function obtained above to get the loss and gradient values at
      # the specific training batch. The function has the same input arguments
      # as the forward function, i.e., `linear_loss()`.
      loss_value, grads_and_vars = self._loss_value_and_grad_fn(xs, ys)
      if verbose:
        print("Iteration %d: loss = %s" % (i, loss_value.numpy()))

      # Send the gradients to the optimizer and update the Variables, i.e., `w`
      # and `b`.
      optimizer.apply_gradients(grads_and_vars)

      if logdir:
        summary_writer.scalar("loss", loss_value)
        summary_writer.step()


def main(_):
  # Ground-truth constants.
  true_w = np.array([[-2.0], [4.0], [1.0]], dtype=np.float32)
  true_b = np.array([0.5], dtype=np.float32)
  noise_level = 0.01

  # Training constants.
  batch_size = 64
  learning_rate = 0.1
  num_iters = 20

  print("True w: %s" % true_w)
  print("True b: %s\n" % true_b)

  device = "gpu:0" if tfe.num_gpus() else "cpu:0"
  print("Using device: %s" % device)
  with tf.device(device):
    linear_model = LinearModel()

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    data_gen = DataGenerator(true_w, true_b, noise_level, batch_size)
    linear_model.fit(data_gen.next_batch, optimizer, num_iters, verbose=True,
                     logdir=FLAGS.logdir)

  print("\nAfter training: w = %s" % linear_model.weights)
  print("\nAfter training: b = %s" % linear_model.biases)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--logdir",
      type=str,
      default=None,
      help="logdir in which TensorBoard summaries will be written (optional).")
  FLAGS, unparsed = parser.parse_known_args()

  # Use tfe.run() instead of tf.app.run() for eager execution.
  tfe.run(main=main, argv=[sys.argv[0]] + unparsed)
