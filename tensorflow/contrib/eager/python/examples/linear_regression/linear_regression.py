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
r"""TensorFlow Eager Execution Example: Linear Regression.

This example shows how to use TensorFlow Eager Execution to fit a simple linear
regression model using some synthesized data. Specifically, it illustrates how
to define the forward path of the linear model and the loss function, as well
as how to obtain the gradients of the loss function with respect to the
variables and update the variables with the gradients.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

import tensorflow.contrib.eager as tfe


class LinearModel(tfe.Network):
  """A TensorFlow linear regression model.

  Uses TensorFlow's eager execution.

  For those familiar with TensorFlow graphs, notice the absence of
  `tf.Session`. The `forward()` method here immediately executes and
  returns output values. The `loss()` method immediately compares the
  output of `forward()` with the target and returns the MSE loss value.
  The `fit()` performs gradient-descent training on the model's weights
  and bias.
  """

  def __init__(self):
    """Constructs a LinearModel object."""
    super(LinearModel, self).__init__()
    self._hidden_layer = self.track_layer(tf.layers.Dense(1))

  def call(self, xs):
    """Invoke the linear model.

    Args:
      xs: input features, as a tensor of size [batch_size, ndims].

    Returns:
      ys: the predictions of the linear mode, as a tensor of size [batch_size]
    """
    return self._hidden_layer(xs)


def fit(model, dataset, optimizer, verbose=False, logdir=None):
  """Fit the linear-regression model.

  Args:
    model: The LinearModel to fit.
    dataset: The tf.data.Dataset to use for training data.
    optimizer: The TensorFlow Optimizer object to be used.
    verbose: If true, will print out loss values at every iteration.
    logdir: The directory in which summaries will be written for TensorBoard
      (optional).
  """

  # The loss function to optimize.
  def mean_square_loss(xs, ys):
    return tf.reduce_mean(tf.square(model(xs) - ys))

  loss_and_grads = tfe.implicit_value_and_gradients(mean_square_loss)

  tf.train.get_or_create_global_step()
  if logdir:
    # Support for TensorBoard summaries. Once training has started, use:
    #   tensorboard --logdir=<logdir>
    summary_writer = tf.contrib.summary.create_file_writer(logdir)

  # Training loop.
  for i, (xs, ys) in enumerate(tfe.Iterator(dataset)):
    loss, grads = loss_and_grads(xs, ys)
    if verbose:
      print("Iteration %d: loss = %s" % (i, loss.numpy()))

    optimizer.apply_gradients(grads, global_step=tf.train.get_global_step())

    if logdir:
      with summary_writer.as_default():
        with tf.contrib.summary.always_record_summaries():
          tf.contrib.summary.scalar("loss", loss)


def synthetic_dataset(w, b, noise_level, batch_size, num_batches):
  """tf.data.Dataset that yields synthetic data for linear regression."""

  # w is a matrix with shape [N, M]
  # b is a vector with shape [M]
  # So:
  # - Generate x's as vectors with shape [batch_size N]
  # - y = tf.matmul(x, W) + b + noise
  def batch(_):
    x = tf.random_normal([batch_size, tf.shape(w)[0]])
    y = tf.matmul(x, w) + b + noise_level * tf.random_normal([])
    return x, y

  with tf.device("/device:CPU:0"):
    return tf.data.Dataset.range(num_batches).map(batch)


def main(_):
  tfe.enable_eager_execution()
  # Ground-truth constants.
  true_w = [[-2.0], [4.0], [1.0]]
  true_b = [0.5]
  noise_level = 0.01

  # Training constants.
  batch_size = 64
  learning_rate = 0.1

  print("True w: %s" % true_w)
  print("True b: %s\n" % true_b)

  model = LinearModel()
  dataset = synthetic_dataset(true_w, true_b, noise_level, batch_size, 20)

  device = "gpu:0" if tfe.num_gpus() else "cpu:0"
  print("Using device: %s" % device)
  with tf.device(device):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    fit(model, dataset, optimizer, verbose=True, logdir=FLAGS.logdir)

  print("\nAfter training: w = %s" % model.variables[0].numpy())
  print("\nAfter training: b = %s" % model.variables[1].numpy())


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--logdir",
      type=str,
      default=None,
      help="logdir in which TensorBoard summaries will be written (optional).")
  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
