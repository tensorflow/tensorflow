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
r"""Train an MLP on MNIST using K-FAC.

This library fits a 3-layer, tanh-activated MLP on MNIST using K-FAC. After
~25k steps, this should reach perfect accuracy on the training set.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.kfac.examples import mnist

lc = tf.contrib.kfac.layer_collection
opt = tf.contrib.kfac.optimizer

__all__ = [
    "fc_layer",
    "train_mnist",
]


def fc_layer(layer_id, inputs, output_size):
  """Builds a fully connected layer.

  Args:
    layer_id: int. Integer ID for this layer's variables.
    inputs: Tensor of shape [num_examples, input_size]. Each row corresponds
      to a single example.
    output_size: int. Number of output dimensions after fully connected layer.

  Returns:
    preactivations: Tensor of shape [num_examples, output_size]. Values of the
      layer immediately before the activation function.
    activations: Tensor of shape [num_examples, output_size]. Values of the
      layer immediately after the activation function.
    params: Tuple of (weights, bias), parameters for this layer.
  """
  # TODO(b/67004004): Delete this function and rely on tf.layers exclusively.
  layer = tf.layers.Dense(
      output_size,
      kernel_initializer=tf.random_normal_initializer(),
      name="fc_%d" % layer_id)
  preactivations = layer(inputs)
  activations = tf.nn.tanh(preactivations)

  # layer.weights is a list. This converts it a (hashable) tuple.
  return preactivations, activations, tuple(layer.weights)


def train_mnist(data_dir, num_epochs, use_fake_data=False):
  """Train an MLP on MNIST.

  Args:
    data_dir: string. Directory to read MNIST examples from.
    num_epochs: int. Number of passes to make over the training set.
    use_fake_data: bool. If True, generate a synthetic dataset.

  Returns:
    accuracy of model on the final minibatch of training data.
  """
  # Load a dataset.
  tf.logging.info("Loading MNIST into memory.")
  examples, labels = mnist.load_mnist(
      data_dir,
      num_epochs=num_epochs,
      batch_size=64,
      flatten_images=True,
      use_fake_data=use_fake_data)

  # Build an MLP. For each layer, we'll keep track of the preactivations,
  # activations, weights, and bias.
  tf.logging.info("Building model.")
  pre0, act0, params0 = fc_layer(layer_id=0, inputs=examples, output_size=128)
  pre1, act1, params1 = fc_layer(layer_id=1, inputs=act0, output_size=64)
  pre2, act2, params2 = fc_layer(layer_id=2, inputs=act1, output_size=32)
  logits, _, params3 = fc_layer(layer_id=3, inputs=act2, output_size=10)
  loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits))
  accuracy = tf.reduce_mean(
      tf.cast(tf.equal(labels, tf.argmax(logits, axis=1)), dtype=tf.float32))

  # Register parameters. K-FAC needs to know about the inputs, outputs, and
  # parameters of each layer and the logits powering the posterior probability
  # over classes.
  tf.logging.info("Building KFAC Optimizer.")
  layer_collection = lc.LayerCollection()
  layer_collection.register_fully_connected(params0, examples, pre0)
  layer_collection.register_fully_connected(params1, act0, pre1)
  layer_collection.register_fully_connected(params2, act1, pre2)
  layer_collection.register_fully_connected(params3, act2, logits)
  layer_collection.register_categorical_predictive_distribution(logits)

  # Train with K-FAC. We'll use a decreasing learning rate that's cut in 1/2
  # every 10k iterations.
  global_step = tf.train.get_or_create_global_step()
  optimizer = opt.KfacOptimizer(
      learning_rate=tf.train.exponential_decay(
          0.00002, global_step, 10000, 0.5, staircase=True),
      cov_ema_decay=0.95,
      damping=0.0001,
      layer_collection=layer_collection,
      momentum=0.99)
  train_op = optimizer.minimize(loss, global_step=global_step)

  tf.logging.info("Starting training.")
  with tf.train.MonitoredTrainingSession() as sess:
    while not sess.should_stop():
      # K-FAC has 3 primary ops,
      # - train_op: Update the weights with the minibatch's gradient.
      # - cov_update_op: Update statistics used for building K-FAC's
      #   preconditioner matrix.
      # - inv_update_op: Update preconditioner matrix using statistics.
      #
      # The first 2 of these are cheap and should be done with each step. The
      # latter is more expensive, and should be updated ~100 iterations.
      global_step_, loss_, accuracy_, _, _ = sess.run(
          [global_step, loss, accuracy, train_op, optimizer.cov_update_op])

      if global_step_ % 100 == 0:
        sess.run(optimizer.inv_update_op)

      if global_step_ % 100 == 0:
        tf.logging.info("global_step: %d | loss: %f | accuracy: %f",
                        global_step_, loss_, accuracy_)

  return accuracy_
