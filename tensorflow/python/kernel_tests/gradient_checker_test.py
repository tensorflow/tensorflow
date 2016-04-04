# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Tests for tensorflow.kernels.gradient_checker."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class GradientCheckerTest(tf.test.TestCase):

  def testAddSimple(self):
    np.random.seed(1)  # Fix seed to avoid flakiness
    with self.test_session(use_gpu=False):
      # a test case for Add operation
      size = (2, 3)
      x1 = tf.constant(2.0, shape=size, name="x1")
      x2 = tf.constant(3.0, shape=size, name="x2")
      y = tf.add(x1, x2, name="y")

      # checking gradients for x1
      error = tf.test.compute_gradient_error(x1, size, y, size)
    tf.logging.info("x1 error = %f", error)
    assert error < 1e-4

  def testAddSimpleGPU(self):
    np.random.seed(2)  # Fix seed to avoid flakiness
    with self.test_session(use_gpu=True):
      # a test case for Add operation
      size = (2, 3)
      x1 = tf.constant(2.0, shape=size, name="x1")
      x2 = tf.constant(3.0, shape=size, name="x2")
      y = tf.add(x1, x2, name="y")

      # checking gradients for x1
      error = tf.test.compute_gradient_error(x1, size, y, size)
    tf.logging.info("x1 error = %f", error)
    assert error < 1e-4

  def testAddCustomized(self):
    np.random.seed(3)  # Fix seed to avoid flakiness
    with self.test_session():
      # a test case for Add operation
      size = (2, 3)
      x1 = tf.constant(2.0, shape=size, dtype=tf.float64,
                                name="x1")
      x2 = tf.constant(3.0, shape=size, dtype=tf.float64,
                                name="x2")
      y = tf.add(x1, x2, name="y")

      # checkint gradients for x2 using a special init_value and delta
      x_init_value = np.asarray(np.arange(6, dtype=np.float64).reshape(2, 3))
      error = tf.test.compute_gradient_error(x2,
                                             size,
                                             y,
                                             size,
                                             x_init_value=x_init_value,
                                             delta=1e-2)
    tf.logging.info("x2 error = %f", error)
    assert error < 1e-10

  def testGather(self):
    np.random.seed(4)  # Fix seed to avoid flakiness
    with self.test_session():
      p_shape = (4, 2)
      p_size = 8
      index_values = [1, 3]
      y_shape = [2, 2]
      params = tf.constant(np.arange(p_size).astype(np.float),
                                    shape=p_shape, name="p")
      indices = tf.constant(index_values, name="i")
      y = tf.gather(params, indices, name="y")

      error = tf.test.compute_gradient_error(params, p_shape, y, y_shape)
    tf.logging.info("gather error = %f", error)
    assert error < 1e-4

  def testNestedGather(self):
    np.random.seed(5)  # Fix seed to avoid flakiness
    with self.test_session():
      p_shape = (8, 2)
      p_size = 16
      index_values = [1, 3, 5, 6]
      index_values2 = [0, 2]
      y2_shape = [2, 2]

      params = tf.constant(np.arange(p_size).astype(np.float),
                                    shape=p_shape, name="p")
      indices = tf.constant(index_values, name="i")
      y = tf.gather(params, indices, name="y")
      indices2 = tf.constant(index_values2, name="i2")
      y2 = tf.gather(y, indices2, name="y2")

      error = tf.test.compute_gradient_error(params, p_shape, y2, y2_shape)
    tf.logging.info("nested gather error = %f", error)
    assert error < 1e-4

  def testComplexMul(self):
    with self.test_session():
      size = ()
      c = tf.constant(5 + 7j, dtype=tf.complex64)
      x = tf.constant(11 - 13j, dtype=tf.complex64)
      y = c * x
      analytical, numerical = tf.test.compute_gradient(x, size, y, size)
      correct = np.array([[5, 7], [-7, 5]])
      self.assertAllEqual(correct, analytical)
      self.assertAllClose(correct, numerical, rtol=1e-4)
      self.assertLess(tf.test.compute_gradient_error(x, size, y, size), 2e-4)

  def testComplexConj(self):
    with self.test_session():
      size = ()
      x = tf.constant(11 - 13j, dtype=tf.complex64)
      y = tf.conj(x)
      analytical, numerical = tf.test.compute_gradient(x, size, y, size)
      correct = np.array([[1, 0], [0, -1]])
      self.assertAllEqual(correct, analytical)
      self.assertAllClose(correct, numerical, rtol=3e-6)
      self.assertLess(tf.test.compute_gradient_error(x, size, y, size), 2e-5)


# Gradient checker for MNIST.
def BuildAndTestMiniMNIST(param_index, tag):
  # Fix seed to avoid occasional flakiness
  np.random.seed(6)

  # Hyperparameters
  batch = 3
  inputs = 16
  features = 32
  classes = 10

  # Define the parameters
  inp_data = np.random.random_sample(inputs * batch)
  hidden_weight_data = np.random.randn(inputs * features) / np.sqrt(inputs)
  hidden_bias_data = np.random.random_sample(features)
  sm_weight_data = np.random.randn(features * classes) / np.sqrt(features)
  sm_bias_data = np.random.random_sample(classes)

  # special care for labels since they need to be normalized per batch
  label_data = np.random.random(batch * classes).reshape((batch, classes))
  s = label_data.sum(axis=1)
  label_data /= s[:, None]

  with tf.Session():
    # We treat the inputs as "parameters" here
    inp = tf.constant(inp_data.tolist(), shape=[batch, inputs],
                               dtype=tf.float64, name="inp")
    hidden_weight = tf.constant(hidden_weight_data.tolist(),
                                         shape=[inputs, features],
                                         dtype=tf.float64,
                                         name="hidden_weight")
    hidden_bias = tf.constant(hidden_bias_data.tolist(),
                                       shape=[features],
                                       dtype=tf.float64,
                                       name="hidden_bias")
    softmax_weight = tf.constant(sm_weight_data.tolist(),
                                          shape=[features, classes],
                                          dtype=tf.float64,
                                          name="softmax_weight")
    softmax_bias = tf.constant(sm_bias_data.tolist(), shape=[classes],
                                        dtype=tf.float64,
                                        name="softmax_bias")

    # List all the parameter so that we can test them one at a time
    all_params = [inp, hidden_weight, hidden_bias, softmax_weight, softmax_bias]
    param_sizes = [[batch, inputs],  # inp
                   [inputs, features],  # hidden_weight,
                   [features],  # hidden_bias
                   [features, classes],  # softmax_weight,
                   [classes]]  # softmax_bias

    # Now, Building MNIST
    features = tf.nn.relu(tf.nn.xw_plus_b(inp, hidden_weight, hidden_bias),
                          name="features")
    logits = tf.nn.xw_plus_b(features, softmax_weight, softmax_bias,
                             name="logits")
    labels = tf.constant(label_data.tolist(),
                         shape=[batch, classes],
                         dtype=tf.float64,
                         name="labels")
    cost = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name="cost")

    # Test the gradients.
    err = tf.test.compute_gradient_error(all_params[param_index],
                                         param_sizes[param_index],
                                         cost,
                                         [batch],
                                         delta=1e-5)

  tf.logging.info("Mini MNIST: %s gradient error = %g", tag, err)
  return err


class MiniMNISTTest(tf.test.TestCase):

  def testInputGradient(self):
    self.assertLess(BuildAndTestMiniMNIST(0, "input"), 1e-8)

  def testHiddenWeightGradient(self):
    self.assertLess(BuildAndTestMiniMNIST(1, "hidden_weight"), 1e-8)

  def testHiddenBiasGradient(self):
    self.assertLess(BuildAndTestMiniMNIST(2, "hidden_bias"), 1e-8)

  def testSoftmaxWeightGradient(self):
    self.assertLess(BuildAndTestMiniMNIST(3, "softmax_weight"), 1e-8)

  def testSoftmaxBiasGradient(self):
    self.assertLess(BuildAndTestMiniMNIST(4, "softmax_bias"), 1e-8)


if __name__ == "__main__":
  tf.test.main()
