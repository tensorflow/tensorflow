# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for Relu and ReluGrad."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class ReluTest(tf.test.TestCase):

  def _npRelu(self, np_features):
    return np.maximum(np_features, np.zeros(np_features.shape))

  def testNpRelu(self):
    self.assertAllClose(
        np.array([[0.0, 0.7, 0.0, 0.3, 0.0],
                  [0.1, 0.0, 0.5, 0.0, 0.9]]),
        self._npRelu(np.array([[-0.9, 0.7, -0.5, 0.3, -0.1],
                               [0.1, -0.3, 0.5, -0.7, 0.9]])))

  def _testRelu(self, np_features, use_gpu=False):
    np_relu = self._npRelu(np_features)
    with self.test_session(use_gpu=use_gpu):
      relu = tf.nn.relu(np_features)
      tf_relu = relu.eval()
    self.assertAllClose(np_relu, tf_relu)
    self.assertShapeEqual(np_relu, relu)

  def testNumbers(self):
    for t in [np.int32, np.int64, np.float32, np.float64]:
      self._testRelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          use_gpu=False)
      if t in [np.float32, np.float64]:
        self._testRelu(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
            use_gpu=True)

  # The gradient test for ReLU is a bit tricky as the derivative is not well
  # defined at around zero and we want to avoid that in terms of input values.
  def testGradientFloat32(self):
    with self.test_session():
      x = tf.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5], name="x")
      y = tf.nn.relu(x, name="relu")
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32, order="F")
      err = tf.test.compute_gradient_error(x,
                                           [2, 5],
                                           y,
                                           [2, 5],
                                           x_init_value=x_init)
    print("relu (float32) gradient err = ", err)
    self.assertLess(err, 1e-4)

  def testGradientFloat64(self):
    with self.test_session():
      x = tf.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5], dtype=tf.float64, name="x")
      y = tf.nn.relu(x, name="relu")
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64, order="F")
      err = tf.test.compute_gradient_error(x,
                                           [2, 5],
                                           y,
                                           [2, 5],
                                           x_init_value=x_init)
    print("relu (float64) gradient err = ", err)
    self.assertLess(err, 1e-10)

  def testGradGradFloat32(self):
    with self.test_session():
      x = tf.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5], name="x")
      y = tf.nn.relu(x, name="relu")
      z = tf.gradients(y, x)
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32, order="F")
      err = tf.test.compute_gradient_error(x,
                                           [2, 5],
                                           z[0],
                                           [2, 5],
                                           x_init_value=x_init)
    print("relu (float32) gradient of gradient err = ", err)
    self.assertLess(err, 1e-4)

  def testGradGradFloat64(self):
    with self.test_session():
      x = tf.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5], dtype=tf.float64, name="x")
      y = tf.nn.relu(x, name="relu")
      z = tf.gradients(y, x)
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64, order="F")
      err = tf.test.compute_gradient_error(x,
                                           [2, 5],
                                           z[0],
                                           [2, 5],
                                           x_init_value=x_init)
    print("relu (float64) gradient of gradient err = ", err)
    self.assertLess(err, 1e-10)

  def testGradientScalar(self):
    with self.test_session() as sess:
      x = tf.Variable(100.)
      y = tf.nn.relu(x)
      loss = y**2
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.25)
      train_op = optimizer.minimize(loss)
      sess.run(tf.initialize_all_variables())
      sess.run(train_op)
      self.assertAllClose(x.eval(), 50.0)


class Relu6Test(tf.test.TestCase):

  def _npRelu6(self, np_features):
    sixes = np.copy(np_features)
    sixes.fill(6.0)
    return np.minimum(np.maximum(np_features, np.zeros(np_features.shape)),
                      sixes)

  def testNpRelu6(self):
    self.assertAllClose(
        np.array([[0.0, 0.7, 0.0, 0.3, 6.0],
                  [0.1, 0.0, 6.0, 0.0, 0.9]]),
        self._npRelu6(np.array([[-0.9, 0.7, -0.5, 0.3, 6.0],
                                [0.1, -0.3, 6.5, -0.7, 0.9]])))

  def _testRelu6(self, np_features, use_gpu=False):
    np_relu6 = self._npRelu6(np_features)
    with self.test_session(use_gpu=use_gpu):
      relu6 = tf.nn.relu6(np_features)
      tf_relu6 = relu6.eval()
    self.assertAllClose(np_relu6, tf_relu6)
    self.assertShapeEqual(np_relu6, relu6)

  def testNumbers(self):
    for t in [np.int32, np.int64, np.float32, np.float64]:
      self._testRelu6(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          use_gpu=False)
      if t in [np.float, np.double]:
        self._testRelu6(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
            use_gpu=True)

  # The gradient test for ReLU6 is a bit tricky as the derivative is
  # not well defined at around zero and six and we want to avoid that
  # in terms of input values.
  def testGradientFloat32(self):
    with self.test_session():
      x = tf.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 6.1, 6.3, 6.5, 6.7, 6.9],
          shape=[2, 5], name="x")
      y = tf.nn.relu6(x, name="relu6")
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [6.1, 6.3, 6.5, 6.7, 6.9]],
          dtype=np.float32, order="F")
      err = tf.test.compute_gradient_error(x,
                                           [2, 5],
                                           y,
                                           [2, 5],
                                           x_init_value=x_init)
    print("relu6 (float32) gradient err = ", err)
    self.assertLess(err, 1e-4)

  def testGradientFloat64(self):
    with self.test_session():
      x = tf.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 6.1, 6.3, 6.5, 6.7, 6.9],
          shape=[2, 5], dtype=tf.float64, name="x")
      y = tf.nn.relu6(x, name="relu6")
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [6.1, 6.3, 6.5, 6.7, 6.9]],
          dtype=np.float64, order="F")
      err = tf.test.compute_gradient_error(x,
                                           [2, 5],
                                           y,
                                           [2, 5],
                                           x_init_value=x_init)
    print("relu6 (float64) gradient err = ", err)
    self.assertLess(err, 1e-10)


class EluTest(tf.test.TestCase):

  def _npElu(self, np_features):
    return np.where(np_features < 0, np.exp(np_features) - 1, np_features)

  def testNpElu(self):
    self.assertAllClose(
        np.array([[-0.59343034025, 0.7, -0.39346934028, 0.3, -0.09516258196],
                  [0.1, -0.25918177931, 0.5, -0.5034146962, 0.9]]),
        self._npElu(np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -
                                                             0.7, 0.9]])))

  def _testElu(self, np_features, use_gpu=False):
    np_elu = self._npElu(np_features)
    with self.test_session(use_gpu=use_gpu):
      elu = tf.nn.elu(np_features)
      tf_elu = elu.eval()
    self.assertAllClose(np_elu, tf_elu)
    self.assertShapeEqual(np_elu, elu)

  def testNumbers(self):
    for t in [np.float32, np.float64]:
      self._testElu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          use_gpu=False)
      self._testElu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          use_gpu=True)

  def testGradientFloat32(self):
    with self.test_session():
      x_val = [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
      x = tf.constant(x_val, name="x")
      y = tf.nn.elu(x, name="elu")
      x_init = np.asarray(x_val, dtype=np.float32, order="F")
      err = tf.test.compute_gradient_error(x,
                                           [2, 5],
                                           y,
                                           [2, 5],
                                           x_init_value=x_init)
    print("elu (float32) gradient err = ", err)
    self.assertLess(err, 1e-4)

  def testGradientFloat64(self):
    with self.test_session():
      x_val = [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
      x = tf.constant(x_val, dtype=tf.float64, name="x")
      y = tf.nn.elu(x, name="elu")
      x_init = np.asarray(x_val, dtype=np.float64, order="F")
      err = tf.test.compute_gradient_error(x,
                                           [2, 5],
                                           y,
                                           [2, 5],
                                           x_init_value=x_init)
    print("elu (float64) gradient err = ", err)
    self.assertLess(err, 1e-6)


if __name__ == "__main__":
  tf.test.main()
