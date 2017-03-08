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

"""Functional tests for Ftrl operations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class FtrlOptimizerTest(tf.test.TestCase):

  def testFtrlwithoutRegularization(self):
    for dtype in [tf.half, tf.float32]:
      with self.test_session() as sess:
        var0 = tf.Variable([0.0, 0.0], dtype=dtype)
        var1 = tf.Variable([0.0, 0.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.2], dtype=dtype)
        grads1 = tf.constant([0.01, 0.02], dtype=dtype)
        opt = tf.train.FtrlOptimizer(3.0,
                                     initial_accumulator_value=0.1,
                                     l1_regularization_strength=0.0,
                                     l2_regularization_strength=0.0)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        tf.global_variables_initializer().run()

        v0_val, v1_val = sess.run([var0, var1])
        self.assertAllClose([0.0, 0.0], v0_val)
        self.assertAllClose([0.0, 0.0], v1_val)

        # Run 3 steps FTRL
        for _ in range(3):
          update.run()

        v0_val, v1_val = sess.run([var0, var1])
        self.assertAllCloseAccordingToType(np.array([-2.60260963, -4.29698515]),
                                           v0_val)
        self.assertAllCloseAccordingToType(np.array([-0.28432083, -0.56694895]),
                                           v1_val)

  def testFtrlwithoutRegularization2(self):
    for dtype in [tf.half, tf.float32]:
      with self.test_session() as sess:
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([4.0, 3.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.2], dtype=dtype)
        grads1 = tf.constant([0.01, 0.02], dtype=dtype)

        opt = tf.train.FtrlOptimizer(3.0,
                                     initial_accumulator_value=0.1,
                                     l1_regularization_strength=0.0,
                                     l2_regularization_strength=0.0)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        tf.global_variables_initializer().run()

        v0_val, v1_val = sess.run([var0, var1])
        self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
        self.assertAllCloseAccordingToType([4.0, 3.0], v1_val)

        # Run 3 steps FTRL
        for _ in range(3):
          update.run()
        v0_val, v1_val = sess.run([var0, var1])
        self.assertAllCloseAccordingToType(np.array([-2.55607247, -3.98729396]),
                                           v0_val)
        self.assertAllCloseAccordingToType(np.array([-0.28232238, -0.56096673]),
                                           v1_val)

  def testFtrlWithL1(self):
    for dtype in [tf.half, tf.float32]:
      with self.test_session() as sess:
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([4.0, 3.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.2], dtype=dtype)
        grads1 = tf.constant([0.01, 0.02], dtype=dtype)

        opt = tf.train.FtrlOptimizer(3.0,
                                     initial_accumulator_value=0.1,
                                     l1_regularization_strength=0.001,
                                     l2_regularization_strength=0.0)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        tf.global_variables_initializer().run()

        v0_val, v1_val = sess.run([var0, var1])
        self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
        self.assertAllCloseAccordingToType([4.0, 3.0], v1_val)

        # Run 10 steps FTRL
        for _ in range(10):
          update.run()
        v0_val, v1_val = sess.run([var0, var1])
        self.assertAllCloseAccordingToType(
            np.array([-7.66718769, -10.91273689]),
            v0_val)
        self.assertAllCloseAccordingToType(
            np.array([-0.93460727, -1.86147261]),
            v1_val)

  def testFtrlWithL1_L2(self):
    for dtype in [tf.half, tf.float32]:
      with self.test_session() as sess:
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([4.0, 3.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.2], dtype=dtype)
        grads1 = tf.constant([0.01, 0.02], dtype=dtype)

        opt = tf.train.FtrlOptimizer(3.0,
                                     initial_accumulator_value=0.1,
                                     l1_regularization_strength=0.001,
                                     l2_regularization_strength=2.0)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        tf.global_variables_initializer().run()

        v0_val, v1_val = sess.run([var0, var1])
        self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
        self.assertAllCloseAccordingToType([4.0, 3.0], v1_val)

        # Run 10 steps FTRL
        for _ in range(10):
          update.run()

        v0_val, v1_val = sess.run([var0, var1])
        self.assertAllCloseAccordingToType(np.array([-0.24059935, -0.46829352]),
                                           v0_val)
        self.assertAllCloseAccordingToType(np.array([-0.02406147, -0.04830509]),
                                           v1_val)

  def applyOptimizer(self, opt, dtype, steps=5, is_sparse=False):
    if is_sparse:
      var0 = tf.Variable([[0.0], [0.0]], dtype=dtype)
      var1 = tf.Variable([[0.0], [0.0]], dtype=dtype)
      grads0 = tf.IndexedSlices(tf.constant([0.1], shape=[1, 1], dtype=dtype),
                                tf.constant([0]),
                                tf.constant([2, 1]))
      grads1 = tf.IndexedSlices(tf.constant([0.02], shape=[1, 1], dtype=dtype),
                                tf.constant([1]),
                                tf.constant([2, 1]))
    else:
      var0 = tf.Variable([0.0, 0.0], dtype=dtype)
      var1 = tf.Variable([0.0, 0.0], dtype=dtype)
      grads0 = tf.constant([0.1, 0.2], dtype=dtype)
      grads1 = tf.constant([0.01, 0.02], dtype=dtype)

    update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
    tf.global_variables_initializer().run()

    sess = tf.get_default_session()
    v0_val, v1_val = sess.run([var0, var1])
    if is_sparse:
      self.assertAllCloseAccordingToType([[0.0], [0.0]], v0_val)
      self.assertAllCloseAccordingToType([[0.0], [0.0]], v1_val)
    else:
      self.assertAllCloseAccordingToType([0.0, 0.0], v0_val)
      self.assertAllCloseAccordingToType([0.0, 0.0], v1_val)

    # Run Ftrl for a few steps
    for _ in range(steps):
      update.run()

    v0_val, v1_val = sess.run([var0, var1])
    return v0_val, v1_val

  # When variables are initialized with Zero, FTRL-Proximal has two properties:
  # 1. Without L1&L2 but with fixed learning rate, FTRL-Proximal is identical
  # with GradientDescent.
  # 2. Without L1&L2 but with adaptive learning rate, FTRL-Proximal is identical
  # with Adagrad.
  # So, basing on these two properties, we test if our implementation of
  # FTRL-Proximal performs same updates as Adagrad or GradientDescent.
  def testEquivAdagradwithoutRegularization(self):
    for dtype in [tf.half, tf.float32]:
      with self.test_session():
        val0, val1 = self.applyOptimizer(
            tf.train.FtrlOptimizer(3.0,
                                   # Adagrad learning rate
                                   learning_rate_power=-0.5,
                                   initial_accumulator_value=0.1,
                                   l1_regularization_strength=0.0,
                                   l2_regularization_strength=0.0),
            dtype)

      with self.test_session():
        val2, val3 = self.applyOptimizer(
            tf.train.AdagradOptimizer(3.0, initial_accumulator_value=0.1),
            dtype)

      self.assertAllCloseAccordingToType(val0, val2)
      self.assertAllCloseAccordingToType(val1, val3)

  def testEquivSparseAdagradwithoutRegularization(self):
    for dtype in [tf.half, tf.float32]:
      with self.test_session():
        val0, val1 = self.applyOptimizer(
            tf.train.FtrlOptimizer(3.0,
                                   # Adagrad learning rate
                                   learning_rate_power=-0.5,
                                   initial_accumulator_value=0.1,
                                   l1_regularization_strength=0.0,
                                   l2_regularization_strength=0.0),
            dtype,
            is_sparse=True)

      with self.test_session():
        val2, val3 = self.applyOptimizer(
            tf.train.AdagradOptimizer(3.0, initial_accumulator_value=0.1),
            dtype, is_sparse=True)

      self.assertAllCloseAccordingToType(val0, val2)
      self.assertAllCloseAccordingToType(val1, val3)

  def testEquivSparseGradientDescentwithoutRegularization(self):
    for dtype in [tf.half, tf.float32]:
      with self.test_session():
        val0, val1 = self.applyOptimizer(
            tf.train.FtrlOptimizer(3.0,
                                   # Fixed learning rate
                                   learning_rate_power=-0.0,
                                   initial_accumulator_value=0.1,
                                   l1_regularization_strength=0.0,
                                   l2_regularization_strength=0.0),
            dtype,
            is_sparse=True)

      with self.test_session():
        val2, val3 = self.applyOptimizer(
            tf.train.GradientDescentOptimizer(3.0), dtype, is_sparse=True)

      self.assertAllCloseAccordingToType(val0, val2)
      self.assertAllCloseAccordingToType(val1, val3)

  def testEquivGradientDescentwithoutRegularization(self):
    for dtype in [tf.half, tf.float32]:
      with self.test_session():
        val0, val1 = self.applyOptimizer(
            tf.train.FtrlOptimizer(3.0,
                                   # Fixed learning rate
                                   learning_rate_power=-0.0,
                                   initial_accumulator_value=0.1,
                                   l1_regularization_strength=0.0,
                                   l2_regularization_strength=0.0),
            dtype)

      with self.test_session():
        val2, val3 = self.applyOptimizer(
            tf.train.GradientDescentOptimizer(3.0), dtype)

      self.assertAllCloseAccordingToType(val0, val2)
      self.assertAllCloseAccordingToType(val1, val3)


if __name__ == "__main__":
  tf.test.main()
