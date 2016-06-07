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

"""Functional tests for Proximal Adagrad operations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class ProximalAdagradOptimizerTest(tf.test.TestCase):

  def testProximalAdagradwithoutRegularization(self):
    with self.test_session() as sess:
      var0 = tf.Variable([0.0, 0.0])
      var1 = tf.Variable([0.0, 0.0])
      grads0 = tf.constant([0.1, 0.2])
      grads1 = tf.constant([0.01, 0.02])
      opt = tf.train.ProximalAdagradOptimizer(3.0,
                                              initial_accumulator_value=0.1,
                                              l1_regularization_strength=0.0,
                                              l2_regularization_strength=0.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      tf.initialize_all_variables().run()

      v0_val, v1_val = sess.run([var0, var1])
      self.assertAllClose([0.0, 0.0], v0_val)
      self.assertAllClose([0.0, 0.0], v1_val)

      # Run 3 steps Proximal Adagrad.
      for _ in range(3):
        update.run()

      v0_val, v1_val = sess.run([var0, var1])
      self.assertAllClose(np.array([-2.60260963, -4.29698515]),
                          v0_val)
      self.assertAllClose(np.array([-0.28432083, -0.56694895]),
                          v1_val)

  def testProximalAdagradwithoutRegularization2(self):
    with self.test_session() as sess:
      var0 = tf.Variable([1.0, 2.0])
      var1 = tf.Variable([4.0, 3.0])
      grads0 = tf.constant([0.1, 0.2])
      grads1 = tf.constant([0.01, 0.02])

      opt = tf.train.ProximalAdagradOptimizer(3.0,
                                              initial_accumulator_value=0.1,
                                              l1_regularization_strength=0.0,
                                              l2_regularization_strength=0.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      tf.initialize_all_variables().run()

      v0_val, v1_val = sess.run([var0, var1])
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([4.0, 3.0], v1_val)

      # Run 3 steps Proximal Adagrad.
      for _ in range(3):
        update.run()
      v0_val, v1_val = sess.run([var0, var1])
      self.assertAllClose(np.array([-1.60261, -2.296985]),
                          v0_val)
      self.assertAllClose(np.array([3.715679, 2.433051]),
                          v1_val)

  def testProximalAdagradWithL1(self):
    with self.test_session() as sess:
      var0 = tf.Variable([1.0, 2.0])
      var1 = tf.Variable([4.0, 3.0])
      grads0 = tf.constant([0.1, 0.2])
      grads1 = tf.constant([0.01, 0.02])

      opt = tf.train.ProximalAdagradOptimizer(3.0,
                                              initial_accumulator_value=0.1,
                                              l1_regularization_strength=0.001,
                                              l2_regularization_strength=0.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      tf.initialize_all_variables().run()

      v0_val, v1_val = sess.run([var0, var1])
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([4.0, 3.0], v1_val)

      # Run 10 steps Proximal Adagrad
      for _ in range(10):
        update.run()
      v0_val, v1_val = sess.run([var0, var1])
      self.assertAllClose(np.array([0.662907, 0.767398]),
                          v0_val)
      self.assertAllClose(np.array([2.959304, 1.029232]),
                          v1_val)

  def testProximalAdagradWithL1_L2(self):
    with self.test_session() as sess:
      var0 = tf.Variable([1.0, 2.0])
      var1 = tf.Variable([4.0, 3.0])
      grads0 = tf.constant([0.1, 0.2])
      grads1 = tf.constant([0.01, 0.02])

      opt = tf.train.ProximalAdagradOptimizer(3.0,
                                              initial_accumulator_value=0.1,
                                              l1_regularization_strength=0.001,
                                              l2_regularization_strength=2.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      tf.initialize_all_variables().run()

      v0_val, v1_val = sess.run([var0, var1])
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([4.0, 3.0], v1_val)

      # Run 10 steps Proximal Adagrad.
      for _ in range(10):
        update.run()

      v0_val, v1_val = sess.run([var0, var1])
      self.assertAllClose(np.array([0.043069, 0.080461]),
                          v0_val)
      self.assertAllClose(np.array([0.004069, 0.008578]),
                          v1_val)

  def applyOptimizer(self, opt, steps=5, is_sparse=False):
    if is_sparse:
      var0 = tf.Variable([[1.0], [2.0]])
      var1 = tf.Variable([[3.0], [4.0]])
      grads0 = tf.IndexedSlices(tf.constant([0.1], shape=[1, 1]),
                                tf.constant([0]),
                                tf.constant([2, 1]))
      grads1 = tf.IndexedSlices(tf.constant([0.02], shape=[1, 1]),
                                tf.constant([1]),
                                tf.constant([2, 1]))
    else:
      var0 = tf.Variable([1.0, 2.0])
      var1 = tf.Variable([3.0, 4.0])
      grads0 = tf.constant([0.1, 0.2])
      grads1 = tf.constant([0.01, 0.02])

    update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
    tf.initialize_all_variables().run()

    sess = tf.get_default_session()
    v0_val, v1_val = sess.run([var0, var1])
    if is_sparse:
      self.assertAllClose([[1.0], [2.0]], v0_val)
      self.assertAllClose([[3.0], [4.0]], v1_val)
    else:
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([3.0, 4.0], v1_val)

    # Run ProximalAdagrad for a few steps
    for _ in range(steps):
      update.run()

    v0_val, v1_val = sess.run([var0, var1])
    return v0_val, v1_val

  def testEquivAdagradwithoutRegularization(self):
    with self.test_session():
      val0, val1 = self.applyOptimizer(
          tf.train.ProximalAdagradOptimizer(3.0,
                                            initial_accumulator_value=0.1,
                                            l1_regularization_strength=0.0,
                                            l2_regularization_strength=0.0))

    with self.test_session():
      val2, val3 = self.applyOptimizer(
          tf.train.AdagradOptimizer(3.0, initial_accumulator_value=0.1))

    self.assertAllClose(val0, val2)
    self.assertAllClose(val1, val3)

  def testEquivSparseAdagradwithoutRegularization(self):
    with self.test_session():
      val0, val1 = self.applyOptimizer(
          tf.train.ProximalAdagradOptimizer(3.0,
                                            initial_accumulator_value=0.1,
                                            l1_regularization_strength=0.0,
                                            l2_regularization_strength=0.0),
          is_sparse=True)

    with self.test_session():
      val2, val3 = self.applyOptimizer(
          tf.train.AdagradOptimizer(3.0, initial_accumulator_value=0.1),
          is_sparse=True)

    self.assertAllClose(val0, val2)
    self.assertAllClose(val1, val3)


if __name__ == "__main__":
  tf.test.main()
