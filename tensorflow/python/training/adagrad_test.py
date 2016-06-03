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

"""Functional tests for aggregate operations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class AdagradOptimizerTest(tf.test.TestCase):

  def doTestBasic(self, use_locking=False):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        ada_opt = tf.train.AdagradOptimizer(3.0,
                                            initial_accumulator_value=0.1,
                                            use_locking=use_locking)
        ada_update = ada_opt.apply_gradients(zip(
            [grads0, grads1], [var0, var1]))
        tf.initialize_all_variables().run()
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())
        # Run 3 steps of adagrad
        for _ in range(3):
          ada_update.run()
        # Validate updated params
        self.assertAllCloseAccordingToType(
            np.array([-1.6026098728179932, -0.6026098728179932]), var0.eval())
        self.assertAllCloseAccordingToType(
            np.array([2.715679168701172, 3.715679168701172]), var1.eval())

  def testBasic(self):
    self.doTestBasic(use_locking=False)

  def testBasicLocked(self):
    self.doTestBasic(use_locking=True)

  def testTensorLearningRate(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        ada_opt = tf.train.AdagradOptimizer(
            tf.constant(3.0),
            initial_accumulator_value=0.1)
        ada_update = ada_opt.apply_gradients(zip(
            [grads0, grads1], [var0, var1]))
        tf.initialize_all_variables().run()
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())
        # Run 3 steps of adagrad
        for _ in range(3):
          ada_update.run()
        # Validate updated params
        self.assertAllCloseAccordingToType(
            np.array([-1.6026098728179932, -0.6026098728179932]), var0.eval())
        self.assertAllCloseAccordingToType(
            np.array([2.715679168701172, 3.715679168701172]), var1.eval())

  def testSparseBasic(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([[1.0], [2.0]], dtype=dtype)
        var1 = tf.Variable([[3.0], [4.0]], dtype=dtype)
        grads0 = tf.IndexedSlices(
            tf.constant([0.1], shape=[1, 1], dtype=dtype),
            tf.constant([0]),
            tf.constant([2, 1]))
        grads1 = tf.IndexedSlices(
            tf.constant([0.01], shape=[1, 1], dtype=dtype),
            tf.constant([1]),
            tf.constant([2, 1]))
        ada_opt = tf.train.AdagradOptimizer(3.0, initial_accumulator_value=0.1)
        ada_update = ada_opt.apply_gradients(zip(
            [grads0, grads1], [var0, var1]))
        tf.initialize_all_variables().run()
        # Fetch params to validate initial values
        self.assertAllClose([[1.0], [2.0]], var0.eval())
        self.assertAllClose([[3.0], [4.0]], var1.eval())
        # Run 3 step of sgd
        for _ in range(3):
          ada_update.run()
        # Validate updated params
        self.assertAllCloseAccordingToType(
            np.array([[-1.6026098728179932], [2.0]]), var0.eval())
        self.assertAllCloseAccordingToType(
            np.array([[3.0], [3.715679168701172]]), var1.eval())

  def testSparseStability(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        shape = [1, 6]
        var0 = tf.Variable(
            [[0.00872496, -0.106952, 0.110467, 0.226505, -0.0147257,
              -0.0105945]],
            dtype=dtype)
        grads0 = tf.IndexedSlices(
            tf.constant(
                [[-5.91278e-05, 5.31673e-05, -2.5779e-06, 4.29153e-05,
                  -8.4877e-05, -9.48906e-05]],
                shape=shape,
                dtype=dtype),
            tf.constant([0]),
            tf.constant(shape))
        ada_opt = tf.train.AdagradOptimizer(1.0, initial_accumulator_value=0.1)
        ada_update = ada_opt.apply_gradients(zip([grads0], [var0]))
        self.assertEqual(["accumulator"], ada_opt.get_slot_names())
        slot0 = ada_opt.get_slot(var0, "accumulator")
        init = tf.initialize_all_variables()
        for _ in range(100):
          init.run()
          ada_update.run()
          self.assertAllCloseAccordingToType(
              np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]), slot0.eval())
          self.assertAllCloseAccordingToType(
              np.array([[0.00891194, -0.10712013, 0.11047515, 0.22636929, -
                         0.0144573, -0.01029443]]), var0.eval())

  def testSharing(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        ada_opt = tf.train.AdagradOptimizer(3.0)
        # Apply the optimizer twice.  Both applications will use
        # the same accums.
        ada_update1 = ada_opt.apply_gradients(zip(
            [grads0, grads1], [var0, var1]))
        ada_update2 = ada_opt.apply_gradients(zip(
            [grads0, grads1], [var0, var1]))
        self.assertEqual(["accumulator"], ada_opt.get_slot_names())
        slot0 = ada_opt.get_slot(var0, "accumulator")
        self.assertEquals(slot0.get_shape(), var0.get_shape())
        slot1 = ada_opt.get_slot(var1, "accumulator")
        self.assertEquals(slot1.get_shape(), var1.get_shape())
        tf.initialize_all_variables().run()

        # Fetch params to validate initial values.
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())
        # Mix the first and the second adagrad for 3 steps.
        ada_update1.run()
        ada_update2.run()
        ada_update1.run()
        # Validate updated params (the same as with only 1 Adagrad).
        self.assertAllCloseAccordingToType(
            np.array([-1.6026098728179932, -0.6026098728179932]), var0.eval())
        self.assertAllCloseAccordingToType(
            np.array([2.715679168701172, 3.715679168701172]), var1.eval())


if __name__ == "__main__":
  tf.test.main()
