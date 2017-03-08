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
"""Tests for Adadelta Optimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

class AdadeltaOptimizerTest(tf.test.TestCase):
  def testBasic(self):
    num_updates = 4 # number of ADADELTA steps to perform
    for dtype in [tf.half, tf.float32]:
      for grad in [0.2, 0.1, 0.01]:
        for lr in [1.0, 0.5, 0.1]:
          with self.test_session():
            var0_init = [1.0, 2.0]
            var1_init = [3.0, 4.0]
            var0 = tf.Variable(var0_init, dtype=dtype)
            var1 = tf.Variable(var1_init, dtype=dtype)

            grads = tf.constant([grad, grad], dtype=dtype)

            accum = 0.0
            accum_update = 0.0

            # ADADELTA gradient optimizer
            rho = 0.95
            epsilon = 1e-8
            adadelta_opt = tf.train.AdadeltaOptimizer(lr, rho, epsilon)
            adadelta_update = adadelta_opt.apply_gradients(zip(
              [grads, grads], [var0, var1]))

            tf.initialize_all_variables().run()

            # Assign slots
            slot = [None] * 2
            slot_update = [None] * 2
            self.assertEqual(["accum", "accum_update"],
              adadelta_opt.get_slot_names())
            slot[0] = adadelta_opt.get_slot(var0, "accum")
            self.assertEquals(slot[0].get_shape(), var0.get_shape())
            self.assertFalse(slot[0] in tf.trainable_variables())

            slot_update[0] = adadelta_opt.get_slot(var0, "accum_update")
            self.assertEquals(slot_update[0].get_shape(), var0.get_shape())
            self.assertFalse(slot_update[0] in tf.trainable_variables())

            slot[1] = adadelta_opt.get_slot(var1, "accum")
            self.assertEquals(slot[1].get_shape(), var1.get_shape())
            self.assertFalse(slot[1] in tf.trainable_variables())

            slot_update[1] = adadelta_opt.get_slot(var1, "accum_update")
            self.assertEquals(slot_update[1].get_shape(), var1.get_shape())
            self.assertFalse(slot_update[1] in tf.trainable_variables())

            # Fetch params to validate initial values
            self.assertAllClose(var0_init, var0.eval())
            self.assertAllClose(var1_init, var1.eval())

            update = [None] * num_updates
            tot_update = 0
            for step in range(num_updates):
              # Run adadelta update for comparison
              adadelta_update.run()

              # Perform initial update without previous accum values
              accum = accum * rho + (grad**2) * (1 - rho)
              update[step] = (np.sqrt(accum_update + epsilon) *
                (1. / np.sqrt(accum + epsilon)) * grad)
              accum_update = (accum_update * rho + (update[step]**2) *
                (1.0 - rho))
              tot_update += update[step] * lr

              # Check that the accumulators have been updated
              for slot_idx in range(2):
                self.assertAllCloseAccordingToType(
                  np.array([accum, accum], dtype=dtype.as_numpy_dtype()),
                  slot[slot_idx].eval())

                self.assertAllCloseAccordingToType(
                  np.array([accum_update, accum_update],
                  dtype=dtype.as_numpy_dtype()),
                  slot_update[slot_idx].eval())

              # Check that the parameters have been updated
              self.assertAllCloseAccordingToType(
                np.array([var0_init[0] - tot_update,
                var0_init[1] - tot_update], dtype=dtype.as_numpy_dtype()),
                var0.eval(), rtol=1e-3)

              self.assertAllCloseAccordingToType(
                np.array([var1_init[0] - tot_update,
                var1_init[1] - tot_update], dtype=dtype.as_numpy_dtype()),
                var1.eval(), rtol=1e-3)

if __name__ == "__main__":
  tf.test.main()
