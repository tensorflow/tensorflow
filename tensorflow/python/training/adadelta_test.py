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
"""Tests for Adadelta Optimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


class AdadeltaOptimizerTest(tf.test.TestCase):

  def testBasic(self):
    for dtype in [tf.half, tf.float32]:
      with self.test_session():
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        lr = 1.0
        rho = 0.95
        epsilon = 1e-8

        adadelta_opt = tf.train.AdadeltaOptimizer(lr, rho=rho, epsilon=epsilon)
        adadelta_update = adadelta_opt.apply_gradients(zip(
            [grads0, grads1], [var0, var1]))
        tf.initialize_all_variables().run()

        # Check we have slots
        self.assertEqual(["accum", "accum_update"],
                         adadelta_opt.get_slot_names())
        slot0 = adadelta_opt.get_slot(var0, "accum")
        self.assertEquals(slot0.get_shape(), var0.get_shape())
        self.assertFalse(slot0 in tf.trainable_variables())

        slot0_update = adadelta_opt.get_slot(var0, "accum_update")
        self.assertEquals(slot0_update.get_shape(), var0.get_shape())
        self.assertFalse(slot0_update in tf.trainable_variables())

        slot1 = adadelta_opt.get_slot(var1, "accum")
        self.assertEquals(slot1.get_shape(), var1.get_shape())
        self.assertFalse(slot1 in tf.trainable_variables())

        slot1_update = adadelta_opt.get_slot(var1, "accum_update")
        self.assertEquals(slot1_update.get_shape(), var1.get_shape())
        self.assertFalse(slot1_update in tf.trainable_variables())

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())

        adadelta_update.run()

        # Check that the accumulators have been updated.
        grad = 0.1
        accum = 0
        accum_update = 0

        accum = accum * rho + (grad**2) * (1 - rho)
        update1 = np.sqrt(accum_update + epsilon) * (
            1. / np.sqrt(accum + epsilon)) * grad
        accum_update = accum_update * rho + (update1**2) * (1.0 - rho)

        self.assertAllCloseAccordingToType(
            np.array([accum, accum]), slot0.eval())
        self.assertAllCloseAccordingToType(
            np.array([accum_update, accum_update]), slot0_update.eval())

        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(
            np.array([1.0 - update1 * lr, 2.0 - update1 * lr]),
            var0.eval(),
            rtol=1e-3)

        self.assertAllCloseAccordingToType(
            np.array([3.0 - update1 * lr, 4.0 - update1 * lr]),
            var1.eval(),
            rtol=1e-3)

        # Step 2: the momentum accumulators contain the previous update.
        accum = accum * rho + (grad**2) * (1 - rho)
        update2 = ((accum_update + epsilon)**0.5 *
                   (1. / (accum + epsilon)**0.5) * grad)
        accum_update = accum_update * rho + (update2**2) * (1.0 - rho)

        adadelta_update.run()

        # Check that the momentum accumulators have been updated.
        self.assertAllCloseAccordingToType(
            np.array([accum, accum]), slot0.eval())
        self.assertAllCloseAccordingToType(
            np.array([accum_update, accum_update]), slot0_update.eval())

        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(
            np.array([1.0 - update1 - update2, 2.0 - update1 - update2]),
            var0.eval(),
            rtol=1e-3)

        self.assertAllCloseAccordingToType(
            np.array([3.0 - update1 - update2, 4.0 - update1 - update2]),
            var1.eval(),
            rtol=1e-3)


if __name__ == "__main__":
  tf.test.main()
