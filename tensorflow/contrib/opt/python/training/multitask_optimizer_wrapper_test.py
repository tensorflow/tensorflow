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
"""Tests for MultitaskOptimizerWrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from tensorflow.contrib.opt.python.training import multitask_optimizer_wrapper
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import momentum


class MultitaskOptimizerWrapperTest(test.TestCase):
  """Tests for the multitask optimizer wrapper.
  """

  def testWrapper(self):
    with self.test_session():
      var0 = variables.Variable([1.0, 2.0], dtype=dtypes.float32)
      var1 = variables.Variable([3.0, 4.0], dtype=dtypes.float32)
      grads0 = constant_op.constant([0.1, 0.1], dtype=dtypes.float32)
      grads1 = constant_op.constant([0.01, 0.01], dtype=dtypes.float32)
      grads_allzero = constant_op.constant([0.0, 0.0], dtype=dtypes.float32)
      mom_opt_impl = momentum.MomentumOptimizer(learning_rate=2.0, momentum=0.9)
      mom_opt = multitask_optimizer_wrapper.MultitaskOptimizerWrapper(
          mom_opt_impl)
      mom_update = mom_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      mom_update_partial = mom_opt.apply_gradients(
          zip([grads_allzero, grads1], [var0, var1]))
      mom_update_no_action = mom_opt.apply_gradients(
          zip([grads_allzero, grads_allzero], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())
      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], self.evaluate(var0))
      self.assertAllClose([3.0, 4.0], self.evaluate(var1))

      self.assertEqual(["momentum"], mom_opt.get_slot_names())
      slot0 = mom_opt.get_slot(var0, "momentum")
      self.assertEquals(slot0.get_shape(), var0.get_shape())
      slot1 = mom_opt.get_slot(var1, "momentum")
      self.assertEquals(slot1.get_shape(), var1.get_shape())

      # Step 1: normal momentum update.
      self.evaluate(mom_update)
      # Check that the momentum accumulators have been updated.
      self.assertAllCloseAccordingToType(
          np.array([0.1, 0.1]), self.evaluate(slot0))
      self.assertAllCloseAccordingToType(
          np.array([0.01, 0.01]), self.evaluate(slot1))
      # Check that the parameters have been updated.
      self.assertAllCloseAccordingToType(
          np.array([1.0 - (0.1 * 2.0), 2.0 - (0.1 * 2.0)]), self.evaluate(var0))
      self.assertAllCloseAccordingToType(
          np.array([3.0 - (0.01 * 2.0), 4.0 - (0.01 * 2.0)]),
          self.evaluate(var1))

      # Step 2: momentum update that changes only slot1 but not slot0.
      self.evaluate(mom_update_partial)
      # Check that only the relevant momentum accumulator has been updated.
      self.assertAllCloseAccordingToType(
          np.array([0.1, 0.1]), self.evaluate(slot0))
      self.assertAllCloseAccordingToType(
          np.array([(0.9 * 0.01 + 0.01), (0.9 * 0.01 + 0.01)]),
          self.evaluate(slot1))

      # Step 3: momentum update that does not change anything.
      self.evaluate(mom_update_no_action)
      # Check that the momentum accumulators have *NOT* been updated.
      self.assertAllCloseAccordingToType(
          np.array([0.1, 0.1]), self.evaluate(slot0))
      self.assertAllCloseAccordingToType(
          np.array([(0.9 * 0.01 + 0.01), (0.9 * 0.01 + 0.01)]),
          self.evaluate(slot1))

  def testGradientClipping(self):
    with self.test_session():
      var0 = variables.Variable([1.0, 2.0], dtype=dtypes.float32)
      var1 = variables.Variable([3.0, 4.0], dtype=dtypes.float32)
      var2 = variables.Variable([3.0, 4.0], dtype=dtypes.float32)
      var3 = variables.Variable([3.0, 4.0], dtype=dtypes.float32)
      grads0 = constant_op.constant([10.0, 15.0], dtype=dtypes.float32)
      grads1 = constant_op.constant([0.0, 5.0], dtype=dtypes.float32)
      grads2 = constant_op.constant([0.0, 0.0], dtype=dtypes.float32)
      grads3 = None
      varlist = [var0, var1, var2, var3]
      gradients = [grads0, grads1, grads2, grads3]
      clipped_gradvars, global_norm = (
          multitask_optimizer_wrapper.clip_gradients_by_global_norm(
              six.moves.zip(gradients, varlist), clip_norm=1.0))
      clipped_grads = list(six.moves.zip(*clipped_gradvars))[0]
      reference_global_norm = np.sqrt(np.sum(np.square([10.0, 15.0, 0.0, 5.0])))
      self.assertAllCloseAccordingToType(
          self.evaluate(global_norm), reference_global_norm)
      self.assertAllCloseAccordingToType(
          self.evaluate(clipped_grads[2]), np.array([0., 0.]))
      self.assertEqual(clipped_grads[3], None)


if __name__ == "__main__":
  test.main()
