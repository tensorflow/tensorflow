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
"""Tests for RMSProp optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import rmsprop


class RmspropTest(xla_test.XLATestCase):

  def _rmsprop_update_numpy(self,
                            var,
                            g,
                            mg,
                            rms,
                            mom,
                            lr,
                            decay=0.9,
                            momentum=0.0,
                            epsilon=1e-10,
                            centered=False):
    rms_t = rms * decay + (1 - decay) * g * g
    denom_t = rms_t + epsilon
    if centered:
      mg_t = mg * decay + (1 - decay) * g
      denom_t -= mg_t * mg_t
    else:
      mg_t = mg
    mom_t = momentum * mom + lr * g / np.sqrt(denom_t, dtype=denom_t.dtype)
    var_t = var - mom_t
    return var_t, mg_t, rms_t, mom_t

  def testBasic(self):
    for dtype in self.float_types:
      for centered in [False, True]:
        with self.cached_session(), self.test_scope():
          # Initialize variables for numpy implementation.
          var0_np = np.array([1.0, 2.0], dtype=dtype)
          grads0_np = np.array([0.1, 0.1], dtype=dtype)
          var1_np = np.array([3.0, 4.0], dtype=dtype)
          grads1_np = np.array([0.01, 0.01], dtype=dtype)
          mg0_np = np.array([0.0, 0.0], dtype=dtype)
          mg1_np = np.array([0.0, 0.0], dtype=dtype)
          rms0_np = np.array([1.0, 1.0], dtype=dtype)
          rms1_np = np.array([1.0, 1.0], dtype=dtype)
          mom0_np = np.array([0.0, 0.0], dtype=dtype)
          mom1_np = np.array([0.0, 0.0], dtype=dtype)

          var0 = resource_variable_ops.ResourceVariable(var0_np)
          var1 = resource_variable_ops.ResourceVariable(var1_np)
          grads0 = constant_op.constant(grads0_np)
          grads1 = constant_op.constant(grads1_np)
          learning_rate = 3.0
          rms_opt = rmsprop.RMSPropOptimizer(learning_rate, centered=centered)
          rms_update = rms_opt.apply_gradients(
              zip([grads0, grads1], [var0, var1]))
          self.evaluate(variables.global_variables_initializer())

          mg0 = rms_opt.get_slot(var0, "mg")
          self.assertEqual(mg0 is not None, centered)
          mg1 = rms_opt.get_slot(var1, "mg")
          self.assertEqual(mg1 is not None, centered)
          rms0 = rms_opt.get_slot(var0, "rms")
          self.assertTrue(rms0 is not None)
          rms1 = rms_opt.get_slot(var1, "rms")
          self.assertTrue(rms1 is not None)
          mom0 = rms_opt.get_slot(var0, "momentum")
          self.assertTrue(mom0 is not None)
          mom1 = rms_opt.get_slot(var1, "momentum")
          self.assertTrue(mom1 is not None)

          # Fetch params to validate initial values
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))

          # Run 3 steps of RMSProp
          for _ in range(3):
            self.evaluate(rms_update)

            var0_np, mg0_np, rms0_np, mom0_np = self._rmsprop_update_numpy(
                var0_np,
                grads0_np,
                mg0_np,
                rms0_np,
                mom0_np,
                learning_rate,
                centered=centered)
            var1_np, mg1_np, rms1_np, mom1_np = self._rmsprop_update_numpy(
                var1_np,
                grads1_np,
                mg1_np,
                rms1_np,
                mom1_np,
                learning_rate,
                centered=centered)

            # Validate updated params
            if centered:
              self.assertAllCloseAccordingToType(mg0_np, self.evaluate(mg0))
              self.assertAllCloseAccordingToType(mg1_np, self.evaluate(mg1))
            self.assertAllCloseAccordingToType(rms0_np, self.evaluate(rms0))
            self.assertAllCloseAccordingToType(rms1_np, self.evaluate(rms1))
            self.assertAllCloseAccordingToType(mom0_np, self.evaluate(mom0))
            self.assertAllCloseAccordingToType(mom1_np, self.evaluate(mom1))
            self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
            self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))


if __name__ == "__main__":
  test.main()
