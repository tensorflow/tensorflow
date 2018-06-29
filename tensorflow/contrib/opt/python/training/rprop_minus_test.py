# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Rprop-"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.opt.python.training import rprop_minus
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def rprop_update_numpy(params,
                       g_t,
                       old_g_t,
                       d_t,
                       eta_minus,
                       eta_plus,
                       delta_min,
                       delta_max):

  for i, _ in enumerate(g_t):

    current_g = g_t[i]
    mul_grad = g_t[i] * old_g_t[i]

    if mul_grad > 0:
      d_t[i] = np.fmin(delta_max, d_t[i] * eta_plus)
    elif mul_grad < 0:
      d_t[i] = np.fmax(delta_min, d_t[i] * eta_minus)

    params[i] -= np.sign(current_g) * d_t[i]

  return params, d_t, g_t

class RpropMinusTest(test.TestCase):

  def _testDense(self,
                 eta_minus=0.5,
                 eta_plus=1.2,
                 delta_min=1e-6,
                 delta_max=50,
                 delta_zero=0.5):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.test_session(use_gpu=True):

        # Initialize variables for numpy implementation.
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        neg_grads0_np = np.negative(grads0_np)
        old_grads0 = np.array([0, 0], dtype=dtype.as_numpy_dtype)
        delta0 = np.array([delta_zero, delta_zero], dtype=dtype.as_numpy_dtype)

        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)
        neg_grads1_np = np.negative(grads1_np)
        old_grads1 = np.array([0, 0], dtype=dtype.as_numpy_dtype)
        delta1 = np.array([delta_zero, delta_zero], dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)

        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)

        opt = rprop_minus.RpropMinusOptimizer(eta_minus=eta_minus,
                                              eta_plus=eta_plus,
                                              delta_min=delta_min,
                                              delta_max=delta_max,
                                              delta_zero=delta_zero)

        pos_update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        neg_update = opt.apply_gradients(zip([-grads0, -grads1], [var0, var1]))
        if not context.executing_eagerly():
          self.evaluate(variables.global_variables_initializer())
          # Fetch params to validate initial values
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        # Run 10 steps of rprop-
        # first 3 steps with positive gradient (same consecutive sign)
        # next 3 steps with negative gradient (first iteration only will have negative sign)
        # last 4 steps with alternate gradient (negative sign)
        for t in range(1, 10):

          if t < 4:
            grads0_sign = grads0_np
            grads1_sign = grads1_np
            if not context.executing_eagerly():
              self.evaluate(pos_update)
            elif t > 1:
              opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
              # pos_update
          elif t < 7:
            grads0_sign = neg_grads0_np
            grads1_sign = neg_grads1_np
            if not context.executing_eagerly():
              self.evaluate(neg_update)
            else:
              opt.apply_gradients(zip([-grads0, -grads1], [var0, var1]))
              # neg_update
          else:
            if t % 2 == 0:
              grads0_sign = neg_grads0_np
              grads1_sign = neg_grads1_np
              update = neg_update
            else:
              grads0_sign = grads0_np
              grads1_sign = grads1_np
              update = pos_update
            if not context.executing_eagerly():
              self.evaluate(update)
            else:
              # pos_update
              opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

          var0_np, delta0, old_grads0 = rprop_update_numpy(
              var0_np,
              grads0_sign,
              old_grads0,
              delta0,
              eta_minus=eta_minus,
              eta_plus=eta_plus,
              delta_min=delta_min,
              delta_max=delta_max)
          var1_np, delta1, old_grads1 = rprop_update_numpy(
              var1_np,
              grads1_sign,
              old_grads1,
              delta1,
              eta_minus=eta_minus,
              eta_plus=eta_plus,
              delta_min=delta_min,
              delta_max=delta_max)

          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testDense(self):

    self._testDense()
    self._testDense(eta_minus=0.25,
                    eta_plus=1.5,
                    delta_min=1e-8,
                    delta_max=25,
                    delta_zero=1)

if __name__ == '__main__':
  test.main()
