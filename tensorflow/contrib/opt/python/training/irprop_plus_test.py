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
"""Tests for IRpropPlus."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.opt.python.training import irprop_plus
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables

from tensorflow.python.platform import test

def irprop_update_numpy(params,
                        old_w_t,
                        g_t,
                        old_g_t,
                        delta_t,
                        eta_minus,
                        eta_plus,
                        delta_min,
                        delta_max,
                        error,
                        old_error):

  # Hold an extra var to restore the previous weight
  old_params = np.copy(params)
  old_g = np.copy(g_t)

  for i, _ in enumerate(g_t):
    if g_t[i] * old_g_t[i] > 0:
      delta_t[i] = np.fmin(delta_max, delta_t[i] * eta_plus)
      params[i] -= np.sign(g_t[i]) * delta_t[i]

    elif g_t[i] * old_g_t[i] < 0:
      delta_t[i] = np.fmax(delta_min, delta_t[i] * eta_minus)
      if error > old_error:
        # retract step
        params[i] = old_w_t[i]
      old_g[i] = 0

    else:
      params[i] -= np.sign(g_t[i]) * delta_t[i]
  return params, delta_t, old_g, old_params


class IRpropPlusTest(test.TestCase):

  def _testDense(self,
                 use_resource=False,
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

        # errors
        loss0_np = 0.5
        old_loss0_np = -0.25
        large_loss0_np = 2.5

        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)
        neg_grads1_np = np.negative(grads1_np)
        old_grads1 = np.array([0, 0], dtype=dtype.as_numpy_dtype)
        delta1 = np.array([delta_zero, delta_zero], dtype=dtype.as_numpy_dtype)

        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(var0_np)
          var1 = resource_variable_ops.ResourceVariable(var1_np)
        else:
          var0 = variables.Variable(var0_np)
          var1 = variables.Variable(var1_np)

        loss0_var = variables.Variable(loss0_np)
        large_loss0_var = variables.Variable(large_loss0_np)

        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)

        opt = irprop_plus.IRpropPlusOptimizer(eta_minus=eta_minus,
                                              eta_plus=eta_plus,
                                              delta_min=delta_min,
                                              delta_max=delta_max,
                                              delta_zero=delta_zero)

        pos_update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]),
                                         loss0_var)
        neg_update = opt.apply_gradients(zip([-grads0, -grads1], [var0, var1]),
                                         loss0_var)

        # large loss value to retract the previous weight
        pos_large_update = opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]),
            large_loss0_var)

        neg_large_update = opt.apply_gradients(
            zip([-grads0, -grads1], [var0, var1]),
            large_loss0_var)

        if not context.executing_eagerly():
          self.evaluate(variables.global_variables_initializer())
          # Fetch params to validate initial values
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        old_w_t0 = var0
        old_w_t1 = var1

        # Run 10 steps of irprop+
        # first 3 steps with positive gradient (same consecutive sign)
        # next 3 steps with negative gradient
        # - first iteration with negative sign will retract the weight
        # - second iteration will have 0 sign
        # - next iterations will have the same sign
        # last 4 steps with alternate gradient
        # - first iteration won't retract the weight (neg sign)
        # - second it will have 0 sign
        # - third iteration will retract the weight (neg sign)
        # - fourth iteration 0 sign
        for t in range(1, 11):
          loss0 = loss0_np
          old_loss0 = old_loss0_np

          if t < 4:
            grads0_sign = grads0_np
            grads1_sign = grads1_np
            if not context.executing_eagerly():
              self.evaluate(pos_update)
            elif t > 1:
              opt.apply_gradients(zip([grads0, grads1], [var0, var1]),
                                  loss0_var)

          elif t < 7:
            grads0_sign = neg_grads0_np
            grads1_sign = neg_grads1_np

            if not context.executing_eagerly():
              loss0 = large_loss0_np
              self.evaluate(neg_large_update)
            else:
              opt.apply_gradients(zip([-grads0, -grads1], [var0, var1]),
                                  large_loss0_var)
          else:
            if t & 1 == 0:
              grads0_sign = neg_grads0_np
              grads1_sign = neg_grads1_np
              # retract weight
              update = neg_update
              if context.executing_eagerly():
                opt.apply_gradients(zip([-grads0, -grads1], [var0, var1]),
                                    loss0_var)
            else:
              # pos grad
              grads0_sign = grads0_np
              grads1_sign = grads1_np
              old_loss0 = loss0
              update = pos_update

              if t > 8:
                loss0 = large_loss0_np
                update = pos_large_update
                if context.executing_eagerly():
                  pos_large_update = opt.apply_gradients(zip([grads0, grads1],
                                                             [var0, var1]),
                                                         large_loss0_var)
              else:
                if context.executing_eagerly():
                  opt.apply_gradients(zip([grads0, grads1], [var0, var1]),
                                      loss0_var)

            if not context.executing_eagerly():
              self.evaluate(update)

          var0_np, delta0, old_grads0, old_w_t0 = irprop_update_numpy(
              var0_np,
              old_w_t0,
              grads0_sign,
              old_grads0,
              delta0,
              eta_minus=eta_minus,
              eta_plus=eta_plus,
              delta_min=delta_min,
              delta_max=delta_max,
              error=loss0,
              old_error=old_loss0)

          var1_np, delta1, old_grads1, old_w_t1 = irprop_update_numpy(
              var1_np,
              old_w_t1,
              grads1_sign,
              old_grads1,
              delta1,
              eta_minus=eta_minus,
              eta_plus=eta_plus,
              delta_min=delta_min,
              delta_max=delta_max,
              error=loss0,
              old_error=old_loss0)

          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def _testInvalidLoss(self):

    for dtype in [dtypes.float32]:

      loss_var_none = None
      loss_var_vector = variables.Variable(
          np.array([1.0], dtype=dtype.as_numpy_dtype))

      var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
      grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)

      var0 = constant_op.constant(var0_np)
      grads0 = constant_op.constant(grads0_np)

      opt = irprop_plus.IRpropPlusOptimizer()
      grads = zip([grads0], [var0])
      self.assertRaises(ValueError, opt.apply_gradients, grads, loss_var_none)
      self.assertRaises(ValueError, opt.apply_gradients, grads, loss_var_vector)
      # type error in case no loss variable is passed, sanity check
      self.assertRaises(TypeError, opt.apply_gradients, grads)

  def testDense(self):

    self._testDense(use_resource=False)
    self._testDense(use_resource=False,
                    eta_minus=0.3,
                    eta_plus=1.5,
                    delta_min=1e-8,
                    delta_max=30,
                    delta_zero=0.0125)
    self._testDense(use_resource=True)
    self._testDense(use_resource=True,
                    eta_minus=0.3,
                    eta_plus=1.5,
                    delta_min=1e-8,
                    delta_max=30,
                    delta_zero=0.0125)

    self._testInvalidLoss()

if __name__ == '__main__':
  test.main()
