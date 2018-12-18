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

from tensorflow.contrib.optimizer_v2 import irprop_plus
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

def irprop_update_np(params,
                     old_w_t,
                     g_t,
                     old_g_t,
                     delta_t,
                     error,
                     old_error,
                     eta_minus=0.5,
                     eta_plus=1.2,
                     delta_min=1e-6,
                     delta_max=50):

  # Hold an extra var to restore the previous weight
  old_params = np.copy(params)
  old_g = np.copy(g_t)

  for i, _ in enumerate(g_t):
    if g_t[i] * old_g_t[i] > 0:
      delta_t[i] = np.fmin(delta_max, delta_t[i] * eta_plus)
      params[i] -= np.sign(g_t[i]) * delta_t[i]

    elif g_t[i] * old_g_t[i] < 0:
      delta_t[i] = np.fmax(delta_min, delta_t[i] * eta_minus)

      # retract step
      if error[i] > old_error[i]:
        params[i] = old_w_t[i]
      old_g[i] = 0

    else:
      params[i] -= np.sign(g_t[i]) * delta_t[i]

  old_error = np.copy(error)
  return params, delta_t, old_g, old_params, old_error


class IRpropPlusTest(test.TestCase):

  def testBasic(self):
    with self.cached_session():
      self.doTestBasic(use_resource=False)

  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testResourceBasic(self):
    self.doTestBasic(use_resource=True)

  def doTestBasic(self, use_resource=False):
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.session(graph=ops.Graph()):
        # trainable variables
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)

        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(
              var0_np, name="var0_%d" % i)
          var1 = resource_variable_ops.ResourceVariable(
              var1_np, name="var1_%d" % i)
        else:
          var0 = variables.Variable(var0_np, dtype=dtype)
          var1 = variables.Variable(var1_np, dtype=dtype)

        # vars used to restore the previous weight
        old_w_t0 = var0_np
        old_w_t1 = var1_np

        grads0_np = np.array([10, 20], dtype=dtype.as_numpy_dtype)
        neg_grads0_np = np.negative(grads0_np)

        grads1_np = np.array([3, 3], dtype=dtype.as_numpy_dtype)
        neg_grads1_np = np.negative(grads1_np)

        delta0 = np.array([0.5, 0.5], dtype=dtype.as_numpy_dtype)
        delta1 = np.array([0.5, 0.5], dtype=dtype.as_numpy_dtype)

        old_error0 = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        old_error1 = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)

        old_grads0 = np.array([0, 0], dtype=dtype.as_numpy_dtype)
        old_grads1 = np.array([0, 0], dtype=dtype.as_numpy_dtype)

        opt = irprop_plus.IRpropPlusOptimizer()

        # eager loss has to be a callable
        def cost():
          return 5 * var0 ** 2 + 3 * var1

        if not context.executing_eagerly():
          cost = 5 * var0 **2 + 3 * var1

        gvs = opt.compute_gradients(cost, [var0, var1])
        opt.apply_gradients(gvs)

        opt_variables = opt.variables()
        error, old_error = opt._get_error_values()
        self.assertTrue(error is not None)
        self.assertTrue(old_error is not None)
        self.assertIn(error, opt_variables)
        self.assertIn(old_error, opt_variables)

        with ops.Graph().as_default():
          # Shouldn't return non-slot variables from other graphs.
          self.assertEqual(0, len(opt.variables()))

        if not context.executing_eagerly():
          self.evaluate(variables.global_variables_initializer())
          # Fetch params to validate initial values
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))
          self.assertAllClose([0.0, 0.0], self.evaluate(error))
          self.assertAllClose([0.0, 0.0], self.evaluate(old_error))

        if not context.executing_eagerly():
          self.evaluate(variables.global_variables_initializer())
          pos_update = opt.apply_gradients(zip([grads0_np, grads1_np],
                                               [var0, var1]))
          neg_update = opt.apply_gradients(zip([neg_grads0_np, neg_grads1_np],
                                               [var0, var1]))

        # Run 10 steps of irprop+
        # first 3 steps with positive gradient (same consecutive sign)
        # next 3 steps with negative gradient
        # - 4th iteration with negative sign will not retract the weight
        # - 5th iteration will have 0 sign
        # - next iteration will have the same sign
        # last 4 steps with alternate gradient sign
        # - 7th iteration will retract the weight (neg sign, err>old_err)
        # - 8th it will have 0 sign
        # - 9th iteration will retract the weight (neg sign, err>old_err)
        # - last iteration 0 sign
        for t in range(1, 11):
          # get error for the local update
          error = self._cost(var0_np, var1_np)

          if t < 3:
            grads0 = grads0_np
            grads1 = grads1_np

            if not context.executing_eagerly():
              self.evaluate(pos_update)
            elif t > 1:
              opt.compute_gradients(cost, [var0, var1])
              opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

          elif t < 7:
            grads0 = neg_grads0_np
            grads1 = neg_grads1_np

            if not context.executing_eagerly():
              self.evaluate(neg_update)
            else:
              opt.compute_gradients(cost, [var0, var1])
              opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
          else:
            if t & 1 == 0:
              grads0 = neg_grads0_np
              grads1 = neg_grads1_np
              if not context.executing_eagerly():
                self.evaluate(neg_update)
              else:
                opt.compute_gradients(cost, [var0, var1])
                opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

            else:
              grads0 = grads0_np
              grads1 = grads1_np
              if not context.executing_eagerly():
                self.evaluate(pos_update)
              else:
                opt.compute_gradients(cost, [var0, var1])
                opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

          var0_np, delta0, old_grads0, old_w_t0, old_error0 = irprop_update_np(
              var0_np,
              old_w_t0,
              grads0,
              old_grads0,
              delta0,
              error=error,
              old_error=old_error0)
          var1_np, delta1, old_grads1, old_w_t1, old_error1 = irprop_update_np(
              var1_np,
              old_w_t1,
              grads1,
              old_grads1,
              delta1,
              error=error,
              old_error=old_error1)

          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def _cost(self, x, y):
    return 5 * x **2 + 3 * y

  def testTwoSessions(self):
    optimizer = irprop_plus.IRpropPlusOptimizer()
    g = ops.Graph()
    with g.as_default():
      with session.Session():
        var0 = variables.Variable(np.array([1.0, 2.0]), name="v0")
        cost0 = 2 * var0 ** 2
        gv0 = optimizer.compute_gradients(cost0, [var0])
        optimizer.apply_gradients(gv0)

    gg = ops.Graph()
    with gg.as_default():
      with session.Session():
        var0 = variables.Variable(np.array([1.0, 2.0]), name="v0")
        cost0 = 2 * var0 ** 2
        gv0 = optimizer.compute_gradients(cost0, [var0])
        # If the optimizer saves any state not keyed by graph the following line
        # fails.
        optimizer.apply_gradients(gv0)

  def testSlotsUniqueEager(self):
    with context.eager_mode():
      v1 = resource_variable_ops.ResourceVariable(1.)
      v2 = resource_variable_ops.ResourceVariable(1.)
      opt = irprop_plus.IRpropPlusOptimizer()
      opt.minimize(lambda: v1 + v2)
      # There should be two non-slot variables, and two unique slot variables
      # for v1 and v2 respectively.
      self.assertEqual(6, len(set(opt.variables())))


if __name__ == '__main__':
  test.main()
