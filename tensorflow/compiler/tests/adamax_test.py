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
"""Tests for AdaMax optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.contrib.opt.python.training import adamax
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def adamax_update_numpy(param,
                        g_t,
                        t,
                        m,
                        v,
                        alpha=0.001,
                        beta1=0.9,
                        beta2=0.999,
                        epsilon=1e-8):
  m_t = beta1 * m + (1 - beta1) * g_t
  v_t = np.maximum(beta2 * v, np.abs(g_t))
  param_t = param - (alpha / (1 - beta1**t)) * (m_t / (v_t + epsilon))
  return param_t, m_t, v_t


class AdaMaxOptimizerTest(xla_test.XLATestCase):

  def testBasic(self):
    for i, dtype in enumerate(self.float_types):
      with self.cached_session(), self.test_scope():
        variable_scope.get_variable_scope().set_use_resource(True)
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype)

        var0 = resource_variable_ops.ResourceVariable(
            var0_np, name="var0_%d" % i)
        var1 = resource_variable_ops.ResourceVariable(
            var1_np, name="var1_%d" % i)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)

        opt = adamax.AdaMaxOptimizer()
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        opt_variables = opt.variables()
        beta1_power = opt._get_beta_accumulators()
        self.assertTrue(beta1_power is not None)
        self.assertIn(beta1_power, opt_variables)

        with ops.Graph().as_default():
          # Shouldn't return non-slot variables from other graphs.
          self.assertEqual(0, len(opt.variables()))

        variables.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())

        beta1_power = opt._get_beta_accumulators()

        # Run 3 steps of AdaMax
        for t in range(1, 4):
          update.run()

          self.assertAllCloseAccordingToType(0.9**(t + 1), beta1_power.eval())

          var0_np, m0, v0 = adamax_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adamax_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, var0.eval(), rtol=1e-2)
          self.assertAllCloseAccordingToType(var1_np, var1.eval(), rtol=1e-2)
          self.assertEqual("var0_%d/AdaMax:0" % (i,),
                           opt.get_slot(var=var0, name="m").name)

  def testTensorLearningRate(self):
    for dtype in self.float_types:
      with self.cached_session(), self.test_scope():
        variable_scope.get_variable_scope().set_use_resource(True)
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype)

        var0 = resource_variable_ops.ResourceVariable(var0_np)
        var1 = resource_variable_ops.ResourceVariable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)
        opt = adamax.AdaMaxOptimizer(constant_op.constant(0.001))
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())

        beta1_power = opt._get_beta_accumulators()

        # Run 3 steps of AdaMax
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9**t, beta1_power.eval())
          update.run()

          var0_np, m0, v0 = adamax_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adamax_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, var0.eval())
          self.assertAllCloseAccordingToType(var1_np, var1.eval())

if __name__ == "__main__":
  test.main()
