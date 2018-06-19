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
"""Tests for Adam."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests.xla_test import XLATestCase
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam


def adam_update_numpy(param,
                      g_t,
                      t,
                      m,
                      v,
                      alpha=0.001,
                      beta1=0.9,
                      beta2=0.999,
                      epsilon=1e-8):
  alpha_t = alpha * np.sqrt(1 - beta2**t) / (1 - beta1**t)

  m_t = beta1 * m + (1 - beta1) * g_t
  v_t = beta2 * v + (1 - beta2) * g_t * g_t

  param_t = param - alpha_t * m_t / (np.sqrt(v_t) + epsilon)
  return param_t, m_t, v_t


class AdamOptimizerTest(XLATestCase):

  def testBasic(self):
    for dtype in self.float_types:
      # TODO: test fails for float16 due to excessive precision requirements.
      if dtype == np.float16:
        continue
      with self.test_session(), self.test_scope():
        variable_scope.get_variable_scope().set_use_resource(True)

        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype)

        var0 = resource_variable_ops.ResourceVariable(var0_np)
        var1 = resource_variable_ops.ResourceVariable(var1_np)
        grads0 = array_ops.placeholder(dtype)
        grads1 = array_ops.placeholder(dtype)
        opt = adam.AdamOptimizer()
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Adam
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9**t, beta1_power.eval())
          self.assertAllCloseAccordingToType(0.999**t, beta2_power.eval())
          update.run(feed_dict={grads0: grads0_np, grads1: grads1_np})

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, var0.eval())
          self.assertAllCloseAccordingToType(var1_np, var1.eval())

  def testTensorLearningRate(self):
    for dtype in self.float_types:
      # TODO: test fails for float16 due to excessive precision requirements.
      if dtype == np.float16:
        continue
      with self.test_session(), self.test_scope():
        variable_scope.get_variable_scope().set_use_resource(True)

        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype)

        var0 = resource_variable_ops.ResourceVariable(var0_np)
        var1 = resource_variable_ops.ResourceVariable(var1_np)
        grads0 = array_ops.placeholder(dtype)
        grads1 = array_ops.placeholder(dtype)
        opt = adam.AdamOptimizer(constant_op.constant(0.001))
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Adam
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9**t, beta1_power.eval())
          self.assertAllCloseAccordingToType(0.999**t, beta2_power.eval())
          update.run(feed_dict={grads0: grads0_np, grads1: grads1_np})

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, var0.eval())
          self.assertAllCloseAccordingToType(var1_np, var1.eval())

  def testSharing(self):
    for dtype in self.float_types:
      # TODO: test fails for float16 due to excessive precision requirements.
      if dtype == np.float16:
        continue
      with self.test_session(), self.test_scope():
        variable_scope.get_variable_scope().set_use_resource(True)

        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype)

        var0 = resource_variable_ops.ResourceVariable(var0_np)
        var1 = resource_variable_ops.ResourceVariable(var1_np)
        grads0 = array_ops.placeholder(dtype)
        grads1 = array_ops.placeholder(dtype)
        opt = adam.AdamOptimizer()
        update1 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        update2 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())

        # Run 3 steps of intertwined Adam1 and Adam2.
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9**t, beta1_power.eval())
          self.assertAllCloseAccordingToType(0.999**t, beta2_power.eval())
          if t % 2 == 0:
            update1.run(feed_dict={grads0: grads0_np, grads1: grads1_np})
          else:
            update2.run(feed_dict={grads0: grads0_np, grads1: grads1_np})

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, var0.eval())
          self.assertAllCloseAccordingToType(var1_np, var1.eval())


if __name__ == "__main__":
  test.main()
