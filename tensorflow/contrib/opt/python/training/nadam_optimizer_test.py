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
"""Tests for Nadam."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.opt.python.training import nadam_optimizer
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def nadam_update_numpy(param,
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

  m_bar = (1 - beta1) * g_t + beta1 * m_t

  param_t = param - alpha_t * m_bar / (np.sqrt(v_t) + epsilon)
  return param_t, m_t, v_t


class NadamOptimizerTest(test.TestCase):

  def doTestSparse(self, use_resource=False):
    # need to use a larger value of epsilon here so that
    # np.sqrt(v_t) + epsilon doesn't get rounded to 0 when
    # the dtype is half and np.sqrt(v_t) = 0, as is the case
    # when the gradient is 0
    sparse_epsilon = 1e-7
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0, 0.01], dtype=dtype.as_numpy_dtype)

        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(var0_np)
          var1 = resource_variable_ops.ResourceVariable(var1_np)
        else:
          var0 = variables.Variable(var0_np)
          var1 = variables.Variable(var1_np)
        grads0_np_indices = np.array([0, 2], dtype=np.int32)
        grads0 = ops.IndexedSlices(
            constant_op.constant(grads0_np[grads0_np_indices]),
            constant_op.constant(grads0_np_indices), constant_op.constant([3]))
        grads1_np_indices = np.array([0, 2], dtype=np.int32)
        grads1 = ops.IndexedSlices(
            constant_op.constant(grads1_np[grads1_np_indices]),
            constant_op.constant(grads1_np_indices), constant_op.constant([3]))
        opt = nadam_optimizer.NadamOptimizer(epsilon=sparse_epsilon)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 3.0, 4.0], var1.eval())

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Nadam
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9**t, beta1_power.eval())
          self.assertAllCloseAccordingToType(0.999**t, beta2_power.eval())
          update.run()

          var0_np, m0, v0 = nadam_update_numpy(var0_np, grads0_np, t, m0, v0,
                                               epsilon=sparse_epsilon)
          var1_np, m1, v1 = nadam_update_numpy(var1_np, grads1_np, t, m1, v1,
                                               epsilon=sparse_epsilon)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, var0.eval())
          self.assertAllCloseAccordingToType(var1_np, var1.eval())

  def testSparse(self):
    self.doTestSparse(use_resource=False)

  def testResourceSparse(self):
    self.doTestSparse(use_resource=True)

  def doTestBasic(self, use_resource=False):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(var0_np)
          var1 = resource_variable_ops.ResourceVariable(var1_np)
        else:
          var0 = variables.Variable(var0_np)
          var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)
        opt = nadam_optimizer.NadamOptimizer()
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Nadam
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9**t, beta1_power.eval())
          self.assertAllCloseAccordingToType(0.999**t, beta2_power.eval())
          update.run()

          var0_np, m0, v0 = nadam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = nadam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, var0.eval())
          self.assertAllCloseAccordingToType(var1_np, var1.eval())

  def testBasic(self):
    self.doTestBasic(use_resource=False)

  def testResourceBasic(self):
    self.doTestBasic(use_resource=True)


if __name__ == "__main__":
  test.main()
