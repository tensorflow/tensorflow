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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import nadam
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def get_beta_accumulators(opt, dtype):
  local_step = math_ops.cast(opt.iterations + 1, dtype)
  beta_1_t = math_ops.cast(opt._get_hyper("beta_1"), dtype)
  beta_1_power = math_ops.pow(beta_1_t, local_step)
  beta_2_t = math_ops.cast(opt._get_hyper("beta_2"), dtype)
  beta_2_power = math_ops.pow(beta_2_t, local_step)
  return (beta_1_power, beta_2_power)


def update_m_cache(m_cache, t, beta1=0.9):
  mu_t = beta1 * (1 - 0.5 * 0.96**(0.004 * (t + 1)))
  m_cache_t = m_cache * mu_t
  return m_cache_t


def nadam_update_numpy(param,
                       g_t,
                       t,
                       m,
                       v,
                       m_cache,
                       alpha=0.001,
                       beta1=0.9,
                       beta2=0.999,
                       epsilon=1e-8):

  mu_t = beta1 * (1 - 0.5 * 0.96**(0.004 * (t + 1)))
  mu_t_1 = beta1 * (1 - 0.5 * 0.96**(0.004 * (t + 2)))
  m_cache_t_1 = m_cache * mu_t_1
  g_prime_t = g_t / (1 - m_cache)
  m_t = beta1 * m + (1 - beta1) * g_t
  v_t = beta2 * v + (1 - beta2) * g_t * g_t

  m_prime_t = m_t / (1 - m_cache_t_1)
  v_prime_t = v_t / (1 - beta2**(t + 1))
  m_bar_t = (1 - mu_t) * g_prime_t + mu_t_1 * m_prime_t

  param_t = param - alpha * m_bar_t / (np.sqrt(v_prime_t) + epsilon)
  return param_t, m_t, v_t


class NadamOptimizerTest(test.TestCase):

  def testSparse(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    sparse_epsilon = 1e-7
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with ops.Graph().as_default(), self.cached_session():
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1, mcache = 0.0, 0.0, 0.0, 0.0, 1.0
        var0_np = np.array([1.0, 1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0, 0.01], dtype=dtype.as_numpy_dtype)

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
        opt = nadam.Nadam(epsilon=sparse_epsilon)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 1.0, 2.0], var0)
        self.assertAllClose([3.0, 3.0, 4.0], var1)

        beta1_power, beta2_power = get_beta_accumulators(opt, dtype)

        # Run 3 steps of Nadam
        for t in range(3):
          self.assertAllCloseAccordingToType(0.9**(t + 1), beta1_power)
          self.assertAllCloseAccordingToType(0.999**(t + 1), beta2_power)
          update.run()

          mcache = update_m_cache(mcache, t)
          var0_np, m0, v0 = nadam_update_numpy(
              var0_np, grads0_np, t, m0, v0, mcache, epsilon=sparse_epsilon)
          var1_np, m1, v1 = nadam_update_numpy(
              var1_np, grads1_np, t, m1, v1, mcache, epsilon=sparse_epsilon)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, var0)
          self.assertAllCloseAccordingToType(var1_np, var1)

  def testBasic(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with ops.Graph().as_default(), self.cached_session():
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1, mcache = 0.0, 0.0, 0.0, 0.0, 1.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)
        opt = nadam.Nadam()
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0)
        self.assertAllClose([3.0, 4.0], var1)

        # Run 3 steps of Nadam
        for t in range(3):
          update.run()

          mcache = update_m_cache(mcache, t)
          var0_np, m0, v0 = nadam_update_numpy(var0_np, grads0_np, t, m0, v0,
                                               mcache)
          var1_np, m1, v1 = nadam_update_numpy(var1_np, grads1_np, t, m1, v1,
                                               mcache)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, var0)
          self.assertAllCloseAccordingToType(var1_np, var1)

  def testConstructNAdamWithLR(self):
    opt = nadam.Nadam(lr=1.0)
    opt_2 = nadam.Nadam(learning_rate=0.1, lr=1.0)
    opt_3 = nadam.Nadam(learning_rate=0.1)
    self.assertIsInstance(opt.lr, variables.Variable)
    self.assertIsInstance(opt_2.lr, variables.Variable)
    self.assertIsInstance(opt_3.lr, variables.Variable)

    self.evaluate(variables.global_variables_initializer())
    self.assertAllClose(self.evaluate(opt.lr), (1.0))
    self.assertAllClose(self.evaluate(opt_2.lr), (1.0))
    self.assertAllClose(self.evaluate(opt_3.lr), (0.1))

  def testConstructNAdamWithScheduleDecay(self):
    opt = nadam.Nadam(schedule_decay=0.2)
    self.assertIsInstance(opt.decay, variables.Variable)
    self.evaluate(variables.global_variables_initializer())
    self.assertAllClose(self.evaluate(opt.decay), (0.2))


if __name__ == "__main__":
  test.main()
