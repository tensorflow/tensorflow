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
"""Tests for Adam."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def adam_update_numpy(param,
                      g_t,
                      t,
                      m,
                      v,
                      lr=0.001,
                      beta1=0.9,
                      beta2=0.999,
                      epsilon=1e-7):
  lr_t = lr * np.sqrt(1 - beta2**(t + 1)) / (1 - beta1**(t + 1))

  m_t = beta1 * m + (1 - beta1) * g_t
  v_t = beta2 * v + (1 - beta2) * g_t * g_t

  param_t = param - lr_t * m_t / (np.sqrt(v_t) + epsilon)
  return param_t, m_t, v_t


def adam_update_numpy_amsgrad(param,
                              g_t,
                              t,
                              m,
                              v,
                              vhat,
                              lr=0.001,
                              beta1=0.9,
                              beta2=0.999,
                              epsilon=1e-7):
  lr_t = lr * np.sqrt(1 - beta2**(t + 1)) / (1 - beta1**(t + 1))

  m_t = beta1 * m + (1 - beta1) * g_t
  v_t = beta2 * v + (1 - beta2) * g_t * g_t
  vhat_t = np.maximum(vhat, v_t)

  param_t = param - lr_t * m_t / (np.sqrt(vhat_t) + epsilon)
  return param_t, m_t, v_t, vhat_t


def adam_sparse_update_numpy_amsgrad(param,
                                     indices,
                                     g_t,
                                     t,
                                     m,
                                     v,
                                     vhat,
                                     lr=0.001,
                                     beta1=0.9,
                                     beta2=0.999,
                                     epsilon=1e-7):
  m_t, v_t, vhat_t, param_t = (np.copy(m), np.copy(v), np.copy(vhat),
                               np.copy(param))
  lr_t = lr * np.sqrt(1 - beta2**(t + 1)) / (1 - beta1**(t + 1))
  m_t_slice = beta1 * m[indices] + (1 - beta1) * g_t
  v_t_slice = beta2 * v[indices] + (1 - beta2) * g_t * g_t
  m_t[indices] = m_t_slice
  v_t[indices] = v_t_slice
  v_hat_t = np.maximum(vhat_t, v_t)
  v_hat_t_slice = v_hat_t[indices]
  param_t_slice = param[indices] - (
      lr_t * (m_t_slice / (np.sqrt(v_hat_t_slice) + epsilon)))
  param_t[indices] = param_t_slice
  return param_t, m_t, v_t, vhat_t


def get_beta_accumulators(opt, dtype):
  local_step = math_ops.cast(opt.iterations + 1, dtype)
  beta_1_t = math_ops.cast(opt._get_hyper("beta_1"), dtype)
  beta_1_power = math_ops.pow(beta_1_t, local_step)
  beta_2_t = math_ops.cast(opt._get_hyper("beta_2"), dtype)
  beta_2_power = math_ops.pow(beta_2_t, local_step)
  return (beta_1_power, beta_2_power)


class AdamOptimizerTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testSparse(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session(use_gpu=True):
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.0, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.0, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = resource_variable_ops.ResourceVariable(var0_np)
        var1 = resource_variable_ops.ResourceVariable(var1_np)
        grads0_np_indices = np.array([0, 2], dtype=np.int32)
        grads0 = ops.IndexedSlices(
            constant_op.constant(grads0_np[grads0_np_indices]),
            constant_op.constant(grads0_np_indices), constant_op.constant([3]))
        grads1_np_indices = np.array([0, 2], dtype=np.int32)
        grads1 = ops.IndexedSlices(
            constant_op.constant(grads1_np[grads1_np_indices]),
            constant_op.constant(grads1_np_indices), constant_op.constant([3]))
        opt = adam.Adam()
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 3.0, 4.0], self.evaluate(var1))

        beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
        # Run 3 steps of Adam
        for t in range(3):
          self.assertAllCloseAccordingToType(0.9**(t + 1),
                                             self.evaluate(beta_1_power))
          self.assertAllCloseAccordingToType(0.999**(t + 1),
                                             self.evaluate(beta_2_power))
          update.run()

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  @test_util.run_deprecated_v1
  def testSparseDevicePlacement(self):
    for index_dtype in [dtypes.int32, dtypes.int64]:
      with self.cached_session(force_gpu=test.is_gpu_available()):
        # If a GPU is available, tests that all optimizer ops can be placed on
        # it (i.e. they have GPU kernels).
        var = variables.Variable([[1.0], [2.0]])
        indices = constant_op.constant([0, 1], dtype=index_dtype)
        g_sum = lambda: math_ops.reduce_sum(array_ops.gather(var, indices))  # pylint: disable=cell-var-from-loop
        optimizer = adam.Adam(3.0)
        minimize_op = optimizer.minimize(g_sum, var_list=[var])
        variables.global_variables_initializer().run()
        minimize_op.run()

  @test_util.run_deprecated_v1
  def testSparseRepeatedIndices(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        repeated_index_update_var = variables.Variable(
            [[1.0], [2.0]], dtype=dtype)
        aggregated_update_var = variables.Variable(
            [[1.0], [2.0]], dtype=dtype)
        grad_repeated_index = ops.IndexedSlices(
            constant_op.constant(
                [0.1, 0.1], shape=[2, 1], dtype=dtype),
            constant_op.constant([1, 1]),
            constant_op.constant([2, 1]))
        grad_aggregated = ops.IndexedSlices(
            constant_op.constant(
                [0.2], shape=[1, 1], dtype=dtype),
            constant_op.constant([1]),
            constant_op.constant([2, 1]))
        repeated_update = adam.Adam().apply_gradients(
            [(grad_repeated_index, repeated_index_update_var)])
        aggregated_update = adam.Adam().apply_gradients(
            [(grad_aggregated, aggregated_update_var)])
        variables.global_variables_initializer().run()
        self.assertAllClose(aggregated_update_var.eval(),
                            self.evaluate(repeated_index_update_var))
        for _ in range(3):
          repeated_update.run()
          aggregated_update.run()
          self.assertAllClose(aggregated_update_var.eval(),
                              self.evaluate(repeated_index_update_var))

  def doTestBasic(self, use_callable_params=False):
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.cached_session(use_gpu=True):
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = resource_variable_ops.ResourceVariable(
            var0_np, name="var0_%d" % i)
        var1 = resource_variable_ops.ResourceVariable(
            var1_np, name="var1_%d" % i)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)

        learning_rate = lambda: 0.001
        beta1 = lambda: 0.9
        beta2 = lambda: 0.999
        epsilon = lambda: 1e-8
        if not use_callable_params:
          learning_rate = learning_rate()
          beta1 = beta1()
          beta2 = beta2()
          epsilon = epsilon()

        opt = adam.Adam(learning_rate=learning_rate)
        if not context.executing_eagerly():
          update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

        self.evaluate(variables.global_variables_initializer())
        # Run 3 steps of Adam
        for t in range(3):
          beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
          self.assertAllCloseAccordingToType(0.9**(t + 1),
                                             self.evaluate(beta_1_power))
          self.assertAllCloseAccordingToType(0.999**(t + 1),
                                             self.evaluate(beta_2_power))
          if not context.executing_eagerly():
            self.evaluate(update)
          else:
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testResourceBasic(self):
    self.doTestBasic()

  def testBasicCallableParams(self):
    with context.eager_mode():
      self.doTestBasic(use_callable_params=True)

  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testBasicWithAmsgrad(self):
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.cached_session(use_gpu=True):
        # Initialize variables for numpy implementation.
        m0, v0, v0hat, m1, v1, v1hat = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = resource_variable_ops.ResourceVariable(
            var0_np, name="var0_%d" % i)
        var1 = resource_variable_ops.ResourceVariable(
            var1_np, name="var1_%d" % i)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)

        opt = adam.Adam(amsgrad=True)
        if not context.executing_eagerly():
          update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

        self.evaluate(variables.global_variables_initializer())
        # Run 3 steps of Adam
        for t in range(3):
          beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
          self.assertAllCloseAccordingToType(0.9**(t + 1),
                                             self.evaluate(beta_1_power))
          self.assertAllCloseAccordingToType(0.999**(t + 1),
                                             self.evaluate(beta_2_power))
          if not context.executing_eagerly():
            self.evaluate(update)
          else:
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

          var0_np, m0, v0, v0hat = adam_update_numpy_amsgrad(
              var0_np, grads0_np, t, m0, v0, v0hat)
          var1_np, m1, v1, v1hat = adam_update_numpy_amsgrad(
              var1_np, grads1_np, t, m1, v1, v1hat)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes
  def testSparseWithAmsgrad(self):
    # dtypes.half does not work on gpu + eager.
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.cached_session():
        m0 = np.array([[0.0], [0.0]])
        v0 = np.array([[0.0], [0.0]])
        v0hat = np.array([[0.0], [0.0]])
        indices_np = np.array([1])
        indices = constant_op.constant(indices_np, dtype=dtypes.int32)
        var0_np = np.array([[1.0], [2.0]], dtype=dtype.as_numpy_dtype)
        repeated_index_update_var = variables.Variable(var0_np, dtype=dtype)
        aggregated_update_var = variables.Variable(var0_np, dtype=dtype)
        grads0_np = np.array([[0.2]], dtype=dtype.as_numpy_dtype)
        grad_repeated_index = ops.IndexedSlices(
            constant_op.constant([0.1, 0.1], shape=[2, 1], dtype=dtype),
            constant_op.constant([1, 1]), constant_op.constant([2, 1]))
        grad_aggregated = ops.IndexedSlices(grads0_np, indices,
                                            constant_op.constant([2, 1]))
        opt_repeated = adam.Adam(amsgrad=True)
        opt_aggregated = adam.Adam(amsgrad=True)
        if not context.executing_eagerly():
          repeated_update = opt_repeated.apply_gradients(
              [(grad_repeated_index, repeated_index_update_var)])
          aggregated_update = opt_aggregated.apply_gradients(
              [(grad_aggregated, aggregated_update_var)])
        self.evaluate(variables.global_variables_initializer())
        self.assertAllClose(
            self.evaluate(aggregated_update_var),
            self.evaluate(repeated_index_update_var))
        for t in range(3):
          if not context.executing_eagerly():
            self.evaluate(repeated_update)
            self.evaluate(aggregated_update)
          else:
            opt_repeated.apply_gradients(
                [(grad_repeated_index, repeated_index_update_var)])
            opt_aggregated.apply_gradients(
                [(grad_aggregated, aggregated_update_var)])

          var0_np, m0, v0, v0hat = adam_sparse_update_numpy_amsgrad(
              var0_np, indices_np, grads0_np, t, m0, v0, v0hat)

          # Validate updated params
          self.assertAllCloseAccordingToType(
              var0_np, self.evaluate(aggregated_update_var))
          self.assertAllCloseAccordingToType(
              self.evaluate(aggregated_update_var),
              self.evaluate(repeated_index_update_var))

  @test_util.run_deprecated_v1
  def testBasicWithLearningRateDecay(self):
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.cached_session(use_gpu=True):
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = resource_variable_ops.ResourceVariable(
            var0_np, name="var0_%d" % i)
        var1 = resource_variable_ops.ResourceVariable(
            var1_np, name="var1_%d" % i)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)

        learning_rate = 0.001
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-7
        decay = 0.5

        opt = adam.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            decay=decay)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

        self.evaluate(variables.global_variables_initializer())
        # Run 3 steps of Adam
        for t in range(3):
          self.evaluate(update)
          lr_np = learning_rate / (1 + decay * t)

          var0_np, m0, v0 = adam_update_numpy(
              var0_np, grads0_np, t, m0, v0, lr=lr_np)
          var1_np, m1, v1 = adam_update_numpy(
              var1_np, grads1_np, t, m1, v1, lr=lr_np)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  @test_util.run_deprecated_v1
  def testBasicWithLearningRateInverseTimeDecay(self):
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.cached_session(use_gpu=True):
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = resource_variable_ops.ResourceVariable(
            var0_np, name="var0_%d" % i)
        var1 = resource_variable_ops.ResourceVariable(
            var1_np, name="var1_%d" % i)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)

        learning_rate = 0.001
        decay = 0.5
        lr_schedule = learning_rate_schedule.InverseTimeDecay(
            learning_rate, decay_steps=1.0, decay_rate=decay)
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-7

        opt = adam.Adam(
            learning_rate=lr_schedule,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

        self.evaluate(variables.global_variables_initializer())
        # Run 3 steps of Adam
        for t in range(3):
          self.evaluate(update)

          lr_np = learning_rate / (1 + decay * t)

          var0_np, m0, v0 = adam_update_numpy(
              var0_np, grads0_np, t, m0, v0, lr=lr_np)
          var1_np, m1, v1 = adam_update_numpy(
              var1_np, grads1_np, t, m1, v1, lr=lr_np)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  @test_util.run_deprecated_v1
  def testTensorLearningRate(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session(use_gpu=True):
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)
        opt = adam.Adam(constant_op.constant(0.001))
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
        # Run 3 steps of Adam
        for t in range(3):
          self.assertAllCloseAccordingToType(0.9**(t + 1),
                                             self.evaluate(beta_1_power))
          self.assertAllCloseAccordingToType(0.999**(t + 1),
                                             self.evaluate(beta_2_power))
          update.run()

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  @test_util.run_deprecated_v1
  def testSharing(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session(use_gpu=True):
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)
        opt = adam.Adam()
        update1 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        update2 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        # Run 3 steps of intertwined Adam1 and Adam2.
        for t in range(3):
          self.assertAllCloseAccordingToType(0.9**(t + 1),
                                             self.evaluate(beta_1_power))
          self.assertAllCloseAccordingToType(0.999**(t + 1),
                                             self.evaluate(beta_2_power))
          if t % 2 == 0:
            update1.run()
          else:
            update2.run()

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testSlotsUniqueEager(self):
    with context.eager_mode():
      v1 = resource_variable_ops.ResourceVariable(1.)
      v2 = resource_variable_ops.ResourceVariable(1.)
      opt = adam.Adam(1.)
      opt.minimize(lambda: v1 + v2, var_list=[v1, v2])
      # There should be iteration, and two unique slot variables for v1 and v2.
      self.assertEqual(
          5, len(set([v.experimental_ref() for v in opt.variables()])))
      self.assertEqual(
          self.evaluate(opt.variables()[0]), self.evaluate(opt.iterations))

  def testSetWeightsFromV1AdamWithoutMinimize(self):
    keras_v1_adam = optimizers.Adam()
    keras_v2_adam = adam.Adam()
    keras_v2_adam.set_weights(keras_v1_adam.get_weights())
    keras_v1_iteration = keras_v1_adam.iterations
    keras_v2_iteration = keras_v2_adam.iterations
    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(
        self.evaluate(keras_v1_iteration), self.evaluate(keras_v2_iteration))

  def testConstructAdamWithLR(self):
    opt = adam.Adam(lr=1.0)
    opt_2 = adam.Adam(learning_rate=0.1, lr=1.0)
    opt_3 = adam.Adam(learning_rate=0.1)
    self.assertIsInstance(opt.lr, variables.Variable)
    self.assertIsInstance(opt_2.lr, variables.Variable)
    self.assertIsInstance(opt_3.lr, variables.Variable)

    self.evaluate(variables.global_variables_initializer())
    self.assertAllClose(self.evaluate(opt.lr), (1.0))
    self.assertAllClose(self.evaluate(opt_2.lr), (1.0))
    self.assertAllClose(self.evaluate(opt_3.lr), (0.1))


if __name__ == "__main__":
  test.main()
