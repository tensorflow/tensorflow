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
"""Tests for GGTOptimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.contrib.opt.python.training.ggt import GGTOptimizer
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def ggt_update_numpy(param,
                     g_t,
                     lr,
                     grad_buffer,
                     m,
                     window,
                     t,
                     beta1=0.9,
                     eps=1e-4,
                     svd_eps=1e-6,
                     sigma_eps=1e-2):
  """Tests the correctness of one step of GGT."""
  m_t = m * beta1 + (1 - beta1) * g_t
  grad_buffer[((t - 1) % window), :] = m_t
  m_matrix = np.transpose(grad_buffer / np.sqrt(np.minimum(t, window)))
  mm = np.dot(np.transpose(m_matrix), m_matrix)
  damping = np.eye(window) * svd_eps
  u, sigma, _ = np.linalg.svd(mm + damping)

  sigma_sqrt_inv = np.power(np.sqrt(sigma) + sigma_eps, -3)
  new_step = np.linalg.multi_dot([
      m_matrix, u,
      np.diag(sigma_sqrt_inv),
      np.transpose(u),
      np.transpose(m_matrix), m_t
  ])

  sigma_sqrt_min = np.sqrt(sigma).min()

  if sigma_sqrt_min > eps:
    new_step += (m_t - np.linalg.multi_dot([
        m_matrix, u,
        np.diag(1.0 / sigma),
        np.transpose(u),
        np.transpose(m_matrix), m_t
    ])) * (1.0 / sigma_sqrt_min)

  param_t = param - lr * new_step
  return param_t, m_t, grad_buffer


class GGTOptimizerTest(test.TestCase):

  def doTestBasic(self, use_resource=False):
    # SVD does not support float16
    for i, dtype in enumerate([dtypes.float32, dtypes.float64]):
      with self.session(graph=ops.Graph()):
        # Initialize variables for numpy implementation.
        m0 = 0.0
        window = 3
        grad_buffer = np.zeros((window, 4), dtype=dtype.as_numpy_dtype)
        lr = 0.001
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(
              var0_np, name="var0_%d" % i)
          var1 = resource_variable_ops.ResourceVariable(
              var1_np, name="var1_%d" % i)
        else:
          var0 = variables.Variable(var0_np, name="var0")
          var1 = variables.Variable(var1_np, name="var1")
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)

        opt = GGTOptimizer(learning_rate=lr, window=window)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        opt_variables = opt.variables()

        m_t = opt._get_moment1()
        grad_buffer_t = opt._get_grad_buffer()
        g_t = opt._get_flat_grad()
        self.assertTrue(m_t is not None)
        self.assertTrue(grad_buffer_t is not None)
        self.assertTrue(g_t is not None)
        self.assertIn(m_t, opt_variables)
        self.assertIn(grad_buffer_t, opt_variables)
        self.assertIn(g_t, opt_variables)

        with ops.Graph().as_default():
          # Shouldn't return non-slot variables from other graphs.
          self.assertEqual(0, len(opt.variables()))

        if not context.executing_eagerly():
          self.evaluate(variables.global_variables_initializer())
          # Fetch params to validate initial values
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        m_t = opt._get_moment1()
        grad_buffer_t = opt._get_grad_buffer()
        g_t = opt._get_flat_grad()

        # Run 3 steps of GGT
        for t in range(1, 4):
          if not context.executing_eagerly():
            self.evaluate(update)
          elif t > 1:
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

          if t == 1:
            self.assertAllCloseAccordingToType(
                np.array([0.01, 0.01, 0.001, 0.001]), self.evaluate(m_t))
            self.assertAllCloseAccordingToType(
                np.array([[0.01, 0.01, 0.001, 0.001], [0., 0., 0., 0.],
                          [0., 0., 0., 0.]]), self.evaluate(grad_buffer_t))
          elif t == 2:
            self.assertAllCloseAccordingToType(
                np.array([0.019, 0.019, 0.0019, 0.0019]), self.evaluate(m_t))
            self.assertAllCloseAccordingToType(
                np.array([[0.01, 0.01, 0.001, 0.001],
                          [0.019, 0.019, 0.0019, 0.0019], [0., 0., 0., 0.]]),
                self.evaluate(grad_buffer_t))
          else:
            self.assertAllCloseAccordingToType(
                np.array([0.0271, 0.0271, 0.00271, 0.00271]),
                self.evaluate(m_t))
            self.assertAllCloseAccordingToType(
                np.array([[0.01, 0.01, 0.001,
                           0.001], [0.019, 0.019, 0.0019, 0.0019],
                          [0.0271, 0.0271, 0.00271, 0.00271]]),
                self.evaluate(grad_buffer_t))

          self.assertAllCloseAccordingToType([0.1, 0.1, 0.01, 0.01],
                                             self.evaluate(g_t))

          var_np = np.append(var0_np, var1_np)
          grads_np = np.append(grads0_np, grads1_np)
          var_np, m0, grad_buffer = ggt_update_numpy(var_np, grads_np, lr,
                                                     grad_buffer, m0, window, t)

          var0_np = var_np[:2]
          var1_np = var_np[2:]
          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testBasic(self):
    with self.cached_session():
      self.doTestBasic(use_resource=False)

  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testResourceBasic(self):
    self.doTestBasic(use_resource=True)


if __name__ == "__main__":
  test.main()
