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
"""Tests for tensorflow.learning.training_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.framework.test_util import TensorFlowTestCase
# Import resource_variable_ops for the variables-to-tensor implicit conversion.
from tensorflow.python.ops import resource_variable_ops  # pylint: disable=unused-import
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import training_ops


class TrainingOpsTest(TensorFlowTestCase):

  def _toType(self, dtype):
    if dtype == np.float16:
      return dtypes.float16
    elif dtype == np.float32:
      return dtypes.float32
    elif dtype == np.float64:
      return dtypes.float64
    elif dtype == np.int32:
      return dtypes.int32
    elif dtype == np.int64:
      return dtypes.int64
    else:
      assert False, (dtype)

  def _testTypes(self, x, alpha, delta, use_gpu=None):
    self.setUp()
    with self.session(use_gpu=use_gpu):
      var = variables.VariableV1(x)
      variables.global_variables_initializer().run()
      self.assertAllCloseAccordingToType(x, self.evaluate(var))
      apply_sgd = training_ops.apply_gradient_descent(var, alpha, delta)
      out = self.evaluate(apply_sgd)
      self.assertShapeEqual(out, apply_sgd)
      self.assertAllCloseAccordingToType(x - alpha * delta, out)

  @test_util.run_deprecated_v1
  def testApplyGradientDescent(self):
    for (dtype, use_gpu) in itertools.product(
        [np.float16, np.float32, np.float64], [False, True]):
      x = np.arange(100).astype(dtype)
      alpha = np.array(2.0).astype(dtype)
      delta = np.arange(100).astype(dtype)
      self._testTypes(x, alpha, delta, use_gpu)

  def _testTypesForAdagrad(self, x, y, lr, grad, use_gpu=None):
    self.setUp()
    with self.session(use_gpu=use_gpu):
      var = variables.VariableV1(x)
      accum = variables.VariableV1(y)
      variables.global_variables_initializer().run()

      self.assertAllCloseAccordingToType(x, self.evaluate(var))
      apply_adagrad = training_ops.apply_adagrad(var, accum, lr, grad)
      out = self.evaluate(apply_adagrad)
      self.assertShapeEqual(out, apply_adagrad)
      self.assertAllCloseAccordingToType(x - lr * grad * (y + grad * grad)**
                                         (-0.5), out)
      self.assertAllCloseAccordingToType(y + grad * grad, self.evaluate(accum))

  def _testTypesForFtrl(self,
                        x,
                        y,
                        z,
                        lr,
                        grad,
                        use_gpu=None,
                        l1=0.0,
                        l2=0.0,
                        lr_power=-0.5):
    self.setUp()
    with self.session(use_gpu=use_gpu):
      var = variables.VariableV1(x)
      accum = variables.VariableV1(y)
      linear = variables.VariableV1(z)
      variables.global_variables_initializer().run()

      self.assertAllCloseAccordingToType(x, self.evaluate(var))
      apply_ftrl = training_ops.apply_ftrl(var, accum, linear, grad, lr, l1, l2,
                                           lr_power)
      out = self.evaluate(apply_ftrl)
      self.assertShapeEqual(out, apply_ftrl)
      accum_update = y + grad * grad
      linear_update = z + grad - (accum_update**(-lr_power) - y**
                                  (-lr_power)) / lr * x
      quadratic = 1.0 / (accum_update**(lr_power) * lr) + 2 * l2
      expected_out = np.array([(
          np.sign(linear_update[i]) * l1 - linear_update[i]) / (quadratic[i]) if
                               np.abs(linear_update[i]) > l1 else 0.0
                               for i in range(linear_update.size)])
      self.assertAllCloseAccordingToType(accum_update, self.evaluate(accum))
      if x.dtype == np.float16:
        # The calculations here really are not very precise in float16.
        self.assertAllClose(
            linear_update, self.evaluate(linear), rtol=2e-2, atol=2e-2)
        self.assertAllClose(expected_out, out, rtol=2e-2, atol=2e-2)
      elif x.dtype == np.float32:
        # The calculations here not sufficiently precise in float32.
        self.assertAllClose(
            linear_update, self.evaluate(linear), rtol=1e-5, atol=1e-5)
        self.assertAllClose(expected_out, out, rtol=1e-5, atol=1e-5)
      else:
        self.assertAllClose(linear_update, self.evaluate(linear))
        self.assertAllClose(expected_out, out)

  @test_util.run_deprecated_v1
  def testApplyAdagrad(self):
    for (dtype, use_gpu) in itertools.product(
        [np.float16, np.float32, np.float64], [False, True]):
      x = np.arange(100).astype(dtype)
      y = np.arange(1, 101).astype(dtype)
      lr = np.array(2.0).astype(dtype)
      grad = np.arange(100).astype(dtype)
      self._testTypesForAdagrad(x, y, lr, grad, use_gpu)

  @test_util.run_deprecated_v1
  def testApplyFtrl(self):
    for dtype in [np.float16, np.float32, np.float64]:
      x = np.arange(100).astype(dtype)
      y = np.arange(1, 101).astype(dtype)
      z = np.arange(102, 202).astype(dtype)
      lr = np.array(2.0).astype(dtype)
      l1 = np.array(3.0).astype(dtype)
      l2 = np.array(4.0).astype(dtype)
      grad = np.arange(100).astype(dtype)
      self._testTypesForFtrl(x, y, z, lr, grad, use_gpu=False, l1=l1, l2=l2)

  def _testTypesForSparseAdagrad(self, x, y, lr, grad, indices):
    self.setUp()
    with self.session(use_gpu=False):
      var = variables.VariableV1(x)
      accum = variables.VariableV1(y)
      variables.global_variables_initializer().run()

      self.assertAllCloseAccordingToType(x, self.evaluate(var))
      sparse_apply_adagrad = training_ops.sparse_apply_adagrad(
          var, accum, lr, grad,
          constant_op.constant(indices, self._toType(indices.dtype)))
      out = self.evaluate(sparse_apply_adagrad)
      self.assertShapeEqual(out, sparse_apply_adagrad)

      for (i, index) in enumerate(indices):
        self.assertAllCloseAccordingToType(
            x[index] - lr * grad[i] * (y[index] + grad[i] * grad[i])**(-0.5),
            self.evaluate(var)[index])
        self.assertAllCloseAccordingToType(y[index] + grad[i] * grad[i],
                                           self.evaluate(accum)[index])

  def _testTypesForSparseFtrl(self,
                              x,
                              y,
                              z,
                              lr,
                              grad,
                              indices,
                              l1=0.0,
                              l2=0.0,
                              lr_power=-0.5):
    self.setUp()
    with self.session(use_gpu=False):
      var = variables.VariableV1(x)
      accum = variables.VariableV1(y)
      linear = variables.VariableV1(z)
      variables.global_variables_initializer().run()

      self.assertAllCloseAccordingToType(x, self.evaluate(var))
      sparse_apply_ftrl = training_ops.sparse_apply_ftrl(
          var,
          accum,
          linear,
          grad,
          constant_op.constant(indices, self._toType(indices.dtype)),
          lr,
          l1,
          l2,
          lr_power=lr_power)
      out = self.evaluate(sparse_apply_ftrl)
      self.assertShapeEqual(out, sparse_apply_ftrl)

      for (i, index) in enumerate(indices):
        self.assertAllCloseAccordingToType(
            x[index] - lr * grad[i] * (y[index] + grad[i] * grad[i])**
            (lr_power),
            self.evaluate(var)[index])
        self.assertAllCloseAccordingToType(y[index] + grad[i] * grad[i],
                                           self.evaluate(accum)[index])

  @test_util.run_deprecated_v1
  def testSparseApplyAdagrad(self):
    for (dtype, index_type) in itertools.product(
        [np.float16, np.float32, np.float64], [np.int32, np.int64]):
      x_val = [np.arange(10), np.arange(10, 20), np.arange(20, 30)]
      y_val = [np.arange(1, 11), np.arange(11, 21), np.arange(21, 31)]
      x = np.array(x_val).astype(dtype)
      y = np.array(y_val).astype(dtype)
      lr = np.array(2.0).astype(dtype)
      grad_val = [np.arange(10), np.arange(10)]
      grad = np.array(grad_val).astype(dtype)
      indices = np.array([0, 2]).astype(index_type)
      self._testTypesForSparseAdagrad(x, y, lr, grad, indices)

  @test_util.run_deprecated_v1
  def testSparseApplyAdagradDim1(self):
    for (dtype, index_type) in itertools.product(
        [np.float16, np.float32, np.float64], [np.int32, np.int64]):
      x_val = [[1.0], [2.0], [3.0]]
      y_val = [[4.0], [5.0], [6.0]]
      x = np.array(x_val).astype(dtype)
      y = np.array(y_val).astype(dtype)
      lr = np.array(2.0).astype(dtype)
      grad_val = [[1.5], [2.5]]
      grad = np.array(grad_val).astype(dtype)
      indices = np.array([0, 2]).astype(index_type)
      self._testTypesForSparseAdagrad(x, y, lr, grad, indices)

  @test_util.run_deprecated_v1
  def testSparseApplyFtrlDim1(self):
    for (dtype, index_type) in itertools.product(
        [np.float16, np.float32, np.float64], [np.int32, np.int64]):
      x_val = [[0.0], [0.0], [0.0]]
      y_val = [[4.0], [5.0], [6.0]]
      z_val = [[0.0], [0.0], [0.0]]
      x = np.array(x_val).astype(dtype)
      y = np.array(y_val).astype(dtype)
      z = np.array(z_val).astype(dtype)
      lr = np.array(2.0).astype(dtype)
      grad_val = [[1.5], [2.5]]
      grad = np.array(grad_val).astype(dtype)
      indices = np.array([0, 2]).astype(index_type)
      self._testTypesForSparseFtrl(x, y, z, lr, grad, indices)

  @test_util.run_deprecated_v1
  def testApplyAdam(self):
    for dtype, use_gpu in itertools.product(
        [np.float16, np.float32, np.float64], [False, True]):
      var = np.arange(100).astype(dtype)
      m = np.arange(1, 101).astype(dtype)
      v = np.arange(101, 201).astype(dtype)
      grad = np.arange(100).astype(dtype)
      self._testTypesForAdam(var, m, v, grad, use_gpu)

  def _testTypesForAdam(self, var, m, v, grad, use_gpu):
    self.setUp()
    with self.session(use_gpu=use_gpu):
      var_t = variables.VariableV1(var)
      m_t = variables.VariableV1(m)
      v_t = variables.VariableV1(v)

      t = 1
      beta1 = np.array(0.9, dtype=var.dtype)
      beta2 = np.array(0.999, dtype=var.dtype)
      beta1_power = beta1**t
      beta2_power = beta2**t
      lr = np.array(0.001, dtype=var.dtype)
      epsilon = np.array(1e-8, dtype=var.dtype)
      beta1_t = constant_op.constant(beta1, self._toType(var.dtype), [])
      beta2_t = constant_op.constant(beta2, self._toType(var.dtype), [])
      beta1_power_t = variables.VariableV1(beta1_power)
      beta2_power_t = variables.VariableV1(beta2_power)
      lr_t = constant_op.constant(lr, self._toType(var.dtype), [])
      epsilon_t = constant_op.constant(epsilon, self._toType(var.dtype), [])
      variables.global_variables_initializer().run()

      self.assertAllCloseAccordingToType(var, self.evaluate(var_t))
      new_var, _, _ = self._adamUpdateNumpy(var, grad, t, m, v, lr, beta1,
                                            beta2, epsilon)
      apply_adam = training_ops.apply_adam(var_t, m_t, v_t, beta1_power_t,
                                           beta2_power_t, lr_t, beta1_t,
                                           beta2_t, epsilon_t, grad)
      out = self.evaluate(apply_adam)
      self.assertShapeEqual(out, apply_adam)
      self.assertAllCloseAccordingToType(new_var, out)

  def _adamUpdateNumpy(self, param, g_t, t, m, v, alpha, beta1, beta2, epsilon):
    alpha_t = alpha * np.sqrt(1 - beta2**t) / (1 - beta1**t)

    m_t = beta1 * m + (1 - beta1) * g_t
    v_t = beta2 * v + (1 - beta2) * g_t * g_t

    param_t = param - alpha_t * m_t / (np.sqrt(v_t) + epsilon)
    return param_t, m_t, v_t


if __name__ == '__main__':
  googletest.main()
