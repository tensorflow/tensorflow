"""Tests for tensorflow.learning.training_ops."""

import itertools

import tensorflow.python.platform

import numpy as np

from tensorflow.python.framework import types
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import training_ops


class TrainingOpsTest(TensorFlowTestCase):

  def _toType(self, dtype):
    if dtype == np.float32:
      return types.float32
    elif dtype == np.float64:
      return types.float64
    elif dtype == np.int32:
      return types.int32
    elif dtype == np.int64:
      return types.int64
    else:
      assert False, (dtype)

  def _testTypes(self, x, alpha, delta, use_gpu=None):
    self.setUp()
    with self.test_session(use_gpu=use_gpu):
      var = variables.Variable(x)
      variables.initialize_all_variables().run()
      self.assertAllEqual(x, var.eval())
      apply_sgd = training_ops.apply_gradient_descent(var, alpha, delta)
      out = apply_sgd.eval()
      self.assertShapeEqual(out, apply_sgd)
      self.assertAllEqual(x - alpha * delta, out)

  def testApplyGradientDescent(self):
    for (dtype, use_gpu) in itertools.product(
        [np.float32, np.float64], [False, True]):
      x = np.arange(100).astype(dtype)
      alpha = np.array(2.0).astype(dtype)
      delta = np.arange(100).astype(dtype)
      self._testTypes(x, alpha, delta, use_gpu)

  def _testTypesForAdagrad(self, x, y, lr, grad, use_gpu=None):
    self.setUp()
    with self.test_session(use_gpu=use_gpu):
      var = variables.Variable(x)
      accum = variables.Variable(y)
      variables.initialize_all_variables().run()

      self.assertAllEqual(x, var.eval())
      apply_adagrad = training_ops.apply_adagrad(var, accum, lr, grad)
      out = apply_adagrad.eval()
      self.assertShapeEqual(out, apply_adagrad)
      self.assertAllClose(
          x - lr * grad * (y + grad * grad) ** (-0.5), out)
      self.assertAllEqual(y + grad * grad, accum.eval())

  def testApplyAdagrad(self):
    for (dtype, use_gpu) in itertools.product(
        [np.float32, np.float64], [False, True]):
      x = np.arange(100).astype(dtype)
      y = np.arange(1, 101).astype(dtype)
      lr = np.array(2.0).astype(dtype)
      grad = np.arange(100).astype(dtype)
      self._testTypesForAdagrad(x, y, lr, grad, use_gpu)

  def _testTypesForSparseAdagrad(self, x, y, lr, grad, indices):
    self.setUp()
    with self.test_session(use_gpu=False):
      var = variables.Variable(x)
      accum = variables.Variable(y)
      variables.initialize_all_variables().run()

      self.assertAllEqual(x, var.eval())
      sparse_apply_adagrad = training_ops.sparse_apply_adagrad(
          var, accum, lr, grad,
          constant_op.constant(indices, self._toType(indices.dtype)))
      out = sparse_apply_adagrad.eval()
      self.assertShapeEqual(out, sparse_apply_adagrad)

      for (i, index) in enumerate(indices):
        self.assertAllClose(
            x[index] - lr * grad[i] * (y[index] + grad[i] * grad[i]) ** (-0.5),
            var.eval()[index])
        self.assertAllEqual(y[index] + grad[i] * grad[i], accum.eval()[index])

  def testSparseApplyAdagrad(self):
    for (dtype, index_type) in itertools.product(
        [np.float32, np.float64], [np.int32, np.int64]):
      x_val = [range(10), range(10, 20), range(20, 30)]
      y_val = [range(1, 11), range(11, 21), range(21, 31)]
      x = np.array(x_val).astype(dtype)
      y = np.array(y_val).astype(dtype)
      lr = np.array(2.0).astype(dtype)
      grad_val = [range(10), range(10)]
      grad = np.array(grad_val).astype(dtype)
      indices = np.array([0, 2]).astype(index_type)
      self._testTypesForSparseAdagrad(x, y, lr, grad, indices)

  def testApplyAdam(self):
    for dtype, use_gpu in itertools.product(
        [np.float32, np.float64], [False, True]):
      var = np.arange(100).astype(dtype)
      m = np.arange(1, 101).astype(dtype)
      v = np.arange(101, 201).astype(dtype)
      grad = np.arange(100).astype(dtype)
      self._testTypesForAdam(var, m, v, grad, use_gpu)

  def _testTypesForAdam(self, var, m, v, grad, use_gpu):
    self.setUp()
    with self.test_session(use_gpu=use_gpu):
      var_t = variables.Variable(var)
      m_t = variables.Variable(m)
      v_t = variables.Variable(v)

      t = 1
      beta1 = np.array(0.9, dtype=var.dtype)
      beta2 = np.array(0.999, dtype=var.dtype)
      beta1_power = beta1**t
      beta2_power = beta2**t
      lr = np.array(0.001, dtype=var.dtype)
      epsilon = np.array(1e-8, dtype=var.dtype)
      beta1_t = constant_op.constant(beta1, self._toType(var.dtype), [])
      beta2_t = constant_op.constant(beta2, self._toType(var.dtype), [])
      beta1_power_t = variables.Variable(beta1_power)
      beta2_power_t = variables.Variable(beta2_power)
      lr_t = constant_op.constant(lr, self._toType(var.dtype), [])
      epsilon_t = constant_op.constant(epsilon, self._toType(var.dtype), [])
      variables.initialize_all_variables().run()

      self.assertAllEqual(var, var_t.eval())
      new_var, _, _ = self._adamUpdateNumpy(var, grad, t, m, v,
                                            lr, beta1, beta2, epsilon)
      apply_adam = training_ops.apply_adam(var_t, m_t, v_t, beta1_power_t,
                                           beta2_power_t, lr_t,
                                           beta1_t, beta2_t, epsilon_t, grad)
      out = apply_adam.eval()
      self.assertShapeEqual(out, apply_adam)
      self.assertAllClose(new_var, out)

  def _adamUpdateNumpy(self, param, g_t, t, m, v, alpha, beta1,
                       beta2, epsilon):
    alpha_t = alpha * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

    m_t = beta1 * m + (1 - beta1) * g_t
    v_t = beta2 * v + (1 - beta2) * g_t * g_t

    param_t = param - alpha_t * m_t / (np.sqrt(v_t) + epsilon)
    return param_t, m_t, v_t

if __name__ == '__main__':
  googletest.main()
