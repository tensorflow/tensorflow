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
"""Functional tests for binary coefficient-wise operations."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

_ADD = lambda x, y: x + y
_SUB = lambda x, y: x - y
_MUL = lambda x, y: x * y
_POW = lambda x, y: x**y
_TRUEDIV = lambda x, y: x / y
_FLOORDIV = lambda x, y: x // y
_MOD = lambda x, y: x % y

# x and y must be numpy array object
np_xlogy = lambda x, y: x * np.log(y)
np_xlog1py = lambda x, y: x * np.log1p(y)


# TODO(zongheng): it'd be great to factor out this function and various random
# SparseTensor gen funcs.
def _sparsify(x, thresh=0.5, index_dtype=np.int64):
  x[x < thresh] = 0

  non_zero = np.where(x)
  x_indices = np.vstack(non_zero).astype(index_dtype).T
  x_values = x[non_zero]
  x_shape = x.shape

  return sparse_tensor.SparseTensor(
      indices=x_indices, values=x_values, dense_shape=x_shape), x_values


def _default_tolerance(dtype):
  """Returns a sensible default tolerance for comparing results of a given type.

  Args:
    dtype: A datatype.
  """
  if dtype == np.float16:
    return 5e-3
  elif dtype in (np.float32, np.complex64):
    return 1e-3
  elif dtype in (np.float64, np.complex128):
    return 1e-5
  else:
    return None  # Fail fast for unexpected types


class BinaryOpTest(test.TestCase):

  def _compareCpu(self, x, y, np_func, tf_func, also_compare_variables=False):
    np_ans = np_func(x, y)
    with test_util.force_cpu():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_cpu = self.evaluate(out)
      # Test that the op takes precedence over numpy operators.
      np_left = self.evaluate(tf_func(x, iny))
      np_right = self.evaluate(tf_func(inx, y))

      if also_compare_variables:
        var_x = variables.Variable(x)
        var_y = variables.Variable(y)
        self.evaluate(variables.global_variables_initializer())
        print(type(x), type(y), type(var_x), type(var_y))
        print(type(tf_func(x, var_y)), type(tf_func(var_x, y)))
        np_var_left = self.evaluate(tf_func(x, var_y))
        np_var_right = self.evaluate(tf_func(var_x, y))

    if np_ans.dtype != np.object_:
      self.assertAllClose(np_ans, tf_cpu)
      self.assertAllClose(np_ans, np_left)
      self.assertAllClose(np_ans, np_right)
      if also_compare_variables:
        self.assertAllClose(np_ans, np_var_left)
        self.assertAllClose(np_ans, np_var_right)
    self.assertShapeEqual(np_ans, out)

  _GRAD_TOL = {
      dtypes_lib.float16: 1e-3,
      dtypes_lib.float32: 1e-3,
      dtypes_lib.complex64: 1e-2,
      dtypes_lib.float64: 1e-5,
      dtypes_lib.complex128: 1e-4
  }

  def _compareGradientX(self,
                        x,
                        y,
                        np_func,
                        tf_func,
                        numeric_gradient_type=None):
    z = np_func(x, y)
    zs = list(z.shape)
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      if x.dtype in (np.float32, np.float64):
        out = 1.1 * tf_func(inx, iny)
      else:
        out = tf_func(inx, iny)
      xs = list(x.shape)
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, xs, out, zs, x_init_value=x)
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = ops.convert_to_tensor(xf)
        inyf = ops.convert_to_tensor(yf)
        outf = tf_func(inxf, inyf)
        _, jacob_n = gradient_checker.compute_gradient(
            inxf, xs, outf, zs, x_init_value=xf, delta=1e-3)
        jacob_n = jacob_n.astype(x.dtype)
      tol = self._GRAD_TOL[dtypes_lib.as_dtype(x.dtype)]
      self.assertAllClose(jacob_t, jacob_n, rtol=tol, atol=tol)

  def _compareGradientY(self,
                        x,
                        y,
                        np_func,
                        tf_func,
                        numeric_gradient_type=None):
    z = np_func(x, y)
    zs = list(z.shape)
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      if x.dtype in (np.float32, np.float64):
        out = 1.1 * tf_func(inx, iny)
      else:
        out = tf_func(inx, iny)
      ys = list(np.shape(y))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          iny, ys, out, zs, x_init_value=y)
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = ops.convert_to_tensor(xf)
        inyf = ops.convert_to_tensor(yf)
        outf = tf_func(inxf, inyf)
        _, jacob_n = gradient_checker.compute_gradient(
            inyf, ys, outf, zs, x_init_value=yf)
        jacob_n = jacob_n.astype(x.dtype)
    tol = self._GRAD_TOL[dtypes_lib.as_dtype(x.dtype)]
    self.assertAllClose(jacob_t, jacob_n, rtol=tol, atol=tol)

  def _compareGpu(self, x, y, np_func, tf_func):
    np_ans = np_func(x, y)
    with test_util.use_gpu():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_gpu = self.evaluate(out)
    self.assertAllClose(np_ans, tf_gpu)
    self.assertShapeEqual(np_ans, out)
    # TODO(zhifengc/ke): make gradient checker work on GPU.

  def _compareBoth(self, x, y, np_func, tf_func, also_compare_variables=False):
    self._compareCpu(x, y, np_func, tf_func, also_compare_variables)
    if x.dtype in (np.float16, np.float32, np.float64, np.complex64,
                   np.complex128):
      if tf_func not in (_FLOORDIV, math_ops.floordiv, math_ops.zeta,
                         math_ops.polygamma):
        self._compareGradientX(x, y, np_func, tf_func)
        self._compareGradientY(x, y, np_func, tf_func)
      if tf_func in (math_ops.zeta, math_ops.polygamma):
        # These methods only support gradients in the second parameter
        self._compareGradientY(x, y, np_func, tf_func)
      self._compareGpu(x, y, np_func, tf_func)

  @test_util.run_deprecated_v1
  def testFloatBasic(self):
    x = np.linspace(-5, 20, 15).reshape(1, 3, 5).astype(np.float32)  # pylint: disable=too-many-function-args
    y = np.linspace(20, -5, 15).reshape(1, 3, 5).astype(np.float32)  # pylint: disable=too-many-function-args
    self._compareBoth(x, y, np.add, math_ops.add, also_compare_variables=True)
    self._compareBoth(x, y, np.subtract, math_ops.subtract)
    self._compareBoth(x, y, np.multiply, math_ops.multiply)
    self._compareBoth(x, y + 0.1, np.true_divide, math_ops.truediv)
    self._compareBoth(x, y + 0.1, np.floor_divide, math_ops.floordiv)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    self._compareBoth(x, y + 0.1, np.true_divide, _TRUEDIV)
    self._compareBoth(x, y + 0.1, np.floor_divide, _FLOORDIV)
    self._compareBoth(x, y, np.arctan2, math_ops.atan2)
    x1 = np.random.randn(5, 6).astype(np.float32)
    x2 = np.random.randn(5, 6).astype(np.float32)
    # Remove tiny values--atan2 gradients are flaky near the origin.
    x1[np.abs(x1) < 0.05] = 0.05 * np.sign(x1[np.abs(x1) < 0.05])
    x2[np.abs(x2) < 0.05] = 0.05 * np.sign(x2[np.abs(x2) < 0.05])
    self._compareBoth(x1, x2, np.arctan2, math_ops.atan2)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top
      a_pos_small = np.linspace(0.1, 2, 15).reshape(1, 3, 5).astype(np.float32)  # pylint: disable=too-many-function-args
      x_pos_small = np.linspace(0.1, 10, 15).reshape(1, 3, 5).astype(np.float32)  # pylint: disable=too-many-function-args
      self._compareBoth(a_pos_small, x_pos_small, special.gammainc,
                        math_ops.igamma)
      self._compareBoth(a_pos_small, x_pos_small, special.gammaincc,
                        math_ops.igammac)
      # Need x > 1
      self._compareBoth(x_pos_small + 1, a_pos_small, special.zeta,
                        math_ops.zeta)
      n_small = np.arange(0, 15).reshape(1, 3, 5).astype(np.float32)
      self._compareBoth(n_small, x_pos_small, special.polygamma,
                        math_ops.polygamma)
    except ImportError as e:
      tf_logging.warn("Cannot test special functions: %s" % str(e))

  @test_util.run_deprecated_v1
  def testFloatDifferentShapes(self):
    x = np.array([1, 2, 3, 4]).reshape(2, 2).astype(np.float32)
    y = np.array([1, 2]).reshape(2, 1).astype(np.float32)
    with self.cached_session() as sess:
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      s = math_ops.reduce_sum(inx * iny)
      gx, gy = sess.run(gradients_impl.gradients(s, [inx, iny]))
    # gx is simply the broadcasted y
    self.assertAllEqual(gx,
                        np.array([1, 1, 2, 2]).reshape(2, 2).astype(np.float32))
    # gy is x's column summed up
    self.assertAllEqual(gy, np.array([3, 7]).reshape(2, 1).astype(np.float32))

  def testFloatVariableOverload(self):
    x = np.array([1, 2, 3, 4]).reshape(2, 2).astype(np.int32)
    y = np.array([1, 2]).reshape(2, 1).astype(np.int32)
    var_x = variables.Variable(x)
    var_y = variables.Variable(y)

    self.evaluate([var_x.initializer, var_y.initializer])
    left_result = self.evaluate(var_x * y)
    right_result = self.evaluate(x * var_y)

    np_result = x * y
    self.assertAllEqual(np_result, left_result)
    self.assertAllEqual(np_result, right_result)

  def testBFloat16Basic(self):
    bfloat16 = dtypes_lib.bfloat16.as_numpy_dtype
    x = np.linspace(-20, 20, 10).reshape(1, 2, 5).astype(bfloat16)  # pylint: disable=too-many-function-args
    # y cannot be zero
    y = np.linspace(-20, 20, 10).reshape(1, 2, 5).astype(bfloat16)  # pylint: disable=too-many-function-args
    self._compareCpu(x, y, np.true_divide, math_ops.xdivy)
    self._compareCpu(x, y, np_xlogy, math_ops.xlogy)
    self._compareCpu(x, y, np_xlog1py, math_ops.xlog1py)

  @test_util.run_deprecated_v1
  def testDoubleBasic(self):
    x = np.linspace(-5, 20, 15).reshape(1, 3, 5).astype(np.float64)  # pylint: disable=too-many-function-args
    y = np.linspace(20, -5, 15).reshape(1, 3, 5).astype(np.float64)  # pylint: disable=too-many-function-args
    self._compareBoth(x, y, np.add, math_ops.add)
    self._compareBoth(x, y, np.subtract, math_ops.subtract)
    self._compareBoth(x, y, np.multiply, math_ops.multiply)
    self._compareBoth(x, y + 0.1, np.true_divide, math_ops.truediv)
    self._compareBoth(x, y + 0.1, np.floor_divide, math_ops.floordiv)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    self._compareBoth(x, y + 0.1, np.true_divide, _TRUEDIV)
    self._compareBoth(x, y + 0.1, np.floor_divide, _FLOORDIV)
    self._compareBoth(x, y, np.arctan2, math_ops.atan2)
    x1 = np.random.randn(7, 4).astype(np.float64)
    x2 = np.random.randn(7, 4).astype(np.float64)
    # Remove tiny values--atan2 gradients are flaky near the origin.
    x1[np.abs(x1) < 0.5] = 0.5 * np.sign(x1[np.abs(x1) < 0.5])
    x2[np.abs(x2) < 0.5] = 0.5 * np.sign(x2[np.abs(x2) < 0.5])
    self._compareBoth(x1, x2, np.arctan2, math_ops.atan2)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top
      a_pos_small = np.linspace(0.1, 2, 15).reshape(1, 3, 5).astype(np.float32)  # pylint: disable=too-many-function-args
      x_pos_small = np.linspace(0.1, 10, 15).reshape(1, 3, 5).astype(np.float32)  # pylint: disable=too-many-function-args
      self._compareBoth(a_pos_small, x_pos_small, special.gammainc,
                        math_ops.igamma)
      self._compareBoth(a_pos_small, x_pos_small, special.gammaincc,
                        math_ops.igammac)
    except ImportError as e:
      tf_logging.warn("Cannot test special functions: %s" % str(e))

  def testBfloat16Basic(self):
    bf16_np = dtypes_lib.bfloat16.as_numpy_dtype
    x = np.linspace(-5, 20, 15).reshape(1, 3, 5).astype(bf16_np)  # pylint: disable=too-many-function-args
    y = np.linspace(20, -5, 15).reshape(1, 3, 5).astype(bf16_np)  # pylint: disable=too-many-function-args
    self._compareBoth(x, y, np.add, math_ops.add)
    self._compareBoth(x, y, np.subtract, math_ops.subtract)
    self._compareBoth(x, y, np.multiply, math_ops.multiply)
    self._compareBoth(x, bf16_np(y + 0.1), np.true_divide, math_ops.truediv)
    self._compareBoth(x, bf16_np(y + 0.1), np.floor_divide, math_ops.floordiv)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    self._compareBoth(x, bf16_np(y + 0.1), np.true_divide, _TRUEDIV)
    self._compareBoth(x, bf16_np(y + 0.1), np.floor_divide, _FLOORDIV)
    self._compareBoth(x, y, np.maximum, math_ops.maximum)
    self._compareBoth(x, y, np.minimum, math_ops.minimum)

  def testUint8Basic(self):
    x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.uint8)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.uint8)
    self._compareBoth(x, y, np.add, math_ops.add)
    self._compareBoth(x, y, np.subtract, math_ops.subtract)
    self._compareBoth(x, y, np.subtract, _SUB)

  def testInt8Basic(self):
    x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.int8)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.int8)
    self._compareBoth(x, y, np.subtract, math_ops.subtract)
    self._compareBoth(x, y, np.multiply, math_ops.multiply)
    self._compareBoth(x, y, np.true_divide, math_ops.truediv)
    self._compareBoth(x, y, np.floor_divide, math_ops.floordiv)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    self._compareBoth(x, y, np.true_divide, _TRUEDIV)
    self._compareBoth(x, y, np.floor_divide, _FLOORDIV)

  def testInt16Basic(self):
    x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.int16)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.int16)
    self._compareBoth(x, y, np.subtract, math_ops.subtract)
    self._compareBoth(x, y, np.multiply, math_ops.multiply)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)

  def testUint16Basic(self):
    x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.uint16)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.uint16)
    self._compareBoth(x, y, np.subtract, math_ops.subtract)
    self._compareBoth(x, y, np.multiply, math_ops.multiply)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    self._compareBoth(x, y, np.true_divide, math_ops.truediv)
    self._compareBoth(x, y, np.floor_divide, math_ops.floordiv)
    self._compareBoth(x, y, np.true_divide, _TRUEDIV)
    self._compareBoth(x, y, np.floor_divide, _FLOORDIV)

  def testInt32Basic(self):
    x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.int32)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.int32)
    self._compareBoth(x, y, np.add, math_ops.add)
    self._compareBoth(x, y, np.subtract, math_ops.subtract)
    self._compareBoth(x, y, np.multiply, math_ops.multiply)
    self._compareBoth(x, y, np.true_divide, math_ops.truediv)
    self._compareBoth(x, y, np.floor_divide, math_ops.floordiv)
    self._compareBoth(x, y, np.mod, math_ops.mod)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    self._compareBoth(x, y, np.true_divide, _TRUEDIV)
    self._compareBoth(x, y, np.floor_divide, _FLOORDIV)
    self._compareBoth(x, y, np.mod, _MOD)
    # _compareBoth tests on GPU only for floating point types, so test
    # _MOD for int32 on GPU by calling _compareGpu
    self._compareGpu(x, y, np.mod, _MOD)

  def testUint32Basic(self):
    x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.uint32)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.uint32)
    self._compareBoth(x, y, np.add, math_ops.add_v2)
    self._compareBoth(x, y, np.true_divide, math_ops.truediv)
    self._compareBoth(x, y, np.floor_divide, math_ops.floordiv)
    self._compareBoth(x, y, np.true_divide, _TRUEDIV)
    self._compareBoth(x, y, np.floor_divide, _FLOORDIV)

  def testInt64Basic(self):
    x = np.arange(1 << 40, 13 << 40, 2 << 40).reshape(1, 3, 2).astype(np.int64)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.int64)
    self._compareBoth(x, y, np.subtract, math_ops.subtract)
    self._compareBoth(x, y, np.multiply, math_ops.multiply)
    self._compareBoth(x, y, np.true_divide, math_ops.truediv)
    self._compareBoth(x, y, np.floor_divide, math_ops.floordiv)
    self._compareBoth(x, y, np.mod, math_ops.mod)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    self._compareBoth(x, y, np.true_divide, _TRUEDIV)
    self._compareBoth(x, y, np.floor_divide, _FLOORDIV)
    self._compareBoth(x, y, np.mod, _MOD)

  def testUint64Basic(self):
    x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.uint32)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.uint32)
    self._compareBoth(x, y, np.true_divide, math_ops.truediv)
    self._compareBoth(x, y, np.floor_divide, math_ops.floordiv)
    self._compareBoth(x, y, np.true_divide, _TRUEDIV)
    self._compareBoth(x, y, np.floor_divide, _FLOORDIV)

  @test_util.run_deprecated_v1
  def testComplex64Basic(self):
    x = (1 + 1j) * np.linspace(-10, 10, 6).reshape(1, 3, 2).astype(  # pylint: disable=too-many-function-args
        np.complex64)
    y = (1 + 1j) * np.linspace(20, -20, 6).reshape(1, 3, 2).astype(  # pylint: disable=too-many-function-args
        np.complex64)
    self._compareBoth(x, y, np.add, math_ops.add)
    self._compareBoth(x, y, np.subtract, math_ops.subtract)
    self._compareBoth(x, y, np.multiply, math_ops.multiply)
    self._compareBoth(x, y + 0.1, np.true_divide, math_ops.truediv)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    self._compareBoth(x, y + 0.1, np.true_divide, _TRUEDIV)

  @test_util.run_deprecated_v1
  def testComplex128Basic(self):
    x = (1 + 1j) * np.linspace(-10, 10, 6).reshape(1, 3, 2).astype(  # pylint: disable=too-many-function-args
        np.complex128)
    y = (1 + 1j) * np.linspace(20, -20, 6).reshape(1, 3, 2).astype(  # pylint: disable=too-many-function-args
        np.complex128)
    self._compareBoth(x, y, np.add, math_ops.add)
    self._compareBoth(x, y, np.subtract, math_ops.subtract)
    self._compareBoth(x, y, np.multiply, math_ops.multiply)
    self._compareBoth(x, y + 0.1, np.true_divide, math_ops.truediv)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    self._compareBoth(x, y + 0.1, np.true_divide, _TRUEDIV)

  def testStringComparison(self):
    x = np.array([["abc", "bh"], ["c", ""]])
    y = np.array([["abc", "bh"], ["def", "hi"]])
    with test_util.force_cpu():
      cmp_eq = math_ops.equal(x, y)
      cmp_not_eq = math_ops.not_equal(x, y)
      values = self.evaluate([cmp_eq, cmp_not_eq])
      self.assertAllEqual([[True, True], [False, False]], values[0])
      self.assertAllEqual([[False, False], [True, True]], values[1])

  def testString(self):
    x = np.array([["x_0_0", "x_0_1", "x_0_2"], ["x_1_0", "x_1_1", "x_1_2"],
                  ["x_2_0", "x_2_1", "x_2_2"]],
                 dtype=np.object_)
    y = np.array([["y_0_0", "y_0_1", "y_0_2"], ["y_1_0", "y_1_1", "y_1_2"],
                  ["y_2_0", "y_2_1", "y_2_2"]],
                 dtype=np.object_)
    z = np.array([["z_0", "z_1", "z_2"]], dtype=np.object_)
    w = np.array("w", dtype=np.object_)
    self._compareCpu(x, y, _ADD, _ADD)
    self._compareCpu(x, z, _ADD, _ADD)
    self._compareCpu(x, w, _ADD, _ADD)
    self._compareCpu(z, w, _ADD, _ADD)

  def _compareBCast(self, xs, ys, dtype, np_func, tf_func):
    if dtype in (np.complex64, np.complex128):
      x = (1 + np.linspace(0, 2 + 3j, np.prod(xs))).astype(dtype).reshape(xs)
      y = (1 + np.linspace(0, 2 - 2j, np.prod(ys))).astype(dtype).reshape(ys)
    else:
      x = (1 + np.linspace(0, 5, np.prod(xs))).astype(dtype).reshape(xs)
      y = (1 + np.linspace(0, 5, np.prod(ys))).astype(dtype).reshape(ys)
    self._compareCpu(x, y, np_func, tf_func)
    if x.dtype in (np.float16, np.float32, np.float64):
      # TODO(aselle): Make the test work for dtypes:
      #     (np.complex64, np.complex128).
      if tf_func not in (_FLOORDIV, math_ops.floordiv):
        if x.dtype == np.float16:
          # Compare fp16 theoretical gradients to fp32 numerical gradients,
          # since fp16 numerical gradients are too imprecise unless great
          # care is taken with choosing the inputs and the delta. This is
          # a weaker check (in particular, it does not test the op itself,
          # only its gradient), but it's much better than nothing.
          self._compareGradientX(x, y, np_func, tf_func, np.float64)
          self._compareGradientY(x, y, np_func, tf_func, np.float64)
        else:
          self._compareGradientX(x, y, np_func, tf_func)
          self._compareGradientY(x, y, np_func, tf_func)
      self._compareGpu(x, y, np_func, tf_func)

  # TODO(josh11b,vrv): Refactor this to use parameterized tests.
  def _testBCastByFunc(self, funcs, xs, ys):
    dtypes = [
        np.float16,
        np.float32,
        np.float64,
        np.int32,
        np.int64,
        np.complex64,
        np.complex128,
    ]
    for dtype in dtypes:
      for (np_func, tf_func) in funcs:
        if (dtype in (np.complex64, np.complex128) and
            tf_func in (_FLOORDIV, math_ops.floordiv)):
          continue  # floordiv makes no sense for complex numbers
        self._compareBCast(xs, ys, dtype, np_func, tf_func)
        self._compareBCast(ys, xs, dtype, np_func, tf_func)

  def _testBCastA(self, xs, ys):
    funcs = [
        (np.add, math_ops.add),
        (np.add, _ADD),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def _testBCastB(self, xs, ys):
    funcs = [
        (np.subtract, math_ops.subtract),
        (np.subtract, _SUB),
        (np.power, math_ops.pow),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def _testBCastC(self, xs, ys):
    funcs = [
        (np.multiply, math_ops.multiply),
        (np.multiply, _MUL),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def _testBCastD(self, xs, ys):
    funcs = [
        (np.true_divide, math_ops.truediv),
        (np.floor_divide, math_ops.floordiv),
        (np.true_divide, _TRUEDIV),
        (np.floor_divide, _FLOORDIV),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  @test_util.run_deprecated_v1
  def testBCast_0A(self):
    self._testBCastA([1, 3, 2], [1])

  @test_util.run_deprecated_v1
  def testBCast_0B(self):
    self._testBCastB([1, 3, 2], [1])

  @test_util.run_deprecated_v1
  def testBCast_0C(self):
    self._testBCastC([1, 3, 2], [1])

  @test_util.run_deprecated_v1
  def testBCast_0D(self):
    self._testBCastD([1, 3, 2], [1])

  @test_util.run_deprecated_v1
  def testBCast_1A(self):
    self._testBCastA([1, 3, 2], [2])

  @test_util.run_deprecated_v1
  def testBCast_1B(self):
    self._testBCastB([1, 3, 2], [2])

  @test_util.run_deprecated_v1
  def testBCast_1C(self):
    self._testBCastC([1, 3, 2], [2])

  @test_util.run_deprecated_v1
  def testBCast_1D(self):
    self._testBCastD([1, 3, 2], [2])

  @test_util.run_deprecated_v1
  def testBCast_2A(self):
    self._testBCastA([1, 3, 2], [3, 2])

  @test_util.run_deprecated_v1
  def testBCast_2B(self):
    self._testBCastB([1, 3, 2], [3, 2])

  @test_util.run_deprecated_v1
  def testBCast_2C(self):
    self._testBCastC([1, 3, 2], [3, 2])

  @test_util.run_deprecated_v1
  def testBCast_2D(self):
    self._testBCastD([1, 3, 2], [3, 2])

  @test_util.run_deprecated_v1
  def testBCast_3A(self):
    self._testBCastA([1, 3, 2], [3, 1])

  @test_util.run_deprecated_v1
  def testBCast_3B(self):
    self._testBCastB([1, 3, 2], [3, 1])

  @test_util.run_deprecated_v1
  def testBCast_3C(self):
    self._testBCastC([1, 3, 2], [3, 1])

  @test_util.run_deprecated_v1
  def testBCast_3D(self):
    self._testBCastD([1, 3, 2], [3, 1])

  @test_util.run_deprecated_v1
  def testBCast_4A(self):
    self._testBCastA([1, 3, 2], [1, 3, 2])

  @test_util.run_deprecated_v1
  def testBCast_4B(self):
    self._testBCastB([1, 3, 2], [1, 3, 2])

  @test_util.run_deprecated_v1
  def testBCast_4C(self):
    self._testBCastC([1, 3, 2], [1, 3, 2])

  @test_util.run_deprecated_v1
  def testBCast_4D(self):
    self._testBCastD([1, 3, 2], [1, 3, 2])

  @test_util.run_deprecated_v1
  def testBCast_5A(self):
    self._testBCastA([1, 3, 2], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_5B(self):
    self._testBCastB([1, 3, 2], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_5C(self):
    self._testBCastC([1, 3, 2], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_5D(self):
    self._testBCastD([1, 3, 2], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_6A(self):
    self._testBCastA([1, 3, 2], [2, 1, 1])

  @test_util.run_deprecated_v1
  def testBCast_6B(self):
    self._testBCastB([1, 3, 2], [2, 1, 1])

  @test_util.run_deprecated_v1
  def testBCast_6C(self):
    self._testBCastC([1, 3, 2], [2, 1, 1])

  @test_util.run_deprecated_v1
  def testBCast_6D(self):
    self._testBCastD([1, 3, 2], [2, 1, 1])

  @test_util.run_deprecated_v1
  def testBCast_7A(self):
    self._testBCastA([1, 3, 2], [1, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_7B(self):
    self._testBCastB([1, 3, 2], [1, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_7C(self):
    self._testBCastC([1, 3, 2], [1, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_7D(self):
    self._testBCastD([1, 3, 2], [1, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_8A(self):
    self._testBCastA([2, 1, 5], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_8B(self):
    self._testBCastB([2, 1, 5], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_8C(self):
    self._testBCastC([2, 1, 5], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_8D(self):
    self._testBCastD([2, 1, 5], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_9A(self):
    self._testBCastA([2, 0, 5], [2, 0, 1])

  @test_util.run_deprecated_v1
  def testBCast_9B(self):
    self._testBCastB([2, 0, 5], [2, 0, 1])

  @test_util.run_deprecated_v1
  def testBCast_9C(self):
    self._testBCastC([2, 0, 5], [2, 0, 1])

  @test_util.run_deprecated_v1
  def testBCast_9D(self):
    self._testBCastD([2, 0, 5], [2, 0, 1])

  @test_util.run_deprecated_v1
  def testBCast_10A(self):
    self._testBCastA([2, 3, 0], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_10B(self):
    self._testBCastB([2, 3, 0], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_10C(self):
    self._testBCastC([2, 3, 0], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_10D(self):
    self._testBCastD([2, 3, 0], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_11A(self):
    self._testBCastA([1, 3, 2], [1, 3, 2])

  @test_util.run_deprecated_v1
  def testBCast_11B(self):
    self._testBCastB([1, 3, 2], [1, 3, 2])

  @test_util.run_deprecated_v1
  def testBCast_11C(self):
    self._testBCastC([1, 3, 2], [1, 3, 2])

  @test_util.run_deprecated_v1
  def testBCast_11D(self):
    self._testBCastD([1, 3, 2], [1, 3, 2])

  @test_util.run_deprecated_v1
  def testBCast_12A(self):
    self._testBCastA([1, 1, 1, 1, 3, 2], [1, 3, 2])

  @test_util.run_deprecated_v1
  def testBCast_12B(self):
    self._testBCastB([1, 1, 1, 1, 3, 2], [1, 3, 2])

  @test_util.run_deprecated_v1
  def testBCast_12C(self):
    self._testBCastC([1, 1, 1, 1, 3, 2], [1, 3, 2])

  @test_util.run_deprecated_v1
  def testBCast_12D(self):
    self._testBCastD([1, 1, 1, 1, 3, 2], [1, 3, 2])

  @test_util.run_deprecated_v1
  def testBCast_13A(self):
    self._testBCastA([1, 3, 2, 1, 1], [1])

  @test_util.run_deprecated_v1
  def testBCast_13B(self):
    self._testBCastB([1, 3, 2, 1, 1], [1])

  @test_util.run_deprecated_v1
  def testBCast_13C(self):
    self._testBCastC([1, 3, 2, 1, 1], [1])

  @test_util.run_deprecated_v1
  def testBCast_13D(self):
    self._testBCastD([1, 3, 2, 1, 1], [1])

  @test_util.run_deprecated_v1
  def testBCast_14A(self):
    self._testBCastA([2, 3, 1, 1, 5], [1])

  @test_util.run_deprecated_v1
  def testBCast_14B(self):
    self._testBCastB([2, 3, 1, 1, 5], [1])

  @test_util.run_deprecated_v1
  def testBCast_14C(self):
    self._testBCastC([2, 3, 1, 1, 5], [1])

  @test_util.run_deprecated_v1
  def testBCast_14D(self):
    self._testBCastD([2, 3, 1, 1, 5], [1])

  @test_util.run_deprecated_v1
  def testBCast_15A(self):
    self._testBCastA([10, 3, 1, 2], [3, 1, 2])

  @test_util.run_deprecated_v1
  def testBCast_15B(self):
    self._testBCastB([10, 3, 1, 2], [3, 1, 2])

  @test_util.run_deprecated_v1
  def testBCast_15C(self):
    self._testBCastC([10, 3, 1, 2], [3, 1, 2])

  @test_util.run_deprecated_v1
  def testBCast_15D(self):
    self._testBCastD([10, 3, 1, 2], [3, 1, 2])

  @test_util.run_deprecated_v1
  def testMismatchedDimensions(self):
    for func in [
        math_ops.add, math_ops.subtract, math_ops.multiply, math_ops.div, _ADD,
        _SUB, _MUL, _TRUEDIV, _FLOORDIV
    ]:
      with self.assertRaisesWithPredicateMatch(
          ValueError, lambda e: "Dimensions must" in str(e)):
        func(
            ops.convert_to_tensor([10.0, 20.0, 30.0]),
            ops.convert_to_tensor([[40.0, 50.0], [60.0, 70.0]]))

  @test_util.run_deprecated_v1
  def testZeroPowGrad(self):
    with self.cached_session():
      for dtype in (np.float16, np.float32, np.float64, np.complex64,
                    np.complex128):
        x = constant_op.constant(0.0, dtype=dtype)
        y = constant_op.constant(2.0, dtype=dtype)
        z = math_ops.pow(x, y)
        error = gradient_checker.compute_gradient_error(y, [], z, [])
        self.assertEqual(error, 0)

  @test_util.run_deprecated_v1
  def testComplexPowGrad(self):
    with self.cached_session():
      for dtype in np.complex64, np.complex128:
        for base in 2.0, -2.0:
          x = constant_op.constant(base, dtype=dtype)
          y = constant_op.constant(2.0, dtype=dtype)
          z = math_ops.pow(x, y)
          error = gradient_checker.compute_gradient_error(y, [], z, [])
          self.assertLess(error, 2e-4)

  def testAtan2SpecialValues(self):
    x1l, x2l = zip((+0.0, +0.0), (+0.0, -0.0), (-0.0, +0.0), (-0.0, -0.0),
                   (1.0, 0.0), (-1.0, 0.0), (1.0, -0.0), (-1.0, -0.0),
                   (0.0, 1.0), (0.0, -1.0), (-0.0, 1.0), (-0.0, -1.0),
                   (1.2345, float("inf")), (1.2345, -float("inf")),
                   (-4.321, float("inf")), (-4.125, -float("inf")),
                   (float("inf"), float("inf")), (float("inf"), -float("inf")),
                   (-float("inf"), float("inf")),
                   (-float("inf"), -float("inf")), (float("1"), float("nan")),
                   (float("nan"), float("1")), (float("nan"), float("nan")))
    for dtype in np.float32, np.float64:
      x1 = np.array(x1l).astype(dtype)
      x2 = np.array(x2l).astype(dtype)
      self._compareCpu(x1, x2, np.arctan2, math_ops.atan2)
      self._compareGpu(x1, x2, np.arctan2, math_ops.atan2)

  def testPowNegativeExponentCpu(self):
    for dtype in [np.int32, np.int64]:
      with test_util.force_cpu():
        with self.assertRaisesRegex(
            errors_impl.InvalidArgumentError,
            "Integers to negative integer powers are not allowed"):
          x = np.array([5, 2]).astype(dtype)
          y = np.array([-2, 3]).astype(dtype)
          self.evaluate(math_ops.pow(x, y))

      with test_util.force_cpu():
        with self.assertRaisesRegex(
            errors_impl.InvalidArgumentError,
            "Integers to negative integer powers are not allowed"):
          x = np.array([5, 2]).astype(dtype)
          y = np.array([2, -3]).astype(dtype)
          self.evaluate(math_ops.pow(x, y))

      with test_util.force_cpu():
        with self.assertRaisesRegex(
            errors_impl.InvalidArgumentError,
            "Integers to negative integer powers are not allowed"):
          x = np.array([5, 2]).astype(dtype)
          y = -3
          self.evaluate(math_ops.pow(x, y))

  def testPowNegativeExponentGpu(self):
    if not test_util.is_gpu_available():
      self.skipTest("Requires GPU")
    # Negative integer powers return zero on GPUs for abs(LHS) > 1. Negative
    # integer powers for 1 and -1 will return the correct result.
    x = np.array([2, 3, 1, -1, -1]).astype(np.int64)
    y = np.array([-1, 0, -2, -2, -3]).astype(np.int64)
    z = math_ops.pow(x, y)
    self.assertAllEqual(self.evaluate(z), [0, 1, 1, 1, -1])

  def testFloorModInfDenominator(self):
    """Regression test for GitHub issue #58369."""
    if not test_util.is_gpu_available():
      self.skipTest("Requires GPU")

    dtypes = [
        dtypes_lib.bfloat16.as_numpy_dtype,
        np.float16,
        np.float32,
        np.float64,
    ]

    for dtype in dtypes:
      x = np.array([4, 0, -1, 4, 0, -1], dtype=dtype)
      y = np.array([np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf],
                   dtype=dtype)
      expected = np.array([4, 0, np.inf, -np.inf, 0, -1], dtype=dtype)

      self.assertAllClose(self.evaluate(math_ops.mod(x, y)), expected)


class ComparisonOpTest(test.TestCase):

  def _compareScalar(self, func, x, y, dtype):
    with test_util.use_gpu():
      out = func(
          ops.convert_to_tensor(np.array([x]).astype(dtype)),
          ops.convert_to_tensor(np.array([y]).astype(dtype)))
      ret = self.evaluate(out)
    return ret[0]

  def testScalarCompareScalar(self):
    dtypes = [np.float16, np.float32, np.float64, np.int32, np.int64]
    data = [-1, 0, 1]
    for t in dtypes:
      for x in data:
        for y in data:
          self.assertEqual(self._compareScalar(math_ops.less, x, y, t), x < y)
          self.assertEqual(
              self._compareScalar(math_ops.less_equal, x, y, t), x <= y)
          self.assertEqual(
              self._compareScalar(math_ops.greater, x, y, t), x > y)
          self.assertEqual(
              self._compareScalar(math_ops.greater_equal, x, y, t), x >= y)
          self.assertEqual(self._compareScalar(math_ops.equal, x, y, t), x == y)
          self.assertEqual(
              self._compareScalar(math_ops.not_equal, x, y, t), x != y)
    data = [-1, 0, 1, -1j, 1j, 1 + 1j, 1 - 1j]
    for t in [np.complex64, np.complex128]:
      for x in data:
        for y in data:
          self.assertEqual(self._compareScalar(math_ops.equal, x, y, t), x == y)
          self.assertEqual(
              self._compareScalar(math_ops.not_equal, x, y, t), x != y)

  def _compare(self, x, y, np_func, tf_func):
    np_ans = np_func(x, y)
    with test_util.use_gpu():
      out = tf_func(ops.convert_to_tensor(x), ops.convert_to_tensor(y))
      tf_ans = self.evaluate(out)
    self.assertAllEqual(np_ans, tf_ans)

  def testTensorCompareTensor(self):
    x = np.linspace(-15, 15, 6).reshape(1, 3, 2)  # pylint: disable=too-many-function-args
    y = np.linspace(20, -10, 6).reshape(1, 3, 2)  # pylint: disable=too-many-function-args
    for t in [np.float16, np.float32, np.float64, np.int32, np.int64]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compare(xt, yt, np.less, math_ops.less)
      self._compare(xt, yt, np.less_equal, math_ops.less_equal)
      self._compare(xt, yt, np.greater, math_ops.greater)
      self._compare(xt, yt, np.greater_equal, math_ops.greater_equal)
      self._compare(xt, yt, np.equal, math_ops.equal)
      self._compare(xt, yt, np.not_equal, math_ops.not_equal)
    # Complex types do not support ordering but do support equality tests.
    for t in [np.complex64, np.complex128]:
      xt = x.astype(t)
      xt -= 1j * xt
      yt = y.astype(t)
      yt -= 1j * yt
      self._compare(xt, yt, np.equal, math_ops.equal)
      self._compare(xt, yt, np.not_equal, math_ops.not_equal)

  def _compareBCast(self, xs, ys, dtype, np_func, tf_func):
    x = np.linspace(-15, 15, np.prod(xs)).astype(dtype).reshape(xs)
    y = np.linspace(20, -10, np.prod(ys)).astype(dtype).reshape(ys)
    if dtype in (np.complex64, np.complex128):
      x -= 1j * x
      y -= 1j * y
    self._compare(x, y, np_func, tf_func)
    self._compare(y, x, np_func, tf_func)

  def _testBCastByFunc(self, np_func, tf_func, include_complex=False):
    shapes = [
        ([1, 3, 2], [1]),
        ([1, 3, 2], [2]),
        ([1, 3, 2], [3, 2]),
        ([1, 3, 2], [3, 1]),
        ([1, 3, 2], [1, 3, 2]),
        ([1, 3, 2], [2, 3, 1]),
        ([1, 3, 2], [2, 1, 1]),
        ([1, 3, 2], [1, 3, 1]),
        ([2, 1, 5], [2, 3, 1]),
        ([2, 0, 5], [2, 0, 1]),
        ([2, 3, 0], [2, 3, 1]),
    ]
    dtypes = [
        np.float16,
        np.float32,
        np.float64,
        np.int32,
        np.int64,
    ]
    if include_complex:
      dtypes.extend([np.complex64, np.complex128])

    for (xs, ys) in shapes:
      for dtype in dtypes:
        self._compareBCast(xs, ys, dtype, np_func, tf_func)

  def testBCastLess(self):
    self._testBCastByFunc(np.less, math_ops.less)

  def testBCastLessEqual(self):
    self._testBCastByFunc(np.less_equal, math_ops.less_equal)

  def testBCastGreater(self):
    self._testBCastByFunc(np.greater, math_ops.greater)

  def testBCastGreaterEqual(self):
    self._testBCastByFunc(np.greater_equal, math_ops.greater_equal)

  def testBCastEqual(self):
    self._testBCastByFunc(np.equal, math_ops.equal, include_complex=True)

  def testBCastNotEqual(self):
    self._testBCastByFunc(
        np.not_equal, math_ops.not_equal, include_complex=True)

  def testShapeMismatch(self):
    dtypes = [np.float16, np.float32, np.float64, np.int32, np.int64]
    funcs = [
        math_ops.less, math_ops.less_equal, math_ops.greater,
        math_ops.greater_equal, math_ops.equal, math_ops.not_equal
    ]
    x = np.arange(0, 10).reshape([2, 5])
    y = np.arange(0, 10).reshape([5, 2])
    for t in dtypes:
      for f in funcs:
        with self.assertRaisesIncompatibleShapesError(
            (ValueError, errors.InvalidArgumentError)):
          f(x.astype(t), y.astype(t))

  def testEqualDType(self):
    dtypes = [
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.bool_,
    ]
    x = np.asarray([0, 1, 2, 3, 4])
    y = np.asarray([0, 1, 2, 3, 4])
    for dtype in dtypes:
      xt = x.astype(dtype)
      yt = y.astype(dtype)
      cmp_eq = math_ops.equal(xt, yt)
      cmp_ne = math_ops.not_equal(xt, yt)
      values = self.evaluate([cmp_eq, cmp_ne])
      self.assertAllEqual(
          [[True, True, True, True, True], [False, False, False, False, False]],
          values)
    for dtype in [np.complex64, np.complex128]:
      xt = x.astype(dtype)
      xt -= 1j * xt
      yt = y.astype(dtype)
      yt -= 1j * yt
      cmp_eq = math_ops.equal(xt, yt)
      cmp_ne = math_ops.not_equal(xt, yt)
      values = self.evaluate([cmp_eq, cmp_ne])
      self.assertAllEqual(
          [[True, True, True, True, True], [False, False, False, False, False]],
          values)

  @test_util.disable_tfrt("b/169901260")
  def testEqualQuantizeDType(self):
    dtypes = [
        dtypes_lib.qint8,
        dtypes_lib.qint16,
        dtypes_lib.quint8,
        dtypes_lib.quint16,
        dtypes_lib.qint32,
    ]
    x = np.asarray([0, 1, 2, 3, 4])
    y = np.asarray([0, 1, 2, 3, 4])
    for dtype in dtypes:
      xt = x.astype(dtype.as_numpy_dtype)
      yt = y.astype(dtype.as_numpy_dtype)
      cmp_eq = math_ops.equal(xt, yt)
      cmp_ne = math_ops.not_equal(xt, yt)
      values = self.evaluate([cmp_eq, cmp_ne])
      self.assertAllEqual(
          [[True, True, True, True, True], [False, False, False, False, False]],
          values)


if __name__ == "__main__":
  test.main()
