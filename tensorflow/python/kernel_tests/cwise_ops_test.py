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

"""Functional tests for coefficient-wise operations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import gen_math_ops


_ADD = lambda x, y: x + y
_SUB = lambda x, y: x - y
_MUL = lambda x, y: x * y
_POW = lambda x, y: x ** y
_TRUEDIV = lambda x, y: x / y
_FLOORDIV = lambda x, y: x // y
_MOD = lambda x, y: x % y
_NEG = lambda x: -x
_ABS = abs

_LT = lambda x, y: x < y
_LE = lambda x, y: x <= y
_GT = lambda x, y: x > y
_GE = lambda x, y: x >= y

_AND = lambda x, y: x & y
_OR = lambda x, y: x | y
_XOR = lambda x, y: x ^ y
_INV = lambda x: ~x


# TODO(zongheng): it'd be great to factor out this function and various random
# SparseTensor gen funcs.
def _sparsify(x, thresh=0.5, index_dtype=np.int64):
  x[x < thresh] = 0

  non_zero = np.where(x)
  x_indices = np.vstack(non_zero).astype(index_dtype).T
  x_values = x[non_zero]
  x_shape = x.shape

  return tf.SparseTensor(
      indices=x_indices, values=x_values, shape=x_shape), x_values

class UnaryOpTest(tf.test.TestCase):

  def _compareCpu(self, x, np_func, tf_func):
    np_ans = np_func(x)
    with self.test_session(use_gpu=False):
      inx = tf.convert_to_tensor(x)
      if x.dtype in (np.float32, np.float64):
        y = 1.1 * tf_func(inx)
        np_ans *= 1.1
      else:
        y = tf_func(inx)
      tf_cpu = y.eval()
      self.assertShapeEqual(np_ans, y)
      if x.dtype == np.float16:
        self.assertAllClose(np_ans, tf_cpu, rtol=1e-3, atol=1e-3)
      else:
        self.assertAllClose(np_ans, tf_cpu)

      if x.dtype in (np.complex64, np.complex128) and tf_func == tf.sign:
        return  # Return early

      if x.dtype == np.float16:
        s = list(np.shape(x))
        jacob_t, _ = tf.test.compute_gradient(inx,
                                              s,
                                              y,
                                              s,
                                              x_init_value=x)
        xf = x.astype(np.float)
        inxf = tf.convert_to_tensor(xf)
        yf = tf_func(inxf)
        _, jacob_n = tf.test.compute_gradient(inxf,
                                              s,
                                              yf,
                                              s,
                                              x_init_value=xf)
        jacob_n = jacob_n.astype(np.float16)
        self.assertAllClose(jacob_t, jacob_n, rtol=5e-3, atol=5e-3)
      elif x.dtype in (np.float32, np.complex64):
        s = list(np.shape(x))
        jacob_t, jacob_n = tf.test.compute_gradient(inx,
                                                    s,
                                                    y,
                                                    s,
                                                    x_init_value=x)
        self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
      elif x.dtype in (np.float64, np.complex128):
        s = list(np.shape(x))
        jacob_t, jacob_n = tf.test.compute_gradient(inx,
                                                    s,
                                                    y,
                                                    s,
                                                    x_init_value=x)
        self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _check(self, result_tensor, result_np, input_sp_t, tol):
    self.assertTrue(isinstance(result_tensor, tf.SparseTensor))
    self.assertTrue(isinstance(input_sp_t, tf.SparseTensor))
    self.assertAllEqual(input_sp_t.indices.eval(), result_tensor.indices.eval())
    self.assertAllEqual(input_sp_t.shape.eval(), result_tensor.shape.eval())
    if tol is None:
      self.assertAllClose(result_np, result_tensor.values.eval())
    else:
      self.assertAllClose(result_np, result_tensor.values.eval(), rtol=tol,
                          atol=tol)

  def _compareSparseCpu(self, x, np_func, tf_func, tol):
    x_sp, x_sp_vals = _sparsify(x)
    res_np = np_func(x_sp_vals)
    with self.test_session(use_gpu=False):
      self._check(tf_func(x_sp), res_np, x_sp, tol)

  def _compareGpu(self, x, np_func, tf_func):
    np_ans = np_func(x)
    with self.test_session(use_gpu=True):
      result = tf_func(tf.convert_to_tensor(x))
      tf_gpu = result.eval()
    if x.dtype == np.float16:
      self.assertAllClose(np_ans, tf_gpu, rtol=1e-3, atol=1e-3)
    else:
      self.assertAllClose(np_ans, tf_gpu)
    # TODO(zhifengc/ke): make gradient checker work on GPU.

  def _compareSparseGpu(self, x, np_func, tf_func, tol):
    x_sp, x_sp_vals = _sparsify(x)
    res_np = np_func(x_sp_vals)
    with self.test_session(use_gpu=True):
      self._check(tf_func(x_sp), res_np, x_sp, tol)

  def _compareBoth(self, x, np_func, tf_func):
    self._compareCpu(x, np_func, tf_func)
    self._compareGpu(x, np_func, tf_func)

  def _compareBothSparse(self, x, np_func, tf_func, tol=None):
    self._compareSparseCpu(x, np_func, tf_func, tol)
    self._compareSparseGpu(x, np_func, tf_func, tol)

  def _inv(self, x):
    return 1.0 / x

  def _rsqrt(self, x):
    return self._inv(np.sqrt(x))

  def _sigmoid(self, x):
    return 1.0 / (1.0 + np.exp(-x))

  def _replace_domain_error_with_inf(self, fn):
    def func(x):
      try:
        return fn(x)
      except ValueError as e:
        if "domain error" in str(e):
          return np.inf * np.ones_like(x)
        else:
          raise e
    return func

  def testFloatBasic(self):
    x = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float32)
    y = (x + .5).astype(np.float32)     # no zero
    z = (x + 15.5).astype(np.float32)   # all positive
    k = np.arange(-0.90, 0.90, 0.25).astype(np.float32) # between -1 and 1

    self._compareBoth(x, np.abs, tf.abs)
    self._compareBoth(x, np.abs, _ABS)
    self._compareBoth(x, np.negative, tf.neg)
    self._compareBoth(x, np.negative, _NEG)
    self._compareBoth(y, self._inv, tf.inv)
    self._compareBoth(x, np.square, tf.square)
    self._compareBoth(z, np.sqrt, tf.sqrt)
    self._compareBoth(z, self._rsqrt, tf.rsqrt)
    self._compareBoth(x, np.exp, tf.exp)
    self._compareBoth(z, np.log, tf.log)
    self._compareBoth(x, np.tanh, tf.tanh)
    self._compareBoth(x, self._sigmoid, tf.sigmoid)
    self._compareBoth(y, np.sign, tf.sign)
    self._compareBoth(x, np.sin, tf.sin)
    self._compareBoth(x, np.cos, tf.cos)
    self._compareBoth(k, np.arcsin, tf.asin)
    self._compareBoth(k, np.arccos, tf.acos)
    self._compareBoth(x, np.arctan, tf.atan)
    self._compareBoth(x, np.tan, tf.tan)
    self._compareBoth(
        y,
        np.vectorize(self._replace_domain_error_with_inf(math.lgamma)),
        tf.lgamma)
    self._compareBoth(x, np.vectorize(math.erf), tf.erf)
    self._compareBoth(x, np.vectorize(math.erfc), tf.erfc)

    self._compareBothSparse(x, np.abs, tf.abs)
    self._compareBothSparse(x, np.negative, tf.neg)
    self._compareBothSparse(x, np.square, tf.square)
    self._compareBothSparse(z, np.sqrt, tf.sqrt, tol=1e-3)
    self._compareBothSparse(x, np.tanh, tf.tanh)
    self._compareBothSparse(y, np.sign, tf.sign)
    self._compareBothSparse(x, np.vectorize(math.erf), tf.erf)

  def testFloatTanhEdge(self):
    x = np.arange(40, 40 + 6).reshape(6).astype(np.float32)
    self._compareBoth(x, np.tanh, tf.tanh)
    x = np.arange(-40, -40 + 6).reshape(6).astype(np.float32)
    self._compareBoth(x, np.tanh, tf.tanh)

  def testFloatEmpty(self):
    x = np.empty((2, 0, 5), dtype=np.float32)
    self._compareBoth(x, np.abs, tf.abs)
    self._compareBoth(x, np.abs, _ABS)
    self._compareBoth(x, np.negative, tf.neg)
    self._compareBoth(x, np.negative, _NEG)
    self._compareBoth(x, self._inv, tf.inv)
    self._compareBoth(x, np.square, tf.square)
    self._compareBoth(x, np.sqrt, tf.sqrt)
    self._compareBoth(x, self._rsqrt, tf.rsqrt)
    self._compareBoth(x, np.exp, tf.exp)
    self._compareBoth(x, np.log, tf.log)
    self._compareBoth(x, np.tanh, tf.tanh)
    self._compareBoth(x, self._sigmoid, tf.sigmoid)
    self._compareBoth(x, np.sign, tf.sign)
    self._compareBoth(x, np.sin, tf.sin)
    self._compareBoth(x, np.cos, tf.cos)
    # Can't use vectorize below, so just use some arbitrary function
    self._compareBoth(x, np.sign, tf.lgamma)
    self._compareBoth(x, np.sign, tf.erf)
    self._compareBoth(x, np.sign, tf.erfc)
    self._compareBoth(x, np.tan, tf.tan)
    self._compareBoth(x, np.arcsin, tf.asin)
    self._compareBoth(x, np.arccos, tf.acos)
    self._compareBoth(x, np.arctan, tf.atan)

    self._compareBothSparse(x, np.abs, tf.abs)
    self._compareBothSparse(x, np.negative, tf.neg)
    self._compareBothSparse(x, np.square, tf.square)
    self._compareBothSparse(x, np.sqrt, tf.sqrt, tol=1e-3)
    self._compareBothSparse(x, np.tanh, tf.tanh)
    self._compareBothSparse(x, np.sign, tf.sign)
    self._compareBothSparse(x, np.sign, tf.erf)

  def testDoubleBasic(self):
    x = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float64)
    y = (x + .5).astype(np.float64)    # no zero
    z = (x + 15.5).astype(np.float64)  # all positive
    k = np.arange(-0.90, 0.90, 0.35).reshape(1, 3, 2).astype(np.float64) # between -1 and 1
    self._compareBoth(x, np.abs, tf.abs)
    self._compareBoth(x, np.abs, _ABS)
    self._compareBoth(x, np.negative, tf.neg)
    self._compareBoth(x, np.negative, _NEG)
    self._compareBoth(y, self._inv, tf.inv)
    self._compareBoth(x, np.square, tf.square)
    self._compareBoth(z, np.sqrt, tf.sqrt)
    self._compareBoth(z, self._rsqrt, tf.rsqrt)
    self._compareBoth(x, np.exp, tf.exp)
    self._compareBoth(z, np.log, tf.log)
    self._compareBoth(x, np.tanh, tf.tanh)
    self._compareBoth(x, self._sigmoid, tf.sigmoid)
    self._compareBoth(y, np.sign, tf.sign)
    self._compareBoth(x, np.sin, tf.sin)
    self._compareBoth(x, np.cos, tf.cos)
    self._compareBoth(
        y,
        np.vectorize(self._replace_domain_error_with_inf(math.lgamma)),
        tf.lgamma)
    self._compareBoth(x, np.vectorize(math.erf), tf.erf)
    self._compareBoth(x, np.vectorize(math.erfc), tf.erfc)
    self._compareBoth(x, np.arctan, tf.atan)
    self._compareBoth(k, np.arcsin, tf.asin)
    self._compareBoth(k, np.arccos, tf.acos)
    self._compareBoth(k, np.tan, tf.tan)

    self._compareBothSparse(x, np.abs, tf.abs)
    self._compareBothSparse(x, np.negative, tf.neg)
    self._compareBothSparse(x, np.square, tf.square)
    self._compareBothSparse(z, np.sqrt, tf.sqrt, tol=1e-3)
    self._compareBothSparse(x, np.tanh, tf.tanh)
    self._compareBothSparse(y, np.sign, tf.sign)
    self._compareBothSparse(x, np.vectorize(math.erf), tf.erf)

  def testHalfBasic(self):
    x = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float16)
    y = (x + .5).astype(np.float16)    # no zero
    z = (x + 15.5).astype(np.float16)  # all positive
    self._compareBoth(x, np.abs, tf.abs)
    self._compareBoth(x, np.abs, _ABS)
    self._compareBoth(x, np.negative, tf.neg)
    self._compareBoth(x, np.negative, _NEG)
    self._compareBoth(y, self._inv, tf.inv)
    self._compareBoth(x, np.square, tf.square)
    self._compareBoth(z, np.sqrt, tf.sqrt)
    self._compareBoth(z, self._rsqrt, tf.rsqrt)
    self._compareBoth(x, np.exp, tf.exp)
    self._compareBoth(z, np.log, tf.log)
    self._compareBoth(x, np.tanh, tf.tanh)
    self._compareBoth(x, self._sigmoid, tf.sigmoid)
    self._compareBoth(y, np.sign, tf.sign)
    self._compareBoth(x, np.sin, tf.sin)
    self._compareBoth(x, np.cos, tf.cos)
    self._compareBoth(
        y,
        np.vectorize(self._replace_domain_error_with_inf(math.lgamma)),
        tf.lgamma)
    self._compareBoth(x, np.vectorize(math.erf), tf.erf)
    self._compareBoth(x, np.vectorize(math.erfc), tf.erfc)

    self._compareBothSparse(x, np.abs, tf.abs)
    self._compareBothSparse(x, np.negative, tf.neg)
    self._compareBothSparse(x, np.square, tf.square)
    self._compareBothSparse(z, np.sqrt, tf.sqrt, tol=1e-3)
    self._compareBothSparse(x, np.tanh, tf.tanh)
    self._compareBothSparse(y, np.sign, tf.sign)
    self._compareBothSparse(x, np.vectorize(math.erf), tf.erf, tol=1e-3)

  def testInt32Basic(self):
    x = np.arange(-6, 6, 2).reshape(1, 3, 2).astype(np.int32)
    self._compareCpu(x, np.abs, tf.abs)
    self._compareCpu(x, np.abs, _ABS)
    self._compareBoth(x, np.negative, tf.neg)
    self._compareBoth(x, np.negative, _NEG)
    self._compareBoth(x, np.square, tf.square)
    self._compareCpu(x, np.sign, tf.sign)

    self._compareBothSparse(x, np.abs, tf.abs)
    self._compareBothSparse(x, np.negative, tf.neg)
    self._compareBothSparse(x, np.square, tf.square)
    self._compareBothSparse(x, np.sign, tf.sign)

  def testInt64Basic(self):
    x = np.arange(
        -6 << 40, 6 << 40, 2 << 40).reshape(1, 3, 2).astype(np.int64)
    self._compareCpu(x, np.abs, tf.abs)
    self._compareCpu(x, np.abs, _ABS)
    self._compareCpu(x, np.negative, tf.neg)
    self._compareCpu(x, np.negative, _NEG)
    self._compareCpu(x, np.square, tf.square)
    self._compareCpu(x, np.sign, tf.sign)

    self._compareBothSparse(x, np.abs, tf.abs)
    self._compareBothSparse(x, np.negative, tf.neg)
    self._compareBothSparse(x, np.square, tf.square)
    self._compareBothSparse(x, np.sign, tf.sign)

  def testComplex64Basic(self):
    x = np.complex(1, 1) * np.arange(-3, 3).reshape(1, 3, 2).astype(
        np.complex64)
    y = x + 0.5  # no zeros
    self._compareCpu(x, np.abs, tf.complex_abs)
    self._compareCpu(x, np.abs, _ABS)
    self._compareCpu(x, np.negative, tf.neg)
    self._compareCpu(x, np.negative, _NEG)
    self._compareCpu(y, self._inv, tf.inv)
    self._compareCpu(x, np.square, tf.square)
    self._compareCpu(y, np.sqrt, tf.sqrt)
    self._compareCpu(y, self._rsqrt, tf.rsqrt)
    self._compareCpu(x, np.exp, tf.exp)
    self._compareCpu(y, np.log, tf.log)
    self._compareCpu(x, np.tanh, tf.tanh)
    self._compareCpu(x, self._sigmoid, tf.sigmoid)
    self._compareCpu(x, np.sin, tf.sin)
    self._compareCpu(x, np.cos, tf.cos)

    self._compareBothSparse(x, np.abs, tf.abs)
    self._compareBothSparse(x, np.negative, tf.neg)
    self._compareBothSparse(x, np.square, tf.square)
    self._compareBothSparse(x, np.sqrt, tf.sqrt, 1e-3)
    self._compareBothSparse(x, np.tanh, tf.tanh)

    # Numpy uses an incorrect definition of sign; use the right one instead.
    def complex_sign(x):
      return x / np.abs(x)
    self._compareCpu(y, complex_sign, tf.sign)
    self._compareBothSparse(y, complex_sign, tf.sign)

  def testComplex128Basic(self):
    x = np.complex(1, 1) * np.arange(-3, 3).reshape(1, 3, 2).astype(
        np.complex128)
    y = x + 0.5  # no zeros
    self._compareCpu(x, np.abs, tf.abs)
    self._compareCpu(x, np.abs, _ABS)
    self._compareCpu(x, np.negative, tf.neg)
    self._compareCpu(x, np.negative, _NEG)
    self._compareCpu(y, self._inv, tf.inv)
    self._compareCpu(x, np.square, tf.square)
    self._compareCpu(y, np.sqrt, tf.sqrt)
    self._compareCpu(y, self._rsqrt, tf.rsqrt)
    self._compareCpu(x, np.exp, tf.exp)
    self._compareCpu(y, np.log, tf.log)
    self._compareCpu(x, np.tanh, tf.tanh)
    self._compareCpu(x, self._sigmoid, tf.sigmoid)
    self._compareCpu(x, np.sin, tf.sin)
    self._compareCpu(x, np.cos, tf.cos)

    self._compareBothSparse(x, np.abs, tf.abs)
    self._compareBothSparse(x, np.negative, tf.neg)
    self._compareBothSparse(x, np.square, tf.square)
    self._compareBothSparse(x, np.sqrt, tf.sqrt, 1e-3)
    self._compareBothSparse(x, np.tanh, tf.tanh)

    # Numpy uses an incorrect definition of sign; use the right one instead.
    def complex_sign(x):
      return x / np.abs(x)
    self._compareCpu(y, complex_sign, tf.sign)
    self._compareBothSparse(y, complex_sign, tf.sign)

  def testGradGrad(self):
    np.random.seed(7)
    shape = (5,)
    dtype_tols = [(np.float32, 5e-4), (np.float64, 1e-6), (np.complex64, 5e-4),
                  (np.complex128, 1e-6)]
    op_range = [(gen_math_ops._inv_grad, [-2, 2]),
                (gen_math_ops._rsqrt_grad, [0.1, 3]),
                (gen_math_ops._sigmoid_grad, [-2, 2]),
                (gen_math_ops._sqrt_grad, [0.1, 3]),
                (gen_math_ops._tanh_grad, [-2, 2]),]

    def rand(dtype):
      x = np.random.uniform(
          real_range[0], real_range[1], size=shape[0]).astype(dtype)
      if dtype in (np.complex64, np.complex128):
        x += 1j * np.random.uniform(-2, 2, size=shape[0]).astype(dtype)
      return x

    for op, real_range in op_range:
      with self.test_session():
        for dtype, tol in dtype_tols:
          x = tf.constant(rand(dtype))
          y = tf.constant(rand(dtype))
          z = op(x, y)
          grads = tf.test.compute_gradient(
              [x, y], [shape, shape],
              z,
              shape,
              x_init_value=[rand(dtype), rand(dtype)])
          if isinstance(grads, tuple):
            grads = [grads]
          for analytical, numerical in grads:
            self.assertAllClose(analytical, numerical, rtol=tol, atol=tol)


class BinaryOpTest(tf.test.TestCase):

  def _compareCpu(self, x, y, np_func, tf_func, also_compare_variables=False):
    np_ans = np_func(x, y)
    with self.test_session(use_gpu=False):
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_cpu = out.eval()
      # Test that the op takes precedence over numpy operators.
      np_left = tf_func(x, iny).eval()
      np_right = tf_func(inx, y).eval()

      if also_compare_variables:
        var_x = tf.Variable(x)
        var_y = tf.Variable(y)
        tf.initialize_all_variables().run()
        print(type(x), type(y), type(var_x), type(var_y))
        print(type(tf_func(x, var_y)), type(tf_func(var_x, y)))
        np_var_left = tf_func(x, var_y).eval()
        np_var_right = tf_func(var_x, y).eval()

    if np_ans.dtype != np.object:
      self.assertAllClose(np_ans, tf_cpu)
      self.assertAllClose(np_ans, np_left)
      self.assertAllClose(np_ans, np_right)
      if also_compare_variables:
        self.assertAllClose(np_ans, np_var_left)
        self.assertAllClose(np_ans, np_var_right)
    self.assertShapeEqual(np_ans, out)

  _GRAD_TOL = {tf.float16: 1e-3,
               tf.float32: 1e-3,
               tf.complex64: 1e-2,
               tf.float64: 1e-5,
               tf.complex128: 1e-4}

  def _compareGradientX(self, x, y, np_func, tf_func,
                        numeric_gradient_type=None):
    z = np_func(x, y)
    zs = list(z.shape)
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      if x.dtype in (np.float32, np.float64):
        out = 1.1 * tf_func(inx, iny)
      else:
        out = tf_func(inx, iny)
      xs = list(x.shape)
      jacob_t, jacob_n = tf.test.compute_gradient(inx,
                                                  xs,
                                                  out,
                                                  zs,
                                                  x_init_value=x)
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = tf.convert_to_tensor(xf)
        inyf = tf.convert_to_tensor(yf)
        outf = tf_func(inxf, inyf)
        _, jacob_n = tf.test.compute_gradient(inxf,
                                              xs,
                                              outf,
                                              zs,
                                              x_init_value=xf,
                                              delta=1e-3)
        jacob_n = jacob_n.astype(x.dtype)
      tol = self._GRAD_TOL[tf.as_dtype(x.dtype)]
      self.assertAllClose(jacob_t, jacob_n, rtol=tol, atol=tol)

  def _compareGradientY(self, x, y, np_func, tf_func,
                        numeric_gradient_type=None):
    z = np_func(x, y)
    zs = list(z.shape)
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      if x.dtype in (np.float32, np.float64):
        out = 1.1 * tf_func(inx, iny)
      else:
        out = tf_func(inx, iny)
      ys = list(np.shape(y))
      jacob_t, jacob_n = tf.test.compute_gradient(iny,
                                                  ys,
                                                  out,
                                                  zs,
                                                  x_init_value=y)
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = tf.convert_to_tensor(xf)
        inyf = tf.convert_to_tensor(yf)
        outf = tf_func(inxf, inyf)
        _, jacob_n = tf.test.compute_gradient(inyf,
                                              ys,
                                              outf,
                                              zs,
                                              x_init_value=yf)
        jacob_n = jacob_n.astype(x.dtype)
    tol = self._GRAD_TOL[tf.as_dtype(x.dtype)]
    self.assertAllClose(jacob_t, jacob_n, rtol=tol, atol=tol)

  def _compareGpu(self, x, y, np_func, tf_func):
    np_ans = np_func(x, y)
    with self.test_session(use_gpu=True):
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_gpu = out.eval()
    self.assertAllClose(np_ans, tf_gpu)
    self.assertShapeEqual(np_ans, out)
    # TODO(zhifengc/ke): make gradient checker work on GPU.

  def _compareBoth(self, x, y, np_func, tf_func, also_compare_variables=False):
    self._compareCpu(x, y, np_func, tf_func, also_compare_variables)
    if x.dtype in (np.float16, np.float32, np.float64):
      if tf_func not in (_FLOORDIV, tf.floordiv, tf.igamma, tf.igammac, tf.zeta, tf.polygamma):
        self._compareGradientX(x, y, np_func, tf_func)
        self._compareGradientY(x, y, np_func, tf_func)
      if tf_func in (tf.igamma, tf.igammac, tf.zeta, tf.polygamma):
        # These methods only support gradients in the second parameter
        self._compareGradientY(x, y, np_func, tf_func)
      self._compareGpu(x, y, np_func, tf_func)

  def testFloatBasic(self):
    x = np.linspace(-5, 20, 15).reshape(1, 3, 5).astype(np.float32)
    y = np.linspace(20, -5, 15).reshape(1, 3, 5).astype(np.float32)
    self._compareBoth(x, y, np.add, tf.add, also_compare_variables=True)
    self._compareBoth(x, y, np.subtract, tf.sub)
    self._compareBoth(x, y, np.multiply, tf.mul)
    self._compareBoth(x, y + 0.1, np.true_divide, tf.truediv)
    self._compareBoth(x, y + 0.1, np.floor_divide, tf.floordiv)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    self._compareBoth(x, y + 0.1, np.true_divide, _TRUEDIV)
    self._compareBoth(x, y + 0.1, np.floor_divide, _FLOORDIV)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top
      a_pos_small = np.linspace(0.1, 2, 15).reshape(1, 3, 5).astype(np.float32)
      x_pos_small = np.linspace(0.1, 10, 15).reshape(1, 3, 5).astype(np.float32)
      self._compareBoth(a_pos_small, x_pos_small, special.gammainc, tf.igamma)
      self._compareBoth(a_pos_small, x_pos_small, special.gammaincc, tf.igammac)
      # Need x > 1
      self._compareBoth(x_pos_small + 1, a_pos_small, special.zeta, tf.zeta)
      n_small = np.arange(0, 15).reshape(1, 3, 5).astype(np.float32)
      self._compareBoth(n_small, x_pos_small, special.polygamma, tf.polygamma)
    except ImportError as e:
      tf.logging.warn("Cannot test special functions: %s" % str(e))

  def testFloatDifferentShapes(self):
    x = np.array([1, 2, 3, 4]).reshape(2, 2).astype(np.float32)
    y = np.array([1, 2]).reshape(2, 1).astype(np.float32)
    with self.test_session() as sess:
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      s = tf.reduce_sum(inx * iny)
      gx, gy = sess.run(tf.gradients(s, [inx, iny]))
    # gx is simply the broadcasted y
    self.assertAllEqual(gx, np.array([1, 1, 2, 2])
                        .reshape(2, 2).astype(np.float32))
    # gy is x's column summed up
    self.assertAllEqual(gy, np.array([3, 7]).
                        reshape(2, 1).astype(np.float32))

  def testFloatVariableOverload(self):
    x = np.array([1, 2, 3, 4]).reshape(2, 2).astype(np.int32)
    y = np.array([1, 2]).reshape(2, 1).astype(np.int32)
    var_x = tf.Variable(x)
    var_y = tf.Variable(y)
    with self.test_session() as sess:
      sess.run([var_x.initializer, var_y.initializer])
      left_result = (var_x * y).eval()
      right_result = (x * var_y).eval()
    np_result = x * y
    self.assertAllEqual(np_result, left_result)
    self.assertAllEqual(np_result, right_result)

  def testDoubleBasic(self):
    x = np.linspace(-5, 20, 15).reshape(1, 3, 5).astype(np.float64)
    y = np.linspace(20, -5, 15).reshape(1, 3, 5).astype(np.float64)
    self._compareBoth(x, y, np.add, tf.add)
    self._compareBoth(x, y, np.subtract, tf.sub)
    self._compareBoth(x, y, np.multiply, tf.mul)
    self._compareBoth(x, y + 0.1, np.true_divide, tf.truediv)
    self._compareBoth(x, y + 0.1, np.floor_divide, tf.floordiv)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    self._compareBoth(x, y + 0.1, np.true_divide, _TRUEDIV)
    self._compareBoth(x, y + 0.1, np.floor_divide, _FLOORDIV)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top
      a_pos_small = np.linspace(0.1, 2, 15).reshape(1, 3, 5).astype(np.float32)
      x_pos_small = np.linspace(0.1, 10, 15).reshape(1, 3, 5).astype(np.float32)
      self._compareBoth(a_pos_small, x_pos_small, special.gammainc, tf.igamma)
      self._compareBoth(a_pos_small, x_pos_small, special.gammaincc, tf.igammac)
    except ImportError as e:
      tf.logging.warn("Cannot test special functions: %s" % str(e))

  def testInt8Basic(self):
    x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.int8)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.int8)
    self._compareBoth(x, y, np.multiply, tf.mul)
    self._compareBoth(x, y, np.multiply, _MUL)

  def testInt16Basic(self):
    x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.int16)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.int16)
    self._compareBoth(x, y, np.multiply, tf.mul)
    self._compareBoth(x, y, np.multiply, _MUL)

  def testUint16Basic(self):
    x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.uint16)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.uint16)
    self._compareBoth(x, y, np.multiply, tf.mul)
    self._compareBoth(x, y, np.multiply, _MUL)
    self._compareBoth(x, y, np.true_divide, tf.truediv)
    self._compareBoth(x, y, np.floor_divide, tf.floordiv)
    self._compareBoth(x, y, np.true_divide, _TRUEDIV)
    self._compareBoth(x, y, np.floor_divide, _FLOORDIV)

  def testInt32Basic(self):
    x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.int32)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.int32)
    self._compareBoth(x, y, np.add, tf.add)
    self._compareBoth(x, y, np.subtract, tf.sub)
    self._compareBoth(x, y, np.multiply, tf.mul)
    self._compareBoth(x, y, np.true_divide, tf.truediv)
    self._compareBoth(x, y, np.floor_divide, tf.floordiv)
    self._compareBoth(x, y, np.mod, tf.mod)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    self._compareBoth(x, y, np.true_divide, _TRUEDIV)
    self._compareBoth(x, y, np.floor_divide, _FLOORDIV)
    self._compareBoth(x, y, np.mod, _MOD)
    # _compareBoth tests on GPU only for floating point types, so test
    # _MOD for int32 on GPU by calling _compareGpu
    self._compareGpu(x, y, np.mod, _MOD)

  def testInt64Basic(self):
    x = np.arange(1 << 40, 13 << 40, 2 << 40).reshape(1, 3, 2).astype(np.int64)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.int64)
    self._compareBoth(x, y, np.subtract, tf.sub)
    self._compareBoth(x, y, np.multiply, tf.mul)
    self._compareBoth(x, y, np.true_divide, tf.truediv)
    self._compareBoth(x, y, np.floor_divide, tf.floordiv)
    self._compareBoth(x, y, np.mod, tf.mod)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    self._compareBoth(x, y, np.true_divide, _TRUEDIV)
    self._compareBoth(x, y, np.floor_divide, _FLOORDIV)
    self._compareBoth(x, y, np.mod, _MOD)

  def testComplex64Basic(self):
    x = np.complex(1, 1) * np.linspace(-10, 10, 6).reshape(1, 3, 2).astype(
        np.complex64)
    y = np.complex(1, 1) * np.linspace(20, -20, 6).reshape(1, 3, 2).astype(
        np.complex64)
    self._compareBoth(x, y, np.add, tf.add)
    self._compareBoth(x, y, np.subtract, tf.sub)
    self._compareBoth(x, y, np.multiply, tf.mul)
    self._compareBoth(x, y + 0.1, np.true_divide, tf.truediv)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    self._compareBoth(x, y + 0.1, np.true_divide, _TRUEDIV)

  def testComplex128Basic(self):
    x = np.complex(1, 1) * np.linspace(-10, 10, 6).reshape(1, 3, 2).astype(
        np.complex128)
    y = np.complex(1, 1) * np.linspace(20, -20, 6).reshape(1, 3, 2).astype(
        np.complex128)
    self._compareBoth(x, y, np.add, tf.add)
    self._compareBoth(x, y, np.subtract, tf.sub)
    self._compareBoth(x, y, np.multiply, tf.mul)
    self._compareBoth(x, y + 0.1, np.true_divide, tf.truediv)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    self._compareBoth(x, y + 0.1, np.true_divide, _TRUEDIV)

  def testStringComparison(self):
    x = np.array([["abc", "bh"], ["c", ""]])
    y = np.array([["abc", "bh"], ["def", "hi"]])
    with self.test_session(use_gpu=False) as sess:
      cmp_eq = tf.equal(x, y)
      cmp_not_eq = tf.not_equal(x, y)
      values = sess.run([cmp_eq, cmp_not_eq])
      self.assertAllEqual([[True, True], [False, False]], values[0])
      self.assertAllEqual([[False, False], [True, True]], values[1])

  def testString(self):
    x = np.array([["x_0_0", "x_0_1", "x_0_2"],
                  ["x_1_0", "x_1_1", "x_1_2"],
                  ["x_2_0", "x_2_1", "x_2_2"]], dtype=np.object)
    y = np.array([["y_0_0", "y_0_1", "y_0_2"],
                  ["y_1_0", "y_1_1", "y_1_2"],
                  ["y_2_0", "y_2_1", "y_2_2"]], dtype=np.object)
    z = np.array([["z_0", "z_1", "z_2"]], dtype=np.object)
    w = np.array("w", dtype=np.object)
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
    if x.dtype in (np.float16, np.float32, np.float64, np.complex64,
                   np.complex128):
      if tf_func not in (_FLOORDIV, tf.floordiv):
        if x.dtype == np.float16:
          # Compare fp16 theoretical gradients to fp32 numerical gradients,
          # since fp16 numerical gradients are too imprecise unless great
          # care is taken with choosing the inputs and the delta. This is
          # a weaker check (in particular, it does not test the op itself,
          # only its gradient), but it's much better than nothing.
          self._compareGradientX(x, y, np_func, tf_func, np.float)
          self._compareGradientY(x, y, np_func, tf_func, np.float)
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
              tf_func in (_FLOORDIV, tf.floordiv)):
          continue  # floordiv makes no sense for complex numbers
        self._compareBCast(xs, ys, dtype, np_func, tf_func)
        self._compareBCast(ys, xs, dtype, np_func, tf_func)

  def _testBCastA(self, xs, ys):
    funcs = [
        (np.add, tf.add),
        (np.add, _ADD),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def _testBCastB(self, xs, ys):
    funcs = [
        (np.subtract, tf.sub),
        (np.subtract, _SUB),
        (np.power, tf.pow),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def _testBCastC(self, xs, ys):
    funcs = [
        (np.multiply, tf.mul),
        (np.multiply, _MUL),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def _testBCastD(self, xs, ys):
    funcs = [
        (np.true_divide, tf.truediv),
        (np.floor_divide, tf.floordiv),
        (np.true_divide, _TRUEDIV),
        (np.floor_divide, _FLOORDIV),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def testBCast_0A(self):
    self._testBCastA([1, 3, 2], [1])

  def testBCast_0B(self):
    self._testBCastB([1, 3, 2], [1])

  def testBCast_0C(self):
    self._testBCastC([1, 3, 2], [1])

  def testBCast_0D(self):
    self._testBCastD([1, 3, 2], [1])

  def testBCast_1A(self):
    self._testBCastA([1, 3, 2], [2])

  def testBCast_1B(self):
    self._testBCastB([1, 3, 2], [2])

  def testBCast_1C(self):
    self._testBCastC([1, 3, 2], [2])

  def testBCast_1D(self):
    self._testBCastD([1, 3, 2], [2])

  def testBCast_2A(self):
    self._testBCastA([1, 3, 2], [3, 2])

  def testBCast_2B(self):
    self._testBCastB([1, 3, 2], [3, 2])

  def testBCast_2C(self):
    self._testBCastC([1, 3, 2], [3, 2])

  def testBCast_2D(self):
    self._testBCastD([1, 3, 2], [3, 2])

  def testBCast_3A(self):
    self._testBCastA([1, 3, 2], [3, 1])

  def testBCast_3B(self):
    self._testBCastB([1, 3, 2], [3, 1])

  def testBCast_3C(self):
    self._testBCastC([1, 3, 2], [3, 1])

  def testBCast_3D(self):
    self._testBCastD([1, 3, 2], [3, 1])

  def testBCast_4A(self):
    self._testBCastA([1, 3, 2], [1, 3, 2])

  def testBCast_4B(self):
    self._testBCastB([1, 3, 2], [1, 3, 2])

  def testBCast_4C(self):
    self._testBCastC([1, 3, 2], [1, 3, 2])

  def testBCast_4D(self):
    self._testBCastD([1, 3, 2], [1, 3, 2])

  def testBCast_5A(self):
    self._testBCastA([1, 3, 2], [2, 3, 1])

  def testBCast_5B(self):
    self._testBCastB([1, 3, 2], [2, 3, 1])

  def testBCast_5C(self):
    self._testBCastC([1, 3, 2], [2, 3, 1])

  def testBCast_5D(self):
    self._testBCastD([1, 3, 2], [2, 3, 1])

  def testBCast_6A(self):
    self._testBCastA([1, 3, 2], [2, 1, 1])

  def testBCast_6B(self):
    self._testBCastB([1, 3, 2], [2, 1, 1])

  def testBCast_6C(self):
    self._testBCastC([1, 3, 2], [2, 1, 1])

  def testBCast_6D(self):
    self._testBCastD([1, 3, 2], [2, 1, 1])

  def testBCast_7A(self):
    self._testBCastA([1, 3, 2], [1, 3, 1])

  def testBCast_7B(self):
    self._testBCastB([1, 3, 2], [1, 3, 1])

  def testBCast_7C(self):
    self._testBCastC([1, 3, 2], [1, 3, 1])

  def testBCast_7D(self):
    self._testBCastD([1, 3, 2], [1, 3, 1])

  def testBCast_8A(self):
    self._testBCastA([2, 1, 5], [2, 3, 1])

  def testBCast_8B(self):
    self._testBCastB([2, 1, 5], [2, 3, 1])

  def testBCast_8C(self):
    self._testBCastC([2, 1, 5], [2, 3, 1])

  def testBCast_8D(self):
    self._testBCastD([2, 1, 5], [2, 3, 1])

  def testBCast_9A(self):
    self._testBCastA([2, 0, 5], [2, 0, 1])

  def testBCast_9B(self):
    self._testBCastB([2, 0, 5], [2, 0, 1])

  def testBCast_9C(self):
    self._testBCastC([2, 0, 5], [2, 0, 1])

  def testBCast_9D(self):
    self._testBCastD([2, 0, 5], [2, 0, 1])

  def testBCast_10A(self):
    self._testBCastA([2, 3, 0], [2, 3, 1])

  def testBCast_10B(self):
    self._testBCastB([2, 3, 0], [2, 3, 1])

  def testBCast_10C(self):
    self._testBCastC([2, 3, 0], [2, 3, 1])

  def testBCast_10D(self):
    self._testBCastD([2, 3, 0], [2, 3, 1])

  def testBCast_11A(self):
    self._testBCastA([1, 3, 2], [1, 3, 2])

  def testBCast_11B(self):
    self._testBCastB([1, 3, 2], [1, 3, 2])

  def testBCast_11C(self):
    self._testBCastC([1, 3, 2], [1, 3, 2])

  def testBCast_11D(self):
    self._testBCastD([1, 3, 2], [1, 3, 2])

  def testBCast_12A(self):
    self._testBCastA([1, 1, 1, 1, 3, 2], [1, 3, 2])

  def testBCast_12B(self):
    self._testBCastB([1, 1, 1, 1, 3, 2], [1, 3, 2])

  def testBCast_12C(self):
    self._testBCastC([1, 1, 1, 1, 3, 2], [1, 3, 2])

  def testBCast_12D(self):
    self._testBCastD([1, 1, 1, 1, 3, 2], [1, 3, 2])

  def testBCast_13A(self):
    self._testBCastA([1, 3, 2, 1, 1], [1])

  def testBCast_13B(self):
    self._testBCastB([1, 3, 2, 1, 1], [1])

  def testBCast_13C(self):
    self._testBCastC([1, 3, 2, 1, 1], [1])

  def testBCast_13D(self):
    self._testBCastD([1, 3, 2, 1, 1], [1])

  def testBCast_14A(self):
    self._testBCastA([2, 3, 1, 1, 5], [1])

  def testBCast_14B(self):
    self._testBCastB([2, 3, 1, 1, 5], [1])

  def testBCast_14C(self):
    self._testBCastC([2, 3, 1, 1, 5], [1])

  def testBCast_14D(self):
    self._testBCastD([2, 3, 1, 1, 5], [1])

  def testBCast_15A(self):
    self._testBCastA([10, 3, 1, 2], [3, 1, 2])

  def testBCast_15B(self):
    self._testBCastB([10, 3, 1, 2], [3, 1, 2])

  def testBCast_15C(self):
    self._testBCastC([10, 3, 1, 2], [3, 1, 2])

  def testBCast_15D(self):
    self._testBCastD([10, 3, 1, 2], [3, 1, 2])

  def testMismatchedDimensions(self):
    for func in [tf.add, tf.sub, tf.mul, tf.div, _ADD, _SUB, _MUL, _TRUEDIV,
                 _FLOORDIV]:
      with self.assertRaisesWithPredicateMatch(
          ValueError, lambda e: "Dimensions must" in str(e)):
        func(tf.convert_to_tensor([10.0, 20.0, 30.0]),
             tf.convert_to_tensor([[40.0, 50.0], [60.0, 70.0]]))

  def testZeroPowGrad(self):
    with self.test_session():
      for dtype in (np.float16, np.float32, np.float64, np.complex64,
                    np.complex128):
        x = tf.constant(0.0, dtype=dtype)
        y = tf.constant(2.0, dtype=dtype)
        z = tf.pow(x, y)
        error = tf.test.compute_gradient_error(y, [], z, [])
        self.assertEqual(error, 0)

  def testComplexPowGrad(self):
    with self.test_session():
      for dtype in np.complex64, np.complex128:
        for base in 2.0, -2.0:
          x = tf.constant(base, dtype=dtype)
          y = tf.constant(2.0, dtype=dtype)
          z = tf.pow(x, y)
          error = tf.test.compute_gradient_error(y, [], z, [])
          self.assertLess(error, 2e-4)


class ComparisonOpTest(tf.test.TestCase):

  def _compare(self, func, x, y, dtype):
    with self.test_session(use_gpu=False):
      out = func(tf.convert_to_tensor(np.array([x]).astype(dtype)),
                 tf.convert_to_tensor(np.array([y]).astype(dtype)))
      ret = out.eval()
    return ret[0]

  def testScalarCompareScalar(self):
    dtypes = [np.float16, np.float32, np.float64, np.int32, np.int64]
    data = [-1, 0, 1]
    for t in dtypes:
      for x in data:
        for y in data:
          self.assertEqual(self._compare(tf.less, x, y, t), x < y)
          self.assertEqual(self._compare(tf.less_equal, x, y, t), x <= y)
          self.assertEqual(self._compare(tf.greater, x, y, t), x > y)
          self.assertEqual(self._compare(tf.greater_equal, x, y, t), x >= y)
          self.assertEqual(self._compare(tf.equal, x, y, t), x == y)
          self.assertEqual(self._compare(tf.not_equal, x, y, t), x != y)
    data = [-1, 0, 1, -1j, 1j, 1 + 1j, 1 - 1j]
    for t in [np.complex64, np.complex128]:
      for x in data:
        for y in data:
          self.assertEqual(self._compare(tf.equal, x, y, t), x == y)
          self.assertEqual(self._compare(tf.not_equal, x, y, t), x != y)

  def _compareCpu(self, x, y, np_func, tf_func):
    np_ans = np_func(x, y)
    with self.test_session(use_gpu=False):
      out = tf_func(tf.convert_to_tensor(x), tf.convert_to_tensor(y))
      tf_cpu = out.eval()
    self.assertAllEqual(np_ans, tf_cpu)

  def _compareGpu(self, x, y, np_func, tf_func):
    np_ans = np_func(x, y)
    with self.test_session(use_gpu=True):
      out = tf_func(tf.convert_to_tensor(x), tf.convert_to_tensor(y))
      tf_gpu = out.eval()
    self.assertAllEqual(np_ans, tf_gpu)

  def _compareBoth(self, x, y, np_func, tf_func):
    self._compareCpu(x, y, np_func, tf_func)
    if x.dtype == np.float16 or x.dtype == np.float32 or x.dtype == np.float64:
      self._compareGpu(x, y, np_func, tf_func)

  def testTensorCompareTensor(self):
    x = np.linspace(-15, 15, 6).reshape(1, 3, 2)
    y = np.linspace(20, -10, 6).reshape(1, 3, 2)
    for t in [np.float16, np.float32, np.float64, np.int32, np.int64]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compareBoth(xt, yt, np.less, tf.less)
      self._compareBoth(xt, yt, np.less_equal, tf.less_equal)
      self._compareBoth(xt, yt, np.greater, tf.greater)
      self._compareBoth(xt, yt, np.greater_equal, tf.greater_equal)
      self._compareBoth(xt, yt, np.equal, tf.equal)
      self._compareBoth(xt, yt, np.not_equal, tf.not_equal)
    # TODO(zhifengc): complex64 doesn't work on GPU yet.
    for t in [np.complex64, np.complex128]:
      self._compareCpu(x.astype(t), y.astype(t), np.equal, tf.equal)
      self._compareCpu(x.astype(t), y.astype(t), np.not_equal, tf.not_equal)

  def _compareBCast(self, xs, ys, dtype, np_func, tf_func):
    x = np.linspace(-15, 15, np.prod(xs)).astype(dtype).reshape(xs)
    y = np.linspace(20, -10, np.prod(ys)).astype(dtype).reshape(ys)
    self._compareCpu(x, y, np_func, tf_func)
    self._compareCpu(y, x, np_func, tf_func)
    if x.dtype == np.float16 or x.dtype == np.float32 or x.dtype == np.float64:
      self._compareGpu(x, y, np_func, tf_func)
      self._compareGpu(y, x, np_func, tf_func)

  def _testBCastByFunc(self, np_func, tf_func):
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
    for (xs, ys) in shapes:
      for dtype in dtypes:
        self._compareBCast(xs, ys, dtype, np_func, tf_func)

  def testBCastLess(self):
    self._testBCastByFunc(np.less, tf.less)

  def testBCastLessEqual(self):
    self._testBCastByFunc(np.less_equal, tf.less_equal)

  def testBCastGreater(self):
    self._testBCastByFunc(np.greater, tf.greater)

  def testBCastGreaterEqual(self):
    self._testBCastByFunc(np.greater_equal, tf.greater_equal)

  def testBCastEqual(self):
    self._testBCastByFunc(np.equal, tf.equal)

  def testBCastNotEqual(self):
    self._testBCastByFunc(np.not_equal, tf.not_equal)

  def testShapeMismatch(self):
    dtypes = [np.float16, np.float32, np.float64, np.int32, np.int64]
    funcs = [tf.less, tf.less_equal, tf.greater,
             tf.greater_equal, tf.equal, tf.not_equal]
    x = np.arange(0, 10).reshape([2, 5])
    y = np.arange(0, 10).reshape([5, 2])
    for t in dtypes:
      for f in funcs:
        with self.assertRaisesWithPredicateMatch(
            ValueError, lambda e: "Dimensions must" in str(e)):
          f(x.astype(t), y.astype(t))


class LogicalOpTest(tf.test.TestCase):

  def _compareBinary(self, x, y, np_func, tf_func, use_gpu=False):
    np_ans = np_func(x, y)
    with self.test_session(use_gpu=use_gpu):
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_val = out.eval()
    self.assertEqual(out.dtype, tf.bool)
    self.assertAllEqual(np_ans, tf_val)
    self.assertShapeEqual(np_ans, out)

  def _not(self, x, use_gpu=False):
    np_ans = np.logical_not(x)
    with self.test_session(use_gpu=use_gpu):
      out = tf.logical_not(tf.convert_to_tensor(x))
      tf_val = out.eval()
    self.assertEqual(out.dtype, tf.bool)
    self.assertAllEqual(np_ans, tf_val)
    self.assertShapeEqual(np_ans, out)

  def testScalar(self):
    data = [np.array([True]), np.array([False])]
    for use_gpu in [True, False]:
      for x in data:
        self._not(x, use_gpu)
      for x in data:
        for y in data:
          self._compareBinary(
              x, y, np.logical_and, tf.logical_and, use_gpu)
          self._compareBinary(
              x, y, np.logical_or, tf.logical_or, use_gpu)
          self._compareBinary(
              x, y, np.logical_xor, tf.logical_xor, use_gpu)

  def testTensor(self):
    x = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    y = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    for use_gpu in [True, False]:
      self._not(x, use_gpu)
      self._compareBinary(x, y, np.logical_and, tf.logical_and, use_gpu)
      self._compareBinary(x, y, np.logical_or, tf.logical_or, use_gpu)
      self._compareBinary(x, y, np.logical_xor, tf.logical_xor, use_gpu)

  def testBCast(self):
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
    for (xs, ys) in shapes:
      x = np.random.randint(0, 2, np.prod(xs)).astype(np.bool).reshape(xs)
      y = np.random.randint(0, 2, np.prod(ys)).astype(np.bool).reshape(ys)
      for use_gpu in [True, False]:
        self._compareBinary(x, y, np.logical_and, tf.logical_and, use_gpu)
        self._compareBinary(x, y, np.logical_or, tf.logical_or, use_gpu)
        self._compareBinary(x, y, np.logical_xor, tf.logical_xor, use_gpu)

  def testShapeMismatch(self):
    x = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    y = np.random.randint(0, 2, 6).astype(np.bool).reshape(3, 2, 1)
    for f in [tf.logical_and, tf.logical_or, tf.logical_xor]:
      with self.assertRaisesWithPredicateMatch(
          ValueError, lambda e: "Dimensions must" in str(e)):
        f(x, y)

  def testUsingAsPythonValueFails(self):
    # Ensure that we raise an error when the user attempts to treat a
    # `Tensor` as a Python `bool`.
    b = tf.constant(False)
    with self.assertRaises(TypeError):
      if b:
        pass

    x = tf.constant(3)
    y = tf.constant(4)
    with self.assertRaises(TypeError):
      if x > y:
        pass

    z = tf.constant(7)

    # The chained comparison should fail because Python computes `x <
    # y` and short-circuits the comparison with `z` if it is `False`.
    with self.assertRaises(TypeError):
      _ = x < y < z


class SelectOpTest(tf.test.TestCase):

  def _compare(self, c, x, y, use_gpu):
    np_ans = np.where(c, x, y)
    with self.test_session(use_gpu=use_gpu):
      out = tf.select(c, x, y)
      tf_ans = out.eval()
    self.assertAllEqual(np_ans, tf_ans)
    self.assertShapeEqual(np_ans, out)

  def _compareGradientX(self, c, x, y, numeric_gradient_type=None):
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf.select(c, inx, iny)
      s = list(np.shape(c))
      jacob_t, jacob_n = tf.test.compute_gradient(inx,
                                                  s,
                                                  out,
                                                  s,
                                                  x_init_value=x)
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = tf.convert_to_tensor(xf)
        inyf = tf.convert_to_tensor(yf)
        outf = tf.select(c, inxf, inyf)
        _, jacob_n = tf.test.compute_gradient(inxf,
                                              s,
                                              outf,
                                              s,
                                              x_init_value=xf)
        jacob_n = jacob_n.astype(x.dtype)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _compareGradientY(self, c, x, y, numeric_gradient_type=None):
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf.select(c, inx, iny)
      s = list(np.shape(c))
      jacob_t, jacob_n = tf.test.compute_gradient(iny,
                                                  s,
                                                  out,
                                                  s,
                                                  x_init_value=y,
                                                  delta=1.0)
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = tf.convert_to_tensor(xf)
        inyf = tf.convert_to_tensor(yf)
        outf = tf.select(c, inxf, inyf)
        _, jacob_n = tf.test.compute_gradient(inyf,
                                              s,
                                              outf,
                                              s,
                                              x_init_value=yf)
        jacob_n = jacob_n.astype(x.dtype)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def testBasic(self):
    c = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 3, 2) * 100
    for t in [np.float16, np.float32, np.float64, np.int32, np.int64,
              np.complex64, np.complex128]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compare(c, xt, yt, use_gpu=False)
      if t in [np.float16, np.float32, np.float64]:
        self._compare(c, xt, yt, use_gpu=True)

  def testGradients(self):
    c = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 3, 2) * 100
    for t in [np.float16, np.float32, np.float64]:
      xt = x.astype(t)
      yt = y.astype(t)
      if t == np.float16:
        # Compare fp16 theoretical gradients to fp32 numerical gradients,
        # since fp16 numerical gradients are too imprecise unless great
        # care is taken with choosing the inputs and the delta. This is
        # a weaker check (in particular, it does not test the op itself,
        # only its gradient), but it's much better than nothing.
        self._compareGradientX(c, xt, yt, np.float)
        self._compareGradientY(c, xt, yt, np.float)
      else:
        self._compareGradientX(c, xt, yt)
        self._compareGradientY(c, xt, yt)

  def testShapeMismatch(self):
    c = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(2, 5, 3) * 100
    for t in [np.float16, np.float32, np.float64, np.int32, np.int64,
              np.complex64, np.complex128]:
      xt = x.astype(t)
      yt = y.astype(t)
      with self.assertRaises(ValueError):
        tf.select(c, xt, yt)

  def testEmptyTensor(self):
    c = np.random.randint(0, 3, 0).astype(np.bool).reshape(1, 3, 0)
    x = np.random.rand(1, 3, 0) * 100
    y = np.random.rand(1, 3, 0) * 100
    z_expected = np.zeros((1, 3, 0), dtype=np.float32)
    with self.test_session():
      xt = x.astype(np.float32)
      yt = y.astype(np.float32)
      z = tf.select(c, xt, yt).eval()
      self.assertAllEqual(z_expected, z)

  def testNan(self):
    """Verify that nans don't propagate where they shouldn't."""
    with self.test_session():
      for c in False, True:
        for a in 7.0, np.nan:
          for b in 5.0, np.nan:
            x = tf.select(c, a, b).eval()
            y = a if c else b
            self.assertEqual(np.isnan(x), np.isnan(y))


class BatchSelectOpTest(tf.test.TestCase):
  """Test broadcasting of Select when 'c' is a vec and 't' &'e' are rank2+."""

  def _compare(self, c, x, y, use_gpu):
    np_ans = np.dstack(
        [x_i if c_i else y_i for c_i, x_i, y_i in zip(c, x, y)]).transpose(
            [2, 0, 1])
    with self.test_session(use_gpu=use_gpu):
      out = tf.select(c, x, y)
      tf_ans = out.eval()
    self.assertAllEqual(np_ans, tf_ans)
    self.assertShapeEqual(np_ans, out)

  def _compareGradientX(self, c, x, y, numeric_gradient_type=None):
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf.select(c, inx, iny)
      s = list(np.shape(x))
      jacob_t, jacob_n = tf.test.compute_gradient(inx,
                                                  s,
                                                  out,
                                                  s,
                                                  x_init_value=x)
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = tf.convert_to_tensor(xf)
        inyf = tf.convert_to_tensor(yf)
        outf = tf.select(c, inxf, inyf)
        _, jacob_n = tf.test.compute_gradient(inxf,
                                              s,
                                              outf,
                                              s,
                                              x_init_value=xf)
        jacob_n = jacob_n.astype(x.dtype)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _compareGradientY(self, c, x, y, numeric_gradient_type=None):
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf.select(c, inx, iny)
      s = list(np.shape(x))
      jacob_t, jacob_n = tf.test.compute_gradient(iny,
                                                  s,
                                                  out,
                                                  s,
                                                  x_init_value=y)
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = tf.convert_to_tensor(xf)
        inyf = tf.convert_to_tensor(yf)
        outf = tf.select(c, inxf, inyf)
        _, jacob_n = tf.test.compute_gradient(inyf,
                                              s,
                                              outf,
                                              s,
                                              x_init_value=yf)
        jacob_n = jacob_n.astype(x.dtype)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def testBasic(self):
    c = np.random.randint(0, 2, 16).astype(np.bool)
    x = np.random.rand(16, 2, 8) * 100
    y = np.random.rand(16, 2, 8) * 100
    for t in [np.float16, np.float32, np.float64, np.int32, np.int64,
              np.complex64, np.complex128]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compare(c, xt, yt, use_gpu=False)
      if t in [np.float16, np.float32, np.float64]:
        self._compare(c, xt, yt, use_gpu=True)

  def testGradients(self):
    c = np.random.randint(0, 2, 16).astype(np.bool)
    x = np.random.rand(16, 2, 8) * 100
    y = np.random.rand(16, 2, 8) * 100
    for t in [np.float16, np.float32, np.float64]:
      xt = x.astype(t)
      yt = y.astype(t)
      if t == np.float16:
        # Compare fp16 theoretical gradients to fp32 numerical gradients,
        # since fp16 numerical gradients are too imprecise unless great
        # care is taken with choosing the inputs and the delta. This is
        # a weaker check (in particular, it does not test the op itself,
        # only its gradient), but it's much better than nothing.
        self._compareGradientX(c, xt, yt, np.float)
        self._compareGradientY(c, xt, yt, np.float)
      else:
        self._compareGradientX(c, xt, yt)
        self._compareGradientY(c, xt, yt)

  def testShapeMismatch(self):
    c = np.random.randint(0, 2, 8).astype(np.bool)
    x = np.random.rand(16, 3, 2) * 100
    y = np.random.rand(16, 3, 2) * 100
    for t in [np.float16, np.float32, np.float64, np.int32, np.int64,
              np.complex64, np.complex128]:
      xt = x.astype(t)
      yt = y.astype(t)
      with self.assertRaises(ValueError):
        tf.select(c, xt, yt)


class MinMaxOpTest(tf.test.TestCase):

  def _compare(self, x, y, use_gpu):
    np_min, np_max = np.minimum(x, y), np.maximum(x, y)
    with self.test_session(use_gpu=use_gpu) as sess:
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      omin, omax = tf.minimum(inx, iny), tf.maximum(inx, iny)
      tf_min, tf_max = sess.run([omin, omax])
    self.assertAllEqual(np_min, tf_min)
    self.assertAllEqual(np_max, tf_max)

  def testBasic(self):
    x = np.random.rand(1, 3, 2) * 100.
    y = np.random.rand(1, 3, 2) * 100.
    for t in [np.float16, np.float32, np.float64, np.int32, np.int64]:
      self._compare(x.astype(t), y.astype(t), use_gpu=False)
      self._compare(x.astype(t), y.astype(t), use_gpu=True)

  def testDifferentShapes(self):
    x = np.random.rand(1, 3, 2) * 100.
    y = np.random.rand(2) * 100.  # should broadcast
    for t in [np.float16, np.float32, np.float64, np.int32, np.int64]:
      self._compare(x.astype(t), y.astype(t), use_gpu=False)
      self._compare(x.astype(t), y.astype(t), use_gpu=True)

  def testScalar(self):
    x = np.random.rand(1, 3, 2) * 100.
    y = np.asscalar(np.random.rand(1) * 100.)  # should broadcast
    # dropped np.float64, int64 because TF automatically converts to 32 bit
    for t in [np.float32, np.int32]:
      self._compare(x.astype(t), t(y), use_gpu=False)
      self._compare(x.astype(t), t(y), use_gpu=True)

  def _compareGradientX(self, func, x, y):
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = func(inx, iny)
      s = list(np.shape(x))
      jacob_t, jacob_n = tf.test.compute_gradient(inx,
                                                  s,
                                                  out,
                                                  s,
                                                  x_init_value=x)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _compareGradientY(self, func, x, y):
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = func(inx, iny)
      s = list(np.shape(x))
      jacob_t, jacob_n = tf.test.compute_gradient(iny,
                                                  s,
                                                  out,
                                                  s,
                                                  x_init_value=y)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def testGradients(self):
    x = np.random.rand(1, 3, 2) * 100.
    # ensure x != y
    y = x + (np.random.randint(2, size=x.shape) - .5) * 2  # -1 or +1
    self._compareGradientX(tf.maximum, x, y)
    self._compareGradientY(tf.maximum, x, y)
    self._compareGradientX(tf.minimum, x, y)
    self._compareGradientY(tf.minimum, x, y)


class MathOpsOverloadTest(tf.test.TestCase):

  def _computeTensorAndLiteral(self, x, y, dtype, func):
    with self.test_session(use_gpu=False):
      inx = tf.convert_to_tensor(x, dtype=dtype)
      z = func(inx, y)  # Should use __add__, __sub__, etc.
      return z.eval()

  def _computeLiteralAndTensor(self, x, y, dtype, func):
    with self.test_session(use_gpu=False):
      iny = tf.convert_to_tensor(y, dtype=dtype)
      z = func(x, iny)  # Should use __radd__, __rsub__, etc.
      return z.eval()

  def _compareBinary(self, x, y, dtype, np_func, tf_func):
    np_ans = np_func(x, y).astype(dtype.as_numpy_dtype)
    self.assertAllClose(np_ans, self._computeTensorAndLiteral(
        x, y, dtype, tf_func))
    self.assertAllClose(np_ans, self._computeLiteralAndTensor(
        x, y, dtype, tf_func))

  def _compareUnary(self, x, dtype, np_func, tf_func):
    np_ans = np_func(x).astype(dtype.as_numpy_dtype)
    with self.test_session(use_gpu=False):
      self.assertAllClose(np_ans, tf_func(tf.convert_to_tensor(x, dtype=dtype)).eval())

  def testOverload(self):
    dtypes = [
        tf.float16,
        tf.float32,
        tf.float64,
        tf.int32,
        tf.int64,
        tf.complex64,
        tf.complex128,
    ]
    funcs = [
        (np.add, _ADD),
        (np.subtract, _SUB),
        (np.multiply, _MUL),
        (np.power, _POW),
        (np.true_divide, _TRUEDIV),
        (np.floor_divide, _FLOORDIV),
    ]
    for dtype in dtypes:
      for np_func, tf_func in funcs:
        if dtype in (tf.complex64, tf.complex128) and tf_func == _FLOORDIV:
          continue  # floordiv makes no sense for complex
        self._compareBinary(10, 5, dtype, np_func, tf_func)
    # Mod only works for int32 and int64.
    for dtype in [tf.int32, tf.int64]:
      self._compareBinary(10, 3, dtype, np.mod, _MOD)

  def testOverloadComparisons(self):
    dtypes = [
        tf.float16,
        tf.float32,
        tf.float64,
        tf.int32,
        tf.int64,
    ]
    funcs = [
        (np.less, _LT),
        (np.less_equal, _LE),
        (np.greater, _GT),
        (np.greater_equal, _GE),
    ]
    for dtype in dtypes:
      for np_func, tf_func in funcs:
        self._compareBinary(10, 5, dtype, np_func, tf_func)
    logical_funcs = [
        (np.logical_and, _AND),
        (np.logical_or, _OR),
        (np.logical_xor, _XOR),
        (np.equal, tf.equal),
        (np.not_equal, tf.not_equal)
    ]
    for np_func, tf_func in logical_funcs:
      self._compareBinary(True, False, tf.bool, np_func, tf_func)
      self._compareBinary(True, True, tf.bool, np_func, tf_func)
      self._compareBinary(False, False, tf.bool, np_func, tf_func)
      self._compareBinary(False, True, tf.bool, np_func, tf_func)
      self._compareBinary([True, True, False, False],
                          [True, False, True, False],
                          tf.bool, np_func, tf_func)
    self._compareUnary(True, tf.bool, np.logical_not, _INV)
    self._compareUnary(False, tf.bool, np.logical_not, _INV)
    self._compareUnary([True, False], tf.bool, np.logical_not, _INV)


class IsFiniteInfNanTest(tf.test.TestCase):

  def _compare(self, x, use_gpu):
    np_finite, np_inf, np_nan = np.isfinite(x), np.isinf(x), np.isnan(x)
    with self.test_session(use_gpu=use_gpu) as sess:
      inx = tf.convert_to_tensor(x)
      ofinite, oinf, onan = tf.is_finite(inx), tf.is_inf(
          inx), tf.is_nan(inx)
      tf_finite, tf_inf, tf_nan = sess.run([ofinite, oinf, onan])
    self.assertAllEqual(np_inf, tf_inf)
    self.assertAllEqual(np_nan, tf_nan)
    self.assertAllEqual(np_finite, tf_finite)
    self.assertShapeEqual(np_inf, oinf)
    self.assertShapeEqual(np_nan, onan)
    self.assertShapeEqual(np_finite, ofinite)

  def _testDtype(self, dtype):
    fi = np.finfo(dtype)
    data = np.array([0, -1, 1, fi.resolution, -fi.resolution, fi.min, fi.max,
                     -np.inf, np.inf, np.nan]).astype(dtype)
    self._compare(data, use_gpu=False)
    self._compare(data, use_gpu=True)

  def testHalf(self):
    self._testDtype(np.float16)

  def testFloat(self):
    self._testDtype(np.float32)

  def testDouble(self):
    self._testDtype(np.float64)


class RoundingTest(tf.test.TestCase):

  def _compare(self, x, use_gpu):
    np_floor, np_ceil = np.floor(x), np.ceil(x)
    with self.test_session(use_gpu=use_gpu) as sess:
      inx = tf.convert_to_tensor(x)
      ofloor, oceil = tf.floor(inx), tf.ceil(inx)
      tf_floor, tf_ceil = sess.run([ofloor, oceil])
    self.assertAllEqual(np_floor, tf_floor)
    self.assertAllEqual(np_ceil, tf_ceil)
    self.assertShapeEqual(np_floor, ofloor)
    self.assertShapeEqual(np_ceil, oceil)

  def _testDtype(self, dtype):
    data = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(dtype)
    self._compare(data, use_gpu=True)
    self._compare(data, use_gpu=True)

  def testTypes(self):
    for dtype in [np.float16, np.float32, np.float64]:
      self._testDtype(dtype)


class ComplexMakeRealImagTest(tf.test.TestCase):

  def _compareMake(self, real, imag, use_gpu):
    np_ans = real + (1j) * imag
    with self.test_session(use_gpu=use_gpu):
      real = tf.convert_to_tensor(real)
      imag = tf.convert_to_tensor(imag)
      tf_ans = tf.complex(real, imag)
      out = tf_ans.eval()
    self.assertAllEqual(np_ans, out)
    self.assertShapeEqual(np_ans, tf_ans)

  def testMake(self):
    real = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(np.float32)
    imag = (np.arange(-3, 3) / 5.).reshape([1, 3, 2]).astype(np.float32)
    for use_gpu in [False, True]:
      self._compareMake(real, imag, use_gpu)
      self._compareMake(real, 12.0, use_gpu)
      self._compareMake(23.0, imag, use_gpu)

  def _compareRealImag(self, cplx, use_gpu):
    np_real, np_imag = np.real(cplx), np.imag(cplx)
    with self.test_session(use_gpu=use_gpu) as sess:
      inx = tf.convert_to_tensor(cplx)
      tf_real = tf.real(inx)
      tf_imag = tf.imag(inx)
      tf_real_val, tf_imag_val = sess.run([tf_real, tf_imag])
    self.assertAllEqual(np_real, tf_real_val)
    self.assertAllEqual(np_imag, tf_imag_val)
    self.assertShapeEqual(np_real, tf_real)
    self.assertShapeEqual(np_imag, tf_imag)

  def testRealImag64(self):
    real = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(np.float32)
    imag = (np.arange(-3, 3) / 5.).reshape([1, 3, 2]).astype(np.float32)
    cplx = real + 1j * imag
    self._compareRealImag(cplx, use_gpu=False)
    self._compareRealImag(cplx, use_gpu=True)

  def testRealImag128(self):
    real = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(np.float64)
    imag = (np.arange(-3, 3) / 5.).reshape([1, 3, 2]).astype(np.float64)
    cplx = real + 1j * imag
    self._compareRealImag(cplx, use_gpu=False)
    self._compareRealImag(cplx, use_gpu=True)

  def testRealReal(self):
    for dtype in tf.int32, tf.int64, tf.float32, tf.float64:
      x = tf.placeholder(dtype)
      y = tf.real(x)
      self.assertEqual(x, y)

  def _compareConj(self, cplx, use_gpu):
    np_ans = np.conj(cplx)
    with self.test_session(use_gpu=use_gpu):
      inx = tf.convert_to_tensor(cplx)
      tf_conj = tf.conj(inx)
      tf_ans = tf_conj.eval()
    self.assertAllEqual(np_ans, tf_ans)
    self.assertShapeEqual(np_ans, tf_conj)

  def testConj64(self):
    real = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(np.float32)
    imag = (np.arange(-3, 3) / 5.).reshape([1, 3, 2]).astype(np.float32)
    cplx = real + 1j * imag
    self._compareConj(cplx, use_gpu=False)
    self._compareConj(cplx, use_gpu=True)

  def testConj128(self):
    real = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(np.float64)
    imag = (np.arange(-3, 3) / 5.).reshape([1, 3, 2]).astype(np.float64)
    cplx = real + 1j * imag
    self._compareConj(cplx, use_gpu=False)
    self._compareConj(cplx, use_gpu=True)

  def testConjReal(self):
    for dtype in tf.int32, tf.int64, tf.float16, tf.float32, tf.float64:
      x = tf.placeholder(dtype)
      y = tf.conj(x)
      self.assertEqual(x, y)

  def testConjString(self):
    x = tf.placeholder(tf.string)
    with self.assertRaisesRegexp(TypeError, r"Expected numeric tensor"):
      tf.conj(x)

  def _compareGradient(self, x):
    # x[:, 0] is real, x[:, 1] is imag.  We combine real and imag into
    # complex numbers. Then, we extract real and imag parts and
    # computes the squared sum. This is obviously the same as sum(real
    # * real) + sum(imag * imag). We just want to make sure the
    # gradient function is checked.
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      real, imag = tf.split(1, 2, inx)
      real, imag = tf.reshape(real, [-1]), tf.reshape(imag, [-1])
      cplx = tf.complex(real, imag)
      cplx = tf.conj(cplx)
      loss = tf.reduce_sum(
          tf.square(tf.real(cplx))) + tf.reduce_sum(
              tf.square(tf.imag(cplx)))
      epsilon = 1e-3
      jacob_t, jacob_n = tf.test.compute_gradient(inx,
                                                  list(x.shape),
                                                  loss,
                                                  [1],
                                                  x_init_value=x,
                                                  delta=epsilon)
    self.assertAllClose(jacob_t, jacob_n, rtol=epsilon, atol=epsilon)

  def _compareBroadcastGradient(self, x):
    x_ = tf.convert_to_tensor(x)
    epsilon = 1e-3
    with self.test_session():
      for args in [(x_, 0.), (0., x_)]:
        z = tf.reduce_sum(tf.complex_abs(tf.complex(*args)))
        jacob_t, jacob_n = tf.test.compute_gradient(x_,
                                                    list(x.shape),
                                                    z,
                                                    [1],
                                                    x_init_value=x,
                                                    delta=epsilon)
        self.assertAllClose(jacob_t, jacob_n, rtol=epsilon, atol=epsilon)

  def testGradient(self):
    # complex64
    data = np.arange(1, 2, 0.10).reshape([5, 2]).astype(np.float32)
    self._compareGradient(data)
    self._compareBroadcastGradient(data)
    # complex128
    data = np.arange(1, 2, 0.10).reshape([5, 2]).astype(np.float64)
    self._compareGradient(data)

  def _compareMulGradient(self, data):
    # data is a float matrix of shape [n, 4].  data[:, 0], data[:, 1],
    # data[:, 2], data[:, 3] are real parts of x, imaginary parts of
    # x, real parts of y and imaginary parts of y.
    with self.test_session():
      inp = tf.convert_to_tensor(data)
      xr, xi, yr, yi = tf.split(1, 4, inp)

      def vec(x):  # Reshape to a vector
        return tf.reshape(x, [-1])
      xr, xi, yr, yi = vec(xr), vec(xi), vec(yr), vec(yi)

      def cplx(r, i):  # Combine to a complex vector
        return tf.complex(r, i)
      x, y = cplx(xr, xi), cplx(yr, yi)
      # z is x times y in complex plane.
      z = x * y
      # Defines the loss function as the sum of all coefficients of z.
      loss = tf.reduce_sum(tf.real(z) + tf.imag(z))
      epsilon = 0.005
      jacob_t, jacob_n = tf.test.compute_gradient(inp,
                                                  list(data.shape),
                                                  loss,
                                                  [1],
                                                  x_init_value=data,
                                                  delta=epsilon)
    self.assertAllClose(jacob_t, jacob_n, rtol=epsilon, atol=epsilon)

  def testMulGradient(self):
    data = np.arange(1, 2, 0.125).reshape([2, 4]).astype(np.float32)
    self._compareMulGradient(data)


class AccumulateTest(tf.test.TestCase):

  def testSimple(self):
    with self.test_session():
      random_arrays = [np.random.rand(16, 16, 16, 16).astype(np.float32)
                       for _ in range(20)]
      random_tensors = [tf.convert_to_tensor(x, dtype=tf.float32)
                        for x in random_arrays]
      tf_val = tf.accumulate_n(random_tensors)
      np_val = random_arrays[0]
      for random_array in random_arrays[1:]:
        np_val += random_array
      self.assertAllClose(np_val, tf_val.eval())

  def testZeroArgs(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tf_val = tf.accumulate_n([])
        tf_val.eval()

if __name__ == "__main__":
  tf.test.main()
