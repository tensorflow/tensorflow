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
"""Tests for tensorflow.ops.tf.BatchMatMul."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import tf2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class BatchMatmulOpTest(test.TestCase):

  # Uses numpy to compute batch_matmul(x, y, adjoint_a, adjoint_b).
  def _npBatchMatmul(self, x, y, adjoint_a, adjoint_b):
    # output's shape depends on adj[0] and adj[1]
    d0 = x.shape[-2] if not adjoint_a else x.shape[-1]
    d2 = y.shape[-1] if not adjoint_b else y.shape[-2]
    batch_dims = x.shape[:-2]
    num = np.prod(batch_dims)
    z = np.empty(list(batch_dims) + [d0, d2], dtype=x.dtype)
    xr = x.reshape([num, x.shape[-2], x.shape[-1]])
    yr = y.reshape([num, y.shape[-2], y.shape[-1]])
    zr = z.reshape([num, z.shape[-2], z.shape[-1]])
    for i in range(num):
      a = np.matrix(xr[i, :, :])
      if adjoint_a:
        a = a.transpose().conj()
      b = np.matrix(yr[i, :, :])
      if adjoint_b:
        b = b.transpose().conj()
      zr[i, :, :] = a * b
    return z

  # Test _npBatchMatMul works.
  def testNpVersion(self):
    x = np.array([0., 1., 2., 3.]).reshape([1, 2, 2])
    y = np.array([1., 2., 3., 4.]).reshape([1, 2, 2])
    z0 = self._npBatchMatmul(x, y, False, False)
    z1 = np.array([3., 4., 11., 16.]).reshape([1, 2, 2])
    self.assertTrue(np.array_equal(z0, z1))

    x = np.array([1., (1j), (-1.), (-1j)]).reshape([1, 2, 2])
    y = x * np.complex(1, 1)  # rotate x 90 degree
    z0 = self._npBatchMatmul(x, y, False, False)
    z1 = np.array([2., (2.j), -2., (-2.j)]).reshape([1, 2, 2])
    self.assertTrue(np.array_equal(z0, z1))

    z0 = self._npBatchMatmul(x, y, False, True)
    z1 = np.array([(2. - 2.j), (-2. + 2.j), (-2. + 2.j), (2. - 2.j)]).reshape(
        [1, 2, 2])
    self.assertTrue(np.array_equal(z0, z1))

    z0 = self._npBatchMatmul(x, y, True, False)
    z1 = np.array([(2. + 2.j), (-2. + 2.j), (2. - 2.j), (2. + 2.j)]).reshape(
        [1, 2, 2])
    self.assertTrue(np.array_equal(z0, z1))

  # Compares _tfpBatchMatmul(x, y, alpha, adj) and _npBatchMatMul(x, y, alpha,
  # adj)
  def _compare(self, x_in, y_in, adjoint_a, adjoint_b, static_shape=True):
    x_t_shape = x_in.shape[:-2] + (x_in.shape[-1], x_in.shape[-2])
    y_t_shape = y_in.shape[:-2] + (y_in.shape[-1], y_in.shape[-2])
    x = x_in if not adjoint_a else x_in.reshape(x_t_shape)
    y = y_in if not adjoint_b else y_in.reshape(y_t_shape)
    is_floating = x.dtype != np.int32
    tol = 100 * np.finfo(x.dtype).eps if is_floating else 0
    with self.cached_session(use_gpu=is_floating) as sess:
      if static_shape:
        z0 = math_ops.matmul(x, y, adjoint_a=adjoint_a, adjoint_b=adjoint_b)
        z0_val = self.evaluate(z0)
      else:
        x_ph = array_ops.placeholder(x.dtype)
        y_ph = array_ops.placeholder(y.dtype)
        z0 = math_ops.matmul(
            x_ph, y_ph, adjoint_a=adjoint_a, adjoint_b=adjoint_b)
        z0_val = sess.run(z0, feed_dict={x_ph: x, y_ph: y})
      z1 = self._npBatchMatmul(x, y, adjoint_a, adjoint_b)
      self.assertAllClose(z0_val, z1, rtol=tol, atol=tol)

  def _rand(self, shape, dtype):
    vals = np.array(np.random.normal(-10, 10, np.prod(shape)), dtype=dtype)
    if dtype in (np.complex64, np.complex128):
      imag = np.array(np.random.normal(-10, 10, np.prod(shape)), dtype=dtype)
      vals += 1j * imag
    return vals.reshape(shape)

  def _testNonEmpty(self, dtype, adjoint_a, adjoint_b, use_static_shape):

    def CompareNonEmpty(self, a_shape, b_shape):
      self._compare(
          self._rand(a_shape, dtype),
          self._rand(b_shape, dtype), adjoint_a, adjoint_b, use_static_shape)

    CompareNonEmpty(self, [1, 2, 3], [1, 3, 5])
    CompareNonEmpty(self, [1, 2, 3], [1, 3, 1])
    CompareNonEmpty(self, [1, 1, 3], [1, 3, 5])
    CompareNonEmpty(self, [1, 2, 3], [1, 3, 5])
    CompareNonEmpty(self, [7, 1, 3], [7, 3, 5])
    CompareNonEmpty(self, [7, 2, 3], [7, 3, 1])
    CompareNonEmpty(self, [7, 2, 3], [7, 3, 5])
    CompareNonEmpty(self, [10, 64, 75], [10, 75, 30])
    CompareNonEmpty(self, [5, 7, 2, 3], [5, 7, 3, 5])

  def _testEmpty(self, dtype, adjoint_a, adjoint_b, use_static_shape):

    def CompareEmpty(self, a_shape, b_shape):
      self._compare(
          np.zeros(a_shape).astype(dtype),
          np.zeros(b_shape).astype(dtype), adjoint_a, adjoint_b,
          use_static_shape)

    CompareEmpty(self, [0, 3, 2], [0, 2, 4])
    CompareEmpty(self, [3, 0, 2], [3, 2, 5])
    CompareEmpty(self, [3, 3, 2], [3, 2, 0])


def _GetBatchMatmulOpTest(dtype, adjoint_a, adjoint_b, use_static_shape):

  def Test(self):
    np.random.seed(42)
    self._testNonEmpty(dtype, adjoint_a, adjoint_b, use_static_shape)
    self._testEmpty(dtype, adjoint_a, adjoint_b, use_static_shape)

  return Test


class BatchMatmulGradientTest(test.TestCase):

  # loss = sum(batch_matmul(x, y)). Verify dl/dx and dl/dy via the
  # gradient checker.
  def _checkGrad(self, x_in, y_in, adjoint_a, adjoint_b):
    x_t_shape = x_in.shape[:-2] + (x_in.shape[-1], x_in.shape[-2])
    y_t_shape = y_in.shape[:-2] + (y_in.shape[-1], y_in.shape[-2])
    x = x_in if not adjoint_a else x_in.reshape(x_t_shape)
    y = y_in if not adjoint_b else y_in.reshape(y_t_shape)
    epsilon = np.finfo(x.dtype).eps
    delta = epsilon**(1.0 / 3.0)
    def Loss(x, y):
      z = math_ops.matmul(x, y, adjoint_a, adjoint_b)
      return math_ops.reduce_sum(z)
    with self.cached_session(use_gpu=True):
      ((x_jacob_t, y_jacob_t),
       (x_jacob_n, y_jacob_n)) = gradient_checker_v2.compute_gradient(
           Loss, [x, y], delta=delta)
      tol = 20 * delta
      self.assertAllClose(x_jacob_t, x_jacob_n, rtol=tol, atol=tol)
      self.assertAllClose(y_jacob_t, y_jacob_n, rtol=tol, atol=tol)

  # Tests a batched matmul of x, and y: x is a 3D tensor of shape [b,
  # n, k] y is a 3D tensor of shape [b, k, m] the batched matmul
  # computes z of shape [b, n, m], where z[i, :, :] = x[i, :, :]
  # matmul y[i, :, :]
  def _compare(self, b, n, k, m, dtype, adjoint_a, adjoint_b):
    np.random.seed(42)
    x = np.random.normal(0, 1, b * n * k).astype(dtype).reshape([b, n, k])
    if dtype in (np.complex64, np.complex128):
      x.imag = np.random.normal(0, 1,
                                b * n * k).astype(dtype).reshape([b, n, k])
    y = np.random.normal(0, 1, b * k * m).astype(dtype).reshape([b, k, m])
    if dtype in (np.complex64, np.complex128):
      y.imag = np.random.normal(0, 1,
                                b * k * m).astype(dtype).reshape([b, k, m])
    self._checkGrad(x, y, adjoint_a, adjoint_b)


def _GetBatchMatmulGradientTest(dtype, adjoint_a, adjoint_b):

  def Test(self):
    self._compare(1, 2, 3, 5, dtype, adjoint_a, adjoint_b)
    self._compare(3, 4, 7, 10, dtype, adjoint_a, adjoint_b)

  return Test


if __name__ == "__main__":
  for dtype_ in [
      np.float16, np.float32, np.float64, np.complex64, np.complex128, np.int32
  ]:
    for adjoint_a_ in False, True:
      for adjoint_b_ in False, True:
        name = "%s_%s_%s" % (dtype_.__name__, adjoint_a_, adjoint_b_)
        for use_static_shape_ in set([True, tf2.enabled()]):
          setattr(BatchMatmulOpTest,
                  "testBatchMatmulOp_" + name + ("_%s" % use_static_shape_),
                  _GetBatchMatmulOpTest(dtype_, adjoint_a_, adjoint_b_,
                                        use_static_shape_))
        if dtype_ is not np.int32:
          setattr(BatchMatmulGradientTest, "testBatchMatmulGradient_" + name,
                  _GetBatchMatmulGradientTest(dtype_, adjoint_a_, adjoint_b_))
  test.main()
