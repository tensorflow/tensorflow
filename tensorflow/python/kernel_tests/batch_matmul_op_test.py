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
import tensorflow as tf


class BatchMatmulOpTest(tf.test.TestCase):

  # Uses numpy to compute batch_matmul(x, y, adj_x, adj_y).
  def _npBatchMatmul(self, x, y, adj_x, adj_y):
    # output's shape depends on adj[0] and adj[1]
    d0 = x.shape[-2] if not adj_x else x.shape[-1]
    d2 = y.shape[-1] if not adj_y else y.shape[-2]
    batch_dims = x.shape[:-2]
    num = np.prod(batch_dims)
    z = np.empty(list(batch_dims) + [d0, d2], dtype=x.dtype)
    xr = x.reshape([num, x.shape[-2], x.shape[-1]])
    yr = y.reshape([num, y.shape[-2], y.shape[-1]])
    zr = z.reshape([num, z.shape[-2], z.shape[-1]])
    for i in range(num):
      a = np.matrix(xr[i, :, :])
      if adj_x:
        a = a.transpose().conj()
      b = np.matrix(yr[i, :, :])
      if adj_y:
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
  def _compare(self, x_in, y_in, adj_x, adj_y):
    x_t_shape = x_in.shape[:-2] + (x_in.shape[-1], x_in.shape[-2])
    y_t_shape = y_in.shape[:-2] + (y_in.shape[-1], y_in.shape[-2])
    x = x_in if not adj_x else x_in.reshape(x_t_shape)
    y = y_in if not adj_y else y_in.reshape(y_t_shape)
    is_floating = x.dtype != np.int32
    tol = 100 * np.finfo(x.dtype).eps if is_floating else 0

    with self.test_session(use_gpu=is_floating):
      z0 = tf.batch_matmul(x, y, adj_x=adj_x, adj_y=adj_y)
      z0_val = z0.eval()
      z1 = self._npBatchMatmul(x, y, adj_x, adj_y)
      self.assertAllClose(z0_val, z1, rtol=tol, atol=tol)

  def _rand(self, shape, dtype):
    vals = np.array(np.random.normal(-10, 10, np.prod(shape)), dtype=dtype)
    if dtype in (np.complex64, np.complex128):
      imag = np.array(np.random.normal(-10, 10, np.prod(shape)), dtype=dtype)
      vals += 1j * imag
    return vals.reshape(shape)

  def _testNonEmpty(self, dtype, adj_x, adj_y):
    self._compare(
        self._rand([7, 2, 3], dtype), self._rand([7, 3, 5], dtype), adj_x,
        adj_y)
    self._compare(
        self._rand([10, 64, 75], dtype), self._rand([10, 75, 30], dtype), adj_x,
        adj_y)
    self._compare(
        self._rand([5, 7, 2, 3], dtype), self._rand([5, 7, 3, 5], dtype), adj_x,
        adj_y)

  def _testEmpty(self, dtype, adj_x, adj_y):
    self._compare(
        np.zeros([0, 3, 2]).astype(dtype), np.zeros([0, 2, 4]).astype(dtype),
        adj_x, adj_y)
    self._compare(
        np.zeros([3, 0, 2]).astype(dtype), np.zeros([3, 2, 5]).astype(dtype),
        adj_x, adj_y)
    self._compare(
        np.zeros([3, 3, 2]).astype(dtype), np.zeros([3, 2, 0]).astype(dtype),
        adj_x, adj_y)


def _GetBatchMatmulOpTest(dtype, adj_x, adj_y):

  def Test(self):
    self._testNonEmpty(dtype, adj_x, adj_y)
    self._testEmpty(dtype, adj_x, adj_y)

  return Test


class BatchMatmulGradientTest(tf.test.TestCase):

  # loss = sum(batch_matmul(x, y)). Verify dl/dx and dl/dy via the
  # gradient checker.
  def _checkGrad(self, x_in, y_in, adj_x, adj_y):
    x_t_shape = x_in.shape[:-2] + (x_in.shape[-1], x_in.shape[-2])
    y_t_shape = y_in.shape[:-2] + (y_in.shape[-1], y_in.shape[-2])
    x = x_in if not adj_x else x_in.reshape(x_t_shape)
    y = y_in if not adj_y else y_in.reshape(y_t_shape)
    epsilon = np.finfo(x.dtype).eps
    delta = epsilon**(1.0 / 3.0)
    with self.test_session(use_gpu=True):
      inx = tf.constant(x)
      iny = tf.constant(y)
      z = tf.batch_matmul(inx, iny, adj_x, adj_y)
      loss = tf.reduce_sum(z)
      ((x_jacob_t, x_jacob_n),
       (y_jacob_t, y_jacob_n)) = tf.test.compute_gradient(
           [inx, iny], [x.shape, y.shape],
           loss, [1],
           x_init_value=[x, y],
           delta=delta)
      tol = 20 * delta
      self.assertAllClose(x_jacob_t, x_jacob_n, rtol=tol, atol=tol)
      self.assertAllClose(y_jacob_t, y_jacob_n, rtol=tol, atol=tol)

  # Tests a batched matmul of x, and y: x is a 3D tensor of shape [b,
  # n, k] y is a 3D tensor of shape [b, k, m] the batched matmul
  # computes z of shape [b, n, m], where z[i, :, :] = x[i, :, :]
  # matmul y[i, :, :]
  def _compare(self, b, n, k, m, dtype, adj_x, adj_y):
    x = np.random.normal(0, 1, b * n * k).astype(dtype).reshape([b, n, k])
    y = np.random.normal(0, 1, b * k * m).astype(dtype).reshape([b, k, m])
    self._checkGrad(x, y, adj_x, adj_y)


def _GetBatchMatmulGradientTest(dtype, adj_x, adj_y):

  def Test(self):
    self._compare(1, 2, 3, 5, dtype, adj_x, adj_y)
    self._compare(3, 4, 7, 10, dtype, adj_x, adj_y)

  return Test


if __name__ == "__main__":
  for dtype_ in [np.float16, np.float32, np.float64, np.complex64,
                 np.complex128, np.int32]:
    for adj_x_ in False, True:
      for adj_y_ in False, True:
        name = "%s_%s_%s" % (dtype_.__name__, adj_x_, adj_y_)
        setattr(BatchMatmulOpTest, "testBatchMatmulOp_" + name,
                _GetBatchMatmulOpTest(dtype_, adj_x_, adj_y_))
        if dtype_ is not np.int32:
          setattr(BatchMatmulGradientTest, "testBatchMatmulGradient_" + name,
                  _GetBatchMatmulGradientTest(dtype_, adj_x_, adj_y_))
  tf.test.main()
