# Copyright 2015 Google Inc. All Rights Reserved.
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
    assert x.ndim >= 3
    assert y.ndim >= 3
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
  def testSimpleNpVersion(self):
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
    z1 = np.array([(2.-2.j), (-2.+2.j), (-2.+2.j), (2.-2.j)]).reshape([1, 2, 2])
    self.assertTrue(np.array_equal(z0, z1))

    z0 = self._npBatchMatmul(x, y, True, False)
    z1 = np.array([(2.+2.j), (-2.+2.j), (2.-2.j), (2.+2.j)]).reshape([1, 2, 2])
    self.assertTrue(np.array_equal(z0, z1))

  # Compares _tfpBatchMatmul(x, y, alpha, adj) and _npBatchMatMul(x, y, alpha,
  # adj)
  def _compare(self, x, y, adj_x, adj_y, use_gpu=False):
    with self.test_session(use_gpu=use_gpu):
      z0 = tf.batch_matmul(x, y, adj_x=adj_x, adj_y=adj_y)
      z0_val = z0.eval()
    z1 = self._npBatchMatmul(x, y, adj_x, adj_y)
    self.assertShapeEqual(z1, z0)
    if z0_val.size != 0:
      err = (np.abs(z0_val - z1) / np.maximum(1, np.abs(z0_val))).max()
      tf.logging.info("error = %f", err)
      self.assertTrue(err < 1e-4)

  # Returns a random float np of "shape".
  def _randFloat(self, shape):
    vals = np.random.normal(0, 1, np.prod(shape)).reshape(shape)
    return np.array(vals, dtype=np.float32)

  def testSimpleFloat(self):
    for use_gpu in [False, True]:
      self._compare(self._randFloat([7, 2, 3]), self._randFloat([7, 3, 5]),
                    False, False, use_gpu)
      self._compare(self._randFloat([7, 2, 3]), self._randFloat([7, 5, 3]),
                    False, True, use_gpu)
      self._compare(self._randFloat([7, 3, 2]), self._randFloat([7, 3, 5]),
                    True, False, use_gpu)
      self._compare(self._randFloat([7, 3, 2]), self._randFloat([7, 5, 3]),
                    True, True, use_gpu)

  def testLargeFloat(self):
    for use_gpu in [False, True]:
      self._compare(self._randFloat([10, 64, 75]),
                    self._randFloat([10, 75, 30]), False, False, use_gpu)
      self._compare(self._randFloat([10, 75, 64]),
                    self._randFloat([10, 75, 30]), True, False, use_gpu)
      self._compare(self._randFloat([10, 64, 75]),
                    self._randFloat([10, 30, 75]), False, True, use_gpu)
      self._compare(self._randFloat([10, 75, 64]),
                    self._randFloat([10, 30, 75]), True, True, use_gpu)

  def testHighNDims(self):
    for use_gpu in [False, True]:
      self._compare(self._randFloat([5, 7, 2, 3]),
                    self._randFloat([5, 7, 3, 5]), False, False, use_gpu)
      self._compare(self._randFloat([5, 7, 3, 2]),
                    self._randFloat([5, 7, 3, 5]), True, False, use_gpu)
      self._compare(self._randFloat([5, 7, 2, 3]),
                    self._randFloat([5, 7, 5, 3]), False, True, use_gpu)
      self._compare(self._randFloat([5, 7, 3, 2]),
                    self._randFloat([5, 7, 5, 3]), True, True, use_gpu)

  # Returns a random complex numpy array of "shape".
  def _randComplex(self, shape):
    real = np.random.normal(0, 1, np.prod(shape))
    imag = np.random.normal(0, 1, np.prod(shape))
    vals = [np.complex(v[0], v[1]) for v in zip(real, imag)]
    return np.array(vals, dtype=np.complex64).reshape(shape)

  def testSimpleComplex(self):
    self._compare(self._randComplex([7, 2, 3]),
                  self._randComplex([7, 3, 5]), False, False)
    self._compare(self._randComplex([7, 2, 3]),
                  self._randComplex([7, 5, 3]), False, True)
    self._compare(self._randComplex([7, 3, 2]),
                  self._randComplex([7, 3, 5]), True, False)
    self._compare(self._randComplex([7, 3, 2]),
                  self._randComplex([7, 5, 3]), True, True)

  def testLargeComplex(self):
    self._compare(self._randComplex([10, 64, 75]),
                  self._randComplex([10, 75, 30]), False,
                  False)
    self._compare(self._randComplex([10, 64, 75]),
                  self._randComplex([10, 30, 75]), False, True)
    self._compare(self._randComplex([10, 75, 64]),
                  self._randComplex([10, 75, 30]), True, False)
    self._compare(self._randComplex([10, 75, 64]),
                  self._randComplex([10, 30, 75]), True, True)

  def testEmpty(self):
    self._compare(np.zeros([0, 3, 2]).astype(np.float32),
                  np.zeros([0, 2, 4]).astype(np.float32), False, False)
    self._compare(np.zeros([3, 2, 0]).astype(np.float32),
                  np.zeros([3, 0, 5]).astype(np.float32), False, False)
    self._compare(np.zeros([3, 0, 2]).astype(np.float32),
                  np.zeros([3, 2, 5]).astype(np.float32), False, False)
    self._compare(np.zeros([3, 3, 2]).astype(np.float32),
                  np.zeros([3, 2, 0]).astype(np.float32), False, False)


class BatchMatmulGradientTest(tf.test.TestCase):

  # loss = sum(batch_matmul(x, y)). Verify dl/dx and dl/dy via the
  # gradient checker.
  def _checkGrad(self, x, y, adj_x, adj_y):
    assert 3 == x.ndim
    assert 3 == y.ndim
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      z = tf.batch_matmul(inx, iny, adj_x, adj_y)
      loss = tf.reduce_sum(z)
      epsilon = 1e-2
      ((x_jacob_t, x_jacob_n),
       (y_jacob_t, y_jacob_n)) = tf.test.compute_gradient(
           [inx, iny],
           [x.shape, y.shape],
           loss,
           [1],
           x_init_value=[x, y],
           delta=epsilon)

    tf.logging.info("x_jacob_t = %s", x_jacob_t.reshape(x.shape))
    tf.logging.info("x_jacob_n = %s", x_jacob_n.reshape(x.shape))
    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=1e-2, atol=epsilon)
    tf.logging.info("y_jacob_t = %s", y_jacob_t.reshape(y.shape))
    tf.logging.info("y_jacob_n = %s", y_jacob_n.reshape(y.shape))
    self.assertAllClose(y_jacob_t, y_jacob_n, rtol=1e-2, atol=epsilon)

  # Tests a batched matmul of x, and y: x is a 3D tensor of shape [b,
  # n, k] y is a 3D tensor of shape [b, k, m] the batched matmul
  # computes z of shape [b, n, m], where z[i, :, :] = x[i, :, :]
  # matmul y[i, :, :]
  def _compare(self, b, n, k, m):
    x = np.random.normal(0, 1, b * n * k).astype(np.float32).reshape([b, n, k])
    y = np.random.normal(0, 1, b * k * m).astype(np.float32).reshape([b, k, m])
    self._checkGrad(x, y, False, False)
    self._checkGrad(x.reshape([b, k, n]), y, True, False)
    self._checkGrad(x, y.reshape([b, m, k]), False, True)
    self._checkGrad(x.reshape([b, k, n]), y.reshape([b, m, k]), True, True)

  def testSmall(self):
    self._compare(1, 2, 3, 5)

  def testMedium(self):
    self._compare(3, 4, 7, 10)

  # Can't do testLarge using very large inputs because gradient
  # checker will take way too long time.


if __name__ == "__main__":
  tf.test.main()
