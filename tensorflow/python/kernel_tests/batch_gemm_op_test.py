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
"""Tests for tensorflow.ops.tf.BatchGemm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random

from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.compat import compat
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test

def GetRandomNormalInput(shape, dtype):
  # float16 has limited range so we reduce the variance of the scalars.
  scale = 10.0 if dtype != np.float16 else 0.1
  loc = -10.0 if dtype != np.float16 else 0.1
  vals = np.array(np.random.normal(loc, scale, np.prod(shape)), dtype=dtype)
  return vals.reshape(shape)


class BatchGemmOpTest(test.TestCase):

  # Uses numpy to compute batch_matmul(x, y, transpose_a, transpose_b).
  def _npBatchGemm(self, x, y, transpose_a, transpose_b):
    # output's shape depends on adj[0] and adj[1]
    if transpose_a:
      x = np.conjugate(np.swapaxes(x, -1, -2))
    if transpose_b:
      y = np.conjugate(np.swapaxes(y, -1, -2))
    return np.matmul(x, y)

  # Compares TensorFlow BatchGemm with NumPy's matmul.
  def _compare(self, x_in, y_in, transpose_a, transpose_b, alpha, beta, static_shape):
    x_t_shape = x_in.shape[:-2] + (x_in.shape[-1], x_in.shape[-2])
    y_t_shape = y_in.shape[:-2] + (y_in.shape[-1], y_in.shape[-2])
    x = x_in if not transpose_a else x_in.reshape(x_t_shape)
    y = y_in if not transpose_b else y_in.reshape(y_t_shape)

    tol = 0
    if x.dtype == np.float32 or x.dtype == np.float16:
      tol = 100 * np.finfo(x.dtype).eps
    elif x.dtype == np.float64:
      # float64 eps is around 10 ** (-14) which is way too small for accumulated differences
      # Bumping it up to 10 ** (-7)
      tol = 10 ** (-7)
    with self.cached_session(use_gpu=True) as sess:
      # Note: Testing with three dimensions only now
      z_in = np.ones((x_in.shape[0], x_in.shape[1], y_in.shape[2])).astype(x.dtype)
      if static_shape:
        z0 = math_ops.batch_gemm(
            x, y, z_in, transpose_a=transpose_a, transpose_b=transpose_b, alpha=alpha, beta=beta)
        z0_val = self.evaluate(z0)
      else:
        x_ph = array_ops.placeholder(x.dtype)
        y_ph = array_ops.placeholder(y.dtype)
        z0 = math_ops.batch_gemm(
            x, y, z_in, transpose_a=transpose_a, transpose_b=transpose_b, alpha=alpha, beta=beta)
        z0_val = sess.run(z0, feed_dict={x_ph: x, y_ph: y})
      z1 = alpha * self._npBatchGemm(x, y, transpose_a, transpose_b) + beta * z_in
      self.assertAllClose(z0_val, z1, rtol=tol, atol=tol)

  def _testNonEmpty(self, dtype, transpose_a, transpose_b, use_static_shape):

    def CompareNonEmpty(self, a_shape, b_shape):
      self._compare(
          GetRandomNormalInput(a_shape, dtype),
          GetRandomNormalInput(b_shape, dtype),
          transpose_a,
          transpose_b,
          random.random(),
          random.random(),
          static_shape=use_static_shape)

    CompareNonEmpty(self, [1, 2, 3], [1, 3, 5])
    CompareNonEmpty(self, [1, 2, 3], [1, 3, 1])
    CompareNonEmpty(self, [1, 1, 3], [1, 3, 5])
    CompareNonEmpty(self, [1, 2, 3], [1, 3, 5])
    CompareNonEmpty(self, [7, 1, 3], [7, 3, 5])
    CompareNonEmpty(self, [7, 2, 3], [7, 3, 1])
    CompareNonEmpty(self, [7, 2, 3], [7, 3, 5])
    CompareNonEmpty(self, [10, 64, 75], [10, 75, 30])

  def _testEmpty(self, dtype, transpose_a, transpose_b, use_static_shape):

    def CompareEmpty(self, a_shape, b_shape):
      self._compare(
          np.zeros(a_shape).astype(dtype),
          np.zeros(b_shape).astype(dtype),
          transpose_a,
          transpose_b,
          random.random(),
          random.random(),
          static_shape=use_static_shape)

    CompareEmpty(self, [0, 3, 2], [0, 2, 4])
    CompareEmpty(self, [3, 0, 2], [3, 2, 5])
    CompareEmpty(self, [3, 3, 2], [3, 2, 0])


def _GetBatchGemmOpTest(dtype, transpose_a, transpose_b, use_static_shape):

  def Test(self):
    np.random.seed(42)
    self._testNonEmpty(dtype, transpose_a, transpose_b, use_static_shape)
    self._testEmpty(dtype, transpose_a, transpose_b, use_static_shape)

  return Test

class BatchGemmGradientTest(test.TestCase):

  # loss = sum(batch_matmul(x, y)). Verify dl/dx and dl/dy via the
  # gradient checker.
  def _checkGrad(self, x_in, y_in, transpose_a, transpose_b, alpha, beta):
    x_t_shape = x_in.shape[:-2] + (x_in.shape[-1], x_in.shape[-2])
    y_t_shape = y_in.shape[:-2] + (y_in.shape[-1], y_in.shape[-2])
    x = x_in if not transpose_a else x_in.reshape(x_t_shape)
    y = y_in if not transpose_b else y_in.reshape(y_t_shape)
    epsilon = np.finfo(x.dtype).eps
    # Since our gradient is linear, a larger delta decreases the error.
    delta = 10 * epsilon**(1.0 / 3.0)

    z = np.ones((x_in.shape[0], x_in.shape[1], y_in.shape[2])).astype(x.dtype)

    # Note: z is counted as a constant for batch_gemm operator therefore should
    # not be fed into the loss function as an independent variable
    def Loss(x, y):
      return math_ops.reduce_sum(math_ops.batch_gemm(x, y, z, transpose_a, transpose_b, alpha, beta))

    with self.cached_session(use_gpu=True):
      ((x_jacob_t, y_jacob_t),
       (x_jacob_n, y_jacob_n)) = gradient_checker_v2.compute_gradient(
           Loss, [x, y], delta=delta)
      tol = 10 * delta
      self.assertAllClose(x_jacob_t, x_jacob_n, rtol=tol, atol=tol)
      self.assertAllClose(y_jacob_t, y_jacob_n, rtol=tol, atol=tol)

  def _compare(self, a_shape, b_shape, dtype, transpose_a, transpose_b):
    np.random.seed(42)
    x = GetRandomNormalInput(a_shape, dtype)
    y = GetRandomNormalInput(b_shape, dtype)
    self._checkGrad(x, y, transpose_a, transpose_b, random.random(), 0.0)


def _GetBatchGemmGradientTest(dtype, transpose_a, transpose_b):

  def Test(self):
    def CheckGradients(self, a_shape, b_shape):
      self._compare(a_shape, b_shape, dtype, transpose_a, transpose_b)

    CheckGradients(self, [1, 2, 3], [1, 3, 5])
    CheckGradients(self, [3, 4, 7], [3, 7, 10])

  return Test

if __name__ == "__main__":
  dtypes_to_test = [np.float16, np.float32, np.float64]
  for dtype_ in dtypes_to_test:
    for transpose_a_ in False, True:
      for transpose_b_ in False, True:
        name = "%s_%s_%s" % (dtype_.__name__, transpose_a_, transpose_b_)
        # TF2 does not support placeholders under eager so we skip it.
        for use_static_shape_ in set([True, tf2.enabled()]):
          setattr(
              BatchGemmOpTest,
              "testBatchGemmOp_" + name + "_{}".format(use_static_shape_),
              _GetBatchGemmOpTest(dtype_, transpose_a_, transpose_b_,
                                    use_static_shape_))
        setattr(BatchGemmGradientTest, "testBatchGemmGradient_" + name,
                _GetBatchGemmGradientTest(dtype_, transpose_a_, transpose_b_))

  test.main()
