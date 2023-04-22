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
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
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
  if dtype in (np.complex64, np.complex128):
    imag = np.array(np.random.normal(loc, scale, np.prod(shape)), dtype=dtype)
    vals += 1j * imag
  return vals.reshape(shape)


class BatchMatmulOpTest(test.TestCase):

  # Uses numpy to compute batch_matmul(x, y, adjoint_a, adjoint_b).
  def _npBatchMatmul(self, x, y, adjoint_a, adjoint_b):
    # output's shape depends on adj[0] and adj[1]
    if adjoint_a:
      x = np.conjugate(np.swapaxes(x, -1, -2))
    if adjoint_b:
      y = np.conjugate(np.swapaxes(y, -1, -2))
    return np.matmul(x, y)

  # Compares TensorFlow BatchMatmul with NumPy's matmul.
  def _compare(self, x_in, y_in, adjoint_a, adjoint_b, static_shape):
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

  def _testNonEmpty(self, dtype, adjoint_a, adjoint_b, use_static_shape):

    def CompareNonEmpty(self, a_shape, b_shape):
      self._compare(
          GetRandomNormalInput(a_shape, dtype),
          GetRandomNormalInput(b_shape, dtype),
          adjoint_a,
          adjoint_b,
          static_shape=use_static_shape)

    CompareNonEmpty(self, [1, 2, 3], [1, 3, 5])
    CompareNonEmpty(self, [1, 2, 3], [1, 3, 1])
    CompareNonEmpty(self, [1, 1, 3], [1, 3, 5])
    CompareNonEmpty(self, [1, 2, 3], [1, 3, 5])
    CompareNonEmpty(self, [7, 1, 3], [7, 3, 5])
    CompareNonEmpty(self, [7, 2, 3], [7, 3, 1])
    CompareNonEmpty(self, [7, 2, 3], [7, 3, 5])
    CompareNonEmpty(self, [10, 64, 75], [10, 75, 30])
    CompareNonEmpty(self, [5, 7, 2, 3], [5, 7, 3, 5])

  def _testBroadcasting(self, dtype, adjoint_a, adjoint_b, use_static_shape):

    def CompareNonEmpty(self, a_shape, b_shape):
      self._compare(
          GetRandomNormalInput(a_shape, dtype),
          GetRandomNormalInput(b_shape, dtype),
          adjoint_a,
          adjoint_b,
          static_shape=use_static_shape)

    CompareNonEmpty(self, [2, 3], [1, 3, 5])
    CompareNonEmpty(self, [1, 2, 3], [3, 5])
    CompareNonEmpty(self, [5, 1, 2, 3], [1, 7, 3, 5])
    CompareNonEmpty(self, [5, 2, 2, 3], [3, 5])
    CompareNonEmpty(self, [2, 3], [5, 2, 3, 5])
    CompareNonEmpty(self, [4, 5, 1, 2, 3], [1, 1, 3, 5])
    CompareNonEmpty(self, [1, 2, 1, 4, 2, 1, 3, 4], [3, 2, 1, 1, 1, 2, 4, 2])

  def _testEmpty(self, dtype, adjoint_a, adjoint_b, use_static_shape):

    def CompareEmpty(self, a_shape, b_shape):
      self._compare(
          np.zeros(a_shape).astype(dtype),
          np.zeros(b_shape).astype(dtype),
          adjoint_a,
          adjoint_b,
          static_shape=use_static_shape)

    CompareEmpty(self, [0, 3, 2], [0, 2, 4])
    CompareEmpty(self, [3, 0, 2], [3, 2, 5])
    CompareEmpty(self, [3, 3, 2], [3, 2, 0])


def _GetBatchMatmulOpTest(dtype, adjoint_a, adjoint_b, use_static_shape):

  @test_util.run_without_tensor_float_32("Tests batch matmul")
  def Test(self):
    np.random.seed(42)
    self._testNonEmpty(dtype, adjoint_a, adjoint_b, use_static_shape)
    self._testEmpty(dtype, adjoint_a, adjoint_b, use_static_shape)

  return Test


def _GetBatchMatmulOpBroadcastingTest(dtype, adjoint_a, adjoint_b,
                                      use_static_shape):

  @test_util.run_without_tensor_float_32("Tests batch matmul")
  def Test(self):
    np.random.seed(42)
    self._testBroadcasting(dtype, adjoint_a, adjoint_b, use_static_shape)

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
    # Since our gradient is linear, a larger delta decreases the error.
    delta = 10 * epsilon**(1.0 / 3.0)

    def Loss(x, y):
      return math_ops.reduce_sum(math_ops.matmul(x, y, adjoint_a, adjoint_b))

    with self.cached_session():
      ((x_jacob_t, y_jacob_t),
       (x_jacob_n, y_jacob_n)) = gradient_checker_v2.compute_gradient(
           Loss, [x, y], delta=delta)
      tol = 10 * delta
      self.assertAllClose(x_jacob_t, x_jacob_n, rtol=tol, atol=tol)
      self.assertAllClose(y_jacob_t, y_jacob_n, rtol=tol, atol=tol)

  # Tests gradients of a batched matmul of x, and y
  def _compare(self, a_shape, b_shape, dtype, adjoint_a, adjoint_b):
    np.random.seed(42)
    x = GetRandomNormalInput(a_shape, dtype)
    y = GetRandomNormalInput(b_shape, dtype)
    self._checkGrad(x, y, adjoint_a, adjoint_b)


def _GetBatchMatmulGradientTest(dtype, adjoint_a, adjoint_b):

  def Test(self):
    def CheckGradients(self, a_shape, b_shape):
      self._compare(a_shape, b_shape, dtype, adjoint_a, adjoint_b)

    CheckGradients(self, [1, 2, 3], [1, 3, 5])
    CheckGradients(self, [3, 4, 7], [3, 7, 10])

  return Test


def _GetBatchMatmulGradientWithBroadcastingTest(dtype, adjoint_a, adjoint_b):

  def Test(self):
    def CheckGradients(self, a_shape, b_shape):
      self._compare(a_shape, b_shape, dtype, adjoint_a, adjoint_b)

    CheckGradients(self, [1, 5, 2, 3], [7, 1, 3, 2])
    CheckGradients(self, [2, 3], [1, 3, 5])
    CheckGradients(self, [2, 3], [5, 3, 5])
    CheckGradients(self, [5, 2, 5], [5, 3])
    CheckGradients(self, [5, 2, 2, 3], [3, 5])
    CheckGradients(self, [4, 5, 1, 2, 3], [1, 1, 3, 5])
    CheckGradients(self, [1, 2, 1, 4, 2, 1, 3, 4], [3, 2, 1, 1, 1, 2, 4, 2])

  return Test


class BatchMatMulBenchmark(test.Benchmark):
  # Batch sizes are 512.
  shape_pairs = [
      # Typical fully connected layer.
      ((4, 8, 4, 2, 1, 1024), (1024, 1024)),
      ((4, 1, 4, 1, 1, 1024), (1, 8, 1, 2, 1024, 1024)),
      # Square matmul.
      ((4, 8, 4, 2, 512, 512), (512, 512)),
      ((4, 1, 4, 1, 512, 512), (1, 8, 1, 2, 512, 512)),
      # Matrix-vector multiplies.
      ((4, 8, 4, 2, 10000, 200), (200, 1)),
      ((4, 1, 4, 1, 10000, 200), (1, 8, 1, 2, 200, 1)),
      # Vector-matrix multiplies.
      ((4, 8, 4, 2, 1, 200), (200, 10000)),
      ((4, 1, 4, 1, 1, 200), (1, 8, 1, 2, 200, 10000)),
  ]

  def benchmarkBatchMatMulBroadcast(self):
    for (a_shape, b_shape) in self.shape_pairs:
      with ops.Graph().as_default(), \
          session.Session(config=benchmark.benchmark_config()) as sess, \
          ops.device("/cpu:0"):
        matrix_a = variables.Variable(
            GetRandomNormalInput(a_shape, np.float32))
        matrix_b = variables.Variable(
            GetRandomNormalInput(b_shape, np.float32))
        self.evaluate(variables.global_variables_initializer())

        # Use batch matmul op's internal broadcasting.
        self.run_op_benchmark(
            sess,
            math_ops.matmul(matrix_a, matrix_b),
            min_iters=50,
            name="batch_matmul_cpu_{}_{}".format(a_shape, b_shape))

        # Manually broadcast the input matrices using the broadcast_to op.
        broadcasted_batch_shape = array_ops.broadcast_static_shape(
            matrix_a.shape[:-2], matrix_b.shape[:-2])
        broadcasted_a_shape = broadcasted_batch_shape.concatenate(
            matrix_a.shape[-2:])
        broadcasted_b_shape = broadcasted_batch_shape.concatenate(
            matrix_b.shape[-2:])
        self.run_op_benchmark(
            sess,
            math_ops.matmul(
                array_ops.broadcast_to(matrix_a, broadcasted_a_shape),
                array_ops.broadcast_to(matrix_b, broadcasted_b_shape)),
            min_iters=50,
            name="batch_matmul_manual_broadcast_cpu_{}_{}".format(
                a_shape, b_shape))


if __name__ == "__main__":
  dtypes_to_test = [
      np.float16, np.float32, np.float64, np.int32, np.complex64, np.complex128
  ]
  for dtype_ in dtypes_to_test:
    for adjoint_a_ in False, True:
      for adjoint_b_ in False, True:
        name = "%s_%s_%s" % (dtype_.__name__, adjoint_a_, adjoint_b_)
        # TF2 does not support placeholders under eager so we skip it.
        for use_static_shape_ in set([True, tf2.enabled()]):
          setattr(
              BatchMatmulOpTest,
              "testBatchMatmulOp_" + name + "_{}".format(use_static_shape_),
              test_util.xla_allow_fallback(
                  "TODO(b/134526360): XLA:CPU hasn't implemented int32 dot.")(
                      _GetBatchMatmulOpTest(dtype_, adjoint_a_, adjoint_b_,
                                            use_static_shape_)))
          # Broadcasting is supported only in v2.
          setattr(
              BatchMatmulOpTest, "testBatchMatmulBroadcasting_" + name +
              ("_%s" % use_static_shape_),
              test_util.xla_allow_fallback(
                  "TODO(b/134526360): XLA:CPU hasn't implemented int32 dot.")(
                      _GetBatchMatmulOpBroadcastingTest(dtype_, adjoint_a_,
                                                        adjoint_b_,
                                                        use_static_shape_)))
        if dtype_ == np.int32:
          continue
        setattr(BatchMatmulGradientTest, "testBatchMatmulGradient_" + name,
                _GetBatchMatmulGradientTest(dtype_, adjoint_a_, adjoint_b_))
        # Broadcasting is supported only in v2.
        setattr(
            BatchMatmulGradientTest,
            "testBatchMatmulGradientWithBroadcasting_" + name,
            _GetBatchMatmulGradientWithBroadcastingTest(dtype_, adjoint_a_,
                                                        adjoint_b_))
  test.main()
