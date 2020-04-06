# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.gen_linalg_ops.matrix_logarithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linalg_impl
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


class LogarithmOpTest(test.TestCase):

  def _verifyLogarithm(self, x, np_type):
    inp = x.astype(np_type)
    with test_util.use_gpu():
      # Verify that expm(logm(A)) == A.
      tf_ans = linalg_impl.matrix_exponential(
          gen_linalg_ops.matrix_logarithm(inp))
      out = self.evaluate(tf_ans)
      self.assertAllClose(inp, out, rtol=1e-4, atol=1e-3)

  def _verifyLogarithmComplex(self, x):
    for np_type in [np.complex64, np.complex128]:
      self._verifyLogarithm(x, np_type)

  def _makeBatch(self, matrix1, matrix2):
    matrix_batch = np.concatenate(
        [np.expand_dims(matrix1, 0),
         np.expand_dims(matrix2, 0)])
    matrix_batch = np.tile(matrix_batch, [2, 3, 1, 1])
    return matrix_batch

  @test_util.run_v1_only("b/120545219")
  def testNonsymmetric(self):
    # 2x2 matrices
    matrix1 = np.array([[1., 2.], [3., 4.]])
    matrix2 = np.array([[1., 3.], [3., 5.]])
    matrix1 = matrix1.astype(np.complex64)
    matrix1 += 1j * matrix1
    matrix2 = matrix2.astype(np.complex64)
    matrix2 += 1j * matrix2
    self._verifyLogarithmComplex(matrix1)
    self._verifyLogarithmComplex(matrix2)
    # Complex batch
    self._verifyLogarithmComplex(self._makeBatch(matrix1, matrix2))

  @test_util.run_v1_only("b/120545219")
  def testSymmetricPositiveDefinite(self):
    # 2x2 matrices
    matrix1 = np.array([[2., 1.], [1., 2.]])
    matrix2 = np.array([[3., -1.], [-1., 3.]])
    matrix1 = matrix1.astype(np.complex64)
    matrix1 += 1j * matrix1
    matrix2 = matrix2.astype(np.complex64)
    matrix2 += 1j * matrix2
    self._verifyLogarithmComplex(matrix1)
    self._verifyLogarithmComplex(matrix2)
    # Complex batch
    self._verifyLogarithmComplex(self._makeBatch(matrix1, matrix2))

  @test_util.run_v1_only("b/120545219")
  def testNonSquareMatrix(self):
    # When the logarithm of a non-square matrix is attempted we should return
    # an error
    with self.assertRaises(ValueError):
      gen_linalg_ops.matrix_logarithm(
          np.array([[1., 2., 3.], [3., 4., 5.]], dtype=np.complex64))

  @test_util.run_v1_only("b/120545219")
  def testWrongDimensions(self):
    # The input to the logarithm should be at least a 2-dimensional tensor.
    tensor3 = constant_op.constant([1., 2.], dtype=dtypes.complex64)
    with self.assertRaises(ValueError):
      gen_linalg_ops.matrix_logarithm(tensor3)

  @test_util.run_v1_only("b/120545219")
  def testEmpty(self):
    self._verifyLogarithmComplex(np.empty([0, 2, 2], dtype=np.complex64))
    self._verifyLogarithmComplex(np.empty([2, 0, 0], dtype=np.complex64))

  @test_util.run_v1_only("b/120545219")
  def testRandomSmallAndLargeComplex64(self):
    np.random.seed(42)
    for batch_dims in [(), (1,), (3,), (2, 2)]:
      for size in 8, 31, 32:
        shape = batch_dims + (size, size)
        matrix = np.random.uniform(
            low=-1.0, high=1.0,
            size=np.prod(shape)).reshape(shape).astype(np.complex64)
        self._verifyLogarithmComplex(matrix)

  @test_util.run_v1_only("b/120545219")
  def testRandomSmallAndLargeComplex128(self):
    np.random.seed(42)
    for batch_dims in [(), (1,), (3,), (2, 2)]:
      for size in 8, 31, 32:
        shape = batch_dims + (size, size)
        matrix = np.random.uniform(
            low=-1.0, high=1.0,
            size=np.prod(shape)).reshape(shape).astype(np.complex128)
        self._verifyLogarithmComplex(matrix)

  @test_util.run_v1_only("b/120545219")
  def testConcurrentExecutesWithoutError(self):
    with self.session(use_gpu=True) as sess:
      matrix1 = math_ops.cast(
          random_ops.random_normal([5, 5], seed=42), dtypes.complex64)
      matrix2 = math_ops.cast(
          random_ops.random_normal([5, 5], seed=42), dtypes.complex64)
      logm1 = gen_linalg_ops.matrix_logarithm(matrix1)
      logm2 = gen_linalg_ops.matrix_logarithm(matrix2)
      logm = self.evaluate([logm1, logm2])
      self.assertAllEqual(logm[0], logm[1])


class MatrixLogarithmBenchmark(test.Benchmark):

  shapes = [
      (4, 4),
      (10, 10),
      (16, 16),
      (101, 101),
      (256, 256),
      (1000, 1000),
      (1024, 1024),
      (2048, 2048),
      (513, 4, 4),
      (513, 16, 16),
      (513, 256, 256),
  ]

  def _GenerateMatrix(self, shape):
    batch_shape = shape[:-2]
    shape = shape[-2:]
    assert shape[0] == shape[1]
    n = shape[0]
    matrix = np.ones(shape).astype(np.complex64) / (2.0 * n) + np.diag(
        np.ones(n).astype(np.complex64))
    return variables.Variable(np.tile(matrix, batch_shape + (1, 1)))

  def benchmarkMatrixLogarithmOp(self):
    for shape in self.shapes:
      with ops.Graph().as_default(), \
          session.Session(config=benchmark.benchmark_config()) as sess, \
          ops.device("/cpu:0"):
        matrix = self._GenerateMatrix(shape)
        logm = gen_linalg_ops.matrix_logarithm(matrix)
        variables.global_variables_initializer().run()
        self.run_op_benchmark(
            sess,
            control_flow_ops.group(logm),
            min_iters=25,
            name="matrix_logarithm_cpu_{shape}".format(shape=shape))


if __name__ == "__main__":
  test.main()
