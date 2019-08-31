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
"""Tests for tensorflow.ops.math_ops.matrix_inverse."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


class InverseOpTest(test.TestCase):

  def _verifyInverse(self, x, np_type):
    for adjoint in False, True:
      y = x.astype(np_type)
      with self.cached_session(use_gpu=True):
        # Verify that x^{-1} * x == Identity matrix.
        inv = linalg_ops.matrix_inverse(y, adjoint=adjoint)
        tf_ans = math_ops.matmul(inv, y, adjoint_b=adjoint)
        np_ans = np.identity(y.shape[-1])
        if x.ndim > 2:
          tiling = list(y.shape)
          tiling[-2:] = [1, 1]
          np_ans = np.tile(np_ans, tiling)
        out = self.evaluate(tf_ans)
        self.assertAllClose(np_ans, out, rtol=1e-4, atol=1e-3)
        self.assertShapeEqual(y, tf_ans)

  def _verifyInverseReal(self, x):
    for np_type in [np.float32, np.float64]:
      self._verifyInverse(x, np_type)

  def _verifyInverseComplex(self, x):
    for np_type in [np.complex64, np.complex128]:
      self._verifyInverse(x, np_type)

  def _makeBatch(self, matrix1, matrix2):
    matrix_batch = np.concatenate(
        [np.expand_dims(matrix1, 0),
         np.expand_dims(matrix2, 0)])
    matrix_batch = np.tile(matrix_batch, [2, 3, 1, 1])
    return matrix_batch

  def testNonsymmetric(self):
    # 2x2 matrices
    matrix1 = np.array([[1., 2.], [3., 4.]])
    matrix2 = np.array([[1., 3.], [3., 5.]])
    self._verifyInverseReal(matrix1)
    self._verifyInverseReal(matrix2)
    # A multidimensional batch of 2x2 matrices
    self._verifyInverseReal(self._makeBatch(matrix1, matrix2))
    if not test.is_built_with_rocm():
      # ROCm does not support BLAS operations for complex types
      # Complex
      matrix1 = matrix1.astype(np.complex64)
      matrix1 += 1j * matrix1
      matrix2 = matrix2.astype(np.complex64)
      matrix2 += 1j * matrix2
      self._verifyInverseComplex(matrix1)
      self._verifyInverseComplex(matrix2)
      # Complex batch
      self._verifyInverseComplex(self._makeBatch(matrix1, matrix2))

  def testSymmetricPositiveDefinite(self):
    # 2x2 matrices
    matrix1 = np.array([[2., 1.], [1., 2.]])
    matrix2 = np.array([[3., -1.], [-1., 3.]])
    self._verifyInverseReal(matrix1)
    self._verifyInverseReal(matrix2)
    # A multidimensional batch of 2x2 matrices
    self._verifyInverseReal(self._makeBatch(matrix1, matrix2))
    if not test.is_built_with_rocm():
      # ROCm does not support BLAS operations for complex types
      # Complex
      matrix1 = matrix1.astype(np.complex64)
      matrix1 += 1j * matrix1
      matrix2 = matrix2.astype(np.complex64)
      matrix2 += 1j * matrix2
      self._verifyInverseComplex(matrix1)
      self._verifyInverseComplex(matrix2)
      # Complex batch
      self._verifyInverseComplex(self._makeBatch(matrix1, matrix2))

  @test_util.deprecated_graph_mode_only
  def testNonSquareMatrix(self):
    # When the inverse of a non-square matrix is attempted we should return
    # an error
    with self.assertRaises(ValueError):
      linalg_ops.matrix_inverse(np.array([[1., 2., 3.], [3., 4., 5.]]))

  @test_util.deprecated_graph_mode_only
  def testWrongDimensions(self):
    # The input to the inverse should be at least a 2-dimensional tensor.
    tensor3 = constant_op.constant([1., 2.])
    with self.assertRaises(ValueError):
      linalg_ops.matrix_inverse(tensor3)

  def testNotInvertible(self):
    # The input should be invertible.
    with self.cached_session():
      with self.assertRaisesOpError("Input is not invertible."):
        # All rows of the matrix below add to zero.
        tensor3 = constant_op.constant([[1., 0., -1.], [-1., 1., 0.],
                                        [0., -1., 1.]])
        linalg_ops.matrix_inverse(tensor3).eval()

  def testEmpty(self):
    self._verifyInverseReal(np.empty([0, 2, 2]))
    self._verifyInverseReal(np.empty([2, 0, 0]))

  def testRandomSmallAndLarge(self):
    np.random.seed(42)
    for dtype in np.float32, np.float64, np.complex64, np.complex128:
      for batch_dims in [(), (1,), (3,), (2, 2)]:
        for size in 8, 31, 32:
          shape = batch_dims + (size, size)
          matrix = np.random.uniform(
              low=-1.0, high=1.0,
              size=np.prod(shape)).reshape(shape).astype(dtype)
          self._verifyInverseReal(matrix)

  @test_util.deprecated_graph_mode_only
  def testConcurrentExecutesWithoutError(self):
    with self.session(use_gpu=True) as sess:
      all_ops = []
      for adjoint_ in True, False:
        matrix1 = random_ops.random_normal([5, 5], seed=42)
        matrix2 = random_ops.random_normal([5, 5], seed=42)
        inv1 = linalg_ops.matrix_inverse(matrix1, adjoint=adjoint_)
        inv2 = linalg_ops.matrix_inverse(matrix2, adjoint=adjoint_)
        all_ops += [inv1, inv2]
      inv = self.evaluate(all_ops)
      self.assertAllEqual(inv[0], inv[1])
      self.assertAllEqual(inv[2], inv[3])


class MatrixInverseBenchmark(test.Benchmark):

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
    matrix = np.ones(shape).astype(np.float32) / (
        2.0 * n) + np.diag(np.ones(n).astype(np.float32))
    return variables.Variable(np.tile(matrix, batch_shape + (1, 1)))

  def benchmarkMatrixInverseOp(self):
    for adjoint in False, True:
      for shape in self.shapes:
        with ops.Graph().as_default(), \
            session.Session(config=benchmark.benchmark_config()) as sess, \
            ops.device("/cpu:0"):
          matrix = self._GenerateMatrix(shape)
          inv = linalg_ops.matrix_inverse(matrix, adjoint=adjoint)
          variables.global_variables_initializer().run()
          self.run_op_benchmark(
              sess,
              control_flow_ops.group(inv),
              min_iters=25,
              name="matrix_inverse_cpu_{shape}_adjoint_{adjoint}".format(
                  shape=shape, adjoint=adjoint))

        if test.is_gpu_available(True):
          with ops.Graph().as_default(), \
              session.Session(config=benchmark.benchmark_config()) as sess, \
              ops.device("/gpu:0"):
            matrix = self._GenerateMatrix(shape)
            inv = linalg_ops.matrix_inverse(matrix, adjoint=adjoint)
            variables.global_variables_initializer().run()
            self.run_op_benchmark(
                sess,
                control_flow_ops.group(inv),
                min_iters=25,
                name="matrix_inverse_gpu_{shape}_adjoint_{adjoint}".format(
                    shape=shape, adjoint=adjoint))


if __name__ == "__main__":
  test.main()
