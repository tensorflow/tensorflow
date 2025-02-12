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
"""Tests for tensorflow.ops.math_ops.matrix_square_root."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.platform import test


class SquareRootOpTest(test.TestCase):

  def _verifySquareRoot(self, matrix, np_type):
    matrix = matrix.astype(np_type)

    # Verify that matmul(sqrtm(A), sqrtm(A)) = A
    sqrt = gen_linalg_ops.matrix_square_root(matrix)
    square = test_util.matmul_without_tf32(sqrt, sqrt)
    self.assertShapeEqual(matrix, square)
    self.assertAllClose(matrix, square, rtol=1e-4, atol=1e-3)

  def _verifySquareRootReal(self, x):
    for np_type in [np.float32, np.float64]:
      self._verifySquareRoot(x, np_type)

  def _verifySquareRootComplex(self, x):
    for np_type in [np.complex64, np.complex128]:
      self._verifySquareRoot(x, np_type)

  def _makeBatch(self, matrix1, matrix2):
    matrix_batch = np.concatenate(
        [np.expand_dims(matrix1, 0),
         np.expand_dims(matrix2, 0)])
    matrix_batch = np.tile(matrix_batch, [2, 3, 1, 1])
    return matrix_batch

  def _testMatrices(self, matrix1, matrix2):
    # Real
    self._verifySquareRootReal(matrix1)
    self._verifySquareRootReal(matrix2)
    self._verifySquareRootReal(self._makeBatch(matrix1, matrix2))
    matrix1 = matrix1.astype(np.complex64)
    matrix2 = matrix2.astype(np.complex64)
    matrix1 += 1j * matrix1
    matrix2 += 1j * matrix2
    self._verifySquareRootComplex(matrix1)
    self._verifySquareRootComplex(matrix2)
    self._verifySquareRootComplex(self._makeBatch(matrix1, matrix2))

  @test_util.run_without_tensor_float_32
  def testSymmetricPositiveDefinite(self):
    matrix1 = np.array([[2., 1.], [1., 2.]])
    matrix2 = np.array([[3., -1.], [-1., 3.]])
    self._testMatrices(matrix1, matrix2)

  @test_util.run_without_tensor_float_32
  def testAsymmetric(self):
    matrix1 = np.array([[0., 4.], [-1., 5.]])
    matrix2 = np.array([[33., 24.], [48., 57.]])
    self._testMatrices(matrix1, matrix2)

  @test_util.run_without_tensor_float_32
  def testIdentityMatrix(self):
    # 2x2
    identity = np.array([[1., 0], [0, 1.]])
    self._verifySquareRootReal(identity)
    # 3x3
    identity = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.]])
    self._verifySquareRootReal(identity)

  @test_util.run_without_tensor_float_32
  def testEmpty(self):
    self._verifySquareRootReal(np.empty([0, 2, 2]))
    self._verifySquareRootReal(np.empty([2, 0, 0]))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  @test_util.run_without_tensor_float_32
  def testWrongDimensions(self):
    # The input to the square root should be at least a 2-dimensional tensor.
    tensor = constant_op.constant([1., 2.])
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      gen_linalg_ops.matrix_square_root(tensor)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  @test_util.run_without_tensor_float_32
  def testNotSquare(self):
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      tensor = constant_op.constant([[1., 0., -1.], [-1., 1., 0.]])
      self.evaluate(gen_linalg_ops.matrix_square_root(tensor))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  @test_util.run_without_tensor_float_32
  def testConcurrentExecutesWithoutError(self):
    matrix_shape = [5, 5]
    seed = [42, 24]
    matrix1 = stateless_random_ops.stateless_random_normal(
        shape=matrix_shape, seed=seed)
    matrix2 = stateless_random_ops.stateless_random_normal(
        shape=matrix_shape, seed=seed)
    self.assertAllEqual(matrix1, matrix2)
    square1 = math_ops.matmul(matrix1, matrix1)
    square2 = math_ops.matmul(matrix2, matrix2)
    sqrt1 = gen_linalg_ops.matrix_square_root(square1)
    sqrt2 = gen_linalg_ops.matrix_square_root(square2)
    all_ops = [sqrt1, sqrt2]
    sqrt = self.evaluate(all_ops)
    self.assertAllClose(sqrt[0], sqrt[1])


if __name__ == "__main__":
  test.main()
