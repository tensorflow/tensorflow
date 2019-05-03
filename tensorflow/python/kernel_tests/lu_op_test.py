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
"""Tests for tensorflow.ops.tf.Lu."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


class LuOpTest(test.TestCase):

  @property
  def float_types(self):
    return set((np.float64, np.float32, np.complex64, np.complex128))

  def _verifyLuBase(self, x, lower, upper, perm, verification,
                    output_idx_type):
    lower_np, upper_np, perm_np, verification_np = self.evaluate(
        [lower, upper, perm, verification])

    self.assertAllClose(x, verification_np)
    self.assertShapeEqual(x, lower)
    self.assertShapeEqual(x, upper)

    self.assertAllEqual(x.shape[:-1], perm.shape.as_list())

    # Check dtypes are as expected.
    self.assertEqual(x.dtype, lower_np.dtype)
    self.assertEqual(x.dtype, upper_np.dtype)
    self.assertEqual(output_idx_type.as_numpy_dtype, perm_np.dtype)

    # Check that the permutation is valid.
    if perm_np.shape[-1] > 0:
      perm_reshaped = np.reshape(perm_np, (-1, perm_np.shape[-1]))
      for perm_vector in perm_reshaped:
        self.assertAllClose(np.arange(len(perm_vector)), np.sort(perm_vector))

  def _verifyLu(self, x, output_idx_type=dtypes.int64):
    # Verify that Px = LU.
    lu, perm = linalg_ops.lu(x, output_idx_type=output_idx_type)

    # Prepare the lower factor of shape num_rows x num_rows
    lu_shape = np.array(lu.shape.as_list())
    batch_shape = lu_shape[:-2]
    num_rows = lu_shape[-2]
    num_cols = lu_shape[-1]

    lower = array_ops.matrix_band_part(lu, -1, 0)

    if num_rows > num_cols:
      eye = linalg_ops.eye(
          num_rows, batch_shape=batch_shape, dtype=lower.dtype)
      lower = array_ops.concat([lower, eye[..., num_cols:]], axis=-1)
    elif num_rows < num_cols:
      lower = lower[..., :num_rows]

    # Fill the diagonal with ones.
    ones_diag = array_ops.ones(
        np.append(batch_shape, num_rows), dtype=lower.dtype)
    lower = array_ops.matrix_set_diag(lower, ones_diag)

    # Prepare the upper factor.
    upper = array_ops.matrix_band_part(lu, 0, -1)

    verification = math_ops.matmul(lower, upper)

    # Permute the rows of product of the Cholesky factors.
    if num_rows > 0:
      # Reshape the product of the triangular factors and permutation indices
      # to a single batch dimension. This makes it easy to apply
      # invert_permutation and gather_nd ops.
      perm_reshaped = array_ops.reshape(perm, [-1, num_rows])
      verification_reshaped = array_ops.reshape(verification,
                                                [-1, num_rows, num_cols])
      # Invert the permutation in each batch.
      inv_perm_reshaped = map_fn.map_fn(array_ops.invert_permutation,
                                        perm_reshaped)
      batch_size = perm_reshaped.shape.as_list()[0]
      # Prepare the batch indices with the same shape as the permutation.
      # The corresponding batch index is paired with each of the `num_rows`
      # permutation indices.
      batch_indices = math_ops.cast(
          array_ops.broadcast_to(
              math_ops.range(batch_size)[:, None], perm_reshaped.shape),
          dtype=output_idx_type)
      permuted_verification_reshaped = array_ops.gather_nd(
          verification_reshaped,
          array_ops.stack([batch_indices, inv_perm_reshaped], axis=-1))

      # Reshape the verification matrix back to the original shape.
      verification = array_ops.reshape(permuted_verification_reshaped,
                                       lu_shape)

    self._verifyLuBase(x, lower, upper, perm, verification,
                       output_idx_type)

  def testBasic(self):
    data = np.array([[4., -1., 2.], [-1., 6., 0], [10., 0., 5.]])

    for dtype in (np.float32, np.float64):
      for output_idx_type in (dtypes.int32, dtypes.int64):
        self._verifyLu(data.astype(dtype), output_idx_type=output_idx_type)

    for dtype in (np.complex64, np.complex128):
      for output_idx_type in (dtypes.int32, dtypes.int64):
        complex_data = np.tril(1j * data, -1).astype(dtype)
        complex_data += np.triu(-1j * data, 1).astype(dtype)
        complex_data += data
        self._verifyLu(complex_data, output_idx_type=output_idx_type)

  def testPivoting(self):
    # This matrix triggers partial pivoting because the first diagonal entry
    # is small.
    data = np.array([[1e-9, 1., 0.], [1., 0., 0], [0., 1., 5]])
    self._verifyLu(data.astype(np.float32))

    for dtype in (np.float32, np.float64):
      self._verifyLu(data.astype(dtype))
      _, p = linalg_ops.lu(data)
      p_val = self.evaluate([p])
      # Make sure p_val is not the identity permutation.
      self.assertNotAllClose(np.arange(3), p_val)

    for dtype in (np.complex64, np.complex128):
      complex_data = np.tril(1j * data, -1).astype(dtype)
      complex_data += np.triu(-1j * data, 1).astype(dtype)
      complex_data += data
      self._verifyLu(complex_data)
      _, p = linalg_ops.lu(data)
      p_val = self.evaluate([p])
      # Make sure p_val is not the identity permutation.
      self.assertNotAllClose(np.arange(3), p_val)

  def testInvalidMatrix(self):
    # LU factorization gives an error when the input is singular.
    # Note: A singular matrix may return without error but it won't be a valid
    # factorization.
    for dtype in self.float_types:
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(
            linalg_ops.lu(
                np.array([[1., 2., 3.], [2., 4., 6.], [2., 3., 4.]],
                         dtype=dtype)))
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(
            linalg_ops.lu(
                np.array([[[1., 2., 3.], [2., 4., 6.], [1., 2., 3.]],
                          [[1., 2., 3.], [3., 4., 5.], [5., 6., 7.]]],
                         dtype=dtype)))

  def testBatch(self):
    simple_array = np.array([[[1., -1.], [2., 5.]]])  # shape (1, 2, 2)
    self._verifyLu(simple_array)
    self._verifyLu(np.vstack((simple_array, simple_array)))
    odd_sized_array = np.array([[[4., -1., 2.], [-1., 6., 0], [2., 0., 5.]]])
    self._verifyLu(np.vstack((odd_sized_array, odd_sized_array)))

    batch_size = 200

    # Generate random matrices.
    np.random.seed(42)
    matrices = np.random.rand(batch_size, 5, 5)
    self._verifyLu(matrices)

    # Generate random complex valued matrices.
    np.random.seed(52)
    matrices = np.random.rand(batch_size, 5,
                              5) + 1j * np.random.rand(batch_size, 5, 5)
    self._verifyLu(matrices)

  def testLargeMatrix(self):
    # Generate random matrices.
    n = 500
    np.random.seed(64)
    data = np.random.rand(n, n)
    self._verifyLu(data)

    # Generate random complex valued matrices.
    np.random.seed(129)
    data = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    self._verifyLu(data)

  @test_util.run_v1_only("b/120545219")
  def testEmpty(self):
    self._verifyLu(np.empty([0, 2, 2]))
    self._verifyLu(np.empty([2, 0, 0]))

  @test_util.run_deprecated_v1
  def testConcurrentExecutesWithoutError(self):
    matrix1 = random_ops.random_normal([5, 5], seed=42)
    matrix2 = random_ops.random_normal([5, 5], seed=42)
    lu1, p1 = linalg_ops.lu(matrix1)
    lu2, p2 = linalg_ops.lu(matrix2)
    lu1_val, p1_val, lu2_val, p2_val = self.evaluate([lu1, p1, lu2, p2])
    self.assertAllEqual(lu1_val, lu2_val)
    self.assertAllEqual(p1_val, p2_val)


class LuBenchmark(test.Benchmark):
  shapes = [
      (4, 4),
      (10, 10),
      (16, 16),
      (101, 101),
      (256, 256),
      (1000, 1000),
      (1024, 1024),
      (2048, 2048),
      (4096, 4096),
      (513, 2, 2),
      (513, 8, 8),
      (513, 256, 256),
      (4, 513, 2, 2),
  ]

  def _GenerateMatrix(self, shape):
    batch_shape = shape[:-2]
    shape = shape[-2:]
    assert shape[0] == shape[1]
    n = shape[0]
    matrix = np.ones(shape).astype(np.float32) / (2.0 * n) + np.diag(
        np.ones(n).astype(np.float32))
    return np.tile(matrix, batch_shape + (1, 1))

  def benchmarkLuOp(self):
    for shape in self.shapes:
      with ops.Graph().as_default(), \
          session.Session(config=benchmark.benchmark_config()) as sess, \
          ops.device("/cpu:0"):
        matrix = variables.Variable(self._GenerateMatrix(shape))
        lu, p = linalg_ops.lu(matrix)
        variables.global_variables_initializer().run()
        self.run_op_benchmark(
            sess,
            control_flow_ops.group(lu, p),
            min_iters=25,
            name="lu_cpu_{shape}".format(shape=shape))

      if test.is_gpu_available(True):
        with ops.Graph().as_default(), \
            session.Session(config=benchmark.benchmark_config()) as sess, \
            ops.device("/device:GPU:0"):
          matrix = variables.Variable(self._GenerateMatrix(shape))
          lu, p = linalg_ops.lu(matrix)
          variables.global_variables_initializer().run()
          self.run_op_benchmark(
              sess,
              control_flow_ops.group(lu, p),
              min_iters=25,
              name="lu_gpu_{shape}".format(shape=shape))


if __name__ == "__main__":
  test.main()
