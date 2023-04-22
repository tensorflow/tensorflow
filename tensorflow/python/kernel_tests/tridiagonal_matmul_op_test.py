# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.linalg.linalg_impl.tridiagonal_matmul."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linalg_impl
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


class TridiagonalMulOpTest(test.TestCase):

  def _testAllFormats(self,
                      superdiag,
                      maindiag,
                      subdiag,
                      rhs,
                      expected,
                      dtype=dtypes.float64):
    superdiag_extended = np.pad(superdiag, [0, 1], 'constant')
    subdiag_extended = np.pad(subdiag, [1, 0], 'constant')
    diags_compact = np.stack([superdiag_extended, maindiag, subdiag_extended])
    diags_matrix = np.diag(superdiag, 1) + np.diag(maindiag, 0) + np.diag(
        subdiag, -1)

    diags_sequence = (constant_op.constant(superdiag_extended, dtype),
                      constant_op.constant(maindiag, dtype),
                      constant_op.constant(subdiag_extended, dtype))
    diags_compact = constant_op.constant(diags_compact, dtype)
    diags_matrix = constant_op.constant(diags_matrix, dtype)
    rhs = constant_op.constant(rhs, dtype)

    rhs_batch = array_ops.stack([rhs, 2 * rhs])
    diags_compact_batch = array_ops.stack([diags_compact, 2 * diags_compact])
    diags_matrix_batch = array_ops.stack([diags_matrix, 2 * diags_matrix])
    diags_sequence_batch = [array_ops.stack([x, 2 * x]) for x in diags_sequence]

    results = [
        linalg_impl.tridiagonal_matmul(
            diags_sequence, rhs, diagonals_format='sequence'),
        linalg_impl.tridiagonal_matmul(
            diags_compact, rhs, diagonals_format='compact'),
        linalg_impl.tridiagonal_matmul(
            diags_matrix, rhs, diagonals_format='matrix')
    ]
    results_batch = [
        linalg_impl.tridiagonal_matmul(
            diags_sequence_batch, rhs_batch, diagonals_format='sequence'),
        linalg_impl.tridiagonal_matmul(
            diags_compact_batch, rhs_batch, diagonals_format='compact'),
        linalg_impl.tridiagonal_matmul(
            diags_matrix_batch, rhs_batch, diagonals_format='matrix')
    ]

    with self.cached_session():
      results = self.evaluate(results)
      results_batch = self.evaluate(results_batch)

    expected = np.array(expected)
    expected_batch = np.stack([expected, 4 * expected])
    for result in results:
      self.assertAllClose(result, expected)
    for result in results_batch:
      self.assertAllClose(result, expected_batch)

  def _makeTridiagonalMatrix(self, superdiag, maindiag, subdiag):
    super_pad = [[0, 0], [0, 1], [1, 0]]
    sub_pad = [[0, 0], [1, 0], [0, 1]]

    super_part = array_ops.pad(array_ops.matrix_diag(superdiag), super_pad)
    main_part = array_ops.matrix_diag(maindiag)
    sub_part = array_ops.pad(array_ops.matrix_diag(subdiag), sub_pad)
    return super_part + main_part + sub_part

  def _randomComplexArray(self, shape):
    np.random.seed(43)
    return (np.random.uniform(-10, 10, shape) +
            np.random.uniform(-10, 10, shape) * 1j)

  def _gradientTest(self, diags, rhs, dtype=dtypes.float64):

    def reference_matmul(diags, rhs):
      matrix = self._makeTridiagonalMatrix(diags[..., 0, :-1], diags[..., 1, :],
                                           diags[..., 2, 1:])
      return math_ops.matmul(matrix, rhs)

    diags = constant_op.constant(diags, dtype=dtype)
    rhs = constant_op.constant(rhs, dtype=dtype)
    with self.cached_session():
      grad_reference, _ = gradient_checker_v2.compute_gradient(
          reference_matmul, [diags, rhs])
      grad_theoretical, grad_numerical = gradient_checker_v2.compute_gradient(
          linalg_impl.tridiagonal_matmul, [diags, rhs])
    self.assertAllClose(grad_theoretical, grad_numerical)
    self.assertAllClose(grad_theoretical, grad_reference)

  def test2x2(self):
    self._testAllFormats([1], [2, 3], [4], [[2, 1], [4, 3]], [[8, 5], [20, 13]])

  def test3x3(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      self._testAllFormats([1, 2], [1, 2, 1], [2, 1], [[1, 1], [2, 2], [3, 3]],
                           [[3, 3], [12, 12], [5, 5]],
                           dtype=dtype)

  def testComplex(self):
    for dtype in [dtypes.complex64, dtypes.complex128]:
      self._testAllFormats([1j, 1j], [1, -1, 0], [1j, 1j],
                           np.array([[1, 1j], [1, 1j], [1, 1j]]),
                           [[1 + 1j, -1 + 1j], [-1 + 2j, -2 - 1j], [1j, -1]],
                           dtype=dtype)

  def testBatch(self):
    b = 20
    m = 10
    n = 15
    superdiag = self._randomComplexArray((b, m - 1))
    maindiag = self._randomComplexArray((b, m))
    subdiag = self._randomComplexArray((b, m - 1))
    rhs = self._randomComplexArray((b, m, n))
    matrix = np.stack([np.diag(superdiag[i], 1) + \
                       np.diag(maindiag[i], 0) + \
                       np.diag(subdiag[i], -1) for i in range(b)])
    expected_result = np.matmul(matrix, rhs)
    result = linalg_impl.tridiagonal_matmul(
        constant_op.constant(matrix, dtype=dtypes.complex128),
        constant_op.constant(rhs, dtype=dtypes.complex128),
        diagonals_format='matrix')

    with self.cached_session():
      result = self.evaluate(result)

    self.assertAllClose(result, expected_result)

  def testGradientSmall(self):
    self._gradientTest([[[1, 2, 0], [1, 2, 3], [0, 1, 2]]],
                       [[[1, 2], [3, 4], [5, 6]]],
                       dtype=dtypes.float64)

  def testGradientComplexSmall(self):
    self._gradientTest(
        np.array([[[1 + 1j, 2j, 0], [1 + 2j, 2j, 3 + 0j], [0, 1j, 2 + 0j]]]),
        np.array([[[1j, 2 + 0j], [3 + 1j, 4j], [5j, 6 + 3j]]]),
        dtype=dtypes.complex128)

  def testGradientComplexWithBatches(self):
    b = 5
    m = 10
    n = 15
    diags = self._randomComplexArray((b, 3, m))
    rhs = self._randomComplexArray((b, m, n))
    self._gradientTest(diags, rhs, dtype=dtypes.complex128)

  # Benchmark
  class TridiagonalMatMulBenchmark(test.Benchmark):
    sizes = [(100000, 1, 1), (1000000, 1, 1), (10000000, 1, 1), (100000, 10, 1),
             (100000, 100, 1), (10000, 1, 100), (10000, 1, 1000),
             (10000, 1, 10000)]

    def baseline(self, upper, diag, lower, vec):
      diag_part = array_ops.expand_dims(diag, -1) * vec
      lower_part = array_ops.pad(
          array_ops.expand_dims(lower[:, 1:], -1) * vec[:, :-1, :],
          [[0, 0], [1, 0], [0, 0]])
      upper_part = array_ops.pad(
          array_ops.expand_dims(upper[:, :-1], -1) * vec[:, 1:, :],
          [[0, 0], [0, 1], [0, 0]])
      return lower_part + diag_part + upper_part

    def _generateData(self, batch_size, m, n, seed=42):
      np.random.seed(seed)
      data = np.random.normal(size=(batch_size, m, 3 + n))
      return (variables.Variable(data[:, :, 0], dtype=dtypes.float64),
              variables.Variable(data[:, :, 1], dtype=dtypes.float64),
              variables.Variable(data[:, :, 2], dtype=dtypes.float64),
              variables.Variable(data[:, :, 3:], dtype=dtypes.float64))

    def benchmarkTridiagonalMulOp(self):
      devices = [('/cpu:0', 'cpu')]
      if test.is_gpu_available(cuda_only=True):
        devices += [('/gpu:0', 'gpu')]

      for device_option, size_option in itertools.product(devices, self.sizes):
        device_id, device_name = device_option
        m, batch_size, n = size_option

        with ops.Graph().as_default(), \
            session.Session(config=benchmark.benchmark_config()) as sess, \
            ops.device(device_id):
          upper, diag, lower, vec = self._generateData(batch_size, m, n)
          x1 = self.baseline(upper, diag, lower, vec)
          x2 = linalg_impl.tridiagonal_matmul((upper, diag, lower),
                                              vec,
                                              diagonals_format='sequence')

          self.evaluate(variables.global_variables_initializer())
          self.run_op_benchmark(
              sess,
              control_flow_ops.group(x1),
              min_iters=10,
              store_memory_usage=False,
              name=('tridiagonal_matmul_baseline_%s'
                    '_batch_size_%d_m_%d_n_%d' %
                    (device_name, batch_size, m, n)))

          self.run_op_benchmark(
              sess,
              control_flow_ops.group(x2),
              min_iters=10,
              store_memory_usage=False,
              name=('tridiagonal_matmul_%s_batch_size_%d_m_%d_n_%d' %
                    (device_name, batch_size, m, n)))


if __name__ == '__main__':
  test.main()
