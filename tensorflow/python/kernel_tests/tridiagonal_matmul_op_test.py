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

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linalg_impl
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


def _tfconst(array):
  return constant_op.constant(array, dtypes.float64)


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

    with self.cached_session(use_gpu=False):
      results = self.evaluate(results)
      results_batch = self.evaluate(results_batch)

    expected = np.array(expected)
    expected_batch = np.stack([expected, 4 * expected])
    for result in results:
      self.assertAllClose(result, expected)
    for result in results_batch:
      self.assertAllClose(result, expected_batch)

  def test1x1(self):
    self._testAllFormats([], [2], [], [[1, 4]], [[2, 8]])

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

  # Benchmark

  class TridiagonalMulBenchmark(test.Benchmark):
    sizes = [(1, 1000000), (1000000, 1), (1000, 1000), (10000, 10000)]

    def baseline(self, upper, diag, lower, vec):
      diag_part = diag * vec
      lower_part = array_ops.pad(lower * vec[:, :-1], [[0, 0], [1, 0]])
      upper_part = array_ops.pad(upper * vec[:, 1:], [[0, 0], [0, 1]])
      return lower_part + diag_part + upper_part

    def _generateData(self, batch_size, matrix_size, seed=42):
      np.random.seed(seed)
      data = np.random.normal(size=(batch_size, matrix_size, 4))
      upper = data[:, 1:, 0]
      diag = data[:, :, 1]
      lower = data[:, 1:, 2]
      vec = data[:, :, 3]

      return (ops.convert_to_tensor(upper, dtype=dtypes.float64),
              ops.convert_to_tensor(diag, dtype=dtypes.float64),
              ops.convert_to_tensor(lower, dtype=dtypes.float64),
              ops.convert_to_tensor(vec, dtype=dtypes.float64))

    def benchmarkTridiagonalMulOp(self):
      devices = [('/cpu:0', 'cpu')]

      for device_id, device_name in devices:
        for batch_size, matrix_size in self.sizes:
          with ops.Graph().as_default(), \
              session.Session(config=benchmark.benchmark_config()) as sess, \
              ops.device(device_id):
            upper, diag, lower, vec = self._generateData(
                batch_size, matrix_size)
            x1 = self.baseline(upper, diag, lower, vec)
            x2 = linalg_impl.tridiagonal_matmul((upper, diag, lower), vec)
            variables.global_variables_initializer().run()
            self.run_op_benchmark(
                sess,
                control_flow_ops.group(x1),
                min_iters=10,
                store_memory_usage=False,
                name=('tridiagonal_matmul_baseline_%s'
                      '_batch_size_%d_matrix_size_%d' %
                      (device_name, batch_size, matrix_size)))

            self.run_op_benchmark(
                sess,
                control_flow_ops.group(x2),
                min_iters=10,
                store_memory_usage=False,
                name=('tridiagonal_matmul_%s_batch_size_%d_matrix_size_%d' %
                      (device_name, batch_size, matrix_size)))


if __name__ == '__main__':
  test.main()
