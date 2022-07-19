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
"""Tests functionality of math operations on ragged tensors."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest


class RaggedSoftmaxTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def _softmax(self, x):
    assert len(x.shape) == 2
    if x.shape[1] == 0:
      return x
    m = x.max(1)[:, np.newaxis]
    u = np.exp(x - m)
    z = u.sum(1)[:, np.newaxis]
    return u / z

  @test_util.run_in_graph_and_eager_modes
  def testOrdinaryValues(self):
    eps = 1e-5
    x_list = [np.log([0.5, 0.25, 0.25]), np.log([0.5, 0.5])]
    x_row_matrices = [[row] for row in x_list]
    y_row_matrices = [
        self._softmax(np.array(row_matrix)).tolist()
        for row_matrix in x_row_matrices
    ]
    y_list = [row_matrix[0] for row_matrix in y_row_matrices]
    y_expected_from_numpy = ragged_factory_ops.constant(
        y_list, dtype=dtypes.float32)
    y_expected = ragged_factory_ops.constant([[0.5, 0.25, 0.25], [0.5, 0.5]],
                                             dtype=dtypes.float32)
    self.assertAllClose(y_expected_from_numpy, y_expected, eps)
    x_tf = ragged_factory_ops.constant(x_list, dtype=dtypes.float32)
    y_tf = nn_ops.softmax_v2(x_tf)
    self.assertAllClose(y_tf, y_expected_from_numpy, eps)

  @test_util.run_in_graph_and_eager_modes
  def testLargeValues(self):
    eps = 1e-5
    x_list = [[-500, -501, -502], [1729, 1729]]
    x_row_matrices = [[row] for row in x_list]
    y_row_matrices = [
        self._softmax(np.array(row_matrix)).tolist()
        for row_matrix in x_row_matrices
    ]
    y_list = [row_matrix[0] for row_matrix in y_row_matrices]
    y_expected_from_numpy = ragged_factory_ops.constant(
        y_list, dtype=dtypes.float32)
    x_tf = ragged_factory_ops.constant(x_list, dtype=dtypes.float32)
    y_tf = nn_ops.softmax_v2(x_tf)
    self.assertAllClose(y_tf, y_expected_from_numpy, eps)

  @test_util.run_in_graph_and_eager_modes
  def testShortTensors(self):
    eps = 1e-5
    x_list = [[], [1]]
    x_row_matrices = [[row] for row in x_list]
    y_row_matrices = [
        self._softmax(np.array(row_matrix)).tolist()
        for row_matrix in x_row_matrices
    ]
    y_list = [row_matrix[0] for row_matrix in y_row_matrices]
    y_expected_from_numpy = ragged_factory_ops.constant(
        y_list, dtype=dtypes.float32)
    x_tf = ragged_factory_ops.constant(x_list, dtype=dtypes.float32)
    y_tf = nn_ops.softmax_v2(x_tf)
    self.assertAllClose(y_tf, y_expected_from_numpy, eps)


def _cumsum_slow(rt, axis=0, exclusive=False, reverse=False, name=None):
  dense = rt.to_tensor()
  result = math_ops.cumsum(dense, axis=axis, exclusive=exclusive,
                           reverse=reverse, name=name)
  return ragged_tensor.RaggedTensor.from_tensor(
      result, lengths=rt.nested_row_lengths())


class RaggedCumsumTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.parameters([
      dict(original=[[1, 2], [3, 4, 5]],
           expected=[[1, 2], [4, 6, 5]]),
      dict(original=[[1, 2], [3, 4, 5]],
           expected=[[4, 6], [3, 4, 5]],
           reverse=True),
      dict(original=[[1, 2], [3, 4, 5]],
           expected=[[3, 4], [0, 0, 0]],
           reverse=True,
           exclusive=True),
      dict(original=[[1, 2], [3, 4, 5]],
           expected=[[0, 0], [1, 2, 0]],
           exclusive=True),
      dict(original=[[1, 2], [3, 4, 5]],
           expected=[[0, 0], [1, 2, 0]],
           exclusive=True),
      dict(original=[[1, 2], [3, 4, 5]],
           expected=[[1, 3], [3, 7, 12]],
           axis=1),
      dict(original=[[1, 2], [3, 4, 5]],
           expected=[[3, 2], [12, 9, 5]],
           axis=1, reverse=True),
      dict(original=[[1, 2], [3, 4, 5]],
           expected=[[2, 0], [9, 5, 0]],
           axis=1, exclusive=True, reverse=True),
      dict(original=[[1, 2], [3, 4, 5]],
           expected=[[0, 1], [0, 3, 7]],
           axis=1, exclusive=True),
  ])
  def test_cumsum(self, original, expected, axis=0, exclusive=False,
                  reverse=False):
    original_rt = ragged_factory_ops.constant(original)
    expected_rt = ragged_factory_ops.constant(expected)
    actual = ragged_math_ops.ragged_cumsum(
        original_rt, axis=axis, exclusive=exclusive, reverse=reverse)
    self.assertAllEqual(actual, expected_rt)
    baseline = _cumsum_slow(original_rt, axis=axis, exclusive=exclusive,
                            reverse=reverse)
    self.assertAllEqual(baseline, expected_rt)

  @parameterized.parameters([
      dict(expected=[[[0, 1], [2, 5]], [[4, 9], [6, 13], [8, 17]]], axis=2),
      dict(expected=[[[0, 0], [0, 2]], [[0, 4], [0, 6], [0, 8]]],
           exclusive=True, axis=2),
      dict(expected=[[[1, 0], [3, 0]], [[5, 0], [7, 0], [9, 0]]],
           exclusive=True, reverse=True, axis=2),
      dict(expected=[[[1, 1], [5, 3]], [[9, 5], [13, 7], [17, 9]]],
           reverse=True, axis=2),
  ])
  def test_cumsum_deep(self, expected, axis=0, exclusive=False, reverse=False):
    # [[[0, 1], [2, 3]], [[4, 5], [6, 7], [8, 9]]]

    original_rt = ragged_tensor.RaggedTensor.from_row_lengths(
        array_ops.reshape(math_ops.range(10), (5, 2)), [2, 3])

    actual = ragged_math_ops.ragged_cumsum(
        original_rt, axis=axis, exclusive=exclusive, reverse=reverse)
    self.assertAllEqual(actual, expected)
    baseline = _cumsum_slow(original_rt, axis=axis, exclusive=exclusive,
                            reverse=reverse)
    self.assertAllEqual(baseline, expected)

if __name__ == '__main__':
  googletest.main()

