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
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test as test_lib


class RaggedSoftmaxTest(test_lib.TestCase, parameterized.TestCase):

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
