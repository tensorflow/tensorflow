# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# maxlengthations under the License.
# ==============================================================================
"""Tests for bincount ops."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import bincount_ops
from tensorflow.python.ops import gen_count_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test


class TestSparseCount(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "_no_maxlength",
          "x": np.array([[3, 2, 1], [5, 4, 4]], dtype=np.int32),
          "expected_indices": [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5]],
          "expected_values": [1, 1, 1, 2, 1],
          "expected_shape": [2, 6]
      }, {
          "testcase_name": "_maxlength",
          "x": np.array([[3, 2, 1, 7], [7, 0, 4, 4]], dtype=np.int32),
          "maxlength": 7,
          "expected_indices": [[0, 1], [0, 2], [0, 3], [1, 0], [1, 4]],
          "expected_values": [1, 1, 1, 1, 2],
          "expected_shape": [2, 7]
      }, {
          "testcase_name": "_maxlength_zero",
          "x": np.array([[3, 2, 1, 7], [7, 0, 4, 4]], dtype=np.int32),
          "maxlength": 0,
          "expected_indices": np.empty([0, 2], dtype=np.int64),
          "expected_values": [],
          "expected_shape": [2, 0]
      }, {
          "testcase_name": "_minlength",
          "x": np.array([[3, 2, 1, 7], [7, 0, 4, 4]], dtype=np.int32),
          "minlength": 9,
          "expected_indices": [[0, 1], [0, 2], [0, 3], [0, 7], [1, 0], [1, 4],
                               [1, 7]],
          "expected_values": [1, 1, 1, 1, 1, 2, 1],
          "expected_shape": [2, 9]
      }, {
          "testcase_name": "_minlength_larger_values",
          "x": np.array([[3, 2, 1, 7], [7, 0, 4, 4]], dtype=np.int32),
          "minlength": 3,
          "expected_indices": [[0, 1], [0, 2], [0, 3], [0, 7], [1, 0], [1, 4],
                               [1, 7]],
          "expected_values": [1, 1, 1, 1, 1, 2, 1],
          "expected_shape": [2, 8]
      }, {
          "testcase_name": "_no_maxlength_binary",
          "x": np.array([[3, 2, 1], [5, 4, 4]], dtype=np.int32),
          "expected_indices": [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5]],
          "expected_values": [1, 1, 1, 1, 1],
          "expected_shape": [2, 6],
          "binary_output": True,
      }, {
          "testcase_name": "_maxlength_binary",
          "x": np.array([[3, 2, 1, 7], [7, 0, 4, 4]], dtype=np.int32),
          "maxlength": 7,
          "expected_indices": [[0, 1], [0, 2], [0, 3], [1, 0], [1, 4]],
          "expected_values": [1, 1, 1, 1, 1],
          "expected_shape": [2, 7],
          "binary_output": True,
      }, {
          "testcase_name": "_minlength_binary",
          "x": np.array([[3, 2, 1, 7], [7, 0, 4, 4]], dtype=np.int32),
          "minlength": 9,
          "expected_indices": [[0, 1], [0, 2], [0, 3], [0, 7], [1, 0], [1, 4],
                               [1, 7]],
          "expected_values": [1, 1, 1, 1, 1, 1, 1],
          "expected_shape": [2, 9],
          "binary_output": True,
      }, {
          "testcase_name": "_minlength_larger_values_binary",
          "x": np.array([[3, 2, 1, 7], [7, 0, 4, 4]], dtype=np.int32),
          "minlength": 3,
          "expected_indices": [[0, 1], [0, 2], [0, 3], [0, 7], [1, 0], [1, 4],
                               [1, 7]],
          "expected_values": [1, 1, 1, 1, 1, 1, 1],
          "expected_shape": [2, 8],
          "binary_output": True,
      }, {
          "testcase_name": "_no_maxlength_weights",
          "x": np.array([[3, 2, 1], [5, 4, 4]], dtype=np.int32),
          "expected_indices": [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5]],
          "expected_values": [2, 1, 0.5, 9, 3],
          "expected_shape": [2, 6],
          "weights": [[0.5, 1, 2], [3, 4, 5]]
      }, {
          "testcase_name": "_maxlength_weights",
          "x": np.array([[3, 2, 1, 7], [7, 0, 4, 4]], dtype=np.int32),
          "maxlength": 7,
          "expected_indices": [[0, 1], [0, 2], [0, 3], [1, 0], [1, 4]],
          "expected_values": [2, 1, 0.5, 3, 9],
          "expected_shape": [2, 7],
          "weights": [[0.5, 1, 2, 11], [7, 3, 4, 5]]
      }, {
          "testcase_name": "_minlength_weights",
          "x": np.array([[3, 2, 1, 7], [7, 0, 4, 4]], dtype=np.int32),
          "minlength": 9,
          "expected_indices": [[0, 1], [0, 2], [0, 3], [0, 7], [1, 0], [1, 4],
                               [1, 7]],
          "expected_values": [2, 1, 0.5, 3, 5, 13, 4],
          "expected_shape": [2, 9],
          "weights": [[0.5, 1, 2, 3], [4, 5, 6, 7]]
      }, {
          "testcase_name": "_minlength_larger_values_weights",
          "x": np.array([[3, 2, 1, 7], [7, 0, 4, 4]], dtype=np.int32),
          "minlength": 3,
          "expected_indices": [[0, 1], [0, 2], [0, 3], [0, 7], [1, 0], [1, 4],
                               [1, 7]],
          "expected_values": [2, 1, 0.5, 3, 5, 13, 4],
          "expected_shape": [2, 8],
          "weights": [[0.5, 1, 2, 3], [4, 5, 6, 7]]
      }, {
          "testcase_name": "_1d",
          "x": np.array([3, 2, 1, 1], dtype=np.int32),
          "expected_indices": [[1], [2], [3]],
          "expected_values": [2, 1, 1],
          "expected_shape": [4]
      }, {
          "testcase_name": "_all_axes",
          "x": np.array([[3, 2, 1], [5, 4, 4]], dtype=np.int32),
          "expected_indices": [[1], [2], [3], [4], [5]],
          "expected_values": [1, 1, 1, 2, 1],
          "expected_shape": [6],
          "axis": None
      }, {
          "testcase_name":
              "_large_inputs",
          "x":
              np.array([[
                  1941591354222760687, 1748591354222760687, 1241591354229760689
              ], [
                  1941591354222760687, 1241591354229760689, 1241591354229760687
              ]],
                       dtype=np.int64),
          "expected_indices": [[1241591354229760687], [1241591354229760689],
                               [1748591354222760687], [1941591354222760687]],
          "expected_values": [1, 2, 1, 2],
          "expected_shape": [1941591354222760687 + 1],
          "axis":
              None
      })
  def test_dense_input(self,
                       x,
                       expected_indices,
                       expected_values,
                       expected_shape,
                       minlength=None,
                       maxlength=None,
                       binary_output=False,
                       weights=None,
                       axis=-1):
    y = bincount_ops.sparse_bincount(
        x,
        weights=weights,
        minlength=minlength,
        maxlength=maxlength,
        binary_output=binary_output,
        axis=axis)
    self.assertAllEqual(expected_indices, y.indices)
    self.assertAllEqual(expected_values, y.values)
    self.assertAllEqual(expected_shape, y.dense_shape)

  @parameterized.named_parameters(
      {
          "testcase_name":
              "_no_maxlength",
          "x":
              np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]],
                       dtype=np.int32),
          "expected_indices": [[0, 1], [0, 3], [2, 4], [2, 5]],
          "expected_values": [1, 1, 2, 1],
          "expected_shape": [3, 6],
      }, {
          "testcase_name":
              "_maxlength",
          "x":
              np.array([[3, 0, 1, 0], [7, 0, 0, 0], [5, 0, 4, 4]],
                       dtype=np.int32),
          "expected_indices": [[0, 1], [0, 3], [2, 4], [2, 5]],
          "expected_values": [1, 1, 2, 1],
          "expected_shape": [3, 7],
          "maxlength":
              7,
      }, {
          "testcase_name":
              "_maxlength_zero",
          "x":
              np.array([[3, 0, 1, 0], [7, 0, 0, 0], [5, 0, 4, 4]],
                       dtype=np.int32),
          "expected_indices":
              np.empty([0, 2], dtype=np.int64),
          "expected_values": [],
          "expected_shape": [3, 0],
          "maxlength":
              0,
      }, {
          "testcase_name":
              "_minlength",
          "x":
              np.array([[3, 0, 1, 0], [7, 0, 0, 0], [5, 0, 4, 4]],
                       dtype=np.int32),
          "expected_indices": [[0, 1], [0, 3], [1, 7], [2, 4], [2, 5]],
          "expected_values": [1, 1, 1, 2, 1],
          "expected_shape": [3, 9],
          "minlength":
              9,
      }, {
          "testcase_name":
              "_minlength_larger_values",
          "x":
              np.array([[3, 0, 1, 0], [7, 0, 0, 0], [5, 0, 4, 4]],
                       dtype=np.int32),
          "expected_indices": [[0, 1], [0, 3], [1, 7], [2, 4], [2, 5]],
          "expected_values": [1, 1, 1, 2, 1],
          "expected_shape": [3, 8],
          "minlength":
              3,
      }, {
          "testcase_name":
              "_no_maxlength_binary",
          "x":
              np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]],
                       dtype=np.int32),
          "expected_indices": [[0, 1], [0, 3], [2, 4], [2, 5]],
          "expected_values": [1, 1, 1, 1],
          "expected_shape": [3, 6],
          "binary_output":
              True,
      }, {
          "testcase_name":
              "_maxlength_binary",
          "x":
              np.array([[3, 0, 1, 0], [0, 0, 7, 0], [5, 0, 4, 4]],
                       dtype=np.int32),
          "expected_indices": [[0, 1], [0, 3], [2, 4], [2, 5]],
          "expected_values": [1, 1, 1, 1],
          "expected_shape": [3, 7],
          "maxlength":
              7,
          "binary_output":
              True,
      }, {
          "testcase_name":
              "_minlength_binary",
          "x":
              np.array([[3, 0, 1, 0], [7, 0, 0, 0], [5, 0, 4, 4]],
                       dtype=np.int32),
          "expected_indices": [[0, 1], [0, 3], [1, 7], [2, 4], [2, 5]],
          "expected_values": [1, 1, 1, 1, 1],
          "expected_shape": [3, 9],
          "minlength":
              9,
          "binary_output":
              True,
      }, {
          "testcase_name":
              "_minlength_larger_values_binary",
          "x":
              np.array([[3, 0, 1, 0], [7, 0, 0, 0], [5, 0, 4, 4]],
                       dtype=np.int32),
          "expected_indices": [[0, 1], [0, 3], [1, 7], [2, 4], [2, 5]],
          "expected_values": [1, 1, 1, 1, 1],
          "expected_shape": [3, 8],
          "minlength":
              3,
          "binary_output":
              True,
      }, {
          "testcase_name":
              "_no_maxlength_weights",
          "x":
              np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]],
                       dtype=np.int32),
          "expected_indices": [[0, 1], [0, 3], [2, 4], [2, 5]],
          "expected_values": [2, 6, 7, 10],
          "expected_shape": [3, 6],
          "weights":
              np.array([[6, 0, 2, 0], [0, 0, 0, 0], [10, 0, 3.5, 3.5]]),
      }, {
          "testcase_name":
              "_maxlength_weights",
          "x":
              np.array([[3, 0, 1, 0], [0, 0, 7, 0], [5, 0, 4, 4]],
                       dtype=np.int32),
          "expected_indices": [[0, 1], [0, 3], [2, 4], [2, 5]],
          "expected_values": [2, 6, 7, 10],
          "expected_shape": [3, 7],
          "maxlength":
              7,
          "weights":
              np.array([[6, 0, 2, 0], [0, 0, 14, 0], [10, 0, 3.5, 3.5]]),
      }, {
          "testcase_name":
              "_minlength_weights",
          "x":
              np.array([[3, 0, 1, 0], [7, 0, 0, 0], [5, 0, 4, 4]],
                       dtype=np.int32),
          "expected_indices": [[0, 1], [0, 3], [1, 7], [2, 4], [2, 5]],
          "expected_values": [2, 6, 14, 6.5, 10],
          "expected_shape": [3, 9],
          "minlength":
              9,
          "weights":
              np.array([[6, 0, 2, 0], [14, 0, 0, 0], [10, 0, 3, 3.5]]),
      }, {
          "testcase_name":
              "_minlength_larger_values_weights",
          "x":
              np.array([[3, 0, 1, 0], [7, 0, 0, 0], [5, 0, 4, 4]],
                       dtype=np.int32),
          "expected_indices": [[0, 1], [0, 3], [1, 7], [2, 4], [2, 5]],
          "expected_values": [2, 6, 14, 6.5, 10],
          "expected_shape": [3, 8],
          "minlength":
              3,
          "weights":
              np.array([[6, 0, 2, 0], [14, 0, 0, 0], [10, 0, 3, 3.5]]),
      }, {
          "testcase_name": "_1d",
          "x": np.array([3, 0, 1, 1], dtype=np.int32),
          "expected_indices": [[1], [3]],
          "expected_values": [2, 1],
          "expected_shape": [4],
      }, {
          "testcase_name":
              "_all_axes",
          "x":
              np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]],
                       dtype=np.int32),
          "expected_indices": [[1], [3], [4], [5]],
          "expected_values": [1, 1, 2, 1],
          "expected_shape": [6],
          "axis":
              None,
      }, {
          "testcase_name":
              "_large_inputs",
          "x":
              np.array([[1941591354222760687, 0, 1241591354229760689],
                        [0, 1241591354229760689, 1241591354229760687]],
                       dtype=np.int64),
          "expected_indices": [[1241591354229760687], [1241591354229760689],
                               [1941591354222760687]],
          "expected_values": [1, 2, 1],
          "expected_shape": [1941591354222760687 + 1],
          "axis":
              None
      })
  def test_sparse_input(self,
                        x,
                        expected_indices,
                        expected_values,
                        expected_shape,
                        maxlength=None,
                        minlength=None,
                        binary_output=False,
                        weights=None,
                        axis=-1):
    x_sparse = sparse_ops.from_dense(x)
    w_sparse = sparse_ops.from_dense(weights) if weights is not None else None
    y = bincount_ops.sparse_bincount(
        x_sparse,
        weights=w_sparse,
        minlength=minlength,
        maxlength=maxlength,
        binary_output=binary_output,
        axis=axis)
    self.assertAllEqual(expected_indices, y.indices)
    self.assertAllEqual(expected_values, y.values)
    self.assertAllEqual(expected_shape, y.dense_shape)

  @parameterized.named_parameters(
      {
          "testcase_name": "_no_maxlength",
          "x": [[], [], [3, 0, 1], [], [5, 0, 4, 4]],
          "expected_indices": [[2, 0], [2, 1], [2, 3], [4, 0], [4, 4], [4, 5]],
          "expected_values": [1, 1, 1, 1, 2, 1],
          "expected_shape": [5, 6],
      }, {
          "testcase_name": "_maxlength",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "maxlength": 7,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [4, 0], [4, 4], [4, 5]],
          "expected_values": [1, 1, 1, 1, 2, 1],
          "expected_shape": [5, 7],
      }, {
          "testcase_name": "_maxlength_zero",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "maxlength": 0,
          "expected_indices": np.empty([0, 2], dtype=np.int64),
          "expected_values": [],
          "expected_shape": [5, 0],
      }, {
          "testcase_name": "_minlength",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "minlength": 9,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [3, 7], [4, 0], [4, 4],
                               [4, 5]],
          "expected_values": [1, 1, 1, 1, 1, 2, 1],
          "expected_shape": [5, 9],
      }, {
          "testcase_name": "_minlength_larger_values",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "minlength": 3,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [3, 7], [4, 0], [4, 4],
                               [4, 5]],
          "expected_values": [1, 1, 1, 1, 1, 2, 1],
          "expected_shape": [5, 8],
      }, {
          "testcase_name": "_no_maxlength_binary",
          "x": [[], [], [3, 0, 1], [], [5, 0, 4, 4]],
          "expected_indices": [[2, 0], [2, 1], [2, 3], [4, 0], [4, 4], [4, 5]],
          "expected_values": [1, 1, 1, 1, 1, 1],
          "expected_shape": [5, 6],
          "binary_output": True,
      }, {
          "testcase_name": "_maxlength_binary",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "maxlength": 7,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [4, 0], [4, 4], [4, 5]],
          "expected_values": [1, 1, 1, 1, 1, 1],
          "expected_shape": [5, 7],
          "binary_output": True,
      }, {
          "testcase_name": "_minlength_binary",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "minlength": 9,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [3, 7], [4, 0], [4, 4],
                               [4, 5]],
          "expected_values": [1, 1, 1, 1, 1, 1, 1],
          "expected_shape": [5, 9],
          "binary_output": True,
      }, {
          "testcase_name": "_minlength_larger_values_binary",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "minlength": 3,
          "binary_output": True,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [3, 7], [4, 0], [4, 4],
                               [4, 5]],
          "expected_values": [1, 1, 1, 1, 1, 1, 1],
          "expected_shape": [5, 8],
      }, {
          "testcase_name": "_no_maxlength_weights",
          "x": [[], [], [3, 0, 1], [], [5, 0, 4, 4]],
          "expected_indices": [[2, 0], [2, 1], [2, 3], [4, 0], [4, 4], [4, 5]],
          "expected_values": [0.5, 2, 6, 0.25, 8, 10],
          "expected_shape": [5, 6],
          "weights": [[], [], [6, 0.5, 2], [], [10, 0.25, 5, 3]],
      }, {
          "testcase_name": "_maxlength_weights",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "maxlength": 7,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [4, 0], [4, 4], [4, 5]],
          "expected_values": [0.5, 2, 6, 0.25, 8, 10],
          "expected_shape": [5, 7],
          "weights": [[], [], [6, 0.5, 2], [14], [10, 0.25, 5, 3]],
      }, {
          "testcase_name": "_minlength_weights",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "minlength": 9,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [3, 7], [4, 0], [4, 4],
                               [4, 5]],
          "expected_values": [0.5, 2, 6, 14, 0.25, 8, 10],
          "expected_shape": [5, 9],
          "weights": [[], [], [6, 0.5, 2], [14], [10, 0.25, 5, 3]],
      }, {
          "testcase_name": "_minlength_larger_values_weights",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "minlength": 3,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [3, 7], [4, 0], [4, 4],
                               [4, 5]],
          "expected_values": [0.5, 2, 6, 14, 0.25, 8, 10],
          "expected_shape": [5, 8],
          "weights": [[], [], [6, 0.5, 2], [14], [10, 0.25, 5, 3]],
      }, {
          "testcase_name": "_1d",
          "x": [3, 0, 1, 1],
          "expected_indices": [[0], [1], [3]],
          "expected_values": [1, 2, 1],
          "expected_shape": [4],
      }, {
          "testcase_name": "_all_axes",
          "x": [[], [], [3, 0, 1], [], [5, 0, 4, 4]],
          "expected_indices": [[0], [1], [3], [4], [5]],
          "expected_values": [2, 1, 1, 2, 1],
          "expected_shape": [6],
          "axis": None,
      }, {
          "testcase_name": "_large_inputs",
          "x": [[1941591354222760687, 1748591354222760687],
                [1941591354222760687, 1241591354229760689, 1241591354229760687]
               ],
          "expected_indices": [[1241591354229760687], [1241591354229760689],
                               [1748591354222760687], [1941591354222760687]],
          "expected_values": [1, 1, 1, 2],
          "expected_shape": [1941591354222760687 + 1],
          "axis": None
      })
  def test_ragged_input(self,
                        x,
                        expected_indices,
                        expected_values,
                        expected_shape,
                        maxlength=None,
                        minlength=None,
                        binary_output=False,
                        weights=None,
                        axis=-1):
    x_ragged = ragged_factory_ops.constant(x)
    w = ragged_factory_ops.constant(weights) if weights is not None else None
    y = bincount_ops.sparse_bincount(
        x_ragged,
        weights=w,
        minlength=minlength,
        maxlength=maxlength,
        binary_output=binary_output,
        axis=axis)
    self.assertAllEqual(expected_indices, y.indices)
    self.assertAllEqual(expected_values, y.values)
    self.assertAllEqual(expected_shape, y.dense_shape)


class TestDenseBincount(test.TestCase, parameterized.TestCase):

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_input_all_count(self, dtype):
    np.random.seed(42)
    num_rows = 128
    size = 1000
    n_elems = 4096
    inp_indices = np.random.randint(0, num_rows, (n_elems, 1))
    inp_indices = np.concatenate([inp_indices, np.zeros((n_elems, 1))], axis=1)
    inp_vals = np.random.randint(0, size, (n_elems,), dtype=dtype)
    sparse_inp = sparse_tensor.SparseTensor(inp_indices, inp_vals,
                                            [num_rows, 1])

    np_out = np.bincount(inp_vals, minlength=size)
    self.assertAllEqual(
        np_out, self.evaluate(bincount_ops.bincount(sparse_inp, axis=0)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_input_all_count_with_weights(self, dtype):
    np.random.seed(42)
    num_rows = 128
    size = 1000
    n_elems = 4096
    inp_indices = np.random.randint(0, num_rows, (n_elems, 1))
    inp_indices = np.concatenate([inp_indices, np.zeros((n_elems, 1))], axis=1)
    inp_vals = np.random.randint(0, size, (n_elems,), dtype=dtype)
    sparse_inp = sparse_tensor.SparseTensor(inp_indices, inp_vals,
                                            [num_rows, 1])
    weight_vals = np.random.random((n_elems,))
    sparse_weights = sparse_tensor.SparseTensor(inp_indices, weight_vals,
                                                [num_rows, 1])

    np_out = np.bincount(inp_vals, minlength=size, weights=weight_vals)
    self.assertAllEqual(
        np_out,
        self.evaluate(bincount_ops.bincount(
            sparse_inp, sparse_weights, axis=0)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_input_all_binary(self, dtype):
    np.random.seed(42)
    num_rows = 128
    size = 10
    n_elems = 4096
    inp_indices = np.random.randint(0, num_rows, (n_elems, 1))
    inp_indices = np.concatenate([inp_indices, np.zeros((n_elems, 1))], axis=1)
    inp_vals = np.random.randint(0, size, (n_elems,), dtype=dtype)
    sparse_inp = sparse_tensor.SparseTensor(inp_indices, inp_vals,
                                            [num_rows, 1])

    np_out = np.ones((size,))
    self.assertAllEqual(
        np_out,
        self.evaluate(bincount_ops.bincount(sparse_inp, binary_output=True)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_input_col_reduce_count(self, dtype):
    num_rows = 128
    num_cols = 27
    size = 100
    np.random.seed(42)
    inp = np.random.randint(0, size, (num_rows, num_cols), dtype=dtype)
    np_out = np.reshape(
        np.concatenate(
            [np.bincount(inp[j, :], minlength=size) for j in range(num_rows)],
            axis=0), (num_rows, size))
    # from_dense will filter out 0s.
    inp = inp + 1
    # from_dense will cause OOM in GPU.
    with ops.device("/CPU:0"):
      inp_sparse = sparse_ops.from_dense(inp)
      inp_sparse = sparse_tensor.SparseTensor(inp_sparse.indices,
                                              inp_sparse.values - 1,
                                              inp_sparse.dense_shape)
    self.assertAllEqual(
        np_out, self.evaluate(bincount_ops.bincount(arr=inp_sparse, axis=-1)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_input_col_reduce_binary(self, dtype):
    num_rows = 128
    num_cols = 27
    size = 100
    np.random.seed(42)
    inp = np.random.randint(0, size, (num_rows, num_cols), dtype=dtype)
    np_out = np.reshape(
        np.concatenate([
            np.where(np.bincount(inp[j, :], minlength=size) > 0, 1, 0)
            for j in range(num_rows)
        ],
                       axis=0), (num_rows, size))
    # from_dense will filter out 0s.
    inp = inp + 1
    # from_dense will cause OOM in GPU.
    with ops.device("/CPU:0"):
      inp_sparse = sparse_ops.from_dense(inp)
      inp_sparse = sparse_tensor.SparseTensor(inp_sparse.indices,
                                              inp_sparse.values - 1,
                                              inp_sparse.dense_shape)
    self.assertAllEqual(
        np_out,
        self.evaluate(
            bincount_ops.bincount(arr=inp_sparse, axis=-1, binary_output=True)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_ragged_input_count(self, dtype):
    x = ragged_factory_ops.constant([[], [], [3, 0, 1], [], [5, 0, 4, 4]],
                                    dtype)
    # pyformat: disable
    expected_output = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 2, 1]]
    # pyformat: enable
    self.assertAllEqual(expected_output,
                        self.evaluate(bincount_ops.bincount(arr=x, axis=-1)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_ragged_input_binary(self, dtype):
    x = ragged_factory_ops.constant([[], [], [3, 0, 1], [], [5, 0, 4, 4]])
    # pyformat: disable
    expected_output = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 1]]
    # pyformat: enable
    self.assertAllEqual(
        expected_output,
        self.evaluate(
            bincount_ops.bincount(arr=x, axis=-1, binary_output=True)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_ragged_input_count_with_weights(self, dtype):
    x = ragged_factory_ops.constant([[], [], [3, 0, 1], [], [5, 0, 4, 4]])
    weights = ragged_factory_ops.constant([[], [], [.1, .2, .3], [],
                                           [.2, .5, .6, .3]])
    # pyformat: disable
    expected_output = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [.2, .3, 0, .1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [.5, 0, 0, 0, .9, .2]]
    # pyformat: enable
    self.assertAllClose(
        expected_output,
        self.evaluate(bincount_ops.bincount(arr=x, weights=weights, axis=-1)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_ragged_input_count_np(self, dtype):
    np.random.seed(42)
    num_rows = 128
    num_cols = 27
    size = 1000
    inp = np.random.randint(0, size, (num_rows, num_cols), dtype=dtype)
    np_out = np.reshape(
        np.concatenate(
            [np.bincount(inp[j, :], minlength=size) for j in range(num_rows)],
            axis=0), (num_rows, size))
    x = ragged_tensor.RaggedTensor.from_tensor(inp)
    self.assertAllEqual(
        np_out,
        self.evaluate(bincount_ops.bincount(arr=x, minlength=size, axis=-1)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_ragged_input_count_np_with_weights(self, dtype):
    np.random.seed(42)
    num_rows = 128
    num_cols = 27
    size = 1000
    inp = np.random.randint(0, size, (num_rows, num_cols), dtype=dtype)
    np_weight = np.random.random((num_rows, num_cols))
    np_out = np.reshape(
        np.concatenate([
            np.bincount(inp[j, :], weights=np_weight[j, :], minlength=size)
            for j in range(num_rows)
        ],
                       axis=0), (num_rows, size))
    x = ragged_tensor.RaggedTensor.from_tensor(inp)
    weights = ragged_tensor.RaggedTensor.from_tensor(np_weight)
    self.assertAllEqual(
        np_out,
        self.evaluate(
            bincount_ops.bincount(
                arr=x, weights=weights, minlength=size, axis=-1)))


class TestSparseCountFailureModes(test.TestCase):

  def test_dense_input_sparse_weights_fails(self):
    x = np.array([[3, 2, 1], [5, 4, 4]], dtype=np.int32)
    weights = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    with self.assertRaisesRegex(ValueError, "must be a tf.Tensor"):
      self.evaluate(bincount_ops.sparse_bincount(x, weights=weights, axis=-1))

  def test_dense_input_ragged_weights_fails(self):
    x = np.array([[3, 2, 1], [5, 4, 4]], dtype=np.int32)
    weights = ragged_factory_ops.constant([[6, 0.5, 2], [14], [10, 0.25, 5, 3]])
    with self.assertRaisesRegex(ValueError, "must be a tf.Tensor"):
      self.evaluate(bincount_ops.sparse_bincount(x, weights=weights, axis=-1))

  def test_dense_input_wrong_shape_fails(self):
    x = np.array([[3, 2, 1], [5, 4, 4]], dtype=np.int32)
    weights = np.array([[3, 2], [5, 4], [4, 3]])
    # Note: Eager mode and graph mode throw different errors here. Graph mode
    # will fail with a ValueError from the shape checking logic, while Eager
    # will fail with an InvalidArgumentError from the kernel itself.
    if context.executing_eagerly():
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "must have the same shape"):
        self.evaluate(bincount_ops.sparse_bincount(x, weights=weights, axis=-1))
    else:
      with self.assertRaisesRegex(ValueError, "both shapes must be equal"):
        self.evaluate(bincount_ops.sparse_bincount(x, weights=weights, axis=-1))

  def test_sparse_input_dense_weights_fails(self):
    x = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    weights = np.array([[3, 2, 1], [5, 4, 4]], dtype=np.int32)
    with self.assertRaisesRegex(ValueError, "must be a SparseTensor"):
      self.evaluate(bincount_ops.sparse_bincount(x, weights=weights, axis=-1))

  def test_sparse_input_ragged_weights_fails(self):
    x = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    weights = ragged_factory_ops.constant([[6, 0.5, 2], [14], [10, 0.25, 5, 3]])
    with self.assertRaisesRegex(ValueError, "must be a SparseTensor"):
      self.evaluate(bincount_ops.sparse_bincount(x, weights=weights, axis=-1))

  def test_sparse_input_wrong_indices_fails(self):
    x = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    weights = sparse_ops.from_dense(
        np.array([[3, 1, 0, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "must have the same indices"):
      self.evaluate(bincount_ops.sparse_bincount(x, weights=weights, axis=-1))

  def test_sparse_input_too_many_indices_fails(self):
    x = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    weights = sparse_ops.from_dense(
        np.array([[3, 1, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    with self.assertRaisesIncompatibleShapesError():
      self.evaluate(bincount_ops.sparse_bincount(x, weights=weights, axis=-1))

  def test_sparse_input_wrong_shape_fails(self):
    x = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    weights = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4], [0, 0, 0, 0]],
                 dtype=np.int32))
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "must have the same dense shape"):
      self.evaluate(bincount_ops.sparse_bincount(x, weights=weights, axis=-1))

  def test_ragged_input_dense_weights_fails(self):
    x = ragged_factory_ops.constant([[6, 1, 2], [14], [10, 1, 5, 3]])
    weights = np.array([[3, 2, 1], [5, 4, 4]], dtype=np.int32)
    with self.assertRaisesRegex(ValueError, "must be a RaggedTensor"):
      self.evaluate(bincount_ops.sparse_bincount(x, weights=weights, axis=-1))

  def test_ragged_input_sparse_weights_fails(self):
    x = ragged_factory_ops.constant([[6, 1, 2], [14], [10, 1, 5, 3]])
    weights = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    with self.assertRaisesRegex(ValueError, "must be a RaggedTensor"):
      self.evaluate(bincount_ops.sparse_bincount(x, weights=weights, axis=-1))

  def test_ragged_input_different_shape_fails(self):
    x = ragged_factory_ops.constant([[6, 1, 2], [14], [10, 1, 5, 3]])
    weights = ragged_factory_ops.constant([[6, 0.5, 2], [], [10, 0.25, 5, 3]])
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "must have the same row splits"):
      self.evaluate(bincount_ops.sparse_bincount(x, weights=weights, axis=-1))


class RawOpsHeapOobTest(test.TestCase, parameterized.TestCase):

  @test_util.run_v1_only("Test security error")
  def testSparseCountSparseOutputBadIndicesShapeTooSmall(self):
    indices = [1]
    values = [[1]]
    weights = []
    dense_shape = [10]
    with self.assertRaisesRegex(ValueError,
                                "Shape must be rank 2 but is rank 1 for"):
      self.evaluate(
          gen_count_ops.SparseCountSparseOutput(
              indices=indices,
              values=values,
              dense_shape=dense_shape,
              weights=weights,
              binary_output=True))


@test_util.run_all_in_graph_and_eager_modes
@test_util.disable_tfrt
class RawOpsTest(test.TestCase, parameterized.TestCase):

  def testSparseCountSparseOutputBadIndicesShape(self):
    indices = [[[0], [0]], [[0], [1]], [[1], [0]], [[1], [2]]]
    values = [1, 1, 1, 10]
    weights = [1, 2, 4, 6]
    dense_shape = [2, 3]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Input indices must be a 2-dimensional tensor"):
      self.evaluate(
          gen_count_ops.SparseCountSparseOutput(
              indices=indices,
              values=values,
              dense_shape=dense_shape,
              weights=weights,
              binary_output=False))

  def testSparseCountSparseOutputBadWeightsShape(self):
    indices = [[0, 0], [0, 1], [1, 0], [1, 2]]
    values = [1, 1, 1, 10]
    weights = [1, 2, 4]
    dense_shape = [2, 3]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Weights and values must have the same shape"):
      self.evaluate(
          gen_count_ops.SparseCountSparseOutput(
              indices=indices,
              values=values,
              dense_shape=dense_shape,
              weights=weights,
              binary_output=False))

  def testSparseCountSparseOutputBadNumberOfValues(self):
    indices = [[0, 0], [0, 1], [1, 0]]
    values = [1, 1, 1, 10]
    weights = [1, 2, 4, 6]
    dense_shape = [2, 3]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Number of values must match first dimension of indices"):
      self.evaluate(
          gen_count_ops.SparseCountSparseOutput(
              indices=indices,
              values=values,
              dense_shape=dense_shape,
              weights=weights,
              binary_output=False))

  def testSparseCountSparseOutputNegativeValue(self):
    indices = [[0, 0], [0, 1], [1, 0], [1, 2]]
    values = [1, 1, -1, 10]
    dense_shape = [2, 3]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Input values must all be non-negative"):
      self.evaluate(
          gen_count_ops.SparseCountSparseOutput(
              indices=indices,
              values=values,
              dense_shape=dense_shape,
              binary_output=False))

  def testRaggedCountSparseOutput(self):
    splits = [0, 4, 7]
    values = [1, 1, 2, 1, 2, 10, 5]
    weights = [1, 2, 3, 4, 5, 6, 7]
    output_indices, output_values, output_shape = self.evaluate(
        gen_count_ops.RaggedCountSparseOutput(
            splits=splits, values=values, weights=weights, binary_output=False))
    self.assertAllEqual([[0, 1], [0, 2], [1, 2], [1, 5], [1, 10]],
                        output_indices)
    self.assertAllEqual([7, 3, 5, 7, 6], output_values)
    self.assertAllEqual([2, 11], output_shape)

  def testRaggedCountSparseOutputBadWeightsShape(self):
    splits = [0, 4, 7]
    values = [1, 1, 2, 1, 2, 10, 5]
    weights = [1, 2, 3, 4, 5, 6]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Weights and values must have the same shape"):
      self.evaluate(
          gen_count_ops.RaggedCountSparseOutput(
              splits=splits,
              values=values,
              weights=weights,
              binary_output=False))

  def testRaggedCountSparseOutputEmptySplits(self):
    splits = []
    values = [1, 1, 2, 1, 2, 10, 5]
    weights = [1, 2, 3, 4, 5, 6, 7]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Must provide at least 2 elements for the splits argument"):
      self.evaluate(
          gen_count_ops.RaggedCountSparseOutput(
              splits=splits,
              values=values,
              weights=weights,
              binary_output=False))

  def testRaggedCountSparseOutputBadSplitsStart(self):
    splits = [1, 7]
    values = [1, 1, 2, 1, 2, 10, 5]
    weights = [1, 2, 3, 4, 5, 6, 7]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Splits must start with 0"):
      self.evaluate(
          gen_count_ops.RaggedCountSparseOutput(
              splits=splits,
              values=values,
              weights=weights,
              binary_output=False))

  def testRaggedCountSparseOutputBadSplitsEnd(self):
    splits = [0, 5]
    values = [1, 1, 2, 1, 2, 10, 5]
    weights = [1, 2, 3, 4, 5, 6, 7]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Splits must end with the number of values"):
      self.evaluate(
          gen_count_ops.RaggedCountSparseOutput(
              splits=splits,
              values=values,
              weights=weights,
              binary_output=False))

  def testRaggedCountSparseOutputNegativeValue(self):
    splits = [0, 4, 7]
    values = [1, 1, 2, 1, -2, 10, 5]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Input values must all be non-negative"):
      self.evaluate(
          gen_count_ops.RaggedCountSparseOutput(
              splits=splits, values=values, binary_output=False))


if __name__ == "__main__":
  test.main()
