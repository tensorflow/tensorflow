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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.ops import bincount
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
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
    y = bincount.sparse_bincount(
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
      },
      {
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
      },
      {
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
      },
      {
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
      },
      {
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
      },
      {
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
      },
      {
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
      },
      {
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
      },
      {
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
      },
      {
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
      },
      {
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
      },
      {
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
      },
      {
          "testcase_name": "_1d",
          "x": np.array([3, 0, 1, 1], dtype=np.int32),
          "expected_indices": [[1], [3]],
          "expected_values": [2, 1],
          "expected_shape": [4],
      },
      {
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
      },
  )
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
    y = bincount.sparse_bincount(
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
      },
      {
          "testcase_name": "_maxlength",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "maxlength": 7,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [4, 0], [4, 4], [4, 5]],
          "expected_values": [1, 1, 1, 1, 2, 1],
          "expected_shape": [5, 7],
      },
      {
          "testcase_name": "_minlength",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "minlength": 9,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [3, 7], [4, 0], [4, 4],
                               [4, 5]],
          "expected_values": [1, 1, 1, 1, 1, 2, 1],
          "expected_shape": [5, 9],
      },
      {
          "testcase_name": "_minlength_larger_values",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "minlength": 3,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [3, 7], [4, 0], [4, 4],
                               [4, 5]],
          "expected_values": [1, 1, 1, 1, 1, 2, 1],
          "expected_shape": [5, 8],
      },
      {
          "testcase_name": "_no_maxlength_binary",
          "x": [[], [], [3, 0, 1], [], [5, 0, 4, 4]],
          "expected_indices": [[2, 0], [2, 1], [2, 3], [4, 0], [4, 4], [4, 5]],
          "expected_values": [1, 1, 1, 1, 1, 1],
          "expected_shape": [5, 6],
          "binary_output": True,
      },
      {
          "testcase_name": "_maxlength_binary",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "maxlength": 7,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [4, 0], [4, 4], [4, 5]],
          "expected_values": [1, 1, 1, 1, 1, 1],
          "expected_shape": [5, 7],
          "binary_output": True,
      },
      {
          "testcase_name": "_minlength_binary",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "minlength": 9,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [3, 7], [4, 0], [4, 4],
                               [4, 5]],
          "expected_values": [1, 1, 1, 1, 1, 1, 1],
          "expected_shape": [5, 9],
          "binary_output": True,
      },
      {
          "testcase_name": "_minlength_larger_values_binary",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "minlength": 3,
          "binary_output": True,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [3, 7], [4, 0], [4, 4],
                               [4, 5]],
          "expected_values": [1, 1, 1, 1, 1, 1, 1],
          "expected_shape": [5, 8],
      },
      {
          "testcase_name": "_no_maxlength_weights",
          "x": [[], [], [3, 0, 1], [], [5, 0, 4, 4]],
          "expected_indices": [[2, 0], [2, 1], [2, 3], [4, 0], [4, 4], [4, 5]],
          "expected_values": [0.5, 2, 6, 0.25, 8, 10],
          "expected_shape": [5, 6],
          "weights": [[], [], [6, 0.5, 2], [], [10, 0.25, 5, 3]],
      },
      {
          "testcase_name": "_maxlength_weights",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "maxlength": 7,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [4, 0], [4, 4], [4, 5]],
          "expected_values": [0.5, 2, 6, 0.25, 8, 10],
          "expected_shape": [5, 7],
          "weights": [[], [], [6, 0.5, 2], [14], [10, 0.25, 5, 3]],
      },
      {
          "testcase_name": "_minlength_weights",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "minlength": 9,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [3, 7], [4, 0], [4, 4],
                               [4, 5]],
          "expected_values": [0.5, 2, 6, 14, 0.25, 8, 10],
          "expected_shape": [5, 9],
          "weights": [[], [], [6, 0.5, 2], [14], [10, 0.25, 5, 3]],
      },
      {
          "testcase_name": "_minlength_larger_values_weights",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "minlength": 3,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [3, 7], [4, 0], [4, 4],
                               [4, 5]],
          "expected_values": [0.5, 2, 6, 14, 0.25, 8, 10],
          "expected_shape": [5, 8],
          "weights": [[], [], [6, 0.5, 2], [14], [10, 0.25, 5, 3]],
      },
      {
          "testcase_name": "_1d",
          "x": [3, 0, 1, 1],
          "expected_indices": [[0], [1], [3]],
          "expected_values": [1, 2, 1],
          "expected_shape": [4],
      },
      {
          "testcase_name": "_all_axes",
          "x": [[], [], [3, 0, 1], [], [5, 0, 4, 4]],
          "expected_indices": [[0], [1], [3], [4], [5]],
          "expected_values": [2, 1, 1, 2, 1],
          "expected_shape": [6],
          "axis": None,
      },
  )
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
    y = bincount.sparse_bincount(
        x_ragged,
        weights=w,
        minlength=minlength,
        maxlength=maxlength,
        binary_output=binary_output,
        axis=axis)
    self.assertAllEqual(expected_indices, y.indices)
    self.assertAllEqual(expected_values, y.values)
    self.assertAllEqual(expected_shape, y.dense_shape)


class TestSparseCountFailureModes(test.TestCase):

  def test_dense_input_sparse_weights_fails(self):
    x = np.array([[3, 2, 1], [5, 4, 4]], dtype=np.int32)
    weights = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    with self.assertRaisesRegexp(ValueError, "must be a tf.Tensor"):
      self.evaluate(bincount.sparse_bincount(x, weights=weights, axis=-1))

  def test_dense_input_ragged_weights_fails(self):
    x = np.array([[3, 2, 1], [5, 4, 4]], dtype=np.int32)
    weights = ragged_factory_ops.constant([[6, 0.5, 2], [14], [10, 0.25, 5, 3]])
    with self.assertRaisesRegexp(ValueError, "must be a tf.Tensor"):
      self.evaluate(bincount.sparse_bincount(x, weights=weights, axis=-1))

  def test_dense_input_wrong_shape_fails(self):
    x = np.array([[3, 2, 1], [5, 4, 4]], dtype=np.int32)
    weights = np.array([[3, 2], [5, 4], [4, 3]])
    # Note: Eager mode and graph mode throw different errors here. Graph mode
    # will fail with a ValueError from the shape checking logic, while Eager
    # will fail with an InvalidArgumentError from the kernel itself.
    if context.executing_eagerly():
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "must have the same shape"):
        self.evaluate(bincount.sparse_bincount(x, weights=weights, axis=-1))
    else:
      with self.assertRaisesRegexp(ValueError, "both shapes must be equal"):
        self.evaluate(bincount.sparse_bincount(x, weights=weights, axis=-1))

  def test_sparse_input_dense_weights_fails(self):
    x = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    weights = np.array([[3, 2, 1], [5, 4, 4]], dtype=np.int32)
    with self.assertRaisesRegexp(ValueError, "must be a SparseTensor"):
      self.evaluate(bincount.sparse_bincount(x, weights=weights, axis=-1))

  def test_sparse_input_ragged_weights_fails(self):
    x = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    weights = ragged_factory_ops.constant([[6, 0.5, 2], [14], [10, 0.25, 5, 3]])
    with self.assertRaisesRegexp(ValueError, "must be a SparseTensor"):
      self.evaluate(bincount.sparse_bincount(x, weights=weights, axis=-1))

  def test_sparse_input_wrong_indices_fails(self):
    x = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    weights = sparse_ops.from_dense(
        np.array([[3, 1, 0, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "must have the same indices"):
      self.evaluate(bincount.sparse_bincount(x, weights=weights, axis=-1))

  def test_sparse_input_too_many_indices_fails(self):
    x = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    weights = sparse_ops.from_dense(
        np.array([[3, 1, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "Incompatible shapes"):
      self.evaluate(bincount.sparse_bincount(x, weights=weights, axis=-1))

  def test_sparse_input_wrong_shape_fails(self):
    x = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    weights = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4], [0, 0, 0, 0]],
                 dtype=np.int32))
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "must have the same dense shape"):
      self.evaluate(bincount.sparse_bincount(x, weights=weights, axis=-1))

  def test_ragged_input_dense_weights_fails(self):
    x = ragged_factory_ops.constant([[6, 1, 2], [14], [10, 1, 5, 3]])
    weights = np.array([[3, 2, 1], [5, 4, 4]], dtype=np.int32)
    with self.assertRaisesRegexp(ValueError, "must be a RaggedTensor"):
      self.evaluate(bincount.sparse_bincount(x, weights=weights, axis=-1))

  def test_ragged_input_sparse_weights_fails(self):
    x = ragged_factory_ops.constant([[6, 1, 2], [14], [10, 1, 5, 3]])
    weights = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    with self.assertRaisesRegexp(ValueError, "must be a RaggedTensor"):
      self.evaluate(bincount.sparse_bincount(x, weights=weights, axis=-1))

  def test_ragged_input_different_shape_fails(self):
    x = ragged_factory_ops.constant([[6, 1, 2], [14], [10, 1, 5, 3]])
    weights = ragged_factory_ops.constant([[6, 0.5, 2], [], [10, 0.25, 5, 3]])
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "must have the same row splits"):
      self.evaluate(bincount.sparse_bincount(x, weights=weights, axis=-1))


if __name__ == "__main__":
  test.main()
