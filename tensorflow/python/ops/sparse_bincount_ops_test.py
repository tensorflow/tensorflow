# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for sparse bincount ops."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


def _sparse_factory(x):
  return lambda: sparse_ops.from_dense(x)


def _adjust_expected_rank1(x, minlength, maxlength):
  """Trim or pad an expected result based on minlength and maxlength."""
  n = len(x)
  if (minlength is not None) and (n < minlength):
    x = x + [0] * (minlength - n)
  if (maxlength is not None) and (n > maxlength):
    x = x[:maxlength]
  return x


def _adjust_expected_rank2(x, minlength, maxlength):
  return [_adjust_expected_rank1(i, minlength, maxlength) for i in x]


class TestSparseCount(test_util.TensorFlowTestCase, parameterized.TestCase):

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
    y = sparse_ops.sparse_bincount(
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
    y = sparse_ops.sparse_bincount(
        x_sparse,
        weights=w_sparse,
        minlength=minlength,
        maxlength=maxlength,
        binary_output=binary_output,
        axis=axis)
    self.assertAllEqual(expected_indices, y.indices)
    self.assertAllEqual(expected_values, y.values)
    self.assertAllEqual(expected_shape, y.dense_shape)

  @parameterized.product(
      (
          dict(
              tid="_s1",
              x_factory=_sparse_factory([1, 2, 2, 3, 3, 3]),
              expected=[0, 1, 2, 3],
          ),
          dict(
              tid="_s1_some_zeros",
              x_factory=_sparse_factory([1, 0, 0, 3, 3, 3]),
              expected=[0, 1, 0, 3],
          ),
          dict(
              tid="_s1_all_zeros",
              x_factory=_sparse_factory([0, 0, 0, 0, 0, 0]),
              expected=[],
          ),
          dict(
              tid="_s2",
              x_factory=_sparse_factory(
                  [[0, 0, 0], [0, 1, 0], [2, 0, 2], [3, 3, 3]]
              ),
              expected=[0, 1, 2, 3],
          ),
          dict(
              tid="_s3",
              x_factory=_sparse_factory(
                  [[[0, 0, 0], [0, 1, 0]], [[2, 0, 2], [3, 3, 3]]]
              ),
              expected=[0, 1, 2, 3],
          ),
      ),
      (
          dict(minlength=None, maxlength=None),
          dict(minlength=3, maxlength=None),
          dict(minlength=5, maxlength=None),
          dict(minlength=None, maxlength=3),
          dict(minlength=None, maxlength=5),
          dict(minlength=2, maxlength=3),
          dict(minlength=3, maxlength=5),
          dict(minlength=5, maxlength=10),
          dict(minlength=None, maxlength=0),
      ),
  )
  def test_default(
      self,
      x_factory,
      minlength,
      maxlength,
      expected,
      tid=None,
  ):
    x = x_factory()
    expected = _adjust_expected_rank1(expected, minlength, maxlength)
    self.assertAllEqual(
        expected,
        self.evaluate(
            sparse_ops.bincount(x, minlength=minlength, maxlength=maxlength)
        ),
    )
    self.assertAllEqual(
        expected,
        self.evaluate(
            sparse_ops.bincount(
                x, minlength=minlength, maxlength=maxlength, axis=0
            )
        ),
    )

  @parameterized.product(
      (
          dict(
              tid="_s2",
              x_factory=_sparse_factory(
                  [[0, 0, 0], [0, 1, 0], [2, 0, 2], [3, 3, 3]]
              ),
              expected=[[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3]],
          ),
      ),
      (
          dict(minlength=None, maxlength=None),
          dict(minlength=3, maxlength=None),
          dict(minlength=5, maxlength=None),
          dict(minlength=None, maxlength=3),
          dict(minlength=None, maxlength=5),
          dict(minlength=2, maxlength=3),
          dict(minlength=3, maxlength=5),
          dict(minlength=5, maxlength=10),
          dict(minlength=None, maxlength=0),
      ),
  )
  def test_axis_neg_one(
      self, tid, x_factory, minlength, maxlength, expected
  ):
    x = x_factory()
    expected = _adjust_expected_rank2(expected, minlength, maxlength)
    self.assertAllEqual(
        expected,
        self.evaluate(
            sparse_ops.bincount(
                x, minlength=minlength, maxlength=maxlength, axis=-1
            )
        ),
    )

  @parameterized.product(
      (
          dict(
              tid="_s1",
              x_factory=_sparse_factory([1, 2, 2, 3, 3, 3]),
              weights_factory=_sparse_factory([1, 2, 3, 4, 5, 6]),
              expected=[0, 1, 5, 15],
              axis=None,
          ),
          dict(
              tid="_s2",
              x_factory=_sparse_factory(
                  [[0, 0, 0], [0, 1, 0], [2, 0, 2], [3, 3, 3]]
              ),
              # weights have the same shape as x, so when x has an implicit
              # zero, the corresponding weight is as an implicit zero
              weights_factory=_sparse_factory(
                  [[0, 0, 0], [0, 1, 0], [2, 0, 3], [4, 5, 6]]
              ),
              axis=None,
              expected=[0, 1, 5, 15],
          ),
          dict(
              tid="_s3",
              x_factory=_sparse_factory(
                  [[[0, 0, 0], [0, 1, 0]], [[2, 0, 2], [3, 3, 3]]]
              ),
              # weights have the same shape as x, so when x has an implicit
              # zero, the corresponding weight is as an implicit zero
              weights_factory=_sparse_factory(
                  [[[0, 0, 0], [0, 1, 0]], [[2, 0, 3], [4, 5, 6]]]
              ),
              axis=None,
              expected=[0, 1, 5, 15],
          ),
          dict(
              tid="_s2_axis_neg_1",
              x_factory=_sparse_factory(
                  [[0, 0, 0], [0, 1, 0], [2, 0, 2], [3, 3, 3]]
              ),
              # weights have the same shape as x, so when x has an implicit
              # zero, the corresponding weight is as an implicit zero
              weights_factory=_sparse_factory(
                  [[0, 0, 0], [0, 1, 0], [2, 0, 3], [4, 5, 6]]
              ),
              expected=[
                  [0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 5, 0],
                  [0, 0, 0, 15],
              ],
              axis=-1,
          ),
      ),
      (
          dict(minlength=None, maxlength=None),
          dict(minlength=3, maxlength=None),
          dict(minlength=5, maxlength=None),
          dict(minlength=None, maxlength=3),
          dict(minlength=None, maxlength=5),
          dict(minlength=2, maxlength=3),
          dict(minlength=3, maxlength=5),
          dict(minlength=5, maxlength=10),
          dict(minlength=None, maxlength=0),
      ),
  )
  def test_weights(
      self,
      tid,
      x_factory,
      weights_factory,
      minlength,
      maxlength,
      expected,
      axis,
  ):
    device_set = set([d.device_type for d in tf_config.list_physical_devices()])
    if "GPU" in device_set and not test_util.is_xla_enabled():
      self.skipTest(
          "b/263004039 The DenseBincount GPU kernel does not support weights."
          " unsorted_segment_sum should be used instead on GPU."
      )
    x = x_factory()
    weights = weights_factory()
    if axis == -1:
      expected = _adjust_expected_rank2(expected, minlength, maxlength)
    else:
      expected = _adjust_expected_rank1(expected, minlength, maxlength)
    self.assertAllEqual(
        expected,
        self.evaluate(
            sparse_ops.bincount(
                x,
                weights=weights,
                minlength=minlength,
                maxlength=maxlength,
                axis=axis,
            )
        ),
    )

  @parameterized.product(
      (
          dict(
              tid="_s1",
              x_factory=_sparse_factory([1, 2, 2, 3, 3, 3]),
              expected=[0, 1, 1, 1],
              axis=None,
          ),
          dict(
              tid="_s1_zeros",
              x_factory=_sparse_factory([1, 0, 0, 3, 3, 3]),
              expected=[0, 1, 0, 1],
              axis=None,
          ),
          dict(
              tid="_s2",
              x_factory=_sparse_factory(
                  [[0, 0, 0], [0, 1, 0], [2, 0, 2], [3, 3, 3]]
              ),
              expected=[0, 1, 1, 1],
              axis=None,
          ),
          dict(
              tid="_s3",
              x_factory=_sparse_factory(
                  [[[0, 0, 0], [0, 1, 0]], [[2, 0, 2], [3, 3, 3]]]
              ),
              expected=[0, 1, 1, 1],
              axis=None,
          ),
          dict(
              tid="_s2_axis_neg_1",
              x_factory=_sparse_factory(
                  [[0, 0, 0], [0, 1, 0], [2, 0, 2], [3, 3, 3]]
              ),
              expected=[[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
              axis=-1,
          ),
      ),
      (
          dict(minlength=None, maxlength=None),
          dict(minlength=3, maxlength=None),
          dict(minlength=5, maxlength=None),
          dict(minlength=None, maxlength=3),
          dict(minlength=None, maxlength=5),
          dict(minlength=2, maxlength=3),
          dict(minlength=3, maxlength=5),
          dict(minlength=5, maxlength=10),
          dict(minlength=None, maxlength=0),
      ),
  )
  def test_binary_output(
      self,
      tid,
      x_factory,
      minlength,
      maxlength,
      expected,
      axis=None,
  ):
    x = x_factory()
    if axis == -1:
      expected = _adjust_expected_rank2(expected, minlength, maxlength)
    else:
      expected = _adjust_expected_rank1(expected, minlength, maxlength)
    self.assertAllEqual(
        expected,
        self.evaluate(
            sparse_ops.bincount(
                x,
                minlength=minlength,
                maxlength=maxlength,
                binary_output=True,
                axis=axis,
            )
        ),
    )


class TestSparseCountFailureModes(test_util.TensorFlowTestCase):

  def test_dense_input_sparse_weights_fails(self):
    x = np.array([[3, 2, 1], [5, 4, 4]], dtype=np.int32)
    weights = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    with self.assertRaisesRegex(ValueError, "must be a tf.Tensor"):
      self.evaluate(sparse_ops.sparse_bincount(x, weights=weights, axis=-1))

  def test_dense_input_wrong_shape_fails(self):
    x = np.array([[3, 2, 1], [5, 4, 4]], dtype=np.int32)
    weights = np.array([[3, 2], [5, 4], [4, 3]])
    # Note: Eager mode and graph mode throw different errors here. Graph mode
    # will fail with a ValueError from the shape checking logic, while Eager
    # will fail with an InvalidArgumentError from the kernel itself.
    if context.executing_eagerly():
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "must have the same shape"):
        self.evaluate(sparse_ops.sparse_bincount(x, weights=weights, axis=-1))
    else:
      with self.assertRaisesRegex(ValueError, "both shapes must be equal"):
        self.evaluate(sparse_ops.sparse_bincount(x, weights=weights, axis=-1))

  def test_sparse_input_dense_weights_fails(self):
    x = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    weights = np.array([[3, 2, 1], [5, 4, 4]], dtype=np.int32)
    with self.assertRaisesRegex(ValueError, "must be a SparseTensor"):
      self.evaluate(sparse_ops.sparse_bincount(x, weights=weights, axis=-1))

  def test_sparse_input_wrong_indices_fails(self):
    x = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    weights = sparse_ops.from_dense(
        np.array([[3, 1, 0, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "must have the same indices"):
      self.evaluate(sparse_ops.sparse_bincount(x, weights=weights, axis=-1))

  def test_sparse_input_too_many_indices_fails(self):
    x = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    weights = sparse_ops.from_dense(
        np.array([[3, 1, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    with self.assertRaisesIncompatibleShapesError():
      self.evaluate(sparse_ops.sparse_bincount(x, weights=weights, axis=-1))

  def test_sparse_input_wrong_shape_fails(self):
    x = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    weights = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4], [0, 0, 0, 0]],
                 dtype=np.int32))
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "must have the same dense shape"):
      self.evaluate(sparse_ops.sparse_bincount(x, weights=weights, axis=-1))


if __name__ == "__main__":
  test.main()
