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
"""Tests for third_party.tensorflow.python.ops.ragged_tensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_grad  # pylint: disable=unused-import
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensorSpec
from tensorflow.python.ops.ragged.row_partition import RowPartition

from tensorflow.python.platform import googletest
from tensorflow.python.util import nest


def int32array(values):
  return np.array(values, dtype=np.int32)


@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  longMessage = True  # Property in unittest.Testcase. pylint: disable=invalid-name

  #=============================================================================
  # RaggedTensor class docstring examples
  #=============================================================================

  def testClassDocStringExamples(self):
    # From section: "Component Tensors"
    rt = RaggedTensor.from_row_splits(
        values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
    self.assertAllEqual(rt, [[3, 1, 4, 1], [], [5, 9, 2], [6], []])
    del rt

    # From section: "Alternative Row-Partitioning Schemes"
    values = [3, 1, 4, 1, 5, 9, 2, 6]
    rt1 = RaggedTensor.from_row_splits(values, row_splits=[0, 4, 4, 7, 8, 8])
    rt2 = RaggedTensor.from_row_lengths(values, row_lengths=[4, 0, 3, 1, 0])
    rt3 = RaggedTensor.from_value_rowids(
        values, value_rowids=[0, 0, 0, 0, 2, 2, 2, 3], nrows=5)
    rt4 = RaggedTensor.from_row_starts(values, row_starts=[0, 4, 4, 7, 8])
    rt5 = RaggedTensor.from_row_limits(values, row_limits=[4, 4, 7, 8, 8])
    for rt in (rt1, rt2, rt3, rt4, rt5):
      self.assertAllEqual(rt, [[3, 1, 4, 1], [], [5, 9, 2], [6], []])
    del rt1, rt2, rt3, rt4, rt5

    # From section: "Multiple Ragged Dimensions"
    inner_rt = RaggedTensor.from_row_splits(
        values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
    outer_rt = RaggedTensor.from_row_splits(
        values=inner_rt, row_splits=[0, 3, 3, 5])
    self.assertEqual(outer_rt.ragged_rank, 2)
    self.assertAllEqual(outer_rt,
                        [[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]])
    del inner_rt, outer_rt

    # From section: "Multiple Ragged Dimensions"
    rt = RaggedTensor.from_nested_row_splits(
        flat_values=[3, 1, 4, 1, 5, 9, 2, 6],
        nested_row_splits=([0, 3, 3, 5], [0, 4, 4, 7, 8, 8]))
    self.assertAllEqual(rt, [[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]])
    del rt

    # From section: "Uniform Inner Dimensions"
    rt = RaggedTensor.from_row_splits(
        values=array_ops.ones([5, 3]), row_splits=[0, 2, 5])
    self.assertAllEqual(
        rt, [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
    self.assertEqual(rt.shape.as_list(), [2, None, 3])
    del rt

  #=============================================================================
  # RaggedTensorValue Constructor
  #=============================================================================

  def testRaggedTensorValueConstruction(self):
    values = np.array(b'a b c d e f g'.split())
    splits = np.array([0, 2, 5, 6, 6, 7], dtype=np.int64)
    splits2 = np.array([0, 3, 5], dtype=np.int64)

    # Test construction of a RaggedTensorValue with ragged_rank=1.
    rt_value = ragged_tensor_value.RaggedTensorValue(values, splits)
    self.assertEqual(rt_value.row_splits.dtype, np.int64)
    self.assertEqual(rt_value.shape, (5, None))
    self.assertLen(rt_value.nested_row_splits, 1)
    self.assertAllEqual(splits, rt_value.row_splits)
    self.assertAllEqual(values, rt_value.values)
    self.assertAllEqual(splits, rt_value.nested_row_splits[0])
    self.assertAllEqual(values, rt_value.flat_values)

    # Test construction of a RaggedTensorValue with ragged_rank=2.
    rt_value = ragged_tensor_value.RaggedTensorValue(
        values=ragged_tensor_value.RaggedTensorValue(values, splits),
        row_splits=splits2)
    self.assertEqual(rt_value.row_splits.dtype, np.int64)
    self.assertEqual(rt_value.shape, (2, None, None))
    self.assertLen(rt_value.nested_row_splits, 2)
    self.assertAllEqual(splits2, rt_value.row_splits)
    self.assertAllEqual(splits, rt_value.values.row_splits)
    self.assertAllEqual(splits2, rt_value.nested_row_splits[0])
    self.assertAllEqual(splits, rt_value.nested_row_splits[1])
    self.assertAllEqual(values, rt_value.values.values)
    self.assertAllEqual(values, rt_value.flat_values)

  #=============================================================================
  # RaggedTensor Constructor (private)
  #=============================================================================

  def testRaggedTensorConstruction(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    rp = RowPartition.from_row_splits(row_splits)
    rt = RaggedTensor(values=values, row_partition=rp, internal=True)

    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testRaggedTensorConstructionErrors(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    rp = RowPartition.from_row_splits(row_splits)

    with self.assertRaisesRegex(ValueError,
                                'RaggedTensor constructor is private'):
      RaggedTensor(values=values, row_partition=rp)

    with self.assertRaisesRegex(
        TypeError,
        r"""type\(values\) must be one of: 'Tensor, RaggedTensor.*"""):
      RaggedTensor(values=range(7), row_partition=rp, internal=True)

    with self.assertRaisesRegex(TypeError,
                                'row_partition must be a RowPartition'):
      RaggedTensor(
          values=values, row_partition=[0, 2, 2, 5, 6, 7], internal=True)

  #=============================================================================
  # RaggedTensor Factory Ops
  #=============================================================================

  def testFromValueRowIdsWithDerivedNRows(self):
    # nrows is known at graph creation time.
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)

    rt = RaggedTensor.from_value_rowids(values, value_rowids, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_value_rowids = rt.value_rowids()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertIs(rt_value_rowids, value_rowids)  # cached_value_rowids
    self.assertAllEqual(rt_value_rowids, value_rowids)
    self.assertAllEqual(rt_nrows, 5)
    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromValueRowIdsWithDerivedNRowsDynamic(self):
    # nrows is not known at graph creation time.
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    value_rowids = array_ops.placeholder_with_default(value_rowids, shape=None)

    rt = RaggedTensor.from_value_rowids(values, value_rowids, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    if context.executing_eagerly():
      self.assertEqual(rt.shape.as_list(), [5, None])
    else:
      self.assertEqual(rt.shape.as_list(), [None, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_value_rowids = rt.value_rowids()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertIs(rt_value_rowids, value_rowids)  # cached_value_rowids
    self.assertAllEqual(rt_value_rowids, value_rowids)
    self.assertAllEqual(rt_nrows, 5)
    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromValueRowIdsWithExplicitNRows(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(7, dtypes.int64)

    rt = RaggedTensor.from_value_rowids(
        values, value_rowids, nrows, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [7, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_value_rowids = rt.value_rowids()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertIs(rt_value_rowids, value_rowids)  # cached_value_rowids
    self.assertIs(rt_nrows, nrows)  # cached_nrows
    self.assertAllEqual(
        rt, [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g'], [], []])

  def testFromValueRowIdsWithExplicitNRowsEqualToDefault(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(5, dtypes.int64)

    rt = RaggedTensor.from_value_rowids(
        values, value_rowids, nrows, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_value_rowids = rt.value_rowids()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertIs(rt_value_rowids, value_rowids)  # cached_value_rowids
    self.assertIs(rt_nrows, nrows)  # cached_nrows
    self.assertAllEqual(rt_value_rowids, value_rowids)
    self.assertAllEqual(rt_nrows, nrows)
    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromValueRowIdsWithEmptyValues(self):
    rt = RaggedTensor.from_value_rowids([], [])
    rt_nrows = rt.nrows()
    self.assertEqual(rt.dtype, dtypes.float32)
    self.assertEqual(rt.shape.as_list(), [0, None])
    self.assertEqual(rt.ragged_rank, 1)
    self.assertEqual(rt.values.shape.as_list(), [0])
    self.assertEqual(rt.value_rowids().shape.as_list(), [0])
    self.assertAllEqual(rt_nrows, 0)
    self.assertAllEqual(rt, [])

  def testFromRowSplits(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)

    rt = RaggedTensor.from_row_splits(values, row_splits, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_row_splits = rt.row_splits
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertIs(rt_row_splits, row_splits)
    self.assertAllEqual(rt_nrows, 5)
    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromRowSplitsWithDifferentSplitTypes(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    splits1 = [0, 2, 2, 5, 6, 7]
    splits2 = np.array([0, 2, 2, 5, 6, 7], np.int64)
    splits3 = np.array([0, 2, 2, 5, 6, 7], np.int32)
    splits4 = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    splits5 = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int32)
    rt1 = RaggedTensor.from_row_splits(values, splits1)
    rt2 = RaggedTensor.from_row_splits(values, splits2)
    rt3 = RaggedTensor.from_row_splits(values, splits3)
    rt4 = RaggedTensor.from_row_splits(values, splits4)
    rt5 = RaggedTensor.from_row_splits(values, splits5)
    self.assertEqual(rt1.row_splits.dtype, dtypes.int64)
    self.assertEqual(rt2.row_splits.dtype, dtypes.int64)
    self.assertEqual(rt3.row_splits.dtype, dtypes.int32)
    self.assertEqual(rt4.row_splits.dtype, dtypes.int64)
    self.assertEqual(rt5.row_splits.dtype, dtypes.int32)

  def testFromRowSplitsWithEmptySplits(self):
    err_msg = 'row_splits tensor may not be empty'
    with self.assertRaisesRegex(ValueError, err_msg):
      RaggedTensor.from_row_splits([], [])

  def testFromRowStarts(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_starts = constant_op.constant([0, 2, 2, 5, 6], dtypes.int64)

    rt = RaggedTensor.from_row_starts(values, row_starts, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_row_starts = rt.row_starts()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertAllEqual(rt_nrows, 5)
    self.assertAllEqual(rt_row_starts, row_starts)
    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromRowLimits(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_limits = constant_op.constant([2, 2, 5, 6, 7], dtypes.int64)

    rt = RaggedTensor.from_row_limits(values, row_limits, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_row_limits = rt.row_limits()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertAllEqual(rt_nrows, 5)
    self.assertAllEqual(rt_row_limits, row_limits)
    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromRowLengths(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_lengths = constant_op.constant([2, 0, 3, 1, 1], dtypes.int64)

    rt = RaggedTensor.from_row_lengths(values, row_lengths, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_row_lengths = rt.row_lengths()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertIs(rt_row_lengths, row_lengths)  # cached_nrows
    self.assertAllEqual(rt_nrows, 5)
    self.assertAllEqual(rt_row_lengths, row_lengths)
    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromRowLengthsInt32(self):
    rt = RaggedTensor.from_row_lengths([1, 2, 3, 4],
                                       constant_op.constant([1, 0, 3],
                                                            dtype=dtypes.int32))
    rt2 = RaggedTensor.from_row_lengths(rt, [2, 1, 0])
    self.assertAllEqual([2, 1, 0], rt2.row_lengths())

  def testFromUniformRowLength(self):
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    a1 = RaggedTensor.from_uniform_row_length(values, 2)
    a2 = RaggedTensor.from_uniform_row_length(values, 2, 8)
    self.assertAllEqual(
        a1,
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
    self.assertAllEqual(a1, a2)
    self.assertEqual(a1.shape.as_list(), [8, 2])
    self.assertEqual(a2.shape.as_list(), [8, 2])

    b1 = RaggedTensor.from_uniform_row_length(a1, 2)
    b2 = RaggedTensor.from_uniform_row_length(a1, 2, 4)
    self.assertAllEqual(b1, [[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                             [[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    self.assertAllEqual(b1, b2)
    self.assertEqual(b1.shape.as_list(), [4, 2, 2])
    self.assertEqual(b2.shape.as_list(), [4, 2, 2])

    c1 = RaggedTensor.from_uniform_row_length(b1, 2)
    c2 = RaggedTensor.from_uniform_row_length(b1, 2, 2)
    self.assertAllEqual(c1, [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                             [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
    self.assertAllEqual(c1, c2)
    self.assertEqual(c1.shape.as_list(), [2, 2, 2, 2])
    self.assertEqual(c2.shape.as_list(), [2, 2, 2, 2])

  def testFromUniformRowLengthWithEmptyValues(self):
    empty_values = []
    a = RaggedTensor.from_uniform_row_length(empty_values, 0, nrows=10)
    self.assertEqual(a.shape.as_list(), [10, 0])

    b = RaggedTensor.from_uniform_row_length(a, 2)
    self.assertEqual(b.shape.as_list(), [5, 2, 0])

    # Make sure we avoid divide-by-zero when finding nrows for nvals=rowlen=0.
    c = RaggedTensor.from_uniform_row_length(empty_values, 0)
    self.assertEqual(c.shape.as_list(), [0, 0])
    d = RaggedTensor.from_uniform_row_length(empty_values, 0, nrows=0)
    self.assertEqual(d.shape.as_list(), [0, 0])

  def testFromUniformRowLengthWithPlaceholders(self):
    ph_values = array_ops.placeholder_with_default([1, 2, 3, 4, 5, 6], [None])
    ph_rowlen = array_ops.placeholder_with_default(3, None)
    rt1 = RaggedTensor.from_uniform_row_length(ph_values, 3)
    rt2 = RaggedTensor.from_uniform_row_length(ph_values, ph_rowlen)
    rt3 = RaggedTensor.from_uniform_row_length([1, 2, 3, 4, 5, 6], ph_rowlen)
    self.assertAllEqual(rt1, [[1, 2, 3], [4, 5, 6]])
    self.assertAllEqual(rt2, [[1, 2, 3], [4, 5, 6]])
    self.assertAllEqual(rt3, [[1, 2, 3], [4, 5, 6]])
    if context.executing_eagerly():
      self.assertEqual(rt1.shape.as_list(), [2, 3])
      self.assertEqual(rt2.shape.as_list(), [2, 3])
      self.assertEqual(rt3.shape.as_list(), [2, 3])
    else:
      self.assertEqual(rt1.shape.as_list(), [None, 3])
      self.assertEqual(rt2.shape.as_list(), [None, None])
      self.assertEqual(rt3.shape.as_list(), [None, None])

    b = RaggedTensor.from_uniform_row_length(rt1, 2)
    self.assertAllEqual(b, [[[1, 2, 3], [4, 5, 6]]])

    # Make sure we avoid divide-by-zero when finding nrows for nvals=rowlen=0.
    ph_empty_values = array_ops.placeholder_with_default(
        array_ops.zeros([0], dtypes.int64), [None])
    ph_zero = array_ops.placeholder_with_default(0, [])
    c = RaggedTensor.from_uniform_row_length(ph_empty_values, ph_zero)
    if context.executing_eagerly():
      self.assertEqual(c.shape.as_list(), [0, 0])
    else:
      self.assertEqual(c.shape.as_list(), [None, None])

  def testFromNestedValueRowIdsWithDerivedNRows(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_value_rowids = [
        constant_op.constant([0, 0, 1, 3, 3], dtypes.int64),
        constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    ]

    rt = RaggedTensor.from_nested_value_rowids(values, nested_value_rowids)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [4, None, None])
    self.assertEqual(rt.ragged_rank, 2)

    rt_values = rt.values
    rt_value_rowids = rt.value_rowids()
    rt_values_values = rt_values.values
    rt_values_value_rowids = rt_values.value_rowids()

    self.assertIs(rt_values_values, values)
    self.assertAllEqual(rt_value_rowids, nested_value_rowids[0])
    self.assertAllEqual(rt_values_value_rowids, nested_value_rowids[1])
    self.assertAllEqual(
        rt, [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [], [[b'f'], [b'g']]])

  def testFromNestedRowPartitions(self):
    flat_values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_row_splits = [[0, 2, 3, 3, 5], [0, 2, 2, 5, 6, 7]]
    nested_row_partition = [
        RowPartition.from_row_splits(constant_op.constant(x, dtypes.int64))
        for x in nested_row_splits
    ]

    rt = RaggedTensor._from_nested_row_partitions(
        flat_values, nested_row_partition, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [4, None, None])
    self.assertEqual(rt.ragged_rank, 2)
    self.assertAllEqual(
        rt, [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [], [[b'f'], [b'g']]])

  def testFromNestedValueRowIdsWithExplicitNRows(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_value_rowids = [
        constant_op.constant([0, 0, 1, 3, 3, 3], dtypes.int64),
        constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    ]
    nrows = [
        constant_op.constant(6, dtypes.int64),
        constant_op.constant(6, dtypes.int64)
    ]

    rt = RaggedTensor.from_nested_value_rowids(values, nested_value_rowids,
                                               nrows)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [6, None, None])
    self.assertEqual(rt.ragged_rank, 2)

    rt_values = rt.values
    rt_value_rowids = rt.value_rowids()
    rt_nrows = rt.nrows()
    rt_values_values = rt_values.values
    rt_values_value_rowids = rt_values.value_rowids()
    rt_values_nrows = rt_values.nrows()

    self.assertIs(rt_values_values, values)
    self.assertAllEqual(rt_value_rowids, nested_value_rowids[0])
    self.assertAllEqual(rt_values_value_rowids, nested_value_rowids[1])
    self.assertAllEqual(rt_nrows, nrows[0])
    self.assertAllEqual(rt_values_nrows, nrows[1])
    self.assertAllEqual(rt, [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [],
                             [[b'f'], [b'g'], []], [], []])

  def testFromNestedValueRowIdsWithExplicitNRowsMismatch(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_value_rowids = [
        constant_op.constant([0, 0, 1, 3, 3, 3], dtypes.int64),
        constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    ]
    nrows = [constant_op.constant(6, dtypes.int64)]
    with self.assertRaisesRegex(
        ValueError, 'nested_nrows must have the same '
        'length as nested_value_rowids'):
      RaggedTensor.from_nested_value_rowids(values, nested_value_rowids, nrows)

  def testFromNestedValueRowIdsWithNonListInput(self):
    with self.assertRaisesRegex(
        TypeError, 'nested_value_rowids must be a list of Tensors'):
      RaggedTensor.from_nested_value_rowids(
          [1, 2, 3], constant_op.constant([[0, 1, 2], [0, 1, 2]], dtypes.int64))
    with self.assertRaisesRegex(TypeError,
                                'nested_nrows must be a list of Tensors'):
      RaggedTensor.from_nested_value_rowids([1, 2, 3], [[0, 1, 2], [0, 1, 2]],
                                            constant_op.constant([3, 3]))

  def testFromNestedRowSplits(self):
    flat_values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_row_splits = [
        constant_op.constant([0, 2, 3, 3, 5], dtypes.int64),
        constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    ]

    rt = RaggedTensor.from_nested_row_splits(
        flat_values, nested_row_splits, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [4, None, None])
    self.assertEqual(rt.ragged_rank, 2)

    rt_values = rt.values
    rt_row_splits = rt.row_splits
    rt_values_values = rt_values.values
    rt_values_row_splits = rt_values.row_splits

    self.assertIs(rt_values_values, flat_values)
    self.assertIs(rt_row_splits, nested_row_splits[0])
    self.assertIs(rt_values_row_splits, nested_row_splits[1])
    self.assertAllEqual(
        rt, [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [], [[b'f'], [b'g']]])

  def testWithRowSplits(self):
    flat_values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_row_splits = [
        constant_op.constant([0, 2, 3, 3, 5], dtypes.int64),
        constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    ]

    rt = RaggedTensor.from_nested_row_splits(
        flat_values, nested_row_splits, validate=False)

    rt = rt.with_row_splits_dtype(dtypes.int32)

    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [4, None, None])
    self.assertEqual(rt.ragged_rank, 2)

    rt_values = rt.values
    rt_row_splits = rt.row_splits
    rt_values_values = rt_values.values
    rt_values_row_splits = rt_values.row_splits

    self.assertAllEqual(rt_values_values, flat_values)
    self.assertAllEqual(rt_row_splits, nested_row_splits[0])
    self.assertAllEqual(rt_values_row_splits, nested_row_splits[1])
    self.assertAllEqual(
        rt, [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [], [[b'f'], [b'g']]])

  def testFromNestedRowSplitsWithNonListInput(self):
    with self.assertRaisesRegex(TypeError,
                                'nested_row_splits must be a list of Tensors'):
      RaggedTensor.from_nested_row_splits(
          [1, 2], constant_op.constant([[0, 1, 2], [0, 1, 2]], dtypes.int64))

  def testFromValueRowIdsWithBadNRows(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(5, dtypes.int64)

    with self.assertRaisesRegex(ValueError, r'Expected nrows >= 0; got -2'):
      RaggedTensor.from_value_rowids(
          values=values,
          value_rowids=array_ops.placeholder_with_default(value_rowids, None),
          nrows=-2)

    with self.assertRaisesRegex(
        ValueError, r'Expected nrows >= value_rowids\[-1\] \+ 1; got nrows=2, '
        r'value_rowids\[-1\]=4'):
      RaggedTensor.from_value_rowids(
          values=values, value_rowids=value_rowids, nrows=2)

    with self.assertRaisesRegex(
        ValueError, r'Expected nrows >= value_rowids\[-1\] \+ 1; got nrows=4, '
        r'value_rowids\[-1\]=4'):
      RaggedTensor.from_value_rowids(
          values=values, value_rowids=value_rowids, nrows=4)

    with self.assertRaisesRegex(ValueError, r'Shape \(7, 1\) must have rank 1'):
      RaggedTensor.from_value_rowids(
          values=values,
          value_rowids=array_ops.expand_dims(value_rowids, 1),
          nrows=nrows)

    with self.assertRaisesRegex(ValueError, r'Shape \(1,\) must have rank 0'):
      RaggedTensor.from_value_rowids(
          values=values,
          value_rowids=value_rowids,
          nrows=array_ops.expand_dims(nrows, 0))

  def testCondWithTensorsFromValueIds(self):
    # b/141166460
    rt = RaggedTensor.from_value_rowids([1, 2, 3], [0, 0, 2])
    c = array_ops.placeholder_with_default(True, None)
    result = control_flow_ops.cond(c, lambda: rt, lambda: rt)
    self.assertAllEqual(rt, result)

  def testGraphMismatch(self):
    if not context.executing_eagerly():
      with ops.Graph().as_default():
        values = constant_op.constant([1, 2, 3], dtypes.int64)
      with ops.Graph().as_default():
        splits = constant_op.constant([0, 2, 3], dtypes.int64)
      with self.assertRaisesRegex(ValueError,
                                  '.* must be from the same graph as .*'):
        RaggedTensor.from_row_splits(values, splits)

  #=============================================================================
  # Ragged Value & Row-Partitioning Tensor Accessors
  #=============================================================================

  def testRaggedTensorAccessors_2d(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    rt1 = RaggedTensor.from_row_splits(values, row_splits)
    rt2 = RaggedTensor.from_value_rowids(values, value_rowids)

    for rt in [rt1, rt2]:
      self.assertAllEqual(
          rt, [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])
      self.assertAllEqual(rt.values, [b'a', b'b', b'c', b'd', b'e', b'f', b'g'])
      self.assertEqual(rt.values.shape.dims[0].value, 7)
      self.assertAllEqual(rt.value_rowids(), [0, 0, 2, 2, 2, 3, 4])
      self.assertAllEqual(rt.nrows(), 5)
      self.assertAllEqual(rt.row_splits, [0, 2, 2, 5, 6, 7])
      self.assertAllEqual(rt.row_starts(), [0, 2, 2, 5, 6])
      self.assertAllEqual(rt.row_limits(), [2, 2, 5, 6, 7])
      self.assertAllEqual(rt.row_lengths(), [2, 0, 3, 1, 1])
      self.assertAllEqual(rt.flat_values,
                          [b'a', b'b', b'c', b'd', b'e', b'f', b'g'])
      self.assertLen(rt.nested_row_splits, 1)
      self.assertAllEqual(rt.nested_row_splits[0], [0, 2, 2, 5, 6, 7])

  def testRaggedTensorAccessors_3d_with_ragged_rank_1(self):
    values = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    row_lengths = constant_op.constant([2, 0, 3, 1, 1])
    rt1 = RaggedTensor.from_row_splits(values, row_splits)
    rt2 = RaggedTensor.from_value_rowids(values, value_rowids)
    rt3 = RaggedTensor.from_row_lengths(values, row_lengths)

    for rt in [rt1, rt2, rt3]:
      self.assertAllEqual(rt, [[[0, 1], [2, 3]], [], [[4, 5], [6, 7], [8, 9]],
                               [[10, 11]], [[12, 13]]])
      self.assertAllEqual(
          rt.values,
          [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]])
      self.assertEqual(rt.values.shape.dims[0].value, 7)
      self.assertAllEqual(rt.value_rowids(), [0, 0, 2, 2, 2, 3, 4])
      self.assertAllEqual(rt.nrows(), 5)
      self.assertAllEqual(rt.row_splits, [0, 2, 2, 5, 6, 7])
      self.assertAllEqual(rt.row_starts(), [0, 2, 2, 5, 6])
      self.assertAllEqual(rt.row_limits(), [2, 2, 5, 6, 7])
      self.assertAllEqual(rt.row_lengths(), [2, 0, 3, 1, 1])
      self.assertAllEqual(
          rt.row_lengths(axis=2), [[2, 2], [], [2, 2, 2], [2], [2]])
      self.assertAllEqual(
          rt.flat_values,
          [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]])
      self.assertLen(rt.nested_row_splits, 1)
      self.assertAllEqual(rt.nested_row_splits[0], [0, 2, 2, 5, 6, 7])
      self.assertLen(rt.nested_value_rowids(), 1)

      self.assertAllEqual(rt.nested_value_rowids()[0], [0, 0, 2, 2, 2, 3, 4])

  def testRaggedTensorAccessors_3d_with_ragged_rank_2(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_row_splits = [
        constant_op.constant([0, 2, 3, 3, 5], dtypes.int64),
        constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    ]
    nested_value_rowids = [
        constant_op.constant([0, 0, 1, 3, 3], dtypes.int64),
        constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    ]
    rt1 = RaggedTensor.from_nested_row_splits(values, nested_row_splits)
    rt2 = RaggedTensor.from_nested_value_rowids(values, nested_value_rowids)

    for rt in [rt1, rt2]:
      self.assertAllEqual(
          rt, [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [], [[b'f'], [b'g']]])
      self.assertAllEqual(
          rt.values, [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])
      self.assertEqual(rt.values.shape.dims[0].value, 5)
      self.assertAllEqual(rt.value_rowids(), [0, 0, 1, 3, 3])
      self.assertAllEqual(rt.nrows(), 4)
      self.assertAllEqual(rt.row_splits, [0, 2, 3, 3, 5])
      self.assertAllEqual(rt.row_starts(), [0, 2, 3, 3])
      self.assertAllEqual(rt.row_limits(), [2, 3, 3, 5])
      self.assertAllEqual(rt.row_lengths(), [2, 1, 0, 2])
      self.assertAllEqual(rt.flat_values,
                          [b'a', b'b', b'c', b'd', b'e', b'f', b'g'])
      self.assertLen(rt.nested_row_splits, 2)
      self.assertAllEqual(rt.nested_row_splits[0], [0, 2, 3, 3, 5])
      self.assertAllEqual(rt.nested_row_splits[1], [0, 2, 2, 5, 6, 7])
      self.assertLen(rt.nested_value_rowids(), 2)
      self.assertAllEqual(rt.nested_value_rowids()[0], [0, 0, 1, 3, 3])
      self.assertAllEqual(rt.nested_value_rowids()[1], [0, 0, 2, 2, 2, 3, 4])

  #=============================================================================
  # RaggedTensor.shape
  #=============================================================================

  def testShape(self):
    """Tests for RaggedTensor.shape."""
    rt1 = RaggedTensor.from_row_splits(b'a b c d e f g'.split(),
                                       [0, 2, 5, 6, 6, 7])
    self.assertEqual(rt1.shape.as_list(), [5, None])

    rt2 = RaggedTensor.from_row_splits(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]],
        [0, 2, 5, 6, 6, 7])
    self.assertEqual(rt2.shape.as_list(), [5, None, 2])

    rt3 = RaggedTensor.from_row_splits(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], [0, 2, 2, 3])
    self.assertEqual(rt3.shape.as_list(), [3, None, 2, 2])

    rt4 = RaggedTensor.from_row_splits(rt3, [0, 1, 3, 3])
    self.assertEqual(rt4.shape.as_list(), [3, None, None, 2, 2])

    if not context.executing_eagerly():
      rt5 = RaggedTensor.from_row_splits(
          array_ops.placeholder(dtype=dtypes.string), [0, 2, 3, 5])
      self.assertIsNone(rt5.shape.ndims)

      rt6 = RaggedTensor.from_row_splits(
          [1, 2, 3], array_ops.placeholder(dtype=dtypes.int64))
      self.assertEqual(rt6.shape.as_list(), [None, None])

  def testGetShape(self):
    rt = RaggedTensor.from_row_splits(b'a b c d e f g'.split(),
                                      [0, 2, 5, 6, 6, 7])
    self.assertEqual(rt.shape.as_list(), rt.get_shape().as_list())

  #=============================================================================
  # RaggedTensor.__str__
  #=============================================================================
  def testRaggedTensorStr(self):
    values = [b'a', b'b', b'c', b'd', b'e', b'f', b'g']
    row_splits = [0, 2, 5, 6, 6, 7]
    rt = RaggedTensor.from_row_splits(values, row_splits, validate=False)
    splits_type = 'int64'
    if context.executing_eagerly():
      expected_repr = '<tf.RaggedTensor {}>'.format([[b'a', b'b'],
                                                     [b'c', b'd', b'e'], [b'f'],
                                                     [], [b'g']])
    else:
      expected_repr = (
          'tf.RaggedTensor(values=Tensor("RaggedFromRowSplits/values:0", '
          'shape=(7,), dtype=string), '
          'row_splits=Tensor('
          '"RaggedFromRowSplits/RowPartitionFromRowSplits/row_splits:0",'
          ' shape=(6,), dtype={}))').format(splits_type)
    self.assertEqual(repr(rt), expected_repr)
    self.assertEqual(str(rt), expected_repr)

  def testRaggedTensorValueStr(self):
    values = [b'a', b'b', b'c', b'd', b'e', b'f', b'g']
    row_splits = [0, 2, 5, 6, 6, 7]
    rt = ragged_tensor_value.RaggedTensorValue(
        np.array(values), np.array(row_splits, dtype=np.int64))
    expected_str = '<tf.RaggedTensorValue {}>'.format([[b'a', b'b'],
                                                       [b'c', b'd', b'e'],
                                                       [b'f'], [], [b'g']])
    expected_repr = ("tf.RaggedTensorValue(values=array({}, dtype='|S1'), "
                     'row_splits=array({}))'.format(values, row_splits))
    self.assertEqual(' '.join(str(rt).split()), expected_str)
    self.assertEqual(' '.join(repr(rt).split()), expected_repr)

  #=============================================================================
  # RaggedTensor.with_values() and RaggedTensor.with_flat_values().
  #=============================================================================

  def testWithValues(self):
    rt1 = ragged_factory_ops.constant([[1, 2], [3, 4, 5], [6], [], [7]])
    rt2 = ragged_factory_ops.constant([[[1, 2], [3, 4, 5]], [[6]], [], [[],
                                                                        [7]]])

    rt1_plus_10 = rt1.with_values(rt1.values + 10)
    rt2_times_10 = rt2.with_flat_values(rt2.flat_values * 10)
    rt1_expanded = rt1.with_values(array_ops.expand_dims(rt1.values, axis=1))

    self.assertAllEqual(rt1_plus_10, [[11, 12], [13, 14, 15], [16], [], [17]])
    self.assertAllEqual(rt2_times_10,
                        [[[10, 20], [30, 40, 50]], [[60]], [], [[], [70]]])
    self.assertAllEqual(rt1_expanded,
                        [[[1], [2]], [[3], [4], [5]], [[6]], [], [[7]]])

  #=============================================================================
  # Session.run
  #=============================================================================
  def testSessionRun(self):
    if context.executing_eagerly():
      return

    rt1 = ragged_factory_ops.constant([[1, 2, 3], [4]])
    rt2 = ragged_factory_ops.constant([[[], [1, 2]], [[3]]])
    with self.test_session() as session:
      result = session.run({'rt1': rt1, 'rt2': rt2})
      self.assertCountEqual(result.keys(), ['rt1', 'rt2'])
      self.assertEqual(result['rt1'].to_list(), [[1, 2, 3], [4]])
      self.assertEqual(result['rt2'].to_list(), [[[], [1, 2]], [[3]]])

  def testSessionRunFeed(self):
    if context.executing_eagerly():
      return

    rt1 = RaggedTensor.from_row_splits(
        array_ops.placeholder(dtypes.int32),
        array_ops.placeholder(dtypes.int64))
    rt2 = RaggedTensor.from_nested_row_splits(
        array_ops.placeholder(dtypes.int32), [
            array_ops.placeholder(dtypes.int64),
            array_ops.placeholder(dtypes.int64)
        ])

    rt1_feed_val = ragged_factory_ops.constant_value([[1, 2, 3], [4]])
    rt2_feed_val = ragged_factory_ops.constant_value([[[], [1, 2]], [[3]]])

    with self.test_session() as session:
      fetches = {'rt1': rt1, 'rt2': rt2}
      feeds = {rt1: rt1_feed_val, rt2: rt2_feed_val}
      result = session.run(fetches, feed_dict=feeds)
      self.assertCountEqual(result.keys(), ['rt1', 'rt2'])
      self.assertEqual(result['rt1'].to_list(), [[1, 2, 3], [4]])
      self.assertEqual(result['rt2'].to_list(), [[[], [1, 2]], [[3]]])

  def testSessionPartialRunFeed(self):
    if context.executing_eagerly():
      return

    # Placeholder inputs.
    a = RaggedTensor.from_row_splits(
        array_ops.placeholder(dtypes.int32, shape=[None], name='a.values'),
        array_ops.placeholder(dtypes.int64, name='a.row_splits'))
    b = RaggedTensor.from_row_splits(
        array_ops.placeholder(dtypes.int32, shape=[None], name='b.values'),
        array_ops.placeholder(dtypes.int64, name='b.row_splits'))
    c = array_ops.placeholder(dtypes.int32, shape=[], name='c')

    # Feed values for placeholder inputs.
    a_val = ragged_factory_ops.constant_value([[1, 2, 3], [4]])
    b_val = ragged_factory_ops.constant_value([[5, 4, 3], [2]])
    c_val = 3

    # Compute some values.
    r1 = ragged_math_ops.reduce_sum(a * b, axis=1)
    r2 = ragged_math_ops.reduce_sum(a + c, axis=1)

    with self.test_session() as session:
      handle = session.partial_run_setup([r1, r2], [a, b, c])

      res1 = session.partial_run(handle, r1, feed_dict={a: a_val, b: b_val})
      self.assertAllEqual(res1, [22, 8])

      res2 = session.partial_run(handle, r2, feed_dict={c: c_val})
      self.assertAllEqual(res2, [15, 7])

  # Test case for GitHub issue 24679.
  def testEagerForLoop(self):
    if not context.executing_eagerly():
      return

    values = [[1., 2.], [3., 4., 5.], [6.]]
    r = ragged_factory_ops.constant(values)
    i = 0
    for elem in r:
      self.assertAllEqual(elem, values[i])
      i += 1

  def testConsumers(self):
    if context.executing_eagerly():
      return

    a = RaggedTensor.from_row_splits(
        array_ops.placeholder(dtypes.int32, shape=[None], name='a.values'),
        array_ops.placeholder(dtypes.int64, name='a.row_splits'),
        validate=False)
    ragged_math_ops.reduce_sum(a)
    self.assertLen(a.consumers(), 1)

  @parameterized.parameters([
      {
          'descr': 'from_value_rowids',
          'factory': RaggedTensor.from_value_rowids,
          'test': RaggedTensor.value_rowids,
          'values': {
              'values': [1, 2, 3, 4, 5, 6],
              'value_rowids': [0, 0, 1, 1, 2, 2],
          },
          'tensor_field': 'value_rowids',
          'value_rowids': [0, 1, 2],
          'nrows': 10
      },
      {
          'descr': 'from_row_splits',
          'factory': RaggedTensor.from_row_splits,
          # row_splits is a property, not a function.
          'test': (lambda rt: rt.row_splits),
          'values': {
              'values': [1, 2, 3, 4, 5, 6],
              'row_splits': [0, 2, 4, 6],
          },
          'tensor_field': 'row_splits',
          'row_splits': [0, 1, 2, 3]
      },
      {
          'descr': 'from_row_lengths',
          'factory': RaggedTensor.from_row_lengths,
          'test': RaggedTensor.row_lengths,
          'values': {
              'values': [1, 2, 3, 4, 5, 6],
              'row_lengths': [2, 2, 2],
          },
          'tensor_field': 'row_lengths',
          'row_lengths': [1, 1, 1],
      },
      # from_row_starts
      {
          'descr': 'from_row_starts',
          'factory': RaggedTensor.from_row_starts,
          'test': RaggedTensor.row_starts,
          'values': {
              'values': [1, 2, 3, 4, 5, 6],
              'row_starts': [0, 2, 4]
          },
          'tensor_field': 'row_starts',
          'row_starts': [0, 1, 2]
      },
      # from_row_limits
      {
          'descr': 'from_row_limits',
          'factory': RaggedTensor.from_row_limits,
          'test': RaggedTensor.row_limits,
          'values': {
              'values': [1, 2, 3, 4, 5, 6],
              'row_limits': [2, 4, 6]
          },
          'tensor_field': 'row_limits',
          'row_limits': [3]
      },
      # from_uniform_row_length
      {
          'descr': 'from_uniform_row_length',
          'factory': RaggedTensor.from_uniform_row_length,
          # One cannot extract uniform_row_length or nvals, so we return
          # nvals//nrows = uniform_row_length, where nvals = 3
          'test': (lambda rt: 3 // (rt.shape[0])),
          'values': {
              'values': [1, 2, 3, 4, 5, 6],
              'uniform_row_length': 2
          },
          'tensor_field': 'uniform_row_length',
          'uniform_row_length': 3
      },
  ])
  def testFactoryTypePreference(self, descr, test, factory, values,
                                tensor_field, **kwargs):
    # When input tensors have shape information, some of these errors will be
    # detected statically.
    def op_cast(k, v):
      if k == tensor_field:
        return constant_op.constant(v, dtype=dtypes.int32)
      else:
        return v

    value_copy = {k: op_cast(k, v) for k, v in values.items()}
    rt = factory(**value_copy)

    kw_copy = {k: v for k, v in kwargs.items()}
    kw_copy['values'] = rt
    rt2 = factory(**kw_copy)
    self.assertAllEqual(kwargs[tensor_field], test(rt2))

  @parameterized.parameters([
      # from_value_rowids
      {
          'descr': 'bad rank for value_rowids',
          'factory': RaggedTensor.from_value_rowids,
          'values': [[1, 2], [3, 4]],
          'value_rowids': [[1, 2], [3, 4]],
          'nrows': 10
      },
      {
          'descr': 'bad rank for nrows',
          'factory': RaggedTensor.from_value_rowids,
          'values': [1, 2, 3, 4],
          'value_rowids': [1, 2, 3, 4],
          'nrows': [10]
      },
      {
          'descr': 'len(values) != len(value_rowids)',
          'factory': RaggedTensor.from_value_rowids,
          'values': [1, 2, 3, 4],
          'value_rowids': [1, 2, 3, 4, 5],
          'nrows': 10
      },
      {
          'descr': 'negative value_rowid',
          'factory': RaggedTensor.from_value_rowids,
          'values': [1, 2, 3, 4],
          'value_rowids': [-5, 2, 3, 4],
          'nrows': 10
      },
      {
          'descr': 'non-monotonic-increasing value_rowid',
          'factory': RaggedTensor.from_value_rowids,
          'values': [1, 2, 3, 4],
          'value_rowids': [4, 3, 2, 1],
          'nrows': 10
      },
      {
          'descr': 'value_rowid > nrows',
          'factory': RaggedTensor.from_value_rowids,
          'values': [1, 2, 3, 4],
          'value_rowids': [1, 2, 3, 4],
          'nrows': 2
      },
      {
          'descr': 'bad rank for values',
          'factory': RaggedTensor.from_value_rowids,
          'values': 10,
          'value_rowids': [1, 2, 3, 4],
          'nrows': 10
      },

      # from_row_splits
      {
          'descr': 'bad rank for row_splits',
          'factory': RaggedTensor.from_row_splits,
          'values': [[1, 2], [3, 4]],
          'row_splits': [[1, 2], [3, 4]]
      },
      {
          'descr': 'row_splits[0] != 0',
          'factory': RaggedTensor.from_row_splits,
          'values': [1, 2, 3, 4],
          'row_splits': [2, 3, 4]
      },
      {
          'descr': 'non-monotonic-increasing row_splits',
          'factory': RaggedTensor.from_row_splits,
          'values': [1, 2, 3, 4],
          'row_splits': [0, 3, 2, 4]
      },
      {
          'descr': 'row_splits[0] != nvals',
          'factory': RaggedTensor.from_row_splits,
          'values': [1, 2, 3, 4],
          'row_splits': [0, 2, 3, 5]
      },
      {
          'descr': 'bad rank for values',
          'factory': RaggedTensor.from_row_splits,
          'values': 10,
          'row_splits': [0, 1]
      },

      # from_row_lengths
      {
          'descr': 'bad rank for row_lengths',
          'factory': RaggedTensor.from_row_lengths,
          'values': [1, 2, 3, 4],
          'row_lengths': [[1, 2], [1, 0]]
      },
      {
          'descr': 'negatve row_lengths',
          'factory': RaggedTensor.from_row_lengths,
          'values': [1, 2, 3, 4],
          'row_lengths': [3, -1, 2]
      },
      {
          'descr': 'sum(row_lengths) != nvals',
          'factory': RaggedTensor.from_row_lengths,
          'values': [1, 2, 3, 4],
          'row_lengths': [2, 4, 2, 8]
      },
      {
          'descr': 'bad rank for values',
          'factory': RaggedTensor.from_row_lengths,
          'values': 10,
          'row_lengths': [0, 1]
      },

      # from_row_starts
      {
          'descr': 'bad rank for row_starts',
          'factory': RaggedTensor.from_row_starts,
          'values': [[1, 2], [3, 4]],
          'row_starts': [[1, 2], [3, 4]]
      },
      {
          'descr': 'row_starts[0] != 0',
          'factory': RaggedTensor.from_row_starts,
          'values': [1, 2, 3, 4],
          'row_starts': [2, 3, 4]
      },
      {
          'descr': 'non-monotonic-increasing row_starts',
          'factory': RaggedTensor.from_row_starts,
          'values': [1, 2, 3, 4],
          'row_starts': [0, 3, 2, 4]
      },
      {
          'descr': 'row_starts[0] > nvals',
          'factory': RaggedTensor.from_row_starts,
          'values': [1, 2, 3, 4],
          'row_starts': [0, 2, 3, 5]
      },
      {
          'descr': 'bad rank for values',
          'factory': RaggedTensor.from_row_starts,
          'values': 10,
          'row_starts': [0, 1]
      },

      # from_row_limits
      {
          'descr': 'bad rank for row_limits',
          'factory': RaggedTensor.from_row_limits,
          'values': [[1, 2], [3, 4]],
          'row_limits': [[1, 2], [3, 4]]
      },
      {
          'descr': 'row_limits[0] < 0',
          'factory': RaggedTensor.from_row_limits,
          'values': [1, 2, 3, 4],
          'row_limits': [-1, 3, 4]
      },
      {
          'descr': 'non-monotonic-increasing row_limits',
          'factory': RaggedTensor.from_row_limits,
          'values': [1, 2, 3, 4],
          'row_limits': [0, 3, 2, 4]
      },
      {
          'descr': 'row_limits[0] != nvals',
          'factory': RaggedTensor.from_row_limits,
          'values': [1, 2, 3, 4],
          'row_limits': [0, 2, 3, 5]
      },
      {
          'descr': 'bad rank for values',
          'factory': RaggedTensor.from_row_limits,
          'values': 10,
          'row_limits': [0, 1]
      },

      # from_uniform_row_length
      {
          'descr': 'rowlen * nrows != nvals (1)',
          'factory': RaggedTensor.from_uniform_row_length,
          'values': [1, 2, 3, 4, 5],
          'uniform_row_length': 3
      },
      {
          'descr': 'rowlen * nrows != nvals (2)',
          'factory': RaggedTensor.from_uniform_row_length,
          'values': [1, 2, 3, 4, 5],
          'uniform_row_length': 6
      },
      {
          'descr': 'rowlen * nrows != nvals (3)',
          'factory': RaggedTensor.from_uniform_row_length,
          'values': [1, 2, 3, 4, 5, 6],
          'uniform_row_length': 3,
          'nrows': 3
      },
      {
          'descr': 'rowlen must be a scalar',
          'factory': RaggedTensor.from_uniform_row_length,
          'values': [1, 2, 3, 4],
          'uniform_row_length': [2]
      },
      {
          'descr': 'rowlen must be nonnegative',
          'factory': RaggedTensor.from_uniform_row_length,
          'values': [1, 2, 3, 4],
          'uniform_row_length': -1
      },
  ])
  def testFactoryValidation(self, descr, factory, **kwargs):
    # When input tensors have shape information, some of these errors will be
    # detected statically.
    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      self.evaluate(factory(**kwargs))

    # Remove shape information (by wrapping tensors in placeholders), and check
    # that we detect the errors when the graph is run.
    if not context.executing_eagerly():

      def wrap_arg(v):
        return array_ops.placeholder_with_default(
            constant_op.constant(v, dtype=dtypes.int64),
            tensor_shape.TensorShape(None))

      kwargs = dict((k, wrap_arg(v)) for (k, v) in kwargs.items())

      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(factory(**kwargs))

  #=============================================================================
  # RaggedTensor Variant conversion
  #=============================================================================

  @parameterized.named_parameters(
      {
          'testcase_name': 'Shape_5_none',
          'ragged_constant': [[1, 2], [3, 4, 5], [6], [], [7]],
          'ragged_rank': 1
      }, {
          'testcase_name': 'Shape_4_none_2',
          'ragged_constant': [[[1, 2]], [], [[3, 4]], []],
          'ragged_rank': 1
      }, {
          'testcase_name': 'Shape_1_none_none',
          'ragged_constant': [[[1], [2, 3, 4, 5, 6, 7]], [[]]],
          'ragged_rank': 2
      })
  def testRaggedToVariant(self, ragged_constant, ragged_rank):
    rt = ragged_factory_ops.constant(ragged_constant, ragged_rank=ragged_rank)
    et = rt._to_variant()
    self.assertEqual(et.shape.as_list(), [])
    self.assertEqual(et.dtype, dtypes.variant)

  @parameterized.parameters(
      {
          'ragged_constant': [[1, 2], [3, 4, 5], [6], [], [7]],
          'ragged_rank': 1,
          'num_batched_elems': 5
      }, {
          'ragged_constant': [[[1, 2]], [], [[3, 4]], []],
          'ragged_rank': 1,
          'num_batched_elems': 4
      }, {
          'ragged_constant': [[[1], [2, 3, 4, 5, 6, 7]], [[]]],
          'ragged_rank': 2,
          'num_batched_elems': 2
      })
  def testRaggedToBatchedVariant(self, ragged_constant, ragged_rank,
                                 num_batched_elems):
    rt = ragged_factory_ops.constant(ragged_constant, ragged_rank=ragged_rank)
    et = rt._to_variant(batched_input=True)
    self.assertEqual(et.shape.as_list(), [num_batched_elems])
    self.assertEqual(et.dtype, dtypes.variant)

  @parameterized.parameters(
      # 2D test cases.
      {
          'ragged_constant': [[]],
          'ragged_rank': 1,
      },
      {
          'ragged_constant': [[1]],
          'ragged_rank': 1,
      },
      {
          'ragged_constant': [[1, 2]],
          'ragged_rank': 1,
      },
      {
          'ragged_constant': [[1], [2], [3]],
          'ragged_rank': 1,
      },
      {
          'ragged_constant': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
          'ragged_rank': 1,
      },
      {
          'ragged_constant': [[1, 2], [3, 4, 5], [6], [], [7]],
          'ragged_rank': 1,
      },
      # 3D test cases.
      {
          'ragged_constant': [[[]]],
          'ragged_rank': 2,
      },
      {
          'ragged_constant': [[[1]]],
          'ragged_rank': 2,
      },
      {
          'ragged_constant': [[[1, 2]]],
          'ragged_rank': 2,
      },
      {
          'ragged_constant': [[[1, 2], [3, 4]]],
          'ragged_rank': 2,
      },
      {
          'ragged_constant': [[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]],
          'ragged_rank': 2,
      },
      {
          'ragged_constant': [[[1], [2]], [[3], [4]], [[5], [6]], [[7], [8]]],
          'ragged_rank': 2,
      },
      {
          'ragged_constant': [[[1, 2]], [], [[3, 4]], []],
          'ragged_rank': 2,
      },
      # 4D test cases.
      {
          'ragged_constant': [[[[1, 2], [3, 4]]],
                              [[[0, 0], [0, 0]], [[5, 6], [7, 8]]], []],
          'ragged_rank': 3,
      },
      # dtype `string`.
      {
          'ragged_constant': [['a'], ['b'], ['c']],
          'ragged_rank': 1,
          'dtype': dtypes.string,
      },
      {
          'ragged_constant': [[['a', 'b'], ['c', 'd']]],
          'ragged_rank': 2,
          'dtype': dtypes.string,
      },
      {
          'ragged_constant': [[[['a', 'b'], ['c', 'd']]],
                              [[['e', 'f'], ['g', 'h']], [['i', 'j'],
                                                          ['k', 'l']]], []],
          'ragged_rank': 3,
          'dtype': dtypes.string,
      })
  def testVariantRoundTrip(self,
                           ragged_constant,
                           ragged_rank,
                           dtype=dtypes.int32):
    rt = ragged_factory_ops.constant(
        ragged_constant, ragged_rank=ragged_rank, dtype=dtype)
    et = rt._to_variant()
    round_trip_rt = RaggedTensor._from_variant(
        et, dtype, output_ragged_rank=ragged_rank)
    self.assertAllEqual(rt, round_trip_rt)

  def testBatchedVariantRoundTripInputRaggedRankInferred(self):
    ragged_rank = 1
    rt = ragged_factory_ops.constant(
        [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
        ragged_rank=ragged_rank)
    batched_variant = rt._to_variant(batched_input=True)
    nested_batched_variant = array_ops.reshape(batched_variant, [5, 2])
    decoded_rt = RaggedTensor._from_variant(
        nested_batched_variant,
        dtype=dtypes.int32,
        output_ragged_rank=ragged_rank + 1)
    expected_rt = ragged_factory_ops.constant([[[0], [1]], [[2], [3]], [[4],
                                                                        [5]],
                                               [[6], [7]], [[8], [9]]])
    self.assertAllEqual(decoded_rt, expected_rt)

  def testBatchedVariantRoundTripWithInputRaggedRank(self):
    ragged_rank = 1
    rt = ragged_factory_ops.constant(
        [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
        ragged_rank=ragged_rank)
    batched_variant = rt._to_variant(batched_input=True)
    nested_batched_variant = array_ops.reshape(batched_variant, [5, 2])
    decoded_rt = RaggedTensor._from_variant(
        nested_batched_variant,
        dtype=dtypes.int32,
        output_ragged_rank=ragged_rank + 1,
        input_ragged_rank=ragged_rank - 1)
    expected_rt = ragged_factory_ops.constant([[[0], [1]], [[2], [3]], [[4],
                                                                        [5]],
                                               [[6], [7]], [[8], [9]]])
    self.assertAllEqual(decoded_rt, expected_rt)

  def testUnbatchVariant(self):  # b/141789000
    rt = ragged_factory_ops.constant([[1, 2, 3], [4, 5], [], [6, 7, 8, 9]])
    batched = rt._to_variant(batched_input=True)
    for i in range(4):
      row = RaggedTensor._from_variant(
          batched[i], dtype=dtypes.int32, output_ragged_rank=0)
      self.assertAllEqual(rt[i], row)

  def testUnbatchVariantInDataset(self):
    rt = ragged_factory_ops.constant([[1, 2, 3], [4, 5], [], [6, 7, 8, 9]])
    ds = dataset_ops.Dataset.from_tensor_slices(rt)
    if context.executing_eagerly():
      for i, value in enumerate(ds):
        self.assertAllEqual(rt[i], value)
    else:
      it = dataset_ops.make_one_shot_iterator(ds)
      out = it.get_next()
      with self.cached_session() as sess:
        for i in range(3):
          self.assertAllEqual(sess.run(rt[i]), out)

  def testFromVariantInvalidParams(self):
    rt = ragged_factory_ops.constant([[0], [1], [2], [3]])
    batched_variant = rt._to_variant(batched_input=True)
    nested_batched_variant = array_ops.reshape(batched_variant, [2, 2])
    with self.assertRaisesRegex(ValueError,
                                'output_ragged_rank must be equal to'):
      RaggedTensor._from_variant(
          nested_batched_variant,
          dtype=dtypes.int32,
          output_ragged_rank=1,
          input_ragged_rank=1)

  def _testRaggedVarientGradient(self, func, x, expected_grad):
    x = constant_op.constant(x)
    if context.executing_eagerly():
      with backprop.GradientTape() as t:
        t.watch(x)
        y = func(x)
        g = t.gradient(y, x)
    else:
      y = func(x)
      g = gradients_impl.gradients(ys=y, xs=x)[0]
    self.assertAllClose(g, expected_grad)

  def testRaggedVariantGradients(self):

    def func(x):
      rt1 = RaggedTensor.from_row_splits(values=x, row_splits=[0, 4, 7, 8])
      rt2 = rt1 * [[10], [100], [1000]]
      v = rt2._to_variant(batched_input=False)
      rt3 = RaggedTensor._from_variant(v, dtype=rt2.dtype, output_ragged_rank=1)
      return rt3.flat_values

    self._testRaggedVarientGradient(
        func, [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
        [10., 10., 10., 10., 100., 100., 100., 1000.])

  def testRaggedVariantGradientsBatched(self):

    def func(x):
      rt1 = RaggedTensor.from_row_splits(values=x, row_splits=[0, 4, 7, 8])
      rt2 = rt1 * [[10], [100], [1000]]
      v = rt2._to_variant(batched_input=True)
      rt3 = RaggedTensor._from_variant(v, dtype=rt2.dtype, output_ragged_rank=1)
      return rt3.flat_values

    self._testRaggedVarientGradient(
        func, [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
        [10., 10., 10., 10., 100., 100., 100., 1000.])

  def testRaggedVariantGradientsBatchedAndSliced(self):

    def func(x, i):
      rt1 = RaggedTensor.from_row_splits(values=x, row_splits=[0, 4, 7, 8])
      rt2 = rt1 * [[10], [100], [1000]]
      v_slice = rt2._to_variant(batched_input=True)[i]
      return RaggedTensor._from_variant(
          v_slice, dtype=rt2.dtype, output_ragged_rank=0)

    self._testRaggedVarientGradient(
        functools.partial(func, i=0), [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
        [10., 10., 10., 10., 0., 0., 0., 0.])
    self._testRaggedVarientGradient(
        functools.partial(func, i=1), [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
        [0., 0., 0., 0., 100., 100., 100., 0.])
    self._testRaggedVarientGradient(
        functools.partial(func, i=2), [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
        [0., 0., 0., 0., 0., 0., 0., 1000.])

  def testRaggedVariantGradientsRaggedRank0(self):

    def func(x):
      x2 = x * 2
      v = gen_ragged_conversion_ops.ragged_tensor_to_variant(
          [], x2, batched_input=False)
      return RaggedTensor._from_variant(v, dtype=x2.dtype, output_ragged_rank=0)

    self._testRaggedVarientGradient(func,
                                    [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
                                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

  def testRaggedVariantGradientsRaggedRank3(self):

    def func(x):
      x2 = x * 2
      rt1 = RaggedTensor.from_nested_row_splits(
          x2, ([0, 0, 3], [0, 2, 2, 3], [0, 4, 7, 8]))
      v = rt1._to_variant(batched_input=False)
      rt3 = RaggedTensor._from_variant(v, dtype=x2.dtype, output_ragged_rank=3)
      return rt3.flat_values

    self._testRaggedVarientGradient(func,
                                    [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
                                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

  def testRaggedVariantGradientsViaMapFn(self):
    rt = RaggedTensor.from_row_splits(
        values=[3, 1.0, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 7, 8])

    def func(x):

      def transform_row(row):
        return math_ops.sqrt(
            math_ops.reduce_mean(math_ops.square(row * x), keepdims=True))

      return math_ops.reduce_sum(map_fn.map_fn(transform_row, rt))

    self._testRaggedVarientGradient(func, 3.0, 14.653377)

  def testRaggedVariantGradientsViaMapFnReduce(self):

    def func(x):
      rt1 = RaggedTensor.from_row_splits(values=x, row_splits=[0, 4, 7, 8])
      return map_fn.map_fn(
          math_ops.reduce_max,
          rt1,
          fn_output_signature=tensor_spec.TensorSpec((), x.dtype))

    self._testRaggedVarientGradient(func,
                                    [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
                                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0])

  def testRaggedVariantGradientsErrors(self):
    if context.executing_eagerly():
      return

    rt = RaggedTensor.from_row_splits([1.0, 2.0], row_splits=[0, 2, 2])
    v1 = rt._to_variant()
    v2 = array_ops.stack([array_ops.stack([v1])])
    y = RaggedTensor._from_variant(v2, rt.dtype, output_ragged_rank=3)

    with self.assertRaisesRegex(
        ValueError, 'Unable to compute gradient: RaggedTensorToVariant '
        'can currently only generate 0D or 1D output.'):
      gradients_impl.gradients(ys=y.flat_values, xs=rt.flat_values)

  def assertNumpyObjectTensorsRecursivelyEqual(self, a, b, msg):
    """Check that two numpy arrays are equal.

    For arrays with dtype=object, check values recursively to see if a and b
    are equal.  (c.f. `np.array_equal`, which checks dtype=object values using
    object identity.)

    Args:
      a: A numpy array.
      b: A numpy array.
      msg: Message to display if a != b.
    """
    if isinstance(a, np.ndarray) and a.dtype == object:
      self.assertEqual(a.dtype, b.dtype, msg)
      self.assertEqual(a.shape, b.shape, msg)
      self.assertLen(a, len(b), msg)
      for a_val, b_val in zip(a, b):
        self.assertNumpyObjectTensorsRecursivelyEqual(a_val, b_val, msg)
    else:
      self.assertAllEqual(a, b, msg)

  @parameterized.named_parameters([
      ('Shape_2_R',
       [[1, 2], [3, 4, 5]],
       np.array([int32array([1, 2]), int32array([3, 4, 5])])),
      ('Shape_2_2',
       [[1, 2], [3, 4]],
       np.array([[1, 2], [3, 4]])),
      ('Shape_2_R_2',
       [[[1, 2], [3, 4]], [[5, 6]]],
       np.array([int32array([[1, 2], [3, 4]]), int32array([[5, 6]])])),
      ('Shape_3_2_R',
       [[[1], []], [[2, 3], [4]], [[], [5, 6, 7]]],
       np.array([[int32array([1]), int32array([])],
                 [int32array([2, 3]), int32array([4])],
                 [int32array([]), int32array([5, 6, 7])]])),
      ('Shape_0_R',
       ragged_factory_ops.constant_value([], ragged_rank=1, dtype=np.int32),
       np.zeros([0, 0], dtype=np.int32)),
      ('Shape_0_R_2',
       ragged_factory_ops.constant_value([], ragged_rank=1,
                                         inner_shape=(2,), dtype=np.int32),
       np.zeros([0, 0, 2], dtype=np.int32)),
  ])  # pyformat: disable
  def testRaggedTensorNumpy(self, rt, expected):
    if isinstance(rt, list):
      rt = ragged_factory_ops.constant(rt, dtype=dtypes.int32)
    else:
      rt = ragged_tensor.convert_to_tensor_or_ragged_tensor(rt)
    if context.executing_eagerly():
      actual = rt.numpy()
      self.assertNumpyObjectTensorsRecursivelyEqual(
          expected, actual, 'Expected %r, got %r' % (expected, actual))
    else:
      with self.assertRaisesRegex(ValueError, 'only supported in eager mode'):
        rt.numpy()

  @parameterized.parameters([
      ([[[1, 2], [3, 4, 5]], [[6]]], 2, None),
      ([[[1, 2], [3, 4, 5]], [[6]]], 2, [None, None, None]),
      ([[[1, 2], [3, 4, 5]], [[6]]], 2, [2, None, None]),
      ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]]], 1, None),
      ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]]], 1, [None, None, None]),
      ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]]], 1, [2, None, None]),
      ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]]], 1, [2, None, 3]),
      ([[[1, 2, 3]]], 1, [1, 1, None]),
      ([[[1, 2, 3]]], 1, [1, 1, 3]),
  ])
  def testRaggedTensorSetShape(self, rt, rt_ragged_rank, shape):
    rt1 = ragged_factory_ops.constant(rt, ragged_rank=rt_ragged_rank)
    rt1._set_shape(shape)
    rt1.shape.assert_is_compatible_with(shape)
    if shape is not None:
      self.assertIsNot(rt1.shape.rank, None)
      for a, b in zip(rt1.shape, shape):
        if b is not None:
          self.assertEqual(a, b)

  @parameterized.parameters([
      ([[[1, 2], [3, 4, 5]], [[6]]], 2, None),
      ([[[1, 2], [3, 4, 5]], [[6]]], 2, [None, None, None]),
      ([[[1, 2], [3, 4, 5]], [[6]]], 2, [2, None, None]),
      ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]]], 1, None),
      ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]]], 1, [None, None, None]),
      ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]]], 1, [2, None, None]),
      ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]]], 1, [2, None, 3]),
      ([[[1, 2, 3]]], 1, [1, 1, None]),
      ([[[1, 2, 3]]], 1, [1, 1, 3]),
  ])
  def testRaggedTensorSetShapeWithPlaceholders(self, rt, rt_ragged_rank, shape):
    rt2 = nest.map_structure(
        lambda x: array_ops.placeholder_with_default(x, None),
        ragged_factory_ops.constant(rt, ragged_rank=rt_ragged_rank),
        expand_composites=True)
    rt2._set_shape(shape)
    rt2.shape.assert_is_compatible_with(shape)
    if shape is not None:
      self.assertIsNot(rt2.shape.rank, None)
      for a, b in zip(rt2.shape, shape):
        if b is not None:
          self.assertEqual(a, b)

  def testRaggedTensorSetShapeUniformRowLength(self):
    rt = [[[1], [2], [3]], [[4], [5], [6]]]

    rt1 = RaggedTensor.from_tensor(rt, ragged_rank=1)
    rt1._set_shape([2, 3, 1])

    rt2 = nest.map_structure(
        lambda x: array_ops.placeholder_with_default(x, None),
        rt1,
        expand_composites=True)
    rt2._set_shape([2, 3, 1])

  def testRaggedTensorSetShapeInconsistentShapeError(self):
    rt = RaggedTensor.from_tensor([[[1], [2], [3]], [[4], [5], [6]]],
                                  ragged_rank=1)
    self.assertEqual(rt.shape.as_list(), [2, 3, 1])
    with self.assertRaises(ValueError):
      rt._set_shape([None, None, 5])
    with self.assertRaisesRegex(ValueError, 'Inconsistent size'):
      rt._set_shape([None, 5, None])
    with self.assertRaises(ValueError):
      rt._set_shape([5, None, None])


@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorSpecTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):

  def assertAllTensorsEqual(self, list1, list2):
    self.assertLen(list1, len(list2))
    for (t1, t2) in zip(list1, list2):
      self.assertAllEqual(t1, t2)

  def testConstruction(self):
    spec1 = RaggedTensorSpec(ragged_rank=1)
    self.assertIsNone(spec1._shape.rank)
    self.assertEqual(spec1._dtype, dtypes.float32)
    self.assertEqual(spec1._row_splits_dtype, dtypes.int64)
    self.assertEqual(spec1._ragged_rank, 1)

    self.assertIsNone(spec1.shape.rank)
    self.assertEqual(spec1.dtype, dtypes.float32)
    self.assertEqual(spec1.row_splits_dtype, dtypes.int64)
    self.assertEqual(spec1.ragged_rank, 1)

    spec2 = RaggedTensorSpec(shape=[None, None, None])
    self.assertEqual(spec2._shape.as_list(), [None, None, None])
    self.assertEqual(spec2._dtype, dtypes.float32)
    self.assertEqual(spec2._row_splits_dtype, dtypes.int64)
    self.assertEqual(spec2._ragged_rank, 2)

    with self.assertRaisesRegex(ValueError, 'Must specify ragged_rank'):
      RaggedTensorSpec()
    with self.assertRaisesRegex(TypeError, 'ragged_rank must be an int'):
      RaggedTensorSpec(ragged_rank=constant_op.constant(1))
    with self.assertRaisesRegex(ValueError,
                                'ragged_rank must be less than rank'):
      RaggedTensorSpec(ragged_rank=2, shape=[None, None])

  def testValueType(self):
    spec1 = RaggedTensorSpec(ragged_rank=1)
    self.assertEqual(spec1.value_type, RaggedTensor)
    spec2 = RaggedTensorSpec(ragged_rank=0)
    self.assertEqual(spec2.value_type, ops.Tensor)

  @parameterized.parameters([
      (RaggedTensorSpec(ragged_rank=1),
       (tensor_shape.TensorShape(None), dtypes.float32, 1, dtypes.int64)),
      (RaggedTensorSpec(shape=[5, None, None]),
       (tensor_shape.TensorShape([5, None, None]), dtypes.float32,
        2, dtypes.int64)),
      (RaggedTensorSpec(shape=[5, None, None], dtype=dtypes.int32),
       (tensor_shape.TensorShape([5, None, None]), dtypes.int32, 2,
        dtypes.int64)),
      (RaggedTensorSpec(ragged_rank=1, row_splits_dtype=dtypes.int32),
       (tensor_shape.TensorShape(None), dtypes.float32, 1, dtypes.int32)),
  ])  # pyformat: disable
  def testSerialize(self, rt_spec, expected):
    serialization = rt_spec._serialize()
    # TensorShape has an unconventional definition of equality, so we can't use
    # assertEqual directly here.  But repr() is deterministic and lossless for
    # the expected values, so we can use that instead.
    self.assertEqual(repr(serialization), repr(expected))

  @parameterized.parameters([
      (RaggedTensorSpec(ragged_rank=0, shape=[5, 3]), [
          tensor_spec.TensorSpec([5, 3], dtypes.float32),
      ]),
      (RaggedTensorSpec(ragged_rank=1), [
          tensor_spec.TensorSpec(None, dtypes.float32),
          tensor_spec.TensorSpec([None], dtypes.int64)
      ]),
      (RaggedTensorSpec(ragged_rank=1, row_splits_dtype=dtypes.int32), [
          tensor_spec.TensorSpec(None, dtypes.float32),
          tensor_spec.TensorSpec([None], dtypes.int32),
      ]),
      (RaggedTensorSpec(ragged_rank=2), [
          tensor_spec.TensorSpec(None, dtypes.float32),
          tensor_spec.TensorSpec([None], dtypes.int64),
          tensor_spec.TensorSpec([None], dtypes.int64),
      ]),
      (RaggedTensorSpec(shape=[5, None, None], dtype=dtypes.string), [
          tensor_spec.TensorSpec([None], dtypes.string),
          tensor_spec.TensorSpec([6], dtypes.int64),
          tensor_spec.TensorSpec([None], dtypes.int64),
      ]),
  ])
  def testComponentSpecs(self, rt_spec, expected):
    self.assertEqual(rt_spec._component_specs, expected)

  @parameterized.parameters([
      {
          'rt_spec': RaggedTensorSpec(ragged_rank=0),
          'rt': [1.0, 2.0, 3.0],
          'components': [[1.0, 2.0, 3.0]]
      },
      {
          'rt_spec': RaggedTensorSpec(ragged_rank=1),
          'rt': [[1.0, 2.0], [3.0]],
          'components': [[1.0, 2.0, 3.0], [0, 2, 3]]
      },
      {
          'rt_spec': RaggedTensorSpec(shape=[2, None, None]),
          'rt': [[[1.0, 2.0], [3.0]], [[], [4.0]]],
          'components': [[1.0, 2.0, 3.0, 4.0], [0, 2, 4], [0, 2, 3, 3, 4]]
      },
  ])
  def testToFromComponents(self, rt_spec, rt, components):
    rt = ragged_factory_ops.constant(rt)
    actual_components = rt_spec._to_components(rt)
    self.assertAllTensorsEqual(actual_components, components)
    rt_reconstructed = rt_spec._from_components(actual_components)
    self.assertAllEqual(rt, rt_reconstructed)

  @test_util.run_v1_only('RaggedTensorValue is deprecated in v2')
  def testFromNumpyComponents(self):
    spec1 = RaggedTensorSpec(ragged_rank=1, dtype=dtypes.int32)
    rt1 = spec1._from_components([np.array([1, 2, 3]), np.array([0, 2, 3])])
    self.assertIsInstance(rt1, ragged_tensor_value.RaggedTensorValue)
    self.assertAllEqual(rt1, [[1, 2], [3]])

    spec2 = RaggedTensorSpec(ragged_rank=2, dtype=dtypes.int32)
    rt2 = spec2._from_components(
        [np.array([1, 2, 3]),
         np.array([0, 2, 3]),
         np.array([0, 0, 2, 3])])
    self.assertIsInstance(rt2, ragged_tensor_value.RaggedTensorValue)
    self.assertAllEqual(rt2, [[[], [1, 2]], [[3]]])

    spec3 = RaggedTensorSpec(ragged_rank=0, dtype=dtypes.int32)
    rt3 = spec3._from_components([np.array([1, 2, 3])])
    self.assertIsInstance(rt3, np.ndarray)
    self.assertAllEqual(rt3, [1, 2, 3])

  @parameterized.parameters([
      RaggedTensorSpec(ragged_rank=0, shape=[5, 3]),
      RaggedTensorSpec(ragged_rank=1),
      RaggedTensorSpec(ragged_rank=1, row_splits_dtype=dtypes.int32),
      RaggedTensorSpec(ragged_rank=2, dtype=dtypes.string),
      RaggedTensorSpec(shape=[5, None, None]),
  ])
  def testFlatTensorSpecs(self, rt_spec):
    self.assertEqual(rt_spec._flat_tensor_specs,
                     [tensor_spec.TensorSpec(None, dtypes.variant)])

  @parameterized.named_parameters([
      {
          'testcase_name': 'RaggedRank0',
          'rt_spec': RaggedTensorSpec(ragged_rank=0),
          'rt': [1.0, 2.0, 3.0],
      },
      {
          'testcase_name': 'RaggedRank1',
          'rt_spec': RaggedTensorSpec(ragged_rank=1),
          'rt': [[1.0, 2.0], [3.0]]
      },
      {
          'testcase_name': 'RaggedRank2',
          'rt_spec': RaggedTensorSpec(shape=[2, None, None]),
          'rt': [[[1.0, 2.0], [3.0]], [[], [4.0]]]
      },
  ])
  def testToFromTensorList(self, rt_spec, rt):
    rt = ragged_factory_ops.constant(rt)
    tensor_list = rt_spec._to_tensor_list(rt)
    rt_reconstructed = rt_spec._from_tensor_list(tensor_list)
    self.assertAllEqual(rt, rt_reconstructed)

  @parameterized.named_parameters([
      # TODO(b/141789000) Test ragged_rank=0 when support is added.
      {
          'testcase_name': 'RaggedRank1',
          'rt_spec': RaggedTensorSpec(ragged_rank=1),
          'rt': [[1.0, 2.0], [3.0]]
      },
      {
          'testcase_name': 'RaggedRank2',
          'rt_spec': RaggedTensorSpec(shape=[2, None, None]),
          'rt': [[[1.0, 2.0], [3.0]], [[], [4.0]]]
      },
  ])
  def testToFromBatchedTensorList(self, rt_spec, rt):
    rt = ragged_factory_ops.constant(rt)
    tensor_list = rt_spec._to_batched_tensor_list(rt)
    rt_reconstructed = rt_spec._from_tensor_list(tensor_list)
    self.assertAllEqual(rt, rt_reconstructed)
    first_row = rt_spec._unbatch()._from_tensor_list(
        [t[0] for t in tensor_list])
    self.assertAllEqual(rt[0], first_row)

  def testToFromBatchedTensorListPreservesUniformRowLengths(self):
    rt = RaggedTensor.from_tensor(array_ops.zeros([3, 4, 5]), ragged_rank=2)
    rt_spec = rt._type_spec
    tensor_list = rt_spec._to_batched_tensor_list(rt)
    rt_reconstructed = rt_spec._from_tensor_list(tensor_list)
    self.assertAllEqual(rt, rt_reconstructed)
    self.assertTrue(rt.shape.is_fully_defined())
    self.assertTrue(rt_reconstructed.shape.is_fully_defined())
    self.assertEqual(rt.shape.as_list(), rt_reconstructed.shape.as_list())

  @parameterized.parameters([
      (RaggedTensorSpec([2, None], dtypes.float32, 1), 32,
       RaggedTensorSpec([32, 2, None], dtypes.float32, 2)),
      (RaggedTensorSpec([4, None], dtypes.float32, 1), None,
       RaggedTensorSpec([None, 4, None], dtypes.float32, 2)),
      (RaggedTensorSpec([2], dtypes.float32,
                        -1), 32, RaggedTensorSpec([32, 2], dtypes.float32, 0)),
  ])
  def testBatch(self, spec, batch_size, expected):
    self.assertEqual(spec._batch(batch_size), expected)

  @parameterized.parameters([
      (RaggedTensorSpec([32, None, None], dtypes.float32, 2),
       RaggedTensorSpec([None, None], dtypes.float32, 1)),
      (RaggedTensorSpec([None, None, None], dtypes.float32, 2),
       RaggedTensorSpec([None, None], dtypes.float32, 1)),
      (RaggedTensorSpec([32, 2], dtypes.float32, 0),
       RaggedTensorSpec([2], dtypes.float32, -1)),
      (RaggedTensorSpec([32, None, 4], dtypes.float32, 1, dtypes.int32),
       RaggedTensorSpec([None, 4], dtypes.float32, 0, dtypes.int32)),
  ])  # pyformat: disable
  def testUnbatch(self, spec, expected):
    self.assertEqual(spec._unbatch(), expected)

  def testIsCompatibleWith(self):
    spec1 = RaggedTensorSpec([32, None, None], dtypes.float32, 2)
    spec2 = RaggedTensorSpec(None, dtypes.float32, 2)
    spec3 = RaggedTensorSpec(None, dtypes.int32, 1)
    spec4 = RaggedTensorSpec([None], dtypes.int32, 0)

    self.assertTrue(spec1.is_compatible_with(spec2))
    self.assertFalse(spec1.is_compatible_with(spec3))
    self.assertFalse(spec1.is_compatible_with(spec4))
    self.assertFalse(spec2.is_compatible_with(spec3))
    self.assertFalse(spec2.is_compatible_with(spec4))
    self.assertFalse(spec3.is_compatible_with(spec4))
    self.assertTrue(spec4.is_compatible_with(constant_op.constant([1, 2, 3])))


if __name__ == '__main__':
  googletest.main()
