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

import re

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensorSpec
from tensorflow.python.platform import googletest


class _SliceBuilder(object):
  """Helper to construct arguments for __getitem__.

  Usage: _SliceBuilder()[<expr>] slice_spec Python generates for <expr>.
  """

  def __getitem__(self, slice_spec):
    return slice_spec


SLICE_BUILDER = _SliceBuilder()


def _make_tensor_slice_spec(slice_spec, use_constant=True):
  """Wraps all integers in an extended slice spec w/ a tensor.

  This function is used to help test slicing when the slice spec contains
  tensors, rather than integers.

  Args:
    slice_spec: The extended slice spec.
    use_constant: If true, then wrap each integer with a tf.constant.  If false,
      then wrap each integer with a tf.placeholder.

  Returns:
    A copy of slice_spec, but with each integer i replaced with tf.constant(i).
  """

  def make_piece_scalar(piece):
    if isinstance(piece, int):
      scalar = constant_op.constant(piece)
      if use_constant:
        return scalar
      else:
        return array_ops.placeholder_with_default(scalar, [])
    elif isinstance(piece, slice):
      return slice(
          make_piece_scalar(piece.start), make_piece_scalar(piece.stop),
          make_piece_scalar(piece.step))
    else:
      return piece

  if isinstance(slice_spec, tuple):
    return tuple(make_piece_scalar(piece) for piece in slice_spec)
  else:
    return make_piece_scalar(slice_spec)


# Example 2D ragged tensor value with one ragged dimension and with scalar
# values, expressed as nested python lists and as splits+values.
EXAMPLE_RAGGED_TENSOR_2D = [[b'a', b'b'], [b'c', b'd', b'e'], [b'f'], [],
                            [b'g']]
EXAMPLE_RAGGED_TENSOR_2D_SPLITS = [0, 2, 5, 6, 6, 7]
EXAMPLE_RAGGED_TENSOR_2D_VALUES = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

# Example 4D ragged tensor value, with two ragged dimensions and with values
# whose shape is [2], expressed as nested python lists and as splits+values.
EXAMPLE_RAGGED_TENSOR_4D = [
    [                                       # rt[0]
        [[1, 2], [3, 4], [5, 6]],           # rt[0][0]
        [[7, 8], [9, 10], [11, 12]]],       # rt[0][1]
    [],                                     # rt[1]
    [                                       # rt[2]
        [[13, 14], [15, 16], [17, 18]]],    # rt[2][0]
    [                                       # rt[3]
        [[19, 20]]]                         # rt[3][0]
]  # pyformat: disable
EXAMPLE_RAGGED_TENSOR_4D_SPLITS1 = [0, 2, 2, 3, 4]
EXAMPLE_RAGGED_TENSOR_4D_SPLITS2 = [0, 3, 6, 9, 10]
EXAMPLE_RAGGED_TENSOR_4D_VALUES = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                                   [11, 12], [13, 14], [15, 16], [17, 18],
                                   [19, 20]]


@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorTest(test_util.TensorFlowTestCase,
                       parameterized.TestCase):
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
    self.assertAllEqual(
        outer_rt,
        [[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]])
    del inner_rt, outer_rt

    # From section: "Multiple Ragged Dimensions"
    rt = RaggedTensor.from_nested_row_splits(
        flat_values=[3, 1, 4, 1, 5, 9, 2, 6],
        nested_row_splits=([0, 3, 3, 5], [0, 4, 4, 7, 8, 8]))
    self.assertAllEqual(
        rt, [[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]])
    del rt

    # From section: "Uniform Inner Dimensions"
    rt = RaggedTensor.from_row_splits(
        values=array_ops.ones([5, 3]), row_splits=[0, 2, 5])
    self.assertAllEqual(
        rt,
        [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
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
    rt = RaggedTensor(values=values, row_splits=row_splits, internal=True)

    self.assertAllEqual(
        rt,
        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testRaggedTensorConstructionErrors(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)

    with self.assertRaisesRegexp(ValueError,
                                 'RaggedTensor constructor is private'):
      RaggedTensor(values=values, row_splits=row_splits)

    with self.assertRaisesRegexp(TypeError,
                                 'values must be a Tensor or RaggedTensor'):
      RaggedTensor(values=range(7), row_splits=row_splits, internal=True)

    with self.assertRaisesRegexp(TypeError,
                                 'Row-partitioning argument must be a Tensor'):
      RaggedTensor(values=values, row_splits=[0, 2, 2, 5, 6, 7], internal=True)

    with self.assertRaisesRegexp(ValueError,
                                 r'Shape \(6, 1\) must have rank 1'):
      RaggedTensor(
          values=values,
          row_splits=array_ops.expand_dims(row_splits, 1),
          internal=True)

    with self.assertRaisesRegexp(TypeError,
                                 'Cached value must be a Tensor or None.'):
      RaggedTensor(
          values=values,
          row_splits=row_splits,
          cached_row_lengths=[2, 3, 4],
          internal=True)

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
    self.assertAllEqual(
        rt,
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
    self.assertAllEqual(
        rt,
        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromValueRowIdsWithExplicitNRows(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(7, dtypes.int64)

    rt = RaggedTensor.from_value_rowids(values, value_rowids, nrows,
                                        validate=False)
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
        rt,
        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g'], [], []])

  def testFromValueRowIdsWithExplicitNRowsEqualToDefault(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(5, dtypes.int64)

    rt = RaggedTensor.from_value_rowids(values, value_rowids, nrows,
                                        validate=False)
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
    self.assertAllEqual(
        rt,
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
    self.assertAllEqual(
        rt,
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
    with self.assertRaisesRegexp(ValueError, err_msg):
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
    self.assertAllEqual(
        rt,
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
    self.assertAllEqual(
        rt,
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
    self.assertAllEqual(
        rt,
        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromUniformRowLength(self):
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    a1 = RaggedTensor.from_uniform_row_length(values, 2)
    a2 = RaggedTensor.from_uniform_row_length(values, 2, 8)
    self.assertAllEqual(a1, [[1, 2], [3, 4], [5, 6], [7, 8],
                             [9, 10], [11, 12], [13, 14], [15, 16]])
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
        rt,
        [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [], [[b'f'], [b'g']]])

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
    self.assertAllEqual(
        rt, [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [],
             [[b'f'], [b'g'], []], [], []])

  def testFromNestedValueRowIdsWithExplicitNRowsMismatch(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_value_rowids = [
        constant_op.constant([0, 0, 1, 3, 3, 3], dtypes.int64),
        constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    ]
    nrows = [constant_op.constant(6, dtypes.int64)]
    with self.assertRaisesRegexp(
        ValueError, 'nested_nrows must have the same '
        'length as nested_value_rowids'):
      RaggedTensor.from_nested_value_rowids(values, nested_value_rowids, nrows)

  def testFromNestedValueRowIdsWithNonListInput(self):
    with self.assertRaisesRegexp(
        TypeError, 'nested_value_rowids must be a list of Tensors'):
      RaggedTensor.from_nested_value_rowids(
          [1, 2, 3], constant_op.constant([[0, 1, 2], [0, 1, 2]], dtypes.int64))
    with self.assertRaisesRegexp(TypeError,
                                 'nested_nrows must be a list of Tensors'):
      RaggedTensor.from_nested_value_rowids([1, 2, 3], [[0, 1, 2], [0, 1, 2]],
                                            constant_op.constant([3, 3]))

  def testFromNestedRowSplits(self):
    flat_values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_row_splits = [
        constant_op.constant([0, 2, 3, 3, 5], dtypes.int64),
        constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    ]

    rt = RaggedTensor.from_nested_row_splits(flat_values, nested_row_splits,
                                             validate=False)
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
        rt,
        [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [], [[b'f'], [b'g']]])

  def testFromNestedRowSplitsWithNonListInput(self):
    with self.assertRaisesRegexp(TypeError,
                                 'nested_row_splits must be a list of Tensors'):
      RaggedTensor.from_nested_row_splits(
          [1, 2], constant_op.constant([[0, 1, 2], [0, 1, 2]], dtypes.int64))

  def testFromValueRowIdsWithBadNRows(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(5, dtypes.int64)

    with self.assertRaisesRegexp(ValueError, r'Expected nrows >= 0; got -2'):
      RaggedTensor.from_value_rowids(
          values=values,
          value_rowids=array_ops.placeholder_with_default(value_rowids, None),
          nrows=-2)

    with self.assertRaisesRegexp(
        ValueError, r'Expected nrows >= value_rowids\[-1\] \+ 1; got nrows=2, '
        r'value_rowids\[-1\]=4'):
      RaggedTensor.from_value_rowids(
          values=values, value_rowids=value_rowids, nrows=2)

    with self.assertRaisesRegexp(
        ValueError, r'Expected nrows >= value_rowids\[-1\] \+ 1; got nrows=4, '
        r'value_rowids\[-1\]=4'):
      RaggedTensor.from_value_rowids(
          values=values, value_rowids=value_rowids, nrows=4)

    with self.assertRaisesRegexp(ValueError,
                                 r'Shape \(7, 1\) must have rank 1'):
      RaggedTensor.from_value_rowids(
          values=values,
          value_rowids=array_ops.expand_dims(value_rowids, 1),
          nrows=nrows)

    with self.assertRaisesRegexp(ValueError, r'Shape \(1,\) must have rank 0'):
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
      self.assertRaisesRegexp(ValueError,
                              '.* must be from the same graph as .*',
                              RaggedTensor.from_row_splits, values, splits)

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
    rt1 = RaggedTensor.from_row_splits(values, row_splits)
    rt2 = RaggedTensor.from_value_rowids(values, value_rowids)

    for rt in [rt1, rt2]:
      self.assertAllEqual(
          rt,
          [[[0, 1], [2, 3]], [], [[4, 5], [6, 7], [8, 9]], [[10, 11]],
           [[12, 13]]])
      self.assertAllEqual(
          rt.values,
          [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]])
      self.assertEqual(rt.values.shape.dims[0].value, 7)
      self.assertAllEqual(
          rt.value_rowids(), [0, 0, 2, 2, 2, 3, 4])
      self.assertAllEqual(rt.nrows(), 5)
      self.assertAllEqual(rt.row_splits, [0, 2, 2, 5, 6, 7])
      self.assertAllEqual(rt.row_starts(), [0, 2, 2, 5, 6])
      self.assertAllEqual(rt.row_limits(), [2, 2, 5, 6, 7])
      self.assertAllEqual(rt.row_lengths(), [2, 0, 3, 1, 1])
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
          rt,
          [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [], [[b'f'], [b'g']]])
      self.assertAllEqual(
          rt.values,
          [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])
      self.assertEqual(rt.values.shape.dims[0].value, 5)
      self.assertAllEqual(rt.value_rowids(), [0, 0, 1, 3, 3])
      self.assertAllEqual(rt.nrows(), 4)
      self.assertAllEqual(rt.row_splits, [0, 2, 3, 3, 5])
      self.assertAllEqual(rt.row_starts(), [0, 2, 3, 3])
      self.assertAllEqual(rt.row_limits(), [2, 3, 3, 5])
      self.assertAllEqual(rt.row_lengths(), [2, 1, 0, 2])
      self.assertAllEqual(
          rt.flat_values,
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
      self.assertEqual(rt5.shape.ndims, None)

      rt6 = RaggedTensor.from_row_splits(
          [1, 2, 3], array_ops.placeholder(dtype=dtypes.int64))
      self.assertEqual(rt6.shape.as_list(), [None, None])

  #=============================================================================
  # RaggedTensor.__getitem__
  #=============================================================================

  def _TestGetItem(self, rt, slice_spec, expected):
    """Helper function for testing RaggedTensor.__getitem__.

    Checks that calling `rt.__getitem__(slice_spec) returns the expected value.
    Checks three different configurations for each slice spec:

      * Call __getitem__ with the slice spec as-is (with int values)
      * Call __getitem__ with int values in the slice spec wrapped in
        `tf.constant()`.
      * Call __getitem__ with int values in the slice spec wrapped in
        `tf.compat.v1.placeholder()` (so value is not known at graph
        construction time).

    Args:
      rt: The RaggedTensor to test.
      slice_spec: The slice spec.
      expected: The expected value of rt.__getitem__(slice_spec), as a python
        list; or an exception class.
    """
    tensor_slice_spec1 = _make_tensor_slice_spec(slice_spec, True)
    tensor_slice_spec2 = _make_tensor_slice_spec(slice_spec, False)
    value1 = rt.__getitem__(slice_spec)
    value2 = rt.__getitem__(tensor_slice_spec1)
    value3 = rt.__getitem__(tensor_slice_spec2)
    self.assertAllEqual(value1, expected, 'slice_spec=%s' % (slice_spec,))
    self.assertAllEqual(value2, expected, 'slice_spec=%s' % (slice_spec,))
    self.assertAllEqual(value3, expected, 'slice_spec=%s' % (slice_spec,))

  def _TestGetItemException(self, rt, slice_spec, expected, message):
    """Helper function for testing RaggedTensor.__getitem__ exceptions."""
    tensor_slice_spec1 = _make_tensor_slice_spec(slice_spec, True)
    self.assertRaisesRegexp(expected, message, rt.__getitem__, slice_spec)
    self.assertRaisesRegexp(expected, message, rt.__getitem__,
                            tensor_slice_spec1)

  @parameterized.parameters(
      # Tests for rt[i]
      (SLICE_BUILDER[-5], EXAMPLE_RAGGED_TENSOR_2D[-5]),
      (SLICE_BUILDER[-4], EXAMPLE_RAGGED_TENSOR_2D[-4]),
      (SLICE_BUILDER[-1], EXAMPLE_RAGGED_TENSOR_2D[-1]),
      (SLICE_BUILDER[0], EXAMPLE_RAGGED_TENSOR_2D[0]),
      (SLICE_BUILDER[1], EXAMPLE_RAGGED_TENSOR_2D[1]),
      (SLICE_BUILDER[4], EXAMPLE_RAGGED_TENSOR_2D[4]),

      # Tests for rt[i:]
      (SLICE_BUILDER[-6:], EXAMPLE_RAGGED_TENSOR_2D[-6:]),
      (SLICE_BUILDER[-3:], EXAMPLE_RAGGED_TENSOR_2D[-3:]),
      (SLICE_BUILDER[-1:], EXAMPLE_RAGGED_TENSOR_2D[-1:]),
      (SLICE_BUILDER[0:], EXAMPLE_RAGGED_TENSOR_2D[0:]),
      (SLICE_BUILDER[3:], EXAMPLE_RAGGED_TENSOR_2D[3:]),
      (SLICE_BUILDER[5:], EXAMPLE_RAGGED_TENSOR_2D[5:]),

      # Tests for rt[:j]
      (SLICE_BUILDER[:-6], EXAMPLE_RAGGED_TENSOR_2D[:-6]),
      (SLICE_BUILDER[:-3], EXAMPLE_RAGGED_TENSOR_2D[:-3]),
      (SLICE_BUILDER[:-1], EXAMPLE_RAGGED_TENSOR_2D[:-1]),
      (SLICE_BUILDER[:0], EXAMPLE_RAGGED_TENSOR_2D[:0]),
      (SLICE_BUILDER[:3], EXAMPLE_RAGGED_TENSOR_2D[:3]),
      (SLICE_BUILDER[:5], EXAMPLE_RAGGED_TENSOR_2D[:5]),

      # Tests for rt[i:j]
      (SLICE_BUILDER[0:3], EXAMPLE_RAGGED_TENSOR_2D[0:3]),
      (SLICE_BUILDER[3:5], EXAMPLE_RAGGED_TENSOR_2D[3:5]),
      (SLICE_BUILDER[-5:3], EXAMPLE_RAGGED_TENSOR_2D[-5:3]),
      (SLICE_BUILDER[3:1], EXAMPLE_RAGGED_TENSOR_2D[3:1]),
      (SLICE_BUILDER[-1:1], EXAMPLE_RAGGED_TENSOR_2D[-1:1]),
      (SLICE_BUILDER[1:-1], EXAMPLE_RAGGED_TENSOR_2D[1:-1]),

      # Tests for rt[i, j]
      (SLICE_BUILDER[0, 1], EXAMPLE_RAGGED_TENSOR_2D[0][1]),
      (SLICE_BUILDER[1, 2], EXAMPLE_RAGGED_TENSOR_2D[1][2]),
      (SLICE_BUILDER[-1, 0], EXAMPLE_RAGGED_TENSOR_2D[-1][0]),
      (SLICE_BUILDER[-3, 0], EXAMPLE_RAGGED_TENSOR_2D[-3][0]),
      (SLICE_BUILDER[:], EXAMPLE_RAGGED_TENSOR_2D),
      (SLICE_BUILDER[:, :], EXAMPLE_RAGGED_TENSOR_2D),

      # Empty slice spec.
      ([], EXAMPLE_RAGGED_TENSOR_2D),

      # Test for ellipsis
      (SLICE_BUILDER[...], EXAMPLE_RAGGED_TENSOR_2D),
      (SLICE_BUILDER[2, ...], EXAMPLE_RAGGED_TENSOR_2D[2]),
      (SLICE_BUILDER[..., :], EXAMPLE_RAGGED_TENSOR_2D),
      (SLICE_BUILDER[..., 2, 0], EXAMPLE_RAGGED_TENSOR_2D[2][0]),
      (SLICE_BUILDER[2, ..., 0], EXAMPLE_RAGGED_TENSOR_2D[2][0]),
      (SLICE_BUILDER[2, 0, ...], EXAMPLE_RAGGED_TENSOR_2D[2][0]),

      # Test for array_ops.newaxis
      (SLICE_BUILDER[array_ops.newaxis, :], [EXAMPLE_RAGGED_TENSOR_2D]),
      (SLICE_BUILDER[:, array_ops.newaxis],
       [[row] for row in EXAMPLE_RAGGED_TENSOR_2D]),

      # Slicing inner ragged dimensions.
      (SLICE_BUILDER[-1:,
                     1:4], [row[1:4] for row in EXAMPLE_RAGGED_TENSOR_2D[-1:]]),
      (SLICE_BUILDER[:, 1:4], [row[1:4] for row in EXAMPLE_RAGGED_TENSOR_2D]),
      (SLICE_BUILDER[:, -2:], [row[-2:] for row in EXAMPLE_RAGGED_TENSOR_2D]),

      # Strided slices
      (SLICE_BUILDER[::2], EXAMPLE_RAGGED_TENSOR_2D[::2]),
      (SLICE_BUILDER[::-1], EXAMPLE_RAGGED_TENSOR_2D[::-1]),
      (SLICE_BUILDER[::-2], EXAMPLE_RAGGED_TENSOR_2D[::-2]),
      (SLICE_BUILDER[::-3], EXAMPLE_RAGGED_TENSOR_2D[::-3]),
      (SLICE_BUILDER[:, ::2], [row[::2] for row in EXAMPLE_RAGGED_TENSOR_2D]),
      (SLICE_BUILDER[:, ::-1], [row[::-1] for row in EXAMPLE_RAGGED_TENSOR_2D]),
      (SLICE_BUILDER[:, ::-2], [row[::-2] for row in EXAMPLE_RAGGED_TENSOR_2D]),
      (SLICE_BUILDER[:, ::-3], [row[::-3] for row in EXAMPLE_RAGGED_TENSOR_2D]),
      (SLICE_BUILDER[:, 2::-1],
       [row[2::-1] for row in EXAMPLE_RAGGED_TENSOR_2D]),
      (SLICE_BUILDER[:, -1::-1],
       [row[-1::-1] for row in EXAMPLE_RAGGED_TENSOR_2D]),
      (SLICE_BUILDER[..., -1::-1],
       [row[-1::-1] for row in EXAMPLE_RAGGED_TENSOR_2D]),
      (SLICE_BUILDER[:, 2::-2],
       [row[2::-2] for row in EXAMPLE_RAGGED_TENSOR_2D]),
      (SLICE_BUILDER[::-1, ::-1],
       [row[::-1] for row in EXAMPLE_RAGGED_TENSOR_2D[::-1]]),
  )  # pyformat: disable
  def testRaggedTensorGetItemWithRaggedRank1(self, slice_spec, expected):
    """Test that rt.__getitem__(slice_spec) == expected."""
    # Ragged tensor
    rt = RaggedTensor.from_row_splits(EXAMPLE_RAGGED_TENSOR_2D_VALUES,
                                      EXAMPLE_RAGGED_TENSOR_2D_SPLITS)

    self.assertAllEqual(rt, EXAMPLE_RAGGED_TENSOR_2D)
    self._TestGetItem(rt, slice_spec, expected)

  def testStridedSlices(self):
    test_value = [[1, 2, 3, 4, 5], [6, 7], [8, 9, 10], [], [9],
                  [1, 2, 3, 4, 5, 6, 7, 8]]
    rt = ragged_factory_ops.constant(test_value)
    for start in [-2, -1, None, 0, 1, 2]:
      for stop in [-2, -1, None, 0, 1, 2]:
        for step in [-3, -2, -1, 1, 2, 3]:
          # Slice outer dimension
          self.assertAllEqual(rt[start:stop:step], test_value[start:stop:step],
                              'slice=%s:%s:%s' % (start, stop, step))
          # Slice inner dimension
          self.assertAllEqual(rt[:, start:stop:step],
                              [row[start:stop:step] for row in test_value],
                              'slice=%s:%s:%s' % (start, stop, step))

  # pylint: disable=invalid-slice-index
  @parameterized.parameters(
      # Tests for out-of-bound errors
      (SLICE_BUILDER[5], (IndexError, ValueError, errors.InvalidArgumentError),
       '.*out of bounds.*'),
      (SLICE_BUILDER[-6], (IndexError, ValueError, errors.InvalidArgumentError),
       '.*out of bounds.*'),
      (SLICE_BUILDER[0, 2], (IndexError, ValueError,
                             errors.InvalidArgumentError), '.*out of bounds.*'),
      (SLICE_BUILDER[3, 0], (IndexError, ValueError,
                             errors.InvalidArgumentError), '.*out of bounds.*'),

      # Indexing into an inner ragged dimension
      (SLICE_BUILDER[:, 3], ValueError,
       'Cannot index into an inner ragged dimension'),
      (SLICE_BUILDER[:1, 3], ValueError,
       'Cannot index into an inner ragged dimension'),
      (SLICE_BUILDER[..., 3], ValueError,
       'Cannot index into an inner ragged dimension'),

      # Tests for type errors
      (SLICE_BUILDER[0.5], TypeError, re.escape(array_ops._SLICE_TYPE_ERROR)),
      (SLICE_BUILDER[1:3:0.5], TypeError, re.escape(
          array_ops._SLICE_TYPE_ERROR)),
      (SLICE_BUILDER[:, 1:3:0.5], TypeError,
       'slice strides must be integers or None'),
      (SLICE_BUILDER[:, 0.5:1.5], TypeError,
       'slice offsets must be integers or None'),
      (SLICE_BUILDER['foo'], TypeError, re.escape(array_ops._SLICE_TYPE_ERROR)),
      (SLICE_BUILDER[:, 'foo':'foo'], TypeError,
       'slice offsets must be integers or None'),

      # Tests for other errors
      (SLICE_BUILDER[..., 0, 0, 0], IndexError,
       'Too many indices for RaggedTensor'),
  )
  def testRaggedTensorGetItemErrorsWithRaggedRank1(self, slice_spec, expected,
                                                   message):
    """Test that rt.__getitem__(slice_spec) == expected."""
    # Ragged tensor
    rt = RaggedTensor.from_row_splits(EXAMPLE_RAGGED_TENSOR_2D_VALUES,
                                      EXAMPLE_RAGGED_TENSOR_2D_SPLITS)

    self.assertAllEqual(rt, EXAMPLE_RAGGED_TENSOR_2D)
    self._TestGetItemException(rt, slice_spec, expected, message)

  @parameterized.parameters(
      # Tests for rt[index, index, ...]
      (SLICE_BUILDER[2, 0], EXAMPLE_RAGGED_TENSOR_4D[2][0]),
      (SLICE_BUILDER[2, 0, 1], EXAMPLE_RAGGED_TENSOR_4D[2][0][1]),
      (SLICE_BUILDER[2, 0, 1, 1], EXAMPLE_RAGGED_TENSOR_4D[2][0][1][1]),
      (SLICE_BUILDER[2, 0, 1:], EXAMPLE_RAGGED_TENSOR_4D[2][0][1:]),
      (SLICE_BUILDER[2, 0, 1:, 1:], [[16], [18]]),
      (SLICE_BUILDER[2, 0, :, 1], [14, 16, 18]),
      (SLICE_BUILDER[2, 0, 1, :], EXAMPLE_RAGGED_TENSOR_4D[2][0][1]),

      # Tests for rt[index, slice, ...]
      (SLICE_BUILDER[0, :], EXAMPLE_RAGGED_TENSOR_4D[0]),
      (SLICE_BUILDER[1, :], EXAMPLE_RAGGED_TENSOR_4D[1]),
      (SLICE_BUILDER[0, :, :, 1], [[2, 4, 6], [8, 10, 12]]),
      (SLICE_BUILDER[1, :, :, 1], []),
      (SLICE_BUILDER[2, :, :, 1], [[14, 16, 18]]),
      (SLICE_BUILDER[3, :, :, 1], [[20]]),

      # Tests for rt[slice, slice, ...]
      (SLICE_BUILDER[:, :], EXAMPLE_RAGGED_TENSOR_4D),
      (SLICE_BUILDER[:, :, :, 1], [[[2, 4, 6], [8, 10, 12]], [], [[14, 16, 18]],
                                   [[20]]]),
      (SLICE_BUILDER[1:, :, :, 1], [[], [[14, 16, 18]], [[20]]]),
      (SLICE_BUILDER[-3:, :, :, 1], [[], [[14, 16, 18]], [[20]]]),

      # Test for ellipsis
      (SLICE_BUILDER[...], EXAMPLE_RAGGED_TENSOR_4D),
      (SLICE_BUILDER[2, ...], EXAMPLE_RAGGED_TENSOR_4D[2]),
      (SLICE_BUILDER[2, 0, ...], EXAMPLE_RAGGED_TENSOR_4D[2][0]),
      (SLICE_BUILDER[..., 0], [[[1, 3, 5], [7, 9, 11]], [], [[13, 15, 17]],
                               [[19]]]),
      (SLICE_BUILDER[2, ..., 0], [[13, 15, 17]]),
      (SLICE_BUILDER[2, 0, ..., 0], [13, 15, 17]),

      # Test for array_ops.newaxis
      (SLICE_BUILDER[array_ops.newaxis, :], [EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[:, array_ops.newaxis],
       [[row] for row in EXAMPLE_RAGGED_TENSOR_4D]),

      # Empty slice spec.
      ([], EXAMPLE_RAGGED_TENSOR_4D),

      # Slicing inner ragged dimensions.
      (SLICE_BUILDER[:, 1:4], [row[1:4] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[:, -2:], [row[-2:] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[:, :, :-1],
       [[v[:-1] for v in row] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[:, :, 1:2],
       [[v[1:2] for v in row] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[1:, 1:3, 1:2],
       [[v[1:2] for v in row[1:3]] for row in EXAMPLE_RAGGED_TENSOR_4D[1:]]),

      # Strided slices
      (SLICE_BUILDER[::2], EXAMPLE_RAGGED_TENSOR_4D[::2]),
      (SLICE_BUILDER[::-1], EXAMPLE_RAGGED_TENSOR_4D[::-1]),
      (SLICE_BUILDER[::-2], EXAMPLE_RAGGED_TENSOR_4D[::-2]),
      (SLICE_BUILDER[1::2], EXAMPLE_RAGGED_TENSOR_4D[1::2]),
      (SLICE_BUILDER[:, ::2], [row[::2] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[:, 1::2], [row[1::2] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[:, :, ::2],
       [[v[::2] for v in row] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[:, :, 1::2],
       [[v[1::2] for v in row] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[:, :, ::-1],
       [[v[::-1] for v in row] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[:, :, ::-2],
       [[v[::-2] for v in row] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[..., ::-1, :],
       [[v[::-1] for v in row] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[..., ::-1],
       [[[v[::-1] for v in col] for col in row]
        for row in EXAMPLE_RAGGED_TENSOR_4D]),
  )
  def testRaggedTensorGetItemWithRaggedRank2(self, slice_spec, expected):
    """Test that rt.__getitem__(slice_spec) == expected."""
    rt = RaggedTensor.from_nested_row_splits(
        EXAMPLE_RAGGED_TENSOR_4D_VALUES,
        [EXAMPLE_RAGGED_TENSOR_4D_SPLITS1, EXAMPLE_RAGGED_TENSOR_4D_SPLITS2])
    self.assertAllEqual(rt, EXAMPLE_RAGGED_TENSOR_4D)
    self._TestGetItem(rt, slice_spec, expected)

  @parameterized.parameters(
      # Test for errors in unsupported cases
      (SLICE_BUILDER[:, 0], ValueError,
       'Cannot index into an inner ragged dimension.'),
      (SLICE_BUILDER[:, :, 0], ValueError,
       'Cannot index into an inner ragged dimension.'),

      # Test for out-of-bounds errors.
      (SLICE_BUILDER[1, 0], (IndexError, ValueError,
                             errors.InvalidArgumentError), '.*out of bounds.*'),
      (SLICE_BUILDER[0, 0, 3],
       (IndexError, ValueError,
        errors.InvalidArgumentError), '.*out of bounds.*'),
      (SLICE_BUILDER[5], (IndexError, ValueError, errors.InvalidArgumentError),
       '.*out of bounds.*'),
      (SLICE_BUILDER[0, 5], (IndexError, ValueError,
                             errors.InvalidArgumentError), '.*out of bounds.*'),
  )
  def testRaggedTensorGetItemErrorsWithRaggedRank2(self, slice_spec, expected,
                                                   message):
    """Test that rt.__getitem__(slice_spec) == expected."""
    rt = RaggedTensor.from_nested_row_splits(
        EXAMPLE_RAGGED_TENSOR_4D_VALUES,
        [EXAMPLE_RAGGED_TENSOR_4D_SPLITS1, EXAMPLE_RAGGED_TENSOR_4D_SPLITS2])
    self.assertAllEqual(rt, EXAMPLE_RAGGED_TENSOR_4D)
    self._TestGetItemException(rt, slice_spec, expected, message)

  @parameterized.parameters(
      (SLICE_BUILDER[:], []),
      (SLICE_BUILDER[2:], []),
      (SLICE_BUILDER[:-3], []),
  )
  def testRaggedTensorGetItemWithEmptyTensor(self, slice_spec, expected):
    """Test that rt.__getitem__(slice_spec) == expected."""
    rt = RaggedTensor.from_row_splits([], [0])
    self._TestGetItem(rt, slice_spec, expected)

  @parameterized.parameters(
      (SLICE_BUILDER[0], (IndexError, ValueError, errors.InvalidArgumentError),
       '.*out of bounds.*'),
      (SLICE_BUILDER[-1], (IndexError, ValueError, errors.InvalidArgumentError),
       '.*out of bounds.*'),
  )
  def testRaggedTensorGetItemErrorsWithEmptyTensor(self, slice_spec, expected,
                                                   message):
    """Test that rt.__getitem__(slice_spec) == expected."""
    rt = RaggedTensor.from_row_splits([], [0])
    self._TestGetItemException(rt, slice_spec, expected, message)

  @parameterized.parameters(
      (SLICE_BUILDER[-4], EXAMPLE_RAGGED_TENSOR_2D[-4]),
      (SLICE_BUILDER[0], EXAMPLE_RAGGED_TENSOR_2D[0]),
      (SLICE_BUILDER[-3:], EXAMPLE_RAGGED_TENSOR_2D[-3:]),
      (SLICE_BUILDER[:3], EXAMPLE_RAGGED_TENSOR_2D[:3]),
      (SLICE_BUILDER[3:5], EXAMPLE_RAGGED_TENSOR_2D[3:5]),
      (SLICE_BUILDER[0, 1], EXAMPLE_RAGGED_TENSOR_2D[0][1]),
      (SLICE_BUILDER[-3, 0], EXAMPLE_RAGGED_TENSOR_2D[-3][0]),
  )
  def testRaggedTensorGetItemWithPlaceholderShapes(self, slice_spec, expected):
    """Test that rt.__getitem__(slice_spec) == expected."""
    # Intentionally use an unknown shape for `splits`, to force the code path
    # that deals with having nrows unknown at graph construction time.
    splits = constant_op.constant(
        EXAMPLE_RAGGED_TENSOR_2D_SPLITS, dtype=dtypes.int64)
    splits = array_ops.placeholder_with_default(splits, None)
    rt = RaggedTensor.from_row_splits(EXAMPLE_RAGGED_TENSOR_2D_VALUES, splits)
    self.assertAllEqual(rt, EXAMPLE_RAGGED_TENSOR_2D)
    self._TestGetItem(rt, slice_spec, expected)

  @parameterized.parameters(
      (SLICE_BUILDER[..., 2], ValueError,
       'Ellipsis not supported for unknown shape RaggedTensors'),)
  def testRaggedTensorGetItemErrorsWithPlaceholderShapes(
      self, slice_spec, expected, message):
    """Test that rt.__getitem__(slice_spec) == expected."""
    if not context.executing_eagerly():
      # Intentionally use an unknown shape for `values`.
      values = array_ops.placeholder_with_default([0], None)
      rt = RaggedTensor.from_row_splits(values, [0, 1])
      self._TestGetItemException(rt, slice_spec, expected, message)

  def testGetItemNewAxis(self):
    # rt: [[[['a', 'b'], ['c', 'd']], [], [['e', 'f']]], []]
    splits1 = [0, 3, 3]
    splits2 = [0, 2, 2, 3]
    values = constant_op.constant([['a', 'b'], ['c', 'd'], ['e', 'f']])
    rt = RaggedTensor.from_nested_row_splits(values, [splits1, splits2])
    rt_newaxis0 = rt[array_ops.newaxis]
    rt_newaxis1 = rt[:, array_ops.newaxis]
    rt_newaxis2 = rt[:, :, array_ops.newaxis]
    rt_newaxis3 = rt[:, :, :, array_ops.newaxis]
    rt_newaxis4 = rt[:, :, :, :, array_ops.newaxis]

    self.assertAllEqual(
        rt,
        [[[[b'a', b'b'], [b'c', b'd']], [], [[b'e', b'f']]], []])
    self.assertAllEqual(
        rt_newaxis0,
        [[[[[b'a', b'b'], [b'c', b'd']], [], [[b'e', b'f']]], []]])
    self.assertAllEqual(
        rt_newaxis1,
        [[[[[b'a', b'b'], [b'c', b'd']], [], [[b'e', b'f']]]], [[]]])
    self.assertAllEqual(
        rt_newaxis2,
        [[[[[b'a', b'b'], [b'c', b'd']]], [[]], [[[b'e', b'f']]]], []])
    self.assertAllEqual(
        rt_newaxis3,
        [[[[[b'a', b'b']], [[b'c', b'd']]], [], [[[b'e', b'f']]]], []])
    self.assertAllEqual(
        rt_newaxis4,
        [[[[[b'a'], [b'b']], [[b'c'], [b'd']]], [], [[[b'e'], [b'f']]]], []])

    self.assertEqual(rt.ragged_rank, 2)
    self.assertEqual(rt_newaxis0.ragged_rank, 3)
    self.assertEqual(rt_newaxis1.ragged_rank, 3)
    self.assertEqual(rt_newaxis2.ragged_rank, 3)
    self.assertEqual(rt_newaxis3.ragged_rank, 2)
    self.assertEqual(rt_newaxis4.ragged_rank, 2)

    self.assertEqual(rt_newaxis0.shape.as_list(), [1, None, None, None, 2])
    self.assertEqual(rt_newaxis1.shape.as_list(), [2, None, None, None, 2])
    self.assertEqual(rt_newaxis2.shape.as_list(), [2, None, None, None, 2])
    self.assertEqual(rt_newaxis3.shape.as_list(), [2, None, None, 1, 2])
    self.assertEqual(rt_newaxis4.shape.as_list(), [2, None, None, 2, 1])

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
          'shape=(7,), dtype=string), row_splits='
          'Tensor("RaggedFromRowSplits/row_splits:0", '
          'shape=(6,), dtype={}))').format(splits_type)
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

    self.assertAllEqual(
        rt1_plus_10,
        [[11, 12], [13, 14, 15], [16], [], [17]])
    self.assertAllEqual(
        rt2_times_10,
        [[[10, 20], [30, 40, 50]], [[60]], [], [[], [70]]])
    self.assertAllEqual(
        rt1_expanded,
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
      # from_value_rowids
      {'descr': 'bad rank for value_rowids',
       'factory': RaggedTensor.from_value_rowids,
       'values': [[1, 2], [3, 4]],
       'value_rowids': [[1, 2], [3, 4]],
       'nrows': 10},
      {'descr': 'bad rank for nrows',
       'factory': RaggedTensor.from_value_rowids,
       'values': [1, 2, 3, 4],
       'value_rowids': [1, 2, 3, 4],
       'nrows': [10]},
      {'descr': 'len(values) != len(value_rowids)',
       'factory': RaggedTensor.from_value_rowids,
       'values': [1, 2, 3, 4],
       'value_rowids': [1, 2, 3, 4, 5],
       'nrows': 10},
      {'descr': 'negative value_rowid',
       'factory': RaggedTensor.from_value_rowids,
       'values': [1, 2, 3, 4],
       'value_rowids': [-5, 2, 3, 4],
       'nrows': 10},
      {'descr': 'non-monotonic-increasing value_rowid',
       'factory': RaggedTensor.from_value_rowids,
       'values': [1, 2, 3, 4],
       'value_rowids': [4, 3, 2, 1],
       'nrows': 10},
      {'descr': 'value_rowid > nrows',
       'factory': RaggedTensor.from_value_rowids,
       'values': [1, 2, 3, 4],
       'value_rowids': [1, 2, 3, 4],
       'nrows': 2},
      {'descr': 'bad rank for values',
       'factory': RaggedTensor.from_value_rowids,
       'values': 10,
       'value_rowids': [1, 2, 3, 4],
       'nrows': 10},

      # from_row_splits
      {'descr': 'bad rank for row_splits',
       'factory': RaggedTensor.from_row_splits,
       'values': [[1, 2], [3, 4]],
       'row_splits': [[1, 2], [3, 4]]},
      {'descr': 'row_splits[0] != 0',
       'factory': RaggedTensor.from_row_splits,
       'values': [1, 2, 3, 4],
       'row_splits': [2, 3, 4]},
      {'descr': 'non-monotonic-increasing row_splits',
       'factory': RaggedTensor.from_row_splits,
       'values': [1, 2, 3, 4],
       'row_splits': [0, 3, 2, 4]},
      {'descr': 'row_splits[0] != nvals',
       'factory': RaggedTensor.from_row_splits,
       'values': [1, 2, 3, 4],
       'row_splits': [0, 2, 3, 5]},
      {'descr': 'bad rank for values',
       'factory': RaggedTensor.from_row_splits,
       'values': 10,
       'row_splits': [0, 1]},

      # from_row_lengths
      {'descr': 'bad rank for row_lengths',
       'factory': RaggedTensor.from_row_lengths,
       'values': [1, 2, 3, 4],
       'row_lengths': [[1, 2], [1, 0]]},
      {'descr': 'negatve row_lengths',
       'factory': RaggedTensor.from_row_lengths,
       'values': [1, 2, 3, 4],
       'row_lengths': [3, -1, 2]},
      {'descr': 'sum(row_lengths) != nvals',
       'factory': RaggedTensor.from_row_lengths,
       'values': [1, 2, 3, 4],
       'row_lengths': [2, 4, 2, 8]},
      {'descr': 'bad rank for values',
       'factory': RaggedTensor.from_row_lengths,
       'values': 10,
       'row_lengths': [0, 1]},

      # from_row_starts
      {'descr': 'bad rank for row_starts',
       'factory': RaggedTensor.from_row_starts,
       'values': [[1, 2], [3, 4]],
       'row_starts': [[1, 2], [3, 4]]},
      {'descr': 'row_starts[0] != 0',
       'factory': RaggedTensor.from_row_starts,
       'values': [1, 2, 3, 4],
       'row_starts': [2, 3, 4]},
      {'descr': 'non-monotonic-increasing row_starts',
       'factory': RaggedTensor.from_row_starts,
       'values': [1, 2, 3, 4],
       'row_starts': [0, 3, 2, 4]},
      {'descr': 'row_starts[0] > nvals',
       'factory': RaggedTensor.from_row_starts,
       'values': [1, 2, 3, 4],
       'row_starts': [0, 2, 3, 5]},
      {'descr': 'bad rank for values',
       'factory': RaggedTensor.from_row_starts,
       'values': 10,
       'row_starts': [0, 1]},

      # from_row_limits
      {'descr': 'bad rank for row_limits',
       'factory': RaggedTensor.from_row_limits,
       'values': [[1, 2], [3, 4]],
       'row_limits': [[1, 2], [3, 4]]},
      {'descr': 'row_limits[0] < 0',
       'factory': RaggedTensor.from_row_limits,
       'values': [1, 2, 3, 4],
       'row_limits': [-1, 3, 4]},
      {'descr': 'non-monotonic-increasing row_limits',
       'factory': RaggedTensor.from_row_limits,
       'values': [1, 2, 3, 4],
       'row_limits': [0, 3, 2, 4]},
      {'descr': 'row_limits[0] != nvals',
       'factory': RaggedTensor.from_row_limits,
       'values': [1, 2, 3, 4],
       'row_limits': [0, 2, 3, 5]},
      {'descr': 'bad rank for values',
       'factory': RaggedTensor.from_row_limits,
       'values': 10,
       'row_limits': [0, 1]},

      # from_uniform_row_length
      {'descr': 'rowlen * nrows != nvals (1)',
       'factory': RaggedTensor.from_uniform_row_length,
       'values': [1, 2, 3, 4, 5],
       'uniform_row_length': 3},
      {'descr': 'rowlen * nrows != nvals (2)',
       'factory': RaggedTensor.from_uniform_row_length,
       'values': [1, 2, 3, 4, 5],
       'uniform_row_length': 6},
      {'descr': 'rowlen * nrows != nvals (3)',
       'factory': RaggedTensor.from_uniform_row_length,
       'values': [1, 2, 3, 4, 5, 6],
       'uniform_row_length': 3,
       'nrows': 3},
      {'descr': 'rowlen must be a scalar',
       'factory': RaggedTensor.from_uniform_row_length,
       'values': [1, 2, 3, 4],
       'uniform_row_length': [2]},
      {'descr': 'rowlen must be nonnegative',
       'factory': RaggedTensor.from_uniform_row_length,
       'values': [1, 2, 3, 4],
       'uniform_row_length': -1},

  ])
  def testFactoryValidation(self, descr, factory, **kwargs):
    # When input tensors have shape information, some of these errors will be
    # detected statically.
    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      self.evaluate(factory(**kwargs))

    # Remove shape information (by wraping tensors in placeholders), and check
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

  @parameterized.parameters(
      {
          'ragged_constant': [[1, 2], [3, 4, 5], [6], [], [7]],
          'ragged_rank': 1
      }, {
          'ragged_constant': [[[1, 2]], [], [[3, 4]], []],
          'ragged_rank': 1
      }, {
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
    with self.assertRaisesRegexp(ValueError,
                                 'output_ragged_rank must be equal to'):
      RaggedTensor._from_variant(
          nested_batched_variant,
          dtype=dtypes.int32,
          output_ragged_rank=1,
          input_ragged_rank=1)


@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorSpecTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):

  def assertAllTensorsEqual(self, list1, list2):
    self.assertLen(list1, len(list2))
    for (t1, t2) in zip(list1, list2):
      self.assertAllEqual(t1, t2)

  def testConstruction(self):
    spec1 = RaggedTensorSpec(ragged_rank=1)
    self.assertEqual(spec1._shape.rank, None)
    self.assertEqual(spec1._dtype, dtypes.float32)
    self.assertEqual(spec1._row_splits_dtype, dtypes.int64)
    self.assertEqual(spec1._ragged_rank, 1)

    spec2 = RaggedTensorSpec(shape=[None, None, None])
    self.assertEqual(spec2._shape.as_list(), [None, None, None])
    self.assertEqual(spec2._dtype, dtypes.float32)
    self.assertEqual(spec2._row_splits_dtype, dtypes.int64)
    self.assertEqual(spec2._ragged_rank, 2)

    with self.assertRaisesRegexp(ValueError, 'Must specify ragged_rank'):
      RaggedTensorSpec()
    with self.assertRaisesRegexp(TypeError, 'ragged_rank must be an int'):
      RaggedTensorSpec(ragged_rank=constant_op.constant(1))
    with self.assertRaisesRegexp(ValueError,
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
    rt2 = spec2._from_components([np.array([1, 2, 3]), np.array([0, 2, 3]),
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


if __name__ == '__main__':
  googletest.main()
