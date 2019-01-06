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

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.ops.ragged import ragged_test_util
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
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
class RaggedTensorTest(ragged_test_util.RaggedTensorTestCase,
                       parameterized.TestCase):
  longMessage = True  # Property in unittest.Testcase. pylint: disable=invalid-name

  #=============================================================================
  # RaggedTensor class docstring examples
  #=============================================================================

  def testClassDocStringExamples(self):
    # From section: "Component Tensors"
    rt = RaggedTensor.from_row_splits(
        values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
    self.assertRaggedEqual(rt, [[3, 1, 4, 1], [], [5, 9, 2], [6], []])
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
      self.assertRaggedEqual(rt, [[3, 1, 4, 1], [], [5, 9, 2], [6], []])
    del rt1, rt2, rt3, rt4, rt5

    # From section: "Multiple Ragged Dimensions"
    inner_rt = RaggedTensor.from_row_splits(
        values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
    outer_rt = RaggedTensor.from_row_splits(
        values=inner_rt, row_splits=[0, 3, 3, 5])
    self.assertEqual(outer_rt.ragged_rank, 2)
    self.assertEqual(
        self.eval_to_list(outer_rt),
        [[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]])
    del inner_rt, outer_rt

    # From section: "Multiple Ragged Dimensions"
    rt = RaggedTensor.from_nested_row_splits(
        flat_values=[3, 1, 4, 1, 5, 9, 2, 6],
        nested_row_splits=([0, 3, 3, 5], [0, 4, 4, 7, 8, 8]))
    self.assertEqual(
        self.eval_to_list(rt), [[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]])
    del rt

    # From section: "Uniform Inner Dimensions"
    rt = RaggedTensor.from_row_splits(
        values=array_ops.ones([5, 3]), row_splits=[0, 2, 5])
    self.assertEqual(
        self.eval_to_list(rt),
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
    self.assertEqual(len(rt_value.nested_row_splits), 1)
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
    self.assertEqual(len(rt_value.nested_row_splits), 2)
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

    self.assertEqual(
        self.eval_to_list(rt),
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

    rt = RaggedTensor.from_value_rowids(values, value_rowids)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_value_rowids = rt.value_rowids()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertIs(rt_value_rowids, value_rowids)  # cached_value_rowids
    self.assertAllEqual(rt_value_rowids, value_rowids)
    self.assertEqual(self.eval_to_list(rt_nrows), 5)
    self.assertEqual(
        self.eval_to_list(rt),
        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromValueRowIdsWithDerivedNRowsDynamic(self):
    # nrows is not known at graph creation time.
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    value_rowids = array_ops.placeholder_with_default(value_rowids, shape=None)

    rt = RaggedTensor.from_value_rowids(values, value_rowids)
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
    self.assertEqual(self.eval_to_list(rt_nrows), 5)
    self.assertEqual(
        self.eval_to_list(rt),
        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromValueRowIdsWithExplicitNRows(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(7, dtypes.int64)

    rt = RaggedTensor.from_value_rowids(values, value_rowids, nrows)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [7, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_value_rowids = rt.value_rowids()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertIs(rt_value_rowids, value_rowids)  # cached_value_rowids
    self.assertIs(rt_nrows, nrows)  # cached_nrows
    self.assertEqual(
        self.eval_to_list(rt),
        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g'], [], []])

  def testFromValueRowIdsWithExplicitNRowsEqualToDefault(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(5, dtypes.int64)

    rt = RaggedTensor.from_value_rowids(values, value_rowids, nrows)
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
    self.assertEqual(
        self.eval_to_list(rt),
        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromValueRowIdsWithEmptyValues(self):
    rt = RaggedTensor.from_value_rowids([], [])
    rt_nrows = rt.nrows()
    self.assertEqual(rt.dtype, dtypes.float32)
    self.assertEqual(rt.shape.as_list(), [0, None])
    self.assertEqual(rt.ragged_rank, 1)
    self.assertEqual(rt.values.shape.as_list(), [0])
    self.assertEqual(rt.value_rowids().shape.as_list(), [0])
    self.assertEqual(self.eval_to_list(rt_nrows), 0)
    self.assertEqual(self.eval_to_list(rt), [])

  def testFromRowSplits(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)

    rt = RaggedTensor.from_row_splits(values, row_splits)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_row_splits = rt.row_splits
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertIs(rt_row_splits, row_splits)
    self.assertEqual(self.eval_to_list(rt_nrows), 5)
    self.assertEqual(
        self.eval_to_list(rt),
        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromRowSplitsWithEmptySplits(self):
    err_msg = 'row_splits tensor may not be empty'
    with self.assertRaisesRegexp(ValueError, err_msg):
      RaggedTensor.from_row_splits([], [])

  def testFromRowStarts(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_starts = constant_op.constant([0, 2, 2, 5, 6], dtypes.int64)

    rt = RaggedTensor.from_row_starts(values, row_starts)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_row_starts = rt.row_starts()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertEqual(self.eval_to_list(rt_nrows), 5)
    self.assertAllEqual(rt_row_starts, row_starts)
    self.assertEqual(
        self.eval_to_list(rt),
        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromRowLimits(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_limits = constant_op.constant([2, 2, 5, 6, 7], dtypes.int64)

    rt = RaggedTensor.from_row_limits(values, row_limits)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_row_limits = rt.row_limits()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertEqual(self.eval_to_list(rt_nrows), 5)
    self.assertAllEqual(rt_row_limits, row_limits)
    self.assertEqual(
        self.eval_to_list(rt),
        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromRowLengths(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_lengths = constant_op.constant([2, 0, 3, 1, 1], dtypes.int64)

    rt = RaggedTensor.from_row_lengths(values, row_lengths)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_row_lengths = rt.row_lengths()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertIs(rt_row_lengths, row_lengths)  # cached_nrows
    self.assertEqual(self.eval_to_list(rt_nrows), 5)
    self.assertAllEqual(rt_row_lengths, row_lengths)
    self.assertEqual(
        self.eval_to_list(rt),
        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

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
    self.assertEqual(
        self.eval_to_list(rt),
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
    self.assertEqual(
        self.eval_to_list(rt), [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [],
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

    rt = RaggedTensor.from_nested_row_splits(flat_values, nested_row_splits)
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
    self.assertEqual(
        self.eval_to_list(rt),
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
      self.assertRaggedEqual(
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
      self.assertEqual(
          self.eval_to_list(rt),
          [[[0, 1], [2, 3]], [], [[4, 5], [6, 7], [8, 9]], [[10, 11]],
           [[12, 13]]])
      self.assertEqual(
          self.eval_to_list(rt.values),
          [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]])
      self.assertEqual(rt.values.shape.dims[0].value, 7)
      self.assertEqual(
          self.eval_to_list(rt.value_rowids()), [0, 0, 2, 2, 2, 3, 4])
      self.assertEqual(self.eval_to_list(rt.nrows()), 5)
      self.assertEqual(self.eval_to_list(rt.row_splits), [0, 2, 2, 5, 6, 7])
      self.assertEqual(self.eval_to_list(rt.row_starts()), [0, 2, 2, 5, 6])
      self.assertEqual(self.eval_to_list(rt.row_limits()), [2, 2, 5, 6, 7])
      self.assertEqual(self.eval_to_list(rt.row_lengths()), [2, 0, 3, 1, 1])
      self.assertEqual(
          self.eval_to_list(rt.flat_values),
          [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]])
      self.assertEqual([self.eval_to_list(s) for s in rt.nested_row_splits],
                       [[0, 2, 2, 5, 6, 7]])

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
      self.assertEqual(
          self.eval_to_list(rt),
          [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [], [[b'f'], [b'g']]])
      self.assertEqual(
          self.eval_to_list(rt.values),
          [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])
      self.assertEqual(rt.values.shape.dims[0].value, 5)
      self.assertEqual(self.eval_to_list(rt.value_rowids()), [0, 0, 1, 3, 3])
      self.assertEqual(self.eval_to_list(rt.nrows()), 4)
      self.assertEqual(self.eval_to_list(rt.row_splits), [0, 2, 3, 3, 5])
      self.assertEqual(self.eval_to_list(rt.row_starts()), [0, 2, 3, 3])
      self.assertEqual(self.eval_to_list(rt.row_limits()), [2, 3, 3, 5])
      self.assertEqual(self.eval_to_list(rt.row_lengths()), [2, 1, 0, 2])
      self.assertEqual(
          self.eval_to_list(rt.flat_values),
          [b'a', b'b', b'c', b'd', b'e', b'f', b'g'])
      self.assertEqual([self.eval_to_list(s) for s in rt.nested_row_splits],
                       [[0, 2, 3, 3, 5], [0, 2, 2, 5, 6, 7]])

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
        `tf.placeholder()` (so value is not known at graph construction time).

    Args:
      rt: The RaggedTensor to test.
      slice_spec: The slice spec.
      expected: The expected value of rt.__getitem__(slice_spec), as a python
        list; or an exception class.
    """
    tensor_slice_spec1 = _make_tensor_slice_spec(slice_spec, True)
    tensor_slice_spec2 = _make_tensor_slice_spec(slice_spec, False)
    value1 = self.eval_to_list(rt.__getitem__(slice_spec))
    value2 = self.eval_to_list(rt.__getitem__(tensor_slice_spec1))
    value3 = self.eval_to_list(rt.__getitem__(tensor_slice_spec2))
    self.assertEqual(value1, expected, 'slice_spec=%s' % (slice_spec,))
    self.assertEqual(value2, expected, 'slice_spec=%s' % (slice_spec,))
    self.assertEqual(value3, expected, 'slice_spec=%s' % (slice_spec,))

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
      (SLICE_BUILDER[-1:, 1:4],
       [row[1:4] for row in EXAMPLE_RAGGED_TENSOR_2D[-1:]]),
      (SLICE_BUILDER[:, 1:4], [row[1:4] for row in EXAMPLE_RAGGED_TENSOR_2D]),
      (SLICE_BUILDER[:, -2:], [row[-2:] for row in EXAMPLE_RAGGED_TENSOR_2D]),
      # TODO(edloper): Add tests for strided slices, once support is added.
  )
  def testRaggedTensorGetItemWithRaggedRank1(self, slice_spec, expected):
    """Test that rt.__getitem__(slice_spec) == expected."""
    # Ragged tensor
    rt = RaggedTensor.from_row_splits(EXAMPLE_RAGGED_TENSOR_2D_VALUES,
                                      EXAMPLE_RAGGED_TENSOR_2D_SPLITS)

    self.assertEqual(self.eval_to_list(rt), EXAMPLE_RAGGED_TENSOR_2D)
    self._TestGetItem(rt, slice_spec, expected)

  # pylint: disable=invalid-slice-index
  @parameterized.parameters(
      # Tests for out-of-bound errors
      (SLICE_BUILDER[5],
       (IndexError, ValueError, errors.InvalidArgumentError),
       '.*out of bounds.*'),
      (SLICE_BUILDER[-6],
       (IndexError, ValueError, errors.InvalidArgumentError),
       '.*out of bounds.*'),
      (SLICE_BUILDER[0, 2],
       (IndexError, ValueError, errors.InvalidArgumentError),
       '.*out of bounds.*'),
      (SLICE_BUILDER[3, 0],
       (IndexError, ValueError, errors.InvalidArgumentError),
       '.*out of bounds.*'),

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

    self.assertEqual(self.eval_to_list(rt), EXAMPLE_RAGGED_TENSOR_2D)
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
      (SLICE_BUILDER[1::2], EXAMPLE_RAGGED_TENSOR_4D[1::2]),
      (SLICE_BUILDER[:, ::2], [row[::2] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[:, 1::2], [row[1::2] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[:, :, ::2],
       [[v[::2] for v in row] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[:, :, 1::2],
       [[v[1::2] for v in row] for row in EXAMPLE_RAGGED_TENSOR_4D]),

      # TODO(edloper): Add tests for strided slices, once support is added.
      # TODO(edloper): Add tests slicing inner ragged dimensions, one support
      # is added.
  )
  def testRaggedTensorGetItemWithRaggedRank2(self, slice_spec, expected):
    """Test that rt.__getitem__(slice_spec) == expected."""
    rt = RaggedTensor.from_nested_row_splits(
        EXAMPLE_RAGGED_TENSOR_4D_VALUES,
        [EXAMPLE_RAGGED_TENSOR_4D_SPLITS1, EXAMPLE_RAGGED_TENSOR_4D_SPLITS2])
    self.assertEqual(self.eval_to_list(rt), EXAMPLE_RAGGED_TENSOR_4D)
    self._TestGetItem(rt, slice_spec, expected)

  @parameterized.parameters(
      # Test for errors in unsupported cases
      (SLICE_BUILDER[:, 0], ValueError,
       'Cannot index into an inner ragged dimension.'),
      (SLICE_BUILDER[:, :, 0], ValueError,
       'Cannot index into an inner ragged dimension.'),

      # Test for out-of-bounds errors.
      (SLICE_BUILDER[1, 0],
       (IndexError, ValueError, errors.InvalidArgumentError),
       '.*out of bounds.*'),
      (SLICE_BUILDER[0, 0, 3],
       (IndexError, ValueError, errors.InvalidArgumentError),
       '.*out of bounds.*'),
      (SLICE_BUILDER[5],
       (IndexError, ValueError, errors.InvalidArgumentError),
       '.*out of bounds.*'),
      (SLICE_BUILDER[0, 5],
       (IndexError, ValueError, errors.InvalidArgumentError),
       '.*out of bounds.*'),
  )
  def testRaggedTensorGetItemErrorsWithRaggedRank2(self, slice_spec, expected,
                                                   message):
    """Test that rt.__getitem__(slice_spec) == expected."""
    rt = RaggedTensor.from_nested_row_splits(
        EXAMPLE_RAGGED_TENSOR_4D_VALUES,
        [EXAMPLE_RAGGED_TENSOR_4D_SPLITS1, EXAMPLE_RAGGED_TENSOR_4D_SPLITS2])
    self.assertEqual(self.eval_to_list(rt), EXAMPLE_RAGGED_TENSOR_4D)
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
      (SLICE_BUILDER[0],
       (IndexError, ValueError, errors.InvalidArgumentError),
       '.*out of bounds.*'),
      (SLICE_BUILDER[-1],
       (IndexError, ValueError, errors.InvalidArgumentError),
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
    self.assertEqual(self.eval_to_list(rt), EXAMPLE_RAGGED_TENSOR_2D)
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

    self.assertEqual(
        self.eval_to_list(rt),
        [[[[b'a', b'b'], [b'c', b'd']], [], [[b'e', b'f']]], []])
    self.assertEqual(
        self.eval_to_list(rt_newaxis0),
        [[[[[b'a', b'b'], [b'c', b'd']], [], [[b'e', b'f']]], []]])
    self.assertEqual(
        self.eval_to_list(rt_newaxis1),
        [[[[[b'a', b'b'], [b'c', b'd']], [], [[b'e', b'f']]]], [[]]])
    self.assertEqual(
        self.eval_to_list(rt_newaxis2),
        [[[[[b'a', b'b'], [b'c', b'd']]], [[]], [[[b'e', b'f']]]], []])
    self.assertEqual(
        self.eval_to_list(rt_newaxis3),
        [[[[[b'a', b'b']], [[b'c', b'd']]], [], [[[b'e', b'f']]]], []])
    self.assertEqual(
        self.eval_to_list(rt_newaxis4),
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
    rt = RaggedTensor.from_row_splits(values, row_splits)
    if context.executing_eagerly():
      expected_str = '<tf.RaggedTensor {}>'.format([[b'a', b'b'],
                                                    [b'c', b'd', b'e'], [b'f'],
                                                    [], [b'g']])
      expected_repr = (
          'tf.RaggedTensor(values=tf.Tensor([{}], shape=(7,), dtype=string), '
          'row_splits=tf.Tensor([{}], shape=(6,), dtype=int64))'.format(
              ' '.join(repr(x) for x in values), ' '.join(
                  repr(x) for x in row_splits)))
      self.assertEqual(str(rt), expected_str)
      self.assertEqual(repr(rt), expected_repr)
    else:
      expected_repr = (
          'tf.RaggedTensor(values=Tensor("RaggedFromRowSplits/values:0", '
          'shape=(7,), dtype=string), row_splits='
          'Tensor("RaggedFromRowSplits/row_splits:0", '
          'shape=(6,), dtype=int64))')
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

    self.assertEqual(
        self.eval_to_list(rt1_plus_10),
        [[11, 12], [13, 14, 15], [16], [], [17]])
    self.assertEqual(
        self.eval_to_list(rt2_times_10),
        [[[10, 20], [30, 40, 50]], [[60]], [], [[], [70]]])
    self.assertEqual(
        self.eval_to_list(rt1_expanded),
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
      self.assertCountEqual(sorted(result.keys()), ['rt1', 'rt2'])
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
      result = session.run({
          'rt1': rt1,
          'rt2': rt2
      },
                           feed_dict={
                               rt1: rt1_feed_val,
                               rt2: rt2_feed_val
                           })
      self.assertCountEqual(sorted(result.keys()), ['rt1', 'rt2'])
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
      value = values[i]
      i += 1
      self.assertAllEqual(elem, value)

if __name__ == '__main__':
  googletest.main()
