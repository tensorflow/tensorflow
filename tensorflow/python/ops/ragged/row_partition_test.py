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

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged.row_partition import RowPartition
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

# Example 3D ragged tensor with uniform_row_lengths.
EXAMPLE_RAGGED_TENSOR_3D = [[[1, 2, 3], [4], [5, 6]], [[], [7, 8, 9], []]]
EXAMPLE_RAGGED_TENSOR_3D_ROWLEN = 3
EXAMPLE_RAGGED_TENSOR_3D_SPLITS = [0, 3, 4, 6, 6, 9, 9]
EXAMPLE_RAGGED_TENSOR_3D_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def int32array(values):
  return np.array(values, dtype=np.int32)


@test_util.run_all_in_graph_and_eager_modes
class RowPartitionTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  longMessage = True  # Property in unittest.Testcase. pylint: disable=invalid-name

  #=============================================================================
  # RaggedTensor class docstring examples
  #=============================================================================

  def testClassDocStringExamples(self):
    # From section: "Component Tensors"
    rt = RowPartition.from_row_splits(row_splits=[0, 4, 4, 7, 8, 8])
    self.assertAllEqual(rt.row_splits, [0, 4, 4, 7, 8, 8])
    del rt

    # From section: "Alternative Row-Partitioning Schemes"
    rt1 = RowPartition.from_row_splits(row_splits=[0, 4, 4, 7, 8, 8])
    rt2 = RowPartition.from_row_lengths(row_lengths=[4, 0, 3, 1, 0])
    rt3 = RowPartition.from_value_rowids(
        value_rowids=[0, 0, 0, 0, 2, 2, 2, 3], nrows=5)
    rt4 = RowPartition.from_row_starts(row_starts=[0, 4, 4, 7, 8], nvals=8)
    rt5 = RowPartition.from_row_limits(row_limits=[4, 4, 7, 8, 8])
    for rt in (rt1, rt2, rt3, rt4, rt5):
      self.assertAllEqual(rt.row_splits, [0, 4, 4, 7, 8, 8])
    del rt1, rt2, rt3, rt4, rt5

    # From section: "Multiple Ragged Dimensions"
    inner_rt = RowPartition.from_row_splits(row_splits=[0, 4, 4, 7, 8, 8])
    outer_rt = RowPartition.from_row_splits(row_splits=[0, 3, 3, 5])
    del inner_rt, outer_rt

  #=============================================================================
  # RaggedTensor Constructor (private)
  #=============================================================================

  def testRaggedTensorConstruction(self):
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    rt = RowPartition(row_splits=row_splits, internal=True)
    self.assertAllEqual(rt.row_splits, [0, 2, 2, 5, 6, 7])

  def testRaggedTensorConstructionErrors(self):
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)

    with self.assertRaisesRegexp(ValueError,
                                 'RaggedTensor constructor is private'):
      RowPartition(row_splits=row_splits)

    with self.assertRaisesRegexp(TypeError,
                                 'Row-partitioning argument must be a Tensor'):
      RowPartition(row_splits=[0, 2, 2, 5, 6, 7], internal=True)

    with self.assertRaisesRegexp(ValueError,
                                 r'Shape \(6, 1\) must have rank 1'):
      RowPartition(
          row_splits=array_ops.expand_dims(row_splits, 1), internal=True)

    with self.assertRaisesRegexp(TypeError,
                                 'Cached value must be a Tensor or None.'):
      RowPartition(
          row_splits=row_splits, cached_row_lengths=[2, 3, 4], internal=True)

  #=============================================================================
  # RaggedTensor Factory Ops
  #=============================================================================

  def testFromValueRowIdsWithDerivedNRows(self):
    # nrows is known at graph creation time.
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    # TODO(martinz): add nrows
    rt = RowPartition.from_value_rowids(value_rowids, validate=False)
    self.assertEqual(rt.dtype, dtypes.int64)

    rt_row_splits = rt.row_splits
    rt_value_rowids = rt.value_rowids()
    rt_nrows = rt.nrows()

    self.assertIs(rt_value_rowids, value_rowids)  # cached_value_rowids
    self.assertAllEqual(rt_value_rowids, value_rowids)
    self.assertAllEqual(rt_nrows, 5)
    self.assertAllEqual(rt_row_splits, [0, 2, 2, 5, 6, 7])

  def testFromValueRowIdsWithDerivedNRowsDynamic(self):
    # nrows is not known at graph creation time.
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    value_rowids = array_ops.placeholder_with_default(value_rowids, shape=None)

    rt = RowPartition.from_value_rowids(value_rowids, validate=False)

    rt_value_rowids = rt.value_rowids()
    rt_nrows = rt.nrows()

    self.assertIs(rt_value_rowids, value_rowids)  # cached_value_rowids
    self.assertAllEqual(rt_value_rowids, value_rowids)
    self.assertAllEqual(rt_nrows, 5)

  def testFromValueRowIdsWithExplicitNRows(self):
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(7, dtypes.int64)

    rt = RowPartition.from_value_rowids(value_rowids, nrows, validate=False)

    rt_value_rowids = rt.value_rowids()
    rt_nrows = rt.nrows()
    rt_row_splits = rt.row_splits

    self.assertIs(rt_value_rowids, value_rowids)  # cached_value_rowids
    self.assertIs(rt_nrows, nrows)  # cached_nrows
    self.assertAllEqual(rt_row_splits, [0, 2, 2, 5, 6, 7, 7, 7])

  def testFromValueRowIdsWithExplicitNRowsEqualToDefault(self):
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(5, dtypes.int64)

    rt = RowPartition.from_value_rowids(value_rowids, nrows, validate=False)

    rt_value_rowids = rt.value_rowids()
    rt_nrows = rt.nrows()
    rt_row_splits = rt.row_splits

    self.assertIs(rt_value_rowids, value_rowids)  # cached_value_rowids
    self.assertIs(rt_nrows, nrows)  # cached_nrows
    self.assertAllEqual(rt_value_rowids, value_rowids)
    self.assertAllEqual(rt_nrows, nrows)
    self.assertAllEqual(rt_row_splits, [0, 2, 2, 5, 6, 7])

  def testFromValueRowIdsWithEmptyValues(self):
    rt = RowPartition.from_value_rowids([])
    rt_nrows = rt.nrows()
    self.assertEqual(rt.dtype, dtypes.int64)
    self.assertEqual(rt.value_rowids().shape.as_list(), [0])
    self.assertAllEqual(rt_nrows, 0)

  def testFromRowSplits(self):
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)

    rt = RowPartition.from_row_splits(row_splits, validate=False)
    self.assertEqual(rt.dtype, dtypes.int64)

    rt_row_splits = rt.row_splits
    rt_nrows = rt.nrows()

    self.assertIs(rt_row_splits, row_splits)
    self.assertAllEqual(rt_nrows, 5)

  def testFromRowSplitsWithDifferentSplitTypes(self):
    splits1 = [0, 2, 2, 5, 6, 7]
    splits2 = np.array([0, 2, 2, 5, 6, 7], np.int64)
    splits3 = np.array([0, 2, 2, 5, 6, 7], np.int32)
    splits4 = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    splits5 = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int32)
    rt1 = RowPartition.from_row_splits(splits1)
    rt2 = RowPartition.from_row_splits(splits2)
    rt3 = RowPartition.from_row_splits(splits3)
    rt4 = RowPartition.from_row_splits(splits4)
    rt5 = RowPartition.from_row_splits(splits5)
    self.assertEqual(rt1.row_splits.dtype, dtypes.int64)
    self.assertEqual(rt2.row_splits.dtype, dtypes.int64)
    self.assertEqual(rt3.row_splits.dtype, dtypes.int32)
    self.assertEqual(rt4.row_splits.dtype, dtypes.int64)
    self.assertEqual(rt5.row_splits.dtype, dtypes.int32)

  def testFromRowSplitsWithEmptySplits(self):
    err_msg = 'row_splits tensor may not be empty'
    with self.assertRaisesRegexp(ValueError, err_msg):
      RowPartition.from_row_splits([], [])

  def testFromRowStarts(self):
    nvals = constant_op.constant(7)
    row_starts = constant_op.constant([0, 2, 2, 5, 6], dtypes.int64)

    rt = RowPartition.from_row_starts(row_starts, nvals, validate=False)
    self.assertEqual(rt.dtype, dtypes.int64)

    rt_row_starts = rt.row_starts()
    rt_row_splits = rt.row_splits
    rt_nrows = rt.nrows()

    self.assertAllEqual(rt_nrows, 5)
    self.assertAllEqual(rt_row_starts, row_starts)
    self.assertAllEqual(rt_row_splits, [0, 2, 2, 5, 6, 7])

  def testFromRowLimits(self):
    row_limits = constant_op.constant([2, 2, 5, 6, 7], dtypes.int64)

    rt = RowPartition.from_row_limits(row_limits, validate=False)
    self.assertEqual(rt.dtype, dtypes.int64)

    rt_row_limits = rt.row_limits()
    rt_row_splits = rt.row_splits
    rt_nrows = rt.nrows()

    self.assertAllEqual(rt_nrows, 5)
    self.assertAllEqual(rt_row_limits, row_limits)
    self.assertAllEqual(rt_row_splits, [0, 2, 2, 5, 6, 7])

  def testFromRowLengths(self):
    row_lengths = constant_op.constant([2, 0, 3, 1, 1], dtypes.int64)

    rt = RowPartition.from_row_lengths(row_lengths, validate=False)
    self.assertEqual(rt.dtype, dtypes.int64)

    rt_row_lengths = rt.row_lengths()
    rt_nrows = rt.nrows()

    self.assertIs(rt_row_lengths, row_lengths)  # cached_nrows
    self.assertAllEqual(rt_nrows, 5)
    self.assertAllEqual(rt_row_lengths, row_lengths)

  def testFromUniformRowLength(self):
    nvals = 16
    a1 = RowPartition.from_uniform_row_length(nvals, 2)
    self.assertAllEqual(a1.uniform_row_length(), 2)
    self.assertAllEqual(a1.nrows(), 8)

  def testFromUniformRowLengthWithEmptyValues(self):
    a = RowPartition.from_uniform_row_length(
        nvals=0, uniform_row_length=0, nrows=10)
    self.assertEqual(self.evaluate(a.nvals()), 0)
    self.assertEqual(self.evaluate(a.nrows()), 10)

  def testFromUniformRowLengthWithPlaceholders1(self):
    nvals = array_ops.placeholder_with_default(
        constant_op.constant(6, dtype=dtypes.int64), None)
    rt1 = RowPartition.from_uniform_row_length(nvals, 3)
    const_nvals1 = self.evaluate(rt1.nvals())
    self.assertEqual(const_nvals1, 6)

  def testFromUniformRowLengthWithPlaceholders2(self):
    nvals = array_ops.placeholder_with_default(6, None)
    ph_rowlen = array_ops.placeholder_with_default(3, None)
    rt2 = RowPartition.from_uniform_row_length(nvals, ph_rowlen)
    const_nvals2 = self.evaluate(rt2.nvals())
    self.assertEqual(const_nvals2, 6)

  def testFromValueRowIdsWithBadNRows(self):
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(5, dtypes.int64)

    with self.assertRaisesRegexp(ValueError, r'Expected nrows >= 0; got -2'):
      RowPartition.from_value_rowids(
          value_rowids=array_ops.placeholder_with_default(value_rowids, None),
          nrows=-2)

    with self.assertRaisesRegexp(
        ValueError, r'Expected nrows >= value_rowids\[-1\] \+ 1; got nrows=2, '
        r'value_rowids\[-1\]=4'):
      RowPartition.from_value_rowids(value_rowids=value_rowids, nrows=2)

    with self.assertRaisesRegexp(
        ValueError, r'Expected nrows >= value_rowids\[-1\] \+ 1; got nrows=4, '
        r'value_rowids\[-1\]=4'):
      RowPartition.from_value_rowids(value_rowids=value_rowids, nrows=4)

    with self.assertRaisesRegexp(ValueError,
                                 r'Shape \(7, 1\) must have rank 1'):
      RowPartition.from_value_rowids(
          value_rowids=array_ops.expand_dims(value_rowids, 1), nrows=nrows)

    with self.assertRaisesRegexp(ValueError, r'Shape \(1,\) must have rank 0'):
      RowPartition.from_value_rowids(
          value_rowids=value_rowids, nrows=array_ops.expand_dims(nrows, 0))

  #=============================================================================
  # RowPartition.__str__
  #=============================================================================
  def testRowPartitionStr(self):
    row_splits = [0, 2, 5, 6, 6, 7]
    rt = RowPartition.from_row_splits(row_splits, validate=False)
    splits_type = 'int64'
    if context.executing_eagerly():
      expected_repr = ('tf.RowPartition(row_splits=tf.Tensor([0 2 5 6 6 7], '
                       'shape=(6,), dtype=int64))')
    else:
      expected_repr = ('tf.RowPartition(row_splits='
                       'Tensor("RowPartitionFromRowSplits/row_splits:0", '
                       'shape=(6,), dtype={}))').format(splits_type)
    self.assertEqual(repr(rt), expected_repr)
    self.assertEqual(str(rt), expected_repr)

  @parameterized.parameters([
      # from_value_rowids
      {
          'descr': 'bad rank for value_rowids',
          'factory': RowPartition.from_value_rowids,
          'value_rowids': [[1, 2], [3, 4]],
          'nrows': 10
      },
      {
          'descr': 'bad rank for nrows',
          'factory': RowPartition.from_value_rowids,
          'value_rowids': [1, 2, 3, 4],
          'nrows': [10]
      },
      {
          'descr': 'negative value_rowid',
          'factory': RowPartition.from_value_rowids,
          'value_rowids': [-5, 2, 3, 4],
          'nrows': 10
      },
      {
          'descr': 'non-monotonic-increasing value_rowid',
          'factory': RowPartition.from_value_rowids,
          'value_rowids': [4, 3, 2, 1],
          'nrows': 10
      },
      {
          'descr': 'value_rowid > nrows',
          'factory': RowPartition.from_value_rowids,
          'value_rowids': [1, 2, 3, 4],
          'nrows': 2
      },

      # from_row_splits
      {
          'descr': 'bad rank for row_splits',
          'factory': RowPartition.from_row_splits,
          'row_splits': [[1, 2], [3, 4]]
      },
      {
          'descr': 'row_splits[0] != 0',
          'factory': RowPartition.from_row_splits,
          'row_splits': [2, 3, 4]
      },
      {
          'descr': 'non-monotonic-increasing row_splits',
          'factory': RowPartition.from_row_splits,
          'row_splits': [0, 3, 2, 4]
      },

      # from_row_lengths
      {
          'descr': 'bad rank for row_lengths',
          'factory': RowPartition.from_row_lengths,
          'row_lengths': [[1, 2], [1, 0]]
      },
      {
          'descr': 'negatve row_lengths',
          'factory': RowPartition.from_row_lengths,
          'row_lengths': [3, -1, 2]
      },

      # from_row_starts
      {
          'descr': 'bad rank for row_starts',
          'factory': RowPartition.from_row_starts,
          'nvals': 2,
          'row_starts': [[1, 2], [3, 4]]
      },
      {
          'descr': 'row_starts[0] != 0',
          'factory': RowPartition.from_row_starts,
          'nvals': 5,
          'row_starts': [2, 3, 4]
      },
      {
          'descr': 'non-monotonic-increasing row_starts',
          'factory': RowPartition.from_row_starts,
          'nvals': 4,
          'row_starts': [0, 3, 2, 4]
      },
      {
          'descr': 'row_starts[0] > nvals',
          'factory': RowPartition.from_row_starts,
          'nvals': 4,
          'row_starts': [0, 2, 3, 5]
      },

      # from_row_limits
      {
          'descr': 'bad rank for row_limits',
          'factory': RowPartition.from_row_limits,
          'row_limits': [[1, 2], [3, 4]]
      },
      {
          'descr': 'row_limits[0] < 0',
          'factory': RowPartition.from_row_limits,
          'row_limits': [-1, 3, 4]
      },
      {
          'descr': 'non-monotonic-increasing row_limits',
          'factory': RowPartition.from_row_limits,
          'row_limits': [0, 3, 2, 4]
      },

      # from_uniform_row_length
      {
          'descr': 'rowlen * nrows != nvals (1)',
          'factory': RowPartition.from_uniform_row_length,
          'nvals': 5,
          'uniform_row_length': 3
      },
      {
          'descr': 'rowlen * nrows != nvals (2)',
          'factory': RowPartition.from_uniform_row_length,
          'nvals': 5,
          'uniform_row_length': 6
      },
      {
          'descr': 'rowlen * nrows != nvals (3)',
          'factory': RowPartition.from_uniform_row_length,
          'nvals': 6,
          'uniform_row_length': 3,
          'nrows': 3
      },
      {
          'descr': 'rowlen must be a scalar',
          'factory': RowPartition.from_uniform_row_length,
          'nvals': 4,
          'uniform_row_length': [2]
      },
      {
          'descr': 'rowlen must be nonnegative',
          'factory': RowPartition.from_uniform_row_length,
          'nvals': 4,
          'uniform_row_length': -1
      },
  ])
  def testFactoryValidation(self, descr, factory, **kwargs):
    # When input tensors have shape information, some of these errors will be
    # detected statically.
    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      partition = factory(**kwargs)
      self.evaluate(partition.row_splits)

    # Remove shape information (by wrapping tensors in placeholders), and check
    # that we detect the errors when the graph is run.
    if not context.executing_eagerly():

      def wrap_arg(v):
        return array_ops.placeholder_with_default(
            constant_op.constant(v, dtype=dtypes.int64),
            tensor_shape.TensorShape(None))

      kwargs = dict((k, wrap_arg(v)) for (k, v) in kwargs.items())

      with self.assertRaises(errors.InvalidArgumentError):
        partition = factory(**kwargs)
        self.evaluate(partition.row_splits)


if __name__ == '__main__':
  googletest.main()
