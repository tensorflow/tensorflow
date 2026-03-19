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
# limitations under the License.
# ==============================================================================
"""Tests for tf.ragged.RowPartition."""

import copy
from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import row_partition
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.ragged.row_partition import RowPartitionSpec
from tensorflow.python.platform import googletest


def _get_specified_row_partition():
  """Needed for merge_with_spec tests. Normally, nvals isn't set."""
  return RowPartition(
      row_splits=constant_op.constant([0, 3, 8], dtype=dtypes.int64),
      nrows=constant_op.constant(2, dtype=dtypes.int64),
      nvals=constant_op.constant(8, dtype=dtypes.int64),
      internal=row_partition._row_partition_factory_key)


@test_util.run_all_in_graph_and_eager_modes
class RowPartitionTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # =============================================================================
  # RowPartition class docstring examples
  # =============================================================================

  def testClassDocStringExamples(self):
    # From section: "Component Tensors"
    rp = RowPartition.from_row_splits(row_splits=[0, 4, 4, 7, 8, 8])
    self.assertAllEqual(rp.row_splits(), [0, 4, 4, 7, 8, 8])
    del rp

    # From section: "Alternative Row-Partitioning Schemes"
    rt1 = RowPartition.from_row_splits(row_splits=[0, 4, 4, 7, 8, 8])
    rt2 = RowPartition.from_row_lengths(row_lengths=[4, 0, 3, 1, 0])
    rt3 = RowPartition.from_value_rowids(
        value_rowids=[0, 0, 0, 0, 2, 2, 2, 3], nrows=5)
    rt4 = RowPartition.from_row_starts(row_starts=[0, 4, 4, 7, 8], nvals=8)
    rt5 = RowPartition.from_row_limits(row_limits=[4, 4, 7, 8, 8])
    for rp in (rt1, rt2, rt3, rt4, rt5):
      self.assertAllEqual(rp.row_splits(), [0, 4, 4, 7, 8, 8])
    del rt1, rt2, rt3, rt4, rt5

    # From section: "Multiple Ragged Dimensions"
    inner_rt = RowPartition.from_row_splits(row_splits=[0, 4, 4, 7, 8, 8])
    outer_rt = RowPartition.from_row_splits(row_splits=[0, 3, 3, 5])
    del inner_rt, outer_rt

  # =============================================================================
  # RowPartition Constructor (private)
  # =============================================================================

  def testRowPartitionConstruction(self):
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    rp = RowPartition(
        row_splits=row_splits,
        internal=row_partition._row_partition_factory_key)
    self.assertAllEqual(rp.row_splits(), [0, 2, 2, 5, 6, 7])

  def testRowPartitionConstructionErrors(self):
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)

    with self.assertRaisesRegex(ValueError,
                                'RowPartition constructor is private'):
      RowPartition(row_splits=row_splits)

    with self.assertRaisesRegex(TypeError,
                                'Row-partitioning argument must be a Tensor'):
      RowPartition(
          row_splits=[0, 2, 2, 5, 6, 7],
          internal=row_partition._row_partition_factory_key)

    with self.assertRaisesRegex(ValueError, r'Shape \(6, 1\) must have rank 1'):
      RowPartition(
          row_splits=array_ops.expand_dims(row_splits, 1),
          internal=row_partition._row_partition_factory_key)

    with self.assertRaisesRegex(TypeError,
                                'Cached value must be a Tensor or None.'):
      RowPartition(
          row_splits=row_splits,
          row_lengths=[2, 3, 4],
          internal=row_partition._row_partition_factory_key)

    with self.assertRaisesRegex(ValueError, 'Inconsistent dtype'):
      RowPartition(
          row_splits=constant_op.constant([0, 3], dtypes.int64),
          nrows=constant_op.constant(1, dtypes.int32),
          internal=row_partition._row_partition_factory_key)

  # =============================================================================
  # RowPartition Factory Ops
  # =============================================================================

  def testFromValueRowIdsWithDerivedNRows(self):
    # nrows is known at graph creation time.
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    # TODO(martinz): add nrows
    rp = RowPartition.from_value_rowids(value_rowids, validate=False)
    self.assertEqual(rp.dtype, dtypes.int64)

    rp_row_splits = rp.row_splits()
    rp_value_rowids = rp.value_rowids()
    rp_nrows = rp.nrows()

    self.assertIs(rp_value_rowids, value_rowids)  # value_rowids
    self.assertAllEqual(rp_value_rowids, value_rowids)
    self.assertAllEqual(rp_nrows, 5)
    self.assertAllEqual(rp_row_splits, [0, 2, 2, 5, 6, 7])

  def testFromValueRowIdsWithDerivedNRowsDynamic(self):
    # nrows is not known at graph creation time.
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    value_rowids = array_ops.placeholder_with_default(value_rowids, shape=None)

    rp = RowPartition.from_value_rowids(value_rowids, validate=False)

    rp_value_rowids = rp.value_rowids()
    rp_nrows = rp.nrows()

    self.assertIs(rp_value_rowids, value_rowids)  # value_rowids
    self.assertAllEqual(rp_value_rowids, value_rowids)
    self.assertAllEqual(rp_nrows, 5)

  def testFromValueRowIdsWithExplicitNRows(self):
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(7, dtypes.int64)

    rp = RowPartition.from_value_rowids(value_rowids, nrows, validate=False)

    rp_value_rowids = rp.value_rowids()
    rp_nrows = rp.nrows()
    rp_row_splits = rp.row_splits()

    self.assertIs(rp_value_rowids, value_rowids)  # value_rowids
    self.assertIs(rp_nrows, nrows)  # nrows
    self.assertAllEqual(rp_row_splits, [0, 2, 2, 5, 6, 7, 7, 7])

  def testFromValueRowIdsWithExplicitNRowsEqualToDefault(self):
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(5, dtypes.int64)

    rp = RowPartition.from_value_rowids(value_rowids, nrows, validate=False)

    rp_value_rowids = rp.value_rowids()
    rp_nrows = rp.nrows()
    rp_row_splits = rp.row_splits()

    self.assertIs(rp_value_rowids, value_rowids)  # value_rowids
    self.assertIs(rp_nrows, nrows)  # nrows
    self.assertAllEqual(rp_value_rowids, value_rowids)
    self.assertAllEqual(rp_nrows, nrows)
    self.assertAllEqual(rp_row_splits, [0, 2, 2, 5, 6, 7])

  def testFromValueRowIdsWithEmptyValues(self):
    rp = RowPartition.from_value_rowids([])
    rp_nrows = rp.nrows()
    self.assertEqual(rp.dtype, dtypes.int64)
    self.assertEqual(rp.value_rowids().shape.as_list(), [0])
    self.assertAllEqual(rp_nrows, 0)

  def testFromRowSplits(self):
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)

    rp = RowPartition.from_row_splits(row_splits, validate=False)
    self.assertEqual(rp.dtype, dtypes.int64)

    rp_row_splits = rp.row_splits()
    rp_nrows = rp.nrows()

    self.assertIs(rp_row_splits, row_splits)
    self.assertAllEqual(rp_nrows, 5)

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
    self.assertEqual(rt1.row_splits().dtype, dtypes.int64)
    self.assertEqual(rt2.row_splits().dtype, dtypes.int64)
    self.assertEqual(rt3.row_splits().dtype, dtypes.int32)
    self.assertEqual(rt4.row_splits().dtype, dtypes.int64)
    self.assertEqual(rt5.row_splits().dtype, dtypes.int32)

  def testFromRowSplitsWithEmptySplits(self):
    err_msg = 'row_splits tensor may not be empty'
    with self.assertRaisesRegex(ValueError, err_msg):
      RowPartition.from_row_splits([])

  def testFromRowStarts(self):
    nvals = constant_op.constant(7)
    row_starts = constant_op.constant([0, 2, 2, 5, 6], dtypes.int64)

    rp = RowPartition.from_row_starts(row_starts, nvals, validate=False)
    self.assertEqual(rp.dtype, dtypes.int64)

    rp_row_starts = rp.row_starts()
    rp_row_splits = rp.row_splits()
    rp_nrows = rp.nrows()

    self.assertAllEqual(rp_nrows, 5)
    self.assertAllEqual(rp_row_starts, row_starts)
    self.assertAllEqual(rp_row_splits, [0, 2, 2, 5, 6, 7])

  def testFromRowLimits(self):
    row_limits = constant_op.constant([2, 2, 5, 6, 7], dtypes.int64)

    rp = RowPartition.from_row_limits(row_limits, validate=False)
    self.assertEqual(rp.dtype, dtypes.int64)

    rp_row_limits = rp.row_limits()
    rp_row_splits = rp.row_splits()
    rp_nrows = rp.nrows()

    self.assertAllEqual(rp_nrows, 5)
    self.assertAllEqual(rp_row_limits, row_limits)
    self.assertAllEqual(rp_row_splits, [0, 2, 2, 5, 6, 7])

  def testFromRowLengths(self):
    row_lengths = constant_op.constant([2, 0, 3, 1, 1], dtypes.int64)

    rp = RowPartition.from_row_lengths(row_lengths, validate=False)
    self.assertEqual(rp.dtype, dtypes.int64)

    rp_row_lengths = rp.row_lengths()
    rp_nrows = rp.nrows()

    self.assertIs(rp_row_lengths, row_lengths)  # nrows
    self.assertAllEqual(rp_nrows, 5)
    self.assertAllEqual(rp_row_lengths, row_lengths)

  def testFromUniformRowLength(self):
    nvals = 16
    a1 = RowPartition.from_uniform_row_length(
        nvals=nvals, uniform_row_length=2)
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
    rt1 = RowPartition.from_uniform_row_length(
        nvals=nvals, uniform_row_length=3)
    const_nvals1 = self.evaluate(rt1.nvals())
    self.assertEqual(const_nvals1, 6)

  def testFromUniformRowLengthWithPlaceholders2(self):
    nvals = array_ops.placeholder_with_default(6, None)
    ph_rowlen = array_ops.placeholder_with_default(3, None)
    rt2 = RowPartition.from_uniform_row_length(
        nvals=nvals, uniform_row_length=ph_rowlen)
    const_nvals2 = self.evaluate(rt2.nvals())
    self.assertEqual(const_nvals2, 6)

  def testFromValueRowIdsWithBadNRows(self):
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(5, dtypes.int64)

    with self.assertRaisesRegex(ValueError, r'Expected nrows >= 0; got -2'):
      RowPartition.from_value_rowids(
          value_rowids=array_ops.placeholder_with_default(value_rowids, None),
          nrows=-2)

    with self.assertRaisesRegex(
        ValueError, r'Expected nrows >= value_rowids\[-1\] \+ 1; got nrows=2, '
        r'value_rowids\[-1\]=4'):
      RowPartition.from_value_rowids(value_rowids=value_rowids, nrows=2)

    with self.assertRaisesRegex(
        ValueError, r'Expected nrows >= value_rowids\[-1\] \+ 1; got nrows=4, '
        r'value_rowids\[-1\]=4'):
      RowPartition.from_value_rowids(value_rowids=value_rowids, nrows=4)

    with self.assertRaisesRegex(ValueError, r'Shape \(7, 1\) must have rank 1'):
      RowPartition.from_value_rowids(
          value_rowids=array_ops.expand_dims(value_rowids, 1), nrows=nrows)

    with self.assertRaisesRegex(ValueError, r'Shape \(1,\) must have rank 0'):
      RowPartition.from_value_rowids(
          value_rowids=value_rowids, nrows=array_ops.expand_dims(nrows, 0))

  # =============================================================================
  # RowPartition.__str__
  # =============================================================================
  def testRowPartitionStr(self):
    row_splits = [0, 2, 5, 6, 6, 7]
    rp = RowPartition.from_row_splits(row_splits, validate=False)
    if context.executing_eagerly():
      expected_repr = 'tf.RowPartition(row_splits=[0 2 5 6 6 7])'
    else:
      expected_repr = ('tf.RowPartition(row_splits='
                       'Tensor("RowPartitionFromRowSplits/row_splits:0", '
                       'shape=(6,), dtype=int64))')
    self.assertEqual(repr(rp), expected_repr)
    self.assertEqual(str(rp), expected_repr)

  def testRowPartitionStrUniformRowLength(self):
    rp = RowPartition.from_uniform_row_length(5, nvals=10, nrows=2)
    if context.executing_eagerly():
      expected_repr = ('tf.RowPartition(nrows=2, uniform_row_length=5)')
    else:
      expected_repr = (
          'tf.RowPartition(nrows='
          'Tensor("RowPartitionFromUniformRowLength/'
          'nrows:0", shape=(), dtype=int64), '
          'uniform_row_length=Tensor("RowPartitionFromUniformRowLength/'
          'uniform_row_length:0", shape=(), dtype=int64))')
    self.assertEqual(repr(rp), expected_repr)
    self.assertEqual(str(rp), expected_repr)

  @parameterized.parameters([
      # from_value_rowids
      {
          'descr': 'bad rank for value_rowids',
          'factory': RowPartition.from_value_rowids,
          'value_rowids': [[1, 2], [3, 4]],
          'nrows': 10,
      },
      {
          'descr': 'bad rank for nrows',
          'factory': RowPartition.from_value_rowids,
          'value_rowids': [1, 2, 3, 4],
          'nrows': [10],
      },
      {
          'descr': 'negative value_rowid',
          'factory': RowPartition.from_value_rowids,
          'value_rowids': [-5, 2, 3, 4],
          'nrows': 10,
      },
      {
          'descr': 'non-monotonic-increasing value_rowid',
          'factory': RowPartition.from_value_rowids,
          'value_rowids': [4, 3, 2, 1],
          'nrows': 10,
      },
      {
          'descr': 'value_rowid > nrows',
          'factory': RowPartition.from_value_rowids,
          'value_rowids': [1, 2, 3, 4],
          'nrows': 2,
      },
      # from_row_splits
      {
          'descr': 'bad rank for row_splits',
          'factory': RowPartition.from_row_splits,
          'row_splits': [[1, 2], [3, 4]],
      },
      {
          'descr': 'row_splits[0] != 0',
          'factory': RowPartition.from_row_splits,
          'row_splits': [2, 3, 4],
      },
      {
          'descr': 'non-monotonic-increasing row_splits',
          'factory': RowPartition.from_row_splits,
          'row_splits': [0, 3, 2, 4],
      },
      # from_row_lengths
      {
          'descr': 'bad rank for row_lengths',
          'factory': RowPartition.from_row_lengths,
          'row_lengths': [[1, 2], [1, 0]],
      },
      {
          'descr': 'negative row_lengths',
          'factory': RowPartition.from_row_lengths,
          'row_lengths': [3, -1, 2],
      },
      # from_row_starts
      {
          'descr': 'bad rank for row_starts',
          'factory': RowPartition.from_row_starts,
          'nvals': 2,
          'row_starts': [[1, 2], [3, 4]],
      },
      {
          'descr': 'row_starts[0] != 0',
          'factory': RowPartition.from_row_starts,
          'nvals': 5,
          'row_starts': [2, 3, 4],
      },
      {
          'descr': 'non-monotonic-increasing row_starts',
          'factory': RowPartition.from_row_starts,
          'nvals': 4,
          'row_starts': [0, 3, 2, 4],
      },
      {
          'descr': 'row_starts[0] > nvals',
          'factory': RowPartition.from_row_starts,
          'nvals': 4,
          'row_starts': [0, 2, 3, 5],
      },
      # from_row_limits
      {
          'descr': 'bad rank for row_limits',
          'factory': RowPartition.from_row_limits,
          'row_limits': [[1, 2], [3, 4]],
      },
      {
          'descr': 'row_limits[0] < 0',
          'factory': RowPartition.from_row_limits,
          'row_limits': [-1, 3, 4],
      },
      {
          'descr': 'non-monotonic-increasing row_limits',
          'factory': RowPartition.from_row_limits,
          'row_limits': [0, 3, 2, 4],
      },
      # from_uniform_row_length
      {
          'descr': 'rowlen * nrows != nvals (1)',
          'factory': RowPartition.from_uniform_row_length,
          'nvals': 5,
          'uniform_row_length': 3,
      },
      {
          'descr': 'rowlen * nrows != nvals (2)',
          'factory': RowPartition.from_uniform_row_length,
          'nvals': 5,
          'uniform_row_length': 6,
      },
      {
          'descr': 'rowlen * nrows != nvals (3)',
          'factory': RowPartition.from_uniform_row_length,
          'nvals': 6,
          'uniform_row_length': 3,
          'nrows': 3,
      },
      {
          'descr': 'rowlen must be a scalar',
          'factory': RowPartition.from_uniform_row_length,
          'nvals': 4,
          'uniform_row_length': [2],
      },
      {
          'descr': 'rowlen must be nonnegative',
          'factory': RowPartition.from_uniform_row_length,
          'nvals': 4,
          'uniform_row_length': -1,
      },
  ])
  def testFactoryValidation(self, descr, factory, **kwargs):
    # When input tensors have shape information, some of these errors will be
    # detected statically.
    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      partition = factory(**kwargs)
      self.evaluate(partition.row_splits())

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
        self.evaluate(partition.row_splits())

  @parameterized.named_parameters([
      ('FromRowSplits', lambda: RowPartition.from_row_splits([0, 2, 8]),
       ['row_splits']),
      ('FromRowLengths', lambda: RowPartition.from_row_lengths([3, 0, 8]),
       ['row_splits', 'row_lengths']),
      ('FromValueRowIds',
       lambda: RowPartition.from_value_rowids([0, 0, 3, 4, 4, 4]),
       ['row_splits', 'value_rowids', 'row_lengths', 'nrows']),
      ('FromRowStarts',
       lambda: RowPartition.from_row_starts([0, 3, 7], nvals=10),
       ['row_splits']),
      ('FromRowLimits', lambda: RowPartition.from_row_limits([3, 7, 10]),
       ['row_splits']),
  ])
  def testPrecomputedSplits(self, rp_factory, expected_encodings):
    rp = rp_factory()
    self.assertEqual(rp._has_precomputed_row_splits(),
                     'row_splits' in expected_encodings)
    self.assertEqual(rp._has_precomputed_row_lengths(),
                     'row_lengths' in expected_encodings)
    self.assertEqual(rp._has_precomputed_value_rowids(),
                     'value_rowids' in expected_encodings)
    self.assertEqual(rp._has_precomputed_nrows(), 'nrows' in expected_encodings)

  def testWithPrecomputedSplits(self):
    rp = RowPartition.from_row_splits([0, 2, 8])

    rp_with_row_splits = rp._with_precomputed_row_splits()
    self.assertTrue(rp_with_row_splits._has_precomputed_row_splits())

    self.assertFalse(rp._has_precomputed_row_lengths())
    rp_with_row_lengths = rp._with_precomputed_row_lengths()
    self.assertTrue(rp_with_row_lengths._has_precomputed_row_lengths())

    self.assertFalse(rp._has_precomputed_value_rowids())
    rp_with_value_rowids = rp._with_precomputed_value_rowids()
    self.assertTrue(rp_with_value_rowids._has_precomputed_value_rowids())

    self.assertFalse(rp._has_precomputed_nrows())
    rp_with_nrows = rp._with_precomputed_nrows()
    self.assertTrue(rp_with_nrows._has_precomputed_nrows())

    self.assertFalse(rp._has_precomputed_nvals())
    rp_with_nvals = rp._with_precomputed_nvals()
    self.assertTrue(rp_with_nvals._has_precomputed_nvals())

  @parameterized.named_parameters([
      dict(
          testcase_name='FromRowSplitsAndRowSplits',
          x=lambda: RowPartition.from_row_splits([0, 3, 8]),
          y=lambda: RowPartition.from_row_splits([0, 3, 8]),
          expected_encodings=['row_splits']),
      dict(
          testcase_name='FromRowSplitsAndUniformRowLength',
          x=lambda: RowPartition.from_row_splits([0, 3, 6]),
          y=lambda: RowPartition.from_uniform_row_length(3, nvals=6),
          expected_encodings=['row_splits', 'uniform_row_length', 'nrows']),
      dict(
          testcase_name='FromRowSplitsAndRowLengths',
          x=lambda: RowPartition.from_row_splits([0, 3, 8]),
          y=lambda: RowPartition.from_row_lengths([3, 5]),
          expected_encodings=['row_splits', 'row_lengths']),
      dict(
          testcase_name='FromRowSplitsAndValueRowIds',
          x=lambda: RowPartition.from_row_splits([0, 3, 8]),
          y=lambda: RowPartition.from_value_rowids([0, 0, 0, 1, 1, 1, 1, 1]),
          expected_encodings=[
              'row_splits', 'row_lengths', 'value_rowids', 'nrows'
          ]),
      dict(
          testcase_name='FromRowSplitsAndRowSplitsPlusNRows',
          x=lambda: RowPartition.from_row_splits([0, 3, 8]),
          y=lambda: RowPartition.from_row_splits([0, 3, 8]).
          _with_precomputed_nrows(),
          expected_encodings=['row_splits', 'nrows']),
  ])
  def testMergePrecomputedEncodings(self, x, y, expected_encodings):
    x = x()
    y = y()
    for validate in (True, False):
      result = x._merge_precomputed_encodings(y, validate)
      self.assertEqual(result._has_precomputed_row_splits(),
                       'row_splits' in expected_encodings)
      self.assertEqual(result._has_precomputed_row_lengths(),
                       'row_lengths' in expected_encodings)
      self.assertEqual(result._has_precomputed_value_rowids(),
                       'value_rowids' in expected_encodings)
      self.assertEqual(result._has_precomputed_nrows(),
                       'nrows' in expected_encodings)
      self.assertEqual(result.uniform_row_length() is not None,
                       'uniform_row_length' in expected_encodings)
      for r in (x, y):
        if (r._has_precomputed_row_splits() and
            result._has_precomputed_row_splits()):
          self.assertAllEqual(r.row_splits(), result.row_splits())
        if (r._has_precomputed_row_lengths() and
            result._has_precomputed_row_lengths()):
          self.assertAllEqual(r.row_lengths(), result.row_lengths())
        if (r._has_precomputed_value_rowids() and
            result._has_precomputed_value_rowids()):
          self.assertAllEqual(r.value_rowids(), result.value_rowids())
        if r._has_precomputed_nrows() and result._has_precomputed_nrows():
          self.assertAllEqual(r.nrows(), result.nrows())
        if (r.uniform_row_length() is not None and
            result.uniform_row_length() is not None):
          self.assertAllEqual(r.uniform_row_length(),
                              result.uniform_row_length())

  def testMergePrecomputedEncodingsFastPaths(self):
    # Same object: x gets returned as-is.
    x = RowPartition.from_row_splits([0, 3, 8, 8])
    self.assertIs(x._merge_precomputed_encodings(x), x)

    # Same encoding tensor objects: x gets returned as-is.
    y = RowPartition.from_row_splits(x.row_splits(), validate=False)
    self.assertIs(x._merge_precomputed_encodings(y), x)

  def testMergePrecomputedEncodingsWithMatchingTensors(self):
    # The encoding tensors for `a` are a superset of the encoding tensors
    # for `b`, and where they overlap, they the same tensor objects.
    a = RowPartition.from_value_rowids([0, 0, 3, 4, 4, 4])
    b = RowPartition.from_row_splits(a.row_splits(), validate=False)
    self.assertIs(a._merge_precomputed_encodings(b), a)
    self.assertIs(b._merge_precomputed_encodings(a), a)
    self.assertIsNot(a, b)

  @parameterized.named_parameters([
      dict(
          testcase_name='RowSplitMismatch',
          x=lambda: RowPartition.from_row_splits([0, 3, 8]),
          y=lambda: RowPartition.from_row_splits([0, 3, 8, 9]),
          message='incompatible row_splits'),
      dict(
          testcase_name='RowLengthMismatch',
          x=lambda: RowPartition.from_row_lengths([2, 0, 2]),
          y=lambda: RowPartition.from_row_lengths([2, 0, 2, 1]),
          message='incompatible row_splits'),  # row_splits is checked first
      dict(
          testcase_name='ValueRowIdMismatch',
          x=lambda: RowPartition.from_value_rowids([0, 3, 3, 4]),
          y=lambda: RowPartition.from_value_rowids([0, 3, 4]),
          message='incompatible value_rowids'),
  ])
  def testMergePrecomputedEncodingStaticErrors(self, x, y, message):
    if context.executing_eagerly():
      return
    # Errors that are caught by static shape checks.
    x = x()
    y = y()
    with self.assertRaisesRegex(ValueError, message):
      x._merge_precomputed_encodings(y).row_splits()
    with self.assertRaisesRegex(ValueError, message):
      y._merge_precomputed_encodings(x).row_splits()

  @parameterized.named_parameters([
      dict(
          testcase_name='NRowsMismatchAlt',
          x=lambda: RowPartition.from_uniform_row_length(5, nrows=4, nvals=20),
          y=lambda: RowPartition.from_uniform_row_length(5, nrows=3, nvals=15),
          message='incompatible nrows'),
      dict(
          testcase_name='UniformRowLengthMismatch',
          x=lambda: RowPartition.from_uniform_row_length(5, nvals=20),
          y=lambda: RowPartition.from_uniform_row_length(2, nvals=8),
          message='incompatible (nvals|uniform_row_length)'),
      dict(
          testcase_name='RowSplitMismatch',
          x=lambda: RowPartition.from_row_splits([0, 3, 8]),
          y=lambda: RowPartition.from_row_splits([0, 5, 8]),
          message='incompatible row_splits'),
      dict(
          testcase_name='RowLengthMismatch',
          x=lambda: RowPartition.from_row_lengths([2, 0, 2]),
          y=lambda: RowPartition.from_row_lengths([0, 0, 2]),
          message='incompatible (row_splits|nvals)'),
      dict(
          testcase_name='ValueRowIdMismatch',
          x=lambda: RowPartition.from_value_rowids([0, 3, 3]),
          y=lambda: RowPartition.from_value_rowids([0, 0, 3]),
          message='incompatible row_splits'),  # row_splits is checked first
  ])
  def testMergePrecomputedEncodingRuntimeErrors(self, x, y, message):
    # Errors that are caught by runtime value checks.
    x = x()
    y = y()
    with self.assertRaisesRegex(errors.InvalidArgumentError, message):
      self.evaluate(x._merge_precomputed_encodings(y).row_splits())
    with self.assertRaisesRegex(errors.InvalidArgumentError, message):
      self.evaluate(y._merge_precomputed_encodings(x).row_splits())

  @parameterized.named_parameters([
      # It throws the right error, but it still complains.
      dict(
          testcase_name='NRowsMismatch',
          x=lambda: RowPartition.from_uniform_row_length(5, nvals=20),
          y=lambda: RowPartition.from_uniform_row_length(5, nvals=15),
          message='incompatible nvals',
          emessage='incompatible nrows'),
  ])
  def testMergePrecomputedEncodingStaticErrors2(self, x, y,
                                                message, emessage):
    # Message error and type varies depending upon eager execution.
    x = x()
    y = y()

    error_type = errors_impl.InvalidArgumentError
    expected_message = emessage if context.executing_eagerly() else message
    with self.assertRaisesRegex(error_type, expected_message):
      self.evaluate(x._merge_precomputed_encodings(y).row_splits())
    with self.assertRaisesRegex(error_type, expected_message):
      self.evaluate(y._merge_precomputed_encodings(x).row_splits())

  @parameterized.named_parameters([
      dict(
          testcase_name='NoneSpecified',
          rp=(lambda: RowPartition.from_row_splits([0, 3, 8])),
          spec=RowPartitionSpec(nrows=None, nvals=None, dtype=dtypes.int64)),
      dict(
          testcase_name='NRowsSpecified',
          rp=(lambda: RowPartition.from_row_splits([0, 3, 8])),
          spec=RowPartitionSpec(nrows=2, nvals=None, dtype=dtypes.int64)),
      dict(
          testcase_name='NValsSpecified',
          rp=_get_specified_row_partition,
          spec=RowPartitionSpec(nrows=None, nvals=8, dtype=dtypes.int64))
  ])
  def testMergeWithSpecNoop(self, rp, spec):
    rp = rp()
    actual = rp._merge_with_spec(spec)
    self.assertAllEqual(actual.row_splits(), rp.row_splits())
    self.assertAllEqual(actual.static_nrows, rp.static_nrows)
    self.assertAllEqual(actual.static_nvals, rp.static_nvals)

  @parameterized.named_parameters([
      dict(
          testcase_name='NRowsNValsUpdated',
          rp=(lambda: RowPartition.from_row_splits([0, 3, 8])),
          spec=RowPartitionSpec(nrows=2, nvals=8, dtype=dtypes.int64),
          expected=_get_specified_row_partition),
      dict(
          testcase_name='NValsUpdated',
          rp=(lambda: RowPartition.from_row_splits([0, 3, 8])),
          spec=RowPartitionSpec(nrows=None, nvals=8, dtype=dtypes.int64),
          expected=_get_specified_row_partition)])
  def testMergeWithSpecUpdate(self, rp, spec, expected):
    rp = rp()
    expected = expected()
    actual = rp._merge_with_spec(spec)
    self.assertAllEqual(actual.row_splits(), expected.row_splits())
    self.assertAllEqual(actual.static_nrows, expected.static_nrows)
    self.assertAllEqual(actual.static_nvals, expected.static_nvals)

  @parameterized.named_parameters([
      dict(
          testcase_name='from_uniform_row_length',
          x=lambda: RowPartition.from_uniform_row_length(5, nvals=20),
          expected=True),
      dict(
          testcase_name='from_row_splits',
          x=lambda: RowPartition.from_row_splits([0, 3, 8]),
          expected=False),
      dict(
          testcase_name='from_row_lengths',
          x=lambda: RowPartition.from_row_lengths([2, 0, 2]),
          expected=False),
      dict(
          testcase_name='from_row_lengths_uniform',
          x=lambda: RowPartition.from_row_lengths([3, 3, 3]),
          expected=False),
  ])
  def testIsUniform(self, x, expected):
    x = x()
    self.assertEqual(expected, x.is_uniform())

  @parameterized.named_parameters([
      dict(
          testcase_name='doc_example',
          x=lambda: RowPartition.from_row_lengths([3, 2, 0, 2]),
          expected=[0, 1, 2, 0, 1, 0, 1]),
      dict(
          testcase_name='from_uniform_row_length',
          x=lambda: RowPartition.from_uniform_row_length(4, nvals=12),
          expected=[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]),
      dict(
          testcase_name='from_row_splits',
          x=lambda: RowPartition.from_row_splits([0, 3, 8]),
          expected=[0, 1, 2, 0, 1, 2, 3, 4]),
  ])
  def testOffsetsInRows(self, x, expected):
    x = x()
    actual = x.offsets_in_rows()
    self.assertAllEqual(expected, actual)

  def testFromUniformRowLengthBugConvertToTensor(self):
    # This originally failed to run because nrows was dtypes.int32. I think
    # we may need to consider the semantics of the type of a RowPartition
    # if preferred_dtype is unspecified. Also, looking at convert_to_tensor:
    # dtype specifies the type of the output.
    # preferred_dtype/dtype_hint is a suggestion, and dtype_hint is the new
    # name.
    nrows = constant_op.constant(3, dtype=dtypes.int32)
    nvals = constant_op.constant(12, dtype=dtypes.int64)
    row_length = constant_op.constant(4, dtype=dtypes.int64)
    rp = RowPartition.from_uniform_row_length(row_length, nvals=nvals,
                                              nrows=nrows, dtype=dtypes.int64)
    self.assertEqual(rp.nrows().dtype, dtypes.int64)

  def testFromUniformRowLengthNvalDynamic(self):
    # A key question is whether if nrows and uniform_row_length are known,
    # and nvals is given but not known statically, should we determine nvals?
    # TODO(martinz): Uncomment after nvals is fixed.
    # @def_function.function(
    #     input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    # def foo(nvals):
    #   rp = RowPartition.from_uniform_row_length(12, nvals=nvals, nrows=3)
    #   nval_output = tensor_util.constant_value(rp.nvals())
    #   self.assertEqual(nval_output, 36)
    # foo(constant_op.constant(36, dtype=dtypes.int32))
    pass

  def testFromUniformRowLengthNvalDynamicNoValidate(self):
    # A key question is whether if nrows and uniform_row_length are known,
    # and nvals is given but not known statically, should we determine nvals?
    # TODO(martinz): Uncomment after nvals is fixed.
    # @def_function.function(
    #     input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    # def foo(nvals):
    #   rp = RowPartition.from_uniform_row_length(12, nvals=nvals, nrows=3,
    #                                             validate=False)
    #   nval_output = tensor_util.constant_value(rp.nvals())
    #   self.assertEqual(nval_output, 36)
    # foo(constant_op.constant(36, dtype=dtypes.int32))
    pass

  def testFromUniformRowLengthNvalDynamicWrong(self):
    # A key question is whether if nrows and uniform_row_length are known,
    # and nvals is given but not known statically and WRONG,
    # what should we do? We add a check, but checks are only checked for
    # row_splits.
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    def foo(nvals):
      rp = RowPartition.from_uniform_row_length(12, nvals=nvals, nrows=3)
      return rp.nvals()

    with self.assertRaises(errors.InvalidArgumentError):
      nvals = foo(constant_op.constant(7, dtype=dtypes.int32))
      self.evaluate(nvals)

  def testFromUniformRowLengthNvalDynamicWrongRowSplits(self):
    # A key question is whether if nrows and uniform_row_length are known,
    # and nvals is given but not known statically and WRONG,
    # what should we do?
    # A key question is whether if nrows and uniform_row_length are known,
    # and nvals is given but not known statically and WRONG,
    # what should we do? We add a check, but checks are only checked for
    # row_splits.
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    def foo(nvals):
      rp = RowPartition.from_uniform_row_length(12, nvals=nvals, nrows=3)
      return rp.row_splits()

    with self.assertRaises(errors.InvalidArgumentError):
      rs = foo(constant_op.constant(7, dtype=dtypes.int32))
      self.evaluate(rs)

  def testFromUniformRowPartitionNrows(self):
    rp = RowPartition.from_uniform_row_length(3, nrows=4)
    self.assertAllEqual(4, rp.nrows())
    self.assertAllEqual(3, rp.uniform_row_length())
    self.assertAllEqual(12, rp.static_nvals)

  def testFromUniformRowPartitionNvalsStatic(self):
    rp = RowPartition.from_uniform_row_length(3, nvals=12)
    self.assertAllEqual(4, rp.static_nrows)
    self.assertAllEqual(3, rp.static_uniform_row_length)
    self.assertAllEqual(12, rp.static_nvals)

  def testFromUniformRowPartitionNvalsStaticNoValidate(self):
    rp = RowPartition.from_uniform_row_length(3, nrows=4, nvals=12,
                                              validate=False)
    self.assertAllEqual(4, rp.static_nrows)
    self.assertAllEqual(3, rp.static_uniform_row_length)
    self.assertAllEqual(12, rp.static_nvals)

  def testFromUniformRowPartitionNvalsIs(self):
    # TODO(martinz): Uncomment after nvals is fixed.
    # nvals = constant_op.constant(12)
    # rp = RowPartition.from_uniform_row_length(3, nvals=nvals)
    # self.assertIs(rp.nvals(), nvals)
    pass

  def testFromUniformRowPartitionRowStartsStatic(self):
    rp = RowPartition.from_row_starts([0, 3, 6], nvals=12)
    self.assertAllEqual(12, rp.static_nvals)

  def testStaticNrows(self):
    rp = RowPartition.from_row_splits([0, 3, 4, 5])
    static_nrows = rp.static_nrows
    self.assertIsInstance(static_nrows, int)
    self.assertAllEqual(3, static_nrows)

  def testStaticNrowsUnknown(self):
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    def foo(rs):
      rp = RowPartition.from_row_splits(rs)
      static_nrows = rp.static_nrows
      self.assertIsNone(static_nrows)
    foo(array_ops.constant([0, 3, 4, 5], dtype=dtypes.int32))


@test_util.run_all_in_graph_and_eager_modes
class RowPartitionSpecTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):

  def testDefaultConstruction(self):
    spec = RowPartitionSpec()
    self.assertIsNone(spec.nrows)
    self.assertIsNone(spec.nvals)
    self.assertIsNone(spec.uniform_row_length)
    self.assertEqual(spec.dtype, dtypes.int64)

  @parameterized.parameters([
      (None, None, None, dtypes.int64, None, None, None, dtypes.int64),
      (5, None, None, dtypes.int32, 5, None, None, dtypes.int32),
      (None, 20, None, dtypes.int64, None, 20, None, dtypes.int64),
      (None, None, 8, dtypes.int64, None, None, 8, dtypes.int64),
      (5, None, 8, dtypes.int64, 5, 40, 8, dtypes.int64),  # nvals inferred
      (None, 20, 4, dtypes.int32, 5, 20, 4, dtypes.int32),  # nrows inferred
      (0, None, None, dtypes.int32, 0, 0, None, dtypes.int32),  # nvals inferred
      (None, None, 0, dtypes.int32, None, 0, 0, dtypes.int32),  # nvals inferred
  ])  # pyformat: disable
  def testConstruction(self, nrows, nvals, uniform_row_length, dtype,
                       expected_nrows, expected_nvals,
                       expected_uniform_row_length, expected_dtype):
    spec = RowPartitionSpec(nrows, nvals, uniform_row_length, dtype)
    self.assertEqual(spec.nrows, expected_nrows)
    self.assertEqual(spec.nvals, expected_nvals)
    self.assertEqual(spec.uniform_row_length, expected_uniform_row_length)
    self.assertEqual(spec.dtype, expected_dtype)

  @parameterized.parameters([
      dict(dtype=dtypes.float32, error='dtype must be tf.int32 or tf.int64'),
      dict(nrows=0, nvals=5, error='.* not compatible .*'),
      dict(uniform_row_length=0, nvals=5, error='.* not compatible .*'),
      dict(nvals=11, uniform_row_length=5, error='.* not compatible .*'),
      dict(
          nrows=8, nvals=10, uniform_row_length=5,
          error='.* not compatible .*'),
  ])
  def testConstructionError(self,
                            nrows=None,
                            nvals=None,
                            uniform_row_length=None,
                            dtype=dtypes.int64,
                            error=None):
    with self.assertRaisesRegex(ValueError, error):
      RowPartitionSpec(nrows, nvals, uniform_row_length, dtype)

  @parameterized.parameters([
      RowPartitionSpec(),
      RowPartitionSpec(dtype=dtypes.int64),
      RowPartitionSpec(uniform_row_length=3)
  ])  # pyformat: disable

  def testDeepcopy(self, spec):
    spec = RowPartitionSpec()
    spec_b = copy.deepcopy(spec)
    self.assertEqual(repr(spec), repr(spec_b))

  def testValueType(self):
    spec = RowPartitionSpec()
    self.assertEqual(spec.value_type, RowPartition)

  @parameterized.parameters([
      dict(
          spec=RowPartitionSpec(),
          expected=(tensor_shape.TensorShape([None]),
                    tensor_shape.TensorShape([None]),
                    tensor_shape.TensorShape([None]), dtypes.int64)),
      dict(
          spec=RowPartitionSpec(dtype=dtypes.int32),
          expected=(tensor_shape.TensorShape([None]),
                    tensor_shape.TensorShape([None]),
                    tensor_shape.TensorShape([None]), dtypes.int32)),
      dict(
          spec=RowPartitionSpec(nrows=8, nvals=13),
          expected=(tensor_shape.TensorShape([8]),
                    tensor_shape.TensorShape([13]),
                    tensor_shape.TensorShape([None]), dtypes.int64)),
      dict(
          spec=RowPartitionSpec(nrows=8, uniform_row_length=2),
          expected=(
              tensor_shape.TensorShape([8]),
              tensor_shape.TensorShape([16]),  # inferred
              tensor_shape.TensorShape([2]),
              dtypes.int64)),
  ])
  def testSerialize(self, spec, expected):
    serialization = spec._serialize()
    # TensorShape has an unconventional definition of equality, so we can't use
    # assertEqual directly here.  But repr() is deterministic and lossless for
    # the expected values, so we can use that instead.
    self.assertEqual(repr(serialization), repr(expected))

  @parameterized.parameters([
      dict(
          spec=RowPartitionSpec(),
          expected=tensor_spec.TensorSpec([None], dtypes.int64)),
      dict(
          spec=RowPartitionSpec(dtype=dtypes.int32),
          expected=tensor_spec.TensorSpec([None], dtypes.int32)),
      dict(
          spec=RowPartitionSpec(nrows=17, dtype=dtypes.int32),
          expected=tensor_spec.TensorSpec([18], dtypes.int32)),
      dict(
          spec=RowPartitionSpec(nvals=10, uniform_row_length=2),
          expected=tensor_spec.TensorSpec([6], dtypes.int64)),  # inferred nrow
  ])
  def testComponentSpecs(self, spec, expected):
    self.assertEqual(spec._component_specs, expected)

  @parameterized.parameters([
      dict(
          rp_factory=lambda: RowPartition.from_row_splits([0, 3, 7]),
          components=[0, 3, 7]),
  ])
  def testToFromComponents(self, rp_factory, components):
    rp = rp_factory()
    spec = rp._type_spec
    actual_components = spec._to_components(rp)
    self.assertAllEqual(actual_components, components)
    rp_reconstructed = spec._from_components(actual_components)
    _assert_row_partition_equal(self, rp, rp_reconstructed)

  @parameterized.parameters([
      (RowPartitionSpec(), RowPartitionSpec()),
      (RowPartitionSpec(nrows=8), RowPartitionSpec(nrows=8)),
      (RowPartitionSpec(nrows=8), RowPartitionSpec(nrows=None)),
      (RowPartitionSpec(nvals=8), RowPartitionSpec(nvals=8)),
      (RowPartitionSpec(nvals=8), RowPartitionSpec(nvals=None)),
      (RowPartitionSpec(uniform_row_length=8),
       RowPartitionSpec(uniform_row_length=8)),
      (RowPartitionSpec(uniform_row_length=8),
       RowPartitionSpec(uniform_row_length=None)),
      (RowPartitionSpec(nvals=12), RowPartitionSpec(uniform_row_length=3)),
      (RowPartitionSpec(nrows=12), RowPartitionSpec(uniform_row_length=72)),
      (RowPartitionSpec(nrows=5), RowPartitionSpec(nvals=15)),
      (RowPartitionSpec(nvals=0), RowPartitionSpec(nrows=0)),
      (RowPartitionSpec(nvals=0), RowPartitionSpec(uniform_row_length=0)),
  ])
  def testIsCompatibleWith(self, spec1, spec2):
    self.assertTrue(spec1.is_compatible_with(spec2))

  @parameterized.parameters([
      (RowPartitionSpec(), RowPartitionSpec(dtype=dtypes.int32)),
      (RowPartitionSpec(nvals=5), RowPartitionSpec(uniform_row_length=3)),
      (RowPartitionSpec(nrows=7,
                        nvals=12), RowPartitionSpec(uniform_row_length=3)),
      (RowPartitionSpec(nvals=5), RowPartitionSpec(nrows=0)),
      (RowPartitionSpec(nvals=5), RowPartitionSpec(uniform_row_length=0)),
  ])
  def testIsNotCompatibleWith(self, spec1, spec2):
    self.assertFalse(spec1.is_compatible_with(spec2))

  @parameterized.parameters([
      dict(
          spec1=RowPartitionSpec(nrows=8, nvals=3, dtype=dtypes.int32),
          spec2=RowPartitionSpec(nrows=8, nvals=3, dtype=dtypes.int32),
          expected=RowPartitionSpec(nrows=8, nvals=3, dtype=dtypes.int32)),
      dict(
          spec1=RowPartitionSpec(nrows=8, nvals=None),
          spec2=RowPartitionSpec(nrows=None, nvals=8),
          expected=RowPartitionSpec(nrows=None, nvals=None)),
      dict(
          spec1=RowPartitionSpec(nrows=8, nvals=33),
          spec2=RowPartitionSpec(nrows=3, nvals=13),
          expected=RowPartitionSpec(nrows=None, nvals=None)),
      dict(
          spec1=RowPartitionSpec(nrows=12, uniform_row_length=3),
          spec2=RowPartitionSpec(nrows=3, uniform_row_length=3),
          expected=RowPartitionSpec(nrows=None, uniform_row_length=3)),
      dict(
          spec1=RowPartitionSpec(5, 35, 7),
          spec2=RowPartitionSpec(8, 80, 10),
          expected=RowPartitionSpec(None, None, None)),
  ])
  def testMostSpecificCompatibleType(self, spec1, spec2, expected):
    actual = spec1.most_specific_compatible_type(spec2)
    self.assertEqual(actual, expected)

  @parameterized.parameters([
      (RowPartitionSpec(), RowPartitionSpec(dtype=dtypes.int32)),
  ])
  def testMostSpecificCompatibleTypeError(self, spec1, spec2):
    with self.assertRaisesRegex(ValueError, 'No TypeSpec is compatible'):
      spec1.most_specific_compatible_type(spec2)

  def testNumRowsInt64(self):
    row_splits = array_ops.constant([0, 2, 3, 5], dtype=dtypes.int64)
    values = RowPartition.from_row_splits(row_splits)
    nrows = values.nrows()
    self.assertEqual(dtypes.int64, nrows.dtype)

  def testFromValue(self):
    self.assertEqual(
        RowPartitionSpec.from_value(RowPartition.from_row_splits([0, 2, 8, 8])),
        RowPartitionSpec(nrows=3))
    self.assertEqual(
        RowPartitionSpec.from_value(
            RowPartition.from_row_lengths([5, 3, 0, 2])),
        RowPartitionSpec(nrows=4))
    self.assertEqual(
        RowPartitionSpec.from_value(
            RowPartition.from_value_rowids([0, 2, 2, 8])),
        RowPartitionSpec(nrows=9, nvals=4))
    self.assertEqual(
        RowPartitionSpec.from_value(
            RowPartition.from_uniform_row_length(
                nvals=12, uniform_row_length=3)),
        RowPartitionSpec(nvals=12, uniform_row_length=3))

  @parameterized.parameters([
      dict(original=RowPartitionSpec(),
           dtype=dtypes.int32,
           expected=RowPartitionSpec(dtype=dtypes.int32)),
      dict(original=RowPartitionSpec(dtype=dtypes.int32),
           dtype=dtypes.int64,
           expected=RowPartitionSpec()),
      dict(original=RowPartitionSpec(nvals=20, nrows=4, uniform_row_length=5),
           dtype=dtypes.int32,
           expected=RowPartitionSpec(nvals=20, nrows=4, uniform_row_length=5,
                                     dtype=dtypes.int32)),
  ])
  def testWithDType(self, original, dtype, expected):
    actual = original.with_dtype(dtype)
    self.assertEqual(actual, expected)

  @parameterized.parameters([
      dict(a=RowPartitionSpec(),
           b=RowPartitionSpec(nrows=3, uniform_row_length=5),
           expected=RowPartitionSpec(nrows=3, uniform_row_length=5)),
      dict(a=RowPartitionSpec(nrows=3),
           b=RowPartitionSpec(uniform_row_length=5),
           expected=RowPartitionSpec(nrows=3, uniform_row_length=5)),
      dict(a=RowPartitionSpec(nvals=20),
           b=RowPartitionSpec(nrows=3),
           expected=RowPartitionSpec(nvals=20, nrows=3)),
      dict(a=RowPartitionSpec(nvals=20, dtype=dtypes.int32),
           b=RowPartitionSpec(nrows=3, dtype=dtypes.int32),
           expected=RowPartitionSpec(nvals=20, nrows=3, dtype=dtypes.int32)),
  ])
  def testMergeWith(self, a, b, expected):
    actual = a._merge_with(b)
    actual_rev = b._merge_with(a)
    self.assertEqual(actual, expected)
    self.assertEqual(actual_rev, expected)

  @parameterized.parameters([
      dict(a=RowPartitionSpec(nrows=3, nvals=10),
           b=RowPartitionSpec(uniform_row_length=5),
           error_type=ValueError,
           error_regex='Merging incompatible RowPartitionSpecs'),
      dict(a=RowPartitionSpec(uniform_row_length=5, dtype=dtypes.int32),
           b=RowPartitionSpec(uniform_row_length=5, dtype=dtypes.int64),
           error_type=ValueError,
           error_regex='Merging RowPartitionSpecs with incompatible dtypes'),
  ])
  def testMergeWithRaises(self, a, b, error_type, error_regex):
    with self.assertRaisesRegex(error_type, error_regex):
      a._merge_with(b)


def _assert_row_partition_equal(test_class, actual, expected):
  assert isinstance(test_class, test_util.TensorFlowTestCase)
  assert isinstance(actual, RowPartition)
  assert isinstance(expected, RowPartition)

  test_class.assertEqual(actual._has_precomputed_row_splits(),
                         expected._has_precomputed_row_splits())
  test_class.assertEqual(actual._has_precomputed_row_lengths(),
                         expected._has_precomputed_row_lengths())
  test_class.assertEqual(actual._has_precomputed_value_rowids(),
                         expected._has_precomputed_value_rowids())
  test_class.assertEqual(actual._has_precomputed_nrows(),
                         expected._has_precomputed_nrows())
  test_class.assertEqual(actual.uniform_row_length() is None,
                         expected.uniform_row_length() is None)

  if expected._has_precomputed_row_splits():
    test_class.assertAllEqual(actual.row_splits(), expected.row_splits())
  if expected._has_precomputed_row_lengths():
    test_class.assertAllEqual(actual.row_lengths(), expected.row_lengths())
  if expected._has_precomputed_value_rowids():
    test_class.assertAllEqual(actual.value_rowids(), expected.value_rowids())
  if expected._has_precomputed_nrows():
    test_class.assertAllEqual(actual.nrows(), expected.nrows())
  if expected.uniform_row_length() is not None:
    test_class.assertAllEqual(actual.uniform_row_length(),
                              expected.uniform_row_length())


if __name__ == '__main__':
  googletest.main()
