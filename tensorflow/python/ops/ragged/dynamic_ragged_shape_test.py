# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.ragged.dynamic_ragged_shape."""

from typing import Sequence, Union

from absl.testing import parameterized
import numpy as np

from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.dynamic_ragged_shape import _LayerBroadcaster
from tensorflow.python.ops.ragged.dynamic_ragged_shape import DynamicRaggedShape
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.ragged.row_partition import RowPartitionSpec
from tensorflow.python.platform import googletest


def _to_row_partitions_from_lengths(
    lengths: Sequence[Union[int, Sequence[int]]]) -> Sequence[RowPartition]:
  """Allow ragged and uniform shapes to be specified.

  For example, [2, [2,1], 2] represents a shape like:
  [[[0, 0], [0, 0]], [[0, 0]]]

  Args:
    lengths: a list of integers and lists of integers.

  Returns:
    a sequence of RowPartitions.
  """
  (result,
   _) = dynamic_ragged_shape._to_row_partitions_and_nvals_from_lengths(lengths)
  return result


def _to_ragged_tensor_from_lengths(
    values, lengths: Sequence[Union[int, Sequence[int]]]) -> RaggedTensor:
  """Specify a ragged tensor (or tensor) from lengths and values."""
  row_partitions = _to_row_partitions_from_lengths(lengths)
  values = constant_op.constant(values)
  if not row_partitions:
    return values
  return RaggedTensor._from_nested_row_partitions(values, row_partitions)


def _divides(a, b):
  return b % a == 0


def _next_prime(primes_so_far):
  first_candidate = 2
  if primes_so_far:
    first_candidate = primes_so_far[-1] + 1
  while True:
    if not any([_divides(x, first_candidate) for x in primes_so_far]):
      return first_candidate
    first_candidate = first_candidate + 1


def _lowest_primes(n):
  """Give the lowest n primes."""
  result = []
  for _ in range(n):
    result.append(_next_prime(result))
  return result


def _num_elements_of_lengths_with_rows(rows,
                                       lengths: Sequence[Union[int,
                                                               Sequence[int]]]):
  """Helper function for _num_elements_of_lengths."""
  if not lengths:
    return rows
  next_length = lengths[0]
  if isinstance(next_length, int):
    return _num_elements_of_lengths_with_rows(next_length * rows, lengths[1:])
  else:
    return _num_elements_of_lengths_with_rows(sum(next_length), lengths[1:])


def _num_elements_of_lengths(lengths: Sequence[Union[int, Sequence[int]]]):
  """Static version of DynamicRaggedShape.from_lengths(lengths)._num_elements()."""
  return _num_elements_of_lengths_with_rows(1, lengths)


def _to_prime_tensor_from_lengths(
    lengths: Sequence[Union[int, Sequence[int]]]) -> RaggedTensor:
  """Create a tensor of primes with the shape specified."""
  shape = DynamicRaggedShape.from_lengths(lengths)
  num_elements = _num_elements_of_lengths(lengths)
  return ragged_array_ops.ragged_reshape(_lowest_primes(num_elements), shape)


@test_util.run_all_in_graph_and_eager_modes
class DynamicRaggedShapeTest(test_util.TensorFlowTestCase,
                             parameterized.TestCase):

  def assertRowPartitionEq(self,
                           x: RowPartition,
                           y: RowPartition,
                           msg=None) -> None:
    self.assertAllEqual(x.row_splits(), y.row_splits(), msg=msg)

  def assertShapeEq(self,
                    x: DynamicRaggedShape,
                    y: DynamicRaggedShape,
                    msg=None) -> None:
    assert isinstance(x, DynamicRaggedShape)
    assert isinstance(y, DynamicRaggedShape)
    if msg is None:
      msg = ''
    self.assertLen(
        x.row_partitions, len(y.row_partitions), msg=msg + ': length unequal')
    for i in range(len(x.row_partitions)):
      x_dims = x.row_partitions[i]
      y_dims = y.row_partitions[i]
      self.assertRowPartitionEq(
          x_dims, y_dims, msg=msg + ': row_partition ' + str(i))
    self.assertAllEqual(
        x.inner_shape, y.inner_shape, msg=msg + ': shapes unequal')

  def assertLayerBroadcasterEq(self, x: _LayerBroadcaster,
                               y: _LayerBroadcaster) -> None:
    assert isinstance(x, _LayerBroadcaster)
    assert isinstance(y, _LayerBroadcaster)
    self.assertAllEqual(x.gather_index, y.gather_index)

  def assertBroadcasterEq(self, x: dynamic_ragged_shape._Broadcaster,
                          y: dynamic_ragged_shape._Broadcaster) -> None:
    assert isinstance(x, dynamic_ragged_shape._Broadcaster)
    assert isinstance(y, dynamic_ragged_shape._Broadcaster)
    self.assertShapeEq(x.source_shape, y.source_shape)
    self.assertShapeEq(x.target_shape, y.target_shape)
    self.assertLen(x._layer_broadcasters, len(y._layer_broadcasters))
    for x_layer, y_layer in zip(x._layer_broadcasters, y._layer_broadcasters):
      self.assertLayerBroadcasterEq(x_layer, y_layer)

  @parameterized.parameters([
      dict(value='x', row_partitions=[], inner_shape=()),
      dict(value=['a', 'b', 'c'], row_partitions=[], inner_shape=[3]),
      dict(
          value=[['a', 'b', 'c'], ['d', 'e', 'f']],
          row_partitions=(),
          inner_shape=[2, 3]),
      dict(
          value=[[['a', 'b', 'c'], ['d', 'e', 'f']]],
          row_partitions=(),
          inner_shape=[1, 2, 3]),
      dict(
          value=ragged_factory_ops.constant_value([['a', 'b', 'c'], ['d', 'e']],
                                                  ragged_rank=1),
          row_partitions=[[0, 3, 5]],
          inner_shape=[5]),
      dict(
          value=ragged_factory_ops.constant_value(
              [[['a', 'b', 'c'], ['d', 'e', 'f']]], ragged_rank=1),
          row_partitions=[[0, 2]],
          inner_shape=[2, 3]),
      dict(
          value=ragged_factory_ops.constant_value(
              [[[[1], [2]], [[3], [4]]], [[[5], [6]]]], ragged_rank=1),
          row_partitions=[[0, 2, 3]],
          inner_shape=[3, 2, 1]),
      dict(
          value=ragged_factory_ops.constant_value([[10, 20], [30]]),
          row_partitions=[[0, 2, 3]],
          inner_shape=[3]),
      # Docstring examples:
      dict(value=[[1, 2, 3], [4, 5, 6]], row_partitions=[], inner_shape=[2, 3]),
      dict(
          value=ragged_factory_ops.constant_value([[1, 2], [], [3, 4, 5]]),
          row_partitions=[[0, 2, 2, 5]],
          inner_shape=[5]),
      dict(
          value=ragged_factory_ops.constant_value([[[1, 2], [3, 4]], [[5, 6]]],
                                                  ragged_rank=1),
          row_partitions=[[0, 2, 3]],
          inner_shape=[3, 2]),
      dict(
          value=ragged_factory_ops.constant_value([[[1, 2], [3]], [[4, 5]]]),
          row_partitions=[[0, 2, 3], [0, 2, 3, 5]],
          inner_shape=[5]),
  ])
  def testFromTensor(self, value, row_partitions, inner_shape):
    shape = DynamicRaggedShape.from_tensor(value)
    row_partitions = [RowPartition.from_row_splits(x) for x in row_partitions]
    expected = DynamicRaggedShape(row_partitions, inner_shape)
    self.assertShapeEq(shape, expected)

  # pylint:disable=g-long-lambda
  @parameterized.parameters([
      # from_lengths           | row_partitions            | inner_shape
      # ---------------------- | --------------------------| -------------
      # []                     | []                        | []
      # [2, (3, 2)]            | [RP([3, 2])]              | [5]
      # [2, 2]                 | []                        | [2, 2]
      # [2, (3, 2), 7]         | [RP([3, 2])]              | [5, 7]
      # [2, (2, 2), 3]         | [RP([2, 2])]              | [4, 3]
      # [2, 2, 3]              | []                        | [2, 2, 3]
      # [2, (2, 1), (2, 0, 3)] | [RP(2, 1), RP([2, 0, 3])] | [5]

      dict(lengths=[], row_partitions=[], inner_shape=[]),
      dict(
          lengths=[2, (3, 2)],
          row_partitions=lambda: [RowPartition.from_row_lengths([3, 2])],
          inner_shape=[5]),
      dict(lengths=[2, 2], row_partitions=[], inner_shape=[2, 2]),
      dict(
          lengths=[2, (3, 2), 7],
          row_partitions=lambda: [RowPartition.from_row_lengths([3, 2])],
          inner_shape=[5, 7]),
      dict(
          lengths=[2, (2, 2), 3],
          row_partitions=lambda: [RowPartition.from_row_lengths([2, 2])],
          inner_shape=[4, 3]),
      dict(lengths=[2, 2, 3], row_partitions=[], inner_shape=[2, 2, 3]),
      dict(
          lengths=[2, (2, 1), (2, 0, 3)],
          row_partitions=lambda: [
              RowPartition.from_row_lengths([2, 1]),
              RowPartition.from_row_lengths([2, 0, 3])
          ],
          inner_shape=[5]),
      # from_lengths   | num_row    | row_partitions           | inner_shape
      #                : partitions :                          :
      # ---------------| -----------|--------------------------|------------
      # [2, (3, 2), 2] | 2          | [RP([3, 2]), URP(2, 10)] | [10]
      # [2, 2]         | 1          | [URP(2, 4)]              | [4]
      # [2, 2, 3]      | 0          | []                       | [2, 2, 3]
      # [2, 2, 3]      | 1          | [URP(2, 4)]              | [4, 3]
      # [2, 2, 3]      | 2          | [URP(2, 4), URP(3, 12)]  | [12]
      dict(lengths=[2, (3, 2), 2],
           num_row_partitions=2,
           row_partitions=lambda: [RowPartition.from_row_lengths([3, 2]),
                                   RowPartition.from_uniform_row_length(2, 10)],
           inner_shape=[10]),
      dict(lengths=[2, 2],
           num_row_partitions=1,
           row_partitions=lambda: [RowPartition.from_uniform_row_length(2, 4)],
           inner_shape=[4]),
      dict(lengths=[2, 2, 3],
           num_row_partitions=0,
           row_partitions=[],
           inner_shape=[2, 2, 3]),
      dict(lengths=[2, 2, 3],
           num_row_partitions=1,
           row_partitions=lambda: [RowPartition.from_uniform_row_length(2, 4)],
           inner_shape=[4, 3]),
      dict(lengths=[2, 2, 3],
           num_row_partitions=2,
           row_partitions=lambda: [RowPartition.from_uniform_row_length(2, 4),
                                   RowPartition.from_uniform_row_length(3, 12)],
           inner_shape=[12])
  ])
  def testFromLengths(self,
                      lengths,
                      row_partitions,
                      inner_shape,
                      num_row_partitions=None):
    if callable(row_partitions):
      row_partitions = row_partitions()
    shape = DynamicRaggedShape.from_lengths(
        lengths, num_row_partitions=num_row_partitions)
    expected = DynamicRaggedShape(row_partitions, inner_shape)
    self.assertShapeEq(shape, expected)

  @parameterized.parameters([
      dict(
          lengths=[2, (2, 1, 3)],
          num_row_partitions=1,
          msg='Shape not consistent'),
      dict(
          lengths=[2, 3],
          num_row_partitions=2,
          msg='num_row_partitions should be less than'),
      dict(
          lengths=[],
          num_row_partitions=3,
          msg='num_row_partitions==0 for a scalar shape'),
      dict(
          lengths=[(5, 3), 3],
          num_row_partitions='a',
          msg='num_row_partitions should be an int or None'),
      dict(
          lengths=[(5, 'a'), 3],
          num_row_partitions=0,
          msg='element of lengths should be int or tuple of ints'),
      dict(
          lengths=['a'],
          num_row_partitions=0,
          msg='element of lengths should be int or tuple of ints'),
      dict(lengths=7, num_row_partitions=0, msg='lengths should be a list')
  ])
  def testFromLengthsError(self, lengths, msg, num_row_partitions=None):
    with self.assertRaisesRegex(ValueError, msg):
      DynamicRaggedShape.from_lengths(
          lengths, num_row_partitions=num_row_partitions)

  def testGetItemSliceRankUnknownA(self):
    if not context.executing_eagerly():
      original_t = array_ops.placeholder_with_default(np.array([4, 5, 3]), None)
      sh = DynamicRaggedShape.from_tensor(original_t)
      known = sh[:1]
      self.assertIsNone(known.rank)

  def testGetItemSliceRankUnknownLong(self):
    if not context.executing_eagerly():
      original_t = array_ops.placeholder_with_default(np.array([4, 5, 3]), None)
      sh = DynamicRaggedShape.from_tensor(original_t)
      unknown = sh[:20]
      self.assertIsNone(unknown.rank)

  def testGetItemSliceRankKnownLong(self):
    if not context.executing_eagerly():
      original_t = constant_op.constant([4, 5, 3], dtypes.float32)
      sh = DynamicRaggedShape.from_tensor(original_t)
      unknown = sh[:20]
      self.assertEqual(unknown.rank, 1)

  def testGetBroadcaster(self):
    origin_shape = DynamicRaggedShape(
        [RowPartition.from_uniform_row_length(1, 3)], inner_shape=[3])
    dest_shape = DynamicRaggedShape(
        [RowPartition.from_uniform_row_length(2, 6)], inner_shape=[6])
    actual = dynamic_ragged_shape._get_broadcaster(origin_shape, dest_shape)
    expected = dynamic_ragged_shape._Broadcaster(origin_shape, dest_shape, [
        _LayerBroadcaster.from_gather_index([0, 1, 2]),
        _LayerBroadcaster.from_gather_index([0, 0, 1, 1, 2, 2])
    ])
    self.assertBroadcasterEq(actual, expected)

  def testGetBroadcaster2(self):
    origin_shape = DynamicRaggedShape([], inner_shape=[])
    dest_shape = DynamicRaggedShape([RowPartition.from_row_splits([0, 2, 3])],
                                    inner_shape=[3])
    actual = dynamic_ragged_shape._get_broadcaster(origin_shape, dest_shape)
    expected = dynamic_ragged_shape._Broadcaster(origin_shape, dest_shape, [])
    self.assertBroadcasterEq(actual, expected)

  @parameterized.parameters([
      dict(lengths=[2, 3], axis=0, expected=2),
      dict(lengths=[2, 3], axis=1, expected=6),
      dict(lengths=[2, 3], axis=-1, expected=6),
      dict(lengths=[2, 3], axis=-2, expected=2),
      dict(lengths=[2, 3, 4], axis=0, expected=2),
      dict(lengths=[2, 3, 4], axis=1, expected=6),
      dict(lengths=[2, 3, 4], axis=2, expected=24),
      dict(lengths=[2, 3, 4], axis=-1, expected=24),
      dict(lengths=[2, 3, 4], axis=-2, expected=6),
      dict(lengths=[2, 3, 4], axis=-3, expected=2),
      dict(lengths=[2, (2, 3), 7], axis=0, expected=2),
      dict(lengths=[2, (2, 3), 7], axis=1, expected=5),
      dict(lengths=[2, (2, 3), 7], axis=2, expected=35),
      dict(lengths=[2, (2, 3), 7], axis=-1, expected=35),
      dict(lengths=[2, (2, 3), 7], axis=-2, expected=5),
      dict(lengths=[2, (2, 3), 7], axis=-3, expected=2),
  ])
  def testNumSlicesInDimension(self, lengths, axis, expected):
    original = DynamicRaggedShape.from_lengths(lengths)
    actual = original._num_slices_in_dimension(axis)
    self.assertAllEqual(expected, actual)

  @parameterized.parameters([
      dict(
          lengths=[2, 3],
          axis=0.5,
          error_type=TypeError,
          error_regex='axis must be an integer'),
  ])
  def testNumSlicesInDimensionRaises(self, lengths, axis, error_type,
                                     error_regex):
    original = DynamicRaggedShape.from_lengths(lengths)
    with self.assertRaisesRegex(error_type, error_regex):
      original._num_slices_in_dimension(axis)

  @parameterized.parameters([
      dict(
          lengths=[2, (1, 2), 4],
          new_dense_rank=3,
          error_type=ValueError,
          error_regex='Cannot get an inner shape'),
      dict(
          lengths=[],
          new_dense_rank=3,
          error_type=ValueError,
          error_regex='old inner_rank cannot be zero'),
      dict(
          lengths=[2, 3],
          new_dense_rank=0,
          error_type=ValueError,
          error_regex='new_inner_rank cannot be zero'),
  ])
  def testAltInnerShapeRaises(self, lengths, new_dense_rank, error_type,
                              error_regex):
    original = DynamicRaggedShape.from_lengths(lengths)
    with self.assertRaisesRegex(error_type, error_regex):
      original._alt_inner_shape(new_dense_rank)

  @parameterized.parameters([
      dict(
          lengths=[2, (1, 2), 4], new_dense_rank=2, expected_inner_shape=[3,
                                                                          4]),
  ])
  def testAltInnerShape(self, lengths, new_dense_rank, expected_inner_shape):
    original = DynamicRaggedShape.from_lengths(lengths)
    actual = original._alt_inner_shape(new_dense_rank)
    self.assertAllEqual(actual, expected_inner_shape)

  def testWithNumRowPartitionsDynamic(self):
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([3], dtypes.int64)])
    def fun(x):
      shape = DynamicRaggedShape([
          RowPartition.from_row_lengths([1, 3], dtype=dtypes.int64),
          RowPartition.from_row_lengths([2, 3, 4, 5], dtype=dtypes.int64)
      ], x)
      result = shape._with_num_row_partitions(3)
      expected = DynamicRaggedShape([
          RowPartition.from_row_lengths([1, 3], dtype=dtypes.int64),
          RowPartition.from_row_lengths([2, 3, 4, 5], dtype=dtypes.int64),
          RowPartition.from_uniform_row_length(
              2, nrows=14, nvals=28, dtype=dtypes.int64)
      ], [14 * 2, 3])
      self.assertShapeEq(expected, result)
    fun(constant_op.constant([14, 2, 3], dtype=dtypes.int64))

  @parameterized.parameters([
      dict(
          lengths=[2],
          new_dense_rank=2,
          error_type=ValueError,
          error_regex='Cannot change inner_rank if'),
  ])
  def testWithDenseRankRaises(self, lengths, new_dense_rank, error_type,
                              error_regex):
    original = DynamicRaggedShape.from_lengths(lengths)
    with self.assertRaisesRegex(error_type, error_regex):
      original._with_inner_rank(new_dense_rank)

  @parameterized.parameters([
      dict(
          lengths=[2, (1, 2)],
          num_row_partitions=2,
          error_type=ValueError,
          error_regex='num_row_partitions must be less than rank'),
      dict(
          lengths=[2],
          num_row_partitions=-1,
          error_type=ValueError,
          error_regex='num_row_partitions must be nonnegative'),
      dict(
          lengths=[2],
          num_row_partitions=0.5,
          error_type=ValueError,
          error_regex='num_row_partitions must be an int'),
  ])
  def testWithNumRowPartitionsRaises(self, lengths, num_row_partitions,
                                     error_type, error_regex):
    original = DynamicRaggedShape.from_lengths(lengths)
    with self.assertRaisesRegex(error_type, error_regex):
      original._with_num_row_partitions(num_row_partitions)

  def testDimensionRaises(self):
    original = DynamicRaggedShape.from_lengths([2, (1, 2)])
    with self.assertRaisesRegex(TypeError, 'index should be an int'):
      # This error is not exposed directly to the end user.
      original._dimension(0.5)

  @parameterized.parameters([
      # The whole shape (num_row_partitions=0, start=negative, stop=really big)
      dict(lengths=[2, 3], s=slice(-1000, 100), expected_lengths=[2, 3]),
      # The whole shape (num_row_partitions=0, stop=really big)
      dict(lengths=[2, 3], s=slice(0, 100), expected_lengths=[2, 3]),
      # The whole shape (num_row_partitions=0, stop=None)
      dict(lengths=[2, 3], s=slice(0, None), expected_lengths=[2, 3]),
      # start = None, num_row_partitions=1, stop = 3 < rank = 4
      dict(
          lengths=[2, (1, 2), 3, 4],
          s=slice(None, 3),
          expected_lengths=[2, (1, 2), 3]),
      # start = 1, num_row_partitions=1, stop = 4, rank = 4
      dict(
          lengths=[2, 3, 3, 4],
          num_row_partitions=1,
          s=slice(1, 4),
          expected_lengths=[3, 3, 4]),
      # start = 1, num_row_partitions=1, stop = 3 < rank = 4
      dict(
          lengths=[2, 3, 3, 4],
          num_row_partitions=1,
          s=slice(1, 3),
          expected_lengths=[3, 3]),
      # start = 1, num_row_partitions=2, stop = 3 < rank = 4
      dict(
          lengths=[2, 3, 4, 3, 4],
          num_row_partitions=2,
          s=slice(1, 3),
          expected_lengths=[3, 4]),
      # start = 0, num_row_partitions=1, stop = 3 < rank = 4
      dict(
          lengths=[2, (1, 2), 3, 4],
          s=slice(0, 3),
          expected_lengths=[2, (1, 2), 3]),
      # start = 0, num_row_partitions=0, stop < rank
      dict(lengths=[2, 3, 4], s=slice(0, 2), expected_lengths=[2, 3]),
      # start=0 < stop=2 <= num_row_partitions
      dict(
          lengths=[2, (1, 2), (3, 4, 5)],
          s=slice(0, 2),
          expected_lengths=[2, (1, 2)]),
      # start=0 < stop=1 <= num_row_partitions
      dict(lengths=[2, (1, 2), (3, 4, 5)], s=slice(0, 1), expected_lengths=[2]),
      # Reversed indices, gives scalar shape.
      dict(lengths=[2, 3], s=slice(2, 0), expected_lengths=[]),
      # The whole shape (num_row_partitions=0)
      dict(lengths=[2, 3], s=slice(0, 2), expected_lengths=[2, 3]),
  ])
  def testGetItemSlice(self,
                       lengths,
                       s,
                       expected_lengths,
                       num_row_partitions=None):
    original = DynamicRaggedShape.from_lengths(lengths)
    if num_row_partitions is not None:
      original = original._with_num_row_partitions(num_row_partitions)
    expected = DynamicRaggedShape.from_lengths(expected_lengths)
    actual = original[s]
    self.assertShapeEq(expected, actual)

  @parameterized.parameters([
      dict(
          lengths=[2, (1, 2), 3, 4],
          index=0.5,
          error_type=TypeError,
          error_regex='Argument is not an int or a slice'),
      dict(
          lengths=[2, (1, 2), 3, 4],
          index=slice(0, 1, 2),
          error_type=IndexError,
          error_regex='Cannot stride through a shape'),
      dict(
          lengths=[2, (1, 2), 3, 4],
          index=1,
          error_type=ValueError,
          error_regex='Index 1 is not uniform'),
      dict(
          lengths=[2, 3, 3, 4],
          num_row_partitions=1,
          index=-20,
          error_type=IndexError,
          error_regex='Index must be non-negative'),
      dict(
          lengths=[2, 3, 3, 4],
          num_row_partitions=1,
          index=9,
          error_type=IndexError,
          error_regex='Index is too big'),
  ])
  def testGetItemRaisesStatic(self,
                              lengths,
                              index,
                              error_type,
                              error_regex,
                              num_row_partitions=None):
    original = DynamicRaggedShape.from_lengths(lengths)
    if num_row_partitions is not None:
      original = original._with_num_row_partitions(num_row_partitions)
    with self.assertRaisesRegex(error_type, error_regex):
      original[index]  # pylint: disable=pointless-statement

  def testBroadcastToAlt(self):
    origin = RaggedTensor.from_uniform_row_length([3, 4, 5],
                                                  uniform_row_length=1)
    expected = RaggedTensor.from_uniform_row_length([3, 3, 4, 4, 5, 5],
                                                    uniform_row_length=2)
    expected_shape = DynamicRaggedShape.from_tensor(expected)
    actual = dynamic_ragged_shape.broadcast_to(origin, expected_shape)
    self.assertAllEqual(actual, expected)

  @parameterized.parameters([
      dict(
          source_lengths=[3],
          target_lengths=[1, 3],
          target_num_row_partitions=1,
          expected_gather_indices=[[0, 1, 2]]),
      dict(  # BroadcastTensorTo4 broadcaster.
          source_lengths=[2, 3],
          target_lengths=[1, 2, 3],
          target_num_row_partitions=2,
          expected_gather_indices=[[0, 1], [0, 1, 2, 3, 4, 5]]),
      dict(  # raggedTensor1.
          source_lengths=[3, (1, 2, 1), 2, 2],
          source_num_row_partitions=3,
          target_lengths=[1, 1, 3, (1, 2, 1), 2, 2],
          target_num_row_partitions=5,
          expected_gather_indices=[[0, 1, 2], [0, 1, 2, 3],
                                   [0, 1, 2, 3, 4, 5, 6, 7],
                                   [
                                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                       13, 14, 15
                                   ]]),
  ])
  def testBroadcaster(self,
                      source_lengths,
                      target_lengths,
                      expected_gather_indices,
                      source_num_row_partitions=None,
                      target_num_row_partitions=None):
    source = DynamicRaggedShape.from_lengths(source_lengths)
    if source_num_row_partitions is not None:
      source = source._with_num_row_partitions(source_num_row_partitions)
    target = DynamicRaggedShape.from_lengths(target_lengths)
    if target_num_row_partitions is not None:
      target = target._with_num_row_partitions(target_num_row_partitions)

    expected_gather_indices = [
        _LayerBroadcaster.from_gather_index(x) for x in expected_gather_indices
    ]
    actual = dynamic_ragged_shape._get_broadcaster(source, target)
    expected = dynamic_ragged_shape._Broadcaster(source, target,
                                                 expected_gather_indices)
    self.assertBroadcasterEq(actual, expected)

  def testRaggedGradientSimple1(self):
    if context.executing_eagerly():
      return
    def func(x):
      rt1 = RaggedTensor.from_row_splits(
          values=x, row_splits=[0, 4, 7, 8], validate=False)
      rt2 = rt1 * [[10], [100], [1000]]
      return rt2.flat_values

    x = constant_op.constant([3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0])
    y = func(x)
    g = gradients_impl.gradients(ys=y, xs=x)[0]

    self.assertAllClose(ops.convert_to_tensor(g),
                        [10., 10., 10., 10., 100., 100., 100, 1000.])

  def testRaggedGradientSimple2(self):
    if context.executing_eagerly():
      return
    def func(x):
      rt1 = RaggedTensor._from_row_partition(
          x,
          RowPartition.from_row_splits(row_splits=[0, 4, 7, 8], validate=False))
      rt2 = rt1 * [[10], [100], [1000]]
      return rt2.flat_values

    x = constant_op.constant([3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0])
    y = func(x)
    g = gradients_impl.gradients(ys=y, xs=x)[0]

    self.assertAllClose(ops.convert_to_tensor(g),
                        [10., 10., 10., 10., 100., 100., 100, 1000.])

  def testRaggedGradientSimple3(self):
    if context.executing_eagerly():
      return
    def func(x):
      rt1 = RaggedTensor._from_row_partition(
          x,
          RowPartition.from_row_splits(row_splits=[0, 4, 7, 8],
                                       dtype=dtypes.int32, validate=False))
      rt2 = rt1 * [[10], [100], [1000]]
      return rt2.flat_values

    x = constant_op.constant([3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0])
    y = func(x)
    g = gradients_impl.gradients(ys=y, xs=x)[0]

    self.assertAllClose(ops.convert_to_tensor(g),
                        [10., 10., 10., 10., 100., 100., 100, 1000.])

  def testRaggedMul(self):
    if context.executing_eagerly():
      return
    x = constant_op.constant([3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0])
    rt1 = RaggedTensor._from_row_partition(
        x,
        RowPartition.from_row_splits(row_splits=[0, 4, 7, 8],
                                     dtype=dtypes.int64, validate=False))
    rt2 = rt1 * [[10], [100], [1000]]
    self.assertAllClose(rt2.flat_values,
                        [30.0, 10.0, 40.0, 10.0, 100.0, 0.0, 200.0, 1000.0])

  def testBroadcastToGradient(self):
    if context.executing_eagerly():
      return
    def func(x):
      target_shape = DynamicRaggedShape.from_row_partitions(
          [RowPartition.from_row_splits(row_splits=[0, 4, 7, 8])])

      rt = dynamic_ragged_shape.broadcast_to(x, target_shape)
      return rt.flat_values

    x = constant_op.constant([[3.0], [1.0], [4.0]])
    y = func(x)
    g = gradients_impl.gradients(ys=y, xs=x)[0]

    self.assertAllClose(g, [[4.], [3.], [1.]])

  def testBroadcastScalarToScalar(self):
    origin = constant_op.constant(b'x')
    expected = origin
    expected_shape = DynamicRaggedShape.from_tensor(expected)
    actual = dynamic_ragged_shape.broadcast_to(origin, expected_shape)
    self.assertAllEqual(actual, expected)

  @parameterized.parameters([
      dict(lengths=[2, 3], axis=0),
      dict(lengths=[2, 3], axis=1),
      dict(lengths=[2, (2, 3), 7, 4], num_row_partitions=2, axis=0),
      dict(lengths=[2, (2, 3), 7, 4], num_row_partitions=2, axis=2),
      dict(lengths=[2, (2, 3), 7, 4], num_row_partitions=2, axis=3),
  ])
  def testIsUniformTrue(self, lengths, axis, num_row_partitions=None):
    shape = DynamicRaggedShape.from_lengths(lengths)
    if num_row_partitions is not None:
      shape = shape._with_num_row_partitions(num_row_partitions)
    actual = shape.is_uniform(axis)
    self.assertTrue(actual)

  @parameterized.parameters([
      dict(lengths=[2, (2, 3), 7, 4], num_row_partitions=2, axis=1),
      dict(
          lengths=[2, (2, 3), 2, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 4],
          num_row_partitions=3,
          axis=3),
  ])
  def testIsUniformFalse(self, lengths, num_row_partitions, axis):
    shape = DynamicRaggedShape.from_lengths(lengths)._with_num_row_partitions(
        num_row_partitions)
    actual = shape.is_uniform(axis)
    self.assertFalse(actual)

  @parameterized.parameters([
      dict(
          lengths=[2, (2, 3), 7, 4],
          num_row_partitions=2,
          axis=10,
          error_type=IndexError,
          error_regex='Expected axis=10 < rank=4'),
      dict(
          lengths=[2, (2, 3), 7, 4],
          num_row_partitions=2,
          axis=-1,
          error_type=IndexError,
          error_regex='Negative axis values are not supported'),
      dict(
          lengths=[2, (2, 3), 7, 4],
          num_row_partitions=2,
          axis=0.5,
          error_type=TypeError,
          error_regex='axis must be an integer'),
  ])
  def testIsUniformRaises(self, lengths, num_row_partitions, axis, error_type,
                          error_regex):
    shape = DynamicRaggedShape.from_lengths(lengths)._with_num_row_partitions(
        num_row_partitions)
    with self.assertRaisesRegex(error_type, error_regex):
      shape.is_uniform(axis)

  @parameterized.parameters([
      dict(lengths=[2, 3], num_row_partitions_a=0, num_row_partitions_b=1),
      dict(
          lengths=[2, (2, 3), 7, 4],
          num_row_partitions_a=2,
          num_row_partitions_b=1),
      dict(
          lengths=[3, (2, 0, 1), 5],
          num_row_partitions_a=1,
          num_row_partitions_b=2)
  ])
  def testWithNumRowPartitions(self, lengths, num_row_partitions_a,
                               num_row_partitions_b):
    shape = DynamicRaggedShape.from_lengths(lengths)
    original_row_partitions = shape.num_row_partitions
    shape_a = shape._with_num_row_partitions(num_row_partitions_a)
    self.assertEqual(shape_a.num_row_partitions, num_row_partitions_a)
    shape_b = shape_a._with_num_row_partitions(num_row_partitions_b)
    self.assertEqual(shape_b.num_row_partitions, num_row_partitions_b)
    actual = shape_b._with_num_row_partitions(original_row_partitions)
    self.assertShapeEq(actual, shape)

  @parameterized.parameters([
      dict(
          lengths=[2, (2, 3), 7, 4], num_row_partitions=2, axis=-2, expected=7),
      dict(lengths=[2, (2, 3), 7, 4], num_row_partitions=2, axis=0, expected=2),
      dict(lengths=[2, (2, 3), 7, 4], num_row_partitions=2, axis=2, expected=7),
      dict(lengths=[2, (2, 3), 7, 4], num_row_partitions=2, axis=3, expected=4),
      dict(
          lengths=[2, (2, 3), 7, 4, 3],
          num_row_partitions=2,
          axis=4,
          expected=3),
      dict(lengths=[3], axis=0, expected=3),
      dict(lengths=[3, 4, 5], axis=0, expected=3),
      dict(lengths=[3, 4, 5], axis=1, expected=4),
      dict(lengths=[3, 4, 5], axis=2, expected=5),
  ])
  def testGetItem(self, lengths, axis, expected, num_row_partitions=None):
    shape = DynamicRaggedShape.from_lengths(lengths)
    if num_row_partitions is not None:
      shape = shape._with_num_row_partitions(num_row_partitions)
    actual = shape[axis]
    self.assertAllEqual(actual, expected)

  def testNumElements(self):
    shape = DynamicRaggedShape.from_lengths([2, 3, 4,
                                             5])._with_num_row_partitions(2)
    self.assertAllEqual(shape._num_elements(), 120)

  def test_to_row_partitions_from_lengths(self):
    # Testing the test.
    actual = _to_row_partitions_from_lengths([1, 2, 3])
    expected = [
        RowPartition.from_row_splits([0, 2]),
        RowPartition.from_row_splits([0, 3, 6])
    ]
    self.assertRowPartitionEq(actual[0], expected[0])
    self.assertRowPartitionEq(actual[1], expected[1])

  @parameterized.parameters([
      dict(
          origin=b'x',
          expected_lengths=[2, (1, 2)],
          expected=[[b'x'], [b'x', b'x']]),
      dict(
          origin=b'x',
          expected_lengths=[1, 1, 1],
          expected_num_row_partitions=2,
          expected=[[[b'x']]]),
      dict(
          origin=[b'a', b'b', b'c'],
          expected_lengths=[3],
          expected=[b'a', b'b', b'c']),
      dict(
          origin=[b'a', b'b', b'c'],
          expected_lengths=[1, 1, 3],
          expected_num_row_partitions=2,
          expected=[[[b'a', b'b', b'c']]]),
      dict(
          origin=[[b'a', b'b', b'c'], [b'd', b'e', b'f']],
          expected_lengths=[1, 2, 3],
          expected_num_row_partitions=2,
          expected=[[[b'a', b'b', b'c'], [b'd', b'e', b'f']]]),
  ])
  def testBroadcastTensorTo(self,
                            origin,
                            expected_lengths,
                            expected,
                            expected_num_row_partitions=None):
    origin = constant_op.constant(origin)
    expected_shape = DynamicRaggedShape.from_lengths(expected_lengths)
    if expected_num_row_partitions is not None:
      expected_shape = expected_shape._with_num_row_partitions(
          expected_num_row_partitions)
    expected = ragged_factory_ops.constant_value(expected)
    actual = dynamic_ragged_shape.broadcast_to(origin, expected_shape)
    self.assertAllEqual(actual, expected)

  def testBroadcastFlatValues(self):
    origin_lengths = [3, (1, 2, 1), 2, 2]
    dest_lengths = [1, 1, 3, (1, 2, 1), 2, 2]
    origin_values = constant_op.constant([
        b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i', b'j', b'k', b'l',
        b'm', b'n', b'o', b'p'
    ])
    origin_shape = DynamicRaggedShape.from_lengths(
        origin_lengths)._with_num_row_partitions(3)
    dest_shape = DynamicRaggedShape.from_lengths(
        dest_lengths)._with_num_row_partitions(5)

    broadcaster = dynamic_ragged_shape._get_broadcaster(origin_shape,
                                                        dest_shape)

    actual = broadcaster.broadcast_flat_values(origin_values)

    self.assertAllEqual(origin_values, actual)

  @parameterized.parameters([
      dict(
          origin_lengths=[3],
          origin_values=[b'a', b'b', b'c'],
          expected_lengths=[2],
          expected_values=[[b'a', b'b', b'c'], [b'a', b'b', b'c']]),
      dict(
          origin_lengths=[3, (3, 2, 4)],
          origin_values=[7, 4, 5, 6, 1, 2, 3, 7, 89],
          expected_lengths=[3, (3, 2, 4)],
          expected_values=[7, 4, 5, 6, 1, 2, 3, 7, 89]),
      dict(
          origin_lengths=[3, (3, 2, 4)],
          origin_values=[7, 4, 5, 6, 1, 2, 3, 7, 89],
          expected_lengths=[1, 3, (3, 2, 4)],
          expected_values=[7, 4, 5, 6, 1, 2, 3, 7, 89]),
      dict(
          origin_lengths=[3, (3, 2, 4)],
          origin_values=[7, 4, 5, 6, 1, 2, 3, 7, 89],
          expected_lengths=[1, 1, 3, (3, 2, 4)],
          expected_values=[7, 4, 5, 6, 1, 2, 3, 7, 89]),
      # Broadcast [1, 2, (1, 2)] to [2, 2, (1, 2, 1, 2)]
      dict(
          origin_lengths=[1, 2, (1, 2)],
          origin_values=[2, 3, 5],
          expected_lengths=[2, 2, (1, 2, 1, 2)],
          expected_values=[2, 3, 5, 2, 3, 5]),
      # Broadcast [2, 1, (1, 2)] to [2, 2, (1, 1, 2, 2)] (NEW)
      dict(
          origin_lengths=[2, 1, (1, 2)],
          origin_values=[2, 3, 5],
          expected_lengths=[2, 2, (1, 1, 2, 2)],
          expected_values=[2, 2, 3, 5, 3, 5]),
      dict(
          origin_lengths=[2, 1, 1],
          origin_values=[2, 3],  # [[[2]], [[3]]]
          expected_lengths=[2, 1, (3, 3)],
          expected_values=[2, 2, 2, 3, 3, 3]),
      dict(
          origin_lengths=[3],
          origin_values=[b'a', b'b', b'c'],
          expected_lengths=[4, 2, 3],
          expected_values=[
              b'a', b'b', b'c', b'a', b'b', b'c', b'a', b'b', b'c', b'a', b'b',
              b'c', b'a', b'b', b'c', b'a', b'b', b'c', b'a', b'b', b'c', b'a',
              b'b', b'c'
          ]),
      dict(
          origin_lengths=[2, 3],
          origin_values=[b'a', b'b', b'c', b'a', b'b', b'c'],
          expected_lengths=[4, 2, 3],
          expected_values=[
              b'a', b'b', b'c', b'a', b'b', b'c', b'a', b'b', b'c', b'a', b'b',
              b'c', b'a', b'b', b'c', b'a', b'b', b'c', b'a', b'b', b'c', b'a',
              b'b', b'c'
          ]),
      dict(
          origin_lengths=[3, (1, 2, 1), 2, 2],
          origin_values=[
              b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i', b'j', b'k',
              b'l', b'm', b'n', b'o', b'p'
          ],
          expected_lengths=[1, 1, 3, (1, 2, 1), 2, 2],
          expected_values=[
              b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i', b'j', b'k',
              b'l', b'm', b'n', b'o', b'p'
          ]),
      dict(
          origin_lengths=[3, (1, 2, 1), 2, 2],
          origin_values=[7, 4, 5, 6, 1, 2, 3, 7, 7, 4, 5, 6, 1, 2, 3, 7],
          expected_lengths=[1, 1, 3, (1, 2, 1), 2, 2],
          expected_values=[7, 4, 5, 6, 1, 2, 3, 7, 7, 4, 5, 6, 1, 2, 3, 7],
      ),
  ])
  def testBroadcastRaggedTo(self, origin_lengths, origin_values,
                            expected_lengths, expected_values):
    origin = _to_ragged_tensor_from_lengths(origin_values, origin_lengths)
    expected = _to_ragged_tensor_from_lengths(expected_values, expected_lengths)
    expected_shape = DynamicRaggedShape.from_tensor(expected)
    actual = dynamic_ragged_shape.broadcast_to(origin, expected_shape)
    self.assertAllEqual(actual, expected)

  def testDynamicRaggedShapeFromTensor2(self):
    raw_rt = [[[[7, 4], [5, 6]], [[1, 2], [3, 7]]], [[[7, 4], [5, 6]]],
              [[[1, 2], [3, 7]]]]
    raw_rt = ragged_factory_ops.constant_value(raw_rt)
    actual_shape = DynamicRaggedShape.from_tensor(raw_rt)
    expected_shape = DynamicRaggedShape.from_lengths(
        [3, (2, 1, 1), 2, 2])._with_num_row_partitions(3)
    self.assertShapeEq(actual_shape, expected_shape)

  # pylint: disable=g-long-lambda
  @parameterized.parameters([
      # A row partition as opposed to a list of row partitions.
      dict(
          row_partitions=lambda: RowPartition.from_row_splits([0, 2, 3]),
          inner_shape=lambda: [4],
          error_type=TypeError,
          error_regex='row_partitions should be'),
      # A list of lists of integers for row_partitions.
      dict(
          row_partitions=lambda: [[0, 2, 3]],
          inner_shape=lambda: [4],
          error_type=TypeError,
          error_regex='row_partitions contains'),
      # nvals and nrows don't match (3 != 6) statically
      dict(
          row_partitions=lambda: [  # pylint: disable=g-long-lambda
              RowPartition.from_value_rowids([0, 2, 4], nrows=5),
              RowPartition.from_value_rowids([0, 2, 5], nrows=6)
          ],
          inner_shape=lambda: [3],
          validate=False,
          error_type=ValueError,
          error_regex='RowPartitions in DynamicRaggedShape do not'),
      # nvals and inner_shape[0] don't match (3 != 6) statically
      dict(
          row_partitions=lambda: [
              RowPartition.from_value_rowids([0, 2, 4], nrows=5),
          ],
          inner_shape=lambda: [6],
          validate=False,
          error_type=ValueError,
          error_regex='Last row partition does not match inner_shape.'),
  ])
  def testConstructorRaisesStatic(self,
                                  row_partitions,
                                  inner_shape,
                                  error_type,
                                  error_regex,
                                  validate=False,
                                  dtype=None):
    row_partitions = row_partitions()
    inner_shape = inner_shape()
    with self.assertRaisesRegex(error_type, error_regex):
      DynamicRaggedShape(
          row_partitions, inner_shape, dtype=dtype, validate=validate)

  def testConstructorStaticOK(self):
    row_partitions = [
        RowPartition.from_value_rowids([0, 2, 4], nrows=5),
        RowPartition.from_value_rowids([0, 1, 2], nrows=3)
    ]
    inner_shape = [3]
    rts = DynamicRaggedShape(row_partitions, inner_shape, validate=True)
    static_inner_shape = tensor_util.constant_value(rts.inner_shape)
    static_valid_rowids0 = tensor_util.constant_value(
        rts.row_partitions[0].value_rowids())
    static_valid_rowids1 = tensor_util.constant_value(
        rts.row_partitions[1].value_rowids())
    self.assertAllEqual(static_inner_shape, [3])
    self.assertAllEqual(static_valid_rowids0, [0, 2, 4])
    self.assertAllEqual(static_valid_rowids1, [0, 1, 2])

  def testConstructorWithStaticInnerShape(self):
    row_partitions = [
        RowPartition.from_value_rowids([0, 2, 4], nrows=5),
        RowPartition.from_value_rowids([0, 1, 2], nrows=3)
    ]
    inner_shape = [3]
    rts = DynamicRaggedShape(row_partitions, inner_shape, validate=True,
                             static_inner_shape=[3])
    static_inner_shape = tensor_util.constant_value(rts.inner_shape)
    static_valid_rowids0 = tensor_util.constant_value(
        rts.row_partitions[0].value_rowids())
    static_valid_rowids1 = tensor_util.constant_value(
        rts.row_partitions[1].value_rowids())
    self.assertAllEqual(static_inner_shape, [3])
    self.assertAllEqual(static_valid_rowids0, [0, 2, 4])
    self.assertAllEqual(static_valid_rowids1, [0, 1, 2])

  def testZeros(self):
    shape_x = DynamicRaggedShape.from_lengths([3, (1, 3, 2), 4])
    foo = ragged_array_ops.zeros(shape_x)
    self.assertShapeEq(shape_x, DynamicRaggedShape.from_tensor(foo))
    self.assertAllEqual(array_ops.zeros([6, 4]), foo.flat_values)

  def testOnes(self):
    shape_x = DynamicRaggedShape.from_lengths([3, (1, 3, 2), 4])
    foo = ragged_array_ops.ones(shape_x)
    self.assertShapeEq(shape_x, DynamicRaggedShape.from_tensor(foo))
    self.assertAllEqual(array_ops.ones([6, 4]), foo.flat_values)

  def testReshapeTensor(self):
    foo = array_ops.zeros([3, 2, 4])
    shape_b = DynamicRaggedShape.from_lengths([3, (3, 2, 1), 4])
    result = ragged_array_ops.ragged_reshape(foo, shape_b)
    self.assertShapeEq(shape_b, DynamicRaggedShape.from_tensor(result))
    self.assertAllEqual(array_ops.zeros([6, 4]), result.flat_values)

  def test_reshape_ragged_tensor(self):
    shape_x = DynamicRaggedShape.from_lengths([3, (1, 3, 2), 4])
    foo = ragged_array_ops.zeros(shape_x)
    shape_b = DynamicRaggedShape.from_lengths([3, (3, 2, 1), 4])
    result = ragged_array_ops.ragged_reshape(foo, shape_b)
    self.assertShapeEq(shape_b, DynamicRaggedShape.from_tensor(result))
    self.assertAllEqual(array_ops.zeros([6, 4]), result.flat_values)

  @parameterized.parameters([
      dict(
          lengths_a=[3, (1, 4, 2)],
          lengths_b=[3, (1, 4, 2)],
          lengths_e=[3, (1, 4, 2)]),
      dict(
          lengths_a=[1, 2, (1, 4)],
          lengths_b=[3, 2, (1, 4, 1, 4, 1, 4)],
          lengths_e=[3, 2, (1, 4, 1, 4, 1, 4)]),
      dict(
          lengths_a=[1, 1],
          num_row_partitions_a=1,
          lengths_b=[3, 5],
          num_row_partitions_b=1,
          lengths_e=[3, 5],
          num_row_partitions_e=1),
      dict(lengths_a=[1, 4, 5], lengths_b=[3, 1, 1], lengths_e=[3, 4, 5]),
      dict(lengths_a=[3], lengths_b=[4, 2, 1], lengths_e=[4, 2, 3]),
      dict(lengths_a=[2, 3], lengths_b=[4, 2, 1], lengths_e=[4, 2, 3]),
      # Outermost dimension-both partitioned
      # Also, neither has uniform_row_length
      dict(
          lengths_a=[2, (1, 3), 1],
          lengths_b=[2, (1, 3), (3, 4, 5, 6)],
          lengths_e=[2, (1, 3), (3, 4, 5, 6)]),
      # Outermost dimension-Only one is partitioned
      # Also, partitioned dimension doesn't have uniform_row_length
      dict(
          lengths_a=[2, 1, 5],
          lengths_b=[2, (1, 3), 5],
          num_row_partitions_b=2,
          lengths_e=[2, (1, 3), 5],
          num_row_partitions_e=2),

      # Cover [5, R], [1, 5, R]
      dict(
          lengths_a=[5, (1, 2, 0, 3, 1)],
          lengths_b=[1, 5, (1, 2, 0, 3, 1)],
          lengths_e=[1, 5, (1, 2, 0, 3, 1)]),
      # When two uniform row lengths are equal
      dict(
          lengths_a=[1, 5],
          num_row_partitions_a=1,
          lengths_b=[3, 5],
          num_row_partitions_b=1,
          lengths_e=[3, 5],
          num_row_partitions_e=1),
      # Dense + Partitioned dimension has uniform_row_length
      # [1, 3, [5, 1, 6]] and DENSE [2, 1, 1] -> [2, 3, [5, 1, 6, 5, 1, 6]]
      dict(
          lengths_a=[1, 3, (5, 1, 6)],
          lengths_b=[2, 1, 1],
          lengths_e=[2, 3, (5, 1, 6, 5, 1, 6)]),
      # Both partitioned; one has uniform_row_length
      # (uniform_row_length [2,1,1]) and [2,[1,3],[3,4,5,6]]
      dict(
          lengths_a=[2, 1, 1],
          num_row_partitions_a=2,
          lengths_b=[2, (1, 3), (3, 4, 5, 6)],
          lengths_e=[2, (1, 3), (3, 4, 5, 6)]),
      # When broadcasting uniform_row_length to uniform_row_length.
      # Also, both have uniform_row_length
      dict(
          lengths_a=[3, 1, 5],
          num_row_partitions_a=2,
          lengths_b=[3, 4, 5],
          num_row_partitions_b=2,
          lengths_e=[3, 4, 5],
          num_row_partitions_e=2),
      # When broadcasting above a U_R_L
      # [2,1, 5] and [2, [1,3], 5] -> [2, [1,3], 5]
      dict(
          lengths_a=[2, 1, 5],
          num_row_partitions_a=2,
          lengths_b=[2, (1, 3), 5],
          num_row_partitions_b=2,
          lengths_e=[2, (1, 3), 5],
          num_row_partitions_e=2),
      # What if the larger-dimensional shape has uniform_row_length on the
      # matching dim, but has larger dimensions above
      # ([3,1,5],[15]) vs ([2,1],[2]))
      dict(
          lengths_a=[3, 1, 5],
          num_row_partitions_a=2,
          lengths_b=[2, 1],
          num_row_partitions_b=1,
          lengths_e=[3, 2, 5],
          num_row_partitions_e=2),
      # Inner non-ragged dimensions
      # Can delegate to dense broadcast operations.
      # Implementation detail: not testable.
      # ([2, [1,2]],[3,2,1]) and ([2,1],[2,1,3])
      dict(
          lengths_a=[2, (1, 2), 2, 1],
          lengths_b=[2, 1, 1, 3],
          num_row_partitions_b=1,
          lengths_e=[2, (1, 2), 2, 3],
      ),
  ])
  def testBroadcastDynamicShapeExtended(self,
                                        lengths_a,
                                        lengths_b,
                                        lengths_e,
                                        num_row_partitions_a=None,
                                        num_row_partitions_b=None,
                                        num_row_partitions_e=None):
    # This test is predicated on the fact that broadcast_to is correct.
    # Thus, it tests:
    # Whether the shape generated is correct.
    # Whether broadcasting is the same as broadcast_to.
    # Instead of specifying values, it just uses primes.
    shape_a = DynamicRaggedShape.from_lengths(lengths_a)
    if num_row_partitions_a is not None:
      shape_a = shape_a._with_num_row_partitions(num_row_partitions_a)
    shape_b = DynamicRaggedShape.from_lengths(lengths_b)
    if num_row_partitions_b is not None:
      shape_b = shape_b._with_num_row_partitions(num_row_partitions_b)
    shape_e = DynamicRaggedShape.from_lengths(lengths_e)
    if num_row_partitions_e is not None:
      shape_e = shape_e._with_num_row_partitions(num_row_partitions_e)

    [actual, bc_a, bc_b
    ] = dynamic_ragged_shape.broadcast_dynamic_shape_extended(shape_a, shape_b)
    [actual_rev, bc_b_rev, bc_a_rev
    ] = dynamic_ragged_shape.broadcast_dynamic_shape_extended(shape_b, shape_a)
    self.assertShapeEq(actual, shape_e)
    self.assertShapeEq(actual_rev, shape_e)

    rt_a = ragged_array_ops.ragged_reshape(
        _lowest_primes(_num_elements_of_lengths(lengths_a)), shape_a)
    bc_a_actual = bc_a.broadcast(rt_a)
    bc_a_actual_rev = bc_a_rev.broadcast(rt_a)
    bc_a_expected = dynamic_ragged_shape.broadcast_to(rt_a, shape_e)
    self.assertAllEqual(bc_a_expected, bc_a_actual)
    self.assertAllEqual(bc_a_expected, bc_a_actual_rev)

    rt_b = ragged_array_ops.ragged_reshape(
        _lowest_primes(_num_elements_of_lengths(lengths_b)), shape_b)
    bc_b_expected = dynamic_ragged_shape.broadcast_to(rt_b, shape_e)
    bc_b_actual = bc_b.broadcast(rt_b)
    bc_b_actual_rev = bc_b_rev.broadcast(rt_b)
    self.assertAllEqual(bc_b_expected, bc_b_actual)
    self.assertAllEqual(bc_b_expected, bc_b_actual_rev)

  @parameterized.parameters([
      dict(
          lengths=[3, (1, 4, 2)],
          dense_rank=1,
          lengths_e=[3, (1, 4, 2)],
      ),
      dict(
          lengths=[3, (1, 4, 2), 5],
          dense_rank=2,
          lengths_e=[3, (1, 4, 2), 5],
      ),
      dict(
          lengths=[3],
          dense_rank=1,
          lengths_e=[3],
      ),
  ])
  def testWithDenseRank(self, lengths, dense_rank, lengths_e):
    # Makes little sense with from_lengths/_with_num_row_partitions.
    original = DynamicRaggedShape.from_lengths(lengths)
    actual = original._with_inner_rank(dense_rank)
    self.assertAllEqual(actual.inner_rank, dense_rank)
    self.assertAllEqual(actual.static_lengths(), lengths_e)

  @parameterized.parameters([
      dict(
          rps=[3, [1, 4, 2]],
          lengths_e=[3, (1, 4, 2)],
          num_row_partitions_e=1,
      ),
      dict(
          rps=[3, [1, 4, 2], 2],
          lengths_e=[3, (1, 4, 2), 2],
          num_row_partitions_e=2,
      ),
  ])
  def testFromRowPartitions(self, rps, lengths_e, num_row_partitions_e):
    rps = _to_row_partitions_from_lengths(rps)
    actual = DynamicRaggedShape.from_row_partitions(rps)
    expected = DynamicRaggedShape.from_lengths(
        lengths_e)._with_num_row_partitions(num_row_partitions_e)
    self.assertShapeEq(expected, actual)

  def testFromRowPartitionsError(self):
    with self.assertRaisesRegex(ValueError, 'row_partitions cannot be empty'):
      DynamicRaggedShape.from_row_partitions([])

  @parameterized.parameters([
      #=========================================================================
      # dimension[axis] is uniform inner; and row_lengths is a scalar
      #=========================================================================
      # shape: [BROADCAST(UNIFORM), UNIFORM, UNIFORM]
      dict(original_lengths=[1, 4, 5],
           broadcast_lengths=[3, 4, 5]),
      # shape: [UNIFORM, UNIFORM, BROADCAST(UNIFORM)]
      dict(original_lengths=[3, 4, 1],
           broadcast_lengths=[3, 4, 5]),
      # shape: [UNIFORM, RAGGED, BROADCAST(UNIFORM)]
      dict(original_lengths=[3, (3, 2, 8), 1],
           broadcast_lengths=[3, (3, 2, 8), 5]),
      # shape: [UNIFORM, RAGGED, RAGGED, UNIFORM, UNIFORM, BROADCAST(UNIFORM)]
      dict(original_lengths=[2, (2, 1), (3, 2, 8), 3, 4, 1],
           broadcast_lengths=[2, (2, 1), (3, 2, 8), 3, 4, 5]),

      #=========================================================================
      # dimension[axis] is uniform inner; and row_lengths is a vector
      #=========================================================================
      # shape: [UNIFORM, BROADCAST(UNIFORM)]
      dict(original_lengths=[3, 1],
           broadcast_lengths=[3, (2, 0, 1)]),
      # shape: [UNIFORM, BROADCAST(UNIFORM), UNIFORM]
      dict(original_lengths=[3, 1, 5],
           broadcast_lengths=[3, (2, 0, 1), 5]),

      # shape: [UNIFORM, UNIFORM, BROADCAST(UNIFORM)]
      dict(original_lengths=[4, 3, 1],
           broadcast_lengths=[4, 3, (2, 0, 1, 3, 8, 2, 3, 4, 1, 8, 7, 0)]),

      # shape: [UNIFORM, RAGGED, BROADCAST(UNIFORM)]
      dict(original_lengths=[2, (2, 1), 1],
           broadcast_lengths=[2, (2, 1), (2, 5, 3)]),

      # shape: [UNIFORM, RAGGED, UNIFORM, UNIFORM, BROADCAST(UNIFORM), UNIFORM]
      dict(original_lengths=[2, (2, 1), 3, 2, 1, 8],
           broadcast_lengths=[2, (2, 1), 3, 2, tuple(range(18)), 8]),

      #=========================================================================
      # dimension[axis] is uniform partitioned; and row_lengths is a scalar
      #=========================================================================
      # shape: [BROADCAST(UNIFORM), RAGGED]
      dict(original_lengths=[1, (5,)],
           broadcast_lengths=[3, (5, 5, 5)]),

      # shape: [BROADCAST(UNIFORM), UNIFORM, RAGGED]
      dict(original_lengths=[1, 3, (3, 0, 2)],
           broadcast_lengths=[2, 3, (3, 0, 2, 3, 0, 2)]),

      # shape: [BROADCAST(UNIFORM), RAGGED, RAGGED, UNIFORM, UNIFORM]
      dict(original_lengths=[1, (3,), (3, 5, 2), 9, 4, 5],
           broadcast_lengths=[3, (3, 3, 3), (3, 5, 2, 3, 5, 2, 3, 5, 2),
                              9, 4, 5]),

      # shape: [BROADCAST(UNIFORM), UNIFORM, RAGGED, UNIFORM]
      dict(original_lengths=[1, 2, (2, 1), (3, 5, 2), 2],
           broadcast_lengths=[2, 2, (2, 1, 2, 1), (3, 5, 2, 3, 5, 2), 2]),

      # shape: [UNIFORM, BROADCAST(UNIFORM), RAGGED, UNIFORM]
      # This is wrong. should broadcast to [3, 2, (4, 4, 0, 0, 2, 2), 5]
      # dict(original_lengths=[3, 1, [4, 0, 2], 5],
      #      broadcast_lengths=[3, 2, [4, 0, 2, 4, 0, 2], 5]),
      dict(original_lengths=[3, 1, (4, 0, 2), 5],
           broadcast_lengths=[3, 2, (4, 4, 0, 0, 2, 2), 5]),

      # shape: [UNIFORM, BROADCAST(UNIFORM), RAGGED]
      dict(original_lengths=[2, 3, (1, 2, 3, 4, 5, 6)],
           broadcast_lengths=[2, 3, (1, 2, 3, 4, 5, 6)]),

      #=========================================================================
      # dimension[axis] is uniform partitioned; and row_lengths is a vector
      #=========================================================================
      # shape: [UNIFORM, BROADCAST(UNIFORM), RAGGED, UNIFORM]
      dict(original_lengths=[
          3,                          # axis=0
          1,                          # axis=1 (broadcast)
          (3, 1, 2),                  # axis=2
          5],                         # axis=3
           broadcast_lengths=[
               3,                          # axis=0
               (4, 1, 2),                  # axis=1 (broadcast)
               (3, 3, 3, 3, 1, 2, 2),      # axis=2
               5]),                        # axis=3

      # shape: [UNIFORM, BROADCAST(UNIFORM), RAGGED, RAGGED]
      dict(original_lengths=[
          3,                                         # axis=0
          1,                                         # axis=1 (broadcast)
          (3, 1, 2),                                 # axis=2
          (3, 1, 4, 1, 5, 9)],                       # axis=3
           broadcast_lengths=[
               3,                                         # axis=0
               (2, 0, 3),                                 # axis=1 (broadcast)
               (3, 3, 2, 2, 2),                           # axis=2
               (3, 1, 4, 3, 1, 4, 5, 9, 5, 9, 5, 9)]),    # axis=3

      # shape: [UNIFORM, RAGGED, BROADCAST(UNIFORM), RAGGED, RAGGED, UNIFORM]
      dict(original_lengths=[
          3,                                         # axis=0
          (2, 0, 1),                                 # axis=1
          1,                                         # axis=2 (broadcast)
          (3, 2, 1),                                 # axis=3
          (1, 0, 1, 0, 2, 3),                        # axis=4
          5],                                        # axis=5
           broadcast_lengths=[
               3,                                         # axis=0
               (2, 0, 1),                                 # axis=2
               (4, 1, 2),                                 # axis=2 (broadcast)
               (3, 3, 3, 3, 2, 1, 1),                     # axis=3
               (1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0,    # axis=4
                2, 3, 3),
               5]),                                       # axis=5
      dict(original_lengths=[1, 1, 2, (2, 1)],
           broadcast_lengths=[2, 1, 2, (2, 1, 2, 1)]),
      dict(original_lengths=[2, 1, 2, (2, 1, 2, 1)],
           broadcast_lengths=[2, (2, 1), 2, (2, 1, 2, 1, 2, 1)]),
      dict(original_lengths=[2, (2, 1), 2, (2, 1, 2, 1, 2, 1)],
           broadcast_lengths=[2, (2, 1), 2, (2, 1, 2, 1, 2, 1)]),
      dict(original_lengths=[2, (2, 1), 2, 1],
           broadcast_lengths=[2, (2, 1), 2, (2, 1, 2, 1, 2, 1)]),
  ])  # pyformat: disable
  def testBroadcastDimension(self, original_lengths, broadcast_lengths):
    """Tests broadcast_to on a single dimension."""
    original_rt = _to_prime_tensor_from_lengths(original_lengths)
    bcast_shape = DynamicRaggedShape.from_lengths(broadcast_lengths)
    result_rt = dynamic_ragged_shape.broadcast_to(original_rt, bcast_shape)
    result_shape = DynamicRaggedShape.from_tensor(result_rt)

    self.assertShapeEq(bcast_shape, result_shape)

  def testAsRowPartitions(self):
    my_shape = DynamicRaggedShape.from_lengths([3, (2, 0, 1), 5])
    rps = my_shape._as_row_partitions()
    self.assertLen(rps, 2)

  def testAsRowPartitionsRaises(self):
    my_shape = DynamicRaggedShape.from_lengths([])
    with self.assertRaisesRegex(ValueError,
                                'rank must be >= 1 for _as_row_partitions'):
      my_shape._as_row_partitions()

  def testToPrimeTensorFromDimSizes(self):
    """Tests the test utility."""
    original_lengths = [3, (3, 2, 8), 1]
    original_rt = _to_prime_tensor_from_lengths(original_lengths)
    expected_rt = _to_ragged_tensor_from_lengths(
        [[2], [3], [5], [7], [11], [13], [17], [19], [23], [29], [31], [37],
         [41]], [3, (3, 2, 8)])

    self.assertAllEqual(expected_rt, original_rt)

  @parameterized.parameters([
      # Broadcast scalar
      dict(x_dims=[], y_dims=[], expected_dims=[]),
      dict(x_dims=[], y_dims=[2], expected_dims=[2]),
      dict(x_dims=[], y_dims=[2, 3], expected_dims=[2, 3]),
      dict(
          x_dims=[],
          y_dims=[2, (2, 3), (5, 7, 2, 0, 9)],
          expected_dims=[2, (2, 3), (5, 7, 2, 0, 9)]),
      # Broadcast vector
      dict(x_dims=[3], y_dims=[4, 2, 3], expected_dims=[4, 2, 3]),
      dict(x_dims=[1], y_dims=[4, 2, 3], expected_dims=[4, 2, 3]),
      dict(x_dims=[3], y_dims=[4, 2, 1], expected_dims=[4, 2, 3]),
      dict(
          x_dims=[3], y_dims=[3, (2, 3, 1), 1], expected_dims=[3, (2, 3, 1),
                                                               3]),
      dict(x_dims=[1], y_dims=[3, (2, 1, 3)], expected_dims=[3, (2, 1, 3)]),
      dict(
          x_dims=[1], y_dims=[3, (2, 1, 3), 8], expected_dims=[3, (2, 1, 3),
                                                               8]),
      dict(
          x_dims=[1],
          y_dims=[2, (2, 3), (5, 7, 2, 0, 9)],
          expected_dims=[2, (2, 3), (5, 7, 2, 0, 9)]),
      # Mixed broadcasting
      dict(
          x_dims=[
              1,  # axis=0
              3,  # axis=1
              (3, 0, 2),  # axis=2
              1,  # axis=3
              2,  # axis=4
          ],
          y_dims=[
              2,  # axis=0
              1,  # axis=1
              1,  # axis=2
              (7, 2),  # axis=3
              1,  # axis=4
          ],
          expected_dims=[
              2,  # axis=0
              3,  # axis=1
              (3, 0, 2, 3, 0, 2),  # axis=2
              (7, 7, 7, 7, 7, 2, 2, 2, 2, 2),  # axis=3
              2,  # axis=4
          ]),
      dict(
          x_dims=[2, (2, 1), 2, 1],
          y_dims=[1, 1, 2, (2, 1)],
          expected_dims=[2, (2, 1), 2, (2, 1, 2, 1, 2, 1)]),
  ])
  def testBroadcastDynamicShape(self, x_dims, y_dims, expected_dims):
    shape_a = DynamicRaggedShape.from_lengths(x_dims)
    shape_b = DynamicRaggedShape.from_lengths(y_dims)
    shape_e = DynamicRaggedShape.from_lengths(expected_dims)
    [actual, bc_a, bc_b
    ] = dynamic_ragged_shape.broadcast_dynamic_shape_extended(shape_a, shape_b)
    [actual_rev, bc_b_rev, bc_a_rev
    ] = dynamic_ragged_shape.broadcast_dynamic_shape_extended(shape_b, shape_a)
    self.assertShapeEq(actual, shape_e)
    self.assertShapeEq(actual_rev, shape_e)

    rt_a = _to_prime_tensor_from_lengths(x_dims)
    bc_a_actual = bc_a.broadcast(rt_a)
    bc_a_actual_rev = bc_a_rev.broadcast(rt_a)
    bc_a_expected = dynamic_ragged_shape.broadcast_to(rt_a, shape_e)
    self.assertAllEqual(bc_a_expected, bc_a_actual)
    self.assertAllEqual(bc_a_expected, bc_a_actual_rev)

    rt_b = _to_prime_tensor_from_lengths(y_dims)
    bc_b_expected = dynamic_ragged_shape.broadcast_to(rt_b, shape_e)
    bc_b_actual = bc_b.broadcast(rt_b)
    bc_b_actual_rev = bc_b_rev.broadcast(rt_b)
    self.assertAllEqual(bc_b_expected, bc_b_actual)
    self.assertAllEqual(bc_b_expected, bc_b_actual_rev)

    # This just wraps broadcast_dynamic_shape_extended, so nothing
    # deeper is required.
    result1 = dynamic_ragged_shape.broadcast_dynamic_shape(shape_a, shape_b)
    self.assertShapeEq(shape_e, result1)

    # Again, just a wrapper.
    result2 = ragged_array_ops.broadcast_dynamic_shape(shape_a, shape_b)
    self.assertShapeEq(shape_e, result2)

  def testBroadcastDynamicShapeFirstLayer(self):
    a_0 = constant_op.constant(1, dtypes.int64)
    b_0 = constant_op.constant(3, dtypes.int64)
    [a_layer, b_layer
    ] = dynamic_ragged_shape._broadcast_dynamic_shape_first_layer(a_0, b_0)
    expected_a_layer = _LayerBroadcaster.from_gather_index([0, 0, 0])
    expected_b_layer = _LayerBroadcaster.from_gather_index([0, 1, 2])
    self.assertLayerBroadcasterEq(expected_a_layer, a_layer)
    self.assertLayerBroadcasterEq(expected_b_layer, b_layer)

  def testBroadcastDynamicShapeNextLayer(self):
    a_1 = RowPartition.from_uniform_row_length(
        1, nvals=1, nrows=1, dtype_hint=dtypes.int64)
    b_1 = RowPartition.from_row_lengths([2, 1, 3], dtype_hint=dtypes.int64)
    ac_0 = _LayerBroadcaster.from_gather_index(
        constant_op.constant([0, 0, 0], dtype=dtypes.int64))
    bc_0 = _LayerBroadcaster.from_gather_index(
        constant_op.constant([0, 1, 2], dtype=dtypes.int64))
    dynamic_ragged_shape._broadcast_dynamic_shape_next_layer_half_ragged(
        ac_0, bc_0, a_1, b_1)

  def testBroadcastDynamicShapeRaisesLeft(self):
    shape = DynamicRaggedShape.from_tensor(constant_op.constant([1, 2, 3]))
    with self.assertRaisesRegex(TypeError, 'shape_x must be'):
      dynamic_ragged_shape.broadcast_dynamic_shape(1, shape)

  def testBroadcastDynamicShapeRaisesRight(self):
    shape = DynamicRaggedShape.from_tensor(constant_op.constant([1, 2, 3]))
    with self.assertRaisesRegex(TypeError, 'shape_y must be'):
      dynamic_ragged_shape.broadcast_dynamic_shape(shape, 1)

  def testBroadcastToRaises(self):
    rt = constant_op.constant([1, 2, 3])
    with self.assertRaisesRegex(TypeError, 'shape must be'):
      dynamic_ragged_shape.broadcast_to(rt, 1)

  @parameterized.parameters([
      dict(
          x=[[10], [20], [30]],  # shape=[3, 1]
          lengths=[3, 2],
          expected=[[10, 10], [20, 20], [30, 30]]),
      dict(
          x=[[10], [20], [30]],  # shape=[3, 1]
          lengths=[3, (3, 0, 2)],
          expected=ragged_factory_ops.constant_value(
              [[10, 10, 10], [], [30, 30]], dtype=np.int32)),
      dict(
          x=[[[1, 2, 3]], [[4, 5, 6]]],  # shape = [2, 1, 3]
          lengths=[2, (2, 3), 3],
          expected=ragged_factory_ops.constant_value(
              [[[1, 2, 3], [1, 2, 3]], [[4, 5, 6], [4, 5, 6], [4, 5, 6]]],
              dtype=np.int32,
              ragged_rank=1)),
      dict(
          x=[[[1]], [[2]]],  # shape = [2, 1, 1]
          lengths=[2, (2, 3), (0, 2, 1, 2, 0)],
          expected=ragged_factory_ops.constant_value(
              [[[], [1, 1]], [[2], [2, 2], []]], dtype=np.int32,
              ragged_rank=2)),
      dict(
          x=10,
          lengths=[3, (3, 0, 2)],
          expected=ragged_factory_ops.constant_value([[10, 10, 10], [],
                                                      [10, 10]])),
      dict(
          x=ragged_factory_ops.constant_value([[[1], [2]], [[3]]],
                                              ragged_rank=1),
          lengths=[2, (2, 1), 2],
          expected=ragged_factory_ops.constant_value(
              [[[1, 1], [2, 2]], [[3, 3]]], ragged_rank=1)),
  ])
  def testRaggedBroadcastTo(self, x, lengths, expected):
    shape = DynamicRaggedShape.from_lengths(lengths)
    result = dynamic_ragged_shape.broadcast_to(x, shape)
    self.assertEqual(
        getattr(result, 'num_row_partitions', 0),
        getattr(expected, 'num_row_partitions', 0))
    self.assertAllEqual(result, expected)

    # broadcast_to just calls dynamic_ragged_shape.broadcast_to, so
    # this should be sufficient.
    result2 = ragged_array_ops.broadcast_to(x, shape)
    self.assertAllEqual(result2, expected)

  @parameterized.parameters([
      dict(
          doc='x.shape=[3, (D1)]; y.shape=[3, 1]; bcast.shape=[3, (D1)]',
          x=ragged_factory_ops.constant_value([[1, 2, 3], [], [4, 5]],
                                              dtype=np.int32),
          y=[[10], [20], [30]],
          expected=ragged_factory_ops.constant_value([[11, 12, 13], [],
                                                      [34, 35]])),
      dict(
          doc='x.shape=[3, (D1)]; y.shape=[]; bcast.shape=[3, (D1)]',
          x=ragged_factory_ops.constant_value([[1, 2, 3], [], [4, 5]],
                                              dtype=np.int32),
          y=10,
          expected=ragged_factory_ops.constant_value([[11, 12, 13], [],
                                                      [14, 15]])),
      dict(
          doc='x.shape=[1, (D1)]; y.shape=[3, 1]; bcast.shape=[3, (D1)]',
          x=ragged_factory_ops.constant_value([[1, 2, 3]], dtype=np.int32),
          y=[[10], [20], [30]],
          expected=ragged_factory_ops.constant_value(
              [[11, 12, 13], [21, 22, 23], [31, 32, 33]], dtype=np.int32)),
      dict(
          doc=('x.shape=[2, (D1), 1]; y.shape=[1, (D2)]; '
               'bcast.shape=[2, (D1), (D2)]'),
          x=ragged_factory_ops.constant_value([[[1], [2], [3]], [[4]]],
                                              ragged_rank=1),
          y=ragged_factory_ops.constant_value([[10, 20, 30]]),
          expected=ragged_factory_ops.constant_value([[[11, 21,
                                                        31], [12, 22, 32],
                                                       [13, 23, 33]],
                                                      [[14, 24, 34]]])),
      dict(
          doc=('x.shape=[2, (D1), 1]; y.shape=[1, 1, 4]; '
               'bcast.shape=[2, (D1), 4]'),
          x=ragged_factory_ops.constant_value([[[10], [20]], [[30]]],
                                              ragged_rank=1),
          y=[[[1, 2, 3, 4]]],
          expected=ragged_factory_ops.constant_value(
              [[[11, 12, 13, 14], [21, 22, 23, 24]], [[31, 32, 33, 34]]],
              ragged_rank=1)),
      dict(
          doc=('x.shape=[2, (D1), 2, 1]; y.shape=[2, (D2)]; '
               'bcast.shape=[2, (D1), (2), (D2)'),
          x=ragged_factory_ops.constant_value(
              [[[[1], [2]], [[3], [4]]], [[[5], [6]]]], ragged_rank=1),
          y=ragged_factory_ops.constant_value([[10, 20], [30]]),
          expected=ragged_factory_ops.constant_value([[[[11, 21], [32]],
                                                       [[13, 23], [34]]],
                                                      [[[15, 25], [36]]]])),
  ])
  def testRaggedAddWithBroadcasting(self, x, y, expected, doc):
    expected_rrank = getattr(expected, 'num_row_partitions', 0)
    x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x, dtype=dtypes.int32)
    y = ragged_tensor.convert_to_tensor_or_ragged_tensor(y, dtype=dtypes.int32)
    result = x + y
    result_rrank = getattr(result, 'num_row_partitions', 0)
    self.assertEqual(expected_rrank, result_rrank)
    if hasattr(expected, 'tolist'):
      expected = expected.tolist()
    self.assertAllEqual(result, expected)

  @parameterized.parameters([
      dict(lengths_a=[3, (1, 4, 2)], new_impl=True, op_max=10),  # Actual ops: 5
      dict(lengths_a=[3, (1, 4, 2)], new_impl=False, op_max=300),
  ])
  def testAddSelf(self, lengths_a, new_impl, op_max, num_row_partitions_a=None):
    if context.executing_eagerly():
      return
    shape_a0 = DynamicRaggedShape.from_lengths(
        lengths_a, num_row_partitions=num_row_partitions_a)
    rt_a = ragged_array_ops.ragged_reshape(
        _lowest_primes(_num_elements_of_lengths(lengths_a)), shape_a0)
    rt_b = rt_a
    g = rt_a.flat_values.graph if ragged_tensor.is_ragged(rt_a) else rt_a.graph
    nodes_at_a = len(g.as_graph_def().node)
    if new_impl:
      dynamic_ragged_shape.ragged_binary_elementwise_op_impl(
          gen_math_ops.add_v2, rt_a, rt_b)
      nodes_at_b = len(g.as_graph_def().node)
      node_delta = nodes_at_b - nodes_at_a
      self.assertLessEqual(node_delta, op_max)
    else:
      if isinstance(rt_a, RaggedTensor):
        rt_a = rt_a.with_row_splits_dtype(dtypes.int32)
      rt_b = rt_a
      nodes_at_b = len(g.as_graph_def().node)
      rt_a + rt_b  # pylint: disable=pointless-statement
      nodes_at_d = len(g.as_graph_def().node)
      node_delta = nodes_at_d - nodes_at_b
      self.assertLessEqual(node_delta, op_max)

  def testAndSelfBool(self):
    if context.executing_eagerly():
      return
    values = constant_op.constant([True, False, True, True, True])
    rt_a = RaggedTensor.from_row_splits(values, [0, 3, 3, 5])
    result = dynamic_ragged_shape.ragged_binary_elementwise_op_impl(
        gen_math_ops.logical_and, rt_a, rt_a)

    expected_values = values
    expected = RaggedTensor.from_row_splits(expected_values, [0, 3, 3, 5])

    self.assertAllEqual(result, expected)

  def testEquals(self):
    if context.executing_eagerly():
      return

    rt_a = ragged_factory_ops.constant([[3, 1, 3], [3]])
    b = constant_op.constant(3)
    rt_expected = ragged_factory_ops.constant([[True, False, True], [True]])

    result = dynamic_ragged_shape.ragged_binary_elementwise_op_impl(
        math_ops.equal, rt_a, b)
    self.assertAllEqual(result, rt_expected)

  def testEquals2(self):
    splits = constant_op.constant([0, 1])
    a = RaggedTensor.from_row_splits([[1, 2]], splits)
    b = RaggedTensor.from_row_splits([[3, 4, 5]], splits)
    self.assertIs(a == b, False)

  def testEquals3(self):
    a = RaggedTensor.from_row_splits([[1, 2]], [0, 1])
    b = RaggedTensor.from_row_splits([[3, 4, 5]], [0, 1])
    self.assertIs(a == b, False)

  @parameterized.parameters([
      dict(
          lengths_a=[3, (1, 4, 2)], lengths_b=[], new_impl=True,
          max_num_ops=5),  # Actual ops: 1
      dict(
          lengths_a=[3, (1, 4, 2), 3, 2],
          lengths_b=[3, 2],
          new_impl=True,
          max_num_ops=5),  # Actual ops: 1
      dict(
          lengths_a=[3, (1, 4, 2)], lengths_b=[], new_impl=False,
          max_num_ops=5),  # Actual ops: 1
      dict(
          lengths_a=[3, (1, 4, 2), 3, 2],
          lengths_b=[3, 2],
          new_impl=False,
          max_num_ops=5),  # Actual ops: 1
  ])
  def testAdd(self,
              lengths_a,
              lengths_b,
              new_impl,
              max_num_ops,
              num_row_partitions_a=None,
              num_row_partitions_b=None):
    if context.executing_eagerly():
      return

    shape_a0 = DynamicRaggedShape.from_lengths(
        lengths_a, num_row_partitions=num_row_partitions_a)
    shape_b0 = DynamicRaggedShape.from_lengths(
        lengths_b, num_row_partitions=num_row_partitions_b)
    rt_a = ragged_array_ops.ragged_reshape(
        _lowest_primes(_num_elements_of_lengths(lengths_a)), shape_a0)
    rt_b = ragged_array_ops.ragged_reshape(
        _lowest_primes(_num_elements_of_lengths(lengths_b)), shape_b0)
    g = rt_a.flat_values.graph if ragged_tensor.is_ragged(rt_a) else rt_a.graph

    nodes_at_a = len(g.as_graph_def().node)
    if new_impl:
      dynamic_ragged_shape.ragged_binary_elementwise_op_impl(
          gen_math_ops.add_v2,
          rt_a,
          rt_b)
      nodes_at_b = len(g.as_graph_def().node)
      num_nodes = nodes_at_b - nodes_at_a
      self.assertLessEqual(num_nodes, max_num_ops)
    else:
      if isinstance(rt_a, RaggedTensor):
        rt_a = rt_a.with_row_splits_dtype(dtypes.int32)
      if isinstance(rt_b, RaggedTensor):
        rt_b = rt_b.with_row_splits_dtype(dtypes.int32)
      nodes_at_b = len(g.as_graph_def().node)
      rt_a + rt_b  # pylint: disable=pointless-statement
      nodes_at_d = len(g.as_graph_def().node)
      num_nodes = nodes_at_d - nodes_at_b

  @parameterized.parameters([
      dict(
          lengths_a=[3, (1, 4, 2)], lengths_b=[],
          shape_e=[3, None], new_impl=False),
      dict(
          lengths_a=[3, (1, 4, 2)], lengths_b=[],
          shape_e=[3, None], new_impl=True),
      dict(
          lengths_a=[5, (1, 4, 2, 1, 3), 3],
          lengths_b=[5, 1, 3],
          shape_e=[5, None, 3], new_impl=False),
      dict(
          lengths_a=[5, (1, 4, 2, 1, 3), 3],
          lengths_b=[5, 1, 3],
          shape_e=[5, None, 3], new_impl=True),
      dict(
          lengths_a=[3, 2, (1, 4, 2, 1, 3, 1), 3],
          lengths_b=[3, 2, 1, 3],
          shape_e=[3, 2, None, 3], new_impl=False),
      dict(
          lengths_a=[3, 2, (1, 4, 2, 1, 3, 1), 3],
          lengths_b=[3, 2, 1, 3],
          shape_e=[3, 2, None, 3],
          new_impl=True),
      dict(
          lengths_a=[3, (1, 4, 2)], lengths_b=[3, 1],
          shape_e=[3, None], new_impl=False),
      dict(
          lengths_a=[3, (1, 4, 2)], lengths_b=[3, 1],
          shape_e=[3, None], new_impl=True),

  ])
  def testAddShape(self,
                   lengths_a,
                   lengths_b,
                   shape_e,
                   new_impl=False,
                   num_row_partitions_a=None,
                   num_row_partitions_b=None):
    if context.executing_eagerly():
      return
    shape_a = DynamicRaggedShape.from_lengths(
        lengths_a, num_row_partitions=num_row_partitions_a)
    shape_b = DynamicRaggedShape.from_lengths(
        lengths_b, num_row_partitions=num_row_partitions_b)
    rt_a = ragged_array_ops.ragged_reshape(
        _lowest_primes(_num_elements_of_lengths(lengths_a)), shape_a)
    rt_b = ragged_array_ops.ragged_reshape(
        _lowest_primes(_num_elements_of_lengths(lengths_b)), shape_b)
    if new_impl:
      result = dynamic_ragged_shape.ragged_binary_elementwise_op_impl(
          math_ops.add, rt_a, rt_b)
      shape_e = tensor_shape.TensorShape(shape_e)
      self.assertEqual(shape_e.as_list(), result.shape.as_list())
    else:
      if isinstance(rt_a, RaggedTensor):
        rt_a = rt_a.with_row_splits_dtype(dtypes.int32)
      if isinstance(rt_b, RaggedTensor):
        rt_b = rt_b.with_row_splits_dtype(dtypes.int32)
      result = rt_a + rt_b
      shape_e = tensor_shape.TensorShape(shape_e)
      self.assertEqual(shape_e.as_list(), result.shape.as_list())

  @parameterized.parameters([
      dict(
          lengths_a=[3, (1, 4, 2)], lengths_b=[],
          shape_e=[3, (1, 4, 2)]),
      dict(
          lengths_a=[5], lengths_b=[1],
          shape_e=[5]),
      dict(
          lengths_a=[5, (1, 4, 2, 1, 3), 3],
          lengths_b=[5, 1, 3],
          shape_e=[5, None, 3]),
      dict(
          lengths_a=[3, 2, (1, 4, 2, 1, 3, 1), 3],
          lengths_b=[3, 2, 1, 3],
          shape_e=[3, 2, None, 3]),
      dict(lengths_a=[3, (1, 4, 2)], lengths_b=[3, 1], shape_e=[3, None]),
      dict(lengths_a=[5, 1, 3], lengths_b=[2, 3], shape_e=[5, 2, 3]),
      dict(lengths_a=[5, 1, (3, 2, 4, 1, 3)], lengths_b=[2, 1],
           shape_e=[5, 2, None]),
      dict(lengths_a=[5, 4, 1, 3], lengths_b=[2, 1], shape_e=[5, 4, 2, 3]),
  ])
  def testBroadcastDynamicShapeStatic(self,
                                      lengths_a,
                                      lengths_b,
                                      shape_e,
                                      num_row_partitions_a=None,
                                      num_row_partitions_b=None):
    if context.executing_eagerly():
      return
    shape_a = DynamicRaggedShape.from_lengths(
        lengths_a, num_row_partitions=num_row_partitions_a)
    shape_b = DynamicRaggedShape.from_lengths(
        lengths_b, num_row_partitions=num_row_partitions_b)

    result = dynamic_ragged_shape.broadcast_dynamic_shape(shape_a, shape_b)
    result_shape = result._to_tensor_shape()

    tensor_shape_e = [None if isinstance(x, tuple) else x for x in shape_e]
    self.assertEqual(shape_e, result.static_lengths())
    self.assertEqual(tensor_shape_e, result_shape.as_list())

  def testBroadcastDynamicShapePartiallyKnown(self):
    if context.executing_eagerly():
      return
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int64)])
    def fun(x):
      shape_a = DynamicRaggedShape([], array_ops.stack([5, x, 3]))
      shape_b = DynamicRaggedShape.from_lengths([1, 3], dtype=dtypes.int64)
      result = dynamic_ragged_shape.broadcast_dynamic_shape(shape_a, shape_b)
      self.assertAllEqual([5, None, 3], result.static_lengths())
    fun(constant_op.constant(2, dtype=dtypes.int64))

  def testBroadcastDynamicShapePartiallyKnownNiceToHave(self):
    if context.executing_eagerly():
      return
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int64)])
    def fun(x):
      shape_a = DynamicRaggedShape([], array_ops.stack([5, x, 3]))
      shape_b = DynamicRaggedShape.from_lengths([2, 3], dtype=dtypes.int64)
      result = dynamic_ragged_shape.broadcast_dynamic_shape(shape_a, shape_b)
      self.assertAllEqual([5, 2, 3], result.static_lengths())
    fun(constant_op.constant(2, dtype=dtypes.int64))

  def testFromRowPartitionsStatic(self):
    if context.executing_eagerly():
      return
    rp = RowPartition.from_row_lengths([4, 2, 3])
    result = DynamicRaggedShape.from_row_partitions([rp])
    self.assertEqual([3, (4, 2, 3)], result.static_lengths())

  @parameterized.parameters([
      dict(
          lengths_a=[3, (1, 4, 2)], dim=0,
          expected=3),
      dict(
          lengths_a=[5], dim=0,
          expected=5),
      dict(
          lengths_a=[5, (1, 4, 2, 1, 3), 3],
          dim=0,
          expected=5),
      dict(
          lengths_a=[5, (1, 4, 2, 1, 3), 3],
          dim=2,
          expected=3),
      dict(
          lengths_a=[3, 2, (1, 4, 2, 1, 3, 1), 3],
          dim=1,
          expected=2),
      dict(lengths_a=[5, 1, 3], dim=0, expected=5),
  ])
  def testDimStatic(self, lengths_a, dim, expected):
    if context.executing_eagerly():
      return
    shape_a = DynamicRaggedShape.from_lengths(lengths_a)
    result = tensor_util.constant_value(shape_a[dim])
    self.assertEqual(result, expected)

  @parameterized.parameters([
      dict(
          lengths_a=[5, (1, 4, 2, 1, 3), 3],
          shape_e=[5, (1, 4, 2, 1, 3), 3],
          new_num_row_partitions=2),  # Fails
      dict(
          lengths_a=[3, 2, (1, 4, 2, 1, 3, 1), 3],
          shape_e=[3, 2, (1, 4, 2, 1, 3, 1), 3],
          new_num_row_partitions=3),  # Fails
  ])
  def testNumRowPartitionShapeStatic(self,
                                     lengths_a,
                                     shape_e,
                                     new_num_row_partitions,
                                     num_row_partitions_a=None):
    if context.executing_eagerly():
      return
    shape_a = DynamicRaggedShape.from_lengths(
        lengths_a, num_row_partitions=num_row_partitions_a)
    result = shape_a._with_num_row_partitions(new_num_row_partitions)
    self.assertEqual(shape_e, result.static_lengths())

  @parameterized.parameters([
      dict(lengths_a=[5, (1, 4, 2, 1, 3), 3]),
      dict(lengths_a=[3, 2, (1, 4, 2, 1, 3, 1), 3]),
  ])
  def testFromLengthsNRowsStatic(self, lengths_a):
    if context.executing_eagerly():
      return
    shape_a = DynamicRaggedShape.from_lengths(lengths_a)
    for rp in shape_a.row_partitions:
      actual = tensor_util.constant_value(rp.nrows())
      self.assertIsNotNone(actual, 'Failed on ' + str(rp))

  @parameterized.parameters([
      dict(
          lengths_a=[5, (1, 4, 2, 1, 3), 3], inner_shape=[33],
          new_inner_rank=1),
      dict(
          lengths_a=[3, 2, (1, 4, 2, 1, 3, 1), 3],
          inner_shape=[36],
          new_inner_rank=1),
      dict(
          lengths_a=[3, 2, (1, 4, 2, 1, 3, 1), 3, 4],
          inner_shape=[36, 4],
          new_inner_rank=2),
  ])
  def testAltInnerShapeStatic(self,
                              lengths_a,
                              inner_shape,
                              new_inner_rank,
                              num_row_partitions_a=None):
    if context.executing_eagerly():
      return
    shape_a = DynamicRaggedShape.from_lengths(
        lengths_a, num_row_partitions=num_row_partitions_a)
    result = shape_a._alt_inner_shape(new_inner_rank)
    result_static = tensor_util.constant_value_as_shape(result)
    self.assertEqual(inner_shape, result_static.as_list())

  @parameterized.parameters([
      dict(
          lengths=[3, (1, 4, 2)],
          shape_e=[3, None]),
      dict(
          lengths=[3, (1, 4, 2)],
          shape_e=[3, None]),
      dict(
          lengths=[5, (1, 4, 2, 1, 3), 3],
          shape_e=[5, None, 3]),
      dict(
          lengths=[5, (1, 4, 2, 1, 3), 3],
          shape_e=[5, None, 3]),
      dict(
          lengths=[3, 2, (1, 4, 2, 1, 3, 1), 3],
          shape_e=[3, 2, None, 3]),
      dict(
          lengths=[3, 2, (1, 4, 2, 1, 3, 1), 3],
          shape_e=[3, 2, None, 3]),
  ])
  def testStaticShape(self,
                      lengths,
                      shape_e,
                      num_row_partitions=None):
    # Testing the shape has enough information.
    # In particular, any uniform_row_length should be reproduced.
    if context.executing_eagerly():
      return
    shape = DynamicRaggedShape.from_lengths(
        lengths, num_row_partitions=num_row_partitions)
    rt_a = ragged_array_ops.ragged_reshape(
        _lowest_primes(_num_elements_of_lengths(lengths)), shape)
    shape_e = tensor_shape.TensorShape(shape_e)
    self.assertEqual(shape_e.as_list(), rt_a.shape.as_list())

  @parameterized.parameters([
      dict(
          lengths=[5, (1, 4, 2, 1, 3), 3],
          shape_e=[5, (1, 4, 2, 1, 3), 3]),
      dict(
          lengths=[3, 2, (1, 4, 2, 1, 3, 1), 3],
          shape_e=[3, 2, (1, 4, 2, 1, 3, 1), 3]),
  ])
  def testWithNumRowPartitionsStatic(self,
                                     lengths,
                                     shape_e,
                                     num_row_partitions=None):
    # Note that this test loses the later static values.
    if context.executing_eagerly():
      return
    shape = DynamicRaggedShape.from_lengths(
        lengths, num_row_partitions=num_row_partitions)
    shape_b = shape._with_num_row_partitions(shape.rank - 1)
    self.assertEqual(shape_e, shape_b.static_lengths())

  def testWithNumRowPartitionsStaticAlt(self):
    # Note that this test loses the later static values.
    if context.executing_eagerly():
      return
    shape = DynamicRaggedShape.from_lengths(
        [5, 2, 3], num_row_partitions=2)
    shape_b = shape._with_num_row_partitions(0)
    self.assertEqual([5, 2, 3], shape_b.static_lengths())

  def testWithNumRowPartitionsDType(self):
    # Note that this test loses the later static values.
    shape = DynamicRaggedShape([], constant_op.constant([5, 2, 3],
                                                        dtype=dtypes.int32))
    self.assertEqual(shape.dtype, dtypes.int32)

    result = shape._with_num_row_partitions(2)
    self.assertEqual(result.dtype, dtypes.int32)

  @parameterized.parameters([
      dict(
          doc='x.shape=[3, (D1)]; y.shape=[3, 1]; bcast.shape=[3, (D1)]',
          x=ragged_factory_ops.constant_value([[1, 2, 3], [], [4, 5]],
                                              dtype=np.int32),
          y=[[10], [20], [30]],
          expected=ragged_factory_ops.constant_value([[11, 12, 13], [],
                                                      [34, 35]])),
      dict(
          doc='x.shape=[3, (D1)]; y.shape=[]; bcast.shape=[3, (D1)]',
          x=ragged_factory_ops.constant_value([[1, 2, 3], [], [4, 5]],
                                              dtype=np.int32),
          y=10,
          expected=ragged_factory_ops.constant_value([[11, 12, 13], [],
                                                      [14, 15]])),
      dict(
          doc='x.shape=[1, (D1)]; y.shape=[3, 1]; bcast.shape=[3, (D1)]',
          x=ragged_factory_ops.constant_value([[1, 2, 3]], dtype=np.int32),
          y=[[10], [20], [30]],
          expected=ragged_factory_ops.constant_value(
              [[11, 12, 13], [21, 22, 23], [31, 32, 33]], dtype=np.int32)),
      dict(
          doc=('x.shape=[2, (D1), 1]; y.shape=[1, (D2)]; '
               'bcast.shape=[2, (D1), (D2)]'),
          x=ragged_factory_ops.constant_value([[[1], [2], [3]], [[4]]],
                                              ragged_rank=1),
          y=ragged_factory_ops.constant_value([[10, 20, 30]]),
          expected=ragged_factory_ops.constant_value([[[11, 21,
                                                        31], [12, 22, 32],
                                                       [13, 23, 33]],
                                                      [[14, 24, 34]]])),
      dict(
          doc=('x.shape=[2, (D1), 1]; y.shape=[1, 1, 4]; '
               'bcast.shape=[2, (D1), 4]'),
          x=ragged_factory_ops.constant_value([[[10], [20]], [[30]]],
                                              ragged_rank=1),
          y=[[[1, 2, 3, 4]]],
          expected=ragged_factory_ops.constant_value(
              [[[11, 12, 13, 14], [21, 22, 23, 24]], [[31, 32, 33, 34]]],
              ragged_rank=1)),
      dict(
          doc=('x.shape=[2, (D1), 2, 1]; y.shape=[2, (D2)]; '
               'bcast.shape=[2, (D1), (2), (D2)'),
          x=ragged_factory_ops.constant_value(
              [[[[1], [2]], [[3], [4]]], [[[5], [6]]]], ragged_rank=1),
          y=ragged_factory_ops.constant_value([[10, 20], [30]]),
          expected=ragged_factory_ops.constant_value([[[[11, 21], [32]],
                                                       [[13, 23], [34]]],
                                                      [[[15, 25], [36]]]])),
  ])
  def testRaggedDispatchImplWithBroadcasting(self, x, y, expected, doc):
    expected_rrank = getattr(expected, 'num_row_partitions', 0)
    x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x, dtype=dtypes.int32)
    y = ragged_tensor.convert_to_tensor_or_ragged_tensor(y, dtype=dtypes.int32)
    result = dynamic_ragged_shape.ragged_binary_elementwise_op_impl(
        gen_math_ops.add_v2, x, y)
    result_rrank = getattr(result, 'num_row_partitions', 0)
    self.assertEqual(expected_rrank, result_rrank)
    if hasattr(expected, 'tolist'):
      expected = expected.tolist()
    self.assertAllEqual(result, expected)

  def testDimensions(self):
    a = DynamicRaggedShape._from_inner_shape([1, 2, 3])
    self.assertAllEqual(1, a._dimension(0))

  def testGetItemIsInstanceTensor(self):
    a = dynamic_ragged_shape.DynamicRaggedShape._from_inner_shape([1, 2, 3])
    self.assertIsInstance(a[0], ops.Tensor)

  @parameterized.parameters([
      dict(
          lengths=[2, 2],
          num_row_partitions=1,
          expected=[2, 2]),
      dict(lengths=[2, 2], num_row_partitions=0, expected=[2, 2]),
      dict(
          lengths=[2, (1, 2), 2], num_row_partitions=1, expected=[2, (1, 2), 2])
  ])
  def testStaticLengths(self,
                        lengths,
                        num_row_partitions,
                        expected,
                        expected_eager=None):
    a = DynamicRaggedShape.from_lengths(lengths)._with_num_row_partitions(
        num_row_partitions)
    actual = a.static_lengths()
    if context.executing_eagerly() and expected_eager is not None:
      self.assertAllEqual(expected_eager, actual)
    else:
      self.assertAllEqual(expected, actual)

  def testStaticLengthsUnknown(self):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    def foo(row_lengths):
      a = DynamicRaggedShape([RowPartition.from_row_lengths(row_lengths)], [6])
      actual = a.static_lengths()
      self.assertAllEqual([None, None], actual)

    foo([3, 3])

  def testStaticLengthsRankUnknown(self):
    # Note that the rank of the shape is unknown, so we can only provide a
    # prefix of the lengths.
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    def foo(inner_shape):
      a = DynamicRaggedShape([RowPartition.from_row_lengths([3, 3])],
                             inner_shape)
      actual = a.static_lengths()
      self.assertAllEqual([2, (3, 3), ...], actual)

    foo([6, 3])

  def testReprRankKnown(self):
    a = DynamicRaggedShape.from_lengths([2, (1, 2), 3])
    actual = str(a)
    self.assertEqual(
        '<DynamicRaggedShape lengths=[2, (1, 2), 3] num_row_partitions=1>',
        actual)

  def assertDimsEqual(self, x: tensor_shape.TensorShape,
                      y: tensor_shape.TensorShape):
    if x.rank is None:
      self.assertIsNone(
          y.rank,
          'x has an unknown rank, but y does not: x={}, y={}'.format(x, y))
      return
    self.assertIsNotNone(
        y.rank,
        'y has an unknown rank, but x does not: x={}, y={}'.format(x, y))
    self.assertAllEqual(x.as_list(), y.as_list())

  def testToTensorShapeRankKnown(self):
    a = DynamicRaggedShape.from_lengths([2, (1, 2), 3])
    actual = a._to_tensor_shape()
    self.assertDimsEqual(tensor_shape.TensorShape([2, None, 3]), actual)

  def testReprRankUnknown(self):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    def foo(inner_shape):
      a = DynamicRaggedShape([RowPartition.from_row_lengths([3, 3])],
                             inner_shape)
      actual = str(a)
      self.assertEqual(
          '<DynamicRaggedShape lengths=[2, (3, 3), ...] num_row_partitions=1>',
          actual)

    foo([6, 3])

  def testToTensorShapeRankUnknown(self):
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    def foo(inner_shape):
      a = DynamicRaggedShape([RowPartition.from_row_lengths([3, 3])],
                             inner_shape)
      actual = a._to_tensor_shape()
      self.assertDimsEqual(
          tensor_shape.TensorShape(None), actual)

    foo([6, 3])

  def testBroadcastDynamicShapeExtendedRankOne(self):
    a = DynamicRaggedShape._from_inner_shape([1])
    b = DynamicRaggedShape._from_inner_shape([3])
    (c, ac, bc) = dynamic_ragged_shape.broadcast_dynamic_shape_extended(a, b)
    expected_c = DynamicRaggedShape._from_inner_shape([3])
    self.assertShapeEq(c, expected_c)
    ac_result = ac.broadcast(constant_op.constant([4]))
    self.assertAllEqual(ac_result, [4, 4, 4])
    bc_result = bc.broadcast(constant_op.constant([4, 7, 1]))
    self.assertAllEqual(bc_result, [4, 7, 1])

  def testBroadcastDynamicShapeExtendedRankOneRev(self):
    a = DynamicRaggedShape._from_inner_shape([3])
    b = DynamicRaggedShape._from_inner_shape([1])
    (c, ac, bc) = dynamic_ragged_shape.broadcast_dynamic_shape_extended(a, b)
    expected_c = DynamicRaggedShape._from_inner_shape([3])
    self.assertShapeEq(c, expected_c)
    bc_result = bc.broadcast(constant_op.constant([4]))
    self.assertAllEqual(bc_result, [4, 4, 4])
    ac_result = ac.broadcast(constant_op.constant([4, 7, 1]))
    self.assertAllEqual(ac_result, [4, 7, 1])

  def testBroadcastDynamicShapeExtendedRankOneIdentity(self):
    a = DynamicRaggedShape._from_inner_shape([3])
    b = DynamicRaggedShape._from_inner_shape([3])
    (c, ac, bc) = dynamic_ragged_shape.broadcast_dynamic_shape_extended(a, b)
    expected_c = DynamicRaggedShape._from_inner_shape([3])
    self.assertShapeEq(c, expected_c)
    bc_result = bc.broadcast(constant_op.constant([4, 7, 1]))
    self.assertAllEqual(bc_result, [4, 7, 1])
    ac_result = ac.broadcast(constant_op.constant([4, 7, 1]))
    self.assertAllEqual(ac_result, [4, 7, 1])

  def testFromGatherLayerIndexRaises(self):
    bad_gather_index = constant_op.constant([0.0, 0.5, 1.0])
    with self.assertRaisesRegex(ValueError, 'gather_index must be'):
      _LayerBroadcaster.from_gather_index(bad_gather_index)

  ### Tests mostly for code coverage ###########################################

  def testFindPreferredDtypeIntNone(self):
    actual = dynamic_ragged_shape._find_dtype(3, None)
    self.assertIsNone(actual)

  @parameterized.parameters([
      dict(
          source_shape=lambda: DynamicRaggedShape._from_inner_shape([3]),
          target_shape=lambda: DynamicRaggedShape._from_inner_shape([3]),
          layer_broadcasters=lambda: [int],
          dtype=None,
          error_type=TypeError,
          error_regex=r'Not a LayerBroadcaster'),
      dict(
          source_shape=lambda: DynamicRaggedShape._from_inner_shape([3]),
          target_shape=lambda: DynamicRaggedShape._from_inner_shape([3]),
          layer_broadcasters=lambda: _LayerBroadcaster.from_gather_index(
              [0, 1, 2]),
          dtype=None,
          error_type=TypeError,
          error_regex=r'layer'),
      dict(
          source_shape=lambda: DynamicRaggedShape._from_inner_shape([3]),
          target_shape=lambda: None,
          layer_broadcasters=lambda:
          [_LayerBroadcaster.from_gather_index([0, 1, 2])],
          dtype=None,
          error_type=TypeError,
          error_regex='target_shape is not a DynamicRaggedShape'),
      dict(
          source_shape=lambda: None,
          target_shape=lambda: DynamicRaggedShape._from_inner_shape([3]),
          layer_broadcasters=lambda:
          [_LayerBroadcaster.from_gather_index([0, 1, 2])],
          dtype=None,
          error_type=TypeError,
          error_regex='source_shape is not a DynamicRaggedShape')
  ])
  def testBroadcasterInitRaises(self, source_shape, target_shape,
                                layer_broadcasters, dtype, error_type,
                                error_regex):
    source_shape = source_shape()
    target_shape = target_shape()
    layer_broadcasters = layer_broadcasters()
    with self.assertRaisesRegex(error_type, error_regex):
      dynamic_ragged_shape._Broadcaster(
          source_shape, target_shape, layer_broadcasters, dtype=dtype)

  def testBroadcasterRepr(self):
    source_shape = DynamicRaggedShape(
        [RowPartition.from_row_splits(constant_op.constant([0, 1, 2]))],
        constant_op.constant([3]))
    target_shape = DynamicRaggedShape(
        [RowPartition.from_row_splits(constant_op.constant([0, 1, 2]))],
        constant_op.constant([3]))
    layer_broadcasters = [
        _LayerBroadcaster.from_gather_index(constant_op.constant([0, 1, 2])),
        _LayerBroadcaster.from_gather_index(constant_op.constant([0, 1, 2]))
    ]
    bc = dynamic_ragged_shape._Broadcaster(source_shape, target_shape,
                                           layer_broadcasters)
    actual = str(bc)
    self.assertRegex(actual, '.src_shape..DynamicRaggedShape')

  def testBroadcasterWithDtype(self):
    source_shape = DynamicRaggedShape(
        [RowPartition.from_row_splits(constant_op.constant([0, 1, 2]))],
        constant_op.constant([3]))
    target_shape = DynamicRaggedShape(
        [RowPartition.from_row_splits(constant_op.constant([0, 1, 2]))],
        constant_op.constant([3]))
    layer_broadcasters = [
        _LayerBroadcaster.from_gather_index(constant_op.constant([0, 1, 2])),
        _LayerBroadcaster.from_gather_index(constant_op.constant([0, 1, 2]))
    ]
    bc = dynamic_ragged_shape._Broadcaster(
        source_shape, target_shape, layer_broadcasters, dtype=dtypes.int32)

    bc2 = bc.with_dtype(dtypes.int64)
    self.assertEqual(bc2.dtype, dtypes.int64)

  # TODO(martinz): This doesn't work for ragged_tensor_shape.
  # Uncomment when we switch over the implementation.
  #    dict(dtype=dtypes.int32)
  @parameterized.parameters([
      dict(dtype=dtypes.int64)
  ])
  def testBroadcasterWithDenseDType(self, dtype):
    a = constant_op.constant([[4]])
    b = RaggedTensor.from_row_splits([[2], [3], [4], [5]], [0, 3, 4])
    b = b.with_row_splits_dtype(dtype)
    c = a + b
    self.assertEqual(c.row_splits.dtype, dtype)
    d = b + a
    self.assertEqual(d.row_splits.dtype, dtype)

  @parameterized.parameters([
      dict(dtype_left=dtypes.int64,
           dtype_right=dtypes.int32),
      dict(dtype_left=dtypes.int32,
           dtype_right=dtypes.int64)])
  def testBroadcastWithDifferentDenseShapeDTypes(self, dtype_left,
                                                 dtype_right):
    s_left = DynamicRaggedShape._from_inner_shape(
        constant_op.constant([4, 1], dtype_left))
    s_right = DynamicRaggedShape._from_inner_shape(
        constant_op.constant([1, 4], dtype_right))
    s_result = dynamic_ragged_shape.broadcast_dynamic_shape(s_left, s_right)
    self.assertEqual(s_result.dtype, dtypes.int64)

  def testBroadcastFlatValuesToDenseExpand(self):
    source = RaggedTensor.from_uniform_row_length([0, 1, 2, 3], 2)
    target_shape = DynamicRaggedShape._from_inner_shape([1, 2, 2])
    broadcaster = dynamic_ragged_shape._get_broadcaster(
        DynamicRaggedShape.from_tensor(source), target_shape)
    flat_values = broadcaster.broadcast_flat_values(source)
    self.assertAllEqual(flat_values, [[[0, 1], [2, 3]]])

  # TODO(edloper): Confirm that this is the expected behavior.
  def testBroadcastFlatValuesToDenseExpandInnerDimensionsFalse(self):
    source = RaggedTensor.from_uniform_row_length([0, 1, 2, 3], 2)
    target_shape = DynamicRaggedShape._from_inner_shape([1, 2, 2])
    broadcaster = dynamic_ragged_shape._get_broadcaster(
        DynamicRaggedShape.from_tensor(source), target_shape)
    flat_values = broadcaster.broadcast_flat_values(
        source, inner_dimensions=False)
    self.assertAllEqual(flat_values, [[0, 1], [2, 3]])

  def testGetLayerBroadcastersFromRPSRaisesTypeError(self):
    with self.assertRaisesRegex(TypeError, 'Not a _LayerBroadcaster'):
      dynamic_ragged_shape._get_layer_broadcasters_from_rps(int, [], [])

  def testGetBroadcasterRankDrop(self):
    with self.assertRaisesRegex(ValueError, 'Cannot broadcast'):
      a = DynamicRaggedShape._from_inner_shape([3, 4, 5])
      b = DynamicRaggedShape._from_inner_shape([4, 5])
      dynamic_ragged_shape._get_broadcaster(a, b)

  @parameterized.parameters([
      dict(
          ac_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          bc_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          a_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          b_1=lambda: None,
          error_type=TypeError,
          error_regex='b_1 should be a RowPartition'),
      dict(
          ac_0=lambda: None,
          bc_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          a_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          b_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          error_type=TypeError,
          error_regex='ac_0 should be a _LayerBroadcaster'),
      dict(
          ac_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          bc_0=lambda: None,
          a_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          b_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          error_type=TypeError,
          error_regex='bc_0 should be a _LayerBroadcaster'),
      dict(
          ac_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          bc_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          a_1=lambda: None,
          b_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          error_type=TypeError,
          error_regex='a_1 should be a RowPartition')
  ])
  def testBroadcastDynamicShapeNextLayerHalfRaggedRaises(
      self, ac_0, bc_0, a_1, b_1, error_type, error_regex):
    ac_0 = ac_0()
    bc_0 = bc_0()
    a_1 = a_1()
    b_1 = b_1()
    with self.assertRaisesRegex(error_type, error_regex):
      dynamic_ragged_shape._broadcast_dynamic_shape_next_layer_half_ragged(
          ac_0, bc_0, a_1, b_1)

  @parameterized.parameters([
      dict(
          ac_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          bc_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          a_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          b_1=lambda: None,
          error_type=TypeError,
          error_regex='b_1 should be a RowPartition'),
      dict(
          ac_0=lambda: None,
          bc_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          a_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          b_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          error_type=TypeError,
          error_regex='ac_0 should be a _LayerBroadcaster'),
      dict(
          ac_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          bc_0=lambda: None,
          a_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          b_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          error_type=TypeError,
          error_regex='bc_0 should be a _LayerBroadcaster'),
      dict(
          ac_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          bc_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          a_1=lambda: None,
          b_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          error_type=TypeError,
          error_regex='a_1 should be a RowPartition')
  ])
  def testBroadcastDynamicShapeNextLayerBothUniformRaises(
      self, ac_0, bc_0, a_1, b_1, error_type, error_regex):
    ac_0 = ac_0()
    bc_0 = bc_0()
    a_1 = a_1()
    b_1 = b_1()
    with self.assertRaisesRegex(error_type, error_regex):
      dynamic_ragged_shape._broadcast_dynamic_shape_next_layer_both_uniform(
          ac_0, bc_0, a_1, b_1)

  @parameterized.parameters([
      dict(
          ac_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          bc_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          a_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          b_1=lambda: None,
          error_type=TypeError,
          error_regex='b_1 should be a RowPartition'),
      dict(
          ac_0=lambda: None,
          bc_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          a_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          b_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          error_type=TypeError,
          error_regex='ac_0 should be a _LayerBroadcaster'),
      dict(
          ac_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          bc_0=lambda: None,
          a_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          b_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          error_type=TypeError,
          error_regex='bc_0 should be a _LayerBroadcaster'),
      dict(
          ac_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          bc_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          a_1=lambda: None,
          b_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          error_type=TypeError,
          error_regex='a_1 should be a RowPartition')
  ])
  def testBroadcastDynamicShapeNextLayerRaises(self, ac_0, bc_0, a_1, b_1,
                                               error_type, error_regex):
    ac_0 = ac_0()
    bc_0 = bc_0()
    a_1 = a_1()
    b_1 = b_1()
    with self.assertRaisesRegex(error_type, error_regex):
      dynamic_ragged_shape._broadcast_dynamic_shape_next_layer(
          ac_0, bc_0, a_1, b_1)

  @parameterized.parameters([
      dict(
          left_dtype=dtypes.int64,
          right_dtype=dtypes.int64,
          expected_dtype=dtypes.int64),
      dict(
          left_dtype=dtypes.int32,
          right_dtype=dtypes.int32,
          expected_dtype=dtypes.int32)
  ])
  def testAddingRowSplits(self, left_dtype, right_dtype, expected_dtype):
    x = ragged_factory_ops.constant([[1, 2]]).with_row_splits_dtype(left_dtype)
    y = ragged_factory_ops.constant([[1, 2]]).with_row_splits_dtype(right_dtype)
    z = math_ops.add(x, y)
    self.assertEqual(z.row_splits.dtype, expected_dtype)

  @parameterized.parameters([
      dict(left_dtype=dtypes.int32, right_dtype=dtypes.int64),
      dict(left_dtype=dtypes.int64, right_dtype=dtypes.int32),
  ])
  def testAddingRowSplitsError(self, left_dtype, right_dtype):
    x = ragged_factory_ops.constant([[1, 2]]).with_row_splits_dtype(left_dtype)
    y = ragged_factory_ops.constant([[1, 2]]).with_row_splits_dtype(right_dtype)
    with self.assertRaisesRegex(
        ValueError, 'Input RaggedTensors have mismatched row_splits dtypes'):
      math_ops.add(x, y)

  def testAddRowPartitionsInvalidV1(self):
    if not context.executing_eagerly():
      return

    with self.assertRaisesRegex(
        (errors_impl.InvalidArgumentError, ValueError),
        'Last row partition does not match flat_values.'):
      rt = ragged_factory_ops.constant([[3], [4, 5], [6]])
      rt_shape = DynamicRaggedShape.from_tensor(rt)
      new_flat_values = constant_op.constant(['a', 'b', 'c', 'd', 'e'])
      rt_shape._add_row_partitions(new_flat_values, validate=True)

  def testGetItemRankNoneTruncate(self):
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    def foo(x):
      rts = DynamicRaggedShape.from_tensor(x)
      actual = rts[:1]
      self.assertShapeEq(rts, actual)

    foo([1, 2, 3])


class DynamicRaggedShapeErrorTest(parameterized.TestCase):

  @parameterized.parameters([
      # Broadcast [1, 2, (1, 2)] to [1, 2, (2, 1)] (FAIL)
      dict(
          origin_lengths=[2, 1, (1, 2)],
          origin_values=[2, 3, 5],
          expected_lengths=[1, 2, (2, 1)]),
      # Broadcast [2, 1, (1, 1)] -> [2, 1, (5, 5)] (UNSUPPORTED)
      dict(
          origin_lengths=[2, 1, (1, 1)],
          origin_values=[2, 3],
          expected_lengths=[2, 1, (5, 5)]),
      # Broadcast [1, 2, (1, 2)] to [2, 2, (2, 1, 1, 2)] (FAIL)
      dict(
          origin_lengths=[1, 2, (1, 2)],
          origin_values=[2, 3, 5],
          expected_lengths=[2, 2, (2, 1, 1, 2)]),
      # Broadcast w.shape = [2,1,(1,3)] to w'.shape = [2,1,(3,3)] (UNSUPPORTED)
      dict(
          origin_lengths=[2, 1, (1, 3)],
          origin_values=[2, 3, 5, 7],  # [[[2]], [[3, 5, 7]]]
          expected_lengths=[2, 1, (3, 3)]),
  ])
  def testBroadcastRaggedError(self, origin_lengths, origin_values,
                               expected_lengths):
    # I pulled this out of the tensorflow test case, so that I could have
    # more control.
    # However this error is being generated, it confuses assertRaises,
    # but it exists.
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                r'Cannot broadcast'):
      # with self.assertRaisesRegex(errors.InvalidArgumentError,
      #                             r"Cannot broadcast"):
      sess = session.Session()
      with sess.as_default():
        origin = _to_ragged_tensor_from_lengths(origin_values, origin_lengths)
        expected_shape = DynamicRaggedShape.from_lengths(expected_lengths)

        rt = dynamic_ragged_shape.broadcast_to(origin, expected_shape)
        sess.run([rt])

  @parameterized.parameters([
      # nvals and nrows don't match (3 != 4) dynamically
      dict(
          row_partitions=lambda: [  # pylint: disable=g-long-lambda
              RowPartition.from_uniform_row_length(1, 3, nrows=3),
              RowPartition.from_uniform_row_length(1, 4, nrows=4)
          ],
          inner_shape=lambda: [4],
          validate=True,
          error_regex='RowPartitions in DynamicRaggedShape do not'),
      # nvals and inner_shape[0] don't match (3 != 4) dynamically
      dict(
          row_partitions=lambda: [  # pylint: disable=g-long-lambda
              RowPartition.from_uniform_row_length(1, 3, nrows=3),
          ],
          inner_shape=lambda: [4],
          validate=True,
          error_regex='Last row partition does not match inner_shape.'),
  ])
  def testConstructorRaisesDynamic(self,
                                   row_partitions,
                                   inner_shape,
                                   error_regex,
                                   validate=False,
                                   dtype=None):
    with self.assertRaisesRegex((errors_impl.InvalidArgumentError, ValueError),
                                error_regex):
      sess = session.Session()
      with sess.as_default():
        row_partitions = row_partitions()
        inner_shape = inner_shape()
        rts = DynamicRaggedShape(
            row_partitions, inner_shape, dtype=dtype, validate=validate)
        sess.run([rts.inner_shape])

  def testRankNone(self):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    def foo(x):
      rts = DynamicRaggedShape._from_inner_shape(x)
      self.assertIsNone(rts.rank)

    foo([3, 7, 5])

  def testNumSlicesInDimensionRankNone(self):
    with self.assertRaisesRegex(ValueError, 'rank is undefined'):

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
      def foo(x):
        rts = DynamicRaggedShape._from_inner_shape(x)
        rts._num_slices_in_dimension(-1)

      foo([3, 7, 5])

  def testGetItemRankNone(self):
    with self.assertRaisesRegex(ValueError, 'Rank must be known to'):

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
      def foo(x):
        rts = DynamicRaggedShape._from_inner_shape(x)
        rts[-1]  # pylint: disable=pointless-statement

      foo([3, 7, 5])

  def testWithDenseRankRankNone(self):
    with self.assertRaisesRegex(ValueError, 'Rank must be known to'):

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
      def foo(x):
        rts = DynamicRaggedShape._from_inner_shape(x)
        rts._with_inner_rank(1)

      foo([3, 7, 5])

  def testWithRaggedRankRankNone(self):
    with self.assertRaisesRegex(ValueError, 'Rank must be known to'):

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
      def foo(x):
        rts = DynamicRaggedShape._from_inner_shape(x)
        rts._with_num_row_partitions(1)

      foo([3, 7, 5])

  def testAsRowPartitionsRankNone(self):
    # Error is readable, but does not match strings correctly.
    with self.assertRaisesRegex(ValueError, ''):

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
      def foo(x):
        rts = DynamicRaggedShape._from_inner_shape(x)
        rts._as_row_partitions()

      foo([3, 7, 5])

  def testBroadcastDynamicShapeExtendedRankNone(self):
    with self.assertRaisesRegex(ValueError,
                                'Unable to broadcast: unknown rank'):

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
      def foo(x):
        a = DynamicRaggedShape._from_inner_shape(x)
        b = DynamicRaggedShape._from_inner_shape([1, 1, 1])
        dynamic_ragged_shape.broadcast_dynamic_shape_extended(a, b)

      foo([3, 7, 5])

  def testBroadcastDynamicShapeUnmatchedTypes6432(self):
    shape_int64 = DynamicRaggedShape.from_lengths([3, (0, 2, 3)],
                                                  dtype=dtypes.int64)
    shape_int32 = DynamicRaggedShape.from_lengths([3, (0, 2, 3)],
                                                  dtype=dtypes.int32)
    with self.assertRaisesRegex(ValueError, "Dtypes don't match"):
      dynamic_ragged_shape.broadcast_dynamic_shape(shape_int64, shape_int32)

  def testBroadcastDynamicShapeUnmatchedTypes3264(self):
    shape_int64 = DynamicRaggedShape.from_lengths([3, (0, 2, 3)],
                                                  dtype=dtypes.int64)
    shape_int32 = DynamicRaggedShape.from_lengths([3, (0, 2, 3)],
                                                  dtype=dtypes.int32)
    with self.assertRaisesRegex(ValueError, "Dtypes don't match"):
      dynamic_ragged_shape.broadcast_dynamic_shape(shape_int32, shape_int64)

  def testGetIdentityBroadcasterRankNone(self):
    with self.assertRaisesRegex(ValueError, 'Shape must have a'):

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
      def foo(x):
        rts = DynamicRaggedShape._from_inner_shape(x)
        dynamic_ragged_shape._get_identity_broadcaster(rts)

      foo([3, 7, 5])

  def testLayerBroadcasterRepr(self):
    index = constant_op.constant([0, 1, 2], name='testLayerBroadcasterRepr')
    lb = _LayerBroadcaster.from_gather_index(index)
    actual = str(lb)
    self.assertRegex(actual, '.*Tensor.*, shape=.3... dtype=int32.')

  def testGetBroadcasterRankNoneLeft(self):
    with self.assertRaisesRegex(ValueError, 'Rank of source and target must'):

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
      def foo(x):
        rts_a = DynamicRaggedShape._from_inner_shape(x)
        rts_b = DynamicRaggedShape._from_inner_shape(x)
        dynamic_ragged_shape._get_broadcaster(rts_a, rts_b)

      foo([3, 7, 5])

  def testFromTensorDType(self):
    x = ragged_factory_ops.constant([[1, 2]])
    self.assertEqual(x.row_splits.dtype, dtypes.int64)
    shape_x = DynamicRaggedShape.from_tensor(x)
    self.assertEqual(shape_x.dtype, dtypes.int64)

  def testAddingRowSplits(self):
    x = ragged_factory_ops.constant([[1, 2]])
    self.assertEqual(x.row_splits.dtype, dtypes.int64)

    y = math_ops.add(x, x)
    self.assertEqual(y.row_splits.dtype, dtypes.int64)

  def testHashingWithMask(self):
    inp_data = ragged_factory_ops.constant(
        [['omar', 'stringer', 'marlo', 'wire'], ['marlo', 'skywalker', 'wire']],
        dtype=dtypes.string)
    mask = math_ops.equal(inp_data, '')
    values = string_ops.string_to_hash_bucket_strong(
        inp_data, 3, name='hash', key=[0xDECAFCAFFE, 0xDECAFCAFFE])
    values = math_ops.add(values, array_ops.ones_like(values))
    local_zeros = array_ops.zeros_like(values)
    values = array_ops.where(mask, local_zeros, values)

  def testAddRowPartitionsInvalid(self):
    with self.assertRaisesRegex(
        (errors_impl.InvalidArgumentError, ValueError),
        'Last row partition does not match flat_values.'):
      sess = session.Session()
      with sess.as_default():
        rt = ragged_factory_ops.constant([[3], [4, 5], [6]])
        rt_shape = DynamicRaggedShape.from_tensor(rt)
        new_flat_values = constant_op.constant(['a', 'b', 'c'])
        rt2 = rt_shape._add_row_partitions(new_flat_values, validate=True)
        sess.run([rt2])


class DynamicRaggedShapeSpecTest(parameterized.TestCase):

  def assertRowPartitionSpecEqual(self,
                                  a: RowPartitionSpec,
                                  b: RowPartitionSpec,
                                  msg='') -> None:
    self.assertEqual(a.nrows, b.nrows, msg)
    self.assertEqual(a.nvals, b.nvals, msg)
    self.assertEqual(a.uniform_row_length, b.uniform_row_length, msg)
    self.assertEqual(a.dtype, b.dtype, msg)

  def assertTensorShapeEqual(self, a: tensor_shape.TensorShape,
                             b: tensor_shape.TensorShape) -> None:
    self.assertEqual(a, b)

  def assertTensorSpecEqual(self,
                            a: tensor_spec.TensorSpec,
                            b: tensor_spec.TensorSpec) -> None:
    self.assertTensorShapeEqual(a.shape, b.shape)
    self.assertEqual(a.dtype, b.dtype)

  def assertDynamicRaggedShapeSpecEqual(self,
                                        a: DynamicRaggedShape.Spec,
                                        b: DynamicRaggedShape.Spec) -> None:
    self.assertTensorShapeEqual(a._static_inner_shape, b._static_inner_shape)
    self.assertTensorSpecEqual(a._inner_shape, b._inner_shape)
    for i, (a, b) in enumerate(zip(a._row_partitions, b._row_partitions)):
      self.assertRowPartitionSpecEqual(a, b, 'Error in partition ' + str(i))

  @parameterized.parameters([
      # Unknown dimension
      dict(
          shape=tensor_shape.TensorShape(None),
          num_row_partitions=1,
          dtype=dtypes.int32,
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=None,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int32),
                  RowPartitionSpec(
                      nrows=None,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int32)
              ],
              static_inner_shape=tensor_shape.TensorShape(None),
              dtype=dtypes.int32)),
      # Unknown dimension, dense
      dict(
          shape=tensor_shape.TensorShape(None),
          num_row_partitions=0,
          dtype=dtypes.int32,
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape(None),
              dtype=dtypes.int32)),
      # Scalar
      dict(
          shape=tensor_shape.TensorShape([]),
          num_row_partitions=0,
          dtype=dtypes.int32,
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([]),
              dtype=dtypes.int32)),
      # Vector
      dict(
          shape=tensor_shape.TensorShape([7]),
          num_row_partitions=0,
          dtype=dtypes.int32,
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([7]),
              dtype=dtypes.int32)),
      # Generic
      dict(
          shape=tensor_shape.TensorShape([5, 3, None, 4, 2, 5]),
          num_row_partitions=3,
          dtype=dtypes.int32,
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=5,
                      nvals=15,
                      uniform_row_length=3,
                      dtype=dtypes.int32),
                  RowPartitionSpec(
                      nrows=15,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int32),
                  RowPartitionSpec(
                      nrows=None,
                      nvals=None,
                      uniform_row_length=4,
                      dtype=dtypes.int32)
              ],
              static_inner_shape=tensor_shape.TensorShape([None, 2, 5]),
              dtype=dtypes.int32)),
      # Generic, Dense
      dict(
          shape=tensor_shape.TensorShape([5, 3, None, 4, 2, 5]),
          num_row_partitions=0,
          dtype=dtypes.int32,
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape(
                  [5, 3, None, 4, 2, 5]),
              dtype=dtypes.int32)),
  ])
  def test_from_tensor_shape(self, shape, num_row_partitions, dtype, expected):
    spec = DynamicRaggedShape.Spec._from_tensor_shape(shape, num_row_partitions,
                                                      dtype)
    self.assertDynamicRaggedShapeSpecEqual(spec, expected)

  @parameterized.parameters([
      # Ridiculous DType.
      dict(
          shape=tensor_shape.TensorShape(None),
          num_row_partitions=1,
          dtype=dtypes.float32,
          error_type=ValueError,
          error_regex='dtype must be tf.int32 or tf.int64'),
      # num_row_partitions positive for scalar.
      dict(
          shape=tensor_shape.TensorShape([]),
          num_row_partitions=1,
          dtype=dtypes.int32,
          error_type=ValueError,
          error_regex='num_row_partitions should be zero ' +
          'if shape is a scalar or vector.'),
      dict(
          shape=tensor_shape.TensorShape([1, 2, 3]),
          num_row_partitions=3,
          dtype=dtypes.int32,
          error_type=ValueError,
          error_regex='num_row_partitions must be less than rank')
  ])
  def test_from_tensor_shape_raises(self, shape, num_row_partitions, dtype,
                                    error_type, error_regex):
    with self.assertRaisesRegex(error_type, error_regex):
      DynamicRaggedShape.Spec._from_tensor_shape(shape, num_row_partitions,
                                                 dtype)

  def test_from_tensor_shape_raises_dtype(self):
    with self.assertRaisesRegex(ValueError,
                                'dtype must be tf.int32 or tf.int64'):
      DynamicRaggedShape.Spec._from_tensor_shape(
          [], tensor_shape.TensorShape([1, 2, 3]), dtypes.float32)

  def test_from_row_partition_inner_shape_and_dtype_raises_dtype(self):
    with self.assertRaisesRegex(
        ValueError, r'dtype of .* is .*int64.*: expected .*int32.*'):
      DynamicRaggedShape.Spec(
          row_partitions=[
              RowPartitionSpec(
                  nrows=None,
                  nvals=None,
                  uniform_row_length=None,
                  dtype=dtypes.int32),
              RowPartitionSpec(
                  nrows=None,
                  nvals=None,
                  uniform_row_length=None,
                  dtype=dtypes.int64)
          ],
          static_inner_shape=tensor_shape.TensorShape(None),
          dtype=dtypes.int32)

  def test_ranks(self):
    spec = dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
        shape=tensor_shape.TensorShape([5, None, 7, 4, 2, 5]),
        num_row_partitions=2,
        dtype=dtypes.int32)

    self.assertEqual(spec.inner_rank, 4)
    self.assertEqual(spec.num_row_partitions, 2)
    self.assertEqual(spec.rank, 6)

  def test_dimension_simple(self):
    spec = dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
        shape=tensor_shape.TensorShape([5, None, 7, 4, 2, 5]),
        num_row_partitions=2,
        dtype=dtypes.int32)

    self.assertEqual(spec._dimension(0), 5)
    self.assertIsNone(spec._dimension(1))
    self.assertEqual(spec._dimension(2), 7)
    self.assertEqual(spec._dimension(3), 4)
    self.assertEqual(spec._dimension(4), 2)
    self.assertEqual(spec._dimension(5), 5)

  @parameterized.parameters([
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
              None, 0, dtypes.int32),
          dimension=0),
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
              None, 0, dtypes.int32),
          dimension=1),
  ])
  def test_dimension_none(self, spec, dimension):
    actual = spec._dimension(dimension)
    self.assertIsNone(actual)

  @parameterized.parameters([
      # Scalar.
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
              [], 0, dtypes.int32),
          dimension=0,
          error_type=ValueError,
          error_regex='Index out of range: 0.'),
      # Scalar.
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
              [], 0, dtypes.int32),
          dimension=1,
          error_type=ValueError,
          error_regex='Index out of range: 1.'),
  ])
  def test_dimension_raises(self, spec, dimension, error_type, error_regex):
    with self.assertRaisesRegex(error_type, error_regex):
      spec._dimension(dimension)

  def test_num_slices_in_dimension_ragged(self):
    spec = dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
        shape=tensor_shape.TensorShape([5, 3, 7, 4, None, 5]),
        num_row_partitions=2,
        dtype=dtypes.int32)

    self.assertEqual(spec._num_slices_in_dimension(0), 5)
    self.assertEqual(spec._num_slices_in_dimension(1), 5 * 3)
    self.assertEqual(spec._num_slices_in_dimension(2), 5 * 3 * 7)
    self.assertEqual(spec._num_slices_in_dimension(3), 5 * 3 * 7 * 4)
    self.assertIsNone(spec._num_slices_in_dimension(4))
    self.assertIsNone(spec._num_slices_in_dimension(5))
    self.assertIsNone(spec._num_slices_in_dimension(-2))

  def test_num_slices_in_dimension_ragged_alt(self):
    spec = dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
        shape=tensor_shape.TensorShape([5, 3, None, 2]),
        num_row_partitions=3,
        dtype=dtypes.int32)

    self.assertEqual(spec._num_slices_in_dimension(0), 5)
    self.assertEqual(spec._num_slices_in_dimension(1), 5 * 3)
    self.assertIsNone(spec._num_slices_in_dimension(2))
    self.assertIsNone(spec._num_slices_in_dimension(3))

  def test_num_slices_in_dimension_dense_known(self):
    spec = dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
        [5, 3, 4], 0, dtypes.int32)

    self.assertEqual(spec._num_slices_in_dimension(0), 5)
    self.assertEqual(spec._num_slices_in_dimension(1), 15)
    self.assertEqual(spec._num_slices_in_dimension(2), 60)

  @parameterized.parameters([
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
              None, 0, dtypes.int32),
          dimension='CRAZY',
          error_type=TypeError,
          error_regex='axis must be an integer'),
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
              None, 0, dtypes.int32),
          dimension=-1,
          error_type=ValueError,
          error_regex='axis=-1 may only be negative' +
          ' if rank is statically known.')
  ])
  def test_num_slices_in_dimension_raises(self, spec, dimension, error_type,
                                          error_regex):
    with self.assertRaisesRegex(error_type, error_regex):
      spec._num_slices_in_dimension(dimension)

  def test_with_dtype(self):
    spec = DynamicRaggedShape.Spec._from_tensor_shape(
        shape=tensor_shape.TensorShape([5, 3, 7, 4, None, 5]),
        num_row_partitions=2,
        dtype=dtypes.int32)
    actual = spec.with_dtype(dtypes.int64)
    self.assertEqual(actual.dtype, dtypes.int64)
    self.assertEqual(actual._row_partitions[0].dtype, dtypes.int64)
    self.assertEqual(actual._row_partitions[1].dtype, dtypes.int64)

  @parameterized.parameters([
      dict(
          original=DynamicRaggedShape.Spec._from_tensor_shape(
              shape=tensor_shape.TensorShape([5, 3, 7, 4, None, 5]),
              num_row_partitions=2,
              dtype=dtypes.int32),
          num_row_partitions=3,
          expected=DynamicRaggedShape.Spec._from_tensor_shape(
              shape=tensor_shape.TensorShape([5, 3, 7, 4, None, 5]),
              num_row_partitions=3,
              dtype=dtypes.int32)),
      dict(
          original=DynamicRaggedShape.Spec._from_tensor_shape(
              shape=tensor_shape.TensorShape([5, 3, 7, 4, None, 5]),
              num_row_partitions=2,
              dtype=dtypes.int32),
          num_row_partitions=1,
          expected=DynamicRaggedShape.Spec._from_tensor_shape(
              shape=tensor_shape.TensorShape([5, 3, 7, 4, None, 5]),
              num_row_partitions=1,
              dtype=dtypes.int32)),
  ])
  def test_with_num_row_partitions(self, original, num_row_partitions,
                                   expected):
    actual = original._with_num_row_partitions(num_row_partitions)
    self.assertDynamicRaggedShapeSpecEqual(actual, expected)

  @parameterized.parameters([
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
              None, 0, dtypes.int32),
          num_row_partitions=2,
          error_type=ValueError,
          error_regex='Changing num_row_partitions with unknown rank'),
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
              [1, 2, 3, 4], 0, dtypes.int32),
          num_row_partitions=4,
          error_type=ValueError,
          error_regex='Number of row partitions too large'),
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
              [1, 2, 3, 4], 0, dtypes.int32),
          num_row_partitions=-3,
          error_type=ValueError,
          error_regex='Number of row partitions negative'),
  ])
  def test_with_num_row_partitions_raises(self, spec, num_row_partitions,
                                          error_type, error_regex):
    with self.assertRaisesRegex(error_type, error_regex):
      spec._with_num_row_partitions(num_row_partitions)

  def test_truncate(self):
    spec = dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
        shape=tensor_shape.TensorShape([5, 3, 7, 4, None, 5]),
        num_row_partitions=2,
        dtype=dtypes.int32)

    for new_rank in range(7):
      truncation = spec._truncate(new_rank)
      self.assertEqual(truncation.rank, new_rank)
      for i in range(new_rank):
        self.assertEqual(
            truncation._dimension(i), spec._dimension(i),
            'Mismatch on new_rank ' + str(new_rank) + ' on dimension ' + str(i))

  def test_truncate_unknown(self):
    spec = DynamicRaggedShape.Spec(
        row_partitions=[
            RowPartitionSpec(
                nrows=3, nvals=7, uniform_row_length=None, dtype=dtypes.int32),
            RowPartitionSpec(
                nrows=7,
                nvals=None,
                uniform_row_length=None,
                dtype=dtypes.int32)
        ],
        static_inner_shape=tensor_shape.TensorShape(None),
        dtype=dtypes.int32)
    expected = DynamicRaggedShape.Spec(
        row_partitions=[
            RowPartitionSpec(
                nrows=3, nvals=7, uniform_row_length=None, dtype=dtypes.int32),
            RowPartitionSpec(
                nrows=7,
                nvals=None,
                uniform_row_length=None,
                dtype=dtypes.int32)
        ],
        static_inner_shape=tensor_shape.TensorShape([None, None]),
        dtype=dtypes.int32)
    actual = spec._truncate(4)
    self.assertDynamicRaggedShapeSpecEqual(actual, expected)

  @parameterized.parameters([
      # Standard scalar
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([]),
              dtype=dtypes.int32),
          expected=0),
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=None,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([None]),
              dtype=dtypes.int64),
          expected=1),
      # Not knowing the shape of the inner shape is weird.
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=None,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape(None),
              dtype=dtypes.int64),
          expected=None),
  ])
  def test_inner_rank(self, spec, expected):
    actual = spec.inner_rank
    self.assertEqual(expected, actual)

  @parameterized.parameters([
      # Standard scalar
      dict(
          other_spec=tensor_spec.TensorSpec([], dtypes.float32),
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([]),
              dtype=dtypes.int64)),
      dict(
          other_spec=ragged_tensor.RaggedTensorSpec([None, None], dtypes.int32),
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(nrows=None,
                                   nvals=None,
                                   uniform_row_length=None,
                                   dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([None]),
              dtype=dtypes.int64)),
      dict(
          other_spec=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(nrows=None,
                                   nvals=None,
                                   uniform_row_length=None,
                                   dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([None]),
              dtype=dtypes.int64),
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(nrows=None,
                                   nvals=None,
                                   uniform_row_length=None,
                                   dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([None]),
              dtype=dtypes.int64)),
  ])
  def test_from_spec(self, other_spec, expected):
    actual = DynamicRaggedShape.Spec._from_spec(other_spec)
    self.assertDynamicRaggedShapeSpecEqual(expected, actual)

  @parameterized.parameters([
      dict(
          row_partitions=[
              RowPartitionSpec(
                  nrows=None,
                  nvals=None,
                  uniform_row_length=None,
                  dtype=dtypes.int64)
          ],
          static_inner_shape=tensor_shape.TensorShape([None]),
          inner_shape=tensor_spec.TensorSpec([1], dtypes.int64)),
      dict(
          row_partitions=[
              RowPartitionSpec(
                  nrows=6,
                  nvals=None,
                  uniform_row_length=None,
                  dtype=dtypes.int64)
          ],
          static_inner_shape=tensor_shape.TensorShape([None]),
          inner_shape=tensor_spec.TensorSpec([1], dtypes.int64)),
      dict(
          row_partitions=[
              RowPartitionSpec(
                  nrows=6, nvals=60, uniform_row_length=10, dtype=dtypes.int64)
          ],
          static_inner_shape=tensor_shape.TensorShape([60]),
          inner_shape=tensor_spec.TensorSpec([1], dtypes.int64)),
      dict(
          row_partitions=[
              RowPartitionSpec(
                  nrows=6, nvals=60, uniform_row_length=10, dtype=dtypes.int64),
              RowPartitionSpec(
                  nrows=60,
                  nvals=120,
                  uniform_row_length=None,
                  dtype=dtypes.int64)
          ],
          static_inner_shape=tensor_shape.TensorShape([120]),
          inner_shape=tensor_spec.TensorSpec([1], dtypes.int64)),
      dict(
          row_partitions=[
              RowPartitionSpec(
                  nrows=6, nvals=60, uniform_row_length=10, dtype=dtypes.int64)
          ],
          static_inner_shape=tensor_shape.TensorShape(None),
          inner_shape=tensor_spec.TensorSpec([None], dtypes.int64))
  ])
  def test_constructor_idempotent(self, row_partitions, static_inner_shape,
                                  inner_shape):
    # The constructor detects if there is any additional information that
    # can be inferred from what is given.
    original = dynamic_ragged_shape.DynamicRaggedShape.Spec(
        row_partitions, static_inner_shape, inner_shape.dtype)
    self.assertTensorShapeEqual(original._static_inner_shape,
                                static_inner_shape)
    self.assertTensorSpecEqual(original._inner_shape, inner_shape)
    for i, (a, b) in enumerate(zip(original._row_partitions, row_partitions)):
      self.assertRowPartitionSpecEqual(a, b, 'Error in partition ' + str(i))

  @parameterized.parameters([
      dict(
          original=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=3,
                      nvals=None,
                      uniform_row_length=4,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([None]),
              dtype=dtypes.int64),
          expected_row_partitions=[
              RowPartitionSpec(
                  nrows=3, nvals=12, uniform_row_length=4, dtype=dtypes.int64)
          ],
          expected_static_inner_shape=tensor_shape.TensorShape([12]),
          expected_inner_shape=tensor_spec.TensorSpec([1], dtypes.int64)),
      dict(
          original=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=None,
                      nvals=None,
                      uniform_row_length=3,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([30]),
              dtype=dtypes.int64),
          expected_row_partitions=[
              RowPartitionSpec(
                  nrows=10, nvals=30, uniform_row_length=3, dtype=dtypes.int64)
          ],
          expected_static_inner_shape=tensor_shape.TensorShape([30]),
          expected_inner_shape=tensor_spec.TensorSpec([1], dtypes.int64)),
      dict(
          original=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=6,
                      nvals=None,
                      uniform_row_length=10,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([None]),
              dtype=dtypes.int64),
          expected_row_partitions=[
              RowPartitionSpec(
                  nrows=6, nvals=60, uniform_row_length=10, dtype=dtypes.int64)
          ],
          expected_static_inner_shape=tensor_shape.TensorShape([60]),
          expected_inner_shape=tensor_spec.TensorSpec([1], dtypes.int64)),
      dict(
          original=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=6,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int64),
                  RowPartitionSpec(
                      nrows=60,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([120]),
              dtype=dtypes.int64),
          expected_row_partitions=[
              RowPartitionSpec(
                  nrows=6,
                  nvals=60,
                  uniform_row_length=None,
                  dtype=dtypes.int64),
              RowPartitionSpec(
                  nrows=60,
                  nvals=120,
                  uniform_row_length=None,
                  dtype=dtypes.int64)
          ],
          expected_static_inner_shape=tensor_shape.TensorShape([120]),
          expected_inner_shape=tensor_spec.TensorSpec([1], dtypes.int64)),
  ])
  def test_constructor_improvements(self, original, expected_row_partitions,
                                    expected_static_inner_shape,
                                    expected_inner_shape):
    # Note that self_merge is only idempotent if no data is partially present.
    self.assertTensorShapeEqual(original._static_inner_shape,
                                expected_static_inner_shape)
    self.assertTensorSpecEqual(original._inner_shape, expected_inner_shape)
    for i, (a, b) in enumerate(
        zip(original._row_partitions, expected_row_partitions)):
      self.assertRowPartitionSpecEqual(a, b, 'Error in partition ' + str(i))

  @parameterized.parameters([
      dict(
          row_partitions=[
              RowPartitionSpec(
                  nrows=3, nvals=12, uniform_row_length=4, dtype=dtypes.int64)
          ],
          static_inner_shape=tensor_shape.TensorShape([]),
          dtype=dtypes.int64,
          error_type=ValueError,
          msg='If row_partitions are provided, must have inner_rank > 0'),
      dict(
          row_partitions=RowPartitionSpec(
              nrows=3, nvals=12, uniform_row_length=4, dtype=dtypes.int64),
          static_inner_shape=tensor_shape.TensorShape([]),
          dtype=dtypes.int64,
          error_type=TypeError,
          msg='row_partitions should be an Iterable'),
      dict(
          row_partitions=[1, 2, 3],
          static_inner_shape=tensor_shape.TensorShape([12]),
          dtype=dtypes.int64,
          error_type=TypeError,
          msg='row_partitions should be an Iterable of RowPartitionSpecs'),
      dict(
          row_partitions=[
              RowPartitionSpec(
                  nrows=3, nvals=12, uniform_row_length=4, dtype=dtypes.int64)
          ],
          static_inner_shape=3,
          dtype=dtypes.int64,
          error_type=ValueError,
          msg='Dimensions 12 and 3'),
      dict(
          row_partitions=[
              RowPartitionSpec(
                  nrows=3, nvals=12, uniform_row_length=4, dtype=dtypes.int64)
          ],
          static_inner_shape=tensor_shape.TensorShape([2]),
          dtype=456,
          error_type=TypeError,
          msg='Cannot convert'),
      dict(
          row_partitions=[
              RowPartitionSpec(
                  nrows=3, nvals=12, uniform_row_length=4, dtype=dtypes.int64)
          ],
          static_inner_shape=tensor_shape.TensorShape([12]),
          dtype=dtypes.int32,
          error_type=ValueError,
          msg='dtype of RowPartitionSpec'),
      dict(
          row_partitions=[
              RowPartitionSpec(
                  nrows=3, nvals=12, uniform_row_length=4, dtype=dtypes.int64)
          ],
          static_inner_shape=tensor_shape.TensorShape([11]),
          dtype=dtypes.int64,
          error_type=ValueError,
          msg='Dimensions 12 and 11 are not compatible'),
      dict(
          row_partitions=[
              RowPartitionSpec(nvals=3, dtype=dtypes.int64),
              RowPartitionSpec(uniform_row_length=4, dtype=dtypes.int64),
              RowPartitionSpec(nrows=17, dtype=dtypes.int64),
          ],
          static_inner_shape=tensor_shape.TensorShape([20]),
          dtype=dtypes.int64,
          error_type=ValueError,
          msg='Dimensions 17 and 12 are not compatible'),
  ])
  def test_constructor_raises(self, row_partitions, static_inner_shape,
                              dtype, error_type, msg):
    # Note that self_merge is only idempotent if no data is partially present.
    with self.assertRaisesRegex(error_type, msg):
      dynamic_ragged_shape.DynamicRaggedShape.Spec(
          row_partitions=row_partitions,
          static_inner_shape=static_inner_shape,
          dtype=dtype)

  @parameterized.parameters([
      # Unknown rank
      dict(
          original=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape(None),
              dtype=dtypes.int64),
          expected=tensor_shape.TensorShape(None)),
      # Scalar
      dict(
          original=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([]),
              dtype=dtypes.int64),
          expected=tensor_shape.TensorShape([])),
      # Vector
      dict(
          original=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([3]),
              dtype=dtypes.int64),
          expected=tensor_shape.TensorShape([3])),
      # Dense
      dict(
          original=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([3, 2, None]),
              dtype=dtypes.int64),
          expected=tensor_shape.TensorShape([3, 2, None])),
      # Ragged
      dict(
          original=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(nrows=6,
                                   nvals=None,
                                   uniform_row_length=10,
                                   dtype=dtypes.int64),
                  RowPartitionSpec(nrows=60,
                                   nvals=None,
                                   uniform_row_length=None,
                                   dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([120]),
              dtype=dtypes.int64),
          expected=tensor_shape.TensorShape([6, 10, None])),

  ])
  def test_to_tensor_shape(self, original, expected):
    # Note that self_merge is only idempotent if no data is partially present.
    actual = original._to_tensor_shape()
    self.assertEqual(actual, expected)

  @parameterized.parameters([
      dict(
          a=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([]),
              dtype=dtypes.int32),
          b=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([]),
              dtype=dtypes.int32),
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([]),
              dtype=dtypes.int32)),
      dict(
          a=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([3, None]),
              dtype=dtypes.int32),
          b=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([None, 4]),
              dtype=dtypes.int32),
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([3, 4]),
              dtype=dtypes.int32)),
      dict(
          a=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=6,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([None]),
              dtype=dtypes.int64),
          b=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=6,
                      nvals=None,
                      uniform_row_length=10,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([None]),
              dtype=dtypes.int64),
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=6,
                      nvals=60,
                      uniform_row_length=10,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([60]),
              dtype=dtypes.int64)),
      dict(
          a=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=6,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([None]),
              dtype=dtypes.int64),
          b=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([None, 10]),
              dtype=dtypes.int64),
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=6,
                      nvals=60,
                      uniform_row_length=10,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([60]),
              dtype=dtypes.int64))
  ])
  def test_merge_with(self,
                      a: DynamicRaggedShape.Spec,
                      b: DynamicRaggedShape.Spec,
                      expected: DynamicRaggedShape.Spec):
    actual = a._merge_with(b)
    actual_rev = b._merge_with(a)

    self.assertDynamicRaggedShapeSpecEqual(actual, expected)
    self.assertDynamicRaggedShapeSpecEqual(actual_rev, expected)

  def test_repr(self):
    original = dynamic_ragged_shape.DynamicRaggedShape.Spec(
        row_partitions=[
            RowPartitionSpec(
                nrows=6,
                nvals=None,
                uniform_row_length=None,
                dtype=dtypes.int64)
        ],
        static_inner_shape=tensor_shape.TensorShape([None]),
        dtype=dtypes.int64)
    representation = repr(original)
    static_inner_shape = tensor_shape.TensorShape([None])
    expected = ('DynamicRaggedShape.Spec(' +
                'row_partitions=(RowPartitionSpec(' +
                'nrows=6, nvals=None, uniform_row_length=None, ' +
                'dtype=tf.int64),), ' +
                f'static_inner_shape={static_inner_shape!r}, ' +
                'dtype=tf.int64)')
    self.assertEqual(representation, expected)

  @parameterized.parameters([
      dict(
          lengths=[3, 4, 5],
          expected=DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([3, 4, 5]),
              dtype=dtypes.int64)),
      dict(
          lengths=[2, (4, 1), 5],
          expected=DynamicRaggedShape.Spec(
              row_partitions=[RowPartitionSpec(nrows=2, nvals=5)],
              static_inner_shape=tensor_shape.TensorShape([5, 5]),
              dtype=dtypes.int64)),
      dict(
          lengths=[2, (4, 1), 5],
          dtype=dtypes.int32,
          expected=DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(nrows=2, nvals=5, dtype=dtypes.int32)],
              static_inner_shape=tensor_shape.TensorShape([5, 5]),
              dtype=dtypes.int32)),
  ])
  def test_from_value(self, lengths, expected, dtype=None):
    original = DynamicRaggedShape.from_lengths(lengths)
    if dtype is not None:
      original = original.with_dtype(dtype)
    actual = dynamic_ragged_shape.DynamicRaggedShape.Spec.from_value(original)
    self.assertTensorShapeEqual(actual, expected)

if __name__ == '__main__':
  googletest.main()
