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
"""Tests for tf.ragged.ragged_shape."""

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
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_shape
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.ragged_shape import _LayerBroadcaster
from tensorflow.python.ops.ragged.ragged_shape import RaggedShape
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.platform import googletest


def _reshape(x, shape: RaggedShape):
  flat_values = array_ops.reshape(x, shape.inner_shape)
  return RaggedTensor._from_nested_row_partitions(flat_values,
                                                  shape.row_partitions)


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
   _) = ragged_shape._to_row_partitions_and_nvals_from_lengths(lengths)
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
  """Static version of RaggedShape.from_lengths(lengths)._num_elements()."""
  return _num_elements_of_lengths_with_rows(1, lengths)


def _to_prime_tensor_from_lengths(
    lengths: Sequence[Union[int, Sequence[int]]]) -> RaggedTensor:
  """Create a tensor of primes with the shape specified."""
  shape = RaggedShape.from_lengths(lengths)
  num_elements = _num_elements_of_lengths(lengths)
  return _reshape(_lowest_primes(num_elements), shape)


@test_util.run_all_in_graph_and_eager_modes
class RaggedShapeTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def assertRowPartitionEq(self,
                           x: RowPartition,
                           y: RowPartition,
                           msg=None) -> None:
    self.assertAllEqual(x.row_splits(), y.row_splits(), msg=msg)

  def assertShapeEq(self, x: RaggedShape, y: RaggedShape, msg=None) -> None:
    assert isinstance(x, RaggedShape)
    assert isinstance(y, RaggedShape)
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

  def assertBroadcasterEq(self, x: ragged_shape._Broadcaster,
                          y: ragged_shape._Broadcaster) -> None:
    assert isinstance(x, ragged_shape._Broadcaster)
    assert isinstance(y, ragged_shape._Broadcaster)
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
    shape = RaggedShape.from_tensor(value)
    row_partitions = [RowPartition.from_row_splits(x) for x in row_partitions]
    expected = RaggedShape(row_partitions, inner_shape)
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
    shape = RaggedShape.from_lengths(
        lengths, num_row_partitions=num_row_partitions)
    expected = RaggedShape(row_partitions, inner_shape)
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
      RaggedShape.from_lengths(lengths, num_row_partitions=num_row_partitions)

  def testGetBroadcaster(self):
    origin_shape = RaggedShape([RowPartition.from_uniform_row_length(1, 3)],
                               inner_shape=[3])
    dest_shape = RaggedShape([RowPartition.from_uniform_row_length(2, 6)],
                             inner_shape=[6])
    actual = ragged_shape._get_broadcaster(origin_shape, dest_shape)
    expected = ragged_shape._Broadcaster(origin_shape, dest_shape, [
        _LayerBroadcaster.from_gather_index([0, 1, 2]),
        _LayerBroadcaster.from_gather_index([0, 0, 1, 1, 2, 2])
    ])
    self.assertBroadcasterEq(actual, expected)

  def testGetBroadcaster2(self):
    origin_shape = RaggedShape([], inner_shape=[])
    dest_shape = RaggedShape([RowPartition.from_row_splits([0, 2, 3])],
                             inner_shape=[3])
    actual = ragged_shape._get_broadcaster(origin_shape, dest_shape)
    expected = ragged_shape._Broadcaster(origin_shape, dest_shape, [])
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
    original = RaggedShape.from_lengths(lengths)
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
    original = RaggedShape.from_lengths(lengths)
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
    original = RaggedShape.from_lengths(lengths)
    with self.assertRaisesRegex(error_type, error_regex):
      original._alt_inner_shape(new_dense_rank)

  @parameterized.parameters([
      dict(
          lengths=[2, (1, 2), 4], new_dense_rank=2, expected_inner_shape=[3,
                                                                          4]),
  ])
  def testAltInnerShape(self, lengths, new_dense_rank, expected_inner_shape):
    original = RaggedShape.from_lengths(lengths)
    actual = original._alt_inner_shape(new_dense_rank)
    self.assertAllEqual(actual, expected_inner_shape)

  @parameterized.parameters([
      dict(
          lengths=[2],
          new_dense_rank=2,
          error_type=ValueError,
          error_regex='Cannot change inner_rank if'),
  ])
  def testWithDenseRankRaises(self, lengths, new_dense_rank, error_type,
                              error_regex):
    original = RaggedShape.from_lengths(lengths)
    with self.assertRaisesRegex(error_type, error_regex):
      original.with_inner_rank(new_dense_rank)

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
    original = RaggedShape.from_lengths(lengths)
    with self.assertRaisesRegex(error_type, error_regex):
      original._with_num_row_partitions(num_row_partitions)

  def testDimensionRaises(self):
    original = RaggedShape.from_lengths([2, (1, 2)])
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
    original = RaggedShape.from_lengths(lengths)
    if num_row_partitions is not None:
      original = original._with_num_row_partitions(num_row_partitions)
    expected = RaggedShape.from_lengths(expected_lengths)
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
    original = RaggedShape.from_lengths(lengths)
    if num_row_partitions is not None:
      original = original._with_num_row_partitions(num_row_partitions)
    with self.assertRaisesRegex(error_type, error_regex):
      original[index]  # pylint: disable=pointless-statement

  def testBroadcastToAlt(self):
    origin = RaggedTensor.from_uniform_row_length([3, 4, 5],
                                                  uniform_row_length=1)
    expected = RaggedTensor.from_uniform_row_length([3, 3, 4, 4, 5, 5],
                                                    uniform_row_length=2)
    expected_shape = RaggedShape.from_tensor(expected)
    actual = ragged_shape.broadcast_to(origin, expected_shape)
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
    source = RaggedShape.from_lengths(source_lengths)
    if source_num_row_partitions is not None:
      source = source._with_num_row_partitions(source_num_row_partitions)
    target = RaggedShape.from_lengths(target_lengths)
    if target_num_row_partitions is not None:
      target = target._with_num_row_partitions(target_num_row_partitions)

    expected_gather_indices = [
        _LayerBroadcaster.from_gather_index(x) for x in expected_gather_indices
    ]
    actual = ragged_shape._get_broadcaster(source, target)
    expected = ragged_shape._Broadcaster(source, target,
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
      target_shape = RaggedShape.from_row_partitions(
          [RowPartition.from_row_splits(row_splits=[0, 4, 7, 8])])

      rt = ragged_shape.broadcast_to(x, target_shape)
      return rt.flat_values

    x = constant_op.constant([[3.0], [1.0], [4.0]])
    y = func(x)
    g = gradients_impl.gradients(ys=y, xs=x)[0]

    self.assertAllClose(g, [[4.], [3.], [1.]])

  def testBroadcastScalarToScalar(self):
    origin = constant_op.constant(b'x')
    expected = origin
    expected_shape = RaggedShape.from_tensor(expected)
    actual = ragged_shape.broadcast_to(origin, expected_shape)
    self.assertAllEqual(actual, expected)

  @parameterized.parameters([
      dict(lengths=[2, 3], axis=0),
      dict(lengths=[2, 3], axis=1),
      dict(lengths=[2, (2, 3), 7, 4], num_row_partitions=2, axis=0),
      dict(lengths=[2, (2, 3), 7, 4], num_row_partitions=2, axis=2),
      dict(lengths=[2, (2, 3), 7, 4], num_row_partitions=2, axis=3),
  ])
  def testIsUniformTrue(self, lengths, axis, num_row_partitions=None):
    shape = RaggedShape.from_lengths(lengths)
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
    shape = RaggedShape.from_lengths(lengths)._with_num_row_partitions(
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
    shape = RaggedShape.from_lengths(lengths)._with_num_row_partitions(
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
    shape = RaggedShape.from_lengths(lengths)
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
    shape = RaggedShape.from_lengths(lengths)
    if num_row_partitions is not None:
      shape = shape._with_num_row_partitions(num_row_partitions)
    actual = shape[axis]
    self.assertAllEqual(actual, expected)

  def testNumElements(self):
    shape = RaggedShape.from_lengths([2, 3, 4, 5])._with_num_row_partitions(2)
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
    expected_shape = RaggedShape.from_lengths(expected_lengths)
    if expected_num_row_partitions is not None:
      expected_shape = expected_shape._with_num_row_partitions(
          expected_num_row_partitions)
    expected = ragged_factory_ops.constant_value(expected)
    actual = ragged_shape.broadcast_to(origin, expected_shape)
    self.assertAllEqual(actual, expected)

  def testBroadcastFlatValues(self):
    origin_lengths = [3, (1, 2, 1), 2, 2]
    dest_lengths = [1, 1, 3, (1, 2, 1), 2, 2]
    origin_values = constant_op.constant([
        b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i', b'j', b'k', b'l',
        b'm', b'n', b'o', b'p'
    ])
    origin_shape = RaggedShape.from_lengths(
        origin_lengths)._with_num_row_partitions(3)
    dest_shape = RaggedShape.from_lengths(
        dest_lengths)._with_num_row_partitions(5)

    broadcaster = ragged_shape._get_broadcaster(origin_shape, dest_shape)

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
    expected_shape = RaggedShape.from_tensor(expected)
    actual = ragged_shape.broadcast_to(origin, expected_shape)
    self.assertAllEqual(actual, expected)

  def testRaggedShapeFromTensor2(self):
    raw_rt = [[[[7, 4], [5, 6]], [[1, 2], [3, 7]]], [[[7, 4], [5, 6]]],
              [[[1, 2], [3, 7]]]]
    raw_rt = ragged_factory_ops.constant_value(raw_rt)
    actual_shape = RaggedShape.from_tensor(raw_rt)
    expected_shape = RaggedShape.from_lengths([3, (2, 1, 1), 2,
                                               2])._with_num_row_partitions(3)
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
          error_regex='RowPartitions in RaggedShape do not'),
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
      RaggedShape(row_partitions, inner_shape, dtype=dtype, validate=validate)

  def testConstructorStaticOK(self):
    row_partitions = [
        RowPartition.from_value_rowids([0, 2, 4], nrows=5),
        RowPartition.from_value_rowids([0, 1, 2], nrows=3)
    ]
    inner_shape = [3]
    rts = RaggedShape(row_partitions, inner_shape, validate=True)
    static_inner_shape = tensor_util.constant_value(rts.inner_shape)
    static_valid_rowids0 = tensor_util.constant_value(
        rts.row_partitions[0].value_rowids())
    static_valid_rowids1 = tensor_util.constant_value(
        rts.row_partitions[1].value_rowids())
    self.assertAllEqual(static_inner_shape, [3])
    self.assertAllEqual(static_valid_rowids0, [0, 2, 4])
    self.assertAllEqual(static_valid_rowids1, [0, 1, 2])

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
    shape_a = RaggedShape.from_lengths(lengths_a)
    if num_row_partitions_a is not None:
      shape_a = shape_a._with_num_row_partitions(num_row_partitions_a)
    shape_b = RaggedShape.from_lengths(lengths_b)
    if num_row_partitions_b is not None:
      shape_b = shape_b._with_num_row_partitions(num_row_partitions_b)
    shape_e = RaggedShape.from_lengths(lengths_e)
    if num_row_partitions_e is not None:
      shape_e = shape_e._with_num_row_partitions(num_row_partitions_e)

    [actual, bc_a, bc_b
    ] = ragged_shape.broadcast_dynamic_shape_extended(shape_a, shape_b)
    [actual_rev, bc_b_rev, bc_a_rev
    ] = ragged_shape.broadcast_dynamic_shape_extended(shape_b, shape_a)
    self.assertShapeEq(actual, shape_e)
    self.assertShapeEq(actual_rev, shape_e)

    rt_a = _reshape(
        _lowest_primes(_num_elements_of_lengths(lengths_a)), shape_a)
    bc_a_actual = bc_a.broadcast(rt_a)
    bc_a_actual_rev = bc_a_rev.broadcast(rt_a)
    bc_a_expected = ragged_shape.broadcast_to(rt_a, shape_e)
    self.assertAllEqual(bc_a_expected, bc_a_actual)
    self.assertAllEqual(bc_a_expected, bc_a_actual_rev)

    rt_b = _reshape(
        _lowest_primes(_num_elements_of_lengths(lengths_b)), shape_b)
    bc_b_expected = ragged_shape.broadcast_to(rt_b, shape_e)
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
    original = RaggedShape.from_lengths(lengths)
    actual = original.with_inner_rank(dense_rank)
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
    actual = RaggedShape.from_row_partitions(rps)
    expected = RaggedShape.from_lengths(lengths_e)._with_num_row_partitions(
        num_row_partitions_e)
    self.assertShapeEq(expected, actual)

  def testFromRowPartitionsError(self):
    with self.assertRaisesRegex(ValueError, 'row_partitions cannot be empty'):
      RaggedShape.from_row_partitions([])

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
    bcast_shape = RaggedShape.from_lengths(broadcast_lengths)
    result_rt = ragged_shape.broadcast_to(original_rt, bcast_shape)
    result_shape = RaggedShape.from_tensor(result_rt)

    self.assertShapeEq(bcast_shape, result_shape)

  def testAsRowPartitions(self):
    my_shape = RaggedShape.from_lengths([3, (2, 0, 1), 5])
    rps = my_shape._as_row_partitions()
    self.assertLen(rps, 2)

  def testAsRowPartitionsRaises(self):
    my_shape = RaggedShape.from_lengths([])
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
    shape_a = RaggedShape.from_lengths(x_dims)
    shape_b = RaggedShape.from_lengths(y_dims)
    shape_e = RaggedShape.from_lengths(expected_dims)
    [actual, bc_a, bc_b
    ] = ragged_shape.broadcast_dynamic_shape_extended(shape_a, shape_b)
    [actual_rev, bc_b_rev, bc_a_rev
    ] = ragged_shape.broadcast_dynamic_shape_extended(shape_b, shape_a)
    self.assertShapeEq(actual, shape_e)
    self.assertShapeEq(actual_rev, shape_e)

    rt_a = _to_prime_tensor_from_lengths(x_dims)
    bc_a_actual = bc_a.broadcast(rt_a)
    bc_a_actual_rev = bc_a_rev.broadcast(rt_a)
    bc_a_expected = ragged_shape.broadcast_to(rt_a, shape_e)
    self.assertAllEqual(bc_a_expected, bc_a_actual)
    self.assertAllEqual(bc_a_expected, bc_a_actual_rev)

    rt_b = _to_prime_tensor_from_lengths(y_dims)
    bc_b_expected = ragged_shape.broadcast_to(rt_b, shape_e)
    bc_b_actual = bc_b.broadcast(rt_b)
    bc_b_actual_rev = bc_b_rev.broadcast(rt_b)
    self.assertAllEqual(bc_b_expected, bc_b_actual)
    self.assertAllEqual(bc_b_expected, bc_b_actual_rev)

    # This just wraps broadcast_dynamic_shape_extended, so nothing
    # deeper is required.
    result1 = ragged_shape.broadcast_dynamic_shape(shape_a, shape_b)
    self.assertShapeEq(shape_e, result1)

  def testBroadcastDynamicShapeFirstLayer(self):
    a_0 = constant_op.constant(1, dtypes.int64)
    b_0 = constant_op.constant(3, dtypes.int64)
    [a_layer, b_layer
    ] = ragged_shape._broadcast_dynamic_shape_first_layer(a_0, b_0)
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
    ragged_shape._broadcast_dynamic_shape_next_layer_half_ragged(
        ac_0, bc_0, a_1, b_1)

  def testBroadcastDynamicShapeRaisesLeft(self):
    shape = RaggedShape.from_tensor(constant_op.constant([1, 2, 3]))
    with self.assertRaisesRegex(TypeError, 'shape_x must be'):
      ragged_shape.broadcast_dynamic_shape(1, shape)

  def testBroadcastDynamicShapeRaisesRight(self):
    shape = RaggedShape.from_tensor(constant_op.constant([1, 2, 3]))
    with self.assertRaisesRegex(TypeError, 'shape_y must be'):
      ragged_shape.broadcast_dynamic_shape(shape, 1)

  def testBroadcastToRaises(self):
    rt = constant_op.constant([1, 2, 3])
    with self.assertRaisesRegex(TypeError, 'shape must be'):
      ragged_shape.broadcast_to(rt, 1)

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
    shape = RaggedShape.from_lengths(lengths)
    result = ragged_shape.broadcast_to(x, shape)
    self.assertEqual(
        getattr(result, 'num_row_partitions', 0),
        getattr(expected, 'num_row_partitions', 0))
    self.assertAllEqual(result, expected)

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

  def testDimensions(self):
    a = RaggedShape._from_inner_shape([1, 2, 3])
    self.assertAllEqual(1, a._dimension(0))

  @parameterized.parameters([
      dict(
          lengths=[2, 2],
          num_row_partitions=1,
          expected=[2, None, ...],  # Also acceptable: [2, 2]
          expected_eager=[2, 2]),
      dict(lengths=[2, 2], num_row_partitions=0, expected=[2, 2]),
      dict(
          lengths=[2, (1, 2), 2], num_row_partitions=1, expected=[2, (1, 2), 2])
  ])
  def testStaticLengths(self,
                        lengths,
                        num_row_partitions,
                        expected,
                        expected_eager=None):
    a = RaggedShape.from_lengths(lengths)._with_num_row_partitions(
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
      a = RaggedShape([RowPartition.from_row_lengths(row_lengths)], [6])
      actual = a.static_lengths()
      self.assertAllEqual([None, None], actual)

    foo([3, 3])

  def testStaticLengthsRankUnknown(self):
    # Note that the rank of the shape is unknown, so we can only provide a
    # prefix of the lengths.
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    def foo(inner_shape):
      a = RaggedShape([RowPartition.from_row_lengths([3, 3])], inner_shape)
      actual = a.static_lengths()
      self.assertAllEqual([2, (3, 3), ...], actual)

    foo([6, 3])

  def testReprRankKnown(self):
    a = RaggedShape.from_lengths([2, (1, 2), 3])
    actual = str(a)
    self.assertEqual(
        '<RaggedShape lengths=[2, (1, 2), 3] num_row_partitions=1>', actual)

  def testReprRankUnknown(self):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    def foo(inner_shape):
      a = RaggedShape([RowPartition.from_row_lengths([3, 3])], inner_shape)
      actual = str(a)
      self.assertEqual(
          '<RaggedShape lengths=[2, (3, 3), ...] num_row_partitions=1>', actual)

    foo([6, 3])

  def testBroadcastDynamicShapeExtendedRankOne(self):
    a = RaggedShape._from_inner_shape([1])
    b = RaggedShape._from_inner_shape([3])
    (c, ac, bc) = ragged_shape.broadcast_dynamic_shape_extended(a, b)
    expected_c = RaggedShape._from_inner_shape([3])
    self.assertShapeEq(c, expected_c)
    ac_result = ac.broadcast(constant_op.constant([4]))
    self.assertAllEqual(ac_result, [4, 4, 4])
    bc_result = bc.broadcast(constant_op.constant([4, 7, 1]))
    self.assertAllEqual(bc_result, [4, 7, 1])

  def testBroadcastDynamicShapeExtendedRankOneRev(self):
    a = RaggedShape._from_inner_shape([3])
    b = RaggedShape._from_inner_shape([1])
    (c, ac, bc) = ragged_shape.broadcast_dynamic_shape_extended(a, b)
    expected_c = RaggedShape._from_inner_shape([3])
    self.assertShapeEq(c, expected_c)
    bc_result = bc.broadcast(constant_op.constant([4]))
    self.assertAllEqual(bc_result, [4, 4, 4])
    ac_result = ac.broadcast(constant_op.constant([4, 7, 1]))
    self.assertAllEqual(ac_result, [4, 7, 1])

  def testBroadcastDynamicShapeExtendedRankOneIdentity(self):
    a = RaggedShape._from_inner_shape([3])
    b = RaggedShape._from_inner_shape([3])
    (c, ac, bc) = ragged_shape.broadcast_dynamic_shape_extended(a, b)
    expected_c = RaggedShape._from_inner_shape([3])
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
    actual = ragged_shape._find_dtype(3, None)
    self.assertIsNone(actual)

  @parameterized.parameters([
      dict(
          source_shape=lambda: RaggedShape._from_inner_shape([3]),
          target_shape=lambda: RaggedShape._from_inner_shape([3]),
          layer_broadcasters=lambda: [int],
          dtype=None,
          error_type=TypeError,
          error_regex=r'Not a LayerBroadcaster'),
      dict(
          source_shape=lambda: RaggedShape._from_inner_shape([3]),
          target_shape=lambda: RaggedShape._from_inner_shape([3]),
          layer_broadcasters=lambda: _LayerBroadcaster.from_gather_index(
              [0, 1, 2]),
          dtype=None,
          error_type=TypeError,
          error_regex=r'layer'),
      dict(
          source_shape=lambda: RaggedShape._from_inner_shape([3]),
          target_shape=lambda: None,
          layer_broadcasters=lambda:
          [_LayerBroadcaster.from_gather_index([0, 1, 2])],
          dtype=None,
          error_type=TypeError,
          error_regex='target_shape is not a RaggedShape'),
      dict(
          source_shape=lambda: None,
          target_shape=lambda: RaggedShape._from_inner_shape([3]),
          layer_broadcasters=lambda:
          [_LayerBroadcaster.from_gather_index([0, 1, 2])],
          dtype=None,
          error_type=TypeError,
          error_regex='source_shape is not a RaggedShape')
  ])
  def testBroadcasterInitRaises(self, source_shape, target_shape,
                                layer_broadcasters, dtype, error_type,
                                error_regex):
    source_shape = source_shape()
    target_shape = target_shape()
    layer_broadcasters = layer_broadcasters()
    with self.assertRaisesRegex(error_type, error_regex):
      ragged_shape._Broadcaster(
          source_shape, target_shape, layer_broadcasters, dtype=dtype)

  def testBroadcasterRepr(self):
    source_shape = RaggedShape(
        [RowPartition.from_row_splits(constant_op.constant([0, 1, 2]))],
        constant_op.constant([3]))
    target_shape = RaggedShape(
        [RowPartition.from_row_splits(constant_op.constant([0, 1, 2]))],
        constant_op.constant([3]))
    layer_broadcasters = [
        _LayerBroadcaster.from_gather_index(constant_op.constant([0, 1, 2])),
        _LayerBroadcaster.from_gather_index(constant_op.constant([0, 1, 2]))
    ]
    bc = ragged_shape._Broadcaster(source_shape, target_shape,
                                   layer_broadcasters)
    actual = str(bc)
    self.assertRegex(actual, '.src_shape..RaggedShape')

  def testBroadcasterWithDtype(self):
    source_shape = RaggedShape(
        [RowPartition.from_row_splits(constant_op.constant([0, 1, 2]))],
        constant_op.constant([3]))
    target_shape = RaggedShape(
        [RowPartition.from_row_splits(constant_op.constant([0, 1, 2]))],
        constant_op.constant([3]))
    layer_broadcasters = [
        _LayerBroadcaster.from_gather_index(constant_op.constant([0, 1, 2])),
        _LayerBroadcaster.from_gather_index(constant_op.constant([0, 1, 2]))
    ]
    bc = ragged_shape._Broadcaster(
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
    s_left = RaggedShape._from_inner_shape(constant_op.constant([4, 1],
                                                                dtype_left))
    s_right = RaggedShape._from_inner_shape(constant_op.constant([1, 4],
                                                                 dtype_right))
    s_result = ragged_shape.broadcast_dynamic_shape(s_left, s_right)
    self.assertEqual(s_result.dtype, dtypes.int64)

  def testBroadcastFlatValuesToDenseExpand(self):
    source = RaggedTensor.from_uniform_row_length([0, 1, 2, 3], 2)
    target_shape = RaggedShape._from_inner_shape([1, 2, 2])
    broadcaster = ragged_shape._get_broadcaster(
        RaggedShape.from_tensor(source), target_shape)
    flat_values = broadcaster.broadcast_flat_values(source)
    self.assertAllEqual(flat_values, [[[0, 1], [2, 3]]])

  # TODO(edloper): Confirm that this is the expected behavior.
  def testBroadcastFlatValuesToDenseExpandInnerDimensionsFalse(self):
    source = RaggedTensor.from_uniform_row_length([0, 1, 2, 3], 2)
    target_shape = RaggedShape._from_inner_shape([1, 2, 2])
    broadcaster = ragged_shape._get_broadcaster(
        RaggedShape.from_tensor(source), target_shape)
    flat_values = broadcaster.broadcast_flat_values(
        source, inner_dimensions=False)
    self.assertAllEqual(flat_values, [[0, 1], [2, 3]])

  def testGetLayerBroadcastersFromRPSRaisesTypeError(self):
    with self.assertRaisesRegex(TypeError, 'Not a _LayerBroadcaster'):
      ragged_shape._get_layer_broadcasters_from_rps(int, [], [])

  def testGetBroadcasterRankDrop(self):
    with self.assertRaisesRegex(ValueError, 'Cannot broadcast'):
      a = RaggedShape._from_inner_shape([3, 4, 5])
      b = RaggedShape._from_inner_shape([4, 5])
      ragged_shape._get_broadcaster(a, b)

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
      ragged_shape._broadcast_dynamic_shape_next_layer_half_ragged(
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
      ragged_shape._broadcast_dynamic_shape_next_layer_both_uniform(
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
      ragged_shape._broadcast_dynamic_shape_next_layer(
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


class RaggedShapeErrorTest(parameterized.TestCase):

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
        expected_shape = RaggedShape.from_lengths(expected_lengths)

        rt = ragged_shape.broadcast_to(origin, expected_shape)
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
          error_regex='RowPartitions in RaggedShape do not'),
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
        rts = RaggedShape(
            row_partitions, inner_shape, dtype=dtype, validate=validate)
        sess.run([rts.inner_shape])

  def testRankNone(self):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    def foo(x):
      rts = RaggedShape._from_inner_shape(x)
      self.assertIsNone(rts.rank)

    foo([3, 7, 5])

  def testNumSlicesInDimensionRankNone(self):
    with self.assertRaisesRegex(ValueError, 'rank is undefined'):

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
      def foo(x):
        rts = RaggedShape._from_inner_shape(x)
        rts._num_slices_in_dimension(-1)

      foo([3, 7, 5])

  def testGetItemRankNone(self):
    with self.assertRaisesRegex(ValueError, 'Rank must be known to'):

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
      def foo(x):
        rts = RaggedShape._from_inner_shape(x)
        rts[-1]  # pylint: disable=pointless-statement

      foo([3, 7, 5])

  def testWithDenseRankRankNone(self):
    with self.assertRaisesRegex(ValueError, 'Rank must be known to'):

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
      def foo(x):
        rts = RaggedShape._from_inner_shape(x)
        rts.with_inner_rank(1)

      foo([3, 7, 5])

  def testWithRaggedRankRankNone(self):
    with self.assertRaisesRegex(ValueError, 'Rank must be known to'):

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
      def foo(x):
        rts = RaggedShape._from_inner_shape(x)
        rts._with_num_row_partitions(1)

      foo([3, 7, 5])

  def testAsRowPartitionsRankNone(self):
    # Error is readable, but does not match strings correctly.
    with self.assertRaisesRegex(ValueError, ''):

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
      def foo(x):
        rts = RaggedShape._from_inner_shape(x)
        rts._as_row_partitions()

      foo([3, 7, 5])

  def testBroadcastDynamicShapeExtendedRankNone(self):
    with self.assertRaisesRegex(ValueError, 'Rank of both shapes'):

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
      def foo(x):
        a = RaggedShape._from_inner_shape(x)
        b = RaggedShape._from_inner_shape([1, 1, 1])
        ragged_shape.broadcast_dynamic_shape_extended(a, b)

      foo([3, 7, 5])

  def testBroadcastDynamicShapeUnmatchedTypes6432(self):
    shape_int64 = RaggedShape.from_lengths([3, (0, 2, 3)], dtype=dtypes.int64)
    shape_int32 = RaggedShape.from_lengths([3, (0, 2, 3)], dtype=dtypes.int32)
    with self.assertRaisesRegex(ValueError, "Dtypes don't match"):
      ragged_shape.broadcast_dynamic_shape(shape_int64, shape_int32)

  def testBroadcastDynamicShapeUnmatchedTypes3264(self):
    shape_int64 = RaggedShape.from_lengths([3, (0, 2, 3)], dtype=dtypes.int64)
    shape_int32 = RaggedShape.from_lengths([3, (0, 2, 3)], dtype=dtypes.int32)
    with self.assertRaisesRegex(ValueError, "Dtypes don't match"):
      ragged_shape.broadcast_dynamic_shape(shape_int32, shape_int64)

  def testGetIdentityBroadcasterRankNone(self):
    with self.assertRaisesRegex(ValueError, 'Shape must have a'):

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
      def foo(x):
        rts = RaggedShape._from_inner_shape(x)
        ragged_shape._get_identity_broadcaster(rts)

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
        rts_a = RaggedShape._from_inner_shape(x)
        rts_b = RaggedShape._from_inner_shape(x)
        ragged_shape._get_broadcaster(rts_a, rts_b)

      foo([3, 7, 5])

  def testFromTensorDType(self):
    x = ragged_factory_ops.constant([[1, 2]])
    self.assertEqual(x.row_splits.dtype, dtypes.int64)
    shape_x = RaggedShape.from_tensor(x)
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


if __name__ == '__main__':
  googletest.main()
