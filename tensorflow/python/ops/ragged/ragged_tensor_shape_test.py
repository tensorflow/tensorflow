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
"""Tests for tf.ragged.ragged_tensor_shape."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import ragged
from tensorflow.python.platform import googletest


class RaggedTensorShapeTest(test_util.TensorFlowTestCase,
                            parameterized.TestCase):

  def assertShapeEq(self, x, y):
    assert isinstance(x, ragged.RaggedTensorDynamicShape)
    assert isinstance(y, ragged.RaggedTensorDynamicShape)
    x_partitioned_dim_sizes = [
        splits.eval().tolist()  #
        for splits in x.partitioned_dim_sizes
    ]
    y_partitioned_dim_sizes = [
        splits.eval().tolist()  #
        for splits in y.partitioned_dim_sizes
    ]
    self.assertEqual(x_partitioned_dim_sizes, y_partitioned_dim_sizes)
    self.assertEqual(x.inner_dim_sizes.eval().tolist(),
                     y.inner_dim_sizes.eval().tolist())

  @parameterized.parameters([
      dict(value='x', expected_dim_sizes=[]),
      dict(value=['a', 'b', 'c'], expected_dim_sizes=[3]),
      dict(value=[['a', 'b', 'c'], ['d', 'e', 'f']], expected_dim_sizes=[2, 3]),
      dict(
          value=[[['a', 'b', 'c'], ['d', 'e', 'f']]],
          expected_dim_sizes=[1, 2, 3]),
      dict(
          value=ragged.constant_value([['a', 'b', 'c'], ['d', 'e']]),
          expected_dim_sizes=[2, [3, 2]]),
      dict(
          value=ragged.constant_value([[['a', 'b', 'c'], ['d', 'e']]]),
          expected_dim_sizes=[1, [2], [3, 2]]),
      dict(
          value=ragged.constant_value([[['a', 'b', 'c'], ['d', 'e', 'f']]],
                                      ragged_rank=1),
          expected_dim_sizes=[1, [2], 3]),
      dict(
          value=ragged.constant_value([[[[1], [2]], [[3], [4]]],
                                       [[[5], [6]]]], ragged_rank=1),
          expected_dim_sizes=[2, [2, 1], 2, 1]),
      dict(
          value=ragged.constant_value([[10, 20], [30]]),
          expected_dim_sizes=[2, [2, 1]]),
      # Docstring examples:
      dict(value=[[1, 2, 3], [4, 5, 6]], expected_dim_sizes=[2, 3]),
      dict(
          value=ragged.constant_value([[1, 2], [], [3, 4, 5]]),
          expected_dim_sizes=[3, [2, 0, 3]]),
      dict(
          value=ragged.constant_value([[[1, 2], [3, 4]], [[5, 6]]],
                                      ragged_rank=1),
          expected_dim_sizes=[2, [2, 1], 2]),
      dict(
          value=ragged.constant_value([[[1, 2], [3]], [[4, 5]]]),
          expected_dim_sizes=[2, [2, 1], [2, 1, 2]]),
  ])
  @test_util.run_v1_only('b/120545219')
  def testFromTensor(self, value, expected_dim_sizes):
    shape = ragged.RaggedTensorDynamicShape.from_tensor(value)
    expected = ragged.RaggedTensorDynamicShape.from_dim_sizes(
        expected_dim_sizes)
    with self.cached_session():
      self.assertShapeEq(shape, expected)

  @parameterized.parameters([
      dict(dim_sizes=[], rank=0, expected_dim_sizes=[]),
      dict(dim_sizes=[], rank=3, expected_dim_sizes=[1, 1, 1]),
      dict(dim_sizes=[3], rank=1, expected_dim_sizes=[3]),
      dict(dim_sizes=[3], rank=3, expected_dim_sizes=[1, 1, 3]),
      dict(dim_sizes=[2, 3], rank=3, expected_dim_sizes=[1, 2, 3]),
      dict(dim_sizes=[3, [3, 2, 4]], rank=2, expected_dim_sizes=[3, [3, 2, 4]]),
      dict(
          dim_sizes=[3, [3, 2, 4]],
          rank=4,
          expected_dim_sizes=[1, 1, 3, [3, 2, 4]]),
      dict(
          dim_sizes=[3, [3, 2, 4], 2, 3],
          rank=5,
          expected_dim_sizes=[1, 3, [3, 2, 4], 2, 3]),
  ])
  @test_util.run_v1_only('b/120545219')
  def testBroadcastToRank(self, dim_sizes, rank, expected_dim_sizes):
    shape = ragged.RaggedTensorDynamicShape.from_dim_sizes(dim_sizes)
    expected = ragged.RaggedTensorDynamicShape.from_dim_sizes(
        expected_dim_sizes)
    broadcasted_shape = shape.broadcast_to_rank(rank)
    with self.cached_session():
      self.assertShapeEq(broadcasted_shape, expected)
      self.assertEqual(broadcasted_shape.rank, rank)

  @parameterized.parameters([
      #=========================================================================
      # dimension[axis] is uniform inner; and row_lengths is a scalar
      #=========================================================================
      # shape: [BROADCAST(UNIFORM), UNIFORM, UNIFORM]
      dict(axis=0,
           row_length=3,
           original_dim_sizes=[1, 4, 5],
           broadcast_dim_sizes=[3, 4, 5]),

      # shape: [UNIFORM, UNIFORM, BROADCAST(UNIFORM)]
      dict(axis=2,
           row_length=5,
           original_dim_sizes=[3, 4, 1],
           broadcast_dim_sizes=[3, 4, 5]),

      # shape: [UNIFORM, RAGGED, BROADCAST(UNIFORM)]
      dict(axis=2,
           row_length=5,
           original_dim_sizes=[3, [3, 2, 8], 1],
           broadcast_dim_sizes=[3, [3, 2, 8], 5]),

      # shape: [UNIFORM, RAGGED, RAGGED, UNIFORM, UNIFORM, BROADCAST(UNIFORM)]
      dict(axis=5,
           row_length=5,
           original_dim_sizes=[2, [2, 1], [3, 2, 8], 3, 4, 1],
           broadcast_dim_sizes=[2, [2, 1], [3, 2, 8], 3, 4, 5]),

      #=========================================================================
      # dimension[axis] is uniform inner; and row_lengths is a vector
      #=========================================================================
      # shape: [UNIFORM, BROADCAST(UNIFORM)]
      dict(axis=1,
           row_length=[2, 0, 1],
           original_dim_sizes=[3, 1],
           broadcast_dim_sizes=[3, [2, 0, 1]]),
      # shape: [UNIFORM, BROADCAST(UNIFORM), UNIFORM]
      dict(axis=1,
           row_length=[2, 0, 1],
           original_dim_sizes=[3, 1, 5],
           broadcast_dim_sizes=[3, [2, 0, 1], 5]),

      # shape: [UNIFORM, UNIFORM, BROADCAST(UNIFORM)]
      dict(axis=2,
           row_length=[2, 0, 1, 3, 8, 2, 3, 4, 1, 8, 7, 0],
           original_dim_sizes=[4, 3, 1],
           broadcast_dim_sizes=[4, 3, [2, 0, 1, 3, 8, 2, 3, 4, 1, 8, 7, 0]]),

      # shape: [UNIFORM, RAGGED, BROADCAST(UNIFORM)]
      dict(axis=2,
           row_length=[2, 5, 3],
           original_dim_sizes=[2, [2, 1], 1],
           broadcast_dim_sizes=[2, [2, 1], [2, 5, 3]]),

      # shape: [UNIFORM, RAGGED, UNIFORM, UNIFORM, BROADCAST(UNIFORM), UNIFORM]
      dict(axis=4,
           row_length=list(range(18)),
           original_dim_sizes=[2, [2, 1], 3, 2, 1, 8],
           broadcast_dim_sizes=[2, [2, 1], 3, 2, list(range(18)), 8]),

      #=========================================================================
      # dimension[axis] is uniform partitioned; and row_lengths is a scalar
      #=========================================================================
      # shape: [BROADCAST(UNIFORM), RAGGED]
      dict(axis=0,
           row_length=3,
           original_dim_sizes=[1, [5]],
           broadcast_dim_sizes=[3, [5, 5, 5]]),

      # shape: [BROADCAST(UNIFORM), UNIFORM, RAGGED]
      dict(axis=0,
           row_length=2,
           original_dim_sizes=[1, 3, [3, 0, 2]],
           broadcast_dim_sizes=[2, 3, [3, 0, 2, 3, 0, 2]]),

      # shape: [BROADCAST(UNIFORM), RAGGED, RAGGED, UNIFORM, UNIFORM]
      dict(axis=0,
           row_length=3,
           original_dim_sizes=[1, [3], [3, 5, 2], 9, 4, 5],
           broadcast_dim_sizes=[3, [3, 3, 3], [3, 5, 2, 3, 5, 2, 3, 5, 2],
                                9, 4, 5]),

      # shape: [BROADCAST(UNIFORM), UNIFORM, RAGGED, UNIFORM]
      dict(axis=0,
           row_length=2,
           original_dim_sizes=[1, 2, [2, 1], [3, 5, 2], 2],
           broadcast_dim_sizes=[2, 2, [2, 1, 2, 1], [3, 5, 2, 3, 5, 2], 2]),

      # shape: [UNIFORM, BROADCAST(UNIFORM), RAGGED, UNIFORM]
      dict(axis=1,
           row_length=2,
           original_dim_sizes=[3, 1, [4, 0, 2], 5],
           broadcast_dim_sizes=[3, 2, [4, 0, 2, 4, 0, 2], 5]),

      # shape: [UNIFORM, BROADCAST(UNIFORM), RAGGED]
      dict(axis=1,
           row_length=1,
           original_dim_sizes=[2, 3, (1, 2, 3, 4, 5, 6)],
           broadcast_dim_sizes=[2, 3, (1, 2, 3, 4, 5, 6)]),

      #=========================================================================
      # dimension[axis] is uniform partitioned; and row_lengths is a vector
      #=========================================================================
      # shape: [UNIFORM, BROADCAST(UNIFORM), RAGGED, UNIFORM]
      dict(axis=1,
           row_length=[4, 1, 2],
           original_dim_sizes=[
               3,                          # axis=0
               1,                          # axis=1 (broadcast)
               [3, 1, 2],                  # axis=2
               5],                         # axis=3
           broadcast_dim_sizes=[
               3,                          # axis=0
               [4, 1, 2],                  # axis=1 (broadcast)
               [3, 3, 3, 3, 1, 2, 2],      # axis=2
               5]),                        # axis=3

      # shape: [UNIFORM, BROADCAST(UNIFORM), RAGGED, RAGGED]
      dict(axis=1,
           row_length=[2, 0, 3],
           original_dim_sizes=[
               3,                                         # axis=0
               1,                                         # axis=1 (broadcast)
               [3, 1, 2],                                 # axis=2
               [3, 1, 4, 1, 5, 9]],                       # axis=3
           broadcast_dim_sizes=[
               3,                                         # axis=0
               [2, 0, 3],                                 # axis=1 (broadcast)
               [3, 3, 2, 2, 2],                           # axis=2
               [3, 1, 4, 3, 1, 4, 5, 9, 5, 9, 5, 9]]),    # axis=3

      # shape: [UNIFORM, RAGGED, BROADCAST(UNIFORM), RAGGED, RAGGED, UNIFORM]
      dict(axis=2,
           row_length=[4, 1, 2],
           original_dim_sizes=[
               3,                                         # axis=0
               [2, 0, 1],                                 # axis=1
               1,                                         # axis=2 (broadcast)
               [3, 2, 1],                                 # axis=3
               [1, 0, 1, 0, 2, 3],                        # axis=4
               5],                                        # axis=5
           broadcast_dim_sizes=[
               3,                                         # axis=0
               [2, 0, 1],                                 # axis=2
               [4, 1, 2],                                 # axis=2 (broadcast)
               [3, 3, 3, 3, 2, 1, 1],                     # axis=3
               [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0,    # axis=4
                2, 3, 3],
               5]),                                       # axis=5

      dict(axis=0,
           row_length=2,
           original_dim_sizes=[1, 1, 2, (2, 1)],
           broadcast_dim_sizes=[2, 1, 2, (2, 1, 2, 1)]),
      dict(axis=1,
           row_length=(2, 1),
           original_dim_sizes=[2, 1, 2, (2, 1, 2, 1)],
           broadcast_dim_sizes=[2, (2, 1), 2, (2, 1, 2, 1, 2, 1)]),
      dict(axis=2,
           row_length=2,
           original_dim_sizes=[2, (2, 1), 2, (2, 1, 2, 1, 2, 1)],
           broadcast_dim_sizes=[2, (2, 1), 2, (2, 1, 2, 1, 2, 1)]),
      dict(axis=3,
           row_length=(2, 1, 2, 1, 2, 1),
           original_dim_sizes=[2, (2, 1), 2, 1],
           broadcast_dim_sizes=[2, (2, 1), 2, (2, 1, 2, 1, 2, 1)]),
  ])  # pyformat: disable
  @test_util.run_v1_only('b/120545219')
  def testBroadcastDimension(self, axis, row_length, original_dim_sizes,
                             broadcast_dim_sizes):
    """Tests for the broadcast_dimension method.

    Verifies that:

    * `original.broadcast_dimension(axis, row_length) == broadcast`
    * `broadcast.broadcast_dimension(axis, row_length) == broadcast`
    * `broadcast.broadcast_dimension(axis, 1) == broadcast`

    Args:
      axis: The axis to broadcast
      row_length: The slice lengths to broadcast to.
      original_dim_sizes: The dimension sizes before broadcasting.
        original_dim_sizes[axis] should be equal to `1` or `row_length`.
      broadcast_dim_sizes: THe dimension sizes after broadcasting.
    """
    original_shape = ragged.RaggedTensorDynamicShape.from_dim_sizes(
        original_dim_sizes)
    broadcast_shape = ragged.RaggedTensorDynamicShape.from_dim_sizes(
        broadcast_dim_sizes)
    self.assertEqual(original_shape.rank, broadcast_shape.rank)
    with self.cached_session():
      # shape[axis].value == 1 and row_length > 1:
      bcast1 = original_shape.broadcast_dimension(axis, row_length)
      # shape[axis].value > 1 and row_length == shape[axis].value:
      bcast2 = broadcast_shape.broadcast_dimension(axis, row_length)
      # shape[axis].value > 1 and row_length == 1:
      bcast3 = broadcast_shape.broadcast_dimension(axis, 1)

      self.assertShapeEq(bcast1, broadcast_shape)
      self.assertShapeEq(bcast2, broadcast_shape)
      self.assertShapeEq(bcast3, broadcast_shape)

  @parameterized.parameters(
      [
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
              x_dims=[3],
              y_dims=[3, (2, 3, 1), 1],
              expected_dims=[3, (2, 3, 1), 3]),
          dict(x_dims=[1], y_dims=[3, (2, 1, 3)], expected_dims=[3, (2, 1, 3)]),
          dict(
              x_dims=[1],
              y_dims=[3, (2, 1, 3), 8],
              expected_dims=[3, (2, 1, 3), 8]),
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
  @test_util.run_v1_only('b/120545219')
  def testBroadcastDynamicShape(self, x_dims, y_dims, expected_dims):
    x_shape = ragged.RaggedTensorDynamicShape.from_dim_sizes(x_dims)
    y_shape = ragged.RaggedTensorDynamicShape.from_dim_sizes(y_dims)
    expected = ragged.RaggedTensorDynamicShape.from_dim_sizes(expected_dims)
    result1 = ragged.broadcast_dynamic_shape(x_shape, y_shape)
    result2 = ragged.broadcast_dynamic_shape(y_shape, x_shape)
    with self.cached_session():
      self.assertShapeEq(expected, result1)
      self.assertShapeEq(expected, result2)

  def testRepr(self):
    shape = ragged.RaggedTensorDynamicShape.from_dim_sizes([2, (2, 1), 2, 1])
    self.assertRegexpMatches(
        repr(shape),
        r'RaggedTensorDynamicShape\('
        r'partitioned_dim_sizes=\(<[^>]+>, <[^>]+>\), '
        r'inner_dim_sizes=<[^>]+>\)')

  @parameterized.parameters([
      dict(
          x=[[10], [20], [30]],  # shape=[3, 1]
          dim_sizes=[3, 2],
          expected=[[10, 10], [20, 20], [30, 30]]),
      dict(
          x=[[10], [20], [30]],  # shape=[3, 1]
          dim_sizes=[3, [3, 0, 2]],
          expected=ragged.constant_value([[10, 10, 10], [], [30, 30]],
                                         dtype=np.int32)),
      dict(
          x=[[[1, 2, 3]], [[4, 5, 6]]],  # shape = [2, 1, 3]
          dim_sizes=[2, [2, 3], 3],
          expected=ragged.constant_value(
              [[[1, 2, 3], [1, 2, 3]], [[4, 5, 6], [4, 5, 6], [4, 5, 6]]],
              dtype=np.int32,
              ragged_rank=1)),
      dict(
          x=[[[1]], [[2]]],  # shape = [2, 1, 1]
          dim_sizes=[2, [2, 3], [0, 2, 1, 2, 0]],
          expected=ragged.constant_value([[[], [1, 1]], [[2], [2, 2], []]],
                                         dtype=np.int32,
                                         ragged_rank=2)),
      dict(
          x=10,
          dim_sizes=[3, [3, 0, 2]],
          expected=ragged.constant_value([[10, 10, 10], [], [10, 10]])),
  ])
  @test_util.run_v1_only('b/120545219')
  def testRaggedBroadcastTo(self, x, dim_sizes, expected):
    shape = ragged.RaggedTensorDynamicShape.from_dim_sizes(dim_sizes)
    result = ragged.broadcast_to(x, shape)
    with self.cached_session():
      self.assertEqual(
          getattr(result, 'ragged_rank', 0), getattr(expected, 'ragged_rank',
                                                     0))
      if hasattr(expected, 'tolist'):
        expected = expected.tolist()
      self.assertEqual(result.eval().tolist(), expected)

  @parameterized.parameters([
      dict(
          doc='x.shape=[3, (D1)]; y.shape=[3, 1]; bcast.shape=[3, (D1)]',
          x=ragged.constant_value([[1, 2, 3], [], [4, 5]], dtype=np.int32),
          y=[[10], [20], [30]],
          expected=ragged.constant_value([[11, 12, 13], [], [34, 35]])),
      dict(
          doc='x.shape=[3, (D1)]; y.shape=[]; bcast.shape=[3, (D1)]',
          x=ragged.constant_value([[1, 2, 3], [], [4, 5]], dtype=np.int32),
          y=10,
          expected=ragged.constant_value([[11, 12, 13], [], [14, 15]])),
      dict(
          doc='x.shape=[1, (D1)]; y.shape=[3, 1]; bcast.shape=[3, (D1)]',
          x=ragged.constant_value([[1, 2, 3]], dtype=np.int32),
          y=[[10], [20], [30]],
          expected=ragged.constant_value(
              [[11, 12, 13], [21, 22, 23], [31, 32, 33]], dtype=np.int32)),
      dict(
          doc=('x.shape=[2, (D1), 1]; y.shape=[1, (D2)]; '
               'bcast.shape=[2, (D1), (D2)]'),
          x=ragged.constant_value([[[1], [2], [3]], [[4]]], ragged_rank=1),
          y=ragged.constant_value([[10, 20, 30]]),
          expected=ragged.constant_value([[[11, 21, 31], [12, 22, 32],
                                           [13, 23, 33]], [[14, 24, 34]]])),
      dict(
          doc=('x.shape=[2, (D1), 1]; y.shape=[1, 1, 4]; '
               'bcast.shape=[2, (D1), 4]'),
          x=ragged.constant_value([[[10], [20]], [[30]]], ragged_rank=1),
          y=[[[1, 2, 3, 4]]],
          expected=ragged.constant_value(
              [[[11, 12, 13, 14], [21, 22, 23, 24]], [[31, 32, 33, 34]]],
              ragged_rank=1)),
      dict(
          doc=('x.shape=[2, (D1), 2, 1]; y.shape=[2, (D2)]; '
               'bcast.shape=[2, (D1), (2), (D2)'),
          x=ragged.constant_value([[[[1], [2]], [[3], [4]]],
                                   [[[5], [6]]]],
                                  ragged_rank=1),
          y=ragged.constant_value([[10, 20], [30]]),
          expected=ragged.constant_value(
              [[[[11, 21], [32]], [[13, 23], [34]]],
               [[[15, 25], [36]]]])),
  ])
  @test_util.run_v1_only('b/120545219')
  def testRaggedAddWithBroadcasting(self, x, y, expected, doc):
    expected_rrank = getattr(expected, 'ragged_rank', 0)
    x = ragged.convert_to_tensor_or_ragged_tensor(x, dtype=dtypes.int32)
    y = ragged.convert_to_tensor_or_ragged_tensor(y, dtype=dtypes.int32)
    result = x + y
    result_rrank = getattr(result, 'ragged_rank', 0)
    self.assertEqual(expected_rrank, result_rrank)
    if hasattr(expected, 'tolist'):
      expected = expected.tolist()
    with self.cached_session():
      self.assertEqual(result.eval().tolist(), expected)


if __name__ == '__main__':
  googletest.main()
