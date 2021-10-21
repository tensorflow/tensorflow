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
"""Tests for RaggedTensor.from_tensor."""

from absl.testing import parameterized

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorFromTensorOpTest(test_util.TensorFlowTestCase,
                                   parameterized.TestCase):

  def testDocStringExamples(self):
    # The examples from RaggedTensor.from_tensor.__doc__.
    dt = constant_op.constant([[5, 7, 0], [0, 3, 0], [6, 0, 0]])
    self.assertAllEqual(
        RaggedTensor.from_tensor(dt), [[5, 7, 0], [0, 3, 0], [6, 0, 0]])

    self.assertAllEqual(
        RaggedTensor.from_tensor(dt, lengths=[1, 0, 3]), [[5], [], [6, 0, 0]])

    self.assertAllEqual(
        RaggedTensor.from_tensor(dt, padding=0), [[5, 7], [0, 3], [6]])

    dt_3d = constant_op.constant([[[5, 0], [7, 0], [0, 0]],
                                  [[0, 0], [3, 0], [0, 0]],
                                  [[6, 0], [0, 0], [0, 0]]])
    self.assertAllEqual(
        RaggedTensor.from_tensor(dt_3d, lengths=([2, 0, 3], [1, 1, 2, 0, 1])),
        [[[5], [7]], [], [[6, 0], [], [0]]])

  @parameterized.parameters(
      # 2D test cases, no length or padding.
      {
          'tensor': [[]],
          'expected': [[]],
          'expected_shape': [1, 0],
      },
      {
          'tensor': [[1]],
          'expected': [[1]],
          'expected_shape': [1, 1],
      },
      {
          'tensor': [[1, 2]],
          'expected': [[1, 2]],
          'expected_shape': [1, 2],
      },
      {
          'tensor': [[1], [2], [3]],
          'expected': [[1], [2], [3]],
          'expected_shape': [3, 1],
      },
      {
          'tensor': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
          'expected': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
          'expected_shape': [3, 3],
      },
      # 3D test cases, no length or padding
      {
          'tensor': [[[]]],
          'expected': [[[]]],
          'expected_shape': [1, 1, 0],
      },
      {
          'tensor': [[[]]],
          'expected': [[[]]],
          'ragged_rank': 1,
          'expected_shape': [1, 1, 0],
      },
      {
          'tensor': [[[1]]],
          'expected': [[[1]]],
          'expected_shape': [1, 1, 1],
      },
      {
          'tensor': [[[1, 2]]],
          'expected': [[[1, 2]]],
          'expected_shape': [1, 1, 2],
      },
      {
          'tensor': [[[1, 2], [3, 4]]],
          'expected': [[[1, 2], [3, 4]]],
          'expected_shape': [1, 2, 2],
      },
      {
          'tensor': [[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]],
          'expected': [[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]],
          'expected_shape': [4, 1, 2],
      },
      {
          'tensor': [[[1], [2]], [[3], [4]], [[5], [6]], [[7], [8]]],
          'expected': [[[1], [2]], [[3], [4]], [[5], [6]], [[7], [8]]],
          'expected_shape': [4, 2, 1],
      },
      # 2D test cases, with length
      {
          'tensor': [[1]],
          'lengths': [1],
          'expected': [[1]],
          'expected_shape': [1, None],
      },
      {
          'tensor': [[1]],
          'lengths': [0],
          'expected': [[]],
          'expected_shape': [1, None],
      },
      {
          'tensor': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
          'lengths': [0, 1, 2],
          'expected': [[], [4], [7, 8]],
          'expected_shape': [3, None],
      },
      {
          'tensor': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
          'lengths': [0, 0, 0],
          'expected': [[], [], []],
          'expected_shape': [3, None],
      },
      {
          'tensor': [[1, 2], [3, 4]],
          'lengths': [2, 2],
          'expected': [[1, 2], [3, 4]],
          'expected_shape': [2, None],
      },
      {
          'tensor': [[1, 2], [3, 4]],
          'lengths': [7, 8],  # lengths > ncols: truncated to ncols
          'expected': [[1, 2], [3, 4]],
          'expected_shape': [2, None],
      },
      {
          'tensor': [[1, 2], [3, 4]],
          'lengths': [-2, -1],  # lengths < 0: treated as zero
          'expected': [[], []],
          'expected_shape': [2, None],
      },
      # 3D test cases, with length
      {
          'tensor': [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
          'lengths': [0, 0],
          'expected': [[], []],
          'expected_shape': [2, None, 2],
      },
      {
          'tensor': [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
          'lengths': [1, 2],
          'expected': [[[1, 2]], [[5, 6], [7, 8]]],
          'expected_shape': [2, None, 2],
      },
      {
          'tensor': [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
          'lengths': [2, 2],
          'expected': [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
          'expected_shape': [2, None, 2],
      },
      # 2D test cases, with padding
      {
          'tensor': [[1]],
          'padding': 0,
          'expected': [[1]],
          'expected_shape': [1, None],
      },
      {
          'tensor': [[0]],
          'padding': 0,
          'expected': [[]],
          'expected_shape': [1, None],
      },
      {
          'tensor': [[0, 1]],
          'padding': 0,
          'expected': [[0, 1]],
          'expected_shape': [1, None],
      },
      {
          'tensor': [[1, 0]],
          'padding': 0,
          'expected': [[1]],
          'expected_shape': [1, None],
      },
      {
          'tensor': [[1, 0, 1, 0, 0, 1, 0, 0]],
          'padding': 0,
          'expected': [[1, 0, 1, 0, 0, 1]],
          'expected_shape': [1, None],
      },
      {
          'tensor': [[3, 7, 0, 0], [2, 0, 0, 0], [5, 0, 0, 0]],
          'padding': 0,
          'expected': [[3, 7], [2], [5]],
          'expected_shape': [3, None],
      },
      # 3D test cases, with padding
      {
          'tensor': [[[1]]],
          'padding': [0],
          'expected': [[[1]]],
          'expected_shape': [1, None, 1],
      },
      {
          'tensor': [[[0]]],
          'padding': [0],
          'expected': [[]],
          'expected_shape': [1, None, 1],
      },
      {
          'tensor': [[[0, 0], [1, 2]], [[3, 4], [0, 0]]],
          'padding': [0, 0],
          'expected': [[[0, 0], [1, 2]], [[3, 4]]],
          'expected_shape': [2, None, 2],
      },
      # 4D test cases, with padding
      {
          'tensor': [
              [[[1, 2], [3, 4]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
              [[[0, 0], [0, 0]], [[5, 6], [7, 8]], [[0, 0], [0, 0]]],
              [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]
          ],
          'padding': [[0, 0], [0, 0]],
          'expected': [
              [[[1, 2], [3, 4]]],
              [[[0, 0], [0, 0]], [[5, 6], [7, 8]]],
              []
          ],
          'expected_shape': [3, None, 2, 2],
      },
      # 3D test cases, with ragged_rank=2.
      {
          'tensor': [[[1, 0], [2, 3]], [[0, 0], [4, 0]]],
          'ragged_rank': 2,
          'expected': [[[1, 0], [2, 3]], [[0, 0], [4, 0]]],
          'expected_shape': [2, 2, 2],
      },
      {
          'tensor': [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
          'ragged_rank': 2,
          'lengths': [2, 0, 2, 1],
          'expected': [[[1, 2], []], [[5, 6], [7]]],
          'expected_shape': [2, 2, None],
      },
      {
          'tensor': [[[1, 0], [2, 3]], [[0, 0], [4, 0]]],
          'ragged_rank': 2,
          'padding': 0,
          'expected': [[[1], [2, 3]], [[], [4]]],
          'expected_shape': [2, 2, None],
      },
      # 4D test cases, with ragged_rank>1
      {
          'tensor': [[[[1, 0], [2, 3]], [[0, 0], [4, 0]]],
                     [[[5, 6], [7, 0]], [[0, 8], [0, 0]]]],
          'ragged_rank': 2,
          'expected': [[[[1, 0], [2, 3]], [[0, 0], [4, 0]]],
                       [[[5, 6], [7, 0]], [[0, 8], [0, 0]]]],
          'expected_shape': [2, 2, 2, 2],
      },
      {
          'tensor': [[[[1, 0], [2, 3]], [[0, 0], [4, 0]]],
                     [[[5, 6], [7, 0]], [[0, 8], [0, 0]]]],
          'ragged_rank': 3,
          'expected': [[[[1, 0], [2, 3]], [[0, 0], [4, 0]]],
                       [[[5, 6], [7, 0]], [[0, 8], [0, 0]]]],
          'expected_shape': [2, 2, 2, 2],
      },
      {
          'tensor': [[[[1, 0], [2, 3]], [[0, 0], [4, 0]]],
                     [[[5, 6], [7, 0]], [[0, 8], [0, 0]]]],
          'ragged_rank': 2,
          'padding': [0, 0],
          'expected': [[[[1, 0], [2, 3]], [[0, 0], [4, 0]]],
                       [[[5, 6], [7, 0]], [[0, 8]]]],
          'expected_shape': [2, 2, None, 2],
      },
      {
          'tensor': [[[[1, 0], [2, 3]], [[0, 0], [4, 0]]],
                     [[[5, 6], [7, 0]], [[0, 8], [0, 0]]]],
          'lengths': ([2, 2], [1, 2, 2, 1]),
          'expected': [[[[1, 0]], [[0, 0], [4, 0]]],
                       [[[5, 6], [7, 0]], [[0, 8]]]],
          'ragged_rank': 2,
          'use_ragged_rank': False,  # lengths contains nested_row_lengths.
          'expected_shape': [2, None, None, 2],
      },
      {
          'tensor': [[[[1, 0], [2, 3]], [[0, 0], [4, 0]]],
                     [[[5, 6], [7, 0]], [[0, 8], [0, 0]]]],
          'lengths': [[2, 2], [1, 2, 2, 1]],
          'expected': [[[[1, 0]], [[0, 0], [4, 0]]],
                       [[[5, 6], [7, 0]], [[0, 8]]]],
          'ragged_rank': 2,
          'use_ragged_rank': False,  # lengths contains nested_row_lengths.
          'expected_shape': [2, None, None, 2],
      },
      {
          'tensor': [[[[1, 0], [2, 3]], [[0, 0], [4, 0]]],
                     [[[5, 6], [7, 0]], [[0, 8], [0, 0]]]],
          'ragged_rank': 3,
          'padding': 0,
          'expected': [[[[1], [2, 3]], [[], [4]]],
                       [[[5, 6], [7]], [[0, 8], []]]],
          'expected_shape': [2, 2, 2, None],
      },
      {
          'tensor': [[[[1, 0], [2, 3]], [[0, 0], [4, 0]]],
                     [[[5, 6], [7, 0]], [[0, 8], [0, 0]]]],
          'lengths': ([2, 2], [2, 2, 2, 2], [1, 2, 0, 1, 2, 1, 2, 0]),
          'expected': [[[[1], [2, 3]], [[], [4]]],
                       [[[5, 6], [7]], [[0, 8], []]]],
          'ragged_rank': 3,
          'use_ragged_rank': False,  # lengths contains nested_row_lengths.
          'expected_shape': [2, None, None, None],
      },
      {
          'tensor': [[[[1, 0], [2, 3]], [[0, 0], [4, 0]]],
                     [[[5, 6], [7, 0]], [[0, 8], [0, 0]]]],
          'lengths': [[2, 2], [2, 2, 2, 2], [1, 2, 0, 1, 2, 1, 2, 0]],
          'expected': [[[[1], [2, 3]], [[], [4]]],
                       [[[5, 6], [7]], [[0, 8], []]]],
          'ragged_rank': 3,
          'use_ragged_rank': False,  # lengths contains nested_row_lengths.
          'expected_shape': [2, None, None, None],
      },
  )  # pyformat: disable
  def testRaggedFromTensor(self,
                           tensor,
                           expected,
                           lengths=None,
                           padding=None,
                           ragged_rank=1,
                           use_ragged_rank=True,
                           expected_shape=None):
    dt = constant_op.constant(tensor)
    if use_ragged_rank:
      rt = RaggedTensor.from_tensor(dt, lengths, padding, ragged_rank)
    else:
      rt = RaggedTensor.from_tensor(dt, lengths, padding)
    self.assertEqual(type(rt), RaggedTensor)
    self.assertEqual(rt.ragged_rank, ragged_rank)
    self.assertTrue(
        dt.shape.is_compatible_with(rt.shape),
        '%s is incompatible with %s' % (dt.shape, rt.shape))
    if expected_shape is not None:
      self.assertEqual(rt.shape.as_list(), expected_shape)
    self.assertAllEqual(rt, expected)
    self.assertAllEqual(rt, RaggedTensor.from_nested_row_splits(
        rt.flat_values, rt.nested_row_splits, validate=True))

  def testHighDimensions(self):
    # Use distinct prime numbers for all dimension shapes in this test, so
    # we can see any errors that are caused by mixing up dimension sizes.
    dt = array_ops.reshape(
        math_ops.range(3 * 5 * 7 * 11 * 13 * 17), [3, 5, 7, 11, 13, 17])
    for ragged_rank in range(1, 4):
      rt = RaggedTensor.from_tensor(dt, ragged_rank=ragged_rank)
      self.assertEqual(type(rt), RaggedTensor)
      self.assertEqual(rt.ragged_rank, ragged_rank)
      self.assertTrue(
          dt.shape.is_compatible_with(rt.shape),
          '%s is incompatible with %s' % (dt.shape, rt.shape))
      self.assertAllEqual(rt, self.evaluate(dt).tolist())
      self.assertAllEqual(rt, RaggedTensor.from_nested_row_splits(
          rt.flat_values, rt.nested_row_splits, validate=True))

  @parameterized.parameters(
      # With no padding or lengths
      {
          'dt_shape': [0, 0],
          'expected': []
      },
      {
          'dt_shape': [0, 3],
          'expected': []
      },
      {
          'dt_shape': [3, 0],
          'expected': [[], [], []]
      },
      {
          'dt_shape': [0, 2, 3],
          'expected': []
      },
      {
          'dt_shape': [1, 0, 0],
          'expected': [[]]
      },
      {
          'dt_shape': [2, 0, 3],
          'expected': [[], []]
      },
      {
          'dt_shape': [2, 3, 0],
          'expected': [[[], [], []], [[], [], []]]
      },
      {
          'dt_shape': [2, 3, 0, 1],
          'expected': [[[], [], []], [[], [], []]]
      },
      {
          'dt_shape': [2, 3, 1, 0],
          'expected': [[[[]], [[]], [[]]], [[[]], [[]], [[]]]]
      },
      # With padding
      {
          'dt_shape': [0, 0],
          'padding': 0,
          'expected': []
      },
      {
          'dt_shape': [0, 3],
          'padding': 0,
          'expected': []
      },
      {
          'dt_shape': [3, 0],
          'padding': 0,
          'expected': [[], [], []]
      },
      {
          'dt_shape': [0, 2, 3],
          'padding': [0, 0, 0],
          'expected': []
      },
      {
          'dt_shape': [2, 0, 3],
          'padding': [0, 0, 0],
          'expected': [[], []]
      },
      {
          'dt_shape': [2, 3, 0],
          'padding': [],
          'expected': [[], []]
      },
      # With lengths
      {
          'dt_shape': [0, 0],
          'lengths': [],
          'expected': []
      },
      {
          'dt_shape': [0, 3],
          'lengths': [],
          'expected': []
      },
      {
          'dt_shape': [3, 0],
          'lengths': [0, 0, 0],
          'expected': [[], [], []]
      },
      {
          'dt_shape': [3, 0],
          'lengths': [2, 3, 4],  # lengths > ncols: truncated to ncols
          'expected': [[], [], []]
      },
      {
          'dt_shape': [0, 2, 3],
          'lengths': [],
          'expected': []
      },
      {
          'dt_shape': [2, 0, 3],
          'lengths': [0, 0],
          'expected': [[], []]
      },
      {
          'dt_shape': [2, 3, 0],
          'lengths': [0, 0],
          'expected': [[], []]
      },
  )
  def testEmpty(self, dt_shape, expected, lengths=None, padding=None):
    dt = array_ops.zeros(dt_shape)
    for ragged_rank in range(1, len(dt_shape) - 1):
      rt = RaggedTensor.from_tensor(dt, lengths, padding, ragged_rank)
      self.assertEqual(type(rt), RaggedTensor)
      self.assertEqual(rt.ragged_rank, ragged_rank)
      self.assertTrue(dt.shape.is_compatible_with(rt.shape))
      self.assertAllEqual(rt, expected)
      self.assertAllEqual(rt, RaggedTensor.from_nested_row_splits(
          rt.flat_values, rt.nested_row_splits, validate=True))

  @parameterized.named_parameters([
      {
          'testcase_name': '2D_UnknownRank',
          'tensor': [[1, 2], [3, 4]],
          'tensor_shape': None,
      },
      {
          'testcase_name': '2D_Shape_None_None',
          'tensor': [[1, 2], [3, 4]],
          'tensor_shape': [None, None],
      },
      {
          'testcase_name': '2D_Shape_2_None',
          'tensor': [[1, 2], [3, 4]],
          'tensor_shape': [2, None],
      },
      {
          'testcase_name': '2D_Shape_None_2',
          'tensor': [[1, 2], [3, 4]],
          'tensor_shape': [None, 2],
      },
      {
          'testcase_name': '4D_UnknownRank',
          'tensor': np.ones([4, 3, 2, 1]),
          'tensor_shape': None,
      },
      {
          'testcase_name': '4D_Shape_None_None_None_None',
          'tensor': np.ones([4, 3, 2, 1]),
          'tensor_shape': [None, None, None, None],
      },
      {
          'tensor': np.ones([4, 3, 2, 1]),
          'tensor_shape': [4, None, None, 1],
          'testcase_name': '4D_Shape_4_None_None_1',
      },
  ])
  def testPartialShapes(self, tensor, tensor_shape, shape=None,
                        expected=None):
    if expected is None:
      expected = tensor

    if context.executing_eagerly():
      return  # static shapes are always fully defined in eager mode.

    dt = constant_op.constant(tensor)
    for ragged_rank in range(1, len(dt.shape) - 1):
      dt_placeholder = array_ops.placeholder_with_default(tensor, tensor_shape)
      rt = RaggedTensor.from_tensor(dt_placeholder, ragged_rank=ragged_rank)
      self.assertIsInstance(rt, RaggedTensor)
      self.assertEqual(rt.ragged_rank, ragged_rank)
      self.assertTrue(
          dt.shape.is_compatible_with(rt.shape),
          '%s is incompatible with %s' % (dt.shape, rt.shape))
      if shape is not None:
        self.assertEqual(rt.shape.as_list(), shape)
      self.assertAllEqual(rt, expected.tolist())
      self.assertAllEqual(rt, RaggedTensor.from_nested_row_splits(
          rt.flat_values, rt.nested_row_splits, validate=True))

  @parameterized.parameters(
      {
          'tensor': [[1]],
          'lengths': [0],
          'padding':
              0,
          'error': (ValueError,
                    'Specify argument `lengths` or `padding`, but not both.')
      },
      {
          'tensor': [[1]],
          'lengths': [0.5],
          'error': (
              TypeError,
              r'Argument `tensor` \(name\: lengths\) must be of type integer.*')
      },
      {
          'tensor': [[1, 2, 3]],
          'lengths': [[1], [1]],
          'error': (ValueError, r'Shape \(1, 3\) must have rank at least 3')
      },
      {
          'tensor': [[1]],
          'padding': 'a',
          'error': (TypeError, '.*')
      },
      {
          'tensor': [[1]],
          'padding': [1],
          'error': (ValueError, r'Shapes \(1,\) and \(\) are incompatible')
      },
      {
          'tensor': [[[1]]],
          'padding': 1,
          'error': (ValueError, r'Shapes \(\) and \(1,\) are incompatible')
      },
      {
          'tensor': [[1]],
          'ragged_rank':
              'bad',
          'error': (TypeError,
                    r'Argument `ragged_rank` must be an int. Received bad.')
      },
      {
          'tensor': [[1]],
          'ragged_rank':
              0,
          'error':
              (ValueError,
               r'Argument `ragged_rank` must be greater than 0. Received 0.')
      },
      {
          'tensor': [[1]],
          'ragged_rank':
              -1,
          'error':
              (ValueError,
               r'Argument `ragged_rank` must be greater than 0. Received -1.')
      },
      {
          'tensor': [[[[1, 0], [2, 3]], [[0, 0], [4, 0]]],
                     [[[5, 6], [7, 0]], [[0, 8], [0, 0]]]],
          'lengths': ([2, 2], [2, 2, 2, 2]),
          'ragged_rank':
              3,
          'error':
              (ValueError,
               r'If Argument `lengths` is a tuple of row_lengths, argument '
               r'`ragged_rank` must be len\(lengths\): 2. Received '
               r'ragged_rank: 3.')
      },
  )
  def testErrors(self,
                 tensor,
                 lengths=None,
                 padding=None,
                 ragged_rank=1,
                 error=None):
    dt = constant_op.constant(tensor)
    self.assertRaisesRegex(error[0], error[1], RaggedTensor.from_tensor, dt,
                           lengths, padding, ragged_rank)


if __name__ == '__main__':
  googletest.main()
