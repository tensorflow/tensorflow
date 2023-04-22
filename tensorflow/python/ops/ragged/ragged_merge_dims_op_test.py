# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for RaggedTensor.merge_dims."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest
from tensorflow.python.util import nest


@test_util.run_all_in_graph_and_eager_modes
class RaggedMergeDimsOpTest(test_util.TensorFlowTestCase,
                            parameterized.TestCase):

  @parameterized.named_parameters([
      {
          'testcase_name': '2DAxis0To1',
          'rt': [[1, 2], [], [3, 4, 5]],
          'outer_axis': 0,
          'inner_axis': 1,
          'expected': [1, 2, 3, 4, 5],
      },
      {
          'testcase_name': '3DAxis0To1',
          'rt': [[[1, 2], [], [3, 4, 5]], [[6], [7, 8], []]],
          'outer_axis': 0,
          'inner_axis': 1,
          'expected': [[1, 2], [], [3, 4, 5], [6], [7, 8], []],
      },
      {
          'testcase_name': '3DAxis1To2',
          'rt': [[[1, 2], [], [3, 4, 5]], [[6], [7, 8], []]],
          'outer_axis': 1,
          'inner_axis': 2,
          'expected': [[1, 2, 3, 4, 5], [6, 7, 8]],
      },
      {
          'testcase_name': '3DAxis0To2',
          'rt': [[[1, 2], [], [3, 4, 5]], [[6], [7, 8], []]],
          'outer_axis': 0,
          'inner_axis': 2,
          'expected': [1, 2, 3, 4, 5, 6, 7, 8],
      },
      {
          'testcase_name': '3DAxis0To1WithDenseValues',
          'rt': [[[1, 2], [3, 4], [5, 6]], [[7, 8]]],
          'ragged_ranks': (1, 2),
          'outer_axis': 0,
          'inner_axis': 1,
          'expected': [[1, 2], [3, 4], [5, 6], [7, 8]],
      },
      {
          'testcase_name': '3DAxis1To2WithDenseValues',
          'rt': [[[1, 2], [3, 4], [5, 6]], [[7, 8]]],
          'ragged_ranks': (1, 2),
          'outer_axis': 1,
          'inner_axis': 2,
          'expected': [[1, 2, 3, 4, 5, 6], [7, 8]],
      },
      {
          'testcase_name': '4DAxis0To1',
          'rt': [[[[1, 2], [], [3, 4, 5]], [[6], [7, 8], []]], [[[9], [0]]]],
          'outer_axis': 0,
          'inner_axis': 1,
          'expected': [[[1, 2], [], [3, 4, 5]], [[6], [7, 8], []], [[9], [0]]],
      },
      {
          'testcase_name': '4DAxis1To2',
          'rt': [[[[1, 2], [], [3, 4, 5]], [[6], [7, 8], []]], [[[9], [0]]]],
          'outer_axis': 1,
          'inner_axis': 2,
          'expected': [[[1, 2], [], [3, 4, 5], [6], [7, 8], []], [[9], [0]]],
      },
      {
          'testcase_name': '4DAxis2To3',
          'rt': [[[[1, 2], [], [3, 4, 5]], [[6], [7, 8], []]], [[[9], [0]]]],
          'outer_axis': 2,
          'inner_axis': 3,
          'expected': [[[1, 2, 3, 4, 5], [6, 7, 8]], [[9, 0]]],
      },
      {
          'testcase_name': '4DAxis1To3',
          'rt': [[[[1, 2], [], [3, 4, 5]], [[6], [7, 8], []]], [[[9], [0]]]],
          'outer_axis': 1,
          'inner_axis': 3,
          'expected': [[1, 2, 3, 4, 5, 6, 7, 8], [9, 0]],
      },
      {
          'testcase_name': '4DAxis1ToNeg1',
          'rt': [[[[1, 2], [], [3, 4, 5]], [[6], [7, 8], []]], [[[9], [0]]]],
          'outer_axis': 1,
          'inner_axis': -1,
          'expected': [[1, 2, 3, 4, 5, 6, 7, 8], [9, 0]],
      },
      {
          'testcase_name': '4DAxis1To2WithDenseValues',
          'rt': [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]]]],
          'ragged_ranks': (1, 2, 3),
          'outer_axis': 1,
          'inner_axis': 2,
          'expected': [[[1, 2], [3, 4], [5, 6], [7, 8]], [[9, 10], [11, 12]]],
      },
      {
          'testcase_name': '4DAxis2To3WithDenseValues',
          'rt': [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]]]],
          'ragged_ranks': (1, 2, 3),
          'outer_axis': 2,
          'inner_axis': 3,
          'expected': [[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12]]],
      },
      {
          'testcase_name': '4DAxis1To3WithDenseValues',
          'rt': [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]]]],
          'ragged_ranks': (1, 2, 3),
          'outer_axis': 1,
          'inner_axis': 3,
          'expected': [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12]],
      },
      {
          'testcase_name': '5DAxis2To3WithDenseValues',
          'rt': [[[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]],
                 [[[[9, 10], [11, 12]]]]],
          'ragged_ranks': (1, 2, 3, 4),
          'outer_axis': 2,
          'inner_axis': 3,
          'expected': [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                       [[[9, 10], [11, 12]]]],
      },
      {
          'testcase_name': '5DAxis3To4WithDenseValues',
          'rt': [[[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]],
                 [[[[9, 10], [11, 12]]]]],
          'ragged_ranks': (1, 2, 3, 4),
          'outer_axis': 3,
          'inner_axis': 4,
          'expected': [[[[1, 2, 3, 4]], [[5, 6, 7, 8]]], [[[9, 10, 11, 12]]]],
      },
      {
          'testcase_name': '5DAxis1To3WithDenseValues',
          'rt': [[[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]],
                 [[[[9, 10], [11, 12]]]]],
          'ragged_ranks': (1, 2, 3, 4),
          'outer_axis': 1,
          'inner_axis': 3,
          'expected': [[[1, 2], [3, 4], [5, 6], [7, 8]], [[9, 10], [11, 12]]],
      },
      {
          'testcase_name': 'OuterEqualsInner',
          'rt': [[1], [2], [3, 4]],
          'outer_axis': 0,
          'inner_axis': 0,
          'expected': [[1], [2], [3, 4]],
      },
      {
          'testcase_name': 'OuterEqualsInnerWithNegativeAxis',
          'rt': [[1], [2], [3, 4]],
          'outer_axis': 1,
          'inner_axis': -1,
          'expected': [[1], [2], [3, 4]],
      },
  ])  # pyformat: disable
  def testRaggedMergeDims(self,
                          rt,
                          outer_axis,
                          inner_axis,
                          expected,
                          ragged_ranks=(None,)):
    for ragged_rank in ragged_ranks:
      x = ragged_factory_ops.constant(rt, ragged_rank=ragged_rank)

      # Check basic behavior.
      actual = x.merge_dims(outer_axis, inner_axis)
      self.assertAllEqual(expected, actual)
      if outer_axis >= 0 and inner_axis >= 0:
        self.assertEqual(actual.shape.rank,
                         x.shape.rank - (inner_axis - outer_axis))

      # Check behavior with negative axis.
      if outer_axis >= 0 and inner_axis >= 0:
        actual_with_neg_axis = x.merge_dims(outer_axis - x.shape.rank,
                                            inner_axis - x.shape.rank)
        self.assertAllEqual(expected, actual_with_neg_axis)

      # Check behavior with placeholder input (no shape info).
      if (not context.executing_eagerly() and outer_axis >= 0 and
          inner_axis >= 0):
        x_with_placeholders = nest.map_structure(
            lambda t: array_ops.placeholder_with_default(t, None),
            x,
            expand_composites=True)
        actual_with_placeholders = x_with_placeholders.merge_dims(
            outer_axis, inner_axis)
        self.assertAllEqual(expected, actual_with_placeholders)

  @parameterized.parameters([
      {
          'rt': [[1]],
          'outer_axis': {},
          'inner_axis': 1,
          'exception': TypeError,
          'message': 'outer_axis must be an int',
      },
      {
          'rt': [[1]],
          'outer_axis': 1,
          'inner_axis': {},
          'exception': TypeError,
          'message': 'inner_axis must be an int',
      },
      {
          'rt': [[1]],
          'outer_axis': 1,
          'inner_axis': 3,
          'exception': ValueError,
          'message': 'inner_axis=3 out of bounds: expected -2<=inner_axis<2',
      },
      {
          'rt': [[1]],
          'outer_axis': 1,
          'inner_axis': -3,
          'exception': ValueError,
          'message': 'inner_axis=-3 out of bounds: expected -2<=inner_axis<2',
      },
      {
          'rt': [[1]],
          'outer_axis': 1,
          'inner_axis': 0,
          'exception': ValueError,
          'message': 'Expected outer_axis .* to be less than or equal to .*',
      },
      {
          'rt': [[1]],
          'outer_axis': -1,
          'inner_axis': -2,
          'exception': ValueError,
          'message': 'Expected outer_axis .* to be less than or equal to .*',
      },
  ])  # pyformat: disable
  def testRaggedMergeDimsError(self,
                               rt,
                               outer_axis,
                               inner_axis,
                               exception,
                               message=None,
                               ragged_rank=None):
    x = ragged_factory_ops.constant(rt, ragged_rank=ragged_rank)
    with self.assertRaisesRegex(exception, message):
      self.evaluate(x.merge_dims(outer_axis, inner_axis))


if __name__ == '__main__':
  googletest.main()
