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
"""Tests for ragged_range op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from absl.testing import parameterized

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import ragged
from tensorflow.python.platform import googletest


def prod(values):
  val = 1
  for v in values:
    val *= v
  return val
  # return reduce(lambda x, y: x * y, values, 1)


def mean(values):
  return 1.0 * sum(values) / len(values)


def sqrt_n(values):
  return 1.0 * sum(values) / math.sqrt(len(values))


class RaggedSegmentOpsTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):

  def assertNestedListAmostEqual(self, lhs, rhs, places=7, context='value'):
    self.assertEqual(type(lhs), type(rhs))
    if isinstance(lhs, (list, tuple)):
      self.assertEqual(len(lhs), len(rhs), 'Length differs for %s' % context)
      for i in range(len(lhs)):
        self.assertNestedListAmostEqual(lhs[i], rhs[i], places,
                                        '%s[%s]' % (context, i))
    else:
      self.assertAlmostEqual(
          lhs, rhs, places,
          '%s != %s within %s places at %s' % (lhs, rhs, places, context))

  def expected_value(self, data, segment_ids, num_segments, combiner):
    """Find the expected value for a call to ragged_segment_<aggregate>.

    Args:
      data: The input RaggedTensor, expressed as a nested python list.
      segment_ids: The segment ids, as a python list of ints.
      num_segments: The number of segments, as a python int.
      combiner: The Python function used to combine values.
    Returns:
      The expected value, as a nested Python list.
    """
    self.assertEqual(len(data), len(segment_ids))

    # Build an empty (num_segments x ncols) "grouped" matrix
    ncols = max(len(row) for row in data)
    grouped = [[[] for _ in range(ncols)] for row in range(num_segments)]

    # Append values from data[row] to grouped[segment_ids[row]]
    for row in range(len(data)):
      for col in range(len(data[row])):
        grouped[segment_ids[row]][col].append(data[row][col])

    # Combine the values.
    return [[combiner(values)
             for values in grouped_row
             if values]
            for grouped_row in grouped]

  @parameterized.parameters(
      (ragged.segment_sum, sum, [0, 0, 1, 1, 2, 2]),
      (ragged.segment_sum, sum, [0, 0, 0, 1, 1, 1]),
      (ragged.segment_sum, sum, [5, 4, 3, 2, 1, 0]),
      (ragged.segment_sum, sum, [0, 0, 0, 10, 10, 10]),
      (ragged.segment_prod, prod, [0, 0, 1, 1, 2, 2]),
      (ragged.segment_prod, prod, [0, 0, 0, 1, 1, 1]),
      (ragged.segment_prod, prod, [5, 4, 3, 2, 1, 0]),
      (ragged.segment_prod, prod, [0, 0, 0, 10, 10, 10]),
      (ragged.segment_min, min, [0, 0, 1, 1, 2, 2]),
      (ragged.segment_min, min, [0, 0, 0, 1, 1, 1]),
      (ragged.segment_min, min, [5, 4, 3, 2, 1, 0]),
      (ragged.segment_min, min, [0, 0, 0, 10, 10, 10]),
      (ragged.segment_max, max, [0, 0, 1, 1, 2, 2]),
      (ragged.segment_max, max, [0, 0, 0, 1, 1, 1]),
      (ragged.segment_max, max, [5, 4, 3, 2, 1, 0]),
      (ragged.segment_max, max, [0, 0, 0, 10, 10, 10]),
      (ragged.segment_mean, mean, [0, 0, 1, 1, 2, 2]),
      (ragged.segment_mean, mean, [0, 0, 0, 1, 1, 1]),
      (ragged.segment_mean, mean, [5, 4, 3, 2, 1, 0]),
      (ragged.segment_mean, mean, [0, 0, 0, 10, 10, 10]),
  )
  def testRaggedSegment_Int(self, segment_op, combiner, segment_ids):
    rt_as_list = [[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]]
    rt = ragged.constant(rt_as_list)
    num_segments = max(segment_ids) + 1
    expected = self.expected_value(rt_as_list, segment_ids, num_segments,
                                   combiner)

    segmented = segment_op(rt, segment_ids, num_segments)
    with self.test_session():
      self.assertListEqual(self.evaluate(segmented).tolist(), expected)

  @parameterized.parameters(
      (ragged.segment_sum, sum, [0, 0, 1, 1, 2, 2]),
      (ragged.segment_sum, sum, [0, 0, 0, 1, 1, 1]),
      (ragged.segment_sum, sum, [5, 4, 3, 2, 1, 0]),
      (ragged.segment_sum, sum, [0, 0, 0, 10, 10, 10]),
      (ragged.segment_prod, prod, [0, 0, 1, 1, 2, 2]),
      (ragged.segment_prod, prod, [0, 0, 0, 1, 1, 1]),
      (ragged.segment_prod, prod, [5, 4, 3, 2, 1, 0]),
      (ragged.segment_prod, prod, [0, 0, 0, 10, 10, 10]),
      (ragged.segment_min, min, [0, 0, 1, 1, 2, 2]),
      (ragged.segment_min, min, [0, 0, 0, 1, 1, 1]),
      (ragged.segment_min, min, [5, 4, 3, 2, 1, 0]),
      (ragged.segment_min, min, [0, 0, 0, 10, 10, 10]),
      (ragged.segment_max, max, [0, 0, 1, 1, 2, 2]),
      (ragged.segment_max, max, [0, 0, 0, 1, 1, 1]),
      (ragged.segment_max, max, [5, 4, 3, 2, 1, 0]),
      (ragged.segment_max, max, [0, 0, 0, 10, 10, 10]),
      (ragged.segment_mean, mean, [0, 0, 1, 1, 2, 2]),
      (ragged.segment_mean, mean, [0, 0, 0, 1, 1, 1]),
      (ragged.segment_mean, mean, [5, 4, 3, 2, 1, 0]),
      (ragged.segment_mean, mean, [0, 0, 0, 10, 10, 10]),
      (ragged.segment_sqrt_n, sqrt_n, [0, 0, 1, 1, 2, 2]),
      (ragged.segment_sqrt_n, sqrt_n, [0, 0, 0, 1, 1, 1]),
      (ragged.segment_sqrt_n, sqrt_n, [5, 4, 3, 2, 1, 0]),
      (ragged.segment_sqrt_n, sqrt_n, [0, 0, 0, 10, 10, 10]),
  )
  def testRaggedSegment_Float(self, segment_op, combiner, segment_ids):
    rt_as_list = [[0., 1., 2., 3.], [4.], [], [5., 6.], [7.], [8., 9.]]
    rt = ragged.constant(rt_as_list)
    num_segments = max(segment_ids) + 1
    expected = self.expected_value(rt_as_list, segment_ids, num_segments,
                                   combiner)

    segmented = segment_op(rt, segment_ids, num_segments)
    with self.test_session():
      self.assertNestedListAmostEqual(
          self.evaluate(segmented).tolist(), expected, places=5)

  def testRaggedRankTwo(self):
    rt = ragged.constant([
        [[111, 112, 113, 114], [121],],  # row 0
        [],                              # row 1
        [[], [321, 322], [331]],         # row 2
        [[411, 412]]                     # row 3
    ])  # pyformat: disable
    segment_ids1 = [0, 2, 2, 2]
    segmented1 = ragged.segment_sum(rt, segment_ids1, 3)
    expected1 = [[[111, 112, 113, 114], [121]],     # row 0
                 [],                                # row 1
                 [[411, 412], [321, 322], [331]]    # row 2
                ]  # pyformat: disable
    with self.test_session():
      self.assertEqual(self.evaluate(segmented1).tolist(), expected1)

    segment_ids2 = [1, 2, 1, 1]
    segmented2 = ragged.segment_sum(rt, segment_ids2, 3)
    expected2 = [[],
                 [[111+411, 112+412, 113, 114], [121+321, 322], [331]],
                 []]  # pyformat: disable
    with self.test_session():
      self.assertEqual(self.evaluate(segmented2).tolist(), expected2)

  def testRaggedSegmentIds(self):
    rt = ragged.constant([
        [[111, 112, 113, 114], [121],],  # row 0
        [],                              # row 1
        [[], [321, 322], [331]],         # row 2
        [[411, 412]]                     # row 3
    ])  # pyformat: disable
    segment_ids = ragged.constant([[1, 2], [], [1, 1, 2], [2]])
    segmented = ragged.segment_sum(rt, segment_ids, 3)
    expected = [[],
                [111+321, 112+322, 113, 114],
                [121+331+411, 412]]  # pyformat: disable
    with self.test_session():
      self.assertEqual(self.evaluate(segmented).tolist(), expected)

  def testShapeMismatchError1(self):
    dt = constant_op.constant([1, 2, 3, 4, 5, 6])
    segment_ids = ragged.constant([[1, 2], []])
    self.assertRaisesRegexp(
        ValueError, 'segment_ids.shape must be a prefix of data.shape, '
        'but segment_ids is ragged and data is not.', ragged.segment_sum, dt,
        segment_ids, 3)

  def testShapeMismatchError2(self):
    rt = ragged.constant([
        [[111, 112, 113, 114], [121]],  # row 0
        [],                             # row 1
        [[], [321, 322], [331]],        # row 2
        [[411, 412]]                    # row 3
    ])  # pyformat: disable
    segment_ids = ragged.constant([[1, 2], [1], [1, 1, 2], [2]])

    # Error is raised at graph-building time if we can detect it then.
    self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        'segment_ids.shape must be a prefix of data.shape.*',
        ragged.segment_sum, rt, segment_ids, 3)

    # Otherwise, error is raised when we run the graph.
    segment_ids2 = ragged.from_row_splits(
        array_ops.placeholder_with_default(segment_ids.values, None),
        array_ops.placeholder_with_default(segment_ids.row_splits, None))
    segmented2 = ragged.segment_sum(rt, segment_ids2, 3)
    with self.test_session():
      self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          'segment_ids.shape must be a prefix of data.shape.*', segmented2.eval)


if __name__ == '__main__':
  googletest.main()
