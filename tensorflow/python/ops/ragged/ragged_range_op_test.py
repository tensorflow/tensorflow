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

from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_test_util
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedRangeOpTest(ragged_test_util.RaggedTensorTestCase):

  def testDocStringExamples(self):
    """Examples from ragged_range.__doc__."""
    rt1 = ragged_math_ops.range([3, 5, 2])
    self.assertRaggedEqual(rt1, [[0, 1, 2], [0, 1, 2, 3, 4], [0, 1]])

    rt2 = ragged_math_ops.range([0, 5, 8], [3, 3, 12])
    self.assertRaggedEqual(rt2, [[0, 1, 2], [], [8, 9, 10, 11]])

    rt3 = ragged_math_ops.range([0, 5, 8], [3, 3, 12], 2)
    self.assertRaggedEqual(rt3, [[0, 2], [], [8, 10]])

  def testBasicRanges(self):
    # Specify limits only.
    self.assertRaggedEqual(
        ragged_math_ops.range([0, 3, 5]),
        [list(range(0)), list(range(3)),
         list(range(5))])

    # Specify starts and limits.
    self.assertRaggedEqual(
        ragged_math_ops.range([0, 3, 5], [2, 3, 10]),
        [list(range(0, 2)),
         list(range(3, 3)),
         list(range(5, 10))])

    # Specify starts, limits, and deltas.
    self.assertRaggedEqual(
        ragged_math_ops.range([0, 3, 5], [4, 4, 15], [2, 3, 4]),
        [list(range(0, 4, 2)),
         list(range(3, 4, 3)),
         list(range(5, 15, 4))])

  def testFloatRanges(self):
    expected = [[0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6], [3.0],
                [5.0, 7.2, 9.4, 11.6, 13.8]]
    actual = ragged_math_ops.range([0.0, 3.0, 5.0], [3.9, 4.0, 15.0],
                                   [0.4, 1.5, 2.2])
    self.assertEqual(
        expected,
        [[round(v, 5) for v in row] for row in self.eval_to_list(actual)])

  def testNegativeDeltas(self):
    self.assertRaggedEqual(
        ragged_math_ops.range([0, 3, 5], limits=0, deltas=-1),
        [list(range(0, 0, -1)),
         list(range(3, 0, -1)),
         list(range(5, 0, -1))])

    self.assertRaggedEqual(
        ragged_math_ops.range([0, -3, 5], limits=0, deltas=[-1, 1, -2]),
        [list(range(0, 0, -1)),
         list(range(-3, 0, 1)),
         list(range(5, 0, -2))])

  def testBroadcast(self):
    # Specify starts and limits, broadcast deltas.
    self.assertRaggedEqual(
        ragged_math_ops.range([0, 3, 5], [4, 4, 15], 3),
        [list(range(0, 4, 3)),
         list(range(3, 4, 3)),
         list(range(5, 15, 3))])

    # Broadcast all arguments.
    self.assertRaggedEqual(
        ragged_math_ops.range(0, 5, 1), [list(range(0, 5, 1))])

  def testEmptyRanges(self):
    rt1 = ragged_math_ops.range([0, 5, 3], [0, 3, 5])
    rt2 = ragged_math_ops.range([0, 5, 5], [0, 3, 5], -1)
    self.assertRaggedEqual(rt1, [[], [], [3, 4]])
    self.assertRaggedEqual(rt2, [[], [5, 4], []])

  def testShapeFnErrors(self):
    self.assertRaises((ValueError, errors.InvalidArgumentError),
                      ragged_math_ops.range, [[0]], 5)
    self.assertRaises((ValueError, errors.InvalidArgumentError),
                      ragged_math_ops.range, 0, [[5]])
    self.assertRaises((ValueError, errors.InvalidArgumentError),
                      ragged_math_ops.range, 0, 5, [[0]])
    self.assertRaises((ValueError, errors.InvalidArgumentError),
                      ragged_math_ops.range, [0], [1, 2])

  def testKernelErrors(self):
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 r'Requires delta != 0'):
      self.evaluate(ragged_math_ops.range(0, 0, 0))

  def testShape(self):
    self.assertRaggedEqual(
        ragged_math_ops.range(0, 0, 1).shape.as_list(), [1, None])
    self.assertRaggedEqual(
        ragged_math_ops.range([1, 2, 3]).shape.as_list(), [3, None])
    self.assertRaggedEqual(
        ragged_math_ops.range([1, 2, 3], [4, 5, 6]).shape.as_list(), [3, None])


if __name__ == '__main__':
  googletest.main()
