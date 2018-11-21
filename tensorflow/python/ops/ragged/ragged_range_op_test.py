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
from tensorflow.python.ops import ragged
from tensorflow.python.platform import googletest


class RaggedRangeOpTest(test_util.TensorFlowTestCase):

  def testDocStringExamples(self):
    """Examples from ragged_range.__doc__."""
    with self.test_session():
      rt1 = ragged.range([3, 5, 2]).eval().tolist()
      self.assertEqual(rt1, [[0, 1, 2], [0, 1, 2, 3, 4], [0, 1]])

      rt2 = ragged.range([0, 5, 8], [3, 3, 12]).eval().tolist()
      self.assertEqual(rt2, [[0, 1, 2], [], [8, 9, 10, 11]])

      rt3 = ragged.range([0, 5, 8], [3, 3, 12], 2).eval().tolist()
      self.assertEqual(rt3, [[0, 2], [], [8, 10]])

  def testBasicRanges(self):
    with self.test_session():
      # Specify limits only.
      self.assertEqual(
          ragged.range([0, 3, 5]).eval().tolist(),
          [list(range(0)), list(range(3)), list(range(5))])

      # Specify starts and limits.
      self.assertEqual(
          ragged.range([0, 3, 5], [2, 3, 10]).eval().tolist(),
          [list(range(0, 2)), list(range(3, 3)), list(range(5, 10))])

      # Specify starts, limits, and deltas.
      self.assertEqual(
          ragged.range([0, 3, 5], [4, 4, 15], [2, 3, 4]).eval().tolist(),
          [list(range(0, 4, 2)), list(range(3, 4, 3)),
           list(range(5, 15, 4))])

  def testFloatRanges(self):
    with self.test_session():
      expected = [[0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6], [3.0],
                  [5.0, 7.2, 9.4, 11.6, 13.8]]
      actual = ragged.range([0.0, 3.0, 5.0], [3.9, 4.0, 15.0],
                            [0.4, 1.5, 2.2]).eval().tolist()
      self.assertEqual(expected, [[round(v, 5) for v in row] for row in actual])

  def testNegativeDeltas(self):
    with self.test_session():
      self.assertEqual(
          ragged.range([0, 3, 5], limits=0, deltas=-1).eval().tolist(),
          [list(range(0, 0, -1)), list(range(3, 0, -1)),
           list(range(5, 0, -1))])

      self.assertEqual(
          ragged.range([0, -3, 5], limits=0, deltas=[-1, 1,
                                                     -2]).eval().tolist(),
          [list(range(0, 0, -1)), list(range(-3, 0, 1)),
           list(range(5, 0, -2))])

  def testBroadcast(self):
    with self.test_session():
      # Specify starts and limits, broadcast deltas.
      self.assertEqual(
          ragged.range([0, 3, 5], [4, 4, 15], 3).eval().tolist(),
          [list(range(0, 4, 3)), list(range(3, 4, 3)),
           list(range(5, 15, 3))])

      # Broadcast all arguments.
      self.assertEqual(
          ragged.range(0, 5, 1).eval().tolist(), [list(range(0, 5, 1))])

  def testEmptyRanges(self):
    rt1 = ragged.range([0, 5, 3], [0, 3, 5])
    rt2 = ragged.range([0, 5, 5], [0, 3, 5], -1)
    with self.test_session():
      self.assertEqual(rt1.eval().tolist(), [[], [], [3, 4]])
      self.assertEqual(rt2.eval().tolist(), [[], [5, 4], []])

  def testShapeFnErrors(self):
    with self.test_session():
      self.assertRaisesRegexp(ValueError, r'Shape must be at most rank 1.*',
                              ragged.range, [[0]], 5)
      self.assertRaisesRegexp(ValueError, r'Shape must be at most rank 1.*',
                              ragged.range, 0, [[5]])
      self.assertRaisesRegexp(ValueError, r'Shape must be at most rank 1.*',
                              ragged.range, 0, 5, [[0]])
      self.assertRaisesRegexp(ValueError, r'Dimensions must be equal.*',
                              ragged.range, [0], [1, 2])

  def testKernelErrors(self):
    with self.test_session():
      self.assertRaisesRegexp(errors.InvalidArgumentError,
                              r'Requires delta != 0',
                              ragged.range(0, 0, 0).eval)

  def testShape(self):
    self.assertEqual(ragged.range(0, 0, 0).shape.as_list(), [1, None])
    self.assertEqual(ragged.range([1, 2, 3]).shape.as_list(), [3, None])
    self.assertEqual(
        ragged.range([1, 2, 3], [4, 5, 6]).shape.as_list(), [3, None])


if __name__ == '__main__':
  googletest.main()
