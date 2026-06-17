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

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedRangeOpTest(test_util.TensorFlowTestCase):

  def testDocStringExamples(self):
    """Examples from ragged_range.__doc__."""
    rt1 = ragged_math_ops.range([3, 5, 2])
    self.assertAllEqual(rt1, [[0, 1, 2], [0, 1, 2, 3, 4], [0, 1]])

    rt2 = ragged_math_ops.range([0, 5, 8], [3, 3, 12])
    self.assertAllEqual(rt2, [[0, 1, 2], [], [8, 9, 10, 11]])

    rt3 = ragged_math_ops.range([0, 5, 8], [3, 3, 12], 2)
    self.assertAllEqual(rt3, [[0, 2], [], [8, 10]])

  def testBasicRanges(self):
    # Specify limits only.
    self.assertAllEqual(
        ragged_math_ops.range([0, 3, 5]),
        [list(range(0)), list(range(3)),
         list(range(5))])

    # Specify starts and limits.
    self.assertAllEqual(
        ragged_math_ops.range([0, 3, 5], [2, 3, 10]),
        [list(range(0, 2)),
         list(range(3, 3)),
         list(range(5, 10))])

    # Specify starts, limits, and deltas.
    self.assertAllEqual(
        ragged_math_ops.range([0, 3, 5], [4, 4, 15], [2, 3, 4]),
        [list(range(0, 4, 2)),
         list(range(3, 4, 3)),
         list(range(5, 15, 4))])

  def testFloatRanges(self):
    expected = [[0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6], [3.0],
                [5.0, 7.2, 9.4, 11.6, 13.8]]
    actual = ragged_math_ops.range([0.0, 3.0, 5.0], [3.9, 4.0, 15.0],
                                   [0.4, 1.5, 2.2])
    self.assertAllClose(actual, expected)

  def testNegativeDeltas(self):
    self.assertAllEqual(
        ragged_math_ops.range([0, 3, 5], limits=0, deltas=-1),
        [list(range(0, 0, -1)),
         list(range(3, 0, -1)),
         list(range(5, 0, -1))])

    self.assertAllEqual(
        ragged_math_ops.range([0, -3, 5], limits=0, deltas=[-1, 1, -2]),
        [list(range(0, 0, -1)),
         list(range(-3, 0, 1)),
         list(range(5, 0, -2))])

  def testBroadcast(self):
    # Specify starts and limits, broadcast deltas.
    self.assertAllEqual(
        ragged_math_ops.range([0, 3, 5], [4, 4, 15], 3),
        [list(range(0, 4, 3)),
         list(range(3, 4, 3)),
         list(range(5, 15, 3))])

    # Broadcast all arguments.
    self.assertAllEqual(ragged_math_ops.range(0, 5, 1), [list(range(0, 5, 1))])

  def testEmptyRanges(self):
    rt1 = ragged_math_ops.range([0, 5, 3], [0, 3, 5])
    rt2 = ragged_math_ops.range([0, 5, 5], [0, 3, 5], -1)
    self.assertAllEqual(rt1, [[], [], [3, 4]])
    self.assertAllEqual(rt2, [[], [5, 4], []])

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
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                r'Requires delta != 0'):
      self.evaluate(ragged_math_ops.range(0, 0, 0))

    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                r'Requires \(\(limit - start\) / delta\) <='):
      self.evaluate(ragged_math_ops.range(0.1, 1e10, 1e-10))

    with self.assertRaisesRegex(errors.InvalidArgumentError, 'overflowed'):
      self.evaluate(
          gen_ragged_math_ops.ragged_range(
              starts=[0, 0],
              limits=[2**31 - 1, 1],
              deltas=[1, 1],
              Tsplits=dtypes.int32))

  def testShape(self):
    self.assertAllEqual(
        ragged_math_ops.range(0, 0, 1).shape.as_list(), [1, None])
    self.assertAllEqual(
        ragged_math_ops.range([1, 2, 3]).shape.as_list(), [3, None])
    self.assertAllEqual(
        ragged_math_ops.range([1, 2, 3], [4, 5, 6]).shape.as_list(), [3, None])

  def testInt32Overflow(self):
    start = 1136033460
    end = -2110457150
    step = -1849827689
    expected = [np.arange(start, end, step)]
    actual = ragged_math_ops.range(start, end, step)
    self.assertAllEqual(expected, self.evaluate(actual))


if __name__ == '__main__':
  googletest.main()
