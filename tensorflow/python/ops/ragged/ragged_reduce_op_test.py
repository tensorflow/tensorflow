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
"""Tests for ragged.reduce_<AGGREGATE> ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import ragged
from tensorflow.python.platform import googletest

_MAX_INT32 = dtypes.int32.max
_MIN_INT32 = dtypes.int32.min
_NAN = np.nan


def mean(*values):
  return 1.0 * sum(values) / len(values)


class RaggedReduceOpsTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.parameters(
      #=========================================================================
      # Docstring examples.  RaggedTensor for testing is:
      #   [[3, 1, 4],
      #    [1, 5,  ],
      #    [9,     ],
      #    [2, 6   ]]
      #=========================================================================
      dict(
          ragged_reduce_op=ragged.reduce_sum,
          rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]],
          axis=0,
          expected=[15, 12, 4]  # = [3+1+9+2, 1+5+6, 4]
      ),
      dict(
          ragged_reduce_op=ragged.reduce_sum,
          rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]],
          axis=-2,
          expected=[15, 12, 4]  # = [3+1+9+2, 1+5+6, 4]
      ),
      dict(
          ragged_reduce_op=ragged.reduce_sum,
          rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]],
          axis=1,
          expected=[8, 6, 9, 8]  # = [3+1+4, 1+5, 9, 2+6]
      ),
      dict(
          ragged_reduce_op=ragged.reduce_sum,
          rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]],
          axis=-1,
          expected=[8, 6, 9, 8]  # = [3+1+4, 1+5, 9, 2+6]
      ),
      dict(
          ragged_reduce_op=ragged.reduce_prod,
          rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]],
          axis=0,
          expected=[54, 30, 4]  # = [3*1*9*2, 1*5*6, 4]
      ),
      dict(
          ragged_reduce_op=ragged.reduce_prod,
          rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]],
          axis=1,
          expected=[12, 5, 9, 12]  # = [3*1*4, 1*5, 9, 2*6]
      ),
      dict(
          ragged_reduce_op=ragged.reduce_min,
          rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]],
          axis=0,
          expected=[1, 1, 4]  # = [min(3, 1, 9, 2), min(1, 5, 6), 4]
      ),
      dict(
          ragged_reduce_op=ragged.reduce_min,
          rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]],
          axis=1,
          expected=[1, 1, 9, 2]  # = [min(3, 1, 4), min(1, 5), 9, min(2, 6)]
      ),
      dict(
          ragged_reduce_op=ragged.reduce_max,
          rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]],
          axis=0,
          expected=[9, 6, 4]  # = [max(3, 1, 9, 2), max(1, 5, 6), 4]
      ),
      dict(
          ragged_reduce_op=ragged.reduce_max,
          rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]],
          axis=1,
          expected=[4, 5, 9, 6]  # = [max(3, 1, 4), max(1, 5), 9, max(2, 6)]
      ),
      dict(
          ragged_reduce_op=ragged.reduce_mean,
          rt_input=[[3, 1, 4], [1, 5], [9], [2, 6]],
          axis=0,
          expected=[3.75, 4, 4]  # = [mean(3, 1, 9, 2), mean(1, 5, 6), 4]
      ),
      dict(
          ragged_reduce_op=ragged.reduce_any,
          rt_input=[[True, True], [True, True, False, True], [False, True]],
          axis=0,
          expected=[True, True, False, True]),
      dict(
          ragged_reduce_op=ragged.reduce_any,
          rt_input=[[True, True], [True, True, False, True], [False, True]],
          axis=1,
          expected=[True, True, True]),
      dict(
          ragged_reduce_op=ragged.reduce_all,
          rt_input=[[True, True], [True, True, False, True], [False, True]],
          axis=0,
          expected=[False, True, False, True]),
      dict(
          ragged_reduce_op=ragged.reduce_all,
          rt_input=[[True, True], [True, True, False, True], [False, True]],
          axis=1,
          expected=[True, False, False]),

      #=========================================================================
      # Examples with the following RaggedTensor (ragged_rank=1):
      #   [[0, 1, 2, 3],
      #    [4         ],
      #    [          ],
      #    [5, 6      ],
      #    [7         ],
      #    [8, 9      ]]
      #=========================================================================

      # axis=None
      dict(
          ragged_reduce_op=ragged.reduce_sum,
          rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]],
          axis=None,
          expected=0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9),
      dict(
          ragged_reduce_op=ragged.reduce_prod,
          rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]],
          axis=None,
          expected=0 * 1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9),
      dict(
          ragged_reduce_op=ragged.reduce_min,
          rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]],
          axis=None,
          expected=min(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)),
      dict(
          ragged_reduce_op=ragged.reduce_max,
          rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]],
          axis=None,
          expected=max(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)),
      dict(
          ragged_reduce_op=ragged.reduce_mean,
          rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]],
          axis=None,
          expected=mean(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)),
      # axis=0
      dict(
          ragged_reduce_op=ragged.reduce_sum,
          rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]],
          axis=0,
          expected=[0 + 4 + 5 + 7 + 8, 1 + 6 + 9, 2, 3]),
      dict(
          ragged_reduce_op=ragged.reduce_prod,
          rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]],
          axis=0,
          expected=[0 * 4 * 5 * 7 * 8, 1 * 6 * 9, 2, 3]),
      dict(
          ragged_reduce_op=ragged.reduce_min,
          rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]],
          axis=0,
          expected=[min(0, 4, 5, 7, 8), min(1, 6, 9), 2, 3]),
      dict(
          ragged_reduce_op=ragged.reduce_max,
          rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]],
          axis=0,
          expected=[max(0, 4, 5, 7, 8), max(1, 6, 9), 2, 3]),
      dict(
          ragged_reduce_op=ragged.reduce_mean,
          rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]],
          axis=0,
          expected=[mean(0, 4, 5, 7, 8),
                    mean(1, 6, 9), 2, 3]),
      # axis=1
      # Note: we don't test mean here because it gives a NaN, and this will
      # cause assertEqual to fail (since NaN != NaN).  See testMeanNan().
      dict(
          ragged_reduce_op=ragged.reduce_sum,
          rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]],
          axis=1,
          expected=[0 + 1 + 2 + 3, 4, 0, 5 + 6, 7, 8 + 9]),
      dict(
          ragged_reduce_op=ragged.reduce_prod,
          rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]],
          axis=1,
          expected=[0 * 1 * 2 * 3, 4, 1, 5 * 6, 7, 8 * 9]),
      dict(
          ragged_reduce_op=ragged.reduce_min,
          rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]],
          axis=1,
          expected=[min(0, 1, 2, 3), 4, _MAX_INT32,
                    min(5, 6), 7,
                    min(8, 9)]),
      dict(
          ragged_reduce_op=ragged.reduce_max,
          rt_input=[[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]],
          axis=1,
          expected=[max(0, 1, 2, 3), 4, _MIN_INT32,
                    max(5, 6), 7,
                    max(8, 9)]),

      #=========================================================================
      # Examples with ragged_rank=2:
      # [[[1, 2], [ ], [3, 4, 5]],
      #  [[6, 7], [ ], [8      ]],
      #  [                      ],
      #  [[9   ]                ]]
      #=========================================================================
      dict(
          ragged_reduce_op=ragged.reduce_sum,
          rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]],
          axis=[],
          expected=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]]),
      dict(
          ragged_reduce_op=ragged.reduce_sum,
          rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]],
          axis=None,
          expected=sum([1, 2, 3, 4, 5, 6, 7, 8, 9])),
      dict(
          ragged_reduce_op=ragged.reduce_sum,
          rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]],
          axis=0,
          expected=[[1 + 6 + 9, 2 + 7], [], [3 + 8, 4, 5]]),
      dict(
          ragged_reduce_op=ragged.reduce_sum,
          rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]],
          axis=1,
          expected=[[1 + 3, 2 + 4, 5], [6 + 8, 7], [], [9]]),
      dict(
          ragged_reduce_op=ragged.reduce_sum,
          rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]],
          axis=2,
          expected=[[1 + 2, 0, 3 + 4 + 5], [6 + 7, 0, 8], [], [9]]),
      dict(
          ragged_reduce_op=ragged.reduce_sum,
          rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]],
          axis=[0, 1],
          expected=[1 + 3 + 6 + 8 + 9, 2 + 4 + 7, 5]),
      dict(
          ragged_reduce_op=ragged.reduce_sum,
          rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]],
          axis=[0, 2],
          expected=[1 + 6 + 9 + 2 + 7, 0, 3 + 8 + 4 + 5]),
      dict(
          ragged_reduce_op=ragged.reduce_sum,
          rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]],
          axis=[1, 2],
          expected=[1 + 2 + 3 + 4 + 5, 6 + 7 + 8, 0, 9]),
      dict(
          ragged_reduce_op=ragged.reduce_sum,
          rt_input=[[[1, 2], [], [3, 4, 5]], [[6, 7], [], [8]], [], [[9]]],
          axis=[0, 1, 2],
          expected=sum([1, 2, 3, 4, 5, 6, 7, 8, 9])),

      #=========================================================================
      # Examples for ragged_reduce_mean ragged_rank=2:
      # [[[1, 2], [3, 4, 5]],
      #  [[6, 7], [8      ]],
      #  [[9   ]          ]]
      #=========================================================================
      dict(
          ragged_reduce_op=ragged.reduce_mean,
          rt_input=[[[1, 2], [3, 4, 5]], [[6, 7], [8]], [[9]]],
          axis=0,
          expected=[[mean(1, 6, 9), mean(2, 7)], [mean(3, 8), 4, 5]]),
      dict(
          ragged_reduce_op=ragged.reduce_mean,
          rt_input=[[[1, 2], [3, 4, 5]], [[6, 7], [8]], [[9]]],
          axis=1,
          expected=[[mean(1, 3), mean(2, 4), 5], [mean(6, 8), 7], [9]]),
      dict(
          ragged_reduce_op=ragged.reduce_mean,
          rt_input=[[[1, 2], [3, 4, 5]], [[6, 7], [8]], [[9]]],
          axis=2,
          expected=[[mean(1, 2), mean(3, 4, 5)], [mean(6, 7), 8], [9]]),
  )
  @test_util.run_deprecated_v1
  def testReduce(self, ragged_reduce_op, rt_input, axis, expected):
    rt_input = ragged.constant(rt_input)
    reduced = ragged_reduce_op(rt_input, axis)
    with self.test_session():
      self.assertEqual(reduced.eval().tolist(), expected)

  def assertEqualWithNan(self, actual, expected):
    """Like assertEqual, but NaN==NaN."""
    self.assertTrue(
        ((actual == expected) | (np.isnan(actual) & np.isnan(expected))).all())

  @test_util.run_deprecated_v1
  def testMeanNan(self):
    rt_as_list = [[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]]
    expected = (
        np.array([0 + 1 + 2 + 3, 4, 0, 5 + 6, 7, 8 + 9]) / np.array(
            [4, 1, 0, 2, 1, 2]))
    rt_input = ragged.constant(rt_as_list)
    reduced = ragged.reduce_mean(rt_input, axis=1)
    with self.test_session():
      self.assertEqualWithNan(reduced.eval(), expected)

  @test_util.run_deprecated_v1
  def testMeanWithTensorInputs(self):
    tensor = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
    expected = [2.0, 20.0]
    reduced = ragged.reduce_mean(tensor, axis=1)
    with self.test_session():
      self.assertAllEqual(reduced.eval(), expected)

  @test_util.run_deprecated_v1
  def testErrors(self):
    rt_input = ragged.constant([[1, 2, 3], [4, 5]])
    axis = array_ops.placeholder_with_default(constant_op.constant([0]), None)
    self.assertRaisesRegexp(ValueError,
                            r'axis must be known at graph construction time.',
                            ragged.reduce_sum, rt_input, axis)
    self.assertRaisesRegexp(TypeError,
                            r'axis must be an int; got str.*',
                            ragged.reduce_sum, rt_input, ['x'])


if __name__ == '__main__':
  googletest.main()
