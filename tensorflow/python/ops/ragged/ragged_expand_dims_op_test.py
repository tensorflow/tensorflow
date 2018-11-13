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
"""Tests for ragged.expand_dims."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.framework import test_util
from tensorflow.python.ops import ragged
from tensorflow.python.platform import googletest


class RaggedExpandDimsOpTest(test_util.TensorFlowTestCase,
                             parameterized.TestCase):

  # An example 4-d ragged tensor with shape [3, (D2), (D3), 2], and the
  # expected result calling for expand_dims on each axis.  c.f. the table of
  # expected result shapes in the ragged.expand_dims docstring.
  EXAMPLE4D = [[[[1, 1], [2, 2]], [[3, 3]]],
               [],
               [[], [[4, 4], [5, 5], [6, 6]]]]  # pyformat: disable
  EXAMPLE4D_EXPAND_AXIS = {
      0: [EXAMPLE4D],
      1: [[d0] for d0 in EXAMPLE4D],
      2: [[[d1] for d1 in d0] for d0 in EXAMPLE4D],
      3: [[[[d2] for d2 in d1] for d1 in d0] for d0 in EXAMPLE4D],
      4: [[[[[d3] for d3 in d2] for d2 in d1] for d1 in d0] for d0 in EXAMPLE4D]
  }

  @parameterized.parameters([
      #=========================================================================
      # Docstring examples: 2D Ragged Inputs
      dict(rt_input=[[1, 2], [3]],
           axis=0,
           expected=[[[1, 2], [3]]],
           expected_shape=[1, None, None]),
      dict(rt_input=[[1, 2], [3]],
           axis=1,
           expected=[[[1, 2]], [[3]]],
           expected_shape=[2, None, None]),
      dict(rt_input=[[1, 2], [3]],
           axis=2,
           expected=[[[1], [2]], [[3]]],
           expected_shape=[2, None, 1]),

      #=========================================================================
      # 2D Tensor Inputs
      dict(rt_input=[[1, 2], [3, 4], [5, 6]],
           ragged_rank=0,
           axis=0,
           expected=[[[1, 2], [3, 4], [5, 6]]],
           expected_shape=[1, 3, 2]),
      dict(rt_input=[[1, 2], [3, 4], [5, 6]],
           ragged_rank=0,
           axis=1,
           expected=[[[1, 2]], [[3, 4]], [[5, 6]]],
           expected_shape=[3, 1, 2]),
      dict(rt_input=[[1, 2], [3, 4], [5, 6]],
           ragged_rank=0,
           axis=2,
           expected=[[[1], [2]], [[3], [4]], [[5], [6]]],
           expected_shape=[3, 2, 1]),

      #=========================================================================
      # 4D Ragged Inputs: [3, (D2), (D3), 2]
      # c.f. the table of expected result shapes in the expand_dims docstring.
      dict(rt_input=EXAMPLE4D,
           ragged_rank=2,
           axis=0,
           expected=EXAMPLE4D_EXPAND_AXIS[0],
           expected_shape=[1, None, None, None, 2]),
      dict(rt_input=EXAMPLE4D,
           ragged_rank=2,
           axis=1,
           expected=EXAMPLE4D_EXPAND_AXIS[1],
           expected_shape=[3, None, None, None, 2]),
      dict(rt_input=EXAMPLE4D,
           ragged_rank=2,
           axis=2,
           expected=EXAMPLE4D_EXPAND_AXIS[2],
           expected_shape=[3, None, None, None, 2]),
      dict(rt_input=EXAMPLE4D,
           ragged_rank=2,
           axis=3,
           expected=EXAMPLE4D_EXPAND_AXIS[3],
           expected_shape=[3, None, None, 1, 2]),
      dict(rt_input=EXAMPLE4D,
           ragged_rank=2,
           axis=4,
           expected=EXAMPLE4D_EXPAND_AXIS[4],
           expected_shape=[3, None, None, 2, 1]),
  ])  # pyformat: disable
  def testRaggedExpandDims(self,
                           rt_input,
                           axis,
                           expected,
                           ragged_rank=None,
                           expected_shape=None):
    rt = ragged.constant(rt_input, ragged_rank=ragged_rank)
    expanded = ragged.expand_dims(rt, axis=axis)
    self.assertEqual(expanded.shape.ndims, rt.shape.ndims + 1)
    if expected_shape is not None:
      self.assertEqual(expanded.shape.as_list(), expected_shape)

    with self.test_session():
      self.assertEqual(expanded.eval().tolist(), expected)


if __name__ == '__main__':
  googletest.main()
