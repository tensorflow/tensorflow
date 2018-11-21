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
"""Tests for ragged.to_tensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import ragged
from tensorflow.python.platform import googletest


class RaggedTensorToTensorOpTest(test_util.TensorFlowTestCase,
                                 parameterized.TestCase):

  def testDocStringExamples(self):
    """Example from ragged_to_tensor.__doc__."""
    rt = ragged.constant([[9, 8, 7], [], [6, 5], [4]])
    dt = ragged.to_tensor(rt)
    with self.test_session():
      self.assertEqual(str(dt.eval()),
                       '[[9 8 7]\n'
                       ' [0 0 0]\n'
                       ' [6 5 0]\n'
                       ' [4 0 0]]')  # pyformat: disable

  @parameterized.parameters(
      {
          'rt_input': [],
          'ragged_rank': 1,
          'expected': [],
          'expected_shape': [0, 0],
      },
      {
          'rt_input': [[1, 2, 3], [], [4], [5, 6]],
          'expected': [[1, 2, 3], [0, 0, 0], [4, 0, 0], [5, 6, 0]]
      },
      {
          'rt_input': [[1, 2, 3], [], [4], [5, 6]],
          'default': 9,
          'expected': [[1, 2, 3], [9, 9, 9], [4, 9, 9], [5, 6, 9]]
      },
      {
          'rt_input': [[[1], [2], [3]], [], [[4]], [[5], [6]]],
          'ragged_rank':
              1,
          'default': [9],
          'expected': [[[1], [2], [3]], [[9], [9], [9]], [[4], [9], [9]],
                       [[5], [6], [9]]]
      },
      {
          'rt_input': [[[1, 2], [], [3, 4]], [], [[5]], [[6, 7], [8]]],
          'expected': [
              [[1, 2], [0, 0], [3, 4]],  #
              [[0, 0], [0, 0], [0, 0]],  #
              [[5, 0], [0, 0], [0, 0]],  #
              [[6, 7], [8, 0], [0, 0]],  #
          ]
      },
      {
          'rt_input': [[[1, 2], [], [3, 4]], [], [[5]], [[6, 7], [8]]],
          'default':
              9,
          'expected': [
              [[1, 2], [9, 9], [3, 4]],  #
              [[9, 9], [9, 9], [9, 9]],  #
              [[5, 9], [9, 9], [9, 9]],  #
              [[6, 7], [8, 9], [9, 9]],  #
          ]
      },
      {
          'rt_input': [[[1], [2], [3]]],
          'ragged_rank': 1,
          'default': 0,
          'expected': [[[1], [2], [3]]],
      },
      {
          'rt_input': [[[[1], [2]], [], [[3]]]],
          'default': 9,
          'expected': [[[[1], [2]], [[9], [9]], [[3], [9]]]],
      },
  )
  def testRaggedTensorToTensor(self,
                               rt_input,
                               expected,
                               ragged_rank=None,
                               default=None,
                               expected_shape=None):
    rt = ragged.constant(rt_input, ragged_rank=ragged_rank)
    dt = ragged.to_tensor(rt, default)
    self.assertEqual(type(dt), ops.Tensor)
    self.assertEqual(rt.dtype, dt.dtype)
    self.assertTrue(dt.shape.is_compatible_with(rt.shape))
    with self.test_session():
      self.assertEqual(dt.eval().tolist(), expected)
      if expected_shape is not None:
        dt_shape = array_ops.shape(dt)
        self.assertEqual(dt_shape.eval().tolist(), expected_shape)

  @parameterized.parameters(
      {
          'rt_input': [[1, 2, 3]],
          'default': [0],
          'error': (ValueError, r'Shape \(1,\) must have rank at most 0'),
      },
      {
          'rt_input': [[[1, 2], [3, 4]], [[5, 6]]],
          'ragged_rank': 1,
          'default': [7, 8, 9],
          'error': (ValueError, r'Shapes \(3,\) and \(2,\) are incompatible'),
      },
      {
          'rt_input': [[1, 2, 3]],
          'default': 'a',
          'error': (TypeError, "Expected int32, got 'a' of type 'str' instead"),
      },
  )
  def testError(self, rt_input, default, error, ragged_rank=None):
    rt = ragged.constant(rt_input, ragged_rank=ragged_rank)
    with self.assertRaisesRegexp(error[0], error[1]):
      ragged.to_tensor(rt, default)


if __name__ == '__main__':
  googletest.main()
