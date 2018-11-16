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
"""Tests for ragged.boolean_mask."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import ragged
from tensorflow.python.platform import googletest


class RaggedBooleanMaskOpTest(test_util.TensorFlowTestCase,
                              parameterized.TestCase):
  # Define short constants for true & false, so the data & mask can be lined
  # up in the examples below.  This makes it easier to read the examples, to
  # see which values should be kept vs. masked.
  T = True
  F = False

  @parameterized.parameters([
      #=========================================================================
      # Docstring examples
      #=========================================================================
      dict(
          descr='Docstring example 1',
          data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
          mask=[[T, F, T], [F, F, F], [T, F, F]],
          keepdims=False,
          expected=[1, 3, 7]),
      dict(
          descr='Docstring example 2',
          data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
          mask=[[T, F, T], [F, F, F], [T, F, F]],
          keepdims=True,
          expected=ragged.constant_value([[1, 3], [], [7]])),
      dict(
          descr='Docstring example 3',
          data=ragged.constant_value([[1, 2, 3], [4], [5, 6]]),
          mask=ragged.constant_value([[F, F, T], [F], [T, T]]),
          keepdims=False,
          expected=[3, 5, 6]),
      dict(
          descr='Docstring example 4',
          data=ragged.constant_value([[1, 2, 3], [4], [5, 6]]),
          mask=ragged.constant_value([[F, F, T], [F], [T, T]]),
          keepdims=True,
          expected=ragged.constant_value([[3], [], [5, 6]])),
      dict(
          descr='Docstring example 5',
          data=ragged.constant_value([[1, 2, 3], [4], [5, 6]]),
          mask=[True, False, True],
          keepdims=False,
          expected=ragged.constant_value([[1, 2, 3], [5, 6]])),
      #=========================================================================
      # Uniform data and uniform mask.
      #=========================================================================
      dict(
          descr='data.shape=[7]; mask.shape=[7]; keepdims=True',
          data=[1, 2, 3, 4, 5, 6, 7],
          mask=[T, F, T, T, F, F, F],
          keepdims=True,
          expected=[1, 3, 4]),
      dict(
          descr='data.shape=[5, 3]; mask.shape=[5]; keepdims=True',
          data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
          mask=[True, False, True, True, False],
          keepdims=True,
          expected=[[1, 2, 3], [7, 8, 9], [10, 11, 12]]),
      dict(
          descr='data.shape=[5, 3]; mask.shape=[5, 3]; keepdims=True',
          data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2], [3, 4, 5]],
          mask=[[F, F, F], [T, F, T], [T, T, T], [F, F, F], [T, T, F]],
          keepdims=True,
          expected=ragged.constant_value([[], [4, 6], [7, 8, 9], [], [3, 4]])),
      dict(
          descr='data.shape=[3, 2, 2]; mask.shape=[3]; keepdims=True',
          data=[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 4], [6, 8]]],
          mask=[F, F, T],
          keepdims=True,
          expected=[[[2, 4], [6, 8]]]),
      dict(
          descr='data.shape=[3, 2, 2]; mask.shape=[3]; keepdims=False',
          data=[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 4], [6, 8]]],
          mask=[F, F, T],
          keepdims=False,
          expected=[[[2, 4], [6, 8]]]),
      dict(
          descr='data.shape=[3, 2, 2]; mask.shape=[3, 2]; keepdims=True',
          data=[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 4], [6, 8]]],
          mask=[[T, F], [T, T], [F, F]],
          keepdims=True,
          expected=ragged.constant_value([[[1, 2]], [[5, 6], [7, 8]], []],
                                         ragged_rank=1)),
      dict(
          descr='data.shape=[3, 2, 2]; mask.shape=[3, 2]; keepdims=False',
          data=[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 4], [6, 8]]],
          mask=[[T, F], [T, T], [F, F]],
          keepdims=False,
          expected=[[1, 2], [5, 6], [7, 8]]),
      dict(
          descr='data.shape=[3, 2, 2]; mask.shape=[3, 2, 2]; keepdims=True',
          data=[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 4], [6, 8]]],
          mask=[[[T, T], [F, T]], [[F, F], [F, F]], [[T, F], [T, T]]],
          keepdims=True,
          expected=ragged.constant_value(
              [[[1, 2], [4]], [[], []], [[2], [6, 8]]])),
      dict(
          descr='data.shape=mask.shape=[2, 2, 2, 2]; keepdims=True',
          data=[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                [[[2, 4], [6, 8]], [[1, 3], [5, 7]]]],
          mask=[[[[T, T], [F, F]], [[T, F], [F, F]]],
                [[[F, F], [F, F]], [[T, T], [T, F]]]],
          keepdims=True,
          expected=ragged.constant_value(
              [[[[1, 2], []], [[5], []]], [[[], []], [[1, 3], [5]]]])),
      dict(
          descr='data.shape=mask.shape=[2, 2, 2, 2]; keepdims=False',
          data=[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                [[[2, 4], [6, 8]], [[1, 3], [5, 7]]]],
          mask=[[[[T, T], [F, F]], [[T, F], [F, F]]],
                [[[F, F], [F, F]], [[T, T], [T, F]]]],
          keepdims=False,
          expected=[1, 2, 5, 1, 3, 5]),

      #=========================================================================
      # Ragged data and ragged mask.
      #=========================================================================
      dict(
          descr='data.shape=[5, (D2)]; mask.shape=[5, (D2)]',
          data=ragged.constant_value(
              [[1, 2], [3, 4, 5, 6], [7, 8, 9], [], [1, 2, 3]]),
          mask=ragged.constant_value(
              [[F, F], [F, T, F, T], [F, F, F], [], [T, F, T]]),
          keepdims=True,
          expected=ragged.constant_value([[], [4, 6], [], [], [1, 3]])),
      dict(
          descr='data.shape=[3, (D2), (D3)]; mask.shape=[3, (D2)]',
          data=ragged.constant_value(
              [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 4], [6, 8]]]),
          mask=ragged.constant_value([[T, F], [T, T], [F, F]]),
          keepdims=True,
          expected=ragged.constant_value(
              [[[1, 2]], [[5, 6], [7, 8]], []])),
      dict(
          descr='data.shape=[3, (D2), (D3)]; mask.shape=[3, (D2)]',
          data=ragged.constant_value(
              [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 4], [6, 8]]]),
          mask=ragged.constant_value([[T, F], [T, T], [F, F]]),
          keepdims=False,
          expected=ragged.constant_value([[1, 2], [5, 6], [7, 8]])),
      dict(
          descr='data.shape=[3, (D2), D3]; mask.shape=[3, (D2)]',
          data=ragged.constant_value(
              [[[1, 2], [3, 4]], [[5, 6], [7, 8], [2, 4]], [[6, 8]]],
              ragged_rank=1),
          mask=ragged.constant_value([[T, F], [T, T, F], [F]]),
          keepdims=True,
          expected=ragged.constant_value(
              [[[1, 2]], [[5, 6], [7, 8]], []],
              ragged_rank=1)),
      dict(
          descr='data.shape=[3, (D2), D3]; mask.shape=[3, (D2)]',
          data=ragged.constant_value(
              [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 4], [6, 8]]],
              ragged_rank=1),
          mask=ragged.constant_value([[T, F], [T, T], [F, F]]),
          keepdims=False,
          expected=[[1, 2], [5, 6], [7, 8]]),
      dict(
          descr='data.shape=[3, (D2), (D3)]; mask.shape=[3, (D2), (D3)]',
          data=ragged.constant_value(
              [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 4]]]),
          mask=ragged.constant_value(
              [[[T, T], [F, T]], [[F, F], [F, F]], [[T, F]]]),
          keepdims=True,
          expected=ragged.constant_value(
              [[[1, 2], [4]], [[], []], [[2]]])),
      dict(
          descr=('data.shape=[3, (D2), (D3), (D4)]; '
                 'mask.shape=[3, (D2), (D3), (D4)]'),
          data=ragged.constant_value(
              [[[[1, 2], [3, 4]], [[5, 6]]], [[[2, 4], [6, 8]]]]),
          mask=ragged.constant_value(
              [[[[T, T], [F, F]], [[T, F]]], [[[F, F], [T, T]]]]),
          keepdims=True,
          expected=ragged.constant_value(
              [[[[1, 2], []], [[5]]], [[[], [6, 8]]]])),

      #=========================================================================
      # Ragged mask and uniform data
      #=========================================================================
      dict(
          descr='data.shape=[2, 3]; mask.shape=[2, (3)]',
          data=[[1, 2, 3], [4, 5, 6]],
          mask=ragged.constant_value([[T, F, F], [F, T, T]]),
          keepdims=True,
          expected=ragged.constant_value([[1], [5, 6]])),
      dict(
          descr='data.shape=[2, 3, 2]; mask.shape=[2, (3)]',
          data=[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 0], [2, 4]]],
          mask=ragged.constant_value([[T, F, F], [F, T, T]]),
          keepdims=True,
          expected=ragged.constant_value(
              [[[1, 2]], [[9, 0], [2, 4]]],
              ragged_rank=1)),
      dict(
          descr='data.shape=[2, 3, 2]; mask.shape=[2, (3), 2]',
          data=[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 0], [2, 4]]],
          mask=ragged.constant_value(
              [[[T, F], [F, F], [T, T]], [[T, F], [F, T], [F, F]]],
              ragged_rank=1),
          keepdims=True,
          expected=ragged.constant_value([[[1], [], [5, 6]], [[7], [0], []]])),

      #=========================================================================
      # Ragged data and uniform mask.
      #=========================================================================
      dict(
          descr='data.shape=[4, (D2)]; mask.shape=[4]',
          data=ragged.constant_value([[1, 2, 3], [4], [], [5, 6]]),
          mask=[T, F, T, F],
          keepdims=False,
          expected=ragged.constant_value([[1, 2, 3], []])),
      dict(
          descr='data.shape=[4, (D2), (D3)]; mask.shape=[4]',
          data=ragged.constant_value(
              [[[1, 2, 3]], [[4], []], [[5, 6]], []]),
          mask=[T, F, T, T],
          keepdims=False,
          expected=ragged.constant_value([[[1, 2, 3]], [[5, 6]], []])),
      dict(
          descr='data.shape=[4, (D2), 2]; mask.shape=[4]',
          data=ragged.constant_value(
              [[[1, 2], [3, 4]], [], [[5, 6]], [[7, 8], [9, 0], [1, 2]]],
              ragged_rank=1),
          mask=[T, F, F, T],
          keepdims=False,
          expected=ragged.constant_value(
              [[[1, 2], [3, 4]], [[7, 8], [9, 0], [1, 2]]],
              ragged_rank=1)),
      dict(
          descr='data.shape=[4, (D2), 2]; mask.shape=[4]',
          data=ragged.constant_value(
              [[[1, 2], [3, 4]], [], [[5, 6]], [[7, 8], [9, 0], [1, 2]]],
              ragged_rank=1),
          mask=[T, F, F, T],
          keepdims=True,
          expected=ragged.constant_value(
              [[[1, 2], [3, 4]], [[7, 8], [9, 0], [1, 2]]],
              ragged_rank=1)),
      dict(
          descr='data.shape=[1, (2)]; mask.shape=[1, 2]',
          data=ragged.constant_value([[1, 2]]),
          mask=[[T, F]],
          keepdims=True,
          expected=ragged.constant_value([[1]])),
      dict(
          descr='data.shape=[2, (2), (D3)]; mask.shape=[2, 2]',
          data=ragged.constant_value([[[1], [2, 3]], [[], [4, 5, 6]]]),
          mask=[[T, F], [T, T]],
          keepdims=True,
          expected=ragged.constant_value([[[1]], [[], [4, 5, 6]]])),
      dict(
          descr='data.shape=[2, (2), 3]; mask.shape=[2, 2]',
          data=ragged.constant_value(
              [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [2, 4, 6]]],
              ragged_rank=1),
          mask=[[T, F], [T, T]],
          keepdims=True,
          expected=ragged.constant_value(
              [[[1, 2, 3]], [[7, 8, 9], [2, 4, 6]]],
              ragged_rank=1)),
      dict(
          descr='data.shape=[2, (2), 3]; mask.shape=[2, 2, 3]',
          data=ragged.constant_value(
              [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [2, 4, 6]]],
              ragged_rank=1),
          mask=[[[T, F, F], [T, F, T]], [[T, F, T], [F, F, F]]],
          keepdims=True,
          expected=ragged.constant_value([[[1], [4, 6]], [[7, 9], []]])),
  ])  # pyformat: disable
  def testBooleanMask(self, descr, data, mask, keepdims, expected):
    actual = ragged.boolean_mask(data, mask, keepdims=keepdims)
    self.assertEqual(
        getattr(actual, 'ragged_rank', 0), getattr(expected, 'ragged_rank', 0))
    with self.test_session():
      if isinstance(expected, ragged.RaggedTensorValue):
        expected = expected.tolist()
      self.assertEqual(actual.eval().tolist(), expected)

  def testErrors(self):
    self.assertRaisesRegexp(ValueError,
                            r'mask\.shape\.ndims must be kown statically',
                            ragged.boolean_mask, [[1, 2]],
                            array_ops.placeholder(dtypes.bool))

    self.assertRaisesRegexp(TypeError,
                            "Expected bool, got 0 of type 'int' instead.",
                            ragged.boolean_mask, [[1, 2]], [[0, 1]])
    self.assertRaisesRegexp(
        ValueError, 'Tensor conversion requested dtype bool for '
        'RaggedTensor with dtype int32', ragged.boolean_mask,
        ragged.constant([[1, 2]]), ragged.constant([[0, 0]]))

    self.assertRaisesRegexp(
        ValueError, r'Shapes \(1, 2\) and \(1, 3\) are incompatible',
        ragged.boolean_mask, [[1, 2]], [[True, False, True]])

    # self.assertRaisesRegexp(ValueError,
    #                         r'data=.* is non-ragged but mask=.* is ragged',
    #                         ragged.boolean_mask, [[1, 2]],
    #                         ragged.constant([[True, False]]))

    # self.assertRaisesRegexp(
    #     ValueError, r'data=.* is ragged but mask=.* is non-ragged',
    #     ragged.boolean_mask, ragged.constant([[1, 2]]), [[True, False]])

    self.assertRaisesRegexp(errors.InvalidArgumentError,
                            r'Inputs must have identical ragged splits',
                            ragged.boolean_mask, ragged.constant([[1, 2]]),
                            ragged.constant([[True, False, True]]))

    self.assertRaisesRegexp(ValueError, 'mask cannot be scalar',
                            ragged.boolean_mask, [[1, 2]], True)

    self.assertRaisesRegexp(ValueError,
                            'mask cannot be scalar', ragged.boolean_mask,
                            ragged.constant([[1, 2]]), True)


if __name__ == '__main__':
  googletest.main()
