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
"""Tests for ragged_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.platform import googletest

# Example 3d tensor for test cases.  Has shape [4, 2, 3].
TENSOR_3D = [[[('%d%d%d' % (i, j, k)).encode('utf-8')
               for k in range(3)]
              for j in range(2)]
             for i in range(4)]

# Example 4d tensor for test cases.  Has shape [4, 2, 3, 5].
TENSOR_4D = [[[[('%d%d%d%d' % (i, j, k, l)).encode('utf-8')
                for l in range(5)]
               for k in range(3)]
              for j in range(2)]
             for i in range(4)]


class RaggedRepeatTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.parameters([
      # Docstring examples
      dict(
          data=['a', 'b', 'c'],
          repeats=[3, 0, 2],
          axis=0,
          expected=[b'a', b'a', b'a', b'c', b'c']),
      dict(
          data=[[1, 2], [3, 4]],
          repeats=[2, 3],
          axis=0,
          expected=[[1, 2], [1, 2], [3, 4], [3, 4], [3, 4]]),
      dict(
          data=[[1, 2], [3, 4]],
          repeats=[2, 3],
          axis=1,
          expected=[[1, 1, 2, 2, 2], [3, 3, 4, 4, 4]]),

      # Scalar repeats value
      dict(
          data=['a', 'b', 'c'],
          repeats=2,
          axis=0,
          expected=[b'a', b'a', b'b', b'b', b'c', b'c']),
      dict(
          data=[[1, 2], [3, 4]],
          repeats=2,
          axis=0,
          expected=[[1, 2], [1, 2], [3, 4], [3, 4]]),
      dict(
          data=[[1, 2], [3, 4]],
          repeats=2,
          axis=1,
          expected=[[1, 1, 2, 2], [3, 3, 4, 4]]),

      # data & repeats are broadcast to have at least one dimension,
      # so these are all equivalent:
      dict(data=3, repeats=4, axis=0, expected=[3, 3, 3, 3]),
      dict(data=[3], repeats=4, axis=0, expected=[3, 3, 3, 3]),
      dict(data=3, repeats=[4], axis=0, expected=[3, 3, 3, 3]),
      dict(data=[3], repeats=[4], axis=0, expected=[3, 3, 3, 3]),
      # Empty tensor
      dict(data=[], repeats=[], axis=0, expected=[]),
  ])
  @test_util.run_v1_only('b/120545219')
  def testRepeat(self, data, repeats, expected, axis=None):
    result = ragged_util.repeat(data, repeats, axis)
    with self.test_session():
      self.assertEqual(result.eval().tolist(), expected)

  @parameterized.parameters([
      dict(mode=mode, **args)
      for mode in ['constant', 'dynamic', 'unknown_shape']
      for args in [
          # data & repeats are broadcast to have at least one dimension,
          # so these are all equivalent:
          dict(data=3, repeats=4, axis=0),
          dict(data=[3], repeats=4, axis=0),
          dict(data=3, repeats=[4], axis=0),
          dict(data=[3], repeats=[4], axis=0),

          # 1-dimensional data tensor.
          dict(data=[], repeats=5, axis=0),
          dict(data=[1, 2, 3], repeats=5, axis=0),
          dict(data=[1, 2, 3], repeats=[3, 0, 2], axis=0),
          dict(data=[1, 2, 3], repeats=[3, 0, 2], axis=-1),
          dict(data=[b'a', b'b', b'c'], repeats=[3, 0, 2], axis=0),

          # 2-dimensional data tensor.
          dict(data=[[1, 2, 3], [4, 5, 6]], repeats=3, axis=0),
          dict(data=[[1, 2, 3], [4, 5, 6]], repeats=3, axis=1),
          dict(data=[[1, 2, 3], [4, 5, 6]], repeats=[3, 5], axis=0),
          dict(data=[[1, 2, 3], [4, 5, 6]], repeats=[3, 5, 7], axis=1),

          # 3-dimensional data tensor: shape=[4, 2, 3].
          dict(data=TENSOR_3D, repeats=2, axis=0),
          dict(data=TENSOR_3D, repeats=2, axis=1),
          dict(data=TENSOR_3D, repeats=2, axis=2),
          dict(data=TENSOR_3D, repeats=[2, 0, 4, 1], axis=0),
          dict(data=TENSOR_3D, repeats=[3, 2], axis=1),
          dict(data=TENSOR_3D, repeats=[1, 3, 1], axis=2),

          # 4-dimensional data tensor: shape=[4, 2, 3, 5].
          dict(data=TENSOR_4D, repeats=2, axis=0),
          dict(data=TENSOR_4D, repeats=2, axis=1),
          dict(data=TENSOR_4D, repeats=2, axis=2),
          dict(data=TENSOR_4D, repeats=2, axis=3),
          dict(data=TENSOR_4D, repeats=[2, 0, 4, 1], axis=0),
          dict(data=TENSOR_4D, repeats=[3, 2], axis=1),
          dict(data=TENSOR_4D, repeats=[1, 3, 1], axis=2),
          dict(data=TENSOR_4D, repeats=[1, 3, 0, 0, 2], axis=3),
      ]
  ])
  @test_util.run_v1_only('b/120545219')
  def testValuesMatchesNumpy(self, mode, data, repeats, axis):
    # Exception: we can't handle negative axis if data.ndims is unknown.
    if axis < 0 and mode == 'unknown_shape':
      return

    expected = np.repeat(data, repeats, axis)

    if mode == 'constant':
      data = constant_op.constant(data)
      repeats = constant_op.constant(repeats)
    elif mode == 'dynamic':
      data = constant_op.constant(data)
      repeats = constant_op.constant(repeats)
      data = array_ops.placeholder_with_default(data, data.shape)
      repeats = array_ops.placeholder_with_default(repeats, repeats.shape)
    elif mode == 'unknown_shape':
      data = array_ops.placeholder_with_default(data, None)
      repeats = array_ops.placeholder_with_default(repeats, None)

    result = ragged_util.repeat(data, repeats, axis)
    with self.test_session():
      self.assertEqual(result.eval().tolist(), expected.tolist())

  @parameterized.parameters([
      dict(
          descr='axis >= rank(data)',
          mode='dynamic',
          data=[1, 2, 3],
          repeats=[3, 0, 2],
          axis=1,
          error='axis=1 out of bounds: expected -1<=axis<1'),
      dict(
          descr='axis < -rank(data)',
          mode='dynamic',
          data=[1, 2, 3],
          repeats=[3, 0, 2],
          axis=-2,
          error='axis=-2 out of bounds: expected -1<=axis<1'),
      dict(
          descr='len(repeats) != data.shape[axis]',
          mode='dynamic',
          data=[[1, 2, 3], [4, 5, 6]],
          repeats=[2, 3],
          axis=1,
          error='Dimensions 3 and 2 are not compatible'),
      dict(
          descr='rank(repeats) > 1',
          mode='dynamic',
          data=[[1, 2, 3], [4, 5, 6]],
          repeats=[[3], [5]],
          axis=1,
          error=r'Shape \(2, 1\) must have rank at most 1'),
      dict(
          descr='non-integer axis',
          mode='constant',
          data=[1, 2, 3],
          repeats=2,
          axis='foo',
          exception=TypeError,
          error='axis must be an int'),
  ])
  def testError(self,
                descr,
                mode,
                data,
                repeats,
                axis,
                exception=ValueError,
                error=None):
    # Make sure that this is also an error case for numpy.
    with self.assertRaises(exception):
      np.repeat(data, repeats, axis)

    if mode == 'constant':
      data = constant_op.constant(data)
      repeats = constant_op.constant(repeats)
    elif mode == 'dynamic':
      data = constant_op.constant(data)
      repeats = constant_op.constant(repeats)
      data = array_ops.placeholder_with_default(data, data.shape)
      repeats = array_ops.placeholder_with_default(repeats, repeats.shape)
    elif mode == 'unknown_shape':
      data = array_ops.placeholder_with_default(data, None)
      repeats = array_ops.placeholder_with_default(repeats, None)

    with self.assertRaisesRegexp(exception, error):
      ragged_util.repeat(data, repeats, axis)


if __name__ == '__main__':
  googletest.main()
