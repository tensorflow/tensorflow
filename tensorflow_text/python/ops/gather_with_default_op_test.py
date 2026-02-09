# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

"""Tests for gather_with_default op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow_text.python.ops import pointer_ops


def _MakeTestTensor(shape, prefix=b'v'):
  """Constructs a string tensor with the specified shape, for testing."""
  if not shape:
    return prefix
  return [
      _MakeTestTensor(shape[1:], b'%s%s' % (prefix, ('%s' % i).encode('ascii')))
      for i in range(shape[0])
  ]


@test_util.run_all_in_graph_and_eager_modes
class GatherWithDefaultOpTest(test_util.TensorFlowTestCase,
                              parameterized.TestCase):

  def testDocStringExample(self):
    gathered = pointer_ops.gather_with_default(['a', 'b', 'c', 'd'],
                                               [2, 0, -1, 2, -1], '_')
    self.assertAllEqual(gathered, [b'c', b'a', b'_', b'c', b'_'])

  @parameterized.parameters(
      (_MakeTestTensor([8]), -1, b'_'),
      (_MakeTestTensor([8]), 0, b'_'),
      (_MakeTestTensor([8]), 1, b'_'),
      (_MakeTestTensor([8]), 6, b'_'),
      (_MakeTestTensor([8]), 7, b'_'),
  )
  def testScalarIndicesWith1DParams(self, params, indices, default):
    indices_t = constant_op.constant(indices, dtype=dtypes.int32)
    params_t = constant_op.constant(params)
    assert isinstance(indices, int)
    gathered = pointer_ops.gather_with_default(params_t, indices_t, default)
    expected = default if indices == -1 else params[indices]

    self.assertAllEqual(expected, gathered)
    # When there are no -1 indices, check that behavior matches tf.gather.
    if indices != -1:
      self.assertAllEqual(gathered, array_ops.gather(params_t, indices_t))

  @parameterized.parameters(
      (_MakeTestTensor([3, 2]), -1, [b'_', b'_']),
      (_MakeTestTensor([3, 2]), 0, [b'_', b'_']),
      (_MakeTestTensor([3, 2]), 1, [b'_', b'_']),
      (_MakeTestTensor([3, 2]), 2, [b'_', b'_']),
  )
  def testScalarIndicesWith2DParams(self, params, indices, default):
    indices_t = constant_op.constant(indices, dtype=dtypes.int32)
    params_t = constant_op.constant(params)
    assert isinstance(indices, int)
    gathered = pointer_ops.gather_with_default(params_t, indices_t, default)
    expected = default if indices == -1 else params[indices]
    self.assertAllEqual(gathered, expected)
    # When there are no -1 indices, check that behavior matches tf.gather.
    if indices != -1:
      self.assertAllEqual(gathered, array_ops.gather(params_t, indices_t))

  @parameterized.parameters(
      # 1D params
      (_MakeTestTensor([8]), [], '_'),
      (_MakeTestTensor([8]), [0], '_'),
      (_MakeTestTensor([8]), [-1], '_'),
      (_MakeTestTensor([8]), [6], '_'),
      (_MakeTestTensor([8]), [2, 0, 2, -1, 5, -1], '_'),
      (_MakeTestTensor([8]), [2, 0, 2, 1, 5, 3], '_'),
      # 2D params
      (_MakeTestTensor([3, 2]), [], ['_', '_'], [0, 2]),
      (_MakeTestTensor([3, 2]), [0], ['_', '_']),
      (_MakeTestTensor([3, 2]), [1], ['_', '_']),
      (_MakeTestTensor([3, 2]), [-1], ['_', '_']),
      (_MakeTestTensor([3, 2]), [2], ['_', '_']),
      (_MakeTestTensor([3, 2]), [1, 0, -1, 2, -1], ['_', '_']),
      (_MakeTestTensor([3, 2]), [1, 0, 1, 2, 0], ['_', '_']),
  )
  def testVectorIndices(self, params, indices, default, expected_shape=None):
    indices_t = constant_op.constant(indices, dtype=dtypes.int32)
    params_t = constant_op.constant(params)
    gathered = pointer_ops.gather_with_default(params_t, indices_t, default)
    expected = [default if i == -1 else params[i] for i in indices]
    expected = constant_op.constant(expected, shape=expected_shape)
    self.assertAllEqual(gathered, expected)
    # When there are no -1 indices, check that behavior matches tf.gather.
    if not any(i == -1 for i in indices):
      self.assertAllEqual(gathered, array_ops.gather(params_t, indices_t))

  @parameterized.parameters(
      # 1D params
      (_MakeTestTensor([8]), [], '_'),
      (_MakeTestTensor([8]), [[0]], '_'),
      (_MakeTestTensor([8]), [[-1]], '_'),
      (_MakeTestTensor([8]), [[6]], '_'),
      (_MakeTestTensor([8]), [[2, 0], [2, -1], [5, -1]], '_'),
      (_MakeTestTensor([8]), [[2, 0], [2, 1], [5, 2]], '_'),
      # 2D params
      (_MakeTestTensor([3, 2]), [], ['_', '_'], [0, 2]),
      (_MakeTestTensor([3, 2]), [[0]], ['_', '_']),
      (_MakeTestTensor([3, 2]), [[1]], ['_', '_']),
      (_MakeTestTensor([3, 2]), [[-1]], ['_', '_']),
      (_MakeTestTensor([3, 2]), [[2]], ['_', '_']),
      (_MakeTestTensor([3, 2]), [[1, 0], [-1, 2], [-1, -1]], ['_', '_']),
      (_MakeTestTensor([3, 2]), [[1, 0], [1, 2], [0, 0]], ['_', '_']),
  )
  def test2DIndices(self, params, indices, default, expected_shape=None):
    indices_t = constant_op.constant(indices, dtype=dtypes.int32)
    params_t = constant_op.constant(params)
    gathered = pointer_ops.gather_with_default(params_t, indices_t, default)
    expected = [[default if i == -1 else params[i]
                 for i in indices_row]
                for indices_row in indices]
    expected = constant_op.constant(expected, shape=expected_shape)
    self.assertAllEqual(gathered, expected)
    # When there are no -1 indices, check that behavior matches tf.gather.
    if not any(i == -1 for index_row in indices for i in index_row):
      self.assertAllEqual(gathered, array_ops.gather(params_t, indices_t))

  def testAxisGreaterThan0(self):
    params = [['a0', 'a1', 'a2', 'a3', 'a4'],
              ['b0', 'b1', 'b2', 'b3', 'b4'],
              ['c0', 'c1', 'c2', 'c3', 'c4']]  # pyformat: disable
    indices = [2, 0, -1, 4, -1]
    gathered = pointer_ops.gather_with_default(params, indices, '__', axis=1)
    expected = [[b'a2', b'a0', b'__', b'a4', b'__'],
                [b'b2', b'b0', b'__', b'b4', b'__'],
                [b'c2', b'c0', b'__', b'c4', b'__']]  # pyformat: disable
    self.assertAllEqual(gathered, expected)

  def testNegativeAxis(self):
    params_1d = _MakeTestTensor(shape=[3])
    params_2d = _MakeTestTensor(shape=[3, 3])
    params_3d = _MakeTestTensor(shape=[3, 3, 3])
    indices = [2, 0, -1, 1, -1]

    gathered1a = pointer_ops.gather_with_default(
        params_1d, indices, '__', axis=0)
    gathered1b = pointer_ops.gather_with_default(
        params_1d, indices, '__', axis=-1)
    expected1 = [b'v2', b'v0', b'__', b'v1', b'__']

    gathered2a = pointer_ops.gather_with_default(
        params_2d, indices, ['__', '__', '__'], axis=0)
    gathered2b = pointer_ops.gather_with_default(
        params_2d, indices, ['__', '__', '__'], axis=-2)
    expected2 = [[b'v20', b'v21', b'v22'],
                 [b'v00', b'v01', b'v02'],
                 [b'__', b'__', b'__'],
                 [b'v10', b'v11', b'v12'],
                 [b'__', b'__', b'__']]  # pyformat: disable

    gathered3a = pointer_ops.gather_with_default(
        params_2d, indices, '__', axis=1)
    gathered3b = pointer_ops.gather_with_default(
        params_2d, indices, '__', axis=-1)
    expected3 = [[b'v02', b'v00', b'__', b'v01', b'__'],
                 [b'v12', b'v10', b'__', b'v11', b'__'],
                 [b'v22', b'v20', b'__', b'v21', b'__']]  # pyformat: disable

    gathered4a = pointer_ops.gather_with_default(
        params_3d, indices, '__', axis=2)
    gathered4b = pointer_ops.gather_with_default(
        params_3d, indices, '__', axis=-1)
    expected4 = [[[b'v002', b'v000', b'__', b'v001', b'__'],
                  [b'v012', b'v010', b'__', b'v011', b'__'],
                  [b'v022', b'v020', b'__', b'v021', b'__']],
                 [[b'v102', b'v100', b'__', b'v101', b'__'],
                  [b'v112', b'v110', b'__', b'v111', b'__'],
                  [b'v122', b'v120', b'__', b'v121', b'__']],
                 [[b'v202', b'v200', b'__', b'v201', b'__'],
                  [b'v212', b'v210', b'__', b'v211', b'__'],
                  [b'v222', b'v220', b'__', b'v221', b'__']]]

    self.assertAllEqual(gathered1a, expected1)
    self.assertAllEqual(gathered1b, expected1)
    self.assertAllEqual(gathered2a, expected2)
    self.assertAllEqual(gathered2b, expected2)
    self.assertAllEqual(gathered3a, expected3)
    self.assertAllEqual(gathered3b, expected3)
    self.assertAllEqual(gathered4a, expected4)
    self.assertAllEqual(gathered4b, expected4)

  def testAxisGreaterThan0_BehaviorMatchesTfGather(self):
    params = [['a1', 'a2', 'a3', 'a4'], ['b1', 'b2', 'b3', 'b4'],
              ['c1', 'c2', 'c3', 'c4']]
    indices = [2, 0, 2, 1]
    gathered = pointer_ops.gather_with_default(params, indices, '__', axis=1)
    expected = array_ops.gather(params, indices, axis=1)
    self.assertAllEqual(gathered, expected)

  def testBadDefaultShape(self):
    with self.assertRaises((ValueError, errors.InvalidArgumentError)):
      pointer_ops.gather_with_default(
          params=[0, 1, 2, 3], indices=[0], default=[0])
    with self.assertRaises((ValueError, errors.InvalidArgumentError)):
      pointer_ops.gather_with_default(
          params=[[0, 1], [2, 3]], indices=[0], default=0)

  def testBadDefaultDtype(self):
    with self.assertRaisesRegex(
        TypeError, 'Expected int32.*|Cannot convert .*'
    ):
      pointer_ops.gather_with_default(
          params=[0, 1, 2, 3], indices=[0], default='a')

  def testBadAxis(self):
    with self.assertRaises((ValueError, errors.InvalidArgumentError)):
      pointer_ops.gather_with_default(
          params=[0, 1, 2, 3], indices=[0], default=-1, axis=1)
    with self.assertRaises((ValueError, errors.InvalidArgumentError)):
      pointer_ops.gather_with_default(
          params=[[0, 1], [2, 3]], indices=[0], default=[0, 0], axis=2)

  def testIndexOutOfRange(self):
    # Note: because of the way gather_with_default is implemented, these
    # error messages will report values and ranges that are one greater than
    # those that were supplied to gather_with_default.
    with self.assertRaisesRegex(
        errors.InvalidArgumentError, r'indices\[0\] = .* is not in .*'
    ):
      self.evaluate(
          pointer_ops.gather_with_default(
              params=[0, 1, 2, 3], indices=[4], default=0))

    with self.assertRaisesRegex(
        errors.InvalidArgumentError, r'indices\[0\] = .* is not in .*'
    ):
      self.evaluate(
          pointer_ops.gather_with_default(
              params=[0, 1, 2, 3], indices=[-2], default=0))


if __name__ == '__main__':
  test.main()
