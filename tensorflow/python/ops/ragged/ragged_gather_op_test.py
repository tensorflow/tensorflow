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
"""Tests for ragged.gather."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import ragged
from tensorflow.python.platform import googletest


class RaggedTensorOpsTest(test_util.TensorFlowTestCase):

  def testDocStringExamples(self):
    params = constant_op.constant(['a', 'b', 'c', 'd', 'e'])
    indices = constant_op.constant([3, 1, 2, 1, 0])
    ragged_params = ragged.constant([['a', 'b', 'c'], ['d'], [], ['e']])
    ragged_indices = ragged.constant([[3, 1, 2], [1], [], [0]])
    with self.test_session():
      self.assertEqual(
          ragged.gather(params, ragged_indices).eval().tolist(),
          [[b'd', b'b', b'c'], [b'b'], [], [b'a']])
      self.assertEqual(
          ragged.gather(ragged_params, indices).eval().tolist(),
          [[b'e'], [b'd'], [], [b'd'], [b'a', b'b', b'c']])
      self.assertEqual(
          ragged.gather(ragged_params, ragged_indices).eval().tolist(),
          [[[b'e'], [b'd'], []], [[b'd']], [], [[b'a', b'b', b'c']]])

  def testTensorParamsAndTensorIndices(self):
    params = ['a', 'b', 'c', 'd', 'e']
    indices = [2, 0, 2, 1]
    with self.test_session():
      self.assertEqual(
          ragged.gather(params, indices).eval().tolist(),
          [b'c', b'a', b'c', b'b'])
      self.assertEqual(type(ragged.gather(params, indices)), ops.Tensor)

  def testRaggedParamsAndTensorIndices(self):
    params = ragged.constant([['a', 'b'], ['c', 'd', 'e'], ['f'], [], ['g']])
    indices = [2, 0, 2, 1]
    with self.test_session():
      self.assertEqual(
          ragged.gather(params, indices).eval().tolist(),
          [[b'f'], [b'a', b'b'], [b'f'], [b'c', b'd', b'e']])

  def testTensorParamsAndRaggedIndices(self):
    params = ['a', 'b', 'c', 'd', 'e']
    indices = ragged.constant([[2, 1], [1, 2, 0], [3]])
    with self.test_session():
      self.assertEqual(
          ragged.gather(params, indices).eval().tolist(),
          [[b'c', b'b'], [b'b', b'c', b'a'], [b'd']])

  def testRaggedParamsAndRaggedIndices(self):
    params = ragged.constant([['a', 'b'], ['c', 'd', 'e'], ['f'], [], ['g']])
    indices = ragged.constant([[2, 1], [1, 2, 0], [3]])
    with self.test_session():
      self.assertEqual(
          ragged.gather(params, indices).eval().tolist(),
          [[[b'f'], [b'c', b'd', b'e']],                # [[p[2], p[1]      ],
           [[b'c', b'd', b'e'], [b'f'], [b'a', b'b']],  #  [p[1], p[2], p[0]],
           [[]]]                                        #  [p[3]            ]]
      )  # pyformat: disable

  def testRaggedParamsAndScalarIndices(self):
    params = ragged.constant([['a', 'b'], ['c', 'd', 'e'], ['f'], [], ['g']])
    indices = 1
    with self.test_session():
      self.assertEqual(
          ragged.gather(params, indices).eval().tolist(), [b'c', b'd', b'e'])

  def test3DRaggedParamsAnd2DTensorIndices(self):
    params = ragged.constant([[['a', 'b'], []], [['c', 'd'], ['e'], ['f']],
                              [['g']]])
    indices = [[1, 2], [0, 1], [2, 2]]
    with self.test_session():
      self.assertEqual(
          ragged.gather(params, indices).eval().tolist(),
          [[[[b'c', b'd'], [b'e'], [b'f']], [[b'g']]],            # [[p1, p2],
           [[[b'a', b'b'], []], [[b'c', b'd'], [b'e'], [b'f']]],  #  [p0, p1],
           [[[b'g']], [[b'g']]]]                                  #  [p2, p2]]
      )  # pyformat: disable

  def testTensorParamsAnd4DRaggedIndices(self):
    indices = ragged.constant(
        [[[[3, 4], [0, 6]], []], [[[2, 1], [1, 0]], [[2, 5]], [[2, 3]]],
         [[[1, 0]]]],  # pyformat: disable
        ragged_rank=2,
        inner_shape=(2,))
    params = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    with self.test_session():
      self.assertEqual(
          ragged.gather(params, indices).eval().tolist(),
          [[[[b'd', b'e'], [b'a', b'g']], []],
           [[[b'c', b'b'], [b'b', b'a']], [[b'c', b'f']], [[b'c', b'd']]],
           [[[b'b', b'a']]]])  # pyformat: disable

  def testOutOfBoundsError(self):
    tensor_params = ['a', 'b', 'c']
    tensor_indices = [0, 1, 2]
    ragged_params = ragged.constant([['a', 'b'], ['c']])
    ragged_indices = ragged.constant([[0, 3]])
    with self.test_session():
      self.assertRaisesRegexp(errors.InvalidArgumentError,
                              r'indices\[1\] = 3 is not in \[0, 3\)',
                              ragged.gather(tensor_params, ragged_indices).eval)
      self.assertRaisesRegexp(errors.InvalidArgumentError,
                              r'indices\[2\] = 2 is not in \[0, 2\)',
                              ragged.gather(ragged_params, tensor_indices).eval)
      self.assertRaisesRegexp(errors.InvalidArgumentError,
                              r'indices\[1\] = 3 is not in \[0, 2\)',
                              ragged.gather(ragged_params, ragged_indices).eval)

  def testUnknownIndicesRankError(self):
    params = ragged.constant([], ragged_rank=1)
    indices = constant_op.constant([0], dtype=dtypes.int64)
    indices = array_ops.placeholder_with_default(indices, None)
    self.assertRaisesRegexp(ValueError,
                            r'indices\.shape\.ndims must be known statically',
                            ragged.gather, params, indices)


if __name__ == '__main__':
  googletest.main()
