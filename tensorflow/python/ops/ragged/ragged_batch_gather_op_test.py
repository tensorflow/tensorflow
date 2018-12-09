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
"""Tests for tf.ragged.batch_gather."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import ragged
from tensorflow.python.ops.ragged import ragged_test_util
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedBatchGatherOpTest(ragged_test_util.RaggedTensorTestCase,
                              parameterized.TestCase):

  @parameterized.parameters([
      #=========================================================================
      # Docstring Example
      #=========================================================================
      dict(
          descr='Docstring example',
          params=ragged.constant_value([['a', 'b', 'c'], ['d'], [], ['e']]),
          indices=ragged.constant_value([[1, 2, 0], [], [], [0, 0]]),
          expected=ragged.constant_value([[b'b', b'c', b'a'], [], [],
                                          [b'e', b'e']])),
      #=========================================================================
      # 0 Batch Dimensions
      #=========================================================================
      dict(
          descr='params: [P1], indices: [I], result: [I]',
          params=['a', 'b', 'c', 'd'],
          indices=[3, 2],
          expected=[b'd', b'c']),
      dict(
          descr='params: [P1, (P2)], indices: [I], result: [I, (P2)]',
          params=ragged.constant_value([['a', 'b'], [], ['c'], ['d', 'e']]),
          indices=[3, 2],
          expected=ragged.constant_value([[b'd', b'e'], [b'c']])),
      #=========================================================================
      # 1 Batch Dimension
      #=========================================================================
      dict(
          descr='params: [B1, P1], indices: [B1, I], result: [B1, I]',
          params=[['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']],
          indices=[[2, 0], [0, 1], [1, 0]],
          expected=[[b'c', b'a'], [b'd', b'e'], [b'h', b'g']]),
      dict(
          descr='params: [B1, (P1)], indices: [B1, I], result: [B1, I]',
          params=ragged.constant_value([['a', 'b', 'c'], ['d', 'e'], ['g']]),
          indices=[[2, 0], [0, 1], [0, 0]],
          expected=[[b'c', b'a'], [b'd', b'e'], [b'g', b'g']]),
      dict(
          descr='params: [B1, P1], indices: [B1, (I)], result: [B1, (I)]',
          params=[['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']],
          indices=ragged.constant_value([[2, 0, 2], [0], [1]]),
          expected=ragged.constant_value([[b'c', b'a', b'c'], [b'd'], [b'h']])),
      dict(
          descr=('params: [B1, (P1), (P2), P3], indices: [B1, I], '
                 'result: [B1, I, (P2), P3]'),
          params=ragged.constant_value(
              [[[['a']], [['b'], ['c']]], [[['d'], ['e']], [['f']]], [[['g']]]],
              ragged_rank=2),
          indices=[[1, 0], [0, 1], [0, 0]],
          expected=ragged.constant_value(
              [[[[b'b'], [b'c']], [[b'a']]], [[[b'd'], [b'e']], [[b'f']]],
               [[[b'g']], [[b'g']]]],
              ragged_rank=2)),
      #=========================================================================
      # 2 Batch Dimensions
      #=========================================================================
      dict(
          descr=('params: [B1, B2, P1], indices: [B1, B2, I], '
                 'result: [B1, B2, I]'),
          params=[[['a', 'b', 'c']], [['d', 'e', 'f']], [['g', 'h', 'i']]],
          indices=[[[2, 0]], [[0, 1]], [[1, 0]]],
          expected=[[[b'c', b'a']], [[b'd', b'e']], [[b'h', b'g']]]),
      dict(
          descr=('params: [B1, (B2), P1], indices: [B1, (B2), I], '
                 'result: [B1, (B2), I]'),
          params=ragged.constant_value(
              [[['a', 'b', 'c'], ['d', 'e', 'f']], [['g', 'h', 'i']]],
              ragged_rank=1),
          indices=ragged.constant_value([[[2, 0], [0, 1]], [[1, 0]]],
                                        ragged_rank=1),
          expected=ragged.constant_value(
              [[[b'c', b'a'], [b'd', b'e']], [[b'h', b'g']]], ragged_rank=1)),
      dict(
          descr=('params: [B1, (B2), (P1)], indices: [B1, (B2), I], '
                 'result: [B1, (B2), I]'),
          params=ragged.constant_value([[['a', 'b', 'c'], ['d']], [['e', 'f']]],
                                       ragged_rank=2),
          indices=ragged.constant_value([[[2, 0], [0, 0]], [[1, 0]]],
                                        ragged_rank=1),
          expected=ragged.constant_value(
              [[[b'c', b'a'], [b'd', b'd']], [[b'f', b'e']]], ragged_rank=1)),
      dict(
          descr=('params: [B1, (B2), P1], indices: [B1, (B2), (I)], '
                 'result: [B1, (B2), (I)]'),
          params=ragged.constant_value(
              [[['a', 'b', 'c'], ['d', 'e', 'f']], [['g', 'h', 'i']]],
              ragged_rank=1),
          indices=ragged.constant_value([[[2, 1, 0], [0]], [[1, 1]]],
                                        ragged_rank=2),
          expected=ragged.constant_value(
              [[[b'c', b'b', b'a'], [b'd']], [[b'h', b'h']]], ragged_rank=2)),
      #=========================================================================
      # 3 Batch Dimensions
      #=========================================================================
      dict(
          descr=(
              'params: [B1, (B2), (B3), (P1)], indices: [B1, (B2), (B3), I], '
              'result: [B1, (B2), (B3), I]'),
          params=ragged.constant_value(
              [[[['a', 'b', 'c'], ['d']], [['e', 'f']]]], ragged_rank=3),
          indices=ragged.constant_value([[[[2, 0], [0, 0]], [[1, 0]]]],
                                        ragged_rank=2),
          expected=ragged.constant_value(
              [[[[b'c', b'a'], [b'd', b'd']], [[b'f', b'e']]]], ragged_rank=2)),
  ])
  def testRaggedBatchGather(self, descr, params, indices, expected):
    result = ragged.batch_gather(params, indices)
    self.assertRaggedEqual(result, expected)

  def testRaggedBatchGatherUnknownRankError(self):
    if context.executing_eagerly():
      return
    params = [['a', 'b'], ['c', 'd']]
    indices = array_ops.placeholder(dtypes.int32, shape=None)
    ragged_indices = ragged.RaggedTensor.from_row_splits(indices, [0, 2, 4])

    with self.assertRaisesRegexp(
        ValueError, 'batch_gather does not allow indices with unknown shape.'):
      ragged.batch_gather(params, indices)

    with self.assertRaisesRegexp(
        ValueError, 'batch_gather does not allow indices with unknown shape.'):
      ragged.batch_gather(params, ragged_indices)

  @parameterized.parameters([
      dict(
          params=ragged.constant_value([['a'], ['b'], ['c']]),
          indices=ragged.constant_value([[0], [0]]),
          message='Dimensions 3 and 2 are not compatible'),
      dict(
          params=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
          indices=ragged.constant_value([[[0, 0], [0, 0, 0]], [[0]]]),
          message='batch shape from indices does not match params shape'),
      dict(  # rank mismatch
          params=ragged.constant_value([[[0, 0], [0, 0, 0]], [[0]]]),
          indices=ragged.constant_value([[[0, 0]], [[0, 0, 0]], [[0]]]),
          error=(ValueError, errors.InvalidArgumentError)),
      dict(
          params=ragged.constant_value([[[0, 0], [0, 0, 0]], [[0]], [[0]]]),
          indices=ragged.constant_value([[[0, 0]], [[0, 0, 0]], [[0]]]),
          error=errors.InvalidArgumentError,
          message='.*Condition x == y did not hold.*'),
      dict(
          params=ragged.constant_value(['a', 'b', 'c']),
          indices=ragged.constant_value([[0], [0]]),
          message='batch shape from indices does not match params shape'),
      dict(
          params=ragged.constant_value([['a']]),
          indices=0,
          message='indices.rank must be at least 1.'),
      dict(
          params=ragged.constant_value([['a']]),
          indices=[[[0]]],
          message='batch shape from indices does not match params shape'),
  ])
  def testRaggedBatchGatherStaticError(self,
                                       params,
                                       indices,
                                       message=None,
                                       error=ValueError):
    with self.assertRaisesRegexp(error, message):
      ragged.batch_gather(params, indices)


if __name__ == '__main__':
  googletest.main()
