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
"""Tests for ragged_gather_ops.gather_nd."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedGatherNdOpTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):

  DOCSTRING_PARAMS = [[['000', '001'], ['010']],
                      [['100'], ['110', '111', '112'], ['120']],
                      [[], ['210']]]  # pyformat: disable

  @parameterized.parameters([
      #=========================================================================
      # Docstring Examples
      #=========================================================================
      dict(
          descr='Docstring example 1',
          params=ragged_factory_ops.constant_value(DOCSTRING_PARAMS),
          indices=[[2], [0]],
          expected=ragged_factory_ops.constant_value(
              [[[], [b'210']], [[b'000', b'001'], [b'010']]])),
      dict(
          descr='Docstring example 2',
          params=ragged_factory_ops.constant_value(DOCSTRING_PARAMS),
          indices=[[2, 1], [0, 0]],
          expected=ragged_factory_ops.constant_value(
              [[b'210'], [b'000', b'001']])),
      dict(
          descr='Docstring example 3',
          params=ragged_factory_ops.constant_value(DOCSTRING_PARAMS),
          indices=[[0, 0, 1], [1, 1, 2]],
          expected=[b'001', b'112']),
      #=========================================================================
      # Indices with 0 values (selects the entire params)
      #=========================================================================
      dict(
          descr='params: [B1, (B2)], indices: [0], result: [B1, (B2)]',
          params=ragged_factory_ops.constant_value([['a', 'b', 'c'], ['d']]),
          indices=np.zeros([0], dtype=np.int32),
          expected=ragged_factory_ops.constant_value(
              [[b'a', b'b', b'c'], [b'd']])),
      dict(
          descr='params: [B1, (B2)], indices: [A1, 0], result: [A1, B1, (B2)]',
          params=ragged_factory_ops.constant_value([['a', 'b', 'c'], ['d']]),
          indices=np.zeros([3, 0], dtype=np.int32),
          expected=ragged_factory_ops.constant_value(
              [[[b'a', b'b', b'c'], [b'd']],
               [[b'a', b'b', b'c'], [b'd']],
               [[b'a', b'b', b'c'], [b'd']]])),
      dict(
          descr=('params: [B1, (B2)], indices: [A1, A2, 0], '
                 'result: [A1, A2, B1, (B2)]'),
          params=ragged_factory_ops.constant_value([['a', 'b', 'c'], ['d']]),
          indices=np.zeros([1, 3, 0], dtype=np.int32),
          expected=ragged_factory_ops.constant_value(
              [[[[b'a', b'b', b'c'], [b'd']],
                [[b'a', b'b', b'c'], [b'd']],
                [[b'a', b'b', b'c'], [b'd']]]])),
      dict(
          descr='params: [B1], indices: [A1, (A2), 0], result: [A1, (A2), B1]',
          params=['a'],
          indices=ragged_factory_ops.constant_value(
              [[[], []], [[]]],
              ragged_rank=1,
              dtype=np.int32),
          expected=ragged_factory_ops.constant_value(
              [[[b'a'], [b'a']], [[b'a']]],
              ragged_rank=1)),
      #=========================================================================
      # Indices with 1 value (selects row from params)
      #=========================================================================
      dict(
          descr='params: [B1, (B2)], indices: [A1, 1], result: [A1, (B2)]',
          params=ragged_factory_ops.constant_value([['a', 'b', 'c'], ['d']]),
          indices=[[1], [0]],
          expected=ragged_factory_ops.constant_value(
              [[b'd'], [b'a', b'b', b'c']])),
      dict(
          descr=('params: [B1, (B2), (B3)], indices: [A1, 1], '
                 'result: [A1, (B2), (B3)]'),
          params=ragged_factory_ops.constant_value(
              [[['a', 'b', 'c'], ['d']], [['e', 'f']]]),
          indices=[[1], [1]],
          expected=ragged_factory_ops.constant_value(
              [[[b'e', b'f']], [[b'e', b'f']]])),
      dict(
          descr=('params: [B1, B2, B3], indices: [A1, (A2), 1], '
                 'result: [A1, (A2), B2, B3]'),
          params=[[['a']], [['b']]],
          indices=ragged_factory_ops.constant_value([[[0]]], ragged_rank=1),
          expected=ragged_factory_ops.constant_value(
              [[[[b'a']]]], ragged_rank=1)),
      #=========================================================================
      # Indices with 2 values (selects row & col from params)
      #=========================================================================
      dict(
          descr='params: [B1, (B2)], indices: [A1, 2], result: [A1]',
          params=ragged_factory_ops.constant_value([['a', 'b', 'c'], ['d']]),
          indices=[[1, 0], [0, 0], [0, 2]],
          expected=ragged_factory_ops.constant_value([b'd', b'a', b'c'])),
      dict(
          descr=('params: [B1, (B2), (B3)], indices: [A1, 2], '
                 'result: [A1, (B3)]'),
          params=ragged_factory_ops.constant_value(
              [[['a', 'b', 'c'], ['d']], [['e', 'f']]]),
          indices=[[1, 0], [0, 1], [0, 0]],
          expected=ragged_factory_ops.constant_value(
              [[b'e', b'f'], [b'd'], [b'a', b'b', b'c']])),
      dict(
          descr=('params: [B1, (B2), (B3)], indices: [A1, A2, 2], '
                 'result: [A1, (A2), (B3)]'),
          params=ragged_factory_ops.constant_value(
              [[['a', 'b', 'c'], ['d']], [['e', 'f']]]),
          indices=[[[1, 0], [0, 1], [0, 0]]],
          expected=ragged_factory_ops.constant_value(
              [[[b'e', b'f'], [b'd'], [b'a', b'b', b'c']]])),
      dict(
          descr=('params: [B1, (B2), B3], indices: [A1, A2, 2], '
                 'result: [A1, A2, B3]'),
          params=ragged_factory_ops.constant_value(
              [[['a', 'b'], ['c', 'd']],
               [['e', 'f']]],
              ragged_rank=1),
          indices=[[[1, 0], [0, 1], [0, 0]]],
          expected=[[[b'e', b'f'], [b'c', b'd'], [b'a', b'b']]]),
      dict(
          descr=('params: [B1, (B2), B3], indices: [A1, A2, A3, 2], '
                 'result: [A1, A2, A3, B3]'),
          params=ragged_factory_ops.constant_value(
              [[['a', 'b'], ['c', 'd']],
               [['e', 'f']]],
              ragged_rank=1),
          indices=[[[[1, 0], [0, 1], [0, 0]]]],
          expected=[[[[b'e', b'f'], [b'c', b'd'], [b'a', b'b']]]]),
      dict(
          descr=('params: [B1, (B2), (B3)], indices: [A1, (A2), 2], '
                 'result: [A1, (A2), (B3)]'),
          params=ragged_factory_ops.constant_value(
              [[['a', 'b', 'c'], ['d']], [['e', 'f']]]),
          indices=ragged_factory_ops.constant_value(
              [[[1, 0], [0, 1]], [[0, 0]]],
              ragged_rank=1),
          expected=ragged_factory_ops.constant_value(
              [[[b'e', b'f'], [b'd']], [[b'a', b'b', b'c']]])),
      #=========================================================================
      # Indices with 3 values
      #=========================================================================
      dict(
          descr=('params: [B1, (B2), (B3)], indices: [A1, 3], '
                 'result: [A1]'),
          params=ragged_factory_ops.constant_value(
              [[['a', 'b', 'c'], ['d']], [['e', 'f']]]),
          indices=[[1, 0, 1], [0, 0, 0], [0, 1, 0]],
          expected=[b'f', b'a', b'd']),
      dict(
          descr=('params: [B1, (B2), B3], indices: [A1, 3], '
                 'result: [A1]'),
          params=ragged_factory_ops.constant_value(
              [[['a', 'b'], ['c', 'd']], [['e', 'f']]],
              ragged_rank=1),
          indices=[[1, 0, 1], [0, 0, 0], [0, 1, 1]],
          expected=[b'f', b'a', b'd']),
      dict(
          descr=('params: [B1, (B2), (B3), B4], indices: [A1, 3], '
                 'result: [A1, B4]'),
          params=ragged_factory_ops.constant_value(
              [[[['a', 'b'], ['c', 'd']], [['e', 'f']]]],
              ragged_rank=2),
          indices=[[0, 0, 1], [0, 0, 0], [0, 1, 0]],
          expected=[[b'c', b'd'], [b'a', b'b'], [b'e', b'f']]),
      dict(
          descr=('Pass through bad_indices_policy for non ragged params+indices'),
          params=[1, 3, 5, 7],
          indices=[[3], [999], [1], [0]],
          expected=[7, 0, 3, 1],
          bad_indices_policy='IGNORE'),
  ])  # pyformat: disable
  def testRaggedGatherNd(
      self, descr, params, indices, expected, bad_indices_policy=''
  ):
    result = ragged_gather_ops.gather_nd(
        params, indices, bad_indices_policy=bad_indices_policy
    )
    self.assertAllEqual(result, expected)

  def testRaggedGatherNdUnknownRankError(self):
    if context.executing_eagerly():
      return
    params = ragged_factory_ops.constant([['a', 'b'], ['c', 'd']])
    indices1 = array_ops.placeholder(dtypes.int32, shape=None)
    indices2 = array_ops.placeholder(dtypes.int32, shape=[None])

    with self.assertRaisesRegex(ValueError,
                                'indices.rank be statically known.'):
      ragged_gather_ops.gather_nd(params, indices1)
    with self.assertRaisesRegex(
        ValueError, r'indices.shape\[-1\] must be statically known.'):
      ragged_gather_ops.gather_nd(params, indices2)

  @parameterized.parameters([
      dict(
          params=['a'],
          indices=0,
          error=(ValueError, errors.InvalidArgumentError),
      ),
      dict(
          params=ragged_factory_ops.constant_value([['a']]),
          indices=0,
          message='indices.rank must be at least 1.',
      ),
      dict(
          params=['a', 'b', 'c'],
          indices=ragged_factory_ops.constant_value([[0]]),
          message='The innermost dimension of indices may not be ragged',
      ),
      dict(
          params=ragged_factory_ops.constant_value([['a', 'b', 'c'], ['d']]),
          indices=np.zeros([1, 3, 0], dtype=np.int32),
          bad_indices_policy='IGNORE',
          message=(
              'non-default bad_indices_policy not supported for ragged gather'
          ),
      ),
  ])
  def testRaggedGatherNdStaticError(
      self,
      params,
      indices,
      bad_indices_policy='',
      message=None,
      error=ValueError,
  ):
    with self.assertRaisesRegex(error, message):
      ragged_gather_ops.gather_nd(
          params, indices, bad_indices_policy=bad_indices_policy
      )


if __name__ == '__main__':
  googletest.main()
