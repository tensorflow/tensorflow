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
"""Tests for ragged_batch_gather_ops.batch_gather."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_batch_gather_ops
from tensorflow.python.ops.ragged import ragged_batch_gather_with_default_op
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedBatchGatherOpTest(test_util.TensorFlowTestCase,
                              parameterized.TestCase):

  @parameterized.parameters([
      #=========================================================================
      # Docstring Example
      #=========================================================================
      dict(
          descr='Docstring example',
          params=ragged_factory_ops.constant_value([['a', 'b', 'c'], ['d'], [],
                                                    ['e']]),
          indices=ragged_factory_ops.constant_value([[1, 2, 0], [], [], [0,
                                                                         0]]),
          expected=ragged_factory_ops.constant_value([[b'b', b'c', b'a'], [],
                                                      [], [b'e', b'e']])),
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
          params=ragged_factory_ops.constant_value([['a', 'b'], [], ['c'],
                                                    ['d', 'e']]),
          indices=[3, 2],
          expected=ragged_factory_ops.constant_value([[b'd', b'e'], [b'c']])),
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
          params=ragged_factory_ops.constant_value([['a', 'b', 'c'], ['d', 'e'],
                                                    ['g']]),
          indices=[[2, 0], [0, 1], [0, 0]],
          expected=[[b'c', b'a'], [b'd', b'e'], [b'g', b'g']]),
      dict(
          descr='params: [B1, P1], indices: [B1, (I)], result: [B1, (I)]',
          params=[['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']],
          indices=ragged_factory_ops.constant_value([[2, 0, 2], [0], [1]]),
          expected=ragged_factory_ops.constant_value([[b'c', b'a', b'c'],
                                                      [b'd'], [b'h']])),
      dict(
          descr=('params: [B1, (P1), (P2), P3], indices: [B1, I], '
                 'result: [B1, I, (P2), P3]'),
          params=ragged_factory_ops.constant_value(
              [[[['a']], [['b'], ['c']]], [[['d'], ['e']], [['f']]], [[['g']]]],
              ragged_rank=2),
          indices=[[1, 0], [0, 1], [0, 0]],
          expected=ragged_factory_ops.constant_value(
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
          params=ragged_factory_ops.constant_value(
              [[['a', 'b', 'c'], ['d', 'e', 'f']], [['g', 'h', 'i']]],
              ragged_rank=1),
          indices=ragged_factory_ops.constant_value(
              [[[2, 0], [0, 1]], [[1, 0]]], ragged_rank=1),
          expected=ragged_factory_ops.constant_value(
              [[[b'c', b'a'], [b'd', b'e']], [[b'h', b'g']]], ragged_rank=1)),
      dict(
          descr=('params: [B1, (B2), (P1)], indices: [B1, (B2), I], '
                 'result: [B1, (B2), I]'),
          params=ragged_factory_ops.constant_value(
              [[['a', 'b', 'c'], ['d']], [['e', 'f']]], ragged_rank=2),
          indices=ragged_factory_ops.constant_value(
              [[[2, 0], [0, 0]], [[1, 0]]], ragged_rank=1),
          expected=ragged_factory_ops.constant_value(
              [[[b'c', b'a'], [b'd', b'd']], [[b'f', b'e']]], ragged_rank=1)),
      dict(
          descr=('params: [B1, (B2), P1], indices: [B1, (B2), (I)], '
                 'result: [B1, (B2), (I)]'),
          params=ragged_factory_ops.constant_value(
              [[['a', 'b', 'c'], ['d', 'e', 'f']], [['g', 'h', 'i']]],
              ragged_rank=1),
          indices=ragged_factory_ops.constant_value(
              [[[2, 1, 0], [0]], [[1, 1]]], ragged_rank=2),
          expected=ragged_factory_ops.constant_value(
              [[[b'c', b'b', b'a'], [b'd']], [[b'h', b'h']]], ragged_rank=2)),
      #=========================================================================
      # 3 Batch Dimensions
      #=========================================================================
      dict(
          descr=(
              'params: [B1, (B2), (B3), (P1)], indices: [B1, (B2), (B3), I], '
              'result: [B1, (B2), (B3), I]'),
          params=ragged_factory_ops.constant_value(
              [[[['a', 'b', 'c'], ['d']], [['e', 'f']]]], ragged_rank=3),
          indices=ragged_factory_ops.constant_value(
              [[[[2, 0], [0, 0]], [[1, 0]]]], ragged_rank=2),
          expected=ragged_factory_ops.constant_value(
              [[[[b'c', b'a'], [b'd', b'd']], [[b'f', b'e']]]], ragged_rank=2)),
  ])
  def testRaggedBatchGather(self, descr, params, indices, expected):
    result = ragged_batch_gather_ops.batch_gather(params, indices)
    self.assertAllEqual(result, expected)

  @parameterized.parameters([
      # Docstring example:
      dict(
          descr='Docstring example',
          params=[['a', 'b', 'c'], ['d'], [], ['e']],
          indices=[[1, 2, -1], [], [], [0, 10]],
          expected=[['b', 'c', 'FOO'], [], [], ['e', 'FOO']],
          default_value='FOO',
      ),
      # Dimensions:
      # indices: [4]
      # params: [2, (d1), (d2)]
      dict(
          descr='params: [2, (d1), (d2), indices: [4]',
          indices=[1, 100, 0, -1],
          params=[[['The', 'deal', 'came', 'about', '18', 'months', 'after',
                    'Yahoo', '!', 'rejected', 'a', '47.5', '-', 'billion', '-',
                    'dollar', 'takeover', 'offer', 'from', 'Microsoft', '.'],
                   ['Trumpty', 'Dumpty', 'sat', 'on', 'a', 'wall']],
                  [["It's", 'always', 'darkest', 'before', 'the', 'dawn']]],
          expected=[[["It's", 'always', 'darkest', 'before', 'the', 'dawn']],
                    [['$NONE^']],
                    [['The', 'deal', 'came', 'about', '18', 'months', 'after',
                      'Yahoo', '!', 'rejected', 'a', '47.5', '-', 'billion',
                      '-', 'dollar', 'takeover', 'offer', 'from', 'Microsoft',
                      '.'],
                     ['Trumpty', 'Dumpty', 'sat', 'on', 'a', 'wall']],
                    [['$NONE^']]],
      ),
      # Dimensions:
      # params: [1, (d1)]
      # indices: [3]
      dict(
          descr='params: rank 2, indices: rank 1',
          params=[
              ['Bruce', 'Wayne'],
          ],
          indices=[-1, 0, 1000],
          expected=[['$NONE^'], ['Bruce', 'Wayne'], ['$NONE^']]
      ),
      # Dimensions:
      # params: [1, (d1)]
      # indices: [1, (d2)]
      dict(
          descr='Test underbound indices of shape [1, (d2)]',
          params=[
              ['The', 'deal', 'came', 'about', '18', 'months', 'after', 'Yahoo',
               '!', 'rejected', 'a', '47.5', '-', 'billion', '-', 'dollar',
               'takeover', 'offer', 'from', 'Microsoft', '.'],
          ],
          indices=[[8, -1]],
          expected=[['!', '$NONE^']],
      ),
      dict(
          descr='Test underbound indices of shape [2, (d2)]',
          params=[
              ['The', 'deal', 'came', 'about', '18', 'months', 'after', 'Yahoo',
               '!', 'rejected', 'a', '47.5', '-', 'billion', '-', 'dollar',
               'takeover', 'offer', 'from', 'Microsoft', '.'],
              ['Who', 'let', 'the', 'dogs', 'out', '?'],
          ],
          indices=[[8, -1], [1, 100]],
          expected=[['!', '$NONE^'], ['let', '$NONE^']],
      ),
      # Dimensions:
      # params: [2, (d1)]
      # indices: [2, (d2)]
      dict(
          descr='Test underbound indices of rank 2',
          params=[
              ['The', 'deal', 'came', 'about', '18', 'months', 'after', 'Yahoo',
               '!', 'rejected', 'a', '47.5', '-', 'billion', '-', 'dollar',
               'takeover', 'offer', 'from', 'Microsoft', '.'],
              ['He', 'left', 'us', '.', 'Little', 'boys', 'crowded', 'together',
               'on', 'long', 'wooden', 'benches', ',', 'and', 'in', 'the',
               'center', 'of', 'the', 'room', 'sat', 'the', 'teacher', '.',
               'His', 'black', 'beard', 'dripped', 'down', 'over', 'the',
               'front', 'of', 'his', 'coat', '.', 'One', 'white', 'hand',
               'poised', 'a', 'stick', 'above', 'his', 'desk', '.', 'He',
               'turned', 'his', 'surly', ',', 'half', '-', 'closed', 'eyes',
               'toward', 'us', ',', 'stared', 'for', 'a', 'second', ',', 'then',
               'shouted', 'in', 'Yiddish', ',', '``', 'One', ',', 'two', ',',
               'three', "''", '!', '!', 'Rapping', 'the', 'stick', 'against',
               'the', 'desk', '.', 'The', 'little', 'boys', 'shrilled', 'out',
               'a', 'Yiddish', 'translation', 'or', 'interpretation', 'of',
               'the', 'Five', 'Books', 'of', 'Moses', ',', 'which', 'they',
               'had', 'previously', 'chanted', 'in', 'Hebrew', '.']],
          indices=[[8, -1], [3, 23, 35, 45, 75, 83, -121]],
          expected=[['!', '$NONE^'], ['.', '.', '.', '.', '!', '.', '$NONE^']],
      ),
      dict(
          descr='Test overbound indices of rank 2',
          params=[
              ['The', 'deal', 'came', 'about', '18', 'months', 'after', 'Yahoo',
               '!', 'rejected', 'a', '47.5', '-', 'billion', '-', 'dollar',
               'takeover', 'offer', 'from', 'Microsoft', '.'],
              ['He', 'left', 'us', '.', 'Little', 'boys', 'crowded', 'together',
               'on', 'long', 'wooden', 'benches', ',', 'and', 'in', 'the',
               'center', 'of', 'the', 'room', 'sat', 'the', 'teacher', '.',
               'His', 'black', 'beard', 'dripped', 'down', 'over', 'the',
               'front', 'of', 'his', 'coat', '.', 'One', 'white', 'hand',
               'poised', 'a', 'stick', 'above', 'his', 'desk', '.', 'He',
               'turned', 'his', 'surly', ',', 'half', '-', 'closed', 'eyes',
               'toward', 'us', ',', 'stared', 'for', 'a', 'second', ',', 'then',
               'shouted', 'in', 'Yiddish', ',', '``', 'One', ',', 'two', ',',
               'three', "''", '!', '!', 'Rapping', 'the', 'stick', 'against',
               'the', 'desk', '.', 'The', 'little', 'boys', 'shrilled', 'out',
               'a', 'Yiddish', 'translation', 'or', 'interpretation', 'of',
               'the', 'Five', 'Books', 'of', 'Moses', ',', 'which', 'they',
               'had', 'previously', 'chanted', 'in', 'Hebrew', '.']],
          indices=[[8, 8823], [3, 23, 35, 45, 75, 83, 1234]],
          expected=[['!', '$NONE^'], ['.', '.', '.', '.', '!', '.', '$NONE^']],
      ),
      # Dimensions:
      # params: [2, (d1), 2]
      # indices: [2, (d2)]
      dict(
          descr='params: rank 3, indices: rank 2',
          params=[
              [['The', 'deal'], ['takeover', 'offer'], ['from', 'Microsoft']],
              [['Who', 'let'], ['the', 'dogs'], ['out', '?']],
          ],
          ragged_rank=1,
          indices=[[1, -1, 2, 30], [1, 100]],
          indices_ragged_rank=1,
          expected=[[['takeover', 'offer'],
                     ['$NONE^', '$NONE^'],
                     ['from', 'Microsoft'],
                     ['$NONE^', '$NONE^']],
                    [['the', 'dogs'],
                     ['$NONE^', '$NONE^']]],
          expected_ragged_rank=1,
          default_value=['$NONE^', '$NONE^'],
      ),
      # Dimensions:
      # params: [2, (d1), (d2)]
      # indices: [2, (d3)]
      dict(
          descr='params: [2, (d1), (d2)], indices: [2, (d3)]',
          params=[
              [['The', 'deal', 'came', 'about', '18', 'months', 'after',
                'Yahoo', '!', 'rejected', 'a', '47.5', '-', 'billion', '-',
                'dollar', 'takeover', 'offer', 'from', 'Microsoft', '.'],
               ['Trumpty', 'Dumpty', 'sat', 'on', 'a', 'wall'],
              ],
              [['It\'s', 'always', 'darkest', 'before', 'the', 'dawn']]
          ],
          indices=[[1, 100], [0, -1]],
          expected=[[['Trumpty', 'Dumpty', 'sat', 'on', 'a', 'wall'],
                     ['$NONE^']],
                    [["It's", 'always', 'darkest', 'before', 'the', 'dawn'],
                     ['$NONE^']]]
      ),
      # Dimensions:
      # params: [2, (d1), (d2)]
      # indices: [2, (d1), (d3)]
      dict(
          descr='Test overbound indices of rank 3',
          params=[
              [['The', 'deal', 'came', 'about', '18', 'months', 'after',
                'Yahoo', '!', 'rejected', 'a', '47.5', '-', 'billion', '-',
                'dollar', 'takeover', 'offer', 'from', 'Microsoft', '.'],
               ['Foo', 'bar', 'mar']],
              [['He', 'left', 'us', '.', 'Little', 'boys', 'crowded',
                'together', 'on', 'long', 'wooden', 'benches', ',', 'and', 'in',
                'the', 'center', 'of', 'the', 'room', 'sat', 'the', 'teacher',
                '.', 'His', 'black', 'beard', 'dripped', 'down', 'over', 'the',
                'front', 'of', 'his', 'coat', '.', 'One', 'white', 'hand',
                'poised', 'a', 'stick', 'above', 'his', 'desk', '.', 'He',
                'turned', 'his', 'surly', ',', 'half', '-', 'closed', 'eyes',
                'toward', 'us', ',', 'stared', 'for', 'a', 'second', ',',
                'then', 'shouted', 'in', 'Yiddish', ',', '``', 'One', ',',
                'two', ',',
                'three', "''", '!', '!', 'Rapping', 'the', 'stick', 'against',
                'the', 'desk', '.', 'The', 'little', 'boys', 'shrilled', 'out',
                'a', 'Yiddish', 'translation', 'or', 'interpretation', 'of',
                'the', 'Five', 'Books', 'of', 'Moses', ',', 'which', 'they',
                'had', 'previously', 'chanted', 'in', 'Hebrew', '.'],
               ['I', 'too', 'was', 'hustled', 'scammed', 'bamboozled', 'hood',
                'winked', 'lead', 'astray']]
          ],
          indices=[[[8, 8823], [0, 100]], [[3, 23, 35, 45, 75, 83, 1234], [5]]],
          expected=[[['!', '$NONE^'], ['Foo', '$NONE^']],
                    [['.', '.', '.', '.', '!', '.', '$NONE^'],
                     ['bamboozled']]],
      ),
      # params.shape = [2, (d1), 8]
      # indices.shape = [2, (d1), 3]
      dict(
          descr='params = [2, (2, 1), 8], indices = [2, (2, 1), 3]',
          params=[[['h'] * 8, ['w'] * 8], [['b'] * 8]],
          ragged_rank=1,
          indices=[[[0, 100, 1], [0, 1, 0]], [[1, 0, 0]]],
          indices_ragged_rank=1,
          expected=[[['h', '$NONE^', 'h'], ['w', 'w', 'w']], [['b', 'b', 'b']]],
          expected_ragged_rank=1,
      ),
  ])
  def testRaggedBatchGatherWithDefault(
      self, descr, params, indices, expected, indices_ragged_rank=None,
      expected_ragged_rank=None, ragged_rank=None, default_value='$NONE^'):
    params = ragged_factory_ops.constant(params, ragged_rank=ragged_rank)
    indices = ragged_factory_ops.constant(
        indices, ragged_rank=indices_ragged_rank or ragged_rank)
    expected = ragged_factory_ops.constant(
        expected, ragged_rank=expected_ragged_rank or ragged_rank)
    result = ragged_batch_gather_with_default_op.batch_gather_with_default(
        params, indices, default_value)
    self.assertAllEqual(result, expected)

  @parameterized.parameters([
      # Dimensions:
      #  params: dims [2, 5], indices: [2, 2]
      dict(
          descr='params: dims [2, 5], indices: [2, 2]',
          params=[
              ['The', 'deal', 'came', 'about', '18'],
              ['He', 'left', 'us', '.', 'Little']],
          indices=[[0, -1], [3, 121]],
          expected=[['The', '$NONE^'], ['.', '$NONE^']],
          default_value='$NONE^',
      ),
      # Dimensions:
      #  params: dims [2, 2, 5], indices: [2, 2]
      dict(
          descr='params: dims [2, 2, 5], indices: [2, 2]',
          params=[
              [['The', 'deal', 'came', 'about', '18'],
               ['The', 'deal', 'came', 'about', '19'],
              ],
              [['He', 'left', 'us', '.', 'Little'],
               ['The', 'deal', 'came', 'about', '20'],
              ]
          ],
          indices=[[0, -1], [0, 121]],
          expected=[[['The', 'deal', 'came', 'about', '18'],
                     ['$NONE^', '$NONE^', '$NONE^', '$NONE^', '$NONE^']],
                    [['He', 'left', 'us', '.', 'Little'],
                     ['$NONE^', '$NONE^', '$NONE^', '$NONE^', '$NONE^']]],
          default_value='$NONE^',
      ),
      # Test default_value with shape [5]
      dict(
          descr='params: dims [2, 2, 5], indices: [2, 2]',
          params=[
              [['The', 'deal', 'came', 'about', '18'],
               ['The', 'deal', 'came', 'about', '19'],
              ],
              [['He', 'left', 'us', '.', 'Little'],
               ['The', 'deal', 'came', 'about', '20'],
              ]
          ],
          indices=[[0, -1], [0, 121]],
          expected=[[['The', 'deal', 'came', 'about', '18'],
                     [':FOO:', ':FOO:', ':FOO:', ':FOO:', ':FOO:']],
                    [['He', 'left', 'us', '.', 'Little'],
                     [':FOO:', ':FOO:', ':FOO:', ':FOO:', ':FOO:']]],
          default_value=[':FOO:', ':FOO:', ':FOO:', ':FOO:', ':FOO:'],
      ),
  ])
  def testRaggedBatchGatherWithDefaultOnTensors(
      self, descr, params, indices, expected, default_value):
    params = constant_op.constant(params)
    indices = constant_op.constant(indices)
    expected = constant_op.constant(expected)
    result = ragged_batch_gather_with_default_op.batch_gather_with_default(
        params, indices, default_value)
    self.assertAllEqual(expected, result)

  @parameterized.parameters([
      dict(
          params=[['The', 'deal', 'came', 'about', '18', 'months', 'after',
                   'Yahoo', '!', 'rejected', 'a', '47.5', '-', 'billion', '-',
                   'dollar', 'takeover', 'offer', 'from', 'Microsoft', '.']],
          indices=[[[8, -1]]],
          # Exception here because different errors are thrown in eager vs
          # graph mode.
          error=Exception,
          default_value='$NONE^',
      ),
  ])
  def testRankMismatch(
      self, params, indices, default_value, error):
    params = ragged_factory_ops.constant(params)
    indices = ragged_factory_ops.constant(indices)
    with self.assertRaises(error):
      _ = ragged_batch_gather_with_default_op.batch_gather_with_default(
          params, indices, default_value)

  @parameterized.parameters([
      # Dimensions:
      # params: [2, (d1), 2]
      # indices: [2, (d2)]
      # default_value: []
      dict(
          descr='params: rank 3, indices: rank 2, default: rank = [], but'
          ' should be [2]',
          params=[
              [['The', 'deal'], ['takeover', 'offer'], ['from', 'Microsoft']],
              [['Who', 'let'], ['the', 'dogs'], ['out', '?']],
          ],
          ragged_rank=1,
          indices=[[1, -1, 2, 30], [1, 100]],
          indices_ragged_rank=1,
          default_value='$NONE^',
          error=Exception,
      )
  ])
  def testInvalidDefaultValueRank(
      self, descr, params, indices, default_value, error, ragged_rank=None,
      indices_ragged_rank=None):
    params = ragged_factory_ops.constant(params, ragged_rank=ragged_rank)
    indices = ragged_factory_ops.constant(
        indices, ragged_rank=indices_ragged_rank)
    with self.assertRaises(error):
      _ = ragged_batch_gather_with_default_op.batch_gather_with_default(
          params, indices, default_value)

  def testRaggedBatchGatherUnknownRankError(self):
    if context.executing_eagerly():
      return
    params = [['a', 'b'], ['c', 'd']]
    indices = array_ops.placeholder(dtypes.int32, shape=None)
    ragged_indices = ragged_tensor.RaggedTensor.from_row_splits(
        indices, [0, 2, 4])

    with self.assertRaisesRegex(
        ValueError, r'batch_dims may only be negative '
        r'if rank\(indices\) is statically known.'):
      ragged_batch_gather_ops.batch_gather(params, indices)

    with self.assertRaisesRegex(
        ValueError, r'batch_dims may only be negative '
        r'if rank\(indices\) is statically known.'):
      ragged_batch_gather_ops.batch_gather(params, ragged_indices)

  @parameterized.parameters(
      [
          dict(
              params=ragged_factory_ops.constant_value([['a'], ['b'], ['c']]),
              indices=ragged_factory_ops.constant_value([[0], [0]]),
              message=(r'batch shape from indices .* does not match params')),
          dict(
              params=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
              indices=ragged_factory_ops.constant_value([[[0, 0], [0, 0, 0]],
                                                         [[0]]]),
              message='batch shape from indices does not match params shape'),
          dict(  # rank mismatch
              params=ragged_factory_ops.constant_value([[[0, 0], [0, 0, 0]],
                                                        [[0]]]),
              indices=ragged_factory_ops.constant_value([[[0, 0]], [[0, 0, 0]],
                                                         [[0]]]),
              error=(ValueError, errors.InvalidArgumentError)),
          dict(
              params=ragged_factory_ops.constant_value([[[0, 0], [0, 0, 0]],
                                                        [[0]], [[0]]]),
              indices=ragged_factory_ops.constant_value([[[0, 0]], [[0, 0, 0]],
                                                         [[0]]]),
              error=(ValueError, errors.InvalidArgumentError),
              message=(r'batch shape from indices .* does not match '
                       r'params shape|dimension size mismatch')),
          dict(
              params=ragged_factory_ops.constant_value(['a', 'b', 'c']),
              indices=ragged_factory_ops.constant_value([[0], [0]]),
              message=r'batch_dims must be less than rank\(params\)'),
          dict(
              params=ragged_factory_ops.constant_value([['a']]),
              indices=0,
              message='batch_dims=-1 out of bounds: expected 0<=batch_dims<0'),
          dict(
              params=ragged_factory_ops.constant_value([['a']]),
              indices=[[[0]]],
              message=r'batch_dims must be less than rank\(params\)'),
      ])
  def testRaggedBatchGatherStaticError(self,
                                       params,
                                       indices,
                                       message=None,
                                       error=ValueError):
    with self.assertRaisesRegex(error, message):
      ragged_batch_gather_ops.batch_gather(params, indices)


if __name__ == '__main__':
  googletest.main()
