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
"""Tests for ragged.tile."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest


class RaggedTileOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.parameters([
      #=========================================================================
      # Docstring Example
      #=========================================================================
      dict(
          descr='docstring example: ragged_rank=1, repeat axes 0 and 1',
          rt_input=[[1, 2], [3]],
          multiples=[3, 2],
          expected=[
              [1, 2, 1, 2], [3, 3], [1, 2, 1, 2], [3, 3], [1, 2, 1, 2], [3, 3]],
      ),

      #=========================================================================
      # rank=3, ragged_rank=2
      #=========================================================================
      dict(
          descr='rank=3, ragged_rank=2, repeat axis 0',
          rt_input=[[[1, 2], [3]], [], [[4]]],
          multiples=[2, 1, 1],
          expected=[[[1, 2], [3]], [], [[4]],
                    [[1, 2], [3]], [], [[4]]]),
      dict(
          descr='rank=3, ragged_rank=2, repeat axis 1',
          rt_input=[[[1, 2], [3]], [], [[4]]],
          multiples=[1, 2, 1],
          expected=[[[1, 2], [3], [1, 2], [3]], [], [[4], [4]]]),
      dict(
          descr='rank=3, ragged_rank=2, repeat axis 2',
          rt_input=[[[1, 2], [3]], [], [[4]]],
          multiples=[1, 1, 2],
          expected=[[[1, 2, 1, 2], [3, 3]], [], [[4, 4]]]),
      dict(
          descr='rank=3, ragged_rank=2, repeat axes 0 and 1',
          rt_input=[[[1, 2], [3]], [], [[4]]],
          multiples=[2, 2, 1],
          expected=[[[1, 2], [3], [1, 2], [3]], [], [[4], [4]],
                    [[1, 2], [3], [1, 2], [3]], [], [[4], [4]]]),
      dict(
          descr='rank=3, ragged_rank=2, repeat axes 0 and 2',
          rt_input=[[[1, 2], [3]], [], [[4]]],
          multiples=[2, 1, 2],
          expected=[[[1, 2, 1, 2], [3, 3]], [], [[4, 4]],
                    [[1, 2, 1, 2], [3, 3]], [], [[4, 4]]]),
      dict(
          descr='rank=3, ragged_rank=2, repeat axes 1 and 2',
          rt_input=[[[1, 2], [3]], [], [[4]]],
          multiples=[1, 2, 2],
          expected=[[[1, 2, 1, 2], [3, 3], [1, 2, 1, 2], [3, 3]],
                    [], [[4, 4], [4, 4]]]),
      dict(
          descr='rank=3, ragged_rank=2, repeat all axes',
          rt_input=[[['a', 'b'], ['c']], [], [['d']]],
          multiples=[4, 3, 2],
          expected=[[[b'a', b'b']*2, [b'c']*2]*3, []*3, [[b'd']*2]*3]*4),
      #=========================================================================
      # rank=3, ragged_rank=1
      #=========================================================================
      dict(
          descr='rank=3, ragged_rank=1, repeat axis 0',
          ragged_rank=1,
          rt_input=[[[1, 2], [3, 4]], [], [[5, 6]]],
          multiples=[2, 1, 1],
          expected=[[[1, 2], [3, 4]], [], [[5, 6]],
                    [[1, 2], [3, 4]], [], [[5, 6]]]),
      dict(
          descr='rank=3, ragged_rank=1, repeat axis 1',
          ragged_rank=1,
          rt_input=[[[1, 2], [3, 4]], [], [[5, 6]]],
          multiples=[1, 2, 1],
          expected=[[[1, 2], [3, 4], [1, 2], [3, 4]], [], [[5, 6], [5, 6]]]),
      dict(
          descr='rank=3, ragged_rank=1, repeat axis 2',
          ragged_rank=1,
          rt_input=[[[1, 2], [3, 4]], [], [[5, 6]]],
          multiples=[1, 1, 2],
          expected=[[[1, 2, 1, 2], [3, 4, 3, 4]], [], [[5, 6, 5, 6]]]),
      #=========================================================================
      # rank=4, ragged_rank=3
      #=========================================================================
      dict(
          descr='rank=4, ragged_rank=3, repeat axis 0',
          rt_input=[[[[1], [2]], [[3]]], [[]], [[[4, 5]]]],
          multiples=[2, 1, 1, 1],
          expected=[[[[1], [2]], [[3]]], [[]], [[[4, 5]]],
                    [[[1], [2]], [[3]]], [[]], [[[4, 5]]]]),
      dict(
          descr='rank=4, ragged_rank=3, repeat axis 1',
          rt_input=[[[[1], [2]], [[3]]], [[]], [[[4, 5]]]],
          multiples=[1, 2, 1, 1],
          expected=[[[[1], [2]], [[3]], [[1], [2]], [[3]]],
                    [[], []],
                    [[[4, 5]], [[4, 5]]]]),
      dict(
          descr='rank=4, ragged_rank=3, repeat axis 2',
          rt_input=[[[[1], [2]], [[3]]], [[]], [[[4, 5]]]],
          multiples=[1, 1, 2, 1],
          expected=[[[[1], [2], [1], [2]], [[3], [3]]],
                    [[]],
                    [[[4, 5], [4, 5]]]]),
      dict(
          descr='rank=4, ragged_rank=3, repeat axis 3',
          rt_input=[[[[1], [2]], [[3]]], [[]], [[[4, 5]]]],
          multiples=[1, 1, 1, 2],
          expected=[[[[1, 1], [2, 2]], [[3, 3]]], [[]], [[[4, 5, 4, 5]]]]),
      dict(
          descr='rank=4, ragged_rank=3, repeat all axes',
          rt_input=[[[['a'], ['b']], [['c']]], [[]], [[['d', 'e']]]],
          multiples=[5, 4, 3, 2],
          expected=[[[[b'a']*2, [b'b']*2]*3, [[b'c']*2]*3]*4,
                    [[]*3]*4,
                    [[[b'd', b'e']*2]*3]*4]*5),
      dict(
          descr='rank=5, ragged_rank=4, repeat all axes',
          rt_input=[[[[['a']]]]],
          multiples=[6, 5, 4, 3, 2],
          expected=[[[[[b'a']*2]*3]*4]*5]*6),
      #=========================================================================
      # multiple=0
      #=========================================================================
      dict(
          descr='rank=4, ragged_rank=3, repeat axis 0 (multiple=0)',
          rt_input=[[[[1], [2]], [[3]]], [[]], [[[4, 5]]]],
          multiples=[0, 1, 1, 1],
          expected=[]),
      dict(
          descr='rank=4, ragged_rank=3, repeat axis 1 (multiple=0)',
          rt_input=[[[[1], [2]], [[3]]], [[]], [[[4, 5]]]],
          multiples=[1, 0, 1, 1],
          expected=[[], [], []]),
      dict(
          descr='rank=4, ragged_rank=3, repeat axis 2 (multiple=0)',
          rt_input=[[[[1], [2]], [[3]]], [[]], [[[4, 5]]]],
          multiples=[1, 1, 0, 1],
          expected=[[[], []], [[]], [[]]]),
      dict(
          descr='rank=4, ragged_rank=3, repeat axis 3 (multiple=0)',
          rt_input=[[[[1], [2]], [[3]]], [[]], [[[4, 5]]]],
          multiples=[1, 1, 1, 0],
          expected=[[[[], []], [[]]], [[]], [[[]]]]),
      #=========================================================================
      # multiple=1
      #=========================================================================
      dict(
          descr='rank=4, multiples=1 (no repeats)',
          rt_input=[[[[1], [2]], [[3], [4]]], [[[5], [6]]]],
          multiples=[1, 1, 1, 1],
          expected=[[[[1], [2]], [[3], [4]]],
                    [[[5], [6]]]]),

  ])  # pyformat: disable
  def testRaggedTile(self,
                     descr,
                     rt_input,
                     multiples,
                     expected,
                     ragged_rank=None):
    rt = ragged_factory_ops.constant(rt_input, ragged_rank)

    expected_shape = [
        None if dim is None else dim * multiple
        for (dim, multiple) in zip(rt.shape.as_list(), multiples)
    ]

    # Test with both const & non-const multiples: ragged_tile has a few code
    # paths that optimize the case where multiples[d] is known to be 1.
    const_multiples = constant_op.constant(multiples, dtypes.int64)
    non_const_multiples = array_ops.placeholder_with_default(
        const_multiples, shape=[len(multiples)])

    for multiples_tensor in (const_multiples, non_const_multiples):
      tiled = ragged_array_ops.tile(rt, multiples_tensor)
      self.assertEqual(tiled.ragged_rank, rt.ragged_rank)
      self.assertEqual(tiled.shape.ndims, rt.shape.ndims)
      if multiples_tensor is const_multiples:
        self.assertEqual(tiled.shape.as_list(), expected_shape)
      with self.test_session():
        self.assertEqual(tiled.eval().tolist(), expected)

  def testRaggedTileWithTensorInput(self):
    # When the input is a `Tensor`, ragged_tile just delegates to tf.tile.
    dt = constant_op.constant([[1, 2], [3, 4]])
    tiled = ragged_array_ops.tile(dt, [3, 2])
    expected = [[1, 2, 1, 2], [3, 4, 3, 4],
                [1, 2, 1, 2], [3, 4, 3, 4],
                [1, 2, 1, 2], [3, 4, 3, 4]]  # pyformat: disable
    with self.test_session():
      self.assertEqual(tiled.eval().tolist(), expected)


if __name__ == '__main__':
  googletest.main()
