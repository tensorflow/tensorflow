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
"""Tests for ragged_array_ops.where."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import parameterized
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_where_op
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedWhereOpTest(test_util.TensorFlowTestCase,
                        parameterized.TestCase):

  @parameterized.parameters([
      #=========================================================================
      # Docstring Examples
      #=========================================================================
      dict(  # shape=[D1, (D2)]
          condition=ragged_factory_ops.constant_value(
              [[True, False, True], [False, True]]),
          expected=[[0, 0], [0, 2], [1, 1]]),
      dict(  # shape=[D1, (D2)]
          condition=ragged_factory_ops.constant_value(
              [[True, False, True], [False, True]]),
          x=ragged_factory_ops.constant_value(
              [['A', 'B', 'C'], ['D', 'E']]),
          y=ragged_factory_ops.constant_value(
              [['a', 'b', 'c'], ['d', 'e']]),
          expected=ragged_factory_ops.constant_value(
              [[b'A', b'b', b'C'], [b'd', b'E']])),
      dict(  # shape=[D1, (D2)]
          condition=ragged_factory_ops.constant_value([True, False]),
          x=ragged_factory_ops.constant_value([['A', 'B', 'C'], ['D', 'E']]),
          y=ragged_factory_ops.constant_value([['a', 'b', 'c'], ['d', 'e']]),
          expected=ragged_factory_ops.constant_value(
              [[b'A', b'B', b'C'], [b'd', b'e']])),
      #=========================================================================
      # Coordinate-retrieval mode
      #=========================================================================
      dict(  # shape=[D1]
          condition=[True, False, True, False, True],
          expected=[[0], [2], [4]]),
      dict(  # shape=[D1, D2]
          condition=[[True, False], [False, True]],
          expected=[[0, 0], [1, 1]]),
      dict(  # shape=[D1, (D2)]
          condition=ragged_factory_ops.constant_value(
              [[True, False, True], [False, True]]),
          expected=[[0, 0], [0, 2], [1, 1]]),
      dict(  # shape=[D1, (D2), (D3)]
          condition=ragged_factory_ops.constant_value([
              [[True, False, True], [False, True]],
              [[True], [], [False], [False, True, False]]
          ]),
          expected=[[0, 0, 0], [0, 0, 2], [0, 1, 1],
                    [1, 0, 0], [1, 3, 1]]),
      dict(  # shape=[D1, (D2), D3]
          condition=ragged_factory_ops.constant_value([
              [[True, False], [False, True]],
              [[True, False], [False, False], [True, False], [False, True]]
          ], ragged_rank=1),
          expected=[[0, 0, 0], [0, 1, 1],
                    [1, 0, 0], [1, 2, 0], [1, 3, 1]]),
      dict(  # shape=[D1, (D2), (D3), (D4)]
          condition=ragged_factory_ops.constant_value([
              [[[], [True]]],
              [[[True, False, True], [False, True]],
               [[True], [], [False], [False, True, False]]]
          ]),
          expected=[[0, 0, 1, 0],
                    [1, 0, 0, 0], [1, 0, 0, 2], [1, 0, 1, 1],
                    [1, 1, 0, 0], [1, 1, 3, 1]]),

      #=========================================================================
      # Elementwise value-selection mode
      #=========================================================================
      dict(  # shape=[]
          condition=True, x='A', y='a', expected=b'A'),
      dict(  # shape=[]
          condition=False, x='A', y='a', expected=b'a'),
      dict(  # shape=[D1]
          condition=[True, False, True],
          x=['A', 'B', 'C'],
          y=['a', 'b', 'c'],
          expected=[b'A', b'b', b'C']),
      dict(  # shape=[D1, D2]
          condition=[[True, False], [False, True]],
          x=[['A', 'B'], ['D', 'E']],
          y=[['a', 'b'], ['d', 'e']],
          expected=[[b'A', b'b'], [b'd', b'E']]),
      dict(  # shape=[D1, (D2)]
          condition=ragged_factory_ops.constant_value(
              [[True, False, True], [False, True]]),
          x=ragged_factory_ops.constant_value([['A', 'B', 'C'], ['D', 'E']]),
          y=ragged_factory_ops.constant_value([['a', 'b', 'c'], ['d', 'e']]),
          expected=ragged_factory_ops.constant_value(
              [[b'A', b'b', b'C'], [b'd', b'E']])),
      dict(  # shape=[D1, (D2), D3]
          condition=ragged_factory_ops.constant_value([
              [[True, False], [False, True]],
              [[True, False], [False, False], [True, False], [False, True]]
          ], ragged_rank=1),
          x=ragged_factory_ops.constant_value([
              [['A', 'B'], ['C', 'D']],
              [['E', 'F'], ['G', 'H'], ['I', 'J'], ['K', 'L']]
          ], ragged_rank=1),
          y=ragged_factory_ops.constant_value([
              [['a', 'b'], ['c', 'd']],
              [['e', 'f'], ['g', 'h'], ['i', 'j'], ['k', 'l']]
          ], ragged_rank=1),
          expected=ragged_factory_ops.constant_value([
              [[b'A', b'b'], [b'c', b'D']],
              [[b'E', b'f'], [b'g', b'h'], [b'I', b'j'], [b'k', b'L']]
          ], ragged_rank=1)),
      dict(  # shape=[D1, (D2), (D3), (D4)]
          condition=ragged_factory_ops.constant_value([
              [[[], [True]]],
              [[[True, False, True], [False, True]],
               [[True], [], [False], [False, True, False]]]
          ]),
          x=ragged_factory_ops.constant_value([
              [[[], ['A']]],
              [[['B', 'C', 'D'], ['E', 'F']],
               [['G'], [], ['H'], ['I', 'J', 'K']]]
          ]),
          y=ragged_factory_ops.constant_value([
              [[[], ['a']]],
              [[['b', 'c', 'd'], ['e', 'f']],
               [['g'], [], ['h'], ['i', 'j', 'k']]]
          ]),
          expected=ragged_factory_ops.constant_value([
              [[[], [b'A']]],
              [[[b'B', b'c', b'D'], [b'e', b'F']],
               [[b'G'], [], [b'h'], [b'i', b'J', b'k']]]
          ])),

      #=========================================================================
      # Elementwise row-selection mode
      #=========================================================================
      dict(  # x.shape=[D1, D2], y.shape=[D1, D2]
          condition=[True, False, True],
          x=[['A', 'B'], ['C', 'D'], ['E', 'F']],
          y=[['a', 'b'], ['c', 'd'], ['e', 'f']],
          expected=[[b'A', b'B'], [b'c', b'd'], [b'E', b'F']]),
      dict(  # x.shape=[D1, D2], y.shape=[D1, (D2)]
          condition=[True, False, True],
          x=[['A', 'B'], ['C', 'D'], ['E', 'F']],
          y=ragged_factory_ops.constant_value(
              [['a', 'b'], ['c'], ['d', 'e']]),
          expected=ragged_factory_ops.constant_value(
              [[b'A', b'B'], [b'c'], [b'E', b'F']])),
      dict(  # x.shape=[D1, (D2)], y.shape=[D1, (D2)]
          condition=[True, False, True],
          x=ragged_factory_ops.constant_value(
              [['A', 'B', 'C'], ['D', 'E'], ['F', 'G']]),
          y=ragged_factory_ops.constant_value(
              [['a', 'b'], ['c'], ['d', 'e']]),
          expected=ragged_factory_ops.constant_value(
              [[b'A', b'B', b'C'], [b'c'], [b'F', b'G']])),
      dict(  # shape=[D1, (D2), (D3), (D4)]
          condition=ragged_factory_ops.constant_value([True, False]),
          x=ragged_factory_ops.constant_value([
              [[[], ['A']]],
              [[['B', 'C', 'D'], ['E', 'F']],
               [['G'], [], ['H'], ['I', 'J', 'K']]]
          ]),
          y=ragged_factory_ops.constant_value([[[['a']]], [[['b']]]]),
          expected=ragged_factory_ops.constant_value(
              [[[[], [b'A']]], [[[b'b']]]])),
  ])   # pyformat: disable
  def testRaggedWhere(self, condition, expected, x=None, y=None):
    result = ragged_where_op.where(condition, x, y)
    self.assertAllEqual(result, expected)

  @parameterized.parameters([
      dict(
          condition=[True, False],
          x=[1, 2],
          error=ValueError,
          message='x and y must be either both None or both non-None'),
      dict(
          condition=ragged_factory_ops.constant_value([[True, False, True],
                                                       [False, True]]),
          x=ragged_factory_ops.constant_value([['A', 'B', 'C'], ['D', 'E']]),
          y=[['a', 'b'], ['d', 'e']],
          error=ValueError,
          message='Input shapes do not match.'),
  ])
  def testRaggedWhereErrors(self, condition, error, message, x=None, y=None):
    with self.assertRaisesRegexp(error, message):
      ragged_where_op.where(condition, x, y)


if __name__ == '__main__':
  googletest.main()
