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

"""Tests for the pointer_ops.span_overlaps() op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow_text.python.ops import pointer_ops


@test_util.run_all_in_graph_and_eager_modes
class SpanOverlapsOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  maxDiff = 5000  # Display diffs even if they're long.  pylint: disable=invalid-name

  #=============================================================================
  # Source & Target Spans:
  #    Offset: 0    5    10   15   20   25   30   35   40   45   50   55   60
  #            |====|====|====|====|====|====|====|====|====|====|====|====|
  # Source[0]: [-0-]     [-1-] [2] [3]    [4][-5-][-6-][-7-][-8-][-9-]
  # Target[0]: [-0-][-1-]     [-2-][-3-][-4-] [5] [6]    [7]  [-8-][-9-]
  #            |====|====|====|====|====|====|====|====|====|====|====|====|
  # Source[1]: [-0-]     [-1-]     [-2-]     [-3-]     [-4-]     [-5-]
  # Target[1]:  [2]      [-0-]   [----1---]  [3]         [4]
  #            |====|====|====|====|====|====|====|====|====|====|====|====|
  #            [----0----]
  # Source[2]:   [--1--][--3--]
  #               [--2--]
  #
  #              [--0--]
  # Target[2]: [------1------]
  #             [--2--]  [-3-]
  #            |====|====|====|====|====|====|====|====|====|====|====|====|
  #    Offset: 0    5    10   15   20   25   30   35   40   45   50   55   60
  BATCH_SIZE = 3
  SOURCE_START = [[0, 10, 16, 20, 27, 30, 35, 40, 45, 50],
                  [0, 10, 20, 30, 40, 50],
                  [0, 2, 3, 9]]  # pyformat: disable
  SOURCE_LIMIT = [[5, 15, 19, 23, 30, 35, 40, 45, 50, 55],
                  [5, 15, 25, 35, 45, 55],
                  [11, 9, 10, 16]]  # pyformat: disable
  TARGET_START = [[0, 5, 15, 20, 25, 31, 35, 42, 47, 52],
                  [10, 18, 1, 30, 42],
                  [2, 0, 1, 10]]  # pyformat: disable
  TARGET_LIMIT = [[5, 10, 20, 25, 30, 34, 38, 45, 52, 57],
                  [15, 28, 4, 33, 45],
                  [9, 15, 8, 15]]  # pyformat: disable

  # Spans encoded using 1D tensors
  BATCH_ITEM = []
  for i in range(BATCH_SIZE):
    BATCH_ITEM.append(
        dict(
            source_start=SOURCE_START[i],  # <int>[s]
            source_limit=SOURCE_LIMIT[i],  # <int>[s]
            target_start=TARGET_START[i],  # <int>[t]
            target_limit=TARGET_LIMIT[i],  # <int>[t]
        ))

  # Spans encoded using 2D ragged tensors
  RAGGED_BATCH_2D = dict(
      source_start=SOURCE_START,  # <int>[b, (s)]
      source_limit=SOURCE_LIMIT,  # <int>[b, (s)]
      target_start=TARGET_START,  # <int>[b, (t)]
      target_limit=TARGET_LIMIT,  # <int>[b, (t)]
  )

  # Spans encoded using 2D uniform tensors
  UNIFORM_BATCH_2D = dict(
      source_start=[row[:4] for row in SOURCE_START],  # <int>[b, s]
      source_limit=[row[:4] for row in SOURCE_LIMIT],  # <int>[b, s]
      target_start=[row[:4] for row in TARGET_START],  # <int>[b, t]
      target_limit=[row[:4] for row in TARGET_LIMIT],  # <int>[b, t]
  )

  # Spans encoded using a 3D ragged tensor with 2 ragged dimensions
  # (2 batch dimensions)
  RAGGED_BATCH_3D = dict(
      source_start=[SOURCE_START[:2], SOURCE_START[2:]],  # <int>[b1, (b2), (s)]
      source_limit=[SOURCE_LIMIT[:2], SOURCE_LIMIT[2:]],  # <int>[b1, (b2), (s)]
      target_start=[TARGET_START[:2], TARGET_START[2:]],  # <int>[b1, (b2), (t)]
      target_limit=[TARGET_LIMIT[:2], TARGET_LIMIT[2:]],  # <int>[b1, (b2), (t)]
  )

  # Spans encoded using a 3D uniform tensor (2 batch dimensions)
  UNIFORM_BATCH_3D = dict(
      source_start=[UNIFORM_BATCH_2D['source_start']] * 2,  # <int>[b1, b2, s]
      source_limit=[UNIFORM_BATCH_2D['source_limit']] * 2,  # <int>[b1, b2, s]
      target_start=[UNIFORM_BATCH_2D['target_start']] * 2,  # <int>[b1, b2, t]
      target_limit=[UNIFORM_BATCH_2D['target_limit']] * 2,  # <int>[b1, b2, t]
  )

  @parameterized.parameters(
      #=========================================================================
      # This group of tests use the following source & target spans:
      #   Offset: 0    5    10   15   20   25   30   35   40   45   50   55   60
      #           |====|====|====|====|====|====|====|====|====|====|====|====|
      #   Source: [-0-]     [-1-] [2] [3]    [4][-5-][-6-][-7-][-8-][-9-]
      #   Target: [-0-][-1-]     [-2-][-3-][-4-] [5] [6]    [7]  [-8-][-9-]
      #           |====|====|====|====|====|====|====|====|====|====|====|====|
      dict(
          name='test set 1, with default overlap flags',
          expected=[0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
          **BATCH_ITEM[0]),
      dict(
          name='test set 1, with contains=True',
          contains=True,
          expected=[0, -1, -1, -1, -1, 5, 6, 7, -1, -1],
          **BATCH_ITEM[0]),
      dict(
          name='test set 1, with contained_by=True',
          contained_by=True,
          expected=[0, -1, 2, 3, 4, -1, -1, -1, -1, -1],
          **BATCH_ITEM[0]),
      dict(
          name='test set 1, with contains=True and contained_by=True',
          contains=True,
          contained_by=True,
          expected=[0, -1, 2, 3, 4, 5, 6, 7, -1, -1],
          **BATCH_ITEM[0]),
      dict(
          name='test set 1, with partial_overlap=True',
          partial_overlap=True,
          expected=[0, -1, 2, 3, 4, 5, 6, 7, 8, 9],
          **BATCH_ITEM[0]),
      #=========================================================================
      # This group of tests use the following source & target spans:
      #   Offset: 0    5    10   15   20   25   30   35   40   45   50   55
      #           |====|====|====|====|====|====|====|====|====|====|====|
      #   Source: [-0-]     [-1-]     [-2-]     [-3-]     [-4-]     [-5-]
      #   Target:  [2]      [-0-]   [----1---]  [3]         [4]
      #           |====|====|====|====|====|====|====|====|====|====|====|
      dict(
          name='test set 2, with default overlap flags',
          expected=[-1, 0, -1, -1, -1, -1],
          **BATCH_ITEM[1]),
      dict(
          name='test set 2, with contains=True',
          contains=True,
          expected=[2, 0, -1, 3, 4, -1],
          **BATCH_ITEM[1]),
      dict(
          name='test set 2, with contained_by=True',
          contained_by=True,
          expected=[-1, 0, 1, -1, -1, -1],
          **BATCH_ITEM[1]),
      dict(
          name='test set 2, with partial_overlap=True',
          partial_overlap=True,
          expected=[2, 0, 1, 3, 4, -1],
          **BATCH_ITEM[1]),
      #=========================================================================
      # This group of tests use the following source & target spans:
      #   Offset: 0    5    10   15   20
      #           |====|====|====|====|
      #           [----0----]
      #   Source:   [--1--][--3--]
      #              [--2--]
      #           |====|====|====|====|
      #             [--0--]
      #   Target: [------1------]
      #            [--2--]  [-3-]
      #           |====|====|====|====|
      dict(
          name='test set 3, with default overlap flags',
          expected=[-1, 0, -1, -1],
          **BATCH_ITEM[2]),
      dict(
          name='test set 3, with contains=True',
          contains=True,
          expected=[2, 0, -1, 3],
          **BATCH_ITEM[2]),
      dict(
          name='test set 3, with contained_by=True',
          contained_by=True,
          expected=[1, 1, 1, -1],
          **BATCH_ITEM[2]),
      dict(
          name='test set 3, with contains=True and contained_by=True',
          contains=True,
          contained_by=True,
          expected=[2, 1, 1, 3],
          **BATCH_ITEM[2]),
      dict(
          name='test set 3, with partial_overlap=True',
          partial_overlap=True,
          expected=[3, 2, 2, 3],
          **BATCH_ITEM[2]),
      #=========================================================================
      # This group of tests use RAGGED_BATCH_2D.
      # Inputs have a single batch dimension, with shapes [b, (s)] and [b, (t)].
      dict(
          name='default overlap flags',
          expected=[
              [0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
              [-1, 0, -1, -1, -1, -1],
              [-1, 0, -1, -1],
          ],
          **RAGGED_BATCH_2D),
      dict(
          name='contains=True',
          contains=True,
          expected=[
              [0, -1, -1, -1, -1, 5, 6, 7, -1, -1],
              [2, 0, -1, 3, 4, -1],
              [2, 0, -1, 3],
          ],
          **RAGGED_BATCH_2D),
      #=========================================================================
      # This group of tests use UNIFORM_BATCH_2D
      # Inputs have a single batch dimension, with shapes [b, s] and [b, t].
      dict(
          name='default overlap flags',
          expected=[
              [0, -1, -1, -1],
              [-1, 0, -1, -1],
              [-1, 0, -1, -1],
          ],
          ragged_rank=0,
          **UNIFORM_BATCH_2D),
      dict(
          name='contains=True',
          contains=True,
          expected=[
              [0, -1, -1, -1],
              [2, 0, -1, 3],
              [2, 0, -1, 3],
          ],
          ragged_rank=0,
          **UNIFORM_BATCH_2D),
      #=========================================================================
      # This group of tests use RAGGED_BATCH_3D.
      # Inputs have two batch dimensions, with shapes [b1, (b2), (s)] and
      # [b1, (b2), (t)].
      dict(
          name='default overlap flags',
          expected=[
              [[0, -1, -1, -1, -1, -1, -1, -1, -1, -1], [-1, 0, -1, -1, -1,
                                                         -1]],
              [[-1, 0, -1, -1]],
          ],
          **RAGGED_BATCH_3D),
      dict(
          name='contains=True',
          contains=True,
          expected=[
              [[0, -1, -1, -1, -1, 5, 6, 7, -1, -1], [2, 0, -1, 3, 4, -1]],
              [[2, 0, -1, 3]],
          ],
          **RAGGED_BATCH_3D),
      #=========================================================================
      # This group of tests use UNIFORM_BATCH_3D
      # Inputs have two batch dimensions, with shapes [b1, b2, s] and
      # [b1, b2, t].
      dict(
          name='default overlap flags',
          expected=[[
              [0, -1, -1, -1],
              [-1, 0, -1, -1],
              [-1, 0, -1, -1],
          ]] * 2,
          ragged_rank=0,
          **UNIFORM_BATCH_3D),
      dict(
          name='contains=True',
          contains=True,
          expected=[[
              [0, -1, -1, -1],
              [2, 0, -1, 3],
              [2, 0, -1, 3],
          ]] * 2,
          ragged_rank=0,
          **UNIFORM_BATCH_3D),
  )  # pyformat: disable
  def testSpanAlignment(self,
                        name,
                        source_start,
                        source_limit,
                        target_start,
                        target_limit,
                        expected,
                        contains=False,
                        contained_by=False,
                        partial_overlap=False,
                        ragged_rank=None):
    source_start = ragged_factory_ops.constant(
        source_start, ragged_rank=ragged_rank)
    source_limit = ragged_factory_ops.constant(
        source_limit, ragged_rank=ragged_rank)
    target_start = ragged_factory_ops.constant(
        target_start, ragged_rank=ragged_rank)
    target_limit = ragged_factory_ops.constant(
        target_limit, ragged_rank=ragged_rank)
    multivalent_result = False
    alignment = pointer_ops.span_alignment(
        source_start, source_limit, target_start, target_limit, contains,
        contained_by, partial_overlap, multivalent_result)
    self.assertAllEqual(alignment, expected)

  @parameterized.parameters([
      #=========================================================================
      # This group of tests use the following source & target spans:
      #   Offset: 0    5    10   15   20   25   30   35   40   45   50   55   60
      #           |====|====|====|====|====|====|====|====|====|====|====|====|
      #   Source: [-0-]     [-1-] [2] [3]    [4][-5-][-6-][-7-][-8-][-9-]
      #   Target: [-0-][-1-]     [-2-][-3-][-4-] [5] [6]    [7]  [-8-][-9-][10]
      #           |====|====|====|====|====|====|====|====|====|====|====|====|
      dict(
          name='test set 1, with default overlap flags',
          expected=[[0], [], [], [], [], [], [], [], [], []],
          **BATCH_ITEM[0]),
      dict(
          name='test set 1, with contains=True',
          contains=True,
          expected=[[0], [], [], [], [], [5], [6], [7], [], []],
          **BATCH_ITEM[0]),
      dict(
          name='test set 1, with contained_by=True',
          contained_by=True,
          expected=[[0], [], [2], [3], [4], [], [], [], [], []],
          **BATCH_ITEM[0]),
      dict(
          name='test set 1, with contains=True and contained_by=True',
          contains=True,
          contained_by=True,
          expected=[[0], [], [2], [3], [4], [5], [6], [7], [], []],
          **BATCH_ITEM[0]),
      dict(
          name='test set 1, with partial_overlap=True',
          partial_overlap=True,
          expected=[[0], [], [2], [3], [4], [5], [6], [7], [8], [8, 9]],
          **BATCH_ITEM[0]),
      #=========================================================================
      # This group of tests use the following source & target spans:
      #   Offset: 0    5    10   15   20   25   30   35   40   45   50   55
      #           |====|====|====|====|====|====|====|====|====|====|====|
      #   Source: [-0-]     [-1-]     [-2-]     [-3-]     [-4-]     [-5-]
      #   Target:  [2]      [-0-]   [----1---]  [3]         [4]
      #           |====|====|====|====|====|====|====|====|====|====|====|
      dict(
          name='test set 2, with default overlap flags',
          expected=[[], [0], [], [], [], []],
          **BATCH_ITEM[1]),
      dict(
          name='test set 2, with contains=True',
          contains=True,
          expected=[[2], [0], [], [3], [4], []],
          **BATCH_ITEM[1]),
      dict(
          name='test set 2, with contained_by=True',
          contained_by=True,
          expected=[[], [0], [1], [], [], []],
          **BATCH_ITEM[1]),
      dict(
          name='test set 2, with partial_overlap=True',
          partial_overlap=True,
          expected=[[2], [0], [1], [3], [4], []],
          **BATCH_ITEM[1]),
      #=========================================================================
      # This group of tests use the following source & target spans:
      #   Offset: 0    5    10   15   20
      #           |====|====|====|====|
      #           [----0----]
      #   Source:   [--1--][--3--]
      #              [--2--]
      #           |====|====|====|====|
      #             [--0--]
      #   Target: [------1------]
      #            [--2--]  [-3-]
      #           |====|====|====|====|
      dict(
          name='test set 3, with default overlap flags',
          expected=[[], [0], [], []],
          **BATCH_ITEM[2]),
      dict(
          name='test set 3, with contains=True',
          contains=True,
          expected=[[0, 2], [0], [], [3]],
          **BATCH_ITEM[2]),
      dict(
          name='test set 3, with contained_by=True',
          contained_by=True,
          expected=[[1], [0, 1], [1], []],
          **BATCH_ITEM[2]),
      dict(
          name='test set 3, with contains=True and contained_by=True',
          contains=True,
          contained_by=True,
          expected=[[0, 1, 2], [0, 1], [1], [3]],
          **BATCH_ITEM[2]),
      dict(
          name='test set 3, with partial_overlap=True',
          partial_overlap=True,
          expected=[[0, 1, 2, 3], [0, 1, 2], [0, 1, 2], [1, 3]],
          **BATCH_ITEM[2]),
      #=========================================================================
      # This group of tests use RAGGED_BATCH_2D
      # Inputs have a single batch dimension, with shapes [b, (s)] and [b, (t)].
      dict(
          name='default overlap flags',
          expected=[
              [[0], [], [], [], [], [], [], [], [], []],
              [[], [0], [], [], [], []],
              [[], [0], [], []],
          ],
          **RAGGED_BATCH_2D),
      dict(
          name='contains=True',
          contains=True,
          expected=[
              [[0], [], [], [], [], [5], [6], [7], [], []],
              [[2], [0], [], [3], [4], []],
              [[0, 2], [0], [], [3]],
          ],
          **RAGGED_BATCH_2D),
      #=========================================================================
      # This group of tests use UNIFORM_BATCH_2D
      # Inputs have a single batch dimension, with shapes [b, s] and [b, t].
      dict(
          name='default overlap flags',
          expected=[
              [[0], [], [], []],
              [[], [0], [], []],
              [[], [0], [], []],
          ],
          ragged_rank=0,
          **UNIFORM_BATCH_2D),
      dict(
          name='contains=True',
          contains=True,
          expected=[
              [[0], [], [], []],
              [[2], [0], [], [3]],
              [[0, 2], [0], [], [3]],
          ],
          ragged_rank=0,
          **UNIFORM_BATCH_2D),
      #=========================================================================
      # This group of tests use RAGGED_BATCH_3D
      # Inputs have two batch dimensions, with shapes [b1, (b2), (s)] and
      # [b1, (b2), (t)].
      dict(
          name='default overlap flags',
          expected=[
              [[[0], [], [], [], [], [], [], [], [], []],
               [[], [0], [], [], [], []]],
              [[[], [0], [], []]],
          ],
          **RAGGED_BATCH_3D),
      dict(
          name='contains=True',
          contains=True,
          expected=[
              [[[0], [], [], [], [], [5], [6], [7], [], []],
               [[2], [0], [], [3], [4], []]],
              [[[0, 2], [0], [], [3]]],
          ],
          **RAGGED_BATCH_3D),
      #=========================================================================
      # This group of tests use UNIFORM_BATCH_3D
      # Inputs have two batch dimensions, with shapes [b1, b2, s] and
      # [b1, b2, t].
      dict(
          name='default overlap flags',
          expected=[[
              [[0], [], [], []],
              [[], [0], [], []],
              [[], [0], [], []],
          ]] * 2,
          ragged_rank=0,
          **UNIFORM_BATCH_3D),
      dict(
          name='contains=True',
          contains=True,
          expected=[[
              [[0], [], [], []],
              [[2], [0], [], [3]],
              [[0, 2], [0], [], [3]],
          ]] * 2,
          ragged_rank=0,
          **UNIFORM_BATCH_3D),
  ])  # pyformat: disable
  def testSpanMultiAlignment(self,
                             name,
                             source_start,
                             source_limit,
                             target_start,
                             target_limit,
                             expected,
                             contains=False,
                             contained_by=False,
                             partial_overlap=False,
                             ragged_rank=None):
    source_start = ragged_factory_ops.constant(
        source_start, ragged_rank=ragged_rank)
    source_limit = ragged_factory_ops.constant(
        source_limit, ragged_rank=ragged_rank)
    target_start = ragged_factory_ops.constant(
        target_start, ragged_rank=ragged_rank)
    target_limit = ragged_factory_ops.constant(
        target_limit, ragged_rank=ragged_rank)
    multivalent_result = True
    alignment = pointer_ops.span_alignment(
        source_start, source_limit, target_start, target_limit, contains,
        contained_by, partial_overlap, multivalent_result)
    self.assertAllEqual(alignment, expected)


if __name__ == '__main__':
  test.main()
