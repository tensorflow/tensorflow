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

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow_text.python.ops import pointer_ops


@test_util.run_all_in_graph_and_eager_modes
class SpanOverlapsOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  maxDiff = 5000  # Display diffs even if they're long.  pylint: disable=invalid-name

  # =============================================================================
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
  RAGGED_BATCH_3D = dict(
      source_start=[SOURCE_START[:2], SOURCE_START[2:]],  # <int>[b1, (b2), (s)]
      source_limit=[SOURCE_LIMIT[:2], SOURCE_LIMIT[2:]],  # <int>[b1, (b2), (s)]
      target_start=[TARGET_START[:2], TARGET_START[2:]],  # <int>[b1, (b2), (t)]
      target_limit=[TARGET_LIMIT[:2], TARGET_LIMIT[2:]],  # <int>[b1, (b2), (t)]
  )

  @parameterized.parameters(
      #=========================================================================
      # This group of tests use BATCH_ITEM[0]:
      #   Offset: 0    5    10   15   20   25   30   35   40   45   50   55   60
      #           |====|====|====|====|====|====|====|====|====|====|====|====|
      #   Source: [-0-]     [-1-] [2] [3]    [4][-5-][-6-][-7-][-8-][-9-]
      #   Target: [-0-][-1-]     [-2-][-3-][-4-] [5] [6]    [7]  [-8-][-9-]
      #           |====|====|====|====|====|====|====|====|====|====|====|====|
      dict(
          name='test set 1, with default overlap flags',
          expected_overlap_pairs=[(0, 0)],
          **BATCH_ITEM[0]),
      dict(
          name='test set 1, with contains=True',
          contains=True,
          expected_overlap_pairs=[(0, 0), (5, 5), (6, 6), (7, 7)],
          **BATCH_ITEM[0]),
      dict(
          name='test set 1, with contained_by=True',
          contained_by=True,
          expected_overlap_pairs=[(0, 0), (2, 2), (3, 3), (4, 4)],
          **BATCH_ITEM[0]),
      dict(
          name='test set 1, with contains=True and contained_by=True',
          contains=True,
          contained_by=True,
          expected_overlap_pairs=[(0, 0), (2, 2), (3, 3), (4, 4), (5, 5), (6,
                                                                           6),
                                  (7, 7)],
          **BATCH_ITEM[0]),
      dict(
          name='test set 1, with partial_overlap=True',
          partial_overlap=True,
          expected_overlap_pairs=[(0, 0), (2, 2), (3, 3), (4, 4), (5, 5), (6,
                                                                           6),
                                  (7, 7), (8, 8), (9, 8), (9, 9)],
          **BATCH_ITEM[0]),
      #=========================================================================
      # This group of tests use BATCH_ITEM[1]:
      #   Offset: 0    5    10   15   20   25   30   35   40   45   50   55
      #           |====|====|====|====|====|====|====|====|====|====|====|
      #   Source: [-0-]     [-1-]     [-2-]     [-3-]     [-4-]     [-5-]
      #   Target:  [2]      [-0-]   [----1---]  [3]         [4]
      #           |====|====|====|====|====|====|====|====|====|====|====|
      dict(
          name='test set 2, with default overlap flags',
          expected_overlap_pairs=[(1, 0)],
          **BATCH_ITEM[1]),
      dict(
          name='test set 2, with contains=True',
          contains=True,
          expected_overlap_pairs=[(0, 2), (1, 0), (3, 3), (4, 4)],
          **BATCH_ITEM[1]),
      dict(
          name='test set 2, with contained_by=True',
          contained_by=True,
          expected_overlap_pairs=[(1, 0), (2, 1)],
          **BATCH_ITEM[1]),
      dict(
          name='test set 2, with partial_overlap=True',
          partial_overlap=True,
          expected_overlap_pairs=[(0, 2), (1, 0), (2, 1), (3, 3), (4, 4)],
          **BATCH_ITEM[1]),
      #=========================================================================
      # This group of tests use BATCH_ITEM[2]:
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
          expected_overlap_pairs=[(1, 0)],
          **BATCH_ITEM[2]),
      dict(
          name='test set 3, with contains=True',
          contains=True,
          expected_overlap_pairs=[(0, 0), (0, 2), (1, 0), (3, 3)],
          **BATCH_ITEM[2]),
      dict(
          name='test set 3, with contained_by=True',
          contained_by=True,
          expected_overlap_pairs=[(0, 1), (1, 0), (1, 1), (2, 1)],
          **BATCH_ITEM[2]),
      dict(
          name='test set 3, with contains=True and contained_by=True',
          contains=True,
          contained_by=True,
          expected_overlap_pairs=[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2,
                                                                           1),
                                  (3, 3)],
          **BATCH_ITEM[2]),
      dict(
          name='test set 3, with partial_overlap=True',
          partial_overlap=True,
          expected_overlap_pairs=[(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1,
                                                                           1),
                                  (1, 2), (2, 0), (2, 1), (2, 2), (3, 1), (3,
                                                                           3)],
          **BATCH_ITEM[2]),
  )
  def test1DSpanOverlaps(self,
                         name,
                         source_start,
                         source_limit,
                         target_start,
                         target_limit,
                         expected_overlap_pairs,
                         contains=False,
                         contained_by=False,
                         partial_overlap=False):
    # Assemble expected value.  (Writing out the complete expected result
    # matrix takes up a lot of space, so instead we just list the positions
    # in the matrix that should be True.)
    # pylint: disable=g-complex-comprehension
    expected = [[(s, t) in expected_overlap_pairs
                 for t in range(len(target_limit))]
                for s in range(len(source_limit))]

    overlaps = pointer_ops.span_overlaps(source_start, source_limit,
                                         target_start, target_limit, contains,
                                         contained_by, partial_overlap)
    self.assertAllEqual(overlaps, expected)

  @parameterized.parameters([
      #=========================================================================
      # This group of tests use RAGGED_BATCH_2D
      dict(
          name='default overlap flags',
          expected_overlap_pairs=[
              (0, 0, 0),  # batch 0
              (1, 1, 0),  # batch 1
              (2, 1, 0),  # batch 2
              ],
          **RAGGED_BATCH_2D),
      dict(
          name='contains=True',
          contains=True,
          expected_overlap_pairs=[
              (0, 0, 0), (0, 5, 5), (0, 6, 6), (0, 7, 7),  # batch 0
              (1, 0, 2), (1, 1, 0), (1, 3, 3), (1, 4, 4),  # batch 1
              (2, 0, 0), (2, 0, 2), (2, 1, 0), (2, 3, 3),  # batch 2
              ],
          **RAGGED_BATCH_2D),
      dict(
          name='contained_by=True',
          contained_by=True,
          expected_overlap_pairs=[
              (0, 0, 0), (0, 2, 2), (0, 3, 3), (0, 4, 4),    # batch 0
              (1, 1, 0), (1, 2, 1),                          # batch 1
              (2, 0, 1), (2, 1, 0), (2, 1, 1), (2, 2, 1)],   # batch 2
          **RAGGED_BATCH_2D),
      dict(
          name='contains=True and contained_by=True',
          contains=True,
          contained_by=True,
          expected_overlap_pairs=[
              # Batch 0:
              (0, 0, 0), (0, 2, 2), (0, 3, 3), (0, 4, 4), (0, 5, 5),
              (0, 6, 6), (0, 7, 7),
              # Batch 1:
              (1, 0, 2), (1, 1, 0), (1, 2, 1), (1, 3, 3), (1, 4, 4),
              # Batch 2:
              (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 1, 0), (2, 1, 1),
              (2, 2, 1), (2, 3, 3)],
          **RAGGED_BATCH_2D),
      dict(
          name='partial_overlap=True',
          partial_overlap=True,
          expected_overlap_pairs=[
              # Batch 0:
              (0, 0, 0), (0, 2, 2), (0, 3, 3), (0, 4, 4), (0, 5, 5),
              (0, 6, 6), (0, 7, 7), (0, 8, 8), (0, 9, 8), (0, 9, 9),
              # Batch 1:
              (1, 0, 2), (1, 1, 0), (1, 2, 1), (1, 3, 3), (1, 4, 4),
              # Batch 2:
              (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 0, 3), (2, 1, 0),
              (2, 1, 1), (2, 1, 2), (2, 2, 0), (2, 2, 1), (2, 2, 2),
              (2, 3, 1), (2, 3, 3)],
          **RAGGED_BATCH_2D),
      #=========================================================================
      # This group of tests use UNIFORM_BATCH_2D
      dict(
          name='default overlap flags',
          expected_overlap_pairs=[
              (0, 0, 0),  # batch 0
              (1, 1, 0),  # batch 1
              (2, 1, 0),  # batch 2
              ],
          ragged_rank=0,
          **UNIFORM_BATCH_2D),
      dict(
          name='contains=True',
          contains=True,
          expected_overlap_pairs=[
              (0, 0, 0),                                   # batch 0
              (1, 0, 2), (1, 1, 0), (1, 3, 3),             # batch 1
              (2, 0, 0), (2, 0, 2), (2, 1, 0), (2, 3, 3),  # batch 2
              ],
          ragged_rank=0,
          **UNIFORM_BATCH_2D),
  ])  # pyformat: disable
  def test2DSpanOverlaps(self,
                         name,
                         source_start,
                         source_limit,
                         target_start,
                         target_limit,
                         expected_overlap_pairs,
                         contains=False,
                         contained_by=False,
                         partial_overlap=False,
                         ragged_rank=None):
    # Assemble expected value.
    # pylint: disable=g-complex-comprehension
    expected = [[[(b, s, t) in expected_overlap_pairs
                  for t in range(len(target_limit[b]))]
                 for s in range(len(source_limit[b]))]
                for b in range(self.BATCH_SIZE)]

    source_start = ragged_factory_ops.constant(
        source_start, ragged_rank=ragged_rank)
    source_limit = ragged_factory_ops.constant(
        source_limit, ragged_rank=ragged_rank)
    target_start = ragged_factory_ops.constant(
        target_start, ragged_rank=ragged_rank)
    target_limit = ragged_factory_ops.constant(
        target_limit, ragged_rank=ragged_rank)
    overlaps = pointer_ops.span_overlaps(source_start, source_limit,
                                         target_start, target_limit, contains,
                                         contained_by, partial_overlap)
    self.assertAllEqual(overlaps, expected)

  @parameterized.parameters([
      #=========================================================================
      # This group of tests use RAGGED_BATCH_3D
      dict(
          name='default overlap flags',
          expected_overlap_pairs=[
              (0, 0, 0, 0),  # batch [0, 0]
              (0, 1, 1, 0),  # batch [0, 1]
              (1, 0, 1, 0),  # batch [1, 0]
              ],
          **RAGGED_BATCH_3D),
      dict(
          name='contains=True',
          contains=True,
          expected_overlap_pairs=[
              (0, 0, 0, 0), (0, 0, 5, 5), (0, 0, 6, 6), (0, 0, 7, 7),  # b[0, 0]
              (0, 1, 0, 2), (0, 1, 1, 0), (0, 1, 3, 3), (0, 1, 4, 4),  # b[0, 1]
              (1, 0, 0, 0), (1, 0, 0, 2), (1, 0, 1, 0), (1, 0, 3, 3),  # b[1, 0]
              ],
          **RAGGED_BATCH_3D),
  ])  # pyformat: disable
  def test3DSpanOverlaps(self,
                         name,
                         source_start,
                         source_limit,
                         target_start,
                         target_limit,
                         expected_overlap_pairs,
                         contains=False,
                         contained_by=False,
                         partial_overlap=False,
                         ragged_rank=None):
    # Assemble expected value.
    # pylint: disable=g-complex-comprehension
    expected = [[[[(b1, b2, s, t) in expected_overlap_pairs
                   for t in range(len(target_limit[b1][b2]))]
                  for s in range(len(source_limit[b1][b2]))]
                 for b2 in range(len(source_limit[b1]))]
                for b1 in range(2)]

    source_start = ragged_factory_ops.constant(
        source_start, ragged_rank=ragged_rank)
    source_limit = ragged_factory_ops.constant(
        source_limit, ragged_rank=ragged_rank)
    target_start = ragged_factory_ops.constant(
        target_start, ragged_rank=ragged_rank)
    target_limit = ragged_factory_ops.constant(
        target_limit, ragged_rank=ragged_rank)
    overlaps = pointer_ops.span_overlaps(source_start, source_limit,
                                         target_start, target_limit, contains,
                                         contained_by, partial_overlap)
    self.assertAllEqual(overlaps, expected)

  def testErrors(self):
    t = [10, 20, 30, 40, 50]

    with self.assertRaisesRegex(TypeError, 'contains must be bool.'):
      pointer_ops.span_overlaps(t, t, t, t, contains='x')
    with self.assertRaisesRegex(TypeError, 'contained_by must be bool.'):
      pointer_ops.span_overlaps(t, t, t, t, contained_by='x')
    with self.assertRaisesRegex(TypeError, 'partial_overlap must be bool.'):
      pointer_ops.span_overlaps(t, t, t, t, partial_overlap='x')
    with self.assertRaisesRegex(
        TypeError,
        'source_start, source_limit, target_start, and '
        'target_limit must all have the same dtype',
    ):
      pointer_ops.span_overlaps(t, t, t, [1.0, 2.0, 3.0, 4.0, 5.0])
    with self.assertRaisesRegex(
        ValueError, r'Shapes \(5,\) and \(4,\) are incompatible'
    ):
      pointer_ops.span_overlaps(t, t[:4], t, t)
    with self.assertRaisesRegex(
        ValueError, r'Shapes \(4,\) and \(5,\) are incompatible'
    ):
      pointer_ops.span_overlaps(t, t, t[:4], t)
    with self.assertRaisesRegex(
        ValueError, r'Shapes \(1, 5\) and \(5,\) must have the same rank'
    ):
      pointer_ops.span_overlaps([t], [t], t, t)
    if not context.executing_eagerly():
      with self.assertRaisesRegex(
          ValueError,
          'For ragged inputs, the shape.ndims of at least one '
          'span tensor must be statically known.',
      ):
        x = ragged_tensor.RaggedTensor.from_row_splits(
            array_ops.placeholder(dtypes.int32), [0, 3, 8])
        pointer_ops.span_overlaps(x, x, x, x)
    with self.assertRaisesRegex(
        ValueError, 'Span tensors must all have the same ragged_rank'
    ):
      a = [[10, 20, 30], [40, 50, 60]]
      pointer_ops.span_overlaps(a, a, a, ragged_factory_ops.constant(a))
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'Mismatched ragged shapes for batch dimensions',
    ):
      rt1 = ragged_factory_ops.constant([[[1, 2], [3]], [[4, 5]]])
      rt2 = ragged_factory_ops.constant([[[1, 2], [3]], [[4, 5], [6]]])
      pointer_ops.span_overlaps(rt1, rt1, rt2, rt2)


if __name__ == '__main__':
  test.main()
