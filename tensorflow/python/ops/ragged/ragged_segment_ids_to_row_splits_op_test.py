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
"""Tests for the segment_id_ops.segment_ids_to_row_splits() op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedSplitsToSegmentIdsOpTest(test_util.TensorFlowTestCase):

  def testDocStringExample(self):
    segment_ids = [0, 0, 0, 2, 2, 3, 4, 4, 4]
    expected = [0, 3, 3, 5, 6, 9]
    splits = segment_id_ops.segment_ids_to_row_splits(segment_ids)
    self.assertAllEqual(splits, expected)

  def testEmptySegmentIds(self):
    # Note: the splits for an empty ragged tensor contains a single zero.
    segment_ids = segment_id_ops.segment_ids_to_row_splits([])
    self.assertAllEqual(segment_ids, [0])

  def testErrors(self):
    self.assertRaisesRegex(TypeError,
                           r'segment_ids must be an integer tensor.*',
                           segment_id_ops.segment_ids_to_row_splits,
                           constant_op.constant([0.5]))
    self.assertRaisesRegex(ValueError, r'Shape \(\) must have rank 1',
                           segment_id_ops.segment_ids_to_row_splits, 0)
    self.assertRaisesRegex(ValueError, r'Shape \(1, 1\) must have rank 1',
                           segment_id_ops.segment_ids_to_row_splits, [[0]])

  def testNumSegments(self):
    segment_ids = [0, 0, 0, 2, 2, 3, 4, 4, 4]
    num_segments = 7
    expected = [0, 3, 3, 5, 6, 9, 9, 9]
    splits = segment_id_ops.segment_ids_to_row_splits(segment_ids, num_segments)
    self.assertAllEqual(splits, expected)

  def testUnsortedSegmentIds(self):
    # Segment ids are not required to be sorted.
    segment_ids = [0, 4, 3, 2, 4, 4, 2, 0, 0]
    splits1 = segment_id_ops.segment_ids_to_row_splits(segment_ids)
    expected1 = [0, 3, 3, 5, 6, 9]

    splits2 = segment_id_ops.segment_ids_to_row_splits(segment_ids, 7)
    expected2 = [0, 3, 3, 5, 6, 9, 9, 9]
    self.assertAllEqual(splits1, expected1)
    self.assertAllEqual(splits2, expected2)


if __name__ == '__main__':
  googletest.main()
