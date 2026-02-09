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

"""Tests for ops to build and pack segments."""
from absl.testing import parameterized

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow_text.python.ops import segment_combiner_ops


class SegmentBuilderTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      dict(
          descr="Test empty",
          segments=[
              # first segment
              [[], [], []],
          ],
          expected_combined=[
              [101, 102],
              [101, 102],
              [101, 102],
          ],
          expected_segment_ids=[[0, 0], [0, 0], [0, 0]],
      ),
      dict(
          descr="Test custom start and end of sequence ids",
          segments=[
              # first segment
              [[1, 2], [
                  3,
                  4,
              ], [5, 6, 7, 8, 9]],
          ],
          expected_combined=[
              [1001, 1, 2, 1002],
              [1001, 3, 4, 1002],
              [1001, 5, 6, 7, 8, 9, 1002],
          ],
          expected_segment_ids=[[0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]],
          start_id=1001,
          end_id=1002,
      ),
      dict(
          descr="Single segment: test rank 3 input segments",
          segments=[
              # first segment
              [[[1], [2]], [[3], [4]], [[5], [6], [7], [8], [9]]],
          ],
          expected_combined=[
              [[101], [1], [2], [102]],
              [[101], [3], [4], [102]],
              [[101], [5], [6], [7], [8], [9], [102]],
          ],
          expected_segment_ids=[
              [[0], [0], [0], [0]],
              [[0], [0], [0], [0]],
              [[0], [0], [0], [0], [0], [0], [0]],
          ],
      ),
      dict(
          descr="Test single segment",
          segments=[
              # first segment
              [[1, 2], [
                  3,
                  4,
              ], [5, 6, 7, 8, 9]],
          ],
          expected_combined=[
              [101, 1, 2, 102],
              [101, 3, 4, 102],
              [101, 5, 6, 7, 8, 9, 102],
          ],
          expected_segment_ids=[[0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]],
      ),
      dict(
          descr="Test two segments",
          segments=[
              # first segment
              [[1, 2], [
                  3,
                  4,
              ], [5, 6, 7, 8, 9]],
              # second segment
              [[
                  10,
                  20,
              ], [
                  30,
                  40,
                  50,
                  60,
              ], [70, 80]],
          ],
          expected_combined=[
              [101, 1, 2, 102, 10, 20, 102],
              [101, 3, 4, 102, 30, 40, 50, 60, 102],
              [101, 5, 6, 7, 8, 9, 102, 70, 80, 102],
          ],
          expected_segment_ids=[[0, 0, 0, 0, 1, 1, 1],
                                [0, 0, 0, 0, 1, 1, 1, 1, 1],
                                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]],
      ),
      dict(
          descr="Test two rank 3 segments",
          segments=[
              # first segment
              [[[1], [2]], [[3], [4]], [[5], [6], [7], [8], [9]]],
              # second segment
              [[[10], [20]], [[30], [40], [50], [60]], [[70], [80]]],
          ],
          expected_combined=[
              [[101], [1], [2], [102], [10], [20], [102]],
              [[101], [3], [4], [102], [30], [40], [50], [60], [102]],
              [[101], [5], [6], [7], [8], [9], [102], [70], [80], [102]],
          ],
          expected_segment_ids=[[[0], [0], [0], [0], [1], [1], [1]],
                                [[0], [0], [0], [0], [1], [1], [1], [1], [1]],
                                [[0], [0], [0], [0], [0], [0], [0], [1], [1],
                                 [1]]],
      ),
      dict(
          descr="Test that if we have 3 or more segments in the list, the " +
          "segment ids are correct",
          segments=[
              # first segment
              [[1, 2], [3, 4], [5, 6, 7, 8, 9]],
              # second segment
              [[10, 20], [30, 40, 50, 60], [70, 80]],
              # third segment
              [[100, 200, 300, 400], [
                  500,
                  600,
              ], [700, 800]],
          ],
          expected_combined=[[
              101, 1, 2, 102, 10, 20, 102, 100, 200, 300, 400, 102
          ], [101, 3, 4, 102, 30, 40, 50, 60, 102, 500, 600,
              102], [101, 5, 6, 7, 8, 9, 102, 70, 80, 102, 700, 800, 102]],
          expected_segment_ids=[[0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2],
                                [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
                                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2]],
      ),
  ])
  def testSegmentBuilder(self,
                         segments,
                         expected_combined,
                         expected_segment_ids,
                         start_id=101,
                         end_id=102,
                         descr=None):
    for segment_dtype in [dtypes.int32, dtypes.int64]:
      segments_as_tensors = [
          ragged_factory_ops.constant(seg, dtype=segment_dtype)
          for seg in segments
      ]
      actual_combined, actual_segment_ids = (
          segment_combiner_ops.combine_segments(
              segments_as_tensors, constant_op.constant(start_id,
                                                        segment_dtype), end_id))
      self.assertAllEqual(expected_combined, actual_combined)
      self.assertAllEqual(expected_segment_ids, actual_segment_ids)

  @parameterized.parameters([
      dict(
          descr="Test empty",
          segments=[
              # first segment
              [[], [], []],
          ],
          expected_combined=[
              [],
              [],
              [],
          ],
          expected_segment_ids=[[], [], []],
      ),
      dict(
          descr="Single segment: test rank 3 input segments",
          segments=[
              # first segment
              [[[1], [2]], [[3], [4]], [[5], [6], [7], [8], [9]]],
          ],
          expected_combined=[
              [[1], [2]],
              [[3], [4]],
              [[5], [6], [7], [8], [9]],
          ],
          expected_segment_ids=[
              [[0], [0]],
              [[0], [0]],
              [[0], [0], [0], [0], [0]],
          ],
      ),
      dict(
          descr="Test single segment",
          segments=[
              # first segment
              [
                  [1, 2],
                  [3, 4],
                  [5, 6, 7, 8, 9],
              ],
          ],
          expected_combined=[
              [1, 2],
              [3, 4],
              [5, 6, 7, 8, 9],
          ],
          expected_segment_ids=[
              [0, 0],
              [0, 0],
              [0, 0, 0, 0, 0],
          ],
      ),
      dict(
          descr="Test two segments",
          segments=[
              # first segment
              [
                  [1, 2],
                  [3, 4],
                  [5, 6, 7, 8, 9],
              ],
              # second segment
              [
                  [10, 20],
                  [30, 40, 50, 60],
                  [70, 80],
              ],
          ],
          expected_combined=[
              [1, 2, 10, 20],
              [3, 4, 30, 40, 50, 60],
              [5, 6, 7, 8, 9, 70, 80],
          ],
          expected_segment_ids=[
              [0, 0, 1, 1],
              [0, 0, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 1, 1],
          ],
      ),
      dict(
          descr="Test two rank 3 segments",
          segments=[
              # first segment
              [[[1], [2]], [[3], [4]], [[5], [6], [7], [8], [9]]],
              # second segment
              [[[10], [20]], [[30], [40], [50], [60]], [[70], [80]]],
          ],
          expected_combined=[
              [[1], [2], [10], [20]],
              [[3], [4], [30], [40], [50], [60]],
              [[5], [6], [7], [8], [9], [70], [80]],
          ],
          expected_segment_ids=[
              [[0], [0], [1], [1]],
              [[0], [0], [1], [1], [1], [1]],
              [[0], [0], [0], [0], [0], [1], [1]],
          ],
      ),
      dict(
          descr="Test that if we have 3 or more segments in the list, the "
          + "segment ids are correct",
          segments=[
              # first segment
              [[1, 2], [3, 4], [5, 6, 7, 8, 9]],
              # second segment
              [[10, 20], [30, 40, 50, 60], [70, 80]],
              # third segment
              [
                  [100, 200, 300, 400],
                  [500, 600],
                  [700, 800],
              ],
          ],
          expected_combined=[
              [1, 2, 10, 20, 100, 200, 300, 400],
              [3, 4, 30, 40, 50, 60, 500, 600],
              [5, 6, 7, 8, 9, 70, 80, 700, 800],
          ],
          expected_segment_ids=[
              [0, 0, 1, 1, 2, 2, 2, 2],
              [0, 0, 1, 1, 1, 1, 2, 2],
              [0, 0, 0, 0, 0, 1, 1, 2, 2],
          ],
      ),
  ])
  def testConcatenateSegments(
      self, segments, expected_combined, expected_segment_ids, descr=None
  ):
    for segment_dtype in [dtypes.int32, dtypes.int64]:
      segments_as_tensors = [
          ragged_factory_ops.constant(seg, dtype=segment_dtype)
          for seg in segments
      ]
      actual_combined, actual_segment_ids = (
          segment_combiner_ops.concatenate_segments(segments_as_tensors)
      )
      self.assertAllEqual(expected_combined, actual_combined)
      self.assertAllEqual(expected_segment_ids, actual_segment_ids)


if __name__ == "__main__":
  test.main()
