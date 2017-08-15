# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for contrib.seq2seq.python.seq2seq.beam_search_ops."""
# pylint: disable=unused-import,g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# pylint: enable=unused-import

import numpy as np

from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


def _transpose_batch_time(x):
  return np.transpose(x, [1, 0, 2]).astype(np.int32)


class GatherTreeTest(test.TestCase):

  def testGatherTreeOne(self):
    # (max_time = 4, batch_size = 1, beams = 3)
    step_ids = _transpose_batch_time(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -1, -1]]])
    parent_ids = _transpose_batch_time(
        [[[0, 0, 0], [0, 1, 1], [2, 1, 2], [-1, -1, -1]]])
    sequence_length = [[3, 3, 3]]
    expected_result = _transpose_batch_time(
        [[[2, 2, 2], [6, 5, 6], [7, 8, 9], [-1, -1, -1]]])
    beams = beam_search_ops.gather_tree(
        step_ids=step_ids, parent_ids=parent_ids,
        sequence_length=sequence_length)
    with self.test_session(use_gpu=True):
      self.assertAllEqual(expected_result, beams.eval())

  def testBadParentValuesOnCPU(self):
    # (batch_size = 1, max_time = 4, beams = 3)
    # bad parent in beam 1 time 1
    step_ids = _transpose_batch_time(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -1, -1]]])
    parent_ids = _transpose_batch_time(
        [[[0, 0, 0], [0, -1, 1], [2, 1, 2], [-1, -1, -1]]])
    sequence_length = [[3, 3, 3]]
    with ops.device("/cpu:0"):
      beams = beam_search_ops.gather_tree(
          step_ids=step_ids, parent_ids=parent_ids,
          sequence_length=sequence_length)
    with self.test_session():
      with self.assertRaisesOpError(
          r"parent id -1 at \(batch, time, beam\) == \(0, 0, 1\)"):
        _ = beams.eval()

  def testBadParentValuesOnGPU(self):
    # Only want to run this test on CUDA devices, as gather_tree is not
    # registered for SYCL devices.
    if not test.is_gpu_available(cuda_only=True):
      return
    # (max_time = 4, batch_size = 1, beams = 3)
    # bad parent in beam 1 time 1; appears as a negative index at time 0
    step_ids = _transpose_batch_time(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -1, -1]]])
    parent_ids = _transpose_batch_time(
        [[[0, 0, 0], [0, -1, 1], [2, 1, 2], [-1, -1, -1]]])
    sequence_length = [[3, 3, 3]]
    expected_result = _transpose_batch_time(
        [[[2, -1, 2], [6, 5, 6], [7, 8, 9], [-1, -1, -1]]])
    with ops.device("/device:GPU:0"):
      beams = beam_search_ops.gather_tree(
          step_ids=step_ids, parent_ids=parent_ids,
          sequence_length=sequence_length)
    with self.test_session(use_gpu=True):
      self.assertAllEqual(expected_result, beams.eval())

  def testGatherTreeBatch(self):
    # sequence_length is [batch_size, beam_width] = [4, 5]
    sequence_length = [[0] * 5, [1] * 5, [2] * 5, [3] * 5]

    with self.test_session(use_gpu=True):
      # (max_time = 4, batch_size = 4, beam_width = 5)
      step_ids = _transpose_batch_time(
          [[[3, 4, 0, 4, 0],
            [4, 2, 0, 3, 1],
            [1, 1, 3, 2, 2],
            [3, 1, 2, 3, 4]],
           [[3, 4, 0, 4, 0],
            [4, 2, 0, 3, 1],
            [1, 1, 3, 2, 2],
            [3, 1, 2, 3, 4]],
           [[1, 2, 3, 4, 2],
            [2, 1, 1, 3, 2],
            [3, 0, 1, 0, 0],
            [3, 4, 0, 2, 4]],
           [[0, 2, 2, 3, 1],
            [3, 2, 2, 2, 3],
            [3, 4, 3, 0, 3],
            [1, 2, 2, 2, 4]]])
      parent_ids = _transpose_batch_time(
          [[[4, 2, 4, 3, 4],
            [3, 4, 0, 2, 0],
            [3, 1, 3, 2, 2],
            [0, 2, 1, 4, 2]],
           [[4, 2, 4, 3, 4],
            [3, 4, 0, 2, 0],
            [3, 1, 3, 2, 2],
            [0, 2, 1, 4, 2]],
           [[3, 0, 0, 4, 0],
            [1, 2, 4, 2, 2],
            [4, 4, 0, 3, 0],
            [2, 4, 4, 3, 0]],
           [[3, 1, 4, 1, 3],
            [3, 2, 4, 0, 4],
            [1, 0, 1, 4, 2],
            [0, 3, 2, 0, 1]]])
      expected_beams = _transpose_batch_time(
          [[[-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1]],
           [[3, 4, 0, 4, 0],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1]],
           [[2, 3, 2, 3, 3],
            [2, 1, 1, 3, 2],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1]],
           [[2, 3, 2, 1, 1],
            [2, 3, 2, 3, 2],
            [3, 4, 3, 0, 3],
            [-1, -1, -1, -1, -1]]])

      beams = beam_search_ops.gather_tree(
          step_ids=step_ids, parent_ids=parent_ids,
          sequence_length=sequence_length)
      self.assertAllEqual(expected_beams, beams.eval())


if __name__ == "__main__":
  test.main()
