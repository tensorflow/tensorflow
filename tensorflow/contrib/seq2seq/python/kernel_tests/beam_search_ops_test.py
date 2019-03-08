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

import itertools

import numpy as np

from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


def _transpose_batch_time(x):
  return np.transpose(x, [1, 0, 2]).astype(np.int32)


class GatherTreeTest(test.TestCase):

  def testGatherTreeOne(self):
    # (max_time = 4, batch_size = 1, beams = 3)
    end_token = 10
    step_ids = _transpose_batch_time(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -1, -1]]])
    parent_ids = _transpose_batch_time(
        [[[0, 0, 0], [0, 1, 1], [2, 1, 2], [-1, -1, -1]]])
    max_sequence_lengths = [3]
    expected_result = _transpose_batch_time([[[2, 2, 2], [6, 5, 6], [7, 8, 9],
                                              [10, 10, 10]]])
    beams = beam_search_ops.gather_tree(
        step_ids=step_ids,
        parent_ids=parent_ids,
        max_sequence_lengths=max_sequence_lengths,
        end_token=end_token)
    with self.cached_session(use_gpu=True):
      self.assertAllEqual(expected_result, self.evaluate(beams))

  def testBadParentValuesOnCPU(self):
    # (batch_size = 1, max_time = 4, beams = 3)
    # bad parent in beam 1 time 1
    end_token = 10
    step_ids = _transpose_batch_time(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -1, -1]]])
    parent_ids = _transpose_batch_time(
        [[[0, 0, 0], [0, -1, 1], [2, 1, 2], [-1, -1, -1]]])
    max_sequence_lengths = [3]
    with ops.device("/cpu:0"):
      with self.assertRaisesOpError(
          r"parent id -1 at \(batch, time, beam\) == \(0, 0, 1\)"):
        beams = beam_search_ops.gather_tree(
            step_ids=step_ids,
            parent_ids=parent_ids,
            max_sequence_lengths=max_sequence_lengths,
            end_token=end_token)
        self.evaluate(beams)

  def testBadParentValuesOnGPU(self):
    # Only want to run this test on CUDA devices, as gather_tree is not
    # registered for SYCL devices.
    if not test.is_gpu_available(cuda_only=True):
      return
    # (max_time = 4, batch_size = 1, beams = 3)
    # bad parent in beam 1 time 1; appears as a negative index at time 0
    end_token = 10
    step_ids = _transpose_batch_time(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -1, -1]]])
    parent_ids = _transpose_batch_time(
        [[[0, 0, 0], [0, -1, 1], [2, 1, 2], [-1, -1, -1]]])
    max_sequence_lengths = [3]
    expected_result = _transpose_batch_time([[[2, -1, 2], [6, 5, 6], [7, 8, 9],
                                              [10, 10, 10]]])
    with ops.device("/device:GPU:0"):
      beams = beam_search_ops.gather_tree(
          step_ids=step_ids,
          parent_ids=parent_ids,
          max_sequence_lengths=max_sequence_lengths,
          end_token=end_token)
      self.assertAllEqual(expected_result, self.evaluate(beams))

  def testGatherTreeBatch(self):
    batch_size = 10
    beam_width = 15
    max_time = 8
    max_sequence_lengths = [0, 1, 2, 4, 7, 8, 9, 10, 11, 0]
    end_token = 5

    with self.cached_session(use_gpu=True):
      step_ids = np.random.randint(
          0, high=end_token + 1, size=(max_time, batch_size, beam_width))
      parent_ids = np.random.randint(
          0, high=beam_width - 1, size=(max_time, batch_size, beam_width))

      beams = beam_search_ops.gather_tree(
          step_ids=step_ids.astype(np.int32),
          parent_ids=parent_ids.astype(np.int32),
          max_sequence_lengths=max_sequence_lengths,
          end_token=end_token)

      self.assertEqual((max_time, batch_size, beam_width), beams.shape)
      beams_value = self.evaluate(beams)
      for b in range(batch_size):
        # Past max_sequence_lengths[b], we emit all end tokens.
        b_value = beams_value[max_sequence_lengths[b]:, b, :]
        self.assertAllClose(b_value, end_token * np.ones_like(b_value))
      for batch, beam in itertools.product(
          range(batch_size), range(beam_width)):
        v = np.squeeze(beams_value[:, batch, beam])
        if end_token in v:
          found_bad = np.where(v == -1)[0]
          self.assertEqual(0, len(found_bad))
          found = np.where(v == end_token)[0]
          found = found[0]  # First occurrence of end_token.
          # If an end_token is found, everything before it should be a
          # valid id and everything after it should be -1.
          if found > 0:
            self.assertAllEqual(
                v[:found - 1] >= 0, np.ones_like(v[:found - 1], dtype=bool))
          self.assertAllClose(v[found + 1:],
                              end_token * np.ones_like(v[found + 1:]))


if __name__ == "__main__":
  test.main()
