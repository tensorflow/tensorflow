# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for CandidateSamplerOp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class RangeSamplerOpsTest(test.TestCase):

  BATCH_SIZE = 3
  NUM_TRUE = 2
  RANGE = 5
  NUM_SAMPLED = RANGE

  TRUE_LABELS = [[1, 2], [0, 4], [3, 3]]

  def testTrueCandidates(self):
    with self.cached_session() as sess:
      indices = constant_op.constant([0, 0, 1, 1, 2, 2])
      true_candidates_vec = constant_op.constant([1, 2, 0, 4, 3, 3])
      true_candidates_matrix = array_ops.reshape(
          true_candidates_vec, [self.BATCH_SIZE, self.NUM_TRUE])
      indices_val, true_candidates_val = sess.run(
          [indices, true_candidates_matrix])

    self.assertAllEqual(indices_val, [0, 0, 1, 1, 2, 2])
    self.assertAllEqual(true_candidates_val, self.TRUE_LABELS)

  def testSampledCandidates(self):
    with self.cached_session():
      true_classes = constant_op.constant(
          [[1, 2], [0, 4], [3, 3]], dtype=dtypes.int64)
      sampled_candidates, _, _ = candidate_sampling_ops.all_candidate_sampler(
          true_classes, self.NUM_TRUE, self.NUM_SAMPLED, True)
      result = self.evaluate(sampled_candidates)

    expected_ids = [0, 1, 2, 3, 4]
    self.assertAllEqual(result, expected_ids)
    self.assertEqual(sampled_candidates.get_shape(), [self.NUM_SAMPLED])

  def testTrueLogExpectedCount(self):
    with self.cached_session():
      true_classes = constant_op.constant(
          [[1, 2], [0, 4], [3, 3]], dtype=dtypes.int64)
      _, true_expected_count, _ = candidate_sampling_ops.all_candidate_sampler(
          true_classes, self.NUM_TRUE, self.NUM_SAMPLED, True)
      true_log_expected_count = math_ops.log(true_expected_count)
      result = self.evaluate(true_log_expected_count)

    self.assertAllEqual(result, [[0.0] * self.NUM_TRUE] * self.BATCH_SIZE)
    self.assertEqual(true_expected_count.get_shape(),
                     [self.BATCH_SIZE, self.NUM_TRUE])
    self.assertEqual(true_log_expected_count.get_shape(),
                     [self.BATCH_SIZE, self.NUM_TRUE])

  def testSampledLogExpectedCount(self):
    with self.cached_session():
      true_classes = constant_op.constant(
          [[1, 2], [0, 4], [3, 3]], dtype=dtypes.int64)
      _, _, sampled_expected_count = candidate_sampling_ops.all_candidate_sampler(  # pylint: disable=line-too-long
          true_classes, self.NUM_TRUE, self.NUM_SAMPLED, True)
      sampled_log_expected_count = math_ops.log(sampled_expected_count)
      result = self.evaluate(sampled_log_expected_count)

    self.assertAllEqual(result, [0.0] * self.NUM_SAMPLED)
    self.assertEqual(sampled_expected_count.get_shape(), [self.NUM_SAMPLED])
    self.assertEqual(sampled_log_expected_count.get_shape(), [self.NUM_SAMPLED])

  def testAccidentalHits(self):
    with self.cached_session() as sess:
      true_classes = constant_op.constant(
          [[1, 2], [0, 4], [3, 3]], dtype=dtypes.int64)
      sampled_candidates, _, _ = candidate_sampling_ops.all_candidate_sampler(
          true_classes, self.NUM_TRUE, self.NUM_SAMPLED, True)
      accidental_hits = candidate_sampling_ops.compute_accidental_hits(
          true_classes, sampled_candidates, self.NUM_TRUE)
      indices, ids, weights = self.evaluate(accidental_hits)

    self.assertEqual(1, accidental_hits[0].get_shape().ndims)
    self.assertEqual(1, accidental_hits[1].get_shape().ndims)
    self.assertEqual(1, accidental_hits[2].get_shape().ndims)
    for index, id_, weight in zip(indices, ids, weights):
      self.assertTrue(id_ in self.TRUE_LABELS[index])
      self.assertLess(weight, -1.0e37)

  def testSeed(self):

    def draw(seed):
      with self.cached_session():
        true_classes = constant_op.constant(
            [[1, 2], [0, 4], [3, 3]], dtype=dtypes.int64)
        sampled, _, _ = candidate_sampling_ops.log_uniform_candidate_sampler(
            true_classes, self.NUM_TRUE, self.NUM_SAMPLED, True, 5, seed=seed)
        return self.evaluate(sampled)

    # Non-zero seed. Repeatable.
    for seed in [1, 12, 123, 1234]:
      self.assertAllEqual(draw(seed), draw(seed))
    # Seed=0 means random seeds.
    num_same = 0
    for _ in range(10):
      if np.allclose(draw(None), draw(None)):
        num_same += 1
    # Accounts for the fact that the same random seed may be picked
    # twice very rarely.
    self.assertLessEqual(num_same, 2)


if __name__ == "__main__":
  test.main()
