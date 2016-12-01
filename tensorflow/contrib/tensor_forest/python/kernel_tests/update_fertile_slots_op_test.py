# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.contrib.tensor_forest.ops.allocate_deallocate_op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow  # pylint: disable=unused-import

from tensorflow.contrib.tensor_forest.python.ops import training_ops

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class UpdateFertileSlotsTest(test_util.TensorFlowTestCase):

  def setUp(self):
    # tree is:
    #         0
    #     1       2
    #   3   4   5   6
    self.finished = [2]
    self.non_fertile_leaves = [3, 4]
    self.non_fertile_leaf_scores = [10., 15.]
    self.end_of_tree = [5]
    self.node_map = [-1, -1, 0, -1, -1, -1, -1]
    self.total_counts = [[80., 40., 40.]]
    self.ops = training_ops.Load()
    self.stale_leaves = []
    self.node_sums = [[3, 1, 2], [4, 2, 2], [5, 2, 3], [6, 1, 5], [7, 5, 2],
                      [8, 4, 4], [9, 7, 2]]

  def testSimple(self):
    with self.test_session():
      (n2a_map_updates, a2n_map_updates, accumulators_cleared,
       accumulators_allocated) = self.ops.update_fertile_slots(
           self.finished, self.non_fertile_leaves, self.non_fertile_leaf_scores,
           self.end_of_tree, self.total_counts, self.node_map,
           self.stale_leaves, self.node_sums)

      self.assertAllEqual([[2, 4], [-1, 0]], n2a_map_updates.eval())
      self.assertAllEqual([[0], [4]], a2n_map_updates.eval())
      self.assertAllEqual([], accumulators_cleared.eval())
      self.assertAllEqual([0], accumulators_allocated.eval())

  def testNoFinished(self):
    with self.test_session():
      (n2a_map_updates, a2n_map_updates, accumulators_cleared,
       accumulators_allocated) = self.ops.update_fertile_slots(
           [], self.non_fertile_leaves, self.non_fertile_leaf_scores,
           self.end_of_tree, self.total_counts, self.node_map,
           self.stale_leaves, self.node_sums)

      self.assertAllEqual((2, 0), n2a_map_updates.eval().shape)
      self.assertAllEqual((2, 0), a2n_map_updates.eval().shape)
      self.assertAllEqual([], accumulators_cleared.eval())
      self.assertAllEqual([], accumulators_allocated.eval())

  def testPureCounts(self):
    with self.test_session():
      self.node_sums[4] = [10, 0, 10]
      (n2a_map_updates, a2n_map_updates, accumulators_cleared,
       accumulators_allocated) = self.ops.update_fertile_slots(
           self.finished, self.non_fertile_leaves, self.non_fertile_leaf_scores,
           self.end_of_tree, self.total_counts, self.node_map,
           self.stale_leaves, self.node_sums)

      self.assertAllEqual([[2, 3], [-1, 0]], n2a_map_updates.eval())
      self.assertAllEqual([[0], [3]], a2n_map_updates.eval())
      self.assertAllEqual([], accumulators_cleared.eval())
      self.assertAllEqual([0], accumulators_allocated.eval())

  def testBadInput(self):
    del self.non_fertile_leaf_scores[-1]
    with self.test_session():
      with self.assertRaisesOpError(
          'Number of non fertile leaves should be the same in '
          'non_fertile_leaves and non_fertile_leaf_scores.'):
        (n2a_map_updates, _, _, _) = self.ops.update_fertile_slots(
            self.finished, self.non_fertile_leaves,
            self.non_fertile_leaf_scores, self.end_of_tree, self.total_counts,
            self.node_map, self.stale_leaves, self.node_sums)
        self.assertAllEqual((2, 0), n2a_map_updates.eval().shape)


if __name__ == '__main__':
  googletest.main()
