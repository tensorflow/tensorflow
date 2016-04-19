# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Tests for tf.contrib.tensor_forest.ops.count_extremely_random_stats."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.tensor_forest.python.ops import training_ops

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class CountExtremelyRandomStatsTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.input_data = [[-1., 0.], [-1., 2.],  # node 1
                       [1., 0.], [1., -2.]]  # node 2
    self.input_labels = [0, 1, 2, 3]
    self.tree = [[1, 0], [-1, 0], [-1, 0]]
    self.tree_thresholds = [0., 0., 0.]
    self.node_map = [-1, 0, -1]
    self.split_features = [[1], [-1]]
    self.split_thresholds = [[1.], [0.]]
    self.ops = training_ops.Load()

  def testSimple(self):
    with self.test_session():
      (pcw_node, pcw_splits_indices, pcw_splits_delta, pcw_totals_indices,
       pcw_totals_delta, leaves) = (
           self.ops.count_extremely_random_stats(
               self.input_data, self.input_labels, self.tree,
               self.tree_thresholds, self.node_map,
               self.split_features, self.split_thresholds, num_classes=4))

      self.assertAllEqual(
          [[1., 1., 1., 1.], [1., 1., 0., 0.], [0., 0., 1., 1.]],
          pcw_node.eval())
      self.assertAllEqual([[0, 0, 0]], pcw_splits_indices.eval())
      self.assertAllEqual([1.], pcw_splits_delta.eval())
      self.assertAllEqual([[0, 1], [0, 0]], pcw_totals_indices.eval())
      self.assertAllEqual([1., 1.], pcw_totals_delta.eval())
      self.assertAllEqual([1, 1, 2, 2], leaves.eval())

  def testThreaded(self):
    with self.test_session(
        config=tf.ConfigProto(intra_op_parallelism_threads=2)):
      (pcw_node, pcw_splits_indices, pcw_splits_delta, pcw_totals_indices,
       pcw_totals_delta,
       leaves) = (self.ops.count_extremely_random_stats(self.input_data,
                                                        self.input_labels,
                                                        self.tree,
                                                        self.tree_thresholds,
                                                        self.node_map,
                                                        self.split_features,
                                                        self.split_thresholds,
                                                        num_classes=4))

      self.assertAllEqual([[1., 1., 1., 1.], [1., 1., 0., 0.],
                           [0., 0., 1., 1.]],
                          pcw_node.eval())
      self.assertAllEqual([[0, 0, 0]], pcw_splits_indices.eval())
      self.assertAllEqual([1.], pcw_splits_delta.eval())
      self.assertAllEqual([[0, 1], [0, 0]], pcw_totals_indices.eval())
      self.assertAllEqual([1., 1.], pcw_totals_delta.eval())
      self.assertAllEqual([1, 1, 2, 2], leaves.eval())

  def testNoAccumulators(self):
    with self.test_session():
      (pcw_node, pcw_splits_indices, pcw_splits_delta, pcw_totals_indices,
       pcw_totals_delta, leaves) = (
           self.ops.count_extremely_random_stats(
               self.input_data, self.input_labels, self.tree,
               self.tree_thresholds, [-1] * 3,
               self.split_features, self.split_thresholds, num_classes=4))

      self.assertAllEqual([[1., 1., 1., 1.], [1., 1., 0., 0.],
                           [0., 0., 1., 1.]],
                          pcw_node.eval())
      self.assertEquals((0, 3), pcw_splits_indices.eval().shape)
      self.assertAllEqual([], pcw_splits_delta.eval())
      self.assertEquals((0, 2), pcw_totals_indices.eval().shape)
      self.assertAllEqual([], pcw_totals_delta.eval())
      self.assertAllEqual([1, 1, 2, 2], leaves.eval())

  def testBadInput(self):
    del self.node_map[-1]

    with self.test_session():
      with self.assertRaisesOpError(
          'Number of nodes should be the same in '
          'tree, tree_thresholds, and node_to_accumulator'):
        pcw_node, _, _, _, _, _ = (
            self.ops.count_extremely_random_stats(
                self.input_data, self.input_labels, self.tree,
                self.tree_thresholds, self.node_map,
                self.split_features, self.split_thresholds, num_classes=4))

        self.assertAllEqual([], pcw_node.eval())


if __name__ == '__main__':
  googletest.main()
