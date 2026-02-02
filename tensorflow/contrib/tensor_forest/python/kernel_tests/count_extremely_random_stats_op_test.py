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
"""Tests for tf.contrib.tensor_forest.ops.count_extremely_random_stats."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.tensor_forest.python import constants
from tensorflow.contrib.tensor_forest.python.ops import training_ops

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class CountExtremelyRandomStatsClassificationTest(test_util.TensorFlowTestCase):

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
    self.epochs = [0, 1, 1]
    self.current_epoch = [1]
    self.data_spec = [constants.DATA_FLOAT]

  def testSimple(self):
    with self.test_session():
      (pcw_node_sums, _, pcw_splits_indices, pcw_splits_sums, _,
       pcw_totals_indices, pcw_totals_sums, _, leaves) = (
           self.ops.count_extremely_random_stats(
               self.input_data, [], [], [], self.data_spec, self.input_labels,
               self.tree, self.tree_thresholds, self.node_map,
               self.split_features, self.split_thresholds, self.epochs,
               self.current_epoch,
               num_classes=5, regression=False))

      self.assertAllEqual(
          [[4., 1., 1., 1., 1.], [2., 1., 1., 0., 0.], [2., 0., 0., 1., 1.]],
          pcw_node_sums.eval())
      self.assertAllEqual([[0, 0, 0], [0, 0, 1]], pcw_splits_indices.eval())
      self.assertAllEqual([1., 1.], pcw_splits_sums.eval())
      self.assertAllEqual([[0, 2], [0, 0], [0, 1]], pcw_totals_indices.eval())
      self.assertAllEqual([1., 2., 1.], pcw_totals_sums.eval())
      self.assertAllEqual([1, 1, 2, 2], leaves.eval())

  def testMissingLabel(self):
    labels = [0, 1, -1, 3]
    with self.test_session():
      (pcw_node_sums, _, pcw_splits_indices, pcw_splits_sums, _,
       pcw_totals_indices, pcw_totals_sums, _,
       leaves) = (self.ops.count_extremely_random_stats(self.input_data, [], [],
                                                        [],
                                                        self.data_spec,
                                                        labels,
                                                        self.tree,
                                                        self.tree_thresholds,
                                                        self.node_map,
                                                        self.split_features,
                                                        self.split_thresholds,
                                                        self.epochs,
                                                        self.current_epoch,
                                                        num_classes=5,
                                                        regression=False))

      self.assertAllEqual(
          [[3., 1., 1., 0., 1.], [2., 1., 1., 0., 0.], [1., 0., 0., 0., 1.]],
          pcw_node_sums.eval())
      self.assertAllEqual([[0, 0, 0], [0, 0, 1]], pcw_splits_indices.eval())
      self.assertAllEqual([1., 1.], pcw_splits_sums.eval())
      self.assertAllEqual([[0, 2], [0, 0], [0, 1]], pcw_totals_indices.eval())
      self.assertAllEqual([1., 2., 1.], pcw_totals_sums.eval())
      self.assertAllEqual([1, 1, 2, 2], leaves.eval())

  def testSparseInput(self):
    sparse_shape = [4, 10]
    sparse_indices = [[0, 0], [0, 4], [0, 9],
                      [1, 0], [1, 7],
                      [2, 0],
                      [3, 1], [3, 4]]
    sparse_values = [3.0, -1.0, 0.5,
                     1.5, 6.0,
                     -2.0,
                     -0.5, 2.0]
    with self.test_session():
      (pcw_node_sums, _, pcw_splits_indices, pcw_splits_sums, _,
       pcw_totals_indices, pcw_totals_sums, _, leaves) = (
           self.ops.count_extremely_random_stats(
               [], sparse_indices, sparse_values, sparse_shape, self.data_spec,
               self.input_labels, self.tree,
               self.tree_thresholds, self.node_map,
               self.split_features, self.split_thresholds, self.epochs,
               self.current_epoch,
               num_classes=5, regression=False))

      self.assertAllEqual(
          [[4., 1., 1., 1., 1.],
           [2., 0., 0., 1., 1.],
           [2., 1., 1., 0., 0.]],
          pcw_node_sums.eval())
      self.assertAllEqual([[0, 0, 4], [0, 0, 0], [0, 0, 3]],
                          pcw_splits_indices.eval())
      self.assertAllEqual([1., 2., 1.], pcw_splits_sums.eval())
      self.assertAllEqual([[0, 4], [0, 0], [0, 3]], pcw_totals_indices.eval())
      self.assertAllEqual([1., 2., 1.], pcw_totals_sums.eval())
      self.assertAllEqual([2, 2, 1, 1], leaves.eval())

  def testFutureEpoch(self):
    current_epoch = [3]
    with self.test_session():
      (pcw_node_sums, _, _, pcw_splits_sums, _,
       _, pcw_totals_sums, _, leaves) = (
           self.ops.count_extremely_random_stats(
               self.input_data, [], [], [], self.data_spec, self.input_labels,
               self.tree, self.tree_thresholds, self.node_map,
               self.split_features, self.split_thresholds, self.epochs,
               current_epoch, num_classes=5, regression=False))

      self.assertAllEqual(
          [[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]],
          pcw_node_sums.eval())
      self.assertAllEqual([], pcw_splits_sums.eval())
      self.assertAllEqual([], pcw_totals_sums.eval())
      self.assertAllEqual([1, 1, 2, 2], leaves.eval())

  def testThreaded(self):
    with self.test_session(
        config=tf.ConfigProto(intra_op_parallelism_threads=2)):
      (pcw_node_sums, _, pcw_splits_indices, pcw_splits_sums, _,
       pcw_totals_indices, pcw_totals_sums, _, leaves) = (
           self.ops.count_extremely_random_stats(
               self.input_data, [], [], [], self.data_spec, self.input_labels,
               self.tree, self.tree_thresholds, self.node_map,
               self.split_features,
               self.split_thresholds, self.epochs, self.current_epoch,
               num_classes=5, regression=False))

      self.assertAllEqual([[4., 1., 1., 1., 1.], [2., 1., 1., 0., 0.],
                           [2., 0., 0., 1., 1.]],
                          pcw_node_sums.eval())
      self.assertAllEqual([[0, 0, 0], [0, 0, 1]], pcw_splits_indices.eval())
      self.assertAllEqual([1., 1.], pcw_splits_sums.eval())
      self.assertAllEqual([[0, 2], [0, 0], [0, 1]], pcw_totals_indices.eval())
      self.assertAllEqual([1., 2., 1.], pcw_totals_sums.eval())
      self.assertAllEqual([1, 1, 2, 2], leaves.eval())

  def testNoAccumulators(self):
    with self.test_session():
      (pcw_node_sums, _, pcw_splits_indices, pcw_splits_sums, _,
       pcw_totals_indices, pcw_totals_sums, _, leaves) = (
           self.ops.count_extremely_random_stats(
               self.input_data, [], [], [], self.data_spec, self.input_labels,
               self.tree, self.tree_thresholds, [-1] * 3,
               self.split_features, self.split_thresholds, self.epochs,
               self.current_epoch, num_classes=5, regression=False))

      self.assertAllEqual([[4., 1., 1., 1., 1.], [2., 1., 1., 0., 0.],
                           [2., 0., 0., 1., 1.]],
                          pcw_node_sums.eval())
      self.assertEquals((0, 3), pcw_splits_indices.eval().shape)
      self.assertAllEqual([], pcw_splits_sums.eval())
      self.assertEquals((0, 2), pcw_totals_indices.eval().shape)
      self.assertAllEqual([], pcw_totals_sums.eval())
      self.assertAllEqual([1, 1, 2, 2], leaves.eval())

  def testBadInput(self):
    del self.node_map[-1]

    with self.test_session():
      with self.assertRaisesOpError(
          'Number of nodes should be the same in '
          'tree, tree_thresholds, node_to_accumulator, and birth_epoch.'):
        pcw_node, _, _, _, _, _, _, _, _ = (
            self.ops.count_extremely_random_stats(
                self.input_data, [], [], [], self.data_spec, self.input_labels,
                self.tree, self.tree_thresholds, self.node_map,
                self.split_features, self.split_thresholds, self.epochs,
                self.current_epoch, num_classes=5, regression=False))

        self.assertAllEqual([], pcw_node.eval())


class CountExtremelyRandomStatsRegressionTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.input_data = [[-1., 0.], [-1., 2.],  # node 1
                       [1., 0.], [1., -2.]]  # node 2
    self.input_labels = [[3.], [6.], [2.], [3.]]
    self.tree = [[1, 0], [-1, 0], [-1, 0]]
    self.tree_thresholds = [0., 0., 0.]
    self.node_map = [-1, 0, -1]
    self.split_features = [[1], [-1]]
    self.split_thresholds = [[1.], [0.]]
    self.ops = training_ops.Load()
    self.epochs = [0, 1, 1]
    self.current_epoch = [1]
    self.data_spec = [constants.DATA_FLOAT] * 2

  def testSimple(self):
    with self.test_session():
      (pcw_node_sums, pcw_node_squares, pcw_splits_indices, pcw_splits_sums,
       pcw_splits_squares, pcw_totals_indices,
       pcw_totals_sums, pcw_totals_squares, leaves) = (
           self.ops.count_extremely_random_stats(
               self.input_data, [], [], [], self.data_spec, self.input_labels,
               self.tree, self.tree_thresholds, self.node_map,
               self.split_features, self.split_thresholds, self.epochs,
               self.current_epoch, num_classes=2, regression=True))

      self.assertAllEqual(
          [[4., 14.], [2., 9.], [2., 5.]], pcw_node_sums.eval())
      self.assertAllEqual(
          [[4., 58.], [2., 45.], [2., 13.]], pcw_node_squares.eval())
      self.assertAllEqual([[0, 0]], pcw_splits_indices.eval())
      self.assertAllEqual([[1., 3.]], pcw_splits_sums.eval())
      self.assertAllEqual([[1., 9.]], pcw_splits_squares.eval())
      self.assertAllEqual([[0]], pcw_totals_indices.eval())
      self.assertAllEqual([[2., 9.]], pcw_totals_sums.eval())
      self.assertAllEqual([[2., 45.]], pcw_totals_squares.eval())
      self.assertAllEqual([1, 1, 2, 2], leaves.eval())


if __name__ == '__main__':
  googletest.main()
