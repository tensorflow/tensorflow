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
"""Tests for tf.contrib.tensor_forest.ops.finished_nodes_op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow  # pylint: disable=unused-import

from tensorflow.contrib.tensor_forest.python.ops import tensor_forest_ops

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class FinishedNodesTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.leaves = [1, 3, 4]
    self.node_map = [-1, -1, -1, 0, 1, -1]
    self.split_sums = [
        # Accumulator 0
        [[3, 0, 3], [2, 1, 1], [3, 1, 2]],
        # Accumulator 1
        [[6, 3, 3], [6, 2, 4], [5, 0, 5]],
        # Accumulator 2
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        # Accumulator 3
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        # Accumulator 4
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    ]
    self.split_squares = []
    self.accumulator_sums = [[6, 3, 3], [11, 4, 7], [0, 0, 0], [0, 0, 0],
                             [0, 0, 0]]
    self.accumulator_squares = []
    self.birth_epochs = [0, 0, 0, 1, 1, 1]
    self.current_epoch = [1]

  def testSimple(self):
    with self.test_session():
      finished, stale = tensor_forest_ops.finished_nodes(
          self.leaves,
          self.node_map,
          self.split_sums,
          self.split_squares,
          self.accumulator_sums,
          self.accumulator_squares,
          self.birth_epochs,
          self.current_epoch,
          regression=False,
          num_split_after_samples=10,
          min_split_samples=10)

      self.assertAllEqual([4], finished.eval())
      self.assertAllEqual([], stale.eval())

  def testLeavesCanBeNegativeOne(self):
    with self.test_session():
      finished, stale = tensor_forest_ops.finished_nodes(
          [-1, -1, 1, -1, 3, -1, -1, 4, -1, -1, -1],
          self.node_map,
          self.split_sums,
          self.split_squares,
          self.accumulator_sums,
          self.accumulator_squares,
          self.birth_epochs,
          self.current_epoch,
          regression=False,
          num_split_after_samples=10,
          min_split_samples=10)

      self.assertAllEqual([4], finished.eval())
      self.assertAllEqual([], stale.eval())

  def testNoAccumulators(self):
    with self.test_session():
      finished, stale = tensor_forest_ops.finished_nodes(
          self.leaves, [-1] * 6,
          self.split_sums,
          self.split_squares,
          self.accumulator_sums,
          self.accumulator_squares,
          self.birth_epochs,
          self.current_epoch,
          regression=False,
          num_split_after_samples=10,
          min_split_samples=10)

      self.assertAllEqual([], finished.eval())
      self.assertAllEqual([], stale.eval())

  def testBadInput(self):
    with self.test_session():
      with self.assertRaisesOpError(
          'leaf_tensor should be one-dimensional'):
        finished, stale = tensor_forest_ops.finished_nodes(
            [self.leaves],
            self.node_map,
            self.split_sums,
            self.split_squares,
            self.accumulator_sums,
            self.accumulator_squares,
            self.birth_epochs,
            self.current_epoch,
            regression=False,
            num_split_after_samples=10,
            min_split_samples=10)

        self.assertAllEqual([], finished.eval())
        self.assertAllEqual([], stale.eval())

  def testEarlyDominatesHoeffding(self):
    with self.test_session():
      finished, stale = tensor_forest_ops.finished_nodes(
          self.leaves,
          self.node_map,
          self.split_sums,
          self.split_squares,
          self.accumulator_sums,
          self.accumulator_squares,
          self.birth_epochs,
          self.current_epoch,
          dominate_method='hoeffding',
          regression=False,
          num_split_after_samples=10,
          min_split_samples=5)

      self.assertAllEqual([4], finished.eval())
      self.assertAllEqual([], stale.eval())

  def testEarlyDominatesBootstrap(self):
    with self.test_session():
      finished, stale = tensor_forest_ops.finished_nodes(
          self.leaves,
          self.node_map,
          self.split_sums,
          self.split_squares,
          self.accumulator_sums,
          self.accumulator_squares,
          self.birth_epochs,
          self.current_epoch,
          dominate_method='bootstrap',
          regression=False,
          num_split_after_samples=10,
          min_split_samples=5,
          random_seed=1)

      self.assertAllEqual([4], finished.eval())
      self.assertAllEqual([], stale.eval())

  def testEarlyDominatesChebyshev(self):
    with self.test_session():
      finished, stale = tensor_forest_ops.finished_nodes(
          self.leaves,
          self.node_map,
          self.split_sums,
          self.split_squares,
          self.accumulator_sums,
          self.accumulator_squares,
          self.birth_epochs,
          self.current_epoch,
          dominate_method='chebyshev',
          regression=False,
          num_split_after_samples=10,
          min_split_samples=5)

      self.assertAllEqual([4], finished.eval())
      self.assertAllEqual([], stale.eval())


if __name__ == '__main__':
  googletest.main()
