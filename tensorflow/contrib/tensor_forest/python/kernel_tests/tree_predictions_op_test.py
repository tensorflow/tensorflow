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
"""Tests for tf.contrib.tensor_forest.ops.tree_predictions_op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow  # pylint: disable=unused-import

from tensorflow.contrib.tensor_forest.python import constants
from tensorflow.contrib.tensor_forest.python.ops import tensor_forest_ops

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class TreePredictionsTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.data_spec = [constants.DATA_FLOAT]

  def testSimple(self):
    input_data = [[-1., 0.], [-1., 2.],  # node 1
                  [1., 0.], [1., -2.]]  # node 2

    tree = [[1, 0], [-1, 0], [-1, 0]]
    tree_thresholds = [0., 0., 0.]
    node_pcw = [[1.0, 0.3, 0.4, 0.3], [1.0, 0.1, 0.1, 0.8],
                [1.0, 0.5, 0.25, 0.25]]

    with self.test_session():
      predictions = tensor_forest_ops.tree_predictions(
          input_data, [], [], [],
          self.data_spec,
          tree,
          tree_thresholds,
          node_pcw,
          valid_leaf_threshold=1)

      self.assertAllClose([[0.1, 0.1, 0.8], [0.1, 0.1, 0.8],
                           [0.5, 0.25, 0.25], [0.5, 0.25, 0.25]],
                          predictions.eval())

  def testSparseInput(self):
    sparse_shape = [3, 10]
    sparse_indices = [[0, 0], [0, 4], [0, 9],
                      [1, 0], [1, 7],
                      [2, 0]]
    sparse_values = [3.0, -1.0, 0.5,
                     1.5, 6.0,
                     -2.0]
    sparse_data_spec = [constants.DATA_FLOAT]

    tree = [[1, 0], [-1, 0], [-1, 0]]
    tree_thresholds = [0., 0., 0.]
    node_pcw = [[1.0, 0.3, 0.4, 0.3], [1.0, 0.1, 0.1, 0.8],
                [1.0, 0.5, 0.25, 0.25]]

    with self.test_session():
      predictions = tensor_forest_ops.tree_predictions(
          [],
          sparse_indices,
          sparse_values,
          sparse_shape,
          sparse_data_spec,
          tree,
          tree_thresholds,
          node_pcw,
          valid_leaf_threshold=1)

      self.assertAllClose([[0.5, 0.25, 0.25],
                           [0.5, 0.25, 0.25],
                           [0.1, 0.1, 0.8]],
                          predictions.eval())

  def testSparseInputDefaultIsZero(self):
    sparse_shape = [3, 10]
    sparse_indices = [[0, 0], [0, 4], [0, 9],
                      [1, 0], [1, 7],
                      [2, 0]]
    sparse_values = [3.0, -1.0, 0.5,
                     1.5, 6.0,
                     -2.0]
    sparse_data_spec = [constants.DATA_FLOAT] * 10

    tree = [[1, 7], [-1, 0], [-1, 0]]
    tree_thresholds = [3.0, 0., 0.]
    node_pcw = [[1.0, 0.3, 0.4, 0.3], [1.0, 0.1, 0.1, 0.8],
                [1.0, 0.5, 0.25, 0.25]]

    with self.test_session():
      predictions = tensor_forest_ops.tree_predictions(
          [],
          sparse_indices,
          sparse_values,
          sparse_shape,
          sparse_data_spec,
          tree,
          tree_thresholds,
          node_pcw,
          valid_leaf_threshold=1)

      self.assertAllClose([[0.1, 0.1, 0.8],
                           [0.5, 0.25, 0.25],
                           [0.1, 0.1, 0.8]],
                          predictions.eval())

  def testBackoffToParent(self):
    input_data = [[-1., 0.], [-1., 2.],  # node 1
                  [1., 0.], [1., -2.]]  # node 2

    tree = [[1, 0], [-1, 0], [-1, 0]]
    tree_thresholds = [0., 0., 0.]
    node_pcw = [[15.0, 3.0, 9.0, 3.0], [5.0, 1.0, 1.0, 3.0],
                [25.0, 5.0, 20.0, 0.0]]

    with self.test_session():
      predictions = tensor_forest_ops.tree_predictions(
          input_data, [], [], [],
          self.data_spec,
          tree,
          tree_thresholds,
          node_pcw,
          valid_leaf_threshold=10)

      # Node 2 has enough data, but Node 1 needs to combine with the parent
      # counts.
      self.assertAllClose([[0.2, 0.4, 0.4], [0.2, 0.4, 0.4],
                           [0.2, 0.8, 0.0], [0.2, 0.8, 0.0]],
                          predictions.eval())

  def testNoInput(self):
    input_data = []

    tree = [[1, 0], [-1, 0], [-1, 0]]
    tree_thresholds = [0., 0., 0.]
    node_pcw = [[1.0, 0.3, 0.4, 0.3], [1.0, 0.1, 0.1, 0.8],
                [1.0, 0.5, 0.25, 0.25]]

    with self.test_session():
      predictions = tensor_forest_ops.tree_predictions(
          input_data, [], [], [],
          self.data_spec,
          tree,
          tree_thresholds,
          node_pcw,
          valid_leaf_threshold=10)

      self.assertEquals((0, 3), predictions.eval().shape)

  def testBadInput(self):
    input_data = [[-1., 0.], [-1., 2.],  # node 1
                  [1., 0.], [1., -2.]]  # node 2

    tree = [[1, 0], [-1, 0], [-1, 0]]
    tree_thresholds = [0., 0.]  # not enough nodes.
    node_pcw = [[1.0, 0.3, 0.4, 0.3], [1.0, 0.1, 0.1, 0.8],
                [1.0, 0.5, 0.25, 0.25]]

    with self.test_session():
      with self.assertRaisesOpError(
          'Number of nodes should be the same in tree, tree_thresholds '
          'and node_pcw.'):
        predictions = tensor_forest_ops.tree_predictions(
            input_data, [], [], [],
            self.data_spec,
            tree,
            tree_thresholds,
            node_pcw,
            valid_leaf_threshold=10)

        self.assertEquals((0, 3), predictions.eval().shape)


if __name__ == '__main__':
  googletest.main()
