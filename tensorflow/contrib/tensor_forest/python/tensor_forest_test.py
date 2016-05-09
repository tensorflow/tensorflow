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
"""Tests for tf.contrib.tensor_forest.ops.tensor_forest."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.tensor_forest.python import tensor_forest

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class TensorForestTest(test_util.TensorFlowTestCase):

  def testForestHParams(self):
    hparams = tensor_forest.ForestHParams(
        num_classes=2, num_trees=100, max_nodes=1000,
        split_after_samples=25, num_features=60).fill()
    self.assertEquals(2, hparams.num_classes)
    self.assertEquals(3, hparams.num_output_columns)
    # 2 * ceil(log_2(1000)) = 20
    self.assertEquals(20, hparams.max_depth)
    # sqrt(num_features) < 10, so num_splits_to_consider should be 10.
    self.assertEquals(10, hparams.num_splits_to_consider)
    # Don't have more fertile nodes than max # leaves, which is 500.
    self.assertEquals(500, hparams.max_fertile_nodes)
    # Default value of valid_leaf_threshold
    self.assertEquals(1, hparams.valid_leaf_threshold)
    # split_after_samples is larger than 10
    self.assertEquals(1, hparams.split_initializations_per_input)
    self.assertEquals(0, hparams.base_random_seed)

  def testForestHParamsBigTree(self):
    hparams = tensor_forest.ForestHParams(
        num_classes=2, num_trees=100, max_nodes=1000000,
        split_after_samples=25,
        num_features=1000).fill()
    self.assertEquals(40, hparams.max_depth)
    # sqrt(1000) = 31.63...
    self.assertEquals(32, hparams.num_splits_to_consider)
    # 1000000 / 32 = 31250
    self.assertEquals(31250, hparams.max_fertile_nodes)
    # floor(31.63 / 25) = 1
    self.assertEquals(1, hparams.split_initializations_per_input)

  def testTrainingConstructionClassification(self):
    input_data = [[-1., 0.], [-1., 2.],  # node 1
                  [1., 0.], [1., -2.]]  # node 2
    input_labels = [0, 1, 2, 3]

    params = tensor_forest.ForestHParams(
        num_classes=4, num_features=2, num_trees=10, max_nodes=1000,
        split_after_samples=25).fill()

    graph_builder = tensor_forest.RandomForestGraphs(params)
    graph = graph_builder.training_graph(input_data, input_labels)
    self.assertTrue(isinstance(graph, tf.Operation))

  def testTrainingConstructionRegression(self):
    input_data = [[-1., 0.], [-1., 2.],  # node 1
                  [1., 0.], [1., -2.]]  # node 2
    input_labels = [0, 1, 2, 3]

    params = tensor_forest.ForestHParams(
        num_classes=4, num_features=2, num_trees=10, max_nodes=1000,
        split_after_samples=25, regression=True).fill()

    graph_builder = tensor_forest.RandomForestGraphs(params)
    graph = graph_builder.training_graph(input_data, input_labels)
    self.assertTrue(isinstance(graph, tf.Operation))

  def testInferenceConstruction(self):
    input_data = [[-1., 0.], [-1., 2.],  # node 1
                  [1., 0.], [1., -2.]]  # node 2

    params = tensor_forest.ForestHParams(
        num_classes=4, num_features=2, num_trees=10, max_nodes=1000,
        split_after_samples=25).fill()

    graph_builder = tensor_forest.RandomForestGraphs(params)
    graph = graph_builder.inference_graph(input_data)
    self.assertTrue(isinstance(graph, tf.Tensor))

  def testImpurityConstruction(self):
    params = tensor_forest.ForestHParams(
        num_classes=4, num_features=2, num_trees=10, max_nodes=1000,
        split_after_samples=25).fill()

    graph_builder = tensor_forest.RandomForestGraphs(params)
    graph = graph_builder.average_impurity()
    self.assertTrue(isinstance(graph, tf.Tensor))


if __name__ == '__main__':
  googletest.main()
