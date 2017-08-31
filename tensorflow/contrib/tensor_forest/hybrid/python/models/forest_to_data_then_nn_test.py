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
"""Tests for the hybrid tensor forest model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

# pylint: disable=unused-import

from tensorflow.contrib.tensor_forest.hybrid.python.models import forest_to_data_then_nn
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.framework.ops import Operation
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import googletest


class ForestToDataThenNNTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.params = tensor_forest.ForestHParams(
        num_classes=2,
        num_features=31,
        layer_size=11,
        num_layers=13,
        num_trees=3,
        connection_probability=0.1,
        hybrid_tree_depth=4,
        regularization_strength=0.01,
        regularization="",
        base_random_seed=10,
        feature_bagging_fraction=1.0,
        learning_rate=0.01,
        weight_init_mean=0.0,
        weight_init_std=0.1)
    self.params.regression = False
    self.params.num_nodes = 2**self.params.hybrid_tree_depth - 1
    self.params.num_leaves = 2**(self.params.hybrid_tree_depth - 1)

    self.params.num_features_per_node = (self.params.feature_bagging_fraction *
                                         self.params.num_features)

  def testInferenceConstruction(self):
    # pylint: disable=W0612
    data = constant_op.constant(
        [[random.uniform(-1, 1) for i in range(self.params.num_features)]
         for _ in range(100)])

    with variable_scope.variable_scope(
        "ForestToDataThenNNTest_testInferenceContruction"):
      graph_builder = forest_to_data_then_nn.ForestToDataThenNN(self.params)
      graph = graph_builder.inference_graph(data, None)

      self.assertTrue(isinstance(graph, Tensor))

  def testTrainingConstruction(self):
    # pylint: disable=W0612
    data = constant_op.constant(
        [[random.uniform(-1, 1) for i in range(self.params.num_features)]
         for _ in range(100)])

    labels = [1 for _ in range(100)]

    with variable_scope.variable_scope(
        "ForestToDataThenNNTest.testTrainingContruction"):
      graph_builder = forest_to_data_then_nn.ForestToDataThenNN(self.params)
      graph = graph_builder.training_graph(data, labels, None)

      self.assertTrue(isinstance(graph, Operation))


if __name__ == "__main__":
  googletest.main()
