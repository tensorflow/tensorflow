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
"""Tests for the routing function op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.tensor_forest.hybrid.ops import gen_training_ops
from tensorflow.contrib.tensor_forest.hybrid.python.ops import training_ops
from tensorflow.contrib.tensor_forest.python import tensor_forest

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class KFeatureRoutingFunctionTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.input_data = [[-1., 0.], [-1., 2.],
                       [1., 0.], [1., -2.]]
    self.input_labels = [0., 1., 2., 3.]
    self.tree = [[1, 0], [-1, 0], [-1, 0]]
    self.tree_weights = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    self.tree_thresholds = [0., 0., 0.]

    self.ops = training_ops.Load()

    self.params = tensor_forest.ForestHParams(
        num_features=2,
        hybrid_tree_depth=2,
        base_random_seed=10,
        feature_bagging_fraction=1.0,
        regularization_strength=0.01,
        regularization="",
        weight_init_mean=0.0,
        weight_init_std=0.1)
    self.params.num_nodes = 2**self.params.hybrid_tree_depth - 1
    self.params.num_leaves = 2**(self.params.hybrid_tree_depth - 1)
    self.params.num_features_per_node = (
        self.params.feature_bagging_fraction * self.params.num_features)
    self.params.regression = False

  def testParams(self):
    self.assertEquals(self.params.num_nodes, 3)
    self.assertEquals(self.params.num_features, 2)
    self.assertEquals(self.params.num_features_per_node, 2)

  def testRoutingFunction(self):
    with self.test_session():
      route_tensor = gen_training_ops.k_feature_routing_function(
          self.input_data,
          self.tree_weights,
          self.tree_thresholds,
          max_nodes=self.params.num_nodes,
          num_features_per_node=self.params.num_features_per_node,
          layer_num=0,
          random_seed=self.params.base_random_seed)

      route_tensor_shape = route_tensor.get_shape()
      self.assertEquals(len(route_tensor_shape), 2)
      self.assertEquals(route_tensor_shape[0], 4)
      self.assertEquals(route_tensor_shape[1], 3)

      routes = route_tensor.eval()
      print(routes)

      # Point 1
      # Node 1 is a decision node => probability = 1.0
      self.assertAlmostEquals(1.0, routes[0, 0])
      # Probability left output = 1.0 / (1.0 + exp(1.0)) = 0.26894142
      self.assertAlmostEquals(0.26894142, routes[0, 1])
      # Probability right = 1 - 0.2689414 = 0.73105858
      self.assertAlmostEquals(0.73105858, routes[0, 2])


if __name__ == "__main__":
  googletest.main()
