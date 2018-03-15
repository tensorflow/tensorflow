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

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class RoutingFunctionTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.input_data = [[-1., 0.], [-1., 2.],
                       [1., 0.], [1., -2.]]
    self.input_labels = [0., 1., 2., 3.]
    self.tree = [[1, 0], [-1, 0], [-1, 0]]
    self.tree_weights = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    self.tree_thresholds = [0., 0., 0.]
    self.ops = training_ops.Load()

  def testRoutingFunction(self):
    with self.test_session():
      route_tensor = gen_training_ops.routing_function(
          self.input_data, self.tree_weights, self.tree_thresholds, max_nodes=3)

      route_tensor_shape = route_tensor.get_shape()
      self.assertEquals(len(route_tensor_shape), 2)
      self.assertEquals(route_tensor_shape[0], 4)
      self.assertEquals(route_tensor_shape[1], 3)

      routes = route_tensor.eval()

      # Point 1
      # Node 1 is a decision node => probability = 1.0
      self.assertAlmostEquals(1.0, routes[0, 0])
      # Probability left output = 1.0 / (1.0 + exp(1.0)) = 0.26894142
      self.assertAlmostEquals(0.26894142, routes[0, 1])
      # Probability right = 1 - 0.2689414 = 0.73105858
      self.assertAlmostEquals(0.73105858, routes[0, 2])


if __name__ == '__main__':
  googletest.main()
