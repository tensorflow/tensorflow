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

  def testTrainingConstruction(self):
    input_data = [[-1., 0.], [-1., 2.],  # node 1
                  [1., 0.], [1., -2.]]  # node 2
    input_labels = [0, 1, 2, 3]

    params = tensor_forest.ForestHParams(
        num_classes=4, num_features=2, num_trees=10, max_nodes=1000).fill()

    graph_builder = tensor_forest.RandomForestGraphs(params)
    graph = graph_builder.training_graph(input_data, input_labels)
    self.assertTrue(isinstance(graph, tf.Operation))

  def testInferenceConstruction(self):
    input_data = [[-1., 0.], [-1., 2.],  # node 1
                  [1., 0.], [1., -2.]]  # node 2

    params = tensor_forest.ForestHParams(
        num_classes=4, num_features=2, num_trees=10, max_nodes=1000).fill()

    graph_builder = tensor_forest.RandomForestGraphs(params)
    graph = graph_builder.inference_graph(input_data)
    self.assertTrue(isinstance(graph, tf.Tensor))


if __name__ == '__main__':
  googletest.main()
