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

# pylint: disable=unused-import
import tensorflow as tf

from tensorflow.contrib.tensor_forest.hybrid.python import hybrid_model
from tensorflow.contrib.tensor_forest.hybrid.python.layers import fully_connected
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class HybridLayerTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.params = tensor_forest.ForestHParams(num_classes=3,
                                              num_features=7,
                                              layer_size=11,
                                              num_layers=13,
                                              num_trees=17,
                                              connection_probability=0.1,
                                              hybrid_tree_depth=4,
                                              regularization_strength=0.01,
                                              regularization="",
                                              weight_init_mean=0.0,
                                              weight_init_std=0.1)
    self.params.num_nodes = 2**self.params.hybrid_tree_depth - 1
    self.params.num_leaves = 2**(self.params.hybrid_tree_depth - 1)

  def testLayerNums(self):
    l1 = fully_connected.FullyConnectedLayer(self.params, 0, None)
    self.assertEquals(l1.layer_num, 0)

    l2 = fully_connected.FullyConnectedLayer(self.params, 1, None)
    self.assertEquals(l2.layer_num, 1)

    l3 = fully_connected.FullyConnectedLayer(self.params, 2, None)
    self.assertEquals(l3.layer_num, 2)


if __name__ == "__main__":
  googletest.main()
