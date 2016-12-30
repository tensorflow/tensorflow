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
"""A model that combines a decision forest embedding with a neural net."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.tensor_forest.hybrid.python import hybrid_model
from tensorflow.contrib.tensor_forest.hybrid.python.layers import decisions_to_data
from tensorflow.contrib.tensor_forest.hybrid.python.layers import fully_connected
from tensorflow.python.training import adagrad


class ForestToDataThenNN(hybrid_model.HybridModel):
  """A model that combines a decision forest embedding with a neural net."""

  def __init__(self,
               params,
               device_assigner=None,
               optimizer_class=adagrad.AdagradOptimizer,
               **kwargs):
    super(ForestToDataThenNN, self).__init__(
        params,
        device_assigner=device_assigner,
        optimizer_class=optimizer_class,
        **kwargs)

    self.layers = [[decisions_to_data.KFeatureDecisionsToDataLayer(
        params, i, device_assigner)
                    for i in range(self.params.num_trees)],
                   fully_connected.FullyConnectedLayer(
                       params,
                       self.params.num_trees,
                       device_assigner=device_assigner)]
