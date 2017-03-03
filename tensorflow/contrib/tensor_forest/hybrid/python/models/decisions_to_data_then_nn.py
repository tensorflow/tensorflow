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
"""A model that places a decision tree embedding before a neural net."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.tensor_forest.hybrid.python import hybrid_model
from tensorflow.contrib.tensor_forest.hybrid.python.layers import decisions_to_data
from tensorflow.contrib.tensor_forest.hybrid.python.layers import fully_connected
from tensorflow.python.training import adagrad


class DecisionsToDataThenNN(hybrid_model.HybridModel):
  """A model that places a decision tree embedding before a neural net."""

  def __init__(self,
               params,
               device_assigner=None,
               optimizer_class=adagrad.AdagradOptimizer,
               **kwargs):
    super(DecisionsToDataThenNN, self).__init__(
        params,
        device_assigner=device_assigner,
        optimizer_class=optimizer_class,
        **kwargs)

    self.layers = [decisions_to_data.DecisionsToDataLayer(params,
                                                          0, device_assigner),
                   fully_connected.FullyConnectedLayer(
                       params, 1, device_assigner=device_assigner)]
