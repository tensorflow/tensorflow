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
"""Defines the layer abstraction for hybrid models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.ops import variables as framework_variables


class HybridLayer(object):
  """Layers are building blocks for hybrid models."""

  def _define_vars(self,
                   params,
                   **kwargs):
    """Override to define the TensorFlow variables for the layer."""
    raise NotImplementedError

  # pylint: disable=unused-argument
  def __init__(self, params, layer_num, device_assigner, *args, **kwargs):
    self.layer_num = layer_num
    self.device_assigner = (
        device_assigner or framework_variables.VariableDeviceChooser())
    self.params = params
    self._define_vars(params, **kwargs)

  def inference_graph(self, data, data_spec=None):
    raise NotImplementedError
