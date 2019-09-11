# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Classes and functions implementing to Network SavedModel serialization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.saving.saved_model import layer_serialization


# Network serialization is pretty much the same as layer serialization.
class NetworkSavedModelSaver(layer_serialization.LayerSavedModelSaver):

  @property
  def object_identifier(self):
    return '_tf_keras_network'

  def _python_properties_internal(self):
    metadata = super(NetworkSavedModelSaver, self)._python_properties_internal()
    metadata['is_graph_network'] = self.obj._is_graph_network  # pylint: disable=protected-access
    return metadata
