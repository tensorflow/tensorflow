# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Classes and functions implementing Metrics SavedModel serialization."""

from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.trackable import data_structures


class MetricSavedModelSaver(layer_serialization.LayerSavedModelSaver):
  """Metric serialization."""

  @property
  def object_identifier(self):
    return constants.METRIC_IDENTIFIER

  def _python_properties_internal(self):
    metadata = dict(
        class_name=generic_utils.get_registered_name(type(self.obj)),
        name=self.obj.name,
        dtype=self.obj.dtype)
    metadata.update(layer_serialization.get_serialized(self.obj))
    if self.obj._build_input_shape is not None:  # pylint: disable=protected-access
      metadata['build_input_shape'] = self.obj._build_input_shape  # pylint: disable=protected-access
    return metadata

  def _get_serialized_attributes_internal(self, unused_serialization_cache):
    return (dict(variables=data_structures.wrap_or_unwrap(self.obj.variables)),
            dict())  # TODO(b/135550038): save functions to enable saving
                     # custom metrics.
