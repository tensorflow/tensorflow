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
"""Classes and functions implementing to Model SavedModel serialization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import network_serialization
from tensorflow.python.keras.saving.saved_model import save_impl


class ModelSavedModelSaver(network_serialization.NetworkSavedModelSaver):
  """Model SavedModel serialization."""

  @property
  def object_identifier(self):
    return '_tf_keras_model'

  def _python_properties_internal(self):
    metadata = super(ModelSavedModelSaver, self)._python_properties_internal()
    metadata.update(
        saving_utils.model_metadata(
            self.obj, include_optimizer=True, require_config=False))
    return metadata

  def _get_serialized_attributes_internal(self, serialization_cache):
    default_signature = None

    # Create a default signature function if this is the only object in the
    # cache (i.e. this is the root level object).
    if len(serialization_cache[constants.KERAS_CACHE_KEY]) == 1:
      default_signature = save_impl.default_save_signature(self.obj)

    # Other than the default signature function, all other attributes match with
    # the ones serialized by Layer.
    objects, functions = (
        super(ModelSavedModelSaver, self)._get_serialized_attributes_internal(
            serialization_cache))
    functions['_default_save_signature'] = default_signature
    return objects, functions


class SequentialSavedModelSaver(ModelSavedModelSaver):

  @property
  def object_identifier(self):
    return '_tf_keras_sequential'
