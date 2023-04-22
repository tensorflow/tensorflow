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
"""Classes and functions implementing Layer SavedModel serialization."""

from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.saving.saved_model import base_serialization
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import save_impl
from tensorflow.python.keras.saving.saved_model import serialized_attributes
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import nest


class LayerSavedModelSaver(base_serialization.SavedModelSaver):
  """Implements Layer SavedModel serialization."""

  @property
  def object_identifier(self):
    return constants.LAYER_IDENTIFIER

  @property
  def python_properties(self):
    # TODO(kathywu): Add python property validator
    return self._python_properties_internal()

  def _python_properties_internal(self):
    """Returns dictionary of all python properties."""
    # TODO(kathywu): Add support for metrics serialization.
    # TODO(kathywu): Synchronize with the keras spec (go/keras-json-spec) once
    # the python config serialization has caught up.
    metadata = dict(
        name=self.obj.name,
        trainable=self.obj.trainable,
        expects_training_arg=self.obj._expects_training_arg,  # pylint: disable=protected-access
        dtype=policy.serialize(self.obj._dtype_policy),  # pylint: disable=protected-access
        batch_input_shape=getattr(self.obj, '_batch_input_shape', None),
        stateful=self.obj.stateful,
        must_restore_from_config=self.obj._must_restore_from_config,  # pylint: disable=protected-access
    )

    metadata.update(get_serialized(self.obj))
    if self.obj.input_spec is not None:
      # Layer's input_spec has already been type-checked in the property setter.
      metadata['input_spec'] = nest.map_structure(
          lambda x: generic_utils.serialize_keras_object(x) if x else None,
          self.obj.input_spec)
    if (self.obj.activity_regularizer is not None and
        hasattr(self.obj.activity_regularizer, 'get_config')):
      metadata['activity_regularizer'] = generic_utils.serialize_keras_object(
          self.obj.activity_regularizer)
    if self.obj._build_input_shape is not None:  # pylint: disable=protected-access
      metadata['build_input_shape'] = self.obj._build_input_shape  # pylint: disable=protected-access
    return metadata

  def objects_to_serialize(self, serialization_cache):
    return (self._get_serialized_attributes(
        serialization_cache).objects_to_serialize)

  def functions_to_serialize(self, serialization_cache):
    return (self._get_serialized_attributes(
        serialization_cache).functions_to_serialize)

  def _get_serialized_attributes(self, serialization_cache):
    """Generates or retrieves serialized attributes from cache."""
    keras_cache = serialization_cache.setdefault(constants.KERAS_CACHE_KEY, {})
    if self.obj in keras_cache:
      return keras_cache[self.obj]

    serialized_attr = keras_cache[self.obj] = (
        serialized_attributes.SerializedAttributes.new(self.obj))

    if (save_impl.should_skip_serialization(self.obj) or
        self.obj._must_restore_from_config):  # pylint: disable=protected-access
      return serialized_attr

    object_dict, function_dict = self._get_serialized_attributes_internal(
        serialization_cache)

    serialized_attr.set_and_validate_objects(object_dict)
    serialized_attr.set_and_validate_functions(function_dict)
    return serialized_attr

  def _get_serialized_attributes_internal(self, serialization_cache):
    """Returns dictionary of serialized attributes."""
    objects = save_impl.wrap_layer_objects(self.obj, serialization_cache)
    functions = save_impl.wrap_layer_functions(self.obj, serialization_cache)
    # Attribute validator requires that the default save signature is added to
    # function dict, even if the value is None.
    functions['_default_save_signature'] = None
    return objects, functions


# TODO(kathywu): Move serialization utils (and related utils from
# generic_utils.py) to a separate file.
def get_serialized(obj):
  with generic_utils.skip_failed_serialization():
    # Store the config dictionary, which may be used when reviving the object.
    # When loading, the program will attempt to revive the object from config,
    # and if that fails, the object will be revived from the SavedModel.
    return generic_utils.serialize_keras_object(obj)


class InputLayerSavedModelSaver(base_serialization.SavedModelSaver):
  """InputLayer serialization."""

  @property
  def object_identifier(self):
    return constants.INPUT_LAYER_IDENTIFIER

  @property
  def python_properties(self):

    return dict(
        class_name=type(self.obj).__name__,
        name=self.obj.name,
        dtype=self.obj.dtype,
        sparse=self.obj.sparse,
        ragged=self.obj.ragged,
        batch_input_shape=self.obj._batch_input_shape,  # pylint: disable=protected-access
        config=self.obj.get_config())

  def objects_to_serialize(self, serialization_cache):
    return {}

  def functions_to_serialize(self, serialization_cache):
    return {}


class RNNSavedModelSaver(LayerSavedModelSaver):
  """RNN layer serialization."""

  @property
  def object_identifier(self):
    return constants.RNN_LAYER_IDENTIFIER

  def _get_serialized_attributes_internal(self, serialization_cache):
    objects, functions = (
        super(RNNSavedModelSaver, self)._get_serialized_attributes_internal(
            serialization_cache))
    states = data_structures.wrap_or_unwrap(self.obj.states)
    # SaveModel require all the objects to be Trackable when saving.
    # If the states is still a tuple after wrap_or_unwrap, it means it doesn't
    # contain any trackable item within it, eg empty tuple or (None, None) for
    # stateless ConvLSTM2D. We convert them to list so that wrap_or_unwrap can
    # make it a Trackable again for saving. When loaded, ConvLSTM2D is
    # able to handle the tuple/list conversion.
    if isinstance(states, tuple):
      states = data_structures.wrap_or_unwrap(list(states))
    objects['states'] = states
    return objects, functions


class IndexLookupLayerSavedModelSaver(LayerSavedModelSaver):
  """Index lookup layer serialization."""

  @property
  def python_properties(self):
    # TODO(kathywu): Add python property validator
    metadata = self._python_properties_internal()
    if metadata['config'].get('has_static_table', False):
      metadata['config']['vocabulary'] = None
    return metadata
