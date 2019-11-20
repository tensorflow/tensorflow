# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Keras SavedModel deserialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from tensorflow.python.eager import function as defun
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import utils
from tensorflow.python.keras.saving.saved_model.serialized_attributes import CommonEndpoints
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.saved_model import load as tf_load
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking.tracking import delete_tracking
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.lazy_loader import LazyLoader

# To avoid circular dependencies between keras/engine and keras/saving,
# code in keras/saving must delay imports.

# TODO(b/134426265): Switch back to single-quotes to match the rest of the file
# once the issue with copybara is fixed.
# pylint:disable=g-inconsistent-quotes
models_lib = LazyLoader("models_lib", globals(),
                        "tensorflow.python.keras.models")
base_layer = LazyLoader(
    "base_layer", globals(),
    "tensorflow.python.keras.engine.base_layer")
input_layer = LazyLoader(
    "input_layer", globals(),
    "tensorflow.python.keras.engine.input_layer")
network_lib = LazyLoader(
    "network_lib", globals(),
    "tensorflow.python.keras.engine.network")
training_lib = LazyLoader(
    "training_lib", globals(),
    "tensorflow.python.keras.engine.training")
# pylint:enable=g-inconsistent-quotes


PUBLIC_ATTRIBUTES = CommonEndpoints.all_functions.union(
    CommonEndpoints.all_checkpointable_objects)
PUBLIC_ATTRIBUTES.add(constants.KERAS_ATTR)


def load(path, compile=True):  # pylint: disable=redefined-builtin
  """Loads Keras objects from a SavedModel.

  Any Keras layer or model saved to the SavedModel will be loaded back
  as Keras objects. Other objects are loaded as regular trackable objects (same
  as `tf.saved_model.load`).

  Currently, Keras saving/loading only retains the Keras object's weights,
  losses, and call function.

  The loaded model can be re-compiled, but the original optimizer, compiled loss
  functions, and metrics are not retained. This is temporary, and `model.save`
  will soon be able to serialize compiled models.

  Args:
    path: Path to SavedModel.
    compile: If true, compile the model after loading it.

  Returns:
    Object loaded from SavedModel.
  """
  # TODO(kathywu): Add saving/loading of optimizer, compiled losses and metrics.
  # TODO(kathywu): Add code to load from objects that contain all endpoints
  model = tf_load.load_internal(path, loader_cls=KerasObjectLoader)

  # pylint: disable=protected-access
  if isinstance(model, training_lib.Model) and compile:
    # TODO(kathywu): Use compiled objects from SavedModel, instead of
    # creating new objects from the training config.
    training_config = model._serialized_attributes['metadata'].get(
        'training_config', None)
    if training_config is not None:
      model.compile(**saving_utils.compile_args_from_training_config(
          training_config))
  # pylint: disable=protected-access

  return model


def _is_graph_network(node):
  # pylint: disable=protected-access
  return (
      isinstance(node, RevivedNetwork) and
      node._serialized_attributes['metadata'].get('is_graph_network', False) and
      hasattr(node, '_config'))
  # pylint: enable=protected-access


class KerasObjectLoader(tf_load.Loader):
  """Loader that recreates Keras objects."""

  def __init__(self, *args, **kwargs):
    super(KerasObjectLoader, self).__init__(*args, **kwargs)
    self._finalize()

  def _finalize(self):
    # pylint: disable=protected-access

    # Set up call functions for all layers (skip this step for Sequential and
    # Functional models).
    for node in self._nodes:
      if isinstance(node, RevivedLayer):
        node.built = True
        is_graph_network = _is_graph_network(node)
        if not (isinstance(node, models_lib.Sequential) or is_graph_network):
          if hasattr(node.keras_api, 'call_and_return_conditional_losses'):
            node.call = utils.use_wrapped_call(
                node, node.keras_api.call_and_return_conditional_losses,
                return_method=True)
            node._init_call_fn_args()

    for node in self._nodes:
      if isinstance(node, RevivedNetwork):
        call_fn = node.keras_api.call_and_return_conditional_losses
        if call_fn.input_signature is None:
          inputs = infer_inputs_from_restored_call_function(call_fn)
        else:
          inputs = call_fn.input_signature[0]

        # Set model inputs and outputs.
        is_graph_network = _is_graph_network(node)
        if isinstance(node, models_lib.Sequential):
          with trackable.no_automatic_dependency_tracking_scope(node):
            node._layers = []
          for layer in node.keras_api.layers:
            node.add(layer)
        elif is_graph_network:
          # Reconstruct functional model from the config and layers loaded
          # from the SavedModel.
          inputs, outputs, _ = network_lib.reconstruct_from_config(
              node.get_config(),
              created_layers={layer.name: layer for layer in node.layers})
          node._init_graph_network(
              inputs, outputs,
              name=node._serialized_attributes['metadata']['name'])
          # Set the metadata attributes once more, since _init_graph_network
          # resets these attributes.
          _set_network_attributes_from_metadata(node)
        else:  # Model is subclassed.
          node._set_inputs(inputs)

      # Add unconditional losses.
      if isinstance(node, RevivedLayer):
        if hasattr(node.keras_api, 'layer_regularization_losses'):
          losses = getattr(node.keras_api, 'layer_regularization_losses', [])
        else:
          # Some earlier SavedModels may not have layer_regularization_losses
          # serialized separately. Fall back to using the regularization_losses
          # list if it does not exist.
          losses = node._serialized_attributes.get('regularization_losses', [])
        for loss in losses:
          node.add_loss(loss)

        # Use wrapped activity regularizer function if the layer's activity
        # regularizer wasn't created during initialization.
        if node.activity_regularizer is None:
          node.activity_regularizer = getattr(node.keras_api,
                                              'activity_regularizer_fn', None)

        # Now that the node object has been fully loaded and restored from the,
        # checkpoint, the object no longer needs to track objects added from
        # SerializedAttributes. (Note that saving a training checkpoint still
        # functions correctly, because layers and variables are tracked
        # separately by the Layer object.)
        # TODO(kathywu): Instead of outright deleting these nodes (which would
        # make restoring from a different checkpoint tricky), mark them as extra
        # dependencies that are OK to overwrite.
        for name in PUBLIC_ATTRIBUTES:
          delete_tracking(node, name)

    # pylint: enable=protected-access

  def _recreate_base_user_object(self, proto):
    revived_classes = {
        '_tf_keras_layer': (RevivedLayer, base_layer.Layer),
        '_tf_keras_input_layer': (RevivedInputLayer, input_layer.InputLayer),
        '_tf_keras_network': (RevivedNetwork, network_lib.Network),
        '_tf_keras_model': (RevivedNetwork, training_lib.Model),
        '_tf_keras_sequential': (RevivedNetwork, models_lib.Sequential)
    }

    parent_classes = revived_classes.get(proto.identifier, None)

    if parent_classes is not None:
      parent_classes = revived_classes[proto.identifier]
      metadata = json.loads(proto.metadata)
      revived_cls = type(
          compat.as_str(metadata['class_name']),
          parent_classes,
          {'__setattr__': parent_classes[1].__setattr__})
      return revived_cls._init_from_metadata(metadata)  # pylint: disable=protected-access

    return super(KerasObjectLoader, self)._recreate_base_user_object(proto)


# TODO(kathywu): Centrally define keys and functions for both  serialization and
# deserialization.
class RevivedLayer(object):
  """Keras layer loaded from a SavedModel."""

  @classmethod
  def _init_from_metadata(cls, metadata):
    """Create revived layer from metadata stored in the SavedModel proto."""
    init_args = dict(
        name=metadata['name'],
        trainable=metadata['trainable'])
    if metadata.get('dtype') is not None:
      init_args['dtype'] = metadata['dtype']
    if metadata.get('batch_input_shape') is not None:
      init_args['batch_input_shape'] = metadata['batch_input_shape']

    revived_obj = cls(**init_args)

    with trackable.no_automatic_dependency_tracking_scope(revived_obj):
      # pylint:disable=protected-access
      revived_obj._expects_training_arg = metadata['expects_training_arg']
      if metadata.get('config') is not None:
        revived_obj._config = metadata['config']
      if metadata.get('input_spec') is not None:
        revived_obj.input_spec = recursively_deserialize_keras_object(
            metadata['input_spec'],
            module_objects={'InputSpec': input_spec.InputSpec})
      if metadata.get('activity_regularizer') is not None:
        revived_obj.activity_regularizer = regularizers.deserialize(
            metadata['activity_regularizer'])
      if metadata.get('_is_feature_layer') is not None:
        revived_obj._is_feature_layer = metadata['_is_feature_layer']

      # Store attributes revived from SerializedAttributes in a un-tracked
      # dictionary. The attributes are the ones listed in CommonEndpoints or
      # "keras_api" for keras-specific attributes.
      revived_obj._serialized_attributes = {'metadata': metadata}
      # pylint:enable=protected-access

    return revived_obj, _revive_setter

  @property
  def keras_api(self):
    return self._serialized_attributes.get(constants.KERAS_ATTR, None)

  def get_config(self):
    if hasattr(self, '_config'):
      return self._config
    else:
      raise NotImplementedError


def _revive_setter(layer, name, value):
  """Reattaches attributes from the SavedModel to the newly revived object."""
  if name in PUBLIC_ATTRIBUTES:
    # pylint: disable=protected-access
    if isinstance(value, trackable.Trackable):
      layer._track_trackable(value, name=name)
    layer._serialized_attributes[name] = value
    # pylint: enable=protected-access
  else:
    setattr(layer, name, value)


class RevivedInputLayer(object):
  """InputLayer loaded from a SavedModel."""

  @classmethod
  def _init_from_metadata(cls, metadata):
    """Revives the saved InputLayer from the Metadata."""
    init_args = dict(
        name=metadata['name'],
        dtype=metadata['dtype'],
        sparse=metadata['sparse'],
        ragged=metadata['ragged'],
        batch_input_shape=metadata['batch_input_shape'])
    revived_obj = cls(**init_args)
    with trackable.no_automatic_dependency_tracking_scope(revived_obj):
      revived_obj._config = metadata['config']  # pylint:disable=protected-access

    return revived_obj, setattr

  def get_config(self):
    return self._config


def recursively_deserialize_keras_object(config, module_objects=None):
  """Deserialize Keras object from a nested structure."""
  if isinstance(config, dict):
    if 'class_name' in config:
      return deserialize_keras_object(config, module_objects=module_objects)
    else:
      return {key: recursively_deserialize_keras_object(config[key],
                                                        module_objects)
              for key in config}
  if isinstance(config, (tuple, list)):
    return [recursively_deserialize_keras_object(x, module_objects)
            for x in config]
  else:
    raise ValueError('Unable to decode config: {}'.format(config))


def infer_inputs_from_restored_call_function(fn):
  """Returns TensorSpec of inputs from a restored call function.

  Args:
    fn: Restored layer call function. It is assumed that the inputs are entirely
      in the first argument.

  Returns:
    TensorSpec of call function inputs.
  """
  def common_spec(x, y):
    return tensor_spec.TensorSpec(defun.common_shape(x.shape, y.shape),
                                  x.dtype, x.name)
  spec = fn.concrete_functions[0].structured_input_signature[0][0]
  for concrete in fn.concrete_functions[1:]:
    spec2 = concrete.structured_input_signature[0][0]
    spec = nest.map_structure(common_spec, spec, spec2)
  return spec


class RevivedNetwork(RevivedLayer):
  """Keras network of layers loaded from a SavedModel."""

  @classmethod
  def _init_from_metadata(cls, metadata):
    """Create revived network from metadata stored in the SavedModel proto."""
    revived_obj = cls(name=metadata['name'])

    # Store attributes revived from SerializedAttributes in a un-tracked
    # dictionary. The attributes are the ones listed in CommonEndpoints or
    # "keras_api" for keras-specific attributes.
    with trackable.no_automatic_dependency_tracking_scope(revived_obj):
      # pylint:disable=protected-access
      revived_obj._serialized_attributes = {'metadata': metadata}
      _set_network_attributes_from_metadata(revived_obj)
      # pylint:enable=protected-access

    return revived_obj, _revive_setter  # pylint:disable=protected-access


def _set_network_attributes_from_metadata(revived_obj):
  """Sets attributes recorded in the metadata."""
  with trackable.no_automatic_dependency_tracking_scope(revived_obj):
    # pylint:disable=protected-access
    metadata = revived_obj._serialized_attributes['metadata']
    if metadata.get('dtype') is not None:
      revived_obj._set_dtype_policy(metadata['dtype'])
    revived_obj.trainable = metadata['trainable']

    revived_obj._expects_training_arg = metadata['expects_training_arg']
    if metadata.get('config') is not None:
      revived_obj._config = metadata['config']

    if metadata.get('activity_regularizer') is not None:
      revived_obj.activity_regularizer = regularizers.deserialize(
          metadata['activity_regularizer'])
    # pylint:enable=protected-access

