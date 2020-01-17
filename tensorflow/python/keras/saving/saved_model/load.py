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
import re

from tensorflow.python.eager import function as defun
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import utils
from tensorflow.python.keras.saving.saved_model.serialized_attributes import CommonEndpoints
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load as tf_load
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import revived_types
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
layers_module = LazyLoader(
    "layers_module", globals(),
    "tensorflow.python.keras.layers")
input_layer = LazyLoader(
    "input_layer", globals(),
    "tensorflow.python.keras.engine.input_layer")
network_lib = LazyLoader(
    "network_lib", globals(),
    "tensorflow.python.keras.engine.network")
training_lib = LazyLoader(
    "training_lib", globals(),
    "tensorflow.python.keras.engine.training")
training_lib_v1 = LazyLoader(
    "training_lib_v1", globals(),
    "tensorflow.python.keras.engine.training_v1")
# pylint:enable=g-inconsistent-quotes


PUBLIC_ATTRIBUTES = CommonEndpoints.all_functions.union(
    CommonEndpoints.all_checkpointable_objects)
PUBLIC_ATTRIBUTES.add(constants.KERAS_ATTR)


KERAS_OBJECT_IDENTIFIERS = (
    '_tf_keras_layer', '_tf_keras_input_layer', '_tf_keras_network',
    '_tf_keras_model', '_tf_keras_sequential')


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
  # pylint: enable=protected-access

  return model


def _is_graph_network(layer):
  """Determines whether the layer is a graph network."""
  # pylint: disable=protected-access
  if isinstance(layer, RevivedNetwork):
    return False
  elif isinstance(layer, network_lib.Network):
    return (layer._is_graph_network or
            isinstance(layer, models_lib.Sequential))
  return False


class KerasObjectLoader(tf_load.Loader):
  """Loader that recreates Keras objects (e.g. layers, models).

  Layers and models are revived from either the config or SavedModel following
  these rules:
  1. If object is a graph network (i.e. Sequential or Functional) then it will
     be initialized using the structure from the config only after the children
     layers have been created. Graph networks must be initialized with inputs
     and outputs, so all child layers must be created beforehand.
  2. If object's config exists and the class can be found, then revive from
     config.
  3. Object may have already been created if its parent was revived from config.
     In this case, do nothing.
  4. If nothing of the above applies, compose the various artifacts from the
     SavedModel to create a subclassed layer or model. At this time, custom
     metrics are not supported.

  """

  def __init__(self, *args, **kwargs):
    # Maps node id -> (node, revive setter function)
    # Nodes recreated from the config may generate other nodes. This list
    # records all nodes that were generated directly/indirectly from the config,
    # so that they do not get recreated multiple times.
    self._nodes_recreated_from_config = {}
    # Store all node ids that have already been traversed when tracking nodes
    # that were recreated from the config.
    self._traversed_nodes_from_config = []

    # Maps model id -> (blank model obj, list of child layer or their node ids)
    # This tracks all layers in functional and sequential models. These models
    # are only reconstructed after all of their child layers have been created.
    self.model_layer_dependencies = {}
    self._models_to_reconstruct = []

    super(KerasObjectLoader, self).__init__(*args, **kwargs)

  def _load_all(self):
    """Reconstruct the object graph from the SavedModel."""
    # Load layer and model objects from either config or SavedModel. The objects
    # loaded from config may create variables / other objects during
    # initialization. These are recorded in `_nodes_recreated_from_config`.
    self._layer_nodes = self._load_layers()

    # Load all other nodes and functions.
    super(KerasObjectLoader, self)._load_all()

    # Finish setting up layers and models. See function docstring for more info.
    self._finalize_objects()

    # Now that the node object has been fully loaded, the object no longer needs
    # to track objects added from SerializedAttributes. (Note that saving a
    # training checkpoint still functions correctly, because layers and
    # variables are tracked separately by the Layer object.)
    # TODO(kathywu): Instead of outright deleting these nodes (which would
    # make restoring from a different checkpoint tricky), mark them as extra
    # dependencies that are OK to overwrite.
    for node in self._nodes:
      if not isinstance(node, base_layer.Layer):
        continue
      for name in PUBLIC_ATTRIBUTES:
        delete_tracking(node, name)

  @property
  def _expect_partial_checkpoint(self):
    return True

  def _recreate(self, proto, node_id):
    """Creates a Python object from a SavedObject protocol buffer."""
    if node_id in self._layer_nodes:
      return self._layer_nodes[node_id]
    if node_id in self._nodes_recreated_from_config:
      obj, setter = self._nodes_recreated_from_config[node_id]

      # Overwrite variable names with the ones saved in the SavedModel.
      if proto.WhichOneof('kind') == 'variable' and proto.variable.name:
        obj._handle_name = proto.variable.name + ':0'  # pylint: disable=protected-access
    else:
      obj, setter = super(KerasObjectLoader, self)._recreate(proto, node_id)
    return obj, setter

  def _add_children_recreated_from_config(self, obj, proto, node_id):
    """Recursively records objects recreated from config."""
    # pylint: disable=protected-access
    if node_id in self._traversed_nodes_from_config:
      return
    self._traversed_nodes_from_config.append(node_id)
    obj._maybe_initialize_trackable()
    for reference in proto.children:
      obj_child = obj._lookup_dependency(reference.local_name)
      setter = setattr
      if not isinstance(obj_child, trackable.Trackable):
        continue
      if obj_child._object_identifier in revived_types.registered_identifiers():
        setter = lambda *unused: None
      elif obj_child._object_identifier in KERAS_OBJECT_IDENTIFIERS:
        metadata = self._proto.nodes[reference.node_id].user_object.metadata
        setter = _revive_setter
        _add_serialized_attributes(obj_child, json.loads(metadata))
        # pylint: enable=protected-access
      if (reference.node_id in self._nodes_recreated_from_config and
          self._nodes_recreated_from_config[reference.node_id][0] is not
          obj_child):
        # This means that the same trackable object is referenced by two
        # different objects that were recreated from the config.
        logging.warn('Looks like there is an object (perhaps variable or layer)'
                     ' that is shared between different layers/models. This '
                     'may cause issues when training the model. Object: {}'
                     .format(obj_child))
      self._nodes_recreated_from_config[reference.node_id] = obj_child, setter
      self._add_children_recreated_from_config(
          obj_child, self._proto.nodes[reference.node_id], reference.node_id)

  def _load_layers(self):
    layers = {}
    for node_id, proto in enumerate(self._proto.nodes):
      if (proto.WhichOneof('kind') == 'user_object' and
          proto.user_object.identifier in KERAS_OBJECT_IDENTIFIERS):
        layers[node_id] = self._load_layer(proto.user_object, node_id)
    return layers

  def _load_layer(self, proto, node_id):
    """Load a single layer from a SavedUserObject proto."""
    # Detect whether this object can be revived from the config. If not, then
    # revive from the SavedModel instead.
    metadata = json.loads(proto.metadata)
    obj, setter = self._revive_from_config(metadata, node_id)
    if obj is None:
      obj, setter = revive_custom_object(proto.identifier, metadata)

    if setter == _revive_setter:
      # Add an attribute that stores the extra functions/objects saved in the
      # SavedModel. Most of these functions/objects are ignored, but some are
      # used later in the loading process (e.g. the list of regularization
      # losses, or the training config of compiled models).
      _add_serialized_attributes(obj, metadata)
    return obj, setter

  def _revive_from_config(self, metadata, node_id):
    """Revives a layer/model from config, or returns None."""
    obj = (self._revive_graph_network(metadata, node_id) or
           self._revive_layer_from_config(metadata, node_id))
    if obj is None:
      return None, None

    setter = _revive_setter
    self._nodes_recreated_from_config[node_id] = obj, setter
    self._add_children_recreated_from_config(
        obj, self._proto.nodes[node_id], node_id)
    return obj, setter

  def _revive_graph_network(self, metadata, node_id):
    """Revives a graph network from config."""
    class_name = compat.as_str(metadata['class_name'])
    config = metadata.get('config')

    # Determine whether the metadata contains information for reviving a
    # functional or Sequential model.
    model_is_functional_or_sequential = (
        metadata.get('is_graph_network', False) or
        metadata['class_name'] == 'Sequential')
    if (generic_utils.LAYER_UNDEFINED_CONFIG_KEY in config or
        not model_is_functional_or_sequential):
      return None  # Revive as custom model.

    # Revive functional and sequential models as blank model objects for now (
    # must be initialized to enable setattr tracking and attribute caching).
    # Reconstruction of the network is deferred until all of the model's layers
    # have been revived.
    if class_name == 'Sequential':
      model = models_lib.Sequential(name=config['name'])
    else:
      model = models_lib.Model(name=config['name'])

    # Record this model and its layers. This will later be used to reconstruct
    # the model.
    layers = self._get_child_layer_node_ids(node_id, model.name)
    self.model_layer_dependencies[node_id] = (model, layers)

    return model

  def _revive_layer_from_config(self, metadata, node_id):
    """Revives a layer from config, or returns None if infeasible."""
    # Check that the following requirements are met for reviving from config:
    #    1. Object can be deserialized from config.
    #    2. If the object needs to be built, then the build input shape can be
    #       found.
    class_name = metadata.get('class_name')
    config = metadata.get('config')
    if config is None or generic_utils.LAYER_UNDEFINED_CONFIG_KEY in config:
      return None

    try:
      obj = layers_module.deserialize(
          generic_utils.serialize_keras_class_and_config(class_name, config))
    except ValueError:
      return None

    # Use the dtype, name, and trainable status. Often times these are not
    # specified in custom configs, so retrieve their values from the metadata.
    # pylint: disable=protected-access
    obj._name = metadata['name']
    if metadata.get('trainable') is not None:
      obj.trainable = metadata['trainable']
    if metadata.get('dtype') is not None:
      obj._set_dtype_policy(metadata['dtype'])
    # pylint: enable=protected-access

    input_shape = None
    if not isinstance(obj, input_layer.InputLayer):
      input_shape = self._infer_inputs(node_id, convert_to_shapes=True)
      if input_shape is None:
        return None
    obj.build(input_shape)
    obj.built = True

    return obj

  def _load_edges(self):
    """Add edges for all nodes that are not waiting on initialization."""
    for node_id, proto in enumerate(self._proto.nodes):
      if node_id not in self.model_layer_dependencies:
        self._add_object_graph_edges(proto, node_id)

  def _finalize_objects(self):
    """Finish setting up Keras objects.

    This function is executed after all objects and functions have been created.
    Call functions and losses are attached to each layer, and once all layers
    have been fully set up, graph networks are initialized.

    Subclassed models that are revived from the SavedModel are treated like
    layers, and have their call/loss functions attached here.
    """
    # Finish setting up layers and subclassed models. This step attaches call
    # functions and losses to each object, and sets model inputs/outputs.
    layers_revived_from_config = []
    layers_revived_from_saved_model = []
    for node_id, node in enumerate(self._nodes):
      if (not isinstance(node, base_layer.Layer) or
          # Don't finalize models until all layers have finished loading.
          node_id in self.model_layer_dependencies):
        continue

      # No need to apply the finalizing steps to input layers.
      if isinstance(node, input_layer.InputLayer):
        self._unblock_model_reconstruction(node_id, node)
        continue

      if node_id in self._nodes_recreated_from_config:
        layers_revived_from_config.append(node)
      else:
        layers_revived_from_saved_model.append(node)
      self._unblock_model_reconstruction(node_id, node)
    _finalize_saved_model_layers(layers_revived_from_saved_model)
    _finalize_config_layers(layers_revived_from_config)

    # Initialize graph networks, now that layer dependencies have been resolved.
    self._reconstruct_all_models()

  def _unblock_model_reconstruction(self, layer_id, layer):
    """Removes layer from blocking model reconstruction."""
    for model_id, v in self.model_layer_dependencies.items():
      _, layers = v
      if layer_id not in layers:
        continue
      layers[layers.index(layer_id)] = layer
      if all(isinstance(x, base_layer.Layer) for x in layers):
        self._models_to_reconstruct.append(model_id)

  def _reconstruct_all_models(self):
    all_initialized_models = set()
    while self._models_to_reconstruct:
      model_id = self._models_to_reconstruct.pop(0)
      all_initialized_models.add(model_id)
      model, layers = self.model_layer_dependencies[model_id]
      self._reconstruct_model(model_id, model, layers)
      self._add_object_graph_edges(self._proto.nodes[model_id], model_id)
      _finalize_config_layers([model])

    if all_initialized_models != set(self.model_layer_dependencies.keys()):
      # This should not happen.
      uninitialized_model_ids = (
          set(self.model_layer_dependencies.keys()) - all_initialized_models)
      uninitialized_model_names = [
          self.model_layer_dependencies[model_id][0].name
          for model_id in uninitialized_model_ids]
      raise ValueError('Error when loading from SavedModel -- the following '
                       'models could not be initialized: {}'
                       .format(uninitialized_model_names))

  def _reconstruct_model(self, model_id, model, layers):
    config = (
        json.loads(self._proto.nodes[model_id].user_object.metadata)['config'])
    if isinstance(model, models_lib.Sequential):
      if not isinstance(layers[0], input_layer.InputLayer):
        if 'batch_input_shape' in config['layers'][0]['config']:
          batch_input_shape = config['layers'][0]['config']['batch_input_shape']
          layers.insert(0, input_layer.InputLayer(
              input_shape=batch_input_shape[1:],
              batch_size=batch_input_shape[0],
              dtype=layers[0].dtype))
      model.__init__(layers, name=config['name'])
      if not model.inputs:
        first_layer = self._get_child_layer_node_ids(model_id, model.name)[0]
        input_shape = self._infer_inputs(first_layer)
        model._set_inputs(input_shape)  # pylint: disable=protected-access
    else:
      (inputs, outputs, created_layers) = network_lib.reconstruct_from_config(
          config, created_layers={layer.name: layer for layer in layers})
      model.__init__(inputs, outputs, name=config['name'])
      network_lib.connect_ancillary_layers(model, created_layers)

    # Set model dtype and trainable status.
    _set_network_attributes_from_metadata(model)

    # Unblock models that are dependent on this model.
    self._unblock_model_reconstruction(model_id, model)

  def _get_child_layer_node_ids(self, node_id, name):
    # First, retrieve the node.keras_api.layers attribute, which is a list of
    # all the layers in the node.
    keras_attr = self._search_for_child_node(node_id, constants.KERAS_ATTR,
                                             name)
    layers_node = self._search_for_child_node(keras_attr, 'layers', name)
    return [node.node_id for node in self._proto.nodes[layers_node].children]

  def _search_for_child_node(self, node_id, child_name, debugging_name):
    for child in self._proto.nodes[node_id].children:
      if child.local_name == child_name:
        return child.node_id
    raise ValueError(
        'Error when loading {}: could not find attribute {}.\n'
        'Most likely this object was serialized incorrectly.'
        .format(debugging_name, child_name))

  def _infer_inputs(self, layer_node_id, convert_to_shapes=False):
    """Infers input shape of layer from SavedModel functions."""
    coder = nested_structure_coder.StructureCoder()
    try:
      call_fn_id = self._search_for_child_node(
          layer_node_id, 'call_and_return_all_conditional_losses', None)
    except ValueError:
      return None

    concrete_functions = (
        self._proto.nodes[call_fn_id].function.concrete_functions)
    if not concrete_functions:
      return None
    call_fn_name = concrete_functions[0]
    call_fn_proto = self._proto.concrete_functions[call_fn_name]
    structured_input_signature = coder.decode_proto(
        call_fn_proto.canonicalized_input_signature)
    inputs = structured_input_signature[0][0]
    if convert_to_shapes:
      return nest.map_structure(lambda spec: spec.shape, inputs)
    else:
      return inputs


def _finalize_saved_model_layers(layers):
  """Runs the final steps of loading Keras Layers from SavedModel."""
  # pylint: disable=protected-access
  # 1. Set up call functions for all layers (skip this step for Sequential and
  # Functional models).
  for layer in layers:
    layer.built = True
    if hasattr(_get_keras_attr(layer), 'call_and_return_conditional_losses'):
      layer.call = utils.use_wrapped_call(
          layer, _get_keras_attr(layer).call_and_return_conditional_losses,
          return_method=True)
      layer._init_call_fn_args()

  for layer in layers:
    # 2. Set model inputs and outputs.
    if isinstance(layer, RevivedNetwork):
      _set_network_attributes_from_metadata(layer)

      call_fn = _get_keras_attr(layer).call_and_return_conditional_losses
      if call_fn.input_signature is None:
        inputs = infer_inputs_from_restored_call_function(call_fn)
      else:
        inputs = call_fn.input_signature[0]
      layer._set_inputs(inputs)

    # 3. Add losses that aren't generated by the layer.call function.
    _restore_layer_unconditional_losses(layer)
    _restore_layer_activation_loss(layer)
  # pylint: enable=protected-access


def _finalize_config_layers(layers):
  """Runs the final steps of loading Keras Layers from config."""
  for layer in layers:
    # It is assumed that layers define their unconditional losses after being
    # recreated from the config and built. The exceptions to this
    # are Functional and Sequential models, which only store conditional losses
    # (losses dependent on the inputs) in the config. Unconditional losses like
    # weight regularization must be revived from the SavedModel.
    if _is_graph_network(layer):
      _restore_layer_unconditional_losses(layer)

    # Some layers, like Dense, record their activation loss function in the
    # config. However, not all layers do this, so the activation loss may be
    # missing when restored from the config/hdf5.
    # TODO(kathywu): Investigate ways to improve the config to ensure consistent
    # loading behavior between HDF5 and SavedModel.
    _restore_layer_activation_loss(layer)


def _restore_layer_unconditional_losses(layer):
  """Restore unconditional losses from SavedModel."""
  if hasattr(_get_keras_attr(layer), 'layer_regularization_losses'):
    losses = getattr(_get_keras_attr(layer), 'layer_regularization_losses', [])
  else:
    # Some earlier SavedModels may not have layer_regularization_losses
    # serialized separately. Fall back to using the regularization_losses
    # list if it does not exist.
    losses = layer._serialized_attributes.get('regularization_losses', [])  # pylint: disable=protected-access
  for loss in losses:
    layer.add_loss(loss)


def _restore_layer_activation_loss(layer):
  """Restore actiation loss from SavedModel."""
  # Use wrapped activity regularizer function if the layer's activity
  # regularizer wasn't created during initialization.
  activity_regularizer = getattr(_get_keras_attr(layer),
                                 'activity_regularizer_fn', None)
  if activity_regularizer and not layer.activity_regularizer:
    try:
      layer.activity_regularizer = activity_regularizer
    except AttributeError:
      # This may happen if a layer wrapper is saved with an activity
      # regularizer. The wrapper object's activity regularizer is unsettable.
      pass


def revive_custom_object(identifier, metadata):
  """Revives object from SavedModel."""
  if ops.executing_eagerly_outside_functions():
    model_class = training_lib.Model
  else:
    model_class = training_lib_v1.Model

  revived_classes = {
      '_tf_keras_layer': (RevivedLayer, base_layer.Layer),
      '_tf_keras_input_layer': (RevivedInputLayer, input_layer.InputLayer),
      '_tf_keras_network': (RevivedNetwork, network_lib.Network),
      '_tf_keras_model': (RevivedNetwork, model_class),
      '_tf_keras_sequential': (RevivedNetwork, models_lib.Sequential)
  }

  parent_classes = revived_classes.get(identifier, None)

  if parent_classes is not None:
    parent_classes = revived_classes[identifier]
    revived_cls = type(
        compat.as_str(metadata['class_name']), parent_classes, {})
    return revived_cls._init_from_metadata(metadata)  # pylint: disable=protected-access


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
  """Setter function that saves some attributes to separate dictionary."""
  # Many attributes in the SavedModel conflict with properties defined in
  # Layer and Model. Save these attributes to a separate dictionary.
  if name in PUBLIC_ATTRIBUTES:
    # pylint: disable=protected-access
    if isinstance(value, trackable.Trackable):
      layer._track_trackable(value, name=name)
    layer._serialized_attributes[name] = value
    # pylint: enable=protected-access
  elif (isinstance(layer, network_lib.Network) and
        re.match(r'^layer(_with_weights)?-[\d+]', name) is not None):
    # Edges named "layer-n" or "layer_with_weights-n", which are tracked in
    # network._track_layers, should not be added as an attribute.
    pass
  elif getattr(layer, name, None) is not None:
    # Don't overwrite already defined attributes.
    pass
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
      return generic_utils.deserialize_keras_object(
          config, module_objects=module_objects)
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
      revived_obj._expects_training_arg = metadata['expects_training_arg']
      if metadata.get('config') is not None:
        revived_obj._config = metadata['config']

      if metadata.get('activity_regularizer') is not None:
        revived_obj.activity_regularizer = regularizers.deserialize(
            metadata['activity_regularizer'])
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
    # pylint:enable=protected-access


def _add_serialized_attributes(layer, metadata):
  # Store attributes revived from SerializedAttributes in a un-tracked
  # dictionary. The attributes are the ones listed in CommonEndpoints or
  # "keras_api" for keras-specific attributes.
  with trackable.no_automatic_dependency_tracking_scope(layer):
    layer._serialized_attributes = {'metadata': metadata}  # pylint: disable=protected-access


def _get_keras_attr(layer):
  return getattr(layer, '_serialized_attributes', {}).get(constants.KERAS_ATTR,
                                                          None)
