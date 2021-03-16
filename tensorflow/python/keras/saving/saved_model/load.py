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

import os
import re
import types

from google.protobuf import message

from tensorflow.core.framework import versions_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.protobuf import saved_metadata_pb2
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.saving.saved_model import utils
from tensorflow.python.keras.saving.saved_model.serialized_attributes import CommonEndpoints
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load as tf_load
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import revived_types
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import compat
from tensorflow.python.util import nest

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
functional_lib = LazyLoader(
    "functional_lib", globals(),
    "tensorflow.python.keras.engine.functional")
training_lib = LazyLoader(
    "training_lib", globals(),
    "tensorflow.python.keras.engine.training")
training_lib_v1 = LazyLoader(
    "training_lib_v1", globals(),
    "tensorflow.python.keras.engine.training_v1")
metrics = LazyLoader("metrics", globals(),
                     "tensorflow.python.keras.metrics")
recurrent = LazyLoader(
    "recurrent", globals(),
    "tensorflow.python.keras.layers.recurrent")
# pylint:enable=g-inconsistent-quotes


PUBLIC_ATTRIBUTES = CommonEndpoints.all_functions.union(
    CommonEndpoints.all_checkpointable_objects)
PUBLIC_ATTRIBUTES.add(constants.KERAS_ATTR)


def load(path, compile=True, options=None):  # pylint: disable=redefined-builtin
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
    options: Optional `tf.saved_model.LoadOptions` object that specifies
      options for loading from SavedModel.


  Returns:
    Object loaded from SavedModel.
  """
  # TODO(kathywu): Add saving/loading of optimizer, compiled losses and metrics.
  # TODO(kathywu): Add code to load from objects that contain all endpoints

  # Look for metadata file or parse the SavedModel
  metadata = saved_metadata_pb2.SavedMetadata()
  meta_graph_def = loader_impl.parse_saved_model(path).meta_graphs[0]
  object_graph_def = meta_graph_def.object_graph_def
  path_to_metadata_pb = os.path.join(path, constants.SAVED_METADATA_PATH)
  if gfile.Exists(path_to_metadata_pb):
    try:
      with gfile.GFile(path_to_metadata_pb, 'rb') as f:
        file_content = f.read()
      metadata.ParseFromString(file_content)
    except message.DecodeError as e:
      raise IOError('Cannot parse keras metadata {}: {}.'
                    .format(path_to_metadata_pb, str(e)))
  else:
    logging.warning('SavedModel saved prior to TF 2.5 detected when loading '
                    'Keras model. Please ensure that you are saving the model '
                    'with model.save() or tf.keras.models.save_model(), *NOT* '
                    'tf.saved_model.save(). To confirm, there should be a file '
                    'named "keras_metadata.pb" in the SavedModel directory.')
    _read_legacy_metadata(object_graph_def, metadata)

  if not metadata.nodes:
    # When there are no Keras objects, return the results from the core loader
    return tf_load.load(path, options=options)

  # Recreate layers and metrics using the info stored in the metadata.
  keras_loader = KerasObjectLoader(metadata, object_graph_def)
  keras_loader.load_layers(compile=compile)

  # Generate a dictionary of all loaded nodes.
  nodes_to_load = {'root': None}
  for node_id, loaded_node in keras_loader.loaded_nodes.items():
    nodes_to_load[keras_loader.get_path(node_id)] = loaded_node
  loaded = tf_load.load_partial(path, nodes_to_load, options=options)

  # Finalize the loaded layers and remove the extra tracked dependencies.
  keras_loader.finalize_objects()
  keras_loader.del_tracking()

  model = loaded['root']

  # pylint: disable=protected-access
  if isinstance(model, training_lib.Model) and compile:
    # TODO(kathywu): Use compiled objects from SavedModel, instead of
    # creating new objects from the training config.
    training_config = model._serialized_attributes['metadata'].get(
        'training_config', None)
    if training_config is not None:
      model.compile(**saving_utils.compile_args_from_training_config(
          training_config), from_serialized=True)
      saving_utils.try_build_compiled_arguments(model)
    else:
      logging.warning('No training configuration found in save file, so the '
                      'model was *not* compiled. Compile it manually.')
  # pylint: enable=protected-access

  # Force variables and resources to initialize.
  if not context.executing_eagerly():
    sess = backend.get_session()  # Variables are initialized by this call.
    sess.run(ops.get_collection(ops.GraphKeys.TABLE_INITIALIZERS))

  return model


def _read_legacy_metadata(object_graph_def, metadata):
  """Builds a KerasMetadata proto from the SavedModel ObjectGraphDef."""
  # Older SavedModels store the metadata directly in the proto instead of the
  # separate pb file.
  node_paths = _generate_object_paths(object_graph_def)
  for node_id, proto in enumerate(object_graph_def.nodes):
    if (proto.WhichOneof('kind') == 'user_object' and
        proto.user_object.identifier in constants.KERAS_OBJECT_IDENTIFIERS):
      metadata.nodes.add(
          node_id=node_id,
          node_path=node_paths[node_id],
          version=versions_pb2.VersionDef(
              producer=1, min_consumer=1, bad_consumers=[]),
          identifier=proto.user_object.identifier,
          metadata=proto.user_object.metadata)


def _generate_object_paths(object_graph_def):
  """Traverses through an ObjectGraphDef and builds a map of all node paths."""
  paths = {0: 'root'}
  nodes_to_visit = [0]

  while nodes_to_visit:
    current_node = nodes_to_visit.pop()
    current_path = paths[current_node]
    for reference in object_graph_def.nodes[current_node].children:
      if reference.node_id in paths:
        continue
      paths[reference.node_id] = '{}.{}'.format(current_path,
                                                reference.local_name)
      nodes_to_visit.append(reference.node_id)

  return paths


def _is_graph_network(layer):
  """Determines whether the layer is a graph network."""
  # pylint: disable=protected-access
  if isinstance(layer, RevivedNetwork):
    return False
  elif isinstance(layer, functional_lib.Functional):
    return (layer._is_graph_network or
            isinstance(layer, models_lib.Sequential))
  return False


class KerasObjectLoader(object):
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

  def __init__(self, metadata, object_graph_def):
    self._metadata = metadata
    self._proto = object_graph_def

    self._node_paths = {node_data.node_id: node_data.node_path
                        for node_data in metadata.nodes}
    self.loaded_nodes = {}  # Maps node path -> loaded node

    # Store all node ids that have already been traversed when tracking nodes
    # that were recreated from the config.
    self._traversed_nodes_from_config = set()

    # Maps model id -> (blank model obj, list of child layer or their node ids)
    # This tracks all layers in functional and sequential models. These models
    # are only reconstructed after all of their child layers have been created.
    self.model_layer_dependencies = {}
    self._models_to_reconstruct = []

  def del_tracking(self):
    """Removes tracked references that are only used when loading the model."""
    # Now that the node object has been fully loaded, and the checkpoint has
    # been restored, the object no longer needs to track objects added from
    # SerializedAttributes. (Note that saving a training checkpoint still
    # functions correctly, because layers and variables are tracked separately
    # by the Layer object.)
    # TODO(kathywu): Instead of outright deleting these nodes (which would
    # make restoring from a different checkpoint tricky), mark them as extra
    # dependencies that are OK to overwrite.
    for node in self.loaded_nodes.values():
      node = node[0]
      if not isinstance(node, base_layer.Layer):
        # Loaded nodes can contain other trackable objects created when
        # loading layers from the config, such as variables.
        continue
      for name in PUBLIC_ATTRIBUTES:
        node._delete_tracking(name)  # pylint: disable=protected-access

      if isinstance(node, functional_lib.Functional):
        # Delete the temporary layer dependencies, which were used to restore
        # the checkpointed values. When the model is live, the user can delete
        # or add layers to the model at any time, so these layer dependencies
        # may be obsolete.
        dependencies = list(node._self_unconditional_dependency_names)  # pylint: disable=protected-access
        for name in dependencies:
          if re.match(r'^layer(_with_weights)?-[\d+]', name) is not None:
            node._delete_tracking(name)  # pylint: disable=protected-access

  def _add_children_recreated_from_config(self, obj, proto, node_id):
    """Recursively records objects recreated from config."""
    # pylint: disable=protected-access
    if node_id in self._traversed_nodes_from_config:
      return

    parent_path = self._node_paths[node_id]
    self._traversed_nodes_from_config.add(node_id)
    obj._maybe_initialize_trackable()
    if isinstance(obj, base_layer.Layer) and not obj.built:
      metadata = json_utils.decode(proto.user_object.metadata)
      self._try_build_layer(obj, node_id, metadata.get('build_input_shape'))

    # Create list of all possible children
    children = []
    # Look for direct children
    for reference in proto.children:
      obj_child = obj._lookup_dependency(reference.local_name)
      children.append((obj_child, reference.node_id, reference.local_name))

    # Add metrics that may have been added to the layer._metrics list.
    # This is stored in the SavedModel as layer.keras_api.layer_metrics in
    # SavedModels created after Tf 2.2.
    metric_list_node_id = self._search_for_child_node(
        node_id, [constants.KERAS_ATTR, 'layer_metrics'])
    if metric_list_node_id is not None and hasattr(obj, '_metrics'):
      obj_metrics = {m.name: m for m in obj._metrics}
      for reference in self._proto.nodes[metric_list_node_id].children:
        metric = obj_metrics.get(reference.local_name)
        if metric is not None:
          metric_path = '{}.layer_metrics.{}'.format(constants.KERAS_ATTR,
                                                     reference.local_name)
          children.append((metric, reference.node_id, metric_path))

    for (obj_child, child_id, child_name) in children:
      child_proto = self._proto.nodes[child_id]

      if not isinstance(obj_child, trackable.Trackable):
        continue
      if (child_proto.user_object.identifier in
          revived_types.registered_identifiers()):
        setter = revived_types.get_setter(child_proto.user_object)
      elif obj_child._object_identifier in constants.KERAS_OBJECT_IDENTIFIERS:
        setter = _revive_setter
      else:
        setter = setattr
        # pylint: enable=protected-access

      if child_id in self.loaded_nodes:
        if self.loaded_nodes[child_id][0] is not obj_child:
          # This means that the same trackable object is referenced by two
          # different objects that were recreated from the config.
          logging.warn('Looks like there is an object (perhaps variable or '
                       'layer) that is shared between different layers/models. '
                       'This may cause issues when restoring the variable '
                       'values. Object: {}'.format(obj_child))
        continue

      # Overwrite variable names with the ones saved in the SavedModel.
      if (child_proto.WhichOneof('kind') == 'variable' and
          child_proto.variable.name):
        obj_child._handle_name = child_proto.variable.name + ':0'  # pylint: disable=protected-access

      if isinstance(obj_child, data_structures.TrackableDataStructure):
        setter = lambda *args: None

      child_path = '{}.{}'.format(parent_path, child_name)
      self._node_paths[child_id] = child_path
      self._add_children_recreated_from_config(
          obj_child, child_proto, child_id)
      self.loaded_nodes[child_id] = obj_child, setter

  def load_layers(self, compile=True):  # pylint: disable=redefined-builtin
    """Load all layer nodes from the metadata."""
    # Load metrics after models and layers, since it's likely that models
    # and layers will create the metric when initialized (this avoids wasting
    # time by creating objects multiple times).
    metric_list = []
    for node_metadata in self._metadata.nodes:
      if node_metadata.identifier == constants.METRIC_IDENTIFIER:
        metric_list.append(node_metadata)
        continue

      self.loaded_nodes[node_metadata.node_id] = self._load_layer(
          node_metadata.node_id, node_metadata.identifier,
          node_metadata.metadata)

    for node_metadata in metric_list:
      try:
        self.loaded_nodes[node_metadata.node_id] = self._load_layer(
            node_metadata.node_id, node_metadata.identifier,
            node_metadata.metadata)
      except ValueError:
        # Metrics are only needed when the model is compiled later. We ignore
        # errors when trying to load custom metrics when `compile=False` until
        # custom metrics are serialized properly (b/135550038).
        if compile:
          raise
        logging.warning('Unable to restore custom metric. Please ensure that '
                        'the layer implements `get_config` and `from_config` '
                        'when saving. In addition, please use the '
                        '`custom_objects` arg when calling `load_model()`.')

  def _load_layer(self, node_id, identifier, metadata):
    """Load a single layer from a SavedUserObject proto."""
    metadata = json_utils.decode(metadata)

    # If node was already created
    if node_id in self.loaded_nodes:
      node, setter = self.loaded_nodes[node_id]

      # Revive setter requires the object to have a `_serialized_attributes`
      # property. Add it here.
      _maybe_add_serialized_attributes(node, metadata)

      config = metadata.get('config')
      if _is_graph_network(node) and generic_utils.validate_config(config):
        child_nodes = self._get_child_layer_node_ids(node_id)
        self.model_layer_dependencies[node_id] = (node, child_nodes)
        if not child_nodes:
          self._models_to_reconstruct.append(node_id)
      return node, setter

    # Detect whether this object can be revived from the config. If not, then
    # revive from the SavedModel instead.
    obj, setter = self._revive_from_config(identifier, metadata, node_id)
    if obj is None:
      obj, setter = revive_custom_object(identifier, metadata)

    # Add an attribute that stores the extra functions/objects saved in the
    # SavedModel. Most of these functions/objects are ignored, but some are
    # used later in the loading process (e.g. the list of regularization
    # losses, or the training config of compiled models).
    _maybe_add_serialized_attributes(obj, metadata)
    return obj, setter

  def _revive_from_config(self, identifier, metadata, node_id):
    """Revives a layer/model from config, or returns None."""
    if identifier == constants.METRIC_IDENTIFIER:
      obj = self._revive_metric_from_config(metadata)
    else:
      obj = (
          self._revive_graph_network(identifier, metadata, node_id) or
          self._revive_layer_or_model_from_config(metadata, node_id))

    if obj is None:
      return None, None

    setter = self._config_node_setter(_revive_setter)
    self._add_children_recreated_from_config(
        obj, self._proto.nodes[node_id], node_id)
    return obj, setter

  def _revive_graph_network(self, identifier, metadata, node_id):
    """Revives a graph network from config."""
    # Determine whether the metadata contains information for reviving a
    # functional or Sequential model.
    config = metadata.get('config')
    if not generic_utils.validate_config(config):
      return None

    class_name = compat.as_str(metadata['class_name'])
    if generic_utils.get_registered_object(class_name) is not None:
      return None
    model_is_functional_or_sequential = (
        metadata.get('is_graph_network', False) or
        class_name == 'Sequential' or
        class_name == 'Functional')
    if not model_is_functional_or_sequential:
      return None

    # Revive functional and sequential models as blank model objects for now (
    # must be initialized to enable setattr tracking and attribute caching).
    # Reconstruction of the network is deferred until all of the model's layers
    # have been revived.
    if class_name == 'Sequential':
      model = models_lib.Sequential(name=config['name'])
    # The model is a custom Sequential model.
    elif identifier == constants.SEQUENTIAL_IDENTIFIER:
      # Uses the custom class name, since the config does not have one.
      model = models_lib.Sequential(name=class_name)
    else:
      model = models_lib.Functional(
          inputs=[], outputs=[], name=config['name'])

    # Record this model and its layers. This will later be used to reconstruct
    # the model.
    layers = self._get_child_layer_node_ids(node_id)
    self.model_layer_dependencies[node_id] = (model, layers)
    if not layers:
      self._models_to_reconstruct.append(node_id)
    return model

  def _revive_layer_or_model_from_config(self, metadata, node_id):
    """Revives a layer/custom model from config; returns None if infeasible."""
    # Check that the following requirements are met for reviving from config:
    #    1. Object can be deserialized from config.
    #    2. If the object needs to be built, then the build input shape can be
    #       found.
    class_name = metadata.get('class_name')
    config = metadata.get('config')
    shared_object_id = metadata.get('shared_object_id')
    must_restore_from_config = metadata.get('must_restore_from_config')
    if not generic_utils.validate_config(config):
      return None

    try:
      obj = layers_module.deserialize(
          generic_utils.serialize_keras_class_and_config(
              class_name, config, shared_object_id=shared_object_id))
    except ValueError:
      if must_restore_from_config:
        raise RuntimeError(
            'Unable to restore a layer of class {cls}. Layers of '
            'class {cls} require that the class be provided to '
            'the model loading code, either by registering the '
            'class using @keras.utils.register_keras_serializable '
            'on the class def and including that file in your '
            'program, or by passing the class in a '
            'keras.utils.CustomObjectScope that wraps this load '
            'call.'.format(cls=class_name))
      else:
        return None

    # Use the dtype, name, and trainable status. Often times these are not
    # specified in custom configs, so retrieve their values from the metadata.
    # pylint: disable=protected-access
    obj._name = metadata['name']
    if metadata.get('trainable') is not None:
      obj.trainable = metadata['trainable']
    if metadata.get('dtype') is not None:
      obj._set_dtype_policy(metadata['dtype'])
    if metadata.get('stateful') is not None:
      obj.stateful = metadata['stateful']
    # Restore model save spec for subclassed models. (layers do not store a
    # SaveSpec)
    if isinstance(obj, training_lib.Model):
      save_spec = metadata.get('save_spec')
      if save_spec is not None:
        obj._set_save_spec(save_spec)
    # pylint: enable=protected-access

    build_input_shape = metadata.get('build_input_shape')
    built = self._try_build_layer(obj, node_id, build_input_shape)

    if not built:
      # If the layer cannot be built, revive a custom layer instead.
      return None
    return obj

  def _revive_metric_from_config(self, metadata):
    """Revives a metric object using the config saved in the metadata."""
    class_name = compat.as_str(metadata['class_name'])
    config = metadata.get('config')

    if not generic_utils.validate_config(config):
      return None

    try:
      obj = metrics.deserialize(
          generic_utils.serialize_keras_class_and_config(class_name, config))
    except ValueError:
      return None

    build_input_shape = metadata.get('build_input_shape')
    if build_input_shape is not None and hasattr(obj, '_build'):
      obj._build(build_input_shape)  # pylint: disable=protected-access

    return obj

  def _try_build_layer(self, obj, node_id, build_input_shape):
    """Attempts to build the layer."""
    if obj.built or hasattr(obj.build, '_is_default'):
      obj.built = True
      return True

    if build_input_shape is None:
      build_input_shape = self._infer_inputs(node_id, convert_to_shapes=True)

    if build_input_shape is not None:
      obj.build(build_input_shape)
      base_layer.Layer.build(obj, build_input_shape)
      return True

    return False

  def _load_edges(self):
    """Add edges for all nodes that are not waiting on initialization."""
    for node_id, proto in enumerate(self._proto.nodes):
      if node_id not in self.model_layer_dependencies:
        self._add_object_graph_edges(proto, node_id)

  def get_path(self, node_id):
    return self._node_paths[node_id]

  def finalize_objects(self):
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
    for node_id, (node, _) in self.loaded_nodes.items():
      if (not isinstance(node, base_layer.Layer) or
          # Don't finalize models until all layers have finished loading.
          node_id in self.model_layer_dependencies):
        continue

      self._unblock_model_reconstruction(node_id, node)

      if isinstance(node, input_layer.InputLayer):
        continue
      elif isinstance(node, metrics.Metric):
        continue

      if isinstance(node, (RevivedLayer, RevivedInputLayer)):
        layers_revived_from_saved_model.append(node)
      else:
        layers_revived_from_config.append(node)

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
    """Reconstructs the network structure of all models."""
    all_initialized_models = set()
    while self._models_to_reconstruct:
      model_id = self._models_to_reconstruct.pop(0)
      all_initialized_models.add(model_id)
      model, layers = self.model_layer_dependencies[model_id]
      self._reconstruct_model(model_id, model, layers)
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
    """Reconstructs the network structure."""
    config = json_utils.decode(
        self._proto.nodes[model_id].user_object.metadata)['config']

    # Set up model inputs
    if model.inputs:
      # Inputs may already be created if the model is instantiated in another
      # object's __init__.
      pass
    elif isinstance(model, models_lib.Sequential):
      if not layers or not isinstance(layers[0], input_layer.InputLayer):
        if config['layers'][0]['class_name'] == 'InputLayer':
          layers.insert(0, input_layer.InputLayer.from_config(
              config['layers'][0]['config']))
        elif 'batch_input_shape' in config['layers'][0]['config']:
          batch_input_shape = config['layers'][0]['config']['batch_input_shape']
          layers.insert(0, input_layer.InputLayer(
              input_shape=batch_input_shape[1:],
              batch_size=batch_input_shape[0],
              dtype=layers[0].dtype,
              name=layers[0].name + '_input'))
      model.__init__(layers, name=config['name'])
      if not model.inputs:
        first_layer = self._get_child_layer_node_ids(model_id)[0]
        input_specs = self._infer_inputs(first_layer)
        input_shapes = self._infer_inputs(first_layer, convert_to_shapes=True)
        model._set_inputs(input_specs)  # pylint: disable=protected-access
        if not model.built and not isinstance(input_specs, dict):
          model.build(input_shapes)
    else:  # Reconstruct functional model
      (inputs, outputs,
       created_layers) = functional_lib.reconstruct_from_config(
           config, created_layers={layer.name: layer for layer in layers})
      model.__init__(inputs, outputs, name=config['name'])
      functional_lib.connect_ancillary_layers(model, created_layers)

    # Set model dtype.
    _set_network_attributes_from_metadata(model)

    # Unblock models that are dependent on this model.
    self._unblock_model_reconstruction(model_id, model)

  def _get_child_layer_node_ids(self, node_id):
    """Returns the node ids of each layer in a Sequential/Functional model."""
    # Sequential and Functional track layers with names following the format
    # "layer-N". Use this to generate the list of layers.
    num_layers = 0
    child_layers = {}
    pattern = re.compile('layer-(\\d+)')

    for child in self._proto.nodes[node_id].children:
      m = pattern.match(child.local_name)
      if m is None:
        continue
      layer_n = int(m.group(1))
      num_layers = max(layer_n + 1, num_layers)
      child_layers[layer_n] = child.node_id

    ordered = []
    for n in range(num_layers):
      child = child_layers.get(n)
      if child is None:
        break
      ordered.append(child)
    return ordered

  def _search_for_child_node(self, parent_id, path_to_child):
    """Returns node id of child node.

    A helper method for traversing the object graph proto.

    As an example, say that the object graph proto in the SavedModel contains an
    object with the following child and grandchild attributes:

    `parent.child_a.child_b`

    This method can be used to retrieve the node id of `child_b` using the
    parent's node id by calling:

    `_search_for_child_node(parent_id, ['child_a', 'child_b'])`.

    Args:
      parent_id: node id of parent node
      path_to_child: list of children names.

    Returns:
      node_id of child, or None if child isn't found.
    """
    if not path_to_child:
      return parent_id

    for child in self._proto.nodes[parent_id].children:
      if child.local_name == path_to_child[0]:
        return self._search_for_child_node(child.node_id, path_to_child[1:])
    return None

  def _infer_inputs(self, layer_node_id, convert_to_shapes=False):
    """Infers input shape of layer from SavedModel functions."""
    coder = nested_structure_coder.StructureCoder()
    call_fn_id = self._search_for_child_node(
        layer_node_id, ['call_and_return_all_conditional_losses'])
    if call_fn_id is None:
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

  def _config_node_setter(self, setter):
    """Creates edges for nodes that are recreated from config."""
    def setattr_wrapper(obj, name, value):
      # Avoid overwriting attributes of objects recreated from the config.
      if obj._lookup_dependency(name) is None:  # pylint: disable=protected-access
        setter(obj, name, value)
    return setattr_wrapper


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
      layer._init_call_fn_args(
          layer._serialized_attributes['metadata']['expects_training_arg'])
    else:
      layer.call = types.MethodType(
          _unable_to_call_layer_due_to_serialization_issue, layer)

  for layer in layers:
    # 2. Set model inputs and outputs.
    if isinstance(layer, RevivedNetwork):
      _set_network_attributes_from_metadata(layer)

      if hasattr(_get_keras_attr(layer), 'call_and_return_conditional_losses'):
        call_fn = _get_keras_attr(layer).call_and_return_conditional_losses
        if not call_fn.concrete_functions:
          continue
        if call_fn.input_signature is None:
          inputs = infer_inputs_from_restored_call_function(call_fn)
        else:
          inputs = call_fn.input_signature[0]
        layer._set_inputs(inputs)  # pylint: disable=protected-access

    # 3. Add losses that aren't generated by the layer.call function.
    _restore_layer_unconditional_losses(layer)
    _restore_layer_activation_loss(layer)

    # 4. Restore metrics list
    _restore_layer_metrics(layer)

  # pylint: enable=protected-access


def _unable_to_call_layer_due_to_serialization_issue(
    layer, *unused_args, **unused_kwargs):
  """Replaces the `layer.call` if the layer was not fully serialized.

  Keras Model/Layer serialization is relatively relaxed because SavedModels
  are not always loaded back as keras models. Thus, when there is an issue
  tracing a non-signature function, a warning is logged instead of raising an
  error. This results in a SavedModel where the model's call function is saved,
  but the internal layer call functions are not.

  When deserialized with `tf.keras.models.load_model`, the internal layers
  which do not have serialized call functions should raise an error when called.

  Args:
    layer: Layer without the serialized call function.

  Raises:
    ValueError
  """

  raise ValueError(
      'Cannot call custom layer {} of type {}, because the call function was '
      'not serialized to the SavedModel.'
      'Please try one of the following methods to fix this issue:'
      '\n\n(1) Implement `get_config` and `from_config` in the layer/model '
      'class, and pass the object to the `custom_objects` argument when '
      'loading the model. For more details, see: '
      'https://www.tensorflow.org/guide/keras/save_and_serialize'
      '\n\n(2) Ensure that the subclassed model or layer overwrites `call` '
      'and not `__call__`. The input shape and dtype will be automatically '
      'recorded when the object is called, and used when saving. To manually '
      'specify the input shape/dtype, decorate the call function with '
      '`@tf.function(input_signature=...)`.'.format(layer.name, type(layer)))


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

    # Restore metrics list.
    _restore_layer_metrics(layer)

    # Restore RNN layer states
    if (isinstance(layer, recurrent.RNN) and
        layer.stateful and
        hasattr(_get_keras_attr(layer), 'states')):
      layer.states = getattr(_get_keras_attr(layer), 'states', None)
      for variable in nest.flatten(layer.states):
        backend.track_variable(variable)


def _finalize_metric(metric):
  metric.update_state = types.MethodType(metrics_utils.update_state_wrapper(
      metric.keras_api.update_state), metric)
  metric.result = metric.keras_api.result


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
      constants.INPUT_LAYER_IDENTIFIER: (
          RevivedInputLayer, input_layer.InputLayer),
      constants.LAYER_IDENTIFIER: (RevivedLayer, base_layer.Layer),
      constants.MODEL_IDENTIFIER: (RevivedNetwork, model_class),
      constants.NETWORK_IDENTIFIER: (RevivedNetwork, functional_lib.Functional),
      constants.SEQUENTIAL_IDENTIFIER: (RevivedNetwork, models_lib.Sequential),
  }
  parent_classes = revived_classes.get(identifier, None)

  if parent_classes is not None:
    parent_classes = revived_classes[identifier]
    revived_cls = type(
        compat.as_str(metadata['class_name']), parent_classes, {})
    return revived_cls._init_from_metadata(metadata)  # pylint: disable=protected-access
  else:
    raise ValueError('Unable to restore custom object of type {} currently. '
                     'Please make sure that the layer implements `get_config`'
                     'and `from_config` when saving. In addition, please use '
                     'the `custom_objects` arg when calling `load_model()`.'
                     .format(identifier))


def _restore_layer_metrics(layer):
  metrics_list = getattr(_get_keras_attr(layer), 'layer_metrics', {})
  layer_metrics = {m.name: m for m in layer._metrics}  # pylint: disable=protected-access
  for name, metric in metrics_list.items():
    if name not in layer_metrics:
      # Metrics may be added during initialization/building of custom layers.
      layer._metrics.append(metric)  # pylint: disable=protected-access


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
      config = metadata.get('config')
      if generic_utils.validate_config(config):
        revived_obj._config = config
      if metadata.get('input_spec') is not None:
        revived_obj.input_spec = recursively_deserialize_keras_object(
            metadata['input_spec'],
            module_objects={'InputSpec': input_spec.InputSpec})
      if metadata.get('activity_regularizer') is not None:
        revived_obj.activity_regularizer = regularizers.deserialize(
            metadata['activity_regularizer'])
      if metadata.get('_is_feature_layer') is not None:
        revived_obj._is_feature_layer = metadata['_is_feature_layer']
      if metadata.get('stateful') is not None:
        revived_obj.stateful = metadata['stateful']
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
  elif (isinstance(layer, functional_lib.Functional) and
        re.match(r'^layer(_with_weights)?-[\d+]', name) is not None):
    # Edges named "layer-n" or "layer_with_weights-n", which are tracked in
    # network._track_layers, should not be added as an attribute. They should
    # be temporarily added as a dependency so that checkpointed values can be
    # restored. These dependencies are manually deleted in
    # KerasObjectLoader.del_tracking.

    # Set `overwrite=True` in the case that `layer` already tracks a different
    # layer-n. This may cause variable values to not be loaded properly in the
    # original layer-n, but we already warn the users about this
    # (ctrl-f "shared between different layers/models").
    layer._track_trackable(value, name, overwrite=True)  # pylint: disable=protected-access
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


def get_common_shape(x, y):
  """Find a `TensorShape` that is compatible with both `x` and `y`."""
  if x is None != y is None:
    raise RuntimeError(
        'Cannot find a common shape when LHS shape is None but RHS shape '
        'is not (or vice versa): %s vs. %s' % (x, y))
  if x is None:
    return None  # The associated input was not a Tensor, no shape generated.
  if not isinstance(x, tensor_shape.TensorShape):
    raise TypeError('Expected x to be a TensorShape but saw %s' % (x,))
  if not isinstance(y, tensor_shape.TensorShape):
    raise TypeError('Expected y to be a TensorShape but saw %s' % (y,))
  if x.rank != y.rank or x.rank is None:
    return tensor_shape.TensorShape(None)
  dims = []
  for dim_x, dim_y in zip(x.dims, y.dims):
    if (dim_x != dim_y
        or tensor_shape.dimension_value(dim_x) is None
        or tensor_shape.dimension_value(dim_y) is None):
      dims.append(None)
    else:
      dims.append(tensor_shape.dimension_value(dim_x))
  return tensor_shape.TensorShape(dims)


def infer_inputs_from_restored_call_function(fn):
  """Returns TensorSpec of inputs from a restored call function.

  Args:
    fn: Restored layer call function. It is assumed that `fn` has at least
        one concrete function and that the inputs are in the first argument.

  Returns:
    TensorSpec of call function inputs.
  """
  def common_spec(x, y):
    common_shape = get_common_shape(x.shape, y.shape)
    if isinstance(x, sparse_tensor.SparseTensorSpec):
      return sparse_tensor.SparseTensorSpec(common_shape, x.dtype)
    elif isinstance(x, ragged_tensor.RaggedTensorSpec):
      return ragged_tensor.RaggedTensorSpec(common_shape, x.dtype)
    return tensor_spec.TensorSpec(common_shape, x.dtype, x.name)

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
      config = metadata.get('config')
      if generic_utils.validate_config(config):
        revived_obj._config = config

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
    revived_obj._trainable = metadata['trainable']
    # pylint:enable=protected-access


def _maybe_add_serialized_attributes(layer, metadata):
  # Store attributes revived from SerializedAttributes in a un-tracked
  # dictionary. The attributes are the ones listed in CommonEndpoints or
  # "keras_api" for keras-specific attributes.
  if not hasattr(layer, '_serialized_attributes'):
    with trackable.no_automatic_dependency_tracking_scope(layer):
      layer._serialized_attributes = {'metadata': metadata}  # pylint: disable=protected-access


def _get_keras_attr(layer):
  return getattr(layer, '_serialized_attributes', {}).get(constants.KERAS_ATTR,
                                                          None)
