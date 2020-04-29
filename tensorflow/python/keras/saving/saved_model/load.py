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

import re
import types

from tensorflow.python.eager import context
from tensorflow.python.eager import function as defun
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.saving.saved_model import utils
from tensorflow.python.keras.saving.saved_model.serialized_attributes import CommonEndpoints
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load as tf_load
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import revived_types
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking.tracking import delete_tracking
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
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
metrics = LazyLoader("metrics", globals(),
                     "tensorflow.python.keras.metrics")
recurrent = LazyLoader(
    "recurrent", globals(),
    "tensorflow.python.keras.layers.recurrent")
# pylint:enable=g-inconsistent-quotes


PUBLIC_ATTRIBUTES = CommonEndpoints.all_functions.union(
    CommonEndpoints.all_checkpointable_objects)
PUBLIC_ATTRIBUTES.add(constants.KERAS_ATTR)


KERAS_OBJECT_IDENTIFIERS = (
    '_tf_keras_layer', '_tf_keras_input_layer', '_tf_keras_network',
    '_tf_keras_model', '_tf_keras_sequential', '_tf_keras_metric',
    '_tf_keras_rnn_layer')


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
    else:
      logging.warning('No training configuration found in save file, so the '
                      'model was *not* compiled. Compile it manually.')
  # pylint: enable=protected-access

  # Force variables and resources to initialize.
  if not context.executing_eagerly():
    sess = backend.get_session()  # Variables are initialized by this call.
    sess.run(ops.get_collection(ops.GraphKeys.TABLE_INITIALIZERS))

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
    self._all_nodes_recreated_from_config = (
        object_identity.ObjectIdentityWeakSet())
    # Store all node ids that have already been traversed when tracking nodes
    # that were recreated from the config.
    self._traversed_nodes_from_config = []

    # Maps model id -> (blank model obj, list of child layer or their node ids)
    # This tracks all layers in functional and sequential models. These models
    # are only reconstructed after all of their child layers have been created.
    self.model_layer_dependencies = {}
    self._models_to_reconstruct = []

    super(KerasObjectLoader, self).__init__(*args, **kwargs)

    # Now that the node object has been fully loaded, and the checkpoint has
    # been restored, the object no longer needs to track objects added from
    # SerializedAttributes. (Note that saving a training checkpoint still
    # functions correctly, because layers and variables are tracked separately
    # by the Layer object.)
    # TODO(kathywu): Instead of outright deleting these nodes (which would
    # make restoring from a different checkpoint tricky), mark them as extra
    # dependencies that are OK to overwrite.
    for node in self._nodes:
      if not isinstance(node, base_layer.Layer):
        continue
      for name in PUBLIC_ATTRIBUTES:
        delete_tracking(node, name)

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
    if isinstance(obj, base_layer.Layer) and not obj.built:
      metadata = json_utils.decode(proto.user_object.metadata)
      self._try_build_layer(obj, node_id, metadata.get('build_input_shape'))

    # Create list of all possible children
    children = []
    # Look for direct children
    for reference in proto.children:
      obj_child = obj._lookup_dependency(reference.local_name)
      children.append((obj_child, reference.node_id))

    # Add metrics that may have been added to the layer._metrics list.
    # This is stored in the SavedModel as layer.keras_api.layer_metrics in
    # SavedModels created after Tf 2.2.
    metric_list_node_id = self._search_for_child_node(
        node_id, [constants.KERAS_ATTR, 'layer_metrics'], raise_error=False)
    if metric_list_node_id is not None and hasattr(obj, '_metrics'):
      obj_metrics = {m.name: m for m in obj._metrics}
      for reference in self._proto.nodes[metric_list_node_id].children:
        metric = obj_metrics.get(reference.local_name)
        if metric is not None:
          children.append((metric, reference.node_id))

    for (obj_child, child_id) in children:
      child_proto = self._proto.nodes[child_id]

      if not isinstance(obj_child, trackable.Trackable):
        continue
      if (child_proto.user_object.identifier in
          revived_types.registered_identifiers()):
        setter = revived_types.get_setter(child_proto.user_object)
      elif obj_child._object_identifier in KERAS_OBJECT_IDENTIFIERS:
        setter = _revive_setter
      else:
        setter = setattr
        # pylint: enable=protected-access

      if (child_id in self._nodes_recreated_from_config and
          self._nodes_recreated_from_config[child_id][0] is not obj_child):
        # This means that the same trackable object is referenced by two
        # different objects that were recreated from the config.
        logging.warn('Looks like there is an object (perhaps variable or layer)'
                     ' that is shared between different layers/models. This '
                     'may cause issues when restoring the variable values.'
                     'Object: {}'.format(obj_child))
      self._nodes_recreated_from_config[child_id] = (
          obj_child, self._config_node_setter(setter))
      self._all_nodes_recreated_from_config.add(obj_child)
      self._add_children_recreated_from_config(
          obj_child, child_proto, child_id)

  def _load_layers(self):
    layers = {}

    # Load metrics after models and layers, since it's likely that models
    # and layers will create the metric when initialized (this avoids wasting
    # time by creating objects multiple times).
    metric_list = []
    for node_id, proto in enumerate(self._proto.nodes):
      if (proto.WhichOneof('kind') != 'user_object' or
          proto.user_object.identifier not in KERAS_OBJECT_IDENTIFIERS):
        continue
      if proto.user_object.identifier == '_tf_keras_metric':
        metric_list.append((node_id, proto))
        continue

      layers[node_id] = self._load_layer(proto.user_object, node_id)

    for node_id, proto in metric_list:
      layers[node_id] = self._load_layer(proto.user_object, node_id)
    return layers

  def _load_layer(self, proto, node_id):
    """Load a single layer from a SavedUserObject proto."""
    metadata = json_utils.decode(proto.metadata)

    # If node was already created
    if node_id in self._nodes_recreated_from_config:
      node, setter = self._nodes_recreated_from_config[node_id]

      # Revive setter requires the object to have a `_serialized_attributes`
      # property. Add it here.
      _maybe_add_serialized_attributes(node, metadata)

      config = metadata.get('config')
      if _is_graph_network(node) and generic_utils.validate_config(config):
        self.model_layer_dependencies[node_id] = (
            node, self._get_child_layer_node_ids(node_id, node.name))
      return node, setter

    # Detect whether this object can be revived from the config. If not, then
    # revive from the SavedModel instead.
    obj, setter = self._revive_from_config(proto.identifier, metadata, node_id)
    if obj is None:
      obj, setter = revive_custom_object(proto.identifier, metadata)

    # Add an attribute that stores the extra functions/objects saved in the
    # SavedModel. Most of these functions/objects are ignored, but some are
    # used later in the loading process (e.g. the list of regularization
    # losses, or the training config of compiled models).
    _maybe_add_serialized_attributes(obj, metadata)
    return obj, setter

  def _revive_from_config(self, identifier, metadata, node_id):
    """Revives a layer/model from config, or returns None."""
    if identifier == '_tf_keras_metric':
      obj = self._revive_metric_from_config(metadata, node_id)
    else:
      obj = (
          self._revive_graph_network(metadata, node_id) or
          self._revive_layer_from_config(metadata, node_id))

    if obj is None:
      return None, None

    setter = self._config_node_setter(_revive_setter)
    self._nodes_recreated_from_config[node_id] = obj, setter
    self._all_nodes_recreated_from_config.add(obj)
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
    if not (generic_utils.validate_config(config) and
            model_is_functional_or_sequential):
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
    if not generic_utils.validate_config(config):
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
    if metadata.get('stateful') is not None:
      obj.stateful = metadata['stateful']
    # pylint: enable=protected-access

    build_input_shape = metadata.get('build_input_shape')
    built = self._try_build_layer(obj, node_id, build_input_shape)

    if not built:
      # If the layer cannot be built, revive a custom layer instead.
      return None

    return obj

  def _revive_metric_from_config(self, metadata, node_id):
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

      self._unblock_model_reconstruction(node_id, node)

      if isinstance(node, input_layer.InputLayer):
        continue
      elif isinstance(node, metrics.Metric):
        continue

      if node_id in self._nodes_recreated_from_config:
        layers_revived_from_config.append(node)
      else:
        layers_revived_from_saved_model.append(node)

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
    config = json_utils.decode(
        self._proto.nodes[model_id].user_object.metadata)['config']
    if isinstance(model, models_lib.Sequential):
      if config['layers'][0]['class_name'] != 'InputLayer':
        if 'batch_input_shape' in config['layers'][0]['config']:
          batch_input_shape = config['layers'][0]['config']['batch_input_shape']
          layers.insert(0, input_layer.InputLayer(
              input_shape=batch_input_shape[1:],
              batch_size=batch_input_shape[0],
              dtype=layers[0].dtype,
              name=layers[0].name + '_input'))
      model.__init__(layers, name=config['name'])
      if not model.inputs:
        first_layer = self._get_child_layer_node_ids(model_id, model.name)[0]
        input_specs = self._infer_inputs(first_layer)
        input_shapes = self._infer_inputs(first_layer, convert_to_shapes=True)
        model._set_inputs(input_specs)  # pylint: disable=protected-access
        if not model.built and not isinstance(input_specs, dict):
          model.build(input_shapes)
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
    """Returns the node ids of the children layers of a node."""
    # Retrieve the node id of layer.keras_api.layers.
    layer_list = self._search_for_child_node(
        node_id, [constants.KERAS_ATTR, 'layers'], name)
    return [node.node_id for node in self._proto.nodes[layer_list].children]

  def _search_for_child_node(
      self, parent_id, path_to_child, debugging_name=None, raise_error=True):
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
      debugging_name: the name to print out when raising an error.
      raise_error: Whether to raise an error if the child isn't found.

    Returns:
      node_id of child, or None if child isn't found.

    Raises:
      ValueError: if child isn't found and raise_error is True.
    """
    if not path_to_child:
      return parent_id

    for child in self._proto.nodes[parent_id].children:
      if child.local_name == path_to_child[0]:
        return self._search_for_child_node(child.node_id, path_to_child[1:],
                                           debugging_name, raise_error)

    if raise_error:
      raise ValueError(
          'Error when loading {}: could not find attribute {}.\n'
          'Most likely this object was serialized incorrectly.'
          .format(debugging_name or path_to_child[0], path_to_child[0]))
    else:
      return None

  def _infer_inputs(self, layer_node_id, convert_to_shapes=False):
    """Infers input shape of layer from SavedModel functions."""
    coder = nested_structure_coder.StructureCoder()
    call_fn_id = self._search_for_child_node(
        layer_node_id, ['call_and_return_all_conditional_losses'], None,
        raise_error=False)
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

    # 4. Restore metrics list
    _restore_layer_metrics(layer)

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
      '_tf_keras_layer': (RevivedLayer, base_layer.Layer),
      '_tf_keras_input_layer': (RevivedInputLayer, input_layer.InputLayer),
      '_tf_keras_network': (RevivedNetwork, network_lib.Network),
      '_tf_keras_model': (RevivedNetwork, model_class),
      '_tf_keras_sequential': (RevivedNetwork, models_lib.Sequential),
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
    revived_obj.trainable = metadata['trainable']
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
