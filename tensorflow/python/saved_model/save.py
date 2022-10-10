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
"""Exports a SavedModel from a Trackable Python object."""

import collections
import os
import re
import sys
import traceback

from absl import logging

from tensorflow.core.config import flags
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import util as checkpoint_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.eager.polymorphic_function import saved_model_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors
from tensorflow.python.framework import function as framework_fn
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import versions
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import builder_impl
from tensorflow.python.saved_model import function_serialization
from tensorflow.python.saved_model import pywrap_saved_model
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import signature_serialization
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import tracing_utils
from tensorflow.python.saved_model import utils_impl
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import fingerprinting
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import base
from tensorflow.python.trackable import resource
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export

_UNCOPIABLE_DTYPES = frozenset((dtypes.resource, dtypes.variant))

# Container for tensors captured from external functions.
_CapturedTensor = collections.namedtuple("_CapturedTensor",
                                         ["name", "concrete_function"])

# Number of untraced functions to display to user in warning message.
_NUM_DISPLAY_UNTRACED_FUNCTIONS = 5

# API label for SavedModel metrics.
_SAVE_V2_LABEL = "save_v2"


class _AugmentedGraphView(graph_view.ObjectGraphView):
  """An extendable graph which also tracks functions attached to objects.

  Extensions through `add_object` appear in the object graph and any checkpoints
  generated from it, even if they are not dependencies of the node they were
  attached to in the saving program. For example a `.signatures` attribute is
  added to exported SavedModel root objects without modifying the root object
  itself.

  Also tracks functions attached to objects in the graph, through the caching
  `_list_functions` method. Enumerating functions only through this method
  ensures that we get a consistent view of functions, even if object attributes
  create new functions every time they are accessed.
  """

  def __init__(self, root):
    super(_AugmentedGraphView, self).__init__(root)

    # Cache the results of `GraphView.list_children()` to ensure that the
    # `Trackable` children are gathered exactly once.
    self._children_cache = object_identity.ObjectIdentityDictionary()

    # Cache shared between objects in the same object graph. This is passed to
    # `Trackable._trackable_children()`.
    self._serialization_cache = object_identity.ObjectIdentityDictionary()

    # Maps functions -> wrapped functions that capture non-cached variables.
    self._wrapped_functions = {}

    self.untraced_functions = []

  def set_signature(self, signature_map, wrapped_functions):
    """Attach signature to the root object.

    Args:
      signature_map: An object that contains signature functions.
      wrapped_functions: A dictionary mapping functions to functions that are
        guaranteed to not capture cached variables (functions that capture
        cached variables can't be saved).
    """
    self.list_children(self.root)
    # Overrides existing dependency.
    name = signature_serialization.SIGNATURE_ATTRIBUTE_NAME
    self._children_cache[self.root][name] = signature_map
    self._wrapped_functions.update(wrapped_functions)

  def _breadth_first_traversal(self):
    """Returns all trackable objects in the SavedObjectGraph."""
    # This method is overriden to merge all equivalent constant tensors and
    # Assets in the object graph.

    trackable_objects, _ = (
        super(_AugmentedGraphView, self)._breadth_first_traversal())

    asset_paths = object_identity.ObjectIdentityDictionary()
    constant_captures = object_identity.ObjectIdentityDictionary()
    for obj in trackable_objects:
      if isinstance(obj, asset.Asset):
        asset_paths[obj.asset_path] = obj
      if isinstance(obj, saved_model_utils.TrackableConstant):
        constant_captures[obj.capture] = obj

    def _get_merged_trackable(x):
      if isinstance(x, asset.Asset):
        return asset_paths[x.asset_path]
      if isinstance(x, saved_model_utils.TrackableConstant):
        if x.capture in asset_paths:
          return asset_paths[x.capture]
        else:
          return constant_captures[x.capture]
      return x

    for obj in list(self._children_cache.keys()):
      if _get_merged_trackable(obj) is not obj:
        del self._children_cache[obj]
        continue
      for name, child in self._children_cache[obj].items():
        self._children_cache[obj][name] = _get_merged_trackable(child)

    return super(_AugmentedGraphView, self)._breadth_first_traversal()

  def list_children(self, obj):
    """Lists children of `obj` for SavedModel."""
    if obj not in self._children_cache:
      children = self._children_cache[obj] = {}

      for name, child in super(_AugmentedGraphView, self).list_children(
          obj,
          save_type=base.SaveType.SAVEDMODEL,
          cache=self._serialization_cache):
        if isinstance(child, defun.ConcreteFunction):
          child = self._maybe_uncache_variable_captures(child)
        children[name] = child

      # Keep track of untraced functions for later reporting to the user.
      if isinstance(obj, def_function.Function) and not children:
        self.untraced_functions.append(obj.name)

    for name, child in self._children_cache[obj].items():
      yield base.TrackableReference(name, child)

  def get_child(self, obj, name):
    return self._children_cache[obj][name]

  def _maybe_uncache_variable_captures(self, concrete_function):
    if concrete_function in self._wrapped_functions:
      return self._wrapped_functions[concrete_function]
    for capture in concrete_function.captured_inputs:
      if hasattr(capture, "_cached_variable"):
        if concrete_function not in self._wrapped_functions:
          wrapped = self._wrapped_functions[concrete_function] = (
              function_serialization.wrap_cached_variables(concrete_function))
          return wrapped
    return concrete_function

  def list_dependencies(self, obj):
    """Yields `Trackables` that must be loaded before `obj`.

    Dependencies and children are both dictionaries of `Trackables`. Children
    define the object graph structure (used in both checkpoints and SavedModel),
    while dependency defines the order used to load the SavedModel

    Args:
      obj: A `Trackable` object

    Yields:
      Tuple of dependency names and trackable objects.

    Raises:
      TypeError: if any of the returned dependencies are not instances of
        `Trackable`.
    """
    if obj not in self._children_cache:
      # Slot variables do not appear in the children_cache.
      children = {}
    else:
      children = self._children_cache[obj]
    for name, dep in obj._deserialization_dependencies(children).items():  # pylint: disable=protected-access
      if not isinstance(dep, base.Trackable):
        raise TypeError(
            f"The dependency of type {type(dep)} is not an instance `Trackable`"
            ", and can't be saved to SavedModel. Please check the "
            "implementation of `_deserialization_dependencies` in the parent "
            f"object {obj}.")
      yield name, dep


class _SaveableView(object):
  """Provides a frozen view over a trackable root.

  This class helps to create a single stable view over an object to save. The
  saving code should access properties and functions via this class and not via
  the original object as there are cases where an object construct their
  trackable attributes and functions dynamically per call and will yield
  different objects if invoked more than once.

  Changes to the graph, for example adding objects, must happen in
  `augmented_graph_view` (an `_AugmentedGraphView`) before the `_SaveableView`
  is constructed. Changes after the `_SaveableView` has been constructed will be
  ignored.
  """

  def __init__(self, augmented_graph_view, options):
    """Initializes a SaveableView.

    Args:
      augmented_graph_view: A GraphView object.
      options: A SaveOptions instance.
    """

    self.augmented_graph_view = augmented_graph_view
    self._options = options

    (self._trackable_objects, self.node_paths, self.node_ids,
     self._slot_variables, self.object_names) = (
         checkpoint_util.objects_ids_and_slot_variables_and_paths(
             self.augmented_graph_view))

    untraced_functions = self.augmented_graph_view.untraced_functions
    if untraced_functions:
      logging.warning(
          "Found untraced functions such as %s while saving (showing %d of %d)."
          " These functions will not be directly callable after loading.",
          ", ".join(untraced_functions[:_NUM_DISPLAY_UNTRACED_FUNCTIONS]),
          min(_NUM_DISPLAY_UNTRACED_FUNCTIONS, len(untraced_functions)),
          len(untraced_functions))

    self._initialize_save_and_restore_functions()
    self._initialize_nodes_and_concrete_functions()

    self.captured_tensor_node_ids = object_identity.ObjectIdentityDictionary()

  def _initialize_save_and_restore_functions(self):
    """Generates all checkpoint save/restore functions.

    The save and restore functions are generated in the eager context (or in the
    user's Graph/Session) before being copied to the exported GraphDef. These
    functions record the ops for saving/restoring the entire object or
    individual objects (e.g. variables and hash tables).

    The global save and restore functions are generated for compatibility with
    TF1 and loading from C++, and is saved in the `MetaGraphDef.saver_def`.

    The individual functions are generated for the Python TF2 use case, where
    users use the loaded SavedModel as-is, or compose new models using parts
    of the object loaded from the SavedModel. These functions are recorded in
    the `saveable_objects` map in the `SavedObject` proto.
    """
    checkpoint_factory_map, registered_savers = (
        save_util_v1.get_checkpoint_factories_and_keys(self.object_names))
    self._obj_to_registered_saver = object_identity.ObjectIdentityDictionary()
    for saver_name, trackables in registered_savers.items():
      for trackable in trackables.values():
        self._obj_to_registered_saver[trackable] = saver_name
    self._saveable_objects_map = (
        _gen_save_and_restore_functions(checkpoint_factory_map))

  def _initialize_nodes_and_concrete_functions(self):
    """Creates graph with nodes for trackable objects and functions.

    Adds functions for each trackable object to `self.nodes` and associated
    concrete functions to `self.concrete_functions` for serialization.
    """
    self.nodes = list(self._trackable_objects)
    self.gradient_functions = []
    self.gradient_defs = []

    for obj in self.nodes:
      if obj in self._saveable_objects_map:
        for save_fn, restore_fn in self._saveable_objects_map[obj].values():
          self.node_ids[save_fn] = len(self.nodes)
          self.nodes.append(save_fn)

          self.node_ids[restore_fn] = len(self.nodes)
          self.nodes.append(restore_fn)

    self.concrete_functions = [
        obj for obj in self.nodes if isinstance(obj, defun.ConcreteFunction)
    ]

  @property
  def concrete_and_gradient_functions(self):
    return self.concrete_functions + self.gradient_functions

  @property
  def root(self):
    return self.nodes[0]

  def fill_object_graph_proto(self, proto):
    """Populate the nodes, children and slot_variables of a SavedObjectGraph."""
    for node_id, node in enumerate(self.nodes):
      assert self.node_ids[node] == node_id
      object_proto = proto.nodes.add()
      object_proto.slot_variables.extend(self._slot_variables.get(node, ()))
      if isinstance(node, _CapturedTensor):
        continue
      for child in self.augmented_graph_view.list_children(node):
        child_proto = object_proto.children.add()
        child_proto.node_id = self.node_ids[child.ref]
        child_proto.local_name = child.name
      for name, ref in self.augmented_graph_view.list_dependencies(node):
        child_proto = object_proto.dependencies.add()
        child_proto.node_id = self.node_ids[ref]
        child_proto.local_name = name

      if node in self._saveable_objects_map:
        assert node not in self._obj_to_registered_saver, (
            "Objects can't have both SaveableObjects and a registered saver")

        for local_name, (save_fn, restore_fn) in (
            self._saveable_objects_map[node].items()):
          saveable_object_proto = object_proto.saveable_objects[local_name]
          saveable_object_proto.save_function = self.node_ids[save_fn]
          saveable_object_proto.restore_function = self.node_ids[restore_fn]

      elif node in self._obj_to_registered_saver:
        object_proto.registered_saver = self._obj_to_registered_saver[node]

  def map_resources(self):
    """Makes new resource handle ops corresponding to existing resource tensors.

    Creates resource handle ops in the current default graph, whereas
    `accessible_objects` will be from an eager context. Resource mapping adds
    resource handle ops to the main GraphDef of a SavedModel, which allows the
    C++ loader API to interact with resources.

    Returns:
      A tuple of (object_map, tensor_map, asset_info):
        object_map: A dictionary mapping from object in `accessible_objects` to
          replacement objects created to hold the new resource tensors.
        tensor_map: A dictionary mapping from resource tensors extracted from
          `accessible_objects` to newly created resource tensors.
        asset_info: An _AssetInfo tuple describing external assets referenced
          from accessible_objects.
    """
    # Only makes sense when adding to the export Graph
    assert not context.executing_eagerly()
    # TODO(b/205007558): Handle MirroredVariables and other types of variables
    # which may need special casing.
    object_map = object_identity.ObjectIdentityDictionary()
    tensor_map = {}
    asset_info = _AssetInfo(
        asset_defs=[],
        asset_initializers_by_resource={},
        asset_filename_map={},
        asset_index={})

    for node_id in _dependency_sorted_node_ids(self):
      obj = self.nodes[node_id]
      tensors = obj._export_to_saved_model_graph(  # pylint: disable=protected-access
          object_map=object_map,
          tensor_map=tensor_map,
          options=self._options)
      if isinstance(obj, asset.Asset):
        _add_asset_info(obj, asset_info, tensor_map[obj.asset_path])
      if tensors:
        for tensor in tensors:
          self.captured_tensor_node_ids[tensor] = node_id

    return object_map, tensor_map, asset_info

  def add_capture_and_node(self, capture, node):
    node_id = len(self.nodes)
    self.nodes.append(node)
    self.node_ids[capture] = node_id
    self.node_ids[node] = node_id
    self.captured_tensor_node_ids[capture] = node_id
    return node_id

  def get_concrete_resource_initializers(self):
    concrete_initializers = []
    for obj in self.nodes:
      if isinstance(obj, resource.CapturableResource):
        concrete_initializers.append(
            self.augmented_graph_view.get_child(
                obj, "_initialize").get_concrete_function())
    return concrete_initializers


def _gen_save_and_restore_functions(checkpoint_factory_map):
  """Generates global and individual save/restore concrete functions.

  The global functions records the ops to save and restore the entire object to
  a file prefix, while the individual functions save and restore value tensors
  for resources.

  This function is intended to run on the output of
  `save_util_v1.get_checkpoint_factories_and_keys(object_names)`,
  which returns the generated a map of `_CheckpointFactoryData`.

  Args:
    checkpoint_factory_map: A dictionary mapping trackable objects to
      a list of `_CheckpointFactoryData`.

  Returns:
    Tuple of (
      saveable_fn_map: Maps obj -> factory name -> (concrete save, restore)
      )
  """
  # Maps obj -> factory attribute_name -> (concrete save, concrete restore)
  # This
  saveable_fn_map = object_identity.ObjectIdentityDictionary()

  for obj, factory_data_list in checkpoint_factory_map.items():
    if resource_variable_ops.is_resource_variable(obj) or not factory_data_list:
      # There is no need to trace the save and restore functions for variables.
      continue

    if factory_data_list[0].name == trackable_utils.SERIALIZE_TO_TENSORS_NAME:
      # Trace Trackable save and restore functions.
      assert len(factory_data_list) == 1
      saveable_fn_map[obj] = {trackable_utils.SERIALIZE_TO_TENSORS_NAME: (
          tracing_utils.trace_save_and_restore(obj))}
    else:
      # Trace deprecated SaveableObject save and restore functions.
      saveable_fn_map[obj] = (
          saveable_object_util.trace_save_restore_function_map(
              obj, factory_data_list))
  return saveable_fn_map


def _tensor_dict_to_tensorinfo(tensor_dict):
  return {
      key: utils_impl.build_tensor_info_internal(value)
      for key, value in tensor_dict.items()
  }


def _to_safe_name_scope(signature_key, user_input_name):
  """Creates a sanitized name scope from user signature and input names.

  Concatenates signature and input names, sanitizing as needed to be a valid
  scope name.

  Args:
    signature_key: The user-provided key for the signature.
    user_input_name: The user-provided name for the input placeholder.

  Returns:
    A name scope that is safe to be used in tf.name_scope().
  """
  name_scope = "{}_{}".format(signature_key, user_input_name)
  if re.match(r"^[A-Za-z0-9.][A-Za-z0-9_.\\-]*$", name_scope):
    return name_scope
  invalid_prefix_stripped = re.sub(r"^[^A-Za-z0-9.]*", "", name_scope)
  return re.sub(r"[^A-Za-z0-9_.\\-]", "_", invalid_prefix_stripped)


def _map_function_arguments_to_created_inputs(function_arguments, signature_key,
                                              function_name):
  """Creates exterior placeholders in the exported graph for function arguments.

  Functions have two types of inputs: tensors captured from the outside (eager)
  context, and arguments to the function which we expect to receive from the
  user at each call. `_map_captures_to_created_tensors` replaces
  captured tensors with stand-ins (typically these are resource dtype tensors
  associated with variables). `_map_function_inputs_to_created_inputs` runs over
  every argument, creating a new placeholder for each which will belong to the
  exported graph rather than the function body.

  Args:
    function_arguments: A list of argument placeholders in the function body.
    signature_key: The name of the signature being exported, for error messages.
    function_name: The name of the function, for error messages.

  Returns:
    A tuple of (mapped_inputs, exterior_placeholders)
      mapped_inputs: A list with entries corresponding to `function_arguments`
        containing all of the inputs of the function gathered from the exported
        graph (both captured resources and arguments).
      exterior_argument_placeholders: A dictionary mapping from argument names
        to placeholders in the exported graph, containing the explicit arguments
        to the function which a user is expected to provide.

  Raises:
    ValueError: If argument names are not unique.
  """
  # `exterior_argument_placeholders` holds placeholders which are outside the
  # function body, directly contained in a MetaGraph of the SavedModel. The
  # function body itself contains nearly identical placeholders used when
  # running the function, but these exterior placeholders allow Session-based
  # APIs to call the function using feeds and fetches which name Tensors in the
  # MetaGraph.
  exterior_argument_placeholders = {}
  mapped_inputs = []
  for placeholder in function_arguments:
    # `export_captures` contains an exhaustive set of captures, so if we don't
    # find the input there then we now know we have an argument.
    user_input_name = compat.as_str_any(
        placeholder.op.get_attr("_user_specified_name"))
    # If the internal placeholders for a function have names which were
    # uniquified by TensorFlow, then a single user-specified argument name
    # must refer to multiple Tensors. The resulting signatures would be
    # confusing to call. Instead, we throw an exception telling the user to
    # specify explicit names.
    if user_input_name != placeholder.op.name:
      # This should be unreachable, since concrete functions may not be
      # generated with non-unique argument names.
      raise ValueError(
          "Got non-flat/non-unique argument names for SavedModel signature "
          f"'{signature_key}': more than one argument to "
          f"'{compat.as_str_any(function_name)}' was named "
          f"'{user_input_name}'. "
          "Signatures have one Tensor per named input, so to have "
          "predictable names Python functions used to generate these "
          "signatures should avoid *args and Tensors in nested "
          "structures unless unique names are specified for each. Use "
          "tf.TensorSpec(..., name=...) to provide a name for a Tensor "
          "input.")
    arg_placeholder = array_ops.placeholder(
        shape=placeholder.shape,
        dtype=placeholder.dtype,
        name=_to_safe_name_scope(signature_key, user_input_name))
    exterior_argument_placeholders[user_input_name] = arg_placeholder
    mapped_inputs.append(arg_placeholder)
  return mapped_inputs, exterior_argument_placeholders


def _generate_signatures(signature_functions, object_map):
  """Validates and calls `signature_functions` in the exported graph.

  Args:
    signature_functions: A dictionary mapping string keys to concrete TensorFlow
      functions (e.g. from `signature_serialization.canonicalize_signatures`)
      which will be used to generate SignatureDefs.
    object_map: A dictionary that contains mappings from signature functions to
      concrete functions in the exported graph.

  Returns:
    Each function in the `signature_functions` dictionary is called with
    placeholder Tensors, generating a function call operation and output
    Tensors. The placeholder Tensors, the function call operation, and the
    output Tensors from the function call are part of the default Graph.

    This function then returns a dictionary with the same structure as
    `signature_functions`, with the concrete functions replaced by SignatureDefs
    implicitly containing information about how to call each function from a
    TensorFlow 1.x Session / the C++ Loader API. These SignatureDefs reference
    the generated placeholders and Tensor outputs by name.

    The caller is expected to include the default Graph set while calling this
    function as a MetaGraph in a SavedModel, including the returned
    SignatureDefs as part of that MetaGraph.
  """
  signatures = {}
  for signature_key, function in sorted(signature_functions.items()):
    if function.graph.captures:
      argument_inputs = function.graph.inputs[:-len(function.graph.captures)]
    else:
      argument_inputs = function.graph.inputs
    mapped_inputs, exterior_argument_placeholders = (
        _map_function_arguments_to_created_inputs(argument_inputs,
                                                  signature_key, function.name))
    outputs = object_map[function](*mapped_inputs)
    signatures[signature_key] = signature_def_utils.build_signature_def(
        _tensor_dict_to_tensorinfo(exterior_argument_placeholders),
        _tensor_dict_to_tensorinfo(outputs),
        method_name=signature_constants.PREDICT_METHOD_NAME)
  return signatures


_AssetInfo = collections.namedtuple(
    "_AssetInfo",
    [
        # List of AssetFileDef protocol buffers
        "asset_defs",
        # Map from asset variable resource Tensors to their init ops
        "asset_initializers_by_resource",
        # Map from base asset filenames to full paths
        "asset_filename_map",
        # Map from Asset to index of corresponding AssetFileDef
        "asset_index"
    ])


def _add_asset_info(trackable_asset, asset_info, mapped_path_variable):
  """Add `trackable_asset` to `asset_info`."""
  original_path_tensor = trackable_asset.asset_path
  original_path = tensor_util.constant_value(original_path_tensor)
  try:
    original_path = str(original_path.astype(str))
  except AttributeError:
    # Already a string rather than a numpy array
    pass

  path = builder_impl.get_asset_filename_to_add(
      asset_filepath=original_path,
      asset_filename_map=asset_info.asset_filename_map)
  asset_info.asset_filename_map[path] = original_path
  asset_def = meta_graph_pb2.AssetFileDef()
  asset_def.filename = path
  asset_def.tensor_info.name = mapped_path_variable.initial_value.name
  asset_info.asset_defs.append(asset_def)
  asset_info.asset_initializers_by_resource[original_path_tensor] = (
      mapped_path_variable.initializer)
  asset_info.asset_index[trackable_asset] = len(asset_info.asset_defs) - 1


def _iterate_op_types(fn):
  """Iterates through each op in the function and returns the op type and op."""
  if isinstance(fn, framework_fn._DefinedFunction):  # pylint: disable=protected-access
    for node in fn.definition.node_def:
      op_type = node.attr["_gradient_op_type"].s
      if op_type:
        raise ValueError(
            "Unable to save gradient functions when exporting a "
            "_DefinedFunction (generally created through graph freezing utils "
            "or through V1 graph importers). Please save with "
            "`options=tf.SaveOptions(experimental_custom_gradients=False)`")
  else:
    for op in fn.graph.get_operations():
      try:
        op_type = op.get_attr("_gradient_op_type")
      except ValueError:
        continue
      yield op_type, op


def _get_outer_most_capture(fn, capture, func_graph_map):
  """Tries to find the original captured tensor if capture more than once."""
  outer_fn = fn
  while outer_fn is not None and not isinstance(capture, ops.EagerTensor):
    if capture.graph is not outer_fn.graph:
      outer_fn = func_graph_map.get(outer_fn.graph.outer_graph)
    else:
      try:
        capture_index = outer_fn.graph.internal_captures.index(capture)
      except ValueError:
        break  # Capture is a tensor inside function, and not captured from
        # another external function
      capture = outer_fn.graph.external_captures[capture_index]
      outer_fn = func_graph_map.get(outer_fn.graph.outer_graph)
  return outer_fn, capture


def _trace_gradient_functions(graph, saveable_view):
  """Traces gradient functions and records them in the SaveableView."""
  functions = list(graph._functions.values())  # pylint: disable=protected-access
  func_graph_map = {f.graph: f for f in functions if hasattr(f, "graph")}
  seen_op_types = set()

  for fn in functions:
    for op_type, op in _iterate_op_types(fn):
      if op_type in seen_op_types:
        continue
      seen_op_types.add(op_type)

      try:
        custom_gradient = ops.gradient_registry.lookup(op_type)
      except LookupError:
        continue

      try:
        grad_fn = (
            def_function.function(custom_gradient).get_concrete_function(
                None, *op.inputs))
      except Exception as exc:
        traceback.print_exc()
        raise ValueError(
            "Error when tracing gradients for SavedModel.\n\n"
            "Check the error log to see the error that was raised when "
            "converting a gradient function to a concrete function. You may "
            "need to update the custom gradient, or disable saving gradients "
            "with the option "
            "tf.saved_model.SaveOptions(experimental_custom_gradients=False)"
            f".\n\tProblematic op name: {op.name}\n\tGradient inputs: "
            f"{op.inputs}") from exc

      # The gradient function will capture all intermediate values. These
      # captures be serialized so that they can be re-bound to the function when
      # loading.
      bad_captures = []
      for capture in grad_fn.captured_inputs:
        if capture.dtype in _UNCOPIABLE_DTYPES:
          continue
        # Tries to find the outermost capture in case the tensor is a constant
        # or not actually captured in the current function (this could happen if
        # the function is a while loop body, in which case the captured input
        # is not the internal captured tensor).
        outer_fn, outer_capture = _get_outer_most_capture(
            fn, capture, func_graph_map)
        if outer_fn is None or isinstance(outer_capture, ops.EagerTensor):
          if outer_capture not in saveable_view.captured_tensor_node_ids:
            raise ValueError(f"Found invalid capture {outer_capture} when "
                             "saving custom gradients.")
          saveable_view.captured_tensor_node_ids[capture] = (
              saveable_view.captured_tensor_node_ids[outer_capture])
        elif outer_capture.graph is outer_fn.graph:
          capture_name = outer_capture.name
          # It's possible for EagerDefinedFunctions to save different names for
          # input tensors when serialized to FunctionDef (all non-alphanumeric
          # characters are converted to '_').
          if isinstance(outer_fn, defun._EagerDefinedFunction):  # pylint:disable=protected-access
            try:
              arg_index = outer_fn.graph.inputs.index(outer_capture)
              capture_name = outer_fn.signature.input_arg[arg_index].name + ":0"
            except ValueError:
              pass

          node = _CapturedTensor(capture_name, outer_fn.name)
          saveable_view.add_capture_and_node(capture, node)
        else:
          bad_captures.append(capture.name)
      if not bad_captures:
        grad_fn.add_to_graph(graph)
      else:
        raise ValueError(
            f"Cannot save custom gradient {op_type} called in function {fn} "
            "because SavedModel is unable to serialize the captured "
            f"inputs: {bad_captures}")

      saveable_view.gradient_functions.append(grad_fn)
      func_graph_map[grad_fn.graph] = grad_fn

      grad_def = function_pb2.RegisteredGradient()
      grad_def.gradient_func = grad_fn.name
      grad_def.registered_op_type = op_type
      saveable_view.gradient_defs.append(grad_def)


def _fill_meta_graph_def(meta_graph_def, saveable_view, signature_functions,
                         namespace_whitelist, save_custom_gradients):
  """Generates a MetaGraph which calls `signature_functions`.

  Args:
    meta_graph_def: The MetaGraphDef proto to fill.
    saveable_view: The _SaveableView being exported.
    signature_functions: A dictionary mapping signature keys to concrete
      functions containing signatures to add to the MetaGraph.
    namespace_whitelist: List of strings containing whitelisted op namespaces.
    save_custom_gradients: Whether to save custom gradients.

  Returns:
    A tuple of (_AssetInfo, Graph) containing the captured assets and
    exported Graph generated from tracing the saveable_view.
  """
  # List objects from the eager context to make sure Optimizers give us the
  # right Graph-dependent variables.
  resource_initializers = saveable_view.get_concrete_resource_initializers()
  exported_graph = ops.Graph()
  resource_initializer_ops = []
  with exported_graph.as_default():
    object_map, tensor_map, asset_info = saveable_view.map_resources()
    signatures = _generate_signatures(signature_functions, object_map)
    if save_custom_gradients:
      _trace_gradient_functions(exported_graph, saveable_view)

    # Create initializers for assets and resources.
    for resource_initializer_function in resource_initializers:
      asset_dependencies = []
      for capture in resource_initializer_function.graph.external_captures:
        asset_initializer = asset_info.asset_initializers_by_resource.get(
            capture, None)
        if asset_initializer is not None:
          asset_dependencies.append(asset_initializer)
      with ops.control_dependencies(asset_dependencies):
        mapped_initializer = object_map[resource_initializer_function]
        resource_initializer_ops.append(mapped_initializer())
    resource_initializer_ops.extend(
        asset_info.asset_initializers_by_resource.values())
    with ops.control_dependencies(resource_initializer_ops):
      init_op = control_flow_ops.no_op()
    # Add the same op to the main_op collection and to the init_op
    # signature. The collection is for compatibility with older loader APIs;
    # only one will be executed.
    meta_graph_def.collection_def[constants.MAIN_OP_KEY].node_list.value.append(
        init_op.name)
    meta_graph_def.signature_def[constants.INIT_OP_SIGNATURE_KEY].CopyFrom(
        signature_def_utils.op_signature_def(init_op,
                                             constants.INIT_OP_SIGNATURE_KEY))

  # Saving an object-based checkpoint again gathers variables. We need to do the
  # gathering from the eager context so Optimizers save the right set of
  # variables, but want any operations associated with the save/restore to be in
  # the exported graph (thus the `to_graph` argument).
  def call_with_mapped_captures(function, args):
    if function in object_map:
      return object_map[function](*args)
    # Registered saver/restore functions do not appear in `object_map`, because
    # they are not in the object graph.
    return saved_model_utils.ExportedConcreteFunction(
        function, tensor_map)(*args)

  for obj in object_map.values():
    obj._maybe_initialize_trackable()  # pylint: disable=protected-access
  named_saveable_objects, registered_savers = (
      save_util_v1.frozen_saveables_and_savers(
          graph_view=saveable_view.augmented_graph_view,
          object_map=object_map,
          to_graph=exported_graph,
          call_with_mapped_captures=call_with_mapped_captures))
  saver = functional_saver.MultiDeviceSaver.from_saveables(
      named_saveable_objects, registered_savers, call_with_mapped_captures)

  with exported_graph.as_default():
    saver_def = saver.to_proto()
    meta_graph_def.saver_def.CopyFrom(saver_def)

  # At this point all nodes that can be added to the SavedObjectGraph have been
  # added, so run the following to validate deserialization dependencies.
  _dependency_sorted_node_ids(saveable_view)

  graph_def = exported_graph.as_graph_def(add_shapes=True)
  graph_def.library.registered_gradients.extend(saveable_view.gradient_defs)
  _verify_ops(graph_def, namespace_whitelist)

  meta_graph_def.graph_def.CopyFrom(graph_def)
  meta_graph_def.meta_info_def.tags.append(tag_constants.SERVING)
  meta_graph_def.meta_info_def.tensorflow_version = versions.__version__
  meta_graph_def.meta_info_def.tensorflow_git_version = (
      versions.__git_version__)
  # We currently always strip default attributes.
  meta_graph_def.meta_info_def.stripped_default_attrs = True
  meta_graph_def.meta_info_def.stripped_op_list.MergeFrom(
      meta_graph.stripped_op_list_for_graph(meta_graph_def.graph_def))
  meta_graph_def.asset_file_def.extend(asset_info.asset_defs)
  for signature_key, signature in signatures.items():
    meta_graph_def.signature_def[signature_key].CopyFrom(signature)
  meta_graph.strip_graph_default_valued_attrs(meta_graph_def)
  # store tensor_content in litle endian format
  if sys.byteorder == "big":
    utils_impl.swap_function_tensor_content(meta_graph_def, "big", "little")
  return asset_info, exported_graph


def _verify_ops(graph_def, namespace_whitelist):
  """Verifies that all namespaced ops in the graph are whitelisted.

  Args:
   graph_def: the GraphDef to validate.
   namespace_whitelist: a list of namespaces to allow. If `None`, all will be
     allowed. If an op does not have a namespace, it will be allowed.

  Raises:
   ValueError: If the graph contains ops that violate the whitelist.
  """
  # By default, if the user has not specified a whitelist, we want to allow
  # everything.  We check for None directly rather than falseness, since the
  # user may instead want to pass an empty list to disallow all custom
  # namespaced ops.
  if namespace_whitelist is None:
    return

  invalid_ops = []
  invalid_namespaces = set()

  all_operations = []
  all_operations.extend(meta_graph.ops_used_by_graph_def(graph_def))

  for op in all_operations:
    if ">" in op:
      namespace = op.split(">")[0]
      if namespace not in namespace_whitelist:
        invalid_ops.append(op)
        invalid_namespaces.add(namespace)
  if invalid_ops:
    raise ValueError(
        "Attempted to save ops from non-whitelisted namespaces to SavedModel: "
        f"{invalid_ops}.\nPlease verify that these ops should be saved, since "
        "they must be available when loading the SavedModel. If loading from "
        "Python, you must import the library defining these ops. From C++, "
        "link the custom ops to the serving binary. Once you've confirmed this,"
        " add the following namespaces to the `namespace_whitelist` "
        f"argument in tf.saved_model.SaveOptions: {invalid_namespaces}.")


def _dependency_sorted_node_ids(saveable_view):
  """Returns topologically sorted nodes, sorted by dependencies."""
  dependency_map = {}
  for node in saveable_view.nodes:
    node_id = saveable_view.node_ids[node]
    deps = dependency_map[node_id] = []
    # TODO(kathywu): Remove once all of these have been converted to trackable.
    if isinstance(node, _CapturedTensor):
      continue  # These are not `Trackable` and therefore have no dependencies.
    for _, dep in saveable_view.augmented_graph_view.list_dependencies(node):
      if dep not in saveable_view.node_ids:
        node_path = trackable_utils.pretty_print_node_path(
            saveable_view.node_paths[node])
        raise ValueError(
            f"Found an untracked dependency. Object {node_path} depends "
            f"on {dep}, but this dependency isn't listed as a child. "
            "Please track this child by overriding `_trackable_children` "
            "or use `._track_trackable`.")
      deps.append(saveable_view.node_ids[dep])
  try:
    return trackable_utils.order_by_dependency(dependency_map)
  except trackable_utils.CyclicDependencyError as err:
    pretty_printed_nodes = []
    pretty_printed_dependencies = []

    for x, deps in err.leftover_dependency_map.items():
      node_path = trackable_utils.pretty_print_node_path(
          saveable_view.node_paths[saveable_view.nodes[x]])
      pretty_printed_nodes.append(
          f"\tNode {x} = {node_path} (type {type(saveable_view.nodes[x])})")
      pretty_printed_dependencies.append(f"\tNode {x} depends on nodes {deps}")
    pretty_printed_nodes = "\n".join(pretty_printed_nodes)
    pretty_printed_dependencies = "\n".join(pretty_printed_dependencies)
    raise ValueError(
        "There is one or more dependency cycle in the saved Trackable object. "
        "Saving cannot continue until this cycle is resolved."
        f"\n>> Unresolved nodes:\n{pretty_printed_nodes}"
        f"\n>> Unresolved cyclic dependencies:\n{pretty_printed_dependencies}")


def _serialize_object_graph(saveable_view, asset_file_def_index):
  """Save a SavedObjectGraph proto for `root`."""
  # SavedObjectGraph is similar to the TrackableObjectGraph proto in the
  # checkpoint. It will eventually go into the SavedModel.
  proto = saved_object_graph_pb2.SavedObjectGraph()
  saveable_view.fill_object_graph_proto(proto)

  for concrete_function in saveable_view.concrete_and_gradient_functions:
    name = compat.as_text(concrete_function.name)
    serialized = function_serialization.serialize_concrete_function(
        concrete_function, saveable_view.captured_tensor_node_ids)
    if serialized is not None:
      proto.concrete_functions[name].CopyFrom(serialized)

  for obj, obj_proto in zip(saveable_view.nodes, proto.nodes):
    _write_object_proto(obj, obj_proto, asset_file_def_index,
                        saveable_view.augmented_graph_view.list_children)
  return proto


def _write_object_proto(obj, proto, asset_file_def_index, list_children_fn):
  """Saves an object into SavedObject proto."""
  if isinstance(obj, asset.Asset):
    proto.asset.SetInParent()
    proto.asset.asset_file_def_index = asset_file_def_index[obj]
  elif resource_variable_ops.is_resource_variable(obj):
    options = save_context.get_save_options()
    obj._write_object_proto(proto, options)  # pylint: disable=protected-access
  elif isinstance(obj, def_function.Function):
    proto.function.CopyFrom(
        function_serialization.serialize_function(
            obj, [x.ref for x in list_children_fn(obj)]))
  elif isinstance(obj, defun.ConcreteFunction):
    proto.bare_concrete_function.CopyFrom(
        function_serialization.serialize_bare_concrete_function(obj))
  elif isinstance(obj, _CapturedTensor):
    proto.captured_tensor.name = obj.name
    proto.captured_tensor.concrete_function = obj.concrete_function
  elif isinstance(obj, resource.CapturableResource):
    proto.resource.device = obj._resource_device  # pylint: disable=protected-access
  else:
    registered_type_proto = revived_types.serialize(obj)
    if registered_type_proto is None:
      # Fallback for types with no matching registration
      # pylint:disable=protected-access
      registered_type_proto = saved_object_graph_pb2.SavedUserObject(
          identifier=obj._object_identifier,
          version=versions_pb2.VersionDef(
              producer=1, min_consumer=1, bad_consumers=[]))
      # pylint:enable=protected-access
    proto.user_object.CopyFrom(registered_type_proto)

  registered_name = registration.get_registered_class_name(obj)
  if registered_name:
    proto.registered_name = registered_name
    serialized_user_proto = obj._serialize_to_proto(object_proto=proto)  # pylint: disable=protected-access
    if serialized_user_proto is not None:
      proto.serialized_user_proto.Pack(serialized_user_proto)


def _export_debug_info(exported_graph, export_dir):
  """Exports debug information from graph to file.

  Creates and writes GraphDebugInfo with traces for ops in all functions of the
  exported_graph.

  Args:
    exported_graph: A Graph that has been created by tracing a saveable view.
    export_dir: SavedModel directory in which to write the debug info.
  """
  exported_operations = []
  for fn_name in exported_graph._functions:  # pylint: disable=protected-access
    fn = exported_graph._get_function(fn_name)  # pylint: disable=protected-access
    if not isinstance(fn, defun._EagerDefinedFunction):  # pylint: disable=protected-access
      continue

    fn_graph = fn.graph
    for fn_op in fn_graph.get_operations():
      exported_operations.append((fn_name, fn_op))

  graph_debug_info = error_interpolation.create_graph_debug_info_def(
      exported_operations)
  file_io.atomic_write_string_to_file(
      file_io.join(
          utils_impl.get_or_create_debug_dir(export_dir),
          constants.DEBUG_INFO_FILENAME_PB),
      graph_debug_info.SerializeToString(deterministic=True))


@tf_export(
    "saved_model.save",
    v1=["saved_model.save", "saved_model.experimental.save"])
def save(obj, export_dir, signatures=None, options=None):
  # pylint: disable=line-too-long
  """Exports a [tf.Module](https://www.tensorflow.org/api_docs/python/tf/Module) (and subclasses) `obj` to [SavedModel format](https://www.tensorflow.org/guide/saved_model#the_savedmodel_format_on_disk).

  The `obj` must inherit from the [`Trackable`
  class](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/tracking/base.py#L591).

  Example usage:

  >>> class Adder(tf.Module):
  ...   @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
  ...   def add(self, x):
  ...     return x + x

  >>> model = Adder()
  >>> tf.saved_model.save(model, '/tmp/adder')

  The resulting SavedModel is then servable with an input named "x", a scalar
  with dtype float32.

  _Signatures_

  Signatures define the input and output types for a computation. The optional
  save `signatures` argument controls which methods in `obj` will be
  available to programs which consume `SavedModel`s, for example, serving
  APIs. Python functions may be decorated with
  `@tf.function(input_signature=...)` and passed as signatures directly, or
  lazily with a call to `get_concrete_function` on the method decorated with
  `@tf.function`.

  Example:

  >>> class Adder(tf.Module):
  ...   @tf.function
  ...   def add(self, x):
  ...     return x + x

  >>> model = Adder()
  >>> tf.saved_model.save(
  ...   model, '/tmp/adder',signatures=model.add.get_concrete_function(
  ...     tf.TensorSpec([], tf.float32)))

  If a `@tf.function` does not have an input signature and
  `get_concrete_function` is not called on that method, the function will not
  be directly callable in the restored SavedModel.

  Example:

  >>> class Adder(tf.Module):
  ...   @tf.function
  ...   def add(self, x):
  ...     return x + x

  >>> model = Adder()
  >>> tf.saved_model.save(model, '/tmp/adder')
  >>> restored = tf.saved_model.load('/tmp/adder')
  >>> restored.add(1.)
  Traceback (most recent call last):
  ...
  ValueError: Found zero restored functions for caller function.

  If the `signatures` argument is omitted, `obj` will be searched for
  `@tf.function`-decorated methods. If exactly one traced `@tf.function` is
  found, that method will be used as the default signature for the SavedModel.
  Else, any `@tf.function` attached to `obj` or its dependencies will be
  exported for use with `tf.saved_model.load`.

  When invoking a signature in an exported SavedModel, `Tensor` arguments are
  identified by name. These names will come from the Python function's argument
  names by default. They may be overridden by specifying a `name=...` argument
  in the corresponding `tf.TensorSpec` object. Explicit naming is required if
  multiple `Tensor`s are passed through a single argument to the Python
  function.

  The outputs of functions used as `signatures` must either be flat lists, in
  which case outputs will be numbered, or a dictionary mapping string keys to
  `Tensor`, in which case the keys will be used to name outputs.

  Signatures are available in objects returned by `tf.saved_model.load` as a
  `.signatures` attribute. This is a reserved attribute: `tf.saved_model.save`
  on an object with a custom `.signatures` attribute will raise an exception.

  _Using `tf.saved_model.save` with Keras models_

  While Keras has its own [saving and loading
  API](https://www.tensorflow.org/guide/keras/save_and_serialize),
  this function can be used to export Keras models. For example, exporting with
  a signature specified:

  >>> class Adder(tf.keras.Model):
  ...   @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  ...   def concat(self, x):
  ...      return x + x

  >>> model = Adder()
  >>> tf.saved_model.save(model, '/tmp/adder')

  Exporting from a function without a fixed signature:

  >>> class Adder(tf.keras.Model):
  ...   @tf.function
  ...   def concat(self, x):
  ...      return x + x

  >>> model = Adder()
  >>> tf.saved_model.save(
  ...   model, '/tmp/adder',
  ...   signatures=model.concat.get_concrete_function(
  ...     tf.TensorSpec(shape=[], dtype=tf.string, name="string_input")))

  `tf.keras.Model` instances constructed from inputs and outputs already have a
  signature and so do not require a `@tf.function` decorator or a `signatures`
  argument. If neither are specified, the model's forward pass is exported.

  >>> x = tf.keras.layers.Input((4,), name="x")
  >>> y = tf.keras.layers.Dense(5, name="out")(x)
  >>> model = tf.keras.Model(x, y)
  >>> tf.saved_model.save(model, '/tmp/saved_model/')

  The exported SavedModel takes "x" with shape [None, 4] and returns "out"
  with shape [None, 5]

  _Variables and Checkpoints_

  Variables must be tracked by assigning them to an attribute of a tracked
  object or to an attribute of `obj` directly. TensorFlow objects (e.g. layers
  from `tf.keras.layers`, optimizers from `tf.train`) track their variables
  automatically. This is the same tracking scheme that `tf.train.Checkpoint`
  uses, and an exported `Checkpoint` object may be restored as a training
  checkpoint by pointing `tf.train.Checkpoint.restore` to the SavedModel's
  "variables/" subdirectory.

  `tf.function` does not hard-code device annotations from outside the function
  body, instead of using the calling context's device. This means for example
  that exporting a model that runs on a GPU and serving it on a CPU will
  generally work, with some exceptions:

    * `tf.device` annotations inside the body of the function will be hard-coded
      in the exported model; this type of annotation is discouraged.
    * Device-specific operations, e.g. with "cuDNN" in the name or with
      device-specific layouts, may cause issues.
    * For `ConcreteFunctions`, active distribution strategies will cause device
      placements to be hard-coded in the function.

  SavedModels exported with `tf.saved_model.save` [strip default-valued
  attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes)
  automatically, which removes one source of incompatibilities when the consumer
  of a SavedModel is running an older TensorFlow version than the
  producer. There are however other sources of incompatibilities which are not
  handled automatically, such as when the exported model contains operations
  which the consumer does not have definitions for.

  Args:
    obj: A trackable object (e.g. tf.Module or tf.train.Checkpoint) to export.
    export_dir: A directory in which to write the SavedModel.
    signatures: Optional, one of three types: * a `tf.function` with an input
      signature specified, which will use the default serving signature key, *
      the result of `f.get_concrete_function` on a `@tf.function`-decorated
      function `f`, in which case `f` will be used to generate a signature for
      the SavedModel under the default serving signature key, * a dictionary,
      which maps signature keys to either `tf.function` instances with input
      signatures or concrete functions. Keys of such a dictionary may be
      arbitrary strings, but will typically be from the
      `tf.saved_model.signature_constants` module.
    options: `tf.saved_model.SaveOptions` object for configuring save options.

  Raises:
    ValueError: If `obj` is not trackable.

  @compatibility(eager)
  Not well supported when graph building. From TensorFlow 1.x,
  `tf.compat.v1.enable_eager_execution()` should run first. Calling
  tf.saved_model.save in a loop when graph building from TensorFlow 1.x will
  add new save operations to the default graph each iteration.

  May not be called from within a function body.
  @end_compatibility
  """
  if isinstance(export_dir, os.PathLike):
    export_dir = os.fspath(export_dir)
  # pylint: enable=line-too-long
  metrics.IncrementWriteApi(_SAVE_V2_LABEL)
  save_and_return_nodes(obj, export_dir, signatures, options)

  metrics.IncrementWrite(write_version="2")


def save_and_return_nodes(obj,
                          export_dir,
                          signatures=None,
                          options=None,
                          experimental_skip_checkpoint=False):
  """Saves a SavedModel while returning all saved nodes and their paths.

  Please see `tf.saved_model.save` for details.

  Args:
    obj: A trackable object to export.
    export_dir: A directory in which to write the SavedModel.
    signatures: A function or dictionary of functions to save in the SavedModel
      as signatures.
    options: `tf.saved_model.SaveOptions` object for configuring save options.
    experimental_skip_checkpoint: If set to `True`, the checkpoint will not be
      written.

  Returns:
    A tuple of (a list of saved nodes in the order they are serialized to the
      `SavedObjectGraph`, dictionary mapping nodes to one possible path from
      the root node to the key node)
  """
  options = options or save_options.SaveOptions()
  # TODO(b/205008509): Factor out some subset of SavedModelBuilder which is 2.x
  # compatible (no sessions) and share it with this export API rather than
  # making a SavedModel proto and writing it directly.
  saved_model = saved_model_pb2.SavedModel()
  meta_graph_def = saved_model.meta_graphs.add()

  _, exported_graph, object_saver, asset_info, saved_nodes, node_paths = (
      _build_meta_graph(obj, signatures, options, meta_graph_def))
  saved_model.saved_model_schema_version = (
      constants.SAVED_MODEL_SCHEMA_VERSION)

  # Write the checkpoint, copy assets into the assets directory, and write out
  # the SavedModel proto itself.
  if not experimental_skip_checkpoint:
    utils_impl.get_or_create_variables_dir(export_dir)
    ckpt_options = checkpoint_options.CheckpointOptions(
        experimental_io_device=options.experimental_io_device)
    object_saver.save(
        utils_impl.get_variables_path(export_dir), options=ckpt_options)
  builder_impl.copy_assets_to_destination_dir(asset_info.asset_filename_map,
                                              export_dir)
  # Note that this needs to be the last file operation when saving the
  # SavedModel. Users rely on checking saved_model_dir/saved_model.pb as an
  # indication that the SavedModel is completely written.
  if context.executing_eagerly():
    try:
      context.async_wait()  # Ensure save operations have completed.
    except errors.NotFoundError as err:
      raise FileNotFoundError(
          f"{err}\n You may be trying to save on a different device from the "
          "computational device. Consider setting the "
          "`experimental_io_device` option in `tf.saved_model.SaveOptions` "
          "to the io_device such as '/job:localhost'.")

  # We will slowly migrate code in this function to pywrap_saved_model.Save
  # as we build up the C++ API.
  pywrap_saved_model.Save(export_dir)

  saved_model_serialized = saved_model.SerializeToString(deterministic=True)

  # Write fingerprint protobuf, if requested.
  if flags.config().saved_model_fingerprinting.value():
    fingerprint_path = file_io.join(
        compat.as_str(export_dir),
        compat.as_str(constants.FINGERPRINT_FILENAME))
    fingerprint_proto = fingerprinting.CreateFingerprintDef(
        saved_model_serialized, export_dir)
    file_io.atomic_write_string_to_file(fingerprint_path, fingerprint_proto)

  path = file_io.join(
      compat.as_str(export_dir),
      compat.as_str(constants.SAVED_MODEL_FILENAME_PB))
  file_io.atomic_write_string_to_file(
      path, saved_model.SerializeToString(deterministic=True))

  # Save debug info, if requested.
  if options.save_debug_info:
    _export_debug_info(exported_graph, export_dir)
  # Clean reference cycles so repeated export()s don't make work for the garbage
  # collector. Before this point, we need to keep references to captured
  # constants in the saved graph.
  ops.dismantle_graph(exported_graph)

  return saved_nodes, node_paths


def export_meta_graph(obj, filename, signatures=None, options=None):
  """Exports the MetaGraph proto of the `obj` to a file.

  This function goes through the same procedures saved_model.save goes to
  produce the given object's MetaGraph, then saves it to the given file. It
  skips saving checkpoint information, and is useful when all one wants is the
  graph defining the model.

  Args:
    obj: A trackable object to build the MetaGraph from.
    filename: The file into which to write the MetaGraph.
    signatures: Optional, either a `tf.function` with an input signature
      specified or the result of `f.get_concrete_function` on a
      `@tf.function`-decorated function `f`, in which case `f` will be used to
      generate a signature for the SavedModel under the default serving
      signature key. `signatures` may also be a dictionary, in which case it
      maps from signature keys to either `tf.function` instances with input
      signatures or concrete functions. The keys of such a dictionary may be
      arbitrary strings, but will typically be from the
      `tf.saved_model.signature_constants` module.
    options: Optional, `tf.saved_model.SaveOptions` object that specifies
      options for saving.
  """
  options = options or save_options.SaveOptions()
  export_dir = os.path.dirname(filename)
  meta_graph_def, exported_graph, _, _, _, _ = _build_meta_graph(
      obj, signatures, options)

  file_io.atomic_write_string_to_file(
      filename, meta_graph_def.SerializeToString(deterministic=True))

  # Save debug info, if requested.
  if options.save_debug_info:
    _export_debug_info(exported_graph, export_dir)

  # Clean reference cycles so repeated export()s don't make work for the garbage
  # collector. Before this point, we need to keep references to captured
  # constants in the saved graph.
  ops.dismantle_graph(exported_graph)


def _build_meta_graph_impl(obj, signatures, options, meta_graph_def=None):
  """Creates a MetaGraph containing the resources and functions of an object."""
  if ops.inside_function():
    raise AssertionError(
        "`tf.saved_model.save` is not supported inside a traced @tf.function. "
        "Move the call to the outer eagerly-executed context.")
  # pylint: enable=line-too-long
  if not isinstance(obj, base.Trackable):
    raise ValueError(
        "Expected an object of type `Trackable`, such as `tf.Module` or a "
        f"subclass of the `Trackable` class, for export. Got {obj} "
        f"with type {type(obj)}.")
  meta_graph_def = meta_graph_def or meta_graph_pb2.MetaGraphDef()

  augmented_graph_view = _AugmentedGraphView(obj)
  if signatures is None:
    signatures = signature_serialization.find_function_to_export(
        augmented_graph_view)

  signatures, wrapped_functions = (
      signature_serialization.canonicalize_signatures(signatures))
  signature_serialization.validate_augmented_graph_view(augmented_graph_view)
  signature_map = signature_serialization.create_signature_map(signatures)
  augmented_graph_view.set_signature(signature_map, wrapped_functions)

  # Use _SaveableView to provide a frozen listing of properties and functions.
  saveable_view = _SaveableView(augmented_graph_view, options)
  object_saver = checkpoint.TrackableSaver(augmented_graph_view)
  asset_info, exported_graph = _fill_meta_graph_def(
      meta_graph_def, saveable_view, signatures, options.namespace_whitelist,
      options.experimental_custom_gradients)
  if options.function_aliases:
    function_aliases = meta_graph_def.meta_info_def.function_aliases
    for alias, func in options.function_aliases.items():
      for fdef in func._list_all_concrete_functions():  # pylint: disable=protected-access
        function_aliases[fdef.name] = alias

  object_graph_proto = _serialize_object_graph(saveable_view,
                                               asset_info.asset_index)
  meta_graph_def.object_graph_def.CopyFrom(object_graph_proto)

  return (meta_graph_def, exported_graph, object_saver, asset_info,
          saveable_view.nodes, saveable_view.node_paths)


def _build_meta_graph(obj, signatures, options, meta_graph_def=None):
  """Creates a MetaGraph under a save context.

  Args:
    obj: A trackable object to build the MetaGraph from.
    signatures: Can be a `tf.function` with an input signature specified or the
      result of `f.get_concrete_function` on a `@tf.function`-decorated function
      `f`. `signatures` may also be a dictionary, in which case it maps from
      signature keys to `tf.function` instances. If None, finds signature to
      export from the `@tf.function`-decorated methods in `obj`.
    options: `tf.saved_model.SaveOptions` object that specifies options for
      saving.
    meta_graph_def: Optional, the MetaGraphDef proto fill.

  Raises:
    AssertionError: If `export_meta_graph` is executing inside a `tf.function`.
    ValueError: If `obj` is not trackable.

  Returns:
    meta_graph_def: Filled MetaGraphDef proto
    exported_graph: `tf.Graph` object generated from `obj`.
    object_saver: `checkpoint.TrackableSaver` of the `obj` and its dependencies.
    asset_info: `_AssetInfo` tuple containing external assets in the `obj`.
    saveable_view.nodes: _SaveableView nodes.
    saveable_view.node_paths: _SaveableView paths.
  """

  with save_context.save_context(options):
    return _build_meta_graph_impl(obj, signatures, options, meta_graph_def)
