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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import gc
import os
import re
import sys
import traceback

from absl import logging
import numpy

from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.framework import constant_op
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
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import pywrap_saved_model
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import signature_serialization
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils_impl
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training.saving import checkpoint_options
from tensorflow.python.training.saving import functional_saver
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import graph_view
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export

_UNCOPIABLE_DTYPES = frozenset((dtypes.resource, dtypes.variant))

# A container for an EagerTensor constant which has been copied to the exported
# Graph.
_CapturedConstant = collections.namedtuple("_CapturedConstant",
                                           ["eager_tensor", "graph_tensor"])
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
  `list_functions` method. Enumerating functions only through this method
  ensures that we get a consistent view of functions, even if object attributes
  create new functions every time they are accessed.
  """

  def __init__(self, root):
    if (not context.executing_eagerly() and not ops.inside_function()):
      saveables_cache = object_identity.ObjectIdentityWeakKeyDictionary()
    else:
      saveables_cache = None
    super(_AugmentedGraphView, self).__init__(root, saveables_cache)
    # Object -> (name -> dep)
    self._extra_dependencies = object_identity.ObjectIdentityDictionary()
    self._functions = object_identity.ObjectIdentityDictionary()
    # Cache shared between objects in the same object graph. This is passed to
    # each trackable object's `_list_extra_dependencies_for_serialization` and
    # `_list_functions_for_serialization` function.
    self._serialization_cache = object_identity.ObjectIdentityDictionary()

  def add_object(self, parent_node, name_in_parent, subgraph_root):
    """Attach an object to `parent_node`, overriding any existing dependency."""
    self._extra_dependencies.setdefault(parent_node,
                                        {})[name_in_parent] = subgraph_root

  def list_dependencies(self, obj):
    """Overrides a parent method to include `add_object` objects."""
    extra_dependencies = self.list_extra_dependencies(obj)
    extra_dependencies.update(self._extra_dependencies.get(obj, {}))

    used_names = set()
    for name, dep in super(_AugmentedGraphView, self).list_dependencies(obj):
      used_names.add(name)
      if name in extra_dependencies:
        # Extra dependencies (except for `.signatures`, which is always added
        # when saving) should not have naming conflicts with dependencies
        # defined by the user.
        if name != signature_serialization.SIGNATURE_ATTRIBUTE_NAME:
          obj_identifier = obj._object_identifier  # pylint: disable=protected-access
          raise ValueError(
              f"Error when exporting object {obj} with identifier "
              f"'{obj_identifier}'. The object has an attribute named "
              f"'{name}', which is reserved. List of all reserved attributes: "
              f"{list(extra_dependencies.keys())}")

        yield base.TrackableReference(name, extra_dependencies[name])
      else:
        yield base.TrackableReference(name, dep)
    for name, dep in extra_dependencies.items():
      if name in used_names:
        continue
      yield base.TrackableReference(name, dep)

  def list_extra_dependencies(self, obj):
    return obj._list_extra_dependencies_for_serialization(  # pylint: disable=protected-access
        self._serialization_cache)

  def list_functions(self, obj):
    obj_functions = self._functions.get(obj, None)
    if obj_functions is None:
      obj_functions = obj._list_functions_for_serialization(  # pylint: disable=protected-access
          self._serialization_cache)
      self._functions[obj] = obj_functions
    return obj_functions


class _SaveableView(object):
  """Provides a frozen view over a trackable root.

  This class helps to create a single stable view over an object to save. The
  saving code should access properties and functions via this class and not via
  the original object as there are cases where an object construct their
  trackable attributes and functions dynamically per call and will yield
  different objects if invoked more than once.

  Changes to the graph, for example adding objects, must happen in
  `checkpoint_view` (an `_AugmentedGraphView`) before the `_SaveableView` is
  constructed. Changes after the `_SaveableView` has been constructed will be
  ignored.
  """

  def __init__(self, checkpoint_view, options, wrapped_functions=None):
    """Initializes a SaveableView.

    Args:
      checkpoint_view: A GraphView object.
      options: A SaveOptions instance.
      wrapped_functions: Dictionary that maps concrete functions to functions
        that do not capture cached variable values.
    """

    self.checkpoint_view = checkpoint_view
    self._options = options
    # Maps functions -> wrapped functions that capture variables
    self._wrapped_functions = wrapped_functions or {}
    # Run through the nodes in the object graph first for side effects of
    # creating variables.
    self._trace_all_concrete_functions()

    (self._trackable_objects, self.node_paths, self._node_ids,
     self._slot_variables, self.object_names) = (
         self.checkpoint_view.objects_ids_and_slot_variables_and_paths())

    self._initialize_save_and_restore_functions()
    self._initialize_nodes_and_concrete_functions()

    # Maps names of concrete functions in the object to names of wrapped
    # functions. When writing the SavedFunction protos, the names of the
    # wrapped functions should be used in place of the original functions.
    self.function_name_map = {
        compat.as_text(original.name): compat.as_text(wrapped.name)
        for original, wrapped in self._wrapped_functions.items()}
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
    checkpoint_factory_map = graph_view.get_checkpoint_factories_and_keys(
        self.object_names)
    self._saveable_objects_map = (
        _gen_save_and_restore_functions(checkpoint_factory_map))

  def _initialize_nodes_and_concrete_functions(self):
    """Creates graph with nodes for trackable objects and functions.

    Adds functions for each trackable object to `self.nodes` and associated
    concrete functions to `self.concrete_functions` for serialization.
    """
    self.nodes = list(self._trackable_objects)
    self.concrete_functions = []
    self.gradient_functions = []
    self.gradient_defs = []
    self._seen_function_names = set()
    self._untraced_functions = []

    for obj in self._trackable_objects:
      for function in self.checkpoint_view.list_functions(obj).values():
        self._add_function_to_graph(function)

      if obj in self._saveable_objects_map:
        for save_fn, restore_fn in self._saveable_objects_map[obj].values():
          self._add_function_to_graph(save_fn)
          self._add_function_to_graph(restore_fn)

    if self._untraced_functions:
      logging.warning(
          "Found untraced functions such as %s while saving (showing %d of %d)."
          " These functions will not be directly callable after loading.",
          ", ".join(self._untraced_functions[:_NUM_DISPLAY_UNTRACED_FUNCTIONS]),
          min(_NUM_DISPLAY_UNTRACED_FUNCTIONS, len(self._untraced_functions)),
          len(self._untraced_functions))

  @property
  def concrete_and_gradient_functions(self):
    return self.concrete_functions + self.gradient_functions

  def _add_function_to_graph(self, function):
    """Adds function to serialize to object graph."""
    # Updates self.nodes, self._node_ids, self.concrete_functions,
    # and self._untraced_functions.
    if function not in self._node_ids:
      self._node_ids[function] = len(self.nodes)
      # Add the function to nodes as well.
      self.nodes.append(function)
    if isinstance(function, def_function.Function):
      concrete_functions = (
          function._list_all_concrete_functions_for_serialization())  # pylint: disable=protected-access
    else:
      concrete_functions = [function]
    if not concrete_functions:
      self._untraced_functions.append(function._name)  # pylint: disable=protected-access
    for concrete_function in concrete_functions:
      if concrete_function.name not in self._seen_function_names:
        self.concrete_functions.append(concrete_function)
        self._seen_function_names.add(concrete_function.name)

  def _trace_all_concrete_functions(self):
    """Trace concrete functions to force side-effects.

    Lists the concrete functions in order to:
      - populate the cache for functions that have an input_signature
        and have not been called
      - force side effects of creation of concrete functions, e.g. create
        variables on first run.
    """
    for obj in self.checkpoint_view.list_objects():
      for function in self.checkpoint_view.list_functions(obj).values():
        if isinstance(function, def_function.Function):
          function._list_all_concrete_functions_for_serialization()  # pylint: disable=protected-access

  @property
  def root(self):
    return self.nodes[0]

  def fill_object_graph_proto(self, proto):
    """Populate the nodes, children and slot_variables of a SavedObjectGraph."""
    for node_id, node in enumerate(self.nodes):
      assert self._node_ids[node] == node_id
      object_proto = proto.nodes.add()
      object_proto.slot_variables.extend(self._slot_variables.get(node, ()))
      if isinstance(
          node,
          (def_function.Function, defun.ConcreteFunction, _CapturedConstant,
           _CapturedTensor)):
        continue
      for child in self.checkpoint_view.list_dependencies(node):
        child_proto = object_proto.children.add()
        child_proto.node_id = self._node_ids[child.ref]
        child_proto.local_name = child.name
      for local_name, ref_function in (
          self.checkpoint_view.list_functions(node).items()):
        child_proto = object_proto.children.add()
        child_proto.node_id = self._node_ids[ref_function]
        child_proto.local_name = local_name

      if node not in self._saveable_objects_map:
        continue

      for local_name, (save_fn, restore_fn) in (
          self._saveable_objects_map[node].items()):
        saveable_object_proto = object_proto.saveable_objects[local_name]
        saveable_object_proto.save_function = self._node_ids[save_fn]
        saveable_object_proto.restore_function = self._node_ids[restore_fn]

  def map_resources(self):
    """Makes new resource handle ops corresponding to existing resource tensors.

    Creates resource handle ops in the current default graph, whereas
    `accessible_objects` will be from an eager context. Resource mapping adds
    resource handle ops to the main GraphDef of a SavedModel, which allows the
    C++ loader API to interact with resources.

    Returns:
      A tuple of (object_map, resource_map, asset_info):
        object_map: A dictionary mapping from object in `accessible_objects` to
          replacement objects created to hold the new resource tensors.
        resource_map: A dictionary mapping from resource tensors extracted from
          `accessible_objects` to newly created resource tensors.
        asset_info: An _AssetInfo tuple describing external assets referenced
          from accessible_objects.
    """
    # Only makes sense when adding to the export Graph
    assert not context.executing_eagerly()
    # TODO(allenl): Handle MirroredVariables and other types of variables which
    # may need special casing.
    object_map = object_identity.ObjectIdentityDictionary()
    resource_map = {}
    asset_info = _AssetInfo(
        asset_defs=[],
        asset_initializers_by_resource={},
        asset_filename_map={},
        asset_index={})

    for node_id, obj in enumerate(self.nodes):
      if isinstance(obj, tracking.Asset):
        _process_asset(obj, asset_info, resource_map)
        self.captured_tensor_node_ids[obj.asset_path] = node_id
      elif isinstance(obj, base.Trackable):
        node_object_map, node_resource_map = obj._map_resources(self._options)  # pylint: disable=protected-access
        for capturable in node_resource_map.keys():
          self.captured_tensor_node_ids[capturable] = node_id
        object_map.update(node_object_map)
        resource_map.update(node_resource_map)

    for concrete_function in self.concrete_functions:
      if not concrete_function.graph.saveable:
        raise ValueError(
            (f"Unable to save function {concrete_function.name} for the "
             "following reason(s):\n" +
             "\n".join(concrete_function.graph.saving_errors)))
      for capture in concrete_function.captured_inputs:
        if (tensor_util.is_tf_type(capture) and
            capture.dtype not in _UNCOPIABLE_DTYPES and
            capture not in self.captured_tensor_node_ids):
          if hasattr(capture, "_cached_variable"):
            if concrete_function not in self._wrapped_functions:
              wrapped = self._wrapped_functions[concrete_function] = (
                  function_serialization.wrap_cached_variables(
                      concrete_function))
              self.function_name_map[compat.as_text(concrete_function.name)] = (
                  compat.as_text(wrapped.name))
            continue
          capture_constant_value = tensor_util.constant_value(capture)
          if capture_constant_value is None:
            raise ValueError(
                f"Unable to save function {concrete_function.name} because it "
                f"captures graph tensor {capture} from a parent function which "
                "cannot be converted to a constant with `tf.get_static_value`.")

          if numpy.prod(capture.shape.as_list()) > 1 and numpy.all(
              capture_constant_value == capture_constant_value.flat[0]):
            # For the common case of a constant array filled with the same
            # value, rebuidling the constant op specifically with the shape arg,
            # since otherwise the whole array is written into the node def,
            # causing performance and graph proto size issues (protos cannot be
            # bigger than 2GB).
            copied_tensor = constant_op.constant(
                capture_constant_value.flat[0],
                dtype=capture.dtype,
                shape=capture.shape)
          else:
            copied_tensor = constant_op.constant(capture_constant_value)

          node = _CapturedConstant(
              eager_tensor=capture, graph_tensor=copied_tensor)
          self.add_capture_and_node(capture, node)
          resource_map[capture] = copied_tensor

    self.concrete_functions = [
        self._wrapped_functions.get(x, x) for x in self.concrete_functions
    ]
    return object_map, resource_map, asset_info

  def add_capture_and_node(self, capture, node):
    node_id = len(self.nodes)
    self.nodes.append(node)
    self._node_ids[capture] = node_id
    self._node_ids[node] = node_id
    self.captured_tensor_node_ids[capture] = node_id
    return node_id


def _gen_save_and_restore_functions(checkpoint_factory_map):
  """Generates global and individual save/restore concrete functions.

  The global functions records the ops to save and restore the entire object to
  a file prefix, while the individual functions save and restore value tensors
  for resources.

  This function is intended to run on the output of
  `graph_view.get_checkpoint_factories_and_keys(object_names)`, which returns
  the generated a map of `_CheckpointFactoryData`.

  Args:
    checkpoint_factory_map: A dictionary mapping trackable objects to
      _CheckpointFactoryData.

  Returns:
    Tuple of (
      saveable_fn_map: Maps obj -> factory name -> (concrete save, restore)
      )
  """
  # Maps obj -> factory attribute_name -> (concrete save, concrete restore)
  # This
  saveable_fn_map = object_identity.ObjectIdentityDictionary()

  for obj, factory_data_list in checkpoint_factory_map.items():
    for factory_data in factory_data_list:
      saveable_factory = factory_data.factory
      checkpoint_key = factory_data.checkpoint_key
      attribute_name = factory_data.name

      # If object revives as a resource (or TPU/Mirrored) variable,
      # there is no need to trace the save and restore functions.
      if not (resource_variable_ops.is_resource_variable(obj) or
              resource_variable_ops.is_resource_variable(saveable_factory)):
        concrete_save, concrete_restore, saveable = (
            saveable_object_util.build_traceable_saveable(
                saveable_factory, checkpoint_key, obj))
        if not saveable:
          continue
        saveable_fn_map.setdefault(obj, {})[attribute_name] = (
            concrete_save, concrete_restore)
  return saveable_fn_map


def _tensor_dict_to_tensorinfo(tensor_dict):
  return {
      key: utils_impl.build_tensor_info_internal(value)
      for key, value in tensor_dict.items()
  }


def _map_captures_to_created_tensors(original_captures, resource_map):
  """Maps eager tensors captured by a function to Graph resources for export.

  Args:
    original_captures: A dictionary mapping from tensors captured by the
      function to interior placeholders for those tensors (inside the function
      body).
    resource_map: A dictionary mapping from resource tensors owned by the eager
      context to resource tensors in the exported graph.

  Returns:
    A list of stand-in tensors which belong to the exported graph, corresponding
    to the function's captures.

  Raises:
    AssertionError: If the function references a resource which is not part of
      `resource_map`.
  """
  export_captures = []
  for exterior, interior in original_captures:
    mapped_resource = resource_map.get(exterior, None)
    if mapped_resource is None:
      trackable_referrers = []
      # Try to figure out where the resource came from by iterating over objects
      # which reference it. This is slow and doesn't help us figure out how to
      # match it to other objects when loading the SavedModel as a checkpoint,
      # so we can't continue saving. But we can at least tell the user what
      # needs attaching.
      for primary_referrer in gc.get_referrers(exterior):
        if isinstance(primary_referrer, base.Trackable):
          trackable_referrers.append(primary_referrer)
        for secondary_referrer in gc.get_referrers(primary_referrer):
          if isinstance(secondary_referrer, base.Trackable):
            trackable_referrers.append(secondary_referrer)
      raise AssertionError(
          "Tried to export a function which references 'untracked' resource "
          f"{interior}. TensorFlow objects (e.g. tf.Variable) captured by "
          "functions must be 'tracked' by assigning them to an attribute of a "
          "tracked object or assigned to an attribute of the main object "
          "directly.\n\n Trackable Python objects referring to this tensor "
          "(from gc.get_referrers, limited to two hops):\n{}".format("\n".join(
              [repr(obj) for obj in trackable_referrers])))
    export_captures.append(mapped_resource)
  return export_captures


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


def _call_function_with_mapped_captures(function, args, resource_map):
  """Calls `function` in the exported graph, using mapped resource captures."""
  export_captures = _map_captures_to_created_tensors(function.graph.captures,
                                                     resource_map)
  # Calls the function quite directly, since we have new captured resource
  # tensors we need to feed in which weren't part of the original function
  # definition.
  # pylint: disable=protected-access
  outputs = function._call_flat(args, export_captures)
  # pylint: enable=protected-access
  return outputs


def _generate_signatures(signature_functions, resource_map):
  """Validates and calls `signature_functions` in the default graph.

  Args:
    signature_functions: A dictionary mapping string keys to concrete TensorFlow
      functions (e.g. from `signature_serialization.canonicalize_signatures`)
      which will be used to generate SignatureDefs.
    resource_map: A dictionary mapping from resource tensors in the eager
      context to resource tensors in the Graph being exported. This dictionary
      is used to re-bind resources captured by functions to tensors which will
      exist in the SavedModel.

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
    outputs = _call_function_with_mapped_captures(
        function, mapped_inputs, resource_map)
    signatures[signature_key] = signature_def_utils.build_signature_def(
        _tensor_dict_to_tensorinfo(exterior_argument_placeholders),
        _tensor_dict_to_tensorinfo(outputs),
        method_name=signature_constants.PREDICT_METHOD_NAME)
  return signatures


def _trace_resource_initializers(accessible_objects):
  """Create concrete functions from `CapturableResource` objects."""
  resource_initializers = []

  def _wrap_initializer(obj):
    obj._initialize()  # pylint: disable=protected-access
    return constant_op.constant(1.)  # Dummy control output

  def _wrap_obj_initializer(obj):
    return lambda: _wrap_initializer(obj)

  for obj in accessible_objects:
    if isinstance(obj, tracking.CapturableResource):
      resource_initializers.append(
          def_function.function(
              _wrap_obj_initializer(obj),
              # All inputs are captures.
              input_signature=[]).get_concrete_function())
  return resource_initializers


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


def _process_asset(trackable_asset, asset_info, resource_map):
  """Add `trackable_asset` to `asset_info` and `resource_map`."""
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
  # TODO(andresp): Instead of mapping 1-1 between trackable asset
  # and asset in the graph def consider deduping the assets that
  # point to the same file.
  asset_path_initializer = array_ops.placeholder(
      shape=original_path_tensor.shape,
      dtype=dtypes.string,
      name="asset_path_initializer")
  asset_variable = resource_variable_ops.ResourceVariable(
      asset_path_initializer)
  asset_info.asset_filename_map[path] = original_path
  asset_def = meta_graph_pb2.AssetFileDef()
  asset_def.filename = path
  asset_def.tensor_info.name = asset_path_initializer.name
  asset_info.asset_defs.append(asset_def)
  asset_info.asset_initializers_by_resource[original_path_tensor] = (
      asset_variable.initializer)
  asset_info.asset_index[trackable_asset] = len(asset_info.asset_defs) - 1
  resource_map[original_path_tensor] = asset_variable


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
            "with the option tf.saved_model.SaveOptions(custom_gradients=False)"
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
  accessible_objects = saveable_view.nodes
  resource_initializer_functions = _trace_resource_initializers(
      accessible_objects)
  exported_graph = ops.Graph()
  resource_initializer_ops = []
  with exported_graph.as_default():
    object_map, resource_map, asset_info = saveable_view.map_resources()
    for resource_initializer_function in resource_initializer_functions:
      asset_dependencies = []
      for capture in resource_initializer_function.graph.external_captures:
        asset_initializer = asset_info.asset_initializers_by_resource.get(
            capture, None)
        if asset_initializer is not None:
          asset_dependencies.append(asset_initializer)
      with ops.control_dependencies(asset_dependencies):
        resource_initializer_ops.append(
            _call_function_with_mapped_captures(resource_initializer_function,
                                                [], resource_map))
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
  saver = functional_saver.MultiDeviceSaver(
      saveable_view.checkpoint_view.frozen_saveable_objects(
          object_map=object_map, to_graph=exported_graph,
          call_with_mapped_captures=functools.partial(
              _call_function_with_mapped_captures, resource_map=resource_map)))

  with exported_graph.as_default():
    signatures = _generate_signatures(signature_functions, resource_map)
    for concrete_function in saveable_view.concrete_functions:
      concrete_function.add_to_graph()
    if save_custom_gradients:
      _trace_gradient_functions(exported_graph, saveable_view)
    saver_def = saver.to_proto()
    meta_graph_def.saver_def.CopyFrom(saver_def)
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


def _serialize_object_graph(saveable_view, asset_file_def_index):
  """Save a SavedObjectGraph proto for `root`."""
  # SavedObjectGraph is similar to the TrackableObjectGraph proto in the
  # checkpoint. It will eventually go into the SavedModel.
  proto = saved_object_graph_pb2.SavedObjectGraph()
  saveable_view.fill_object_graph_proto(proto)

  coder = nested_structure_coder.StructureCoder()
  for concrete_function in saveable_view.concrete_and_gradient_functions:
    name = compat.as_text(concrete_function.name)
    name = saveable_view.function_name_map.get(name, name)
    serialized = function_serialization.serialize_concrete_function(
        concrete_function, saveable_view.captured_tensor_node_ids, coder)
    if serialized is not None:
      proto.concrete_functions[name].CopyFrom(serialized)

  for obj, obj_proto in zip(saveable_view.nodes, proto.nodes):
    _write_object_proto(obj, obj_proto, asset_file_def_index,
                        saveable_view.function_name_map)
  return proto


def _write_object_proto(obj, proto, asset_file_def_index, function_name_map):
  """Saves an object into SavedObject proto."""
  if isinstance(obj, tracking.Asset):
    proto.asset.SetInParent()
    proto.asset.asset_file_def_index = asset_file_def_index[obj]
  elif resource_variable_ops.is_resource_variable(obj):
    options = save_context.get_save_options()
    obj._write_object_proto(proto, options)  # pylint: disable=protected-access
  elif isinstance(obj, def_function.Function):
    proto.function.CopyFrom(function_serialization.serialize_function(
        obj, function_name_map))
  elif isinstance(obj, defun.ConcreteFunction):
    proto.bare_concrete_function.CopyFrom(
        function_serialization.serialize_bare_concrete_function(
            obj, function_name_map))
  elif isinstance(obj, _CapturedConstant):
    proto.constant.operation = obj.graph_tensor.op.name
  elif isinstance(obj, _CapturedTensor):
    proto.captured_tensor.name = obj.name
    proto.captured_tensor.concrete_function = obj.concrete_function
  elif isinstance(obj, tracking.CapturableResource):
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

  registered_name = registration.get_registered_name(obj)
  if registered_name:
    proto.registered_name = registered_name
    serialized_user_proto = obj._serialize_to_proto()  # pylint: disable=protected-access
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

  The `obj` must inherit from the [`Trackable` class](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/tracking/base.py#L591).

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

  While Keras has its own [saving and loading API](https://www.tensorflow.org/guide/keras/save_and_serialize),
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
    signatures: Optional, one of three types:
      * a `tf.function` with an input signature specified, which will use the
        default serving signature key,
      * the result of `f.get_concrete_function` on a `@tf.function`-decorated
        function `f`, in which case `f` will be used to generate a signature for
        the SavedModel under the default serving signature key,
      * a dictionary, which maps signature keys to either `tf.function`
        instances with input signatures or concrete functions. Keys of such a
        dictionary may be arbitrary strings, but will typically be from the
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
    experimental_skip_checkpoint: If set to `True`, the checkpoint will not
      be written.

  Returns:
    A tuple of (a list of saved nodes in the order they are serialized to the
      `SavedObjectGraph`, dictionary mapping nodes to one possible path from
      the root node to the key node)
  """
  options = options or save_options.SaveOptions()
  # TODO(allenl): Factor out some subset of SavedModelBuilder which is 2.x
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


def _build_meta_graph_impl(obj,
                           signatures,
                           options,
                           meta_graph_def=None):
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

  checkpoint_graph_view = _AugmentedGraphView(obj)
  if signatures is None:
    signatures = signature_serialization.find_function_to_export(
        checkpoint_graph_view)

  signatures, wrapped_functions = (
      signature_serialization.canonicalize_signatures(signatures))
  signature_serialization.validate_saveable_view(checkpoint_graph_view)
  signature_map = signature_serialization.create_signature_map(signatures)
  checkpoint_graph_view.add_object(
      parent_node=checkpoint_graph_view.root,
      name_in_parent=signature_serialization.SIGNATURE_ATTRIBUTE_NAME,
      subgraph_root=signature_map)

  # Use _SaveableView to provide a frozen listing of properties and functions.
  saveable_view = _SaveableView(checkpoint_graph_view, options,
                                wrapped_functions)
  object_saver = util.TrackableSaver(checkpoint_graph_view)
  asset_info, exported_graph = _fill_meta_graph_def(
      meta_graph_def, saveable_view, signatures,
      options.namespace_whitelist, options.experimental_custom_gradients)
  if options.function_aliases:
    function_aliases = meta_graph_def.meta_info_def.function_aliases
    for alias, func in options.function_aliases.items():
      for fdef in func._stateful_fn._function_cache.all_values():  # pylint: disable=protected-access
        function_aliases[fdef.name] = alias
      for fdef in func._stateless_fn._function_cache.all_values():  # pylint: disable=protected-access
        function_aliases[fdef.name] = alias

  object_graph_proto = _serialize_object_graph(
      saveable_view, asset_info.asset_index)
  meta_graph_def.object_graph_def.CopyFrom(object_graph_proto)

  return (meta_graph_def, exported_graph, object_saver, asset_info,
          saveable_view.nodes, saveable_view.node_paths)


def _build_meta_graph(obj,
                      signatures,
                      options,
                      meta_graph_def=None):
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
    object_saver: `util.TrackableSaver` of the `obj` and its dependencies.
    asset_info: `_AssetInfo` tuple containing external assets in the `obj`.
  """

  with save_context.save_context(options):
    return _build_meta_graph_impl(obj, signatures, options, meta_graph_def)
