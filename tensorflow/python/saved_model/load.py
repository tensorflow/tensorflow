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
"""Import a checkpointable object from a SavedModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import function_deserialization
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import saved_object_graph_pb2
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.training.checkpointable import base
from tensorflow.python.training.checkpointable import graph_view
from tensorflow.python.training.checkpointable import tracking
from tensorflow.python.training.checkpointable import util
from tensorflow.python.util import compat


class _Loader(object):
  """Helper class to load an object-based SavedModel."""

  def __init__(self, object_graph_proto, saved_model_proto, export_dir):
    meta_graph = saved_model_proto.meta_graphs[0]
    self._asset_file_def = meta_graph.asset_file_def
    self._operation_attributes = {
        node.name: node.attr for node in meta_graph.graph_def.node}
    self._proto = object_graph_proto
    self._export_dir = export_dir
    self._concrete_functions = (
        function_deserialization.load_function_def_library(
            meta_graph.graph_def.library))
    self._load_all()
    self._setup_functions()
    self._restore_checkpoint()

  def _setup_concrete_function(self, proto, concrete_function, coder):
    """Setup captured tensors and outputs for a single concrete function."""
    bound_inputs = [
        self._get_tensor_from_node(node_id)
        for node_id in proto.bound_inputs]
    bound_variables = [
        self._nodes[node_id]
        for node_id in proto.bound_inputs
        if self._proto.nodes[node_id].WhichOneof("kind") == "variable"
    ]
    # TODO(andresp): This is only injecting the captured inputs into the
    # concrete function, note that we did not modify the FuncGraph
    # itself.
    concrete_function._captured_inputs = bound_inputs  # pylint: disable=protected-access
    concrete_function._func_graph.variables = bound_variables  # pylint: disable=protected-access
    # By setting the structured_outputs directly, we can rely on this
    # function_lib.ConcreteFunction object to perform the output repacking
    # logic. The only limitation of that logic is that it only works
    # with output that is convertible to Tensors and the conversion
    # always happens. For example tf.TensorShape([2, 3]) will be
    # converted to Tensor representing [2, 3].
    original_outputs = coder.decode_proto(proto.output_signature)
    # The original_outputs here had Tensors converted to TensorSpecs, so
    # the restored function's structured_outputs field will not be
    # exactly the same. Fortunately the repacking logic cares only about
    # the structure.
    # TODO(vbardiovsky): Should we just replicate the structures, with
    # Nones instead of real objects?
    concrete_function._func_graph.structured_outputs = original_outputs  # pylint: disable=protected-access
    concrete_function._func_graph.structured_input_signature = (  # pylint: disable=protected-access
        coder.decode_proto(proto.canonicalized_input_signature))

  def _setup_functions(self):
    """Setup captures and output structure in restored functions."""
    coder = nested_structure_coder.StructureCoder()
    for name, concrete_function_proto in self._proto.concrete_functions.items():
      self._setup_concrete_function(
          concrete_function_proto,
          self._concrete_functions[name],
          coder)

  def _get_tensor_from_node(self, node_id):
    obj = self._nodes[node_id]
    if resource_variable_ops.is_resource_variable(obj):
      return obj.handle
    elif isinstance(obj, tracking.TrackableAsset):
      return obj.asset_path.handle
    elif tensor_util.is_tensor(obj):
      return obj
    raise ValueError("Can't convert node %s to tensor" % (type(obj)))

  def _load_all(self):
    """Load all saved objects and wire their properties."""
    self._nodes = []
    node_setters = []
    for proto in self._proto.nodes:
      node, setter = self._recreate(proto)
      self._nodes.append(node)
      node_setters.append(setter)
    # After creating the objects, construct the edges between the objects.
    for obj, object_proto, setter in zip(self._nodes, self._proto.nodes,
                                         node_setters):
      for reference in object_proto.children:
        setter(obj, reference.local_name, self._nodes[reference.node_id])
        # Note: if an object has an attribute `__call__` add a class method
        # that allows `obj()` syntax to work. This is done per-instance to
        # allow `callable` to be used to find out if an object is callable.
        if reference.local_name == "__call__":
          setattr(type(obj), "__call__", _call_attribute)

  def _restore_checkpoint(self):
    """Load state from checkpoint into the deserialized objects."""
    variables_path = saved_model_utils.get_variables_path(self._export_dir)
    # TODO(andresp): Clean use of private methods of CheckpointableSaver.
    # pylint: disable=protected-access
    saver = util.CheckpointableSaver(graph_view.ObjectGraphView(self.get(0)))
    saver._file_prefix_placeholder = constant_op.constant(variables_path)
    load_status = saver.restore(variables_path)
    load_status.assert_existing_objects_matched()
    checkpoint = load_status._checkpoint

    # When running in eager mode, the `restore` call above has already run and
    # restored the state of checkpointables, call `position.restore_ops()` will
    # return an empty list as there is nothing left to do. In graph mode, that
    # will return the list of ops that must run to restore the object on that
    # position. We have to wire them in the initializers of the objects so that
    # they get initialized properly when using common practices (e.g. the ones
    # used by ManagedSession) without further user action.
    for object_id, obj in dict(checkpoint.object_by_proto_id).items():
      position = base.CheckpointPosition(checkpoint=checkpoint,
                                         proto_id=object_id)
      restore_ops = position.restore_ops()
      if restore_ops:
        if resource_variable_ops.is_resource_variable(obj):
          obj._initializer_op = restore_ops
        else:
          raise NotImplementedError(
              ("Missing functionality to restore state of object "
               "%r from the checkpoint." % obj))

  def get(self, node_id):
    return self._nodes[node_id]

  def _recreate(self, proto):
    """Creates a Python object from a SavedObject protocol buffer."""
    factory = {
        "user_object": lambda: self._recreate_user_object(proto.user_object),
        "asset": lambda: self._recreate_asset(proto.asset),
        "function": lambda: self._recreate_function(proto.function),
        "bare_concrete_function": functools.partial(
            self._recreate_bare_concrete_function,
            proto.bare_concrete_function),
        "variable": lambda: self._recreate_variable(proto.variable),
        "constant": lambda: self._recreate_constant(proto.constant),
    }
    kind = proto.WhichOneof("kind")
    if kind not in factory:
      raise ValueError("Unknown SavedObject type: %r" % kind)
    return factory[kind]()

  def _recreate_user_object(self, proto):
    """Instantiates a SavedUserObject."""
    looked_up = revived_types.deserialize(proto)
    if looked_up is None:
      # Note: each user object has its own class. This allows to make each one
      # individually callable by adding a `__call__` method to the classes of
      # the objects instances that have a `__call__` property.

      class _UserObject(tracking.AutoCheckpointable):
        pass

      return _UserObject(), setattr
    return looked_up

  def _recreate_asset(self, proto):
    filename = os.path.join(
        saved_model_utils.get_assets_dir(self._export_dir),
        self._asset_file_def[proto.asset_file_def_index].filename)
    return tracking.TrackableAsset(filename), setattr

  def _recreate_function(self, proto):
    return function_deserialization.recreate_function(
        proto, self._concrete_functions), setattr

  def _recreate_bare_concrete_function(self, proto):
    return function_deserialization.setup_bare_concrete_function(
        proto, self._concrete_functions), setattr

  def _recreate_variable(self, proto):
    # TODO(andresp): Can we use the checkpointed value as initializer?
    dummy_value = init_ops.Zeros(dtype=proto.dtype)(shape=proto.shape)
    return variables.Variable(dummy_value, trainable=proto.trainable), setattr

  def _recreate_constant(self, proto):
    tensor_proto = self._operation_attributes[proto.operation]["value"].tensor
    imported_constant = constant_op.constant(
        tensor_util.MakeNdarray(tensor_proto))
    return imported_constant, setattr


def _call_attribute(instance, *args, **kwargs):
  return instance.__call__(*args, **kwargs)


def _load_saved_object_graph_proto(filename):
  with file_io.FileIO(filename, "rb") as f:
    contents = f.read()
    return saved_object_graph_pb2.SavedObjectGraph.FromString(contents)


def load(export_dir):
  """Load a SavedModel from `export_dir`."""
  saved_model_proto = loader_impl.parse_saved_model(export_dir)
  object_graph_filename = os.path.join(
      compat.as_bytes(export_dir),
      compat.as_bytes(constants.EXTRA_ASSETS_DIRECTORY),
      compat.as_bytes("object_graph.pb"))
  if file_io.file_exists(object_graph_filename):
    object_graph_proto = _load_saved_object_graph_proto(object_graph_filename)
    with ops.init_scope():
      loader = _Loader(object_graph_proto,
                       saved_model_proto,
                       export_dir)
      root = loader.get(0)
  else:
    raise NotImplementedError(
        "Currently only SavedModels exported with `tf.saved_model.save` may be "
        "imported. Other SavedModels may eventually be supported via load().")
  return root
