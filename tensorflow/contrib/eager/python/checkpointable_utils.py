"""Utilities for working with Checkpointable objects."""
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.contrib.eager.proto import checkpointable_object_graph_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import checkpointable as core_checkpointable
from tensorflow.python.training import optimizer as optimizer_lib
from tensorflow.python.training import saver as saver_lib


_ESCAPE_CHAR = "."  # For avoiding conflicts with user-specified names.

# Keyword for identifying that the next bit of a checkpoint variable name is a
# slot name. Checkpoint names for slot variables look like:
#
#   <path to variable>/<_OPTIMIZER_SLOTS_NAME>/<path to optimizer>/<slot name>
#
# Where <path to variable> is a full path from the checkpoint root to the
# variable being slotted for.
_OPTIMIZER_SLOTS_NAME = _ESCAPE_CHAR + "OPTIMIZER_SLOT"
# Keyword for separating the path to an object from the name of an
# attribute in checkpoint names. Used like:
#   <path to variable>/<_OBJECT_ATTRIBUTES_NAME>/<name of attribute>
_OBJECT_ATTRIBUTES_NAME = _ESCAPE_CHAR + "ATTRIBUTES"
# Key where the object graph proto is saved in a TensorBundle
_OBJECT_GRAPH_PROTO_KEY = "_CHECKPOINTABLE_OBJECT_GRAPH"


# TODO(allenl): If this ends up in a public API, consider adding LINT.IfChange
# or consolidating the implementation with get_variable.
def _default_getter(name, shape, dtype, initializer=None,
                    partition_info=None, **kwargs):
  """A pared-down version of get_variable which does not reuse variables."""
  dtype = dtypes.as_dtype(dtype)
  shape_object = tensor_shape.as_shape(shape)
  with ops.init_scope():
    if initializer is None:
      initializer, initializing_from_value = (
          variable_scope._get_default_variable_store()._get_default_initializer(  # pylint: disable=protected-access
              name=name, shape=shape_object, dtype=dtype))
    else:
      initializing_from_value = not callable(initializer)
    # Same logic as get_variable
    variable_dtype = dtype.base_dtype
    if initializing_from_value:
      if shape is not None:
        raise ValueError("If initializer is a constant, do not specify shape.")
      initial_value = initializer
    else:
      # Instantiate initializer if provided initializer is a type object.
      if isinstance(initializer, type(init_ops.Initializer)):
        initializer = initializer(dtype=dtype)
      def initial_value():
        return initializer(
            shape_object.as_list(), dtype=dtype, partition_info=partition_info)
    return resource_variable_ops.ResourceVariable(
        initial_value=initial_value,
        name=name,
        dtype=variable_dtype,
        **kwargs
    )


def add_variable(checkpointable, name, shape=None, dtype=dtypes.float32,
                 initializer=None):
  """Add a variable to a Checkpointable with no scope influence."""
  return checkpointable._add_variable_with_custom_getter(  # pylint: disable=protected-access
      name=name, shape=shape, dtype=dtype,
      initializer=initializer, getter=_default_getter)


def _breadth_first_checkpointable_traversal(root_checkpointable):
  """Find shortest paths to all variables owned by dependencies of root."""
  bfs_sorted = []
  to_visit = collections.deque([root_checkpointable])
  path_to_root = {root_checkpointable: ()}
  while to_visit:
    current_checkpointable = to_visit.popleft()
    current_checkpointable._maybe_initialize_checkpointable()  # pylint: disable=protected-access
    bfs_sorted.append(current_checkpointable)
    for child_checkpointable in (
        current_checkpointable._checkpoint_dependencies):  # pylint: disable=protected-access
      if child_checkpointable.ref not in path_to_root:
        path_to_root[child_checkpointable.ref] = (
            path_to_root[current_checkpointable] + (child_checkpointable,))
        to_visit.append(child_checkpointable.ref)
  return bfs_sorted, path_to_root


def _escape_local_name(name):
  # We need to support slashes in local names for compatibility, since this
  # naming scheme is being patched in to things like Layer.add_variable where
  # slashes were previously accepted. We also want to use slashes to indicate
  # edges traversed to reach the variable, so we escape forward slashes in
  # names.
  return (name.replace(_ESCAPE_CHAR, _ESCAPE_CHAR + _ESCAPE_CHAR)
          .replace(r"/", _ESCAPE_CHAR + "S"))


def _object_prefix_from_path(path_to_root):
  return "/".join(
      (_escape_local_name(checkpointable.name)
       for checkpointable in path_to_root))


def _slot_variable_naming_for_optimizer(optimizer_path):
  """Make a function for naming slot variables in an optimizer."""
  # Name slot variables:
  #
  #   <variable name>/<_OPTIMIZER_SLOTS_NAME>/<optimizer path>/<slot name>
  #
  # where <variable name> is exactly the checkpoint name used for the original
  # variable, including the path from the checkpoint root and the local name in
  # the object which owns it. Note that we only save slot variables if the
  # variable it's slotting for is also being saved.

  optimizer_identifier = "/%s/%s/" % (_OPTIMIZER_SLOTS_NAME, optimizer_path)

  def _name_slot_variable(variable_path, slot_name):
    """With an optimizer specified, name a slot variable."""
    return (variable_path
            + optimizer_identifier
            + _escape_local_name(slot_name))

  return _name_slot_variable


def _serialize_slot_variables(checkpointable_objects, node_ids, object_names):
  """Gather and name slot variables."""
  non_slot_objects = list(checkpointable_objects)
  slot_variables = {}
  for checkpointable in non_slot_objects:
    if isinstance(checkpointable, optimizer_lib.Optimizer):
      naming_scheme = _slot_variable_naming_for_optimizer(
          optimizer_path=object_names[checkpointable])
      slot_names = checkpointable.get_slot_names()
      for slot_name in slot_names:
        for original_variable_node_id, original_variable in enumerate(
            non_slot_objects):
          try:
            slot_variable = checkpointable.get_slot(
                original_variable, slot_name)
          except AttributeError:
            slot_variable = None
          if slot_variable is None:
            continue
          slot_variable._maybe_initialize_checkpointable()  # pylint: disable=protected-access
          if slot_variable._checkpoint_dependencies:  # pylint: disable=protected-access
            # TODO(allenl): Gather dependencies of slot variables.
            raise NotImplementedError(
                "Currently only variables with no dependencies can be saved as "
                "slot variables. File a feature request if this limitation "
                "bothers you.")
          if slot_variable in node_ids:
            raise NotImplementedError(
                "A slot variable was re-used as a dependency of a "
                "Checkpointable object. This is not currently allowed. File a "
                "feature request if this limitation bothers you.")
          checkpoint_name = naming_scheme(
              variable_path=object_names[original_variable],
              slot_name=slot_name)
          object_names[slot_variable] = checkpoint_name
          slot_variable_node_id = len(checkpointable_objects)
          node_ids[slot_variable] = slot_variable_node_id
          checkpointable_objects.append(slot_variable)
          slot_variable_proto = (
              checkpointable_object_graph_pb2.CheckpointableObjectGraph
              .Object.SlotVariableReference(
                  slot_name=slot_name,
                  original_variable_node_id=original_variable_node_id,
                  slot_variable_node_id=slot_variable_node_id))
          slot_variables.setdefault(checkpointable, []).append(
              slot_variable_proto)
  return slot_variables


def _serialize_checkpointables(
    checkpointable_objects, node_ids, object_names, slot_variables):
  """Name non-slot `Checkpointable`s and add them to `object_graph_proto`."""
  object_graph_proto = (
      checkpointable_object_graph_pb2.CheckpointableObjectGraph())
  named_saveables = {}

  for checkpoint_id, checkpointable in enumerate(checkpointable_objects):
    assert node_ids[checkpointable] == checkpoint_id
    object_proto = object_graph_proto.nodes.add()
    object_proto.slot_variables.extend(slot_variables.get(checkpointable, ()))
    object_name = object_names[checkpointable]
    for name, saveable in (
        checkpointable._gather_tensors_for_checkpoint().items()):  # pylint: disable=protected-access
      attribute = object_proto.attributes.add()
      attribute.name = name
      attribute.checkpoint_key = "%s/%s/%s" % (
          object_name, _OBJECT_ATTRIBUTES_NAME, _escape_local_name(name))
      # Figure out the name-based Saver's name for this variable.
      saver_dict = saver_lib.BaseSaverBuilder.OpListToDict(
          [saveable], convert_variable_to_tensor=False)
      attribute.full_name, = saver_dict.keys()
      named_saveables[attribute.checkpoint_key] = saveable

    for child in checkpointable._checkpoint_dependencies:  # pylint: disable=protected-access
      child_proto = object_proto.children.add()
      child_proto.node_id = node_ids[child.ref]
      child_proto.local_name = child.name

  return named_saveables, object_graph_proto


def _serialize_object_graph(root_checkpointable):
  """Determine checkpoint keys for variables and build a serialized graph.

  Non-slot variables are keyed based on a shortest path from the root saveable
  to the object which owns the variable (i.e. the one which called
  `Checkpointable._add_variable` to create it).

  Slot variables are keyed based on a shortest path to the variable being
  slotted for, a shortest path to their optimizer, and the slot name.

  Args:
    root_checkpointable: A `Checkpointable` object whose variables (including
      the variables of dependencies, recursively) should be saved.

  Returns:
    A tuple of (named_variables, object_graph_proto):
      named_variables: A dictionary mapping names to variable objects.
      object_graph_proto: A CheckpointableObjectGraph protocol buffer containing
        the serialized object graph and variable references.

  Raises:
    ValueError: If there are invalid characters in an optimizer's slot names.
  """
  checkpointable_objects, path_to_root = (
      _breadth_first_checkpointable_traversal(root_checkpointable))
  object_names = {
      obj: _object_prefix_from_path(path)
      for obj, path in path_to_root.items()}
  node_ids = {node: node_id for node_id, node
              in enumerate(checkpointable_objects)}
  slot_variables = _serialize_slot_variables(
      checkpointable_objects=checkpointable_objects,
      node_ids=node_ids,
      object_names=object_names)
  return _serialize_checkpointables(
      checkpointable_objects=checkpointable_objects,
      node_ids=node_ids,
      object_names=object_names,
      slot_variables=slot_variables)


class _NoRestoreSaveable(saver_lib.BaseSaverBuilder.SaveableObject):

  def __init__(self, tensor, name):
    spec = saver_lib.BaseSaverBuilder.SaveSpec(tensor, "", name)
    super(_NoRestoreSaveable, self).__init__(tensor, [spec], name)

  def restore(self, restored_tensors, restored_shapes):
    return control_flow_ops.no_op()


def save(file_prefix, root_checkpointable, checkpoint_number=None,
         session=None):
  """Save a training checkpoint.

  Args:
    file_prefix: A prefix to use for the checkpoint filenames
      (/path/to/directory/and_a_prefix). Names are generated based on this
      prefix and the global step, if provided.
    root_checkpointable: A Checkpointable object to save. The checkpoint
      includes variables created by this object and any Checkpointable objects
      it depends on.
    checkpoint_number: An integer variable or Tensor, used to number
      checkpoints. Typically this value is saved along with other variables in
      training checkpoints, which will happen automatically if it was created by
      `root_checkpointable` or one of its dependencies (via
      `Checkpointable._add_variable`).
    session: The session to evaluate variables in. Ignored when executing
      eagerly. If not provided when graph building, the default session is used.

  Returns:
    The full path to the checkpoint.
  """
  named_variables, serialized_graph = _serialize_object_graph(
      root_checkpointable)
  if context.in_graph_mode():
    if session is None:
      session = ops.get_default_session()
  else:
    session = None
  assert _OBJECT_GRAPH_PROTO_KEY not in named_variables
  # TODO(allenl): Feed rather than embedding a constant.
  named_variables[_OBJECT_GRAPH_PROTO_KEY] = _NoRestoreSaveable(
      tensor=constant_op.constant(
          serialized_graph.SerializeToString(), dtype=dtypes.string),
      name=_OBJECT_GRAPH_PROTO_KEY)
  with ops.device("/device:CPU:0"):
    save_path = saver_lib.Saver(var_list=named_variables).save(
        sess=session,
        save_path=file_prefix,
        write_meta_graph=False,
        global_step=checkpoint_number)
  return save_path


class CheckpointLoadStatus(object):
  """Checks the status of checkpoint loading."""

  def __init__(self, checkpoint):
    self._checkpoint = checkpoint

  def assert_consumed(self):
    """Asserts that all objects in the checkpoint have been created/matched."""
    for node_id, node in enumerate(self._checkpoint.object_graph_proto.nodes):
      checkpointable = self._checkpoint.object_by_proto_id.get(node_id, None)
      if checkpointable is None:
        raise AssertionError("Unresolved object in checkpoint: %s" % (node,))
      if checkpointable._update_uid < self._checkpoint.restore_uid:  # pylint: disable=protected-access
        raise AssertionError(
            "Object not assigned a value from checkpoint: %s" % (node,))
    if self._checkpoint.slot_restorations:
      # Sanity check; this collection should be clear if everything has been
      # restored.
      raise AssertionError("Unresolved slot restorations: %s" % (
          self._checkpoint.slot_restorations,))
    return self

  @property
  def restore_ops(self):
    """Operations to restore objects in the dependency graph."""
    return self._checkpoint.restore_ops


def restore(save_path, root_checkpointable, session=None):
  """Restore a training checkpoint.

  Restores the values of variables created with `Checkpointable._add_variable`
  in `root_checkpointable` and any objects that it tracks (transitive). Either
  assigns values immediately if variables to restore have been created already,
  or defers restoration until the variables are created. Dependencies added to
  `root_checkpointable` after this call will be matched if they have a
  corresponding object in the checkpoint.

  When building a graph, restorations are added to the graph but not run. A
  session is required to retrieve checkpoint metadata.

  To disallow deferred loading, assert immediately that all checkpointed
  variables have been matched to variable objects:

  ```python
  restore(path, root).assert_consumed()
  ```

  An exception will be raised unless every object was matched and its variables
  already exist.

  When graph building, `assert_consumed()` indicates that all of the restore ops
  which will be created for this checkpoint have been created. They are
  available in the `restore_ops` property of the status object:

  ```python
  session.run(restore(path, root).assert_consumed().restore_ops)
  ```

  If the checkpoint has not been consumed completely, then the list of
  `restore_ops` will grow as more objects are added to the dependency graph.

  Args:
    save_path: The path to the checkpoint, as returned by `save` or
      `tf.train.latest_checkpoint`. If None (as when there is no latest
      checkpoint for `tf.train.latest_checkpoint` to return), does nothing.
    root_checkpointable: The root of the object graph to restore. Variables to
      restore need not have been created yet, but all dependencies on other
      `Checkpointable` objects should already be declared. Objects in the
      dependency graph are matched to objects in the checkpointed graph, and
      matching objects have their variables restored (or the checkpointed values
      saved for eventual restoration when the variable is created).
    session: The session to retrieve metadata with. Ignored when executing
      eagerly. If not provided when graph building, the default session is used.
  Returns:
    A `CheckpointLoadStatus` object, which can be used to make assertions about
    the status of checkpoint restoration and fetch restore ops.
  """
  if save_path is None:
    return
  if context.in_graph_mode():
    if session is None:
      session = ops.get_default_session()
  else:
    session = None
  object_graph_string, = io_ops.restore_v2(
      prefix=save_path,
      tensor_names=[_OBJECT_GRAPH_PROTO_KEY],
      shape_and_slices=[""],
      dtypes=[dtypes.string],
      name="object_graph_proto_read")
  if session is not None:
    object_graph_string = session.run(object_graph_string)
  else:
    object_graph_string = object_graph_string.numpy()
  object_graph_proto = (
      checkpointable_object_graph_pb2.CheckpointableObjectGraph())
  object_graph_proto.ParseFromString(object_graph_string)
  checkpoint = core_checkpointable._Checkpoint(  # pylint: disable=protected-access
      object_graph_proto=object_graph_proto,
      save_path=save_path)
  core_checkpointable._CheckpointPosition(  # pylint: disable=protected-access
      checkpoint=checkpoint, proto_id=0).restore(root_checkpointable)
  load_status = CheckpointLoadStatus(checkpoint)
  return load_status
