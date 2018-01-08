"""An object-local variable management scheme."""
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
import re

from tensorflow.contrib.eager.proto import checkpointable_object_graph_pb2
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import optimizer as optimizer_lib
from tensorflow.python.training import saver as saver_lib

_CheckpointableReference = collections.namedtuple(
    "_CheckpointableReference",
    [
        "name",  # The local name if explicitly specified, else None.
        "local_uid",  # 0 for the first dependency, 1 for the next, ... Used for
        # routing checkpointed variables to their correct
        # Checkpointables when "name" is not set (see docstring of
        # `track_checkpointable`).
        "ref"  # The Checkpointable object being referenced.
    ])

_OwnedVariable = collections.namedtuple(
    "_OwnedVariable",
    [
        "name",  # The variable's (local) name.
        "variable"  # The owned variable object.
    ])

# Validation regular expression for the local names of Checkpointable
# objects. In particular, disallows "/" in names, and reserves
# underscore-prefixed names.
_VALID_LOCAL_NAME = re.compile(r"^[A-Za-z0-9.][A-Za-z0-9_.-]*$")

# Keyword for identifying that the next bit of a checkpoint variable name is a
# slot name. May not be the local name of a checkpointable. Checkpoint names for
# slot variables look like:
#
#   <path to variable>/<_OPTIMIZER_SLOTS_NAME>/<path to optimizer>/<slot name>
#
# Where <path to variable> is a full path from the checkpoint root to the
# variable being slotted for.
_OPTIMIZER_SLOTS_NAME = "_OPTIMIZER_SLOT"


class Checkpointable(object):
  """Manages variables and dependencies on other objects.

  To make reliable checkpoints, all `Checkpointable`s on which this object
  depends must be registered in the constructor using `track_checkpointable` in
  a deterministic order, and if possible they should be named. Variables may be
  created using `add_variable` outside of the constructor and in any order, but
  only these variables will be saved.
  """

  def __init__(self):
    # Basically less useful OrderedDicts but without the reference cycles.
    # TODO(allenl): Switch these to OrderedDict once TensorFlow supports only
    # Python 3.6+.
    self._checkpoint_dependencies = []  # A list of _CheckpointableReference
    # objects.
    self._dependency_names = set()
    self._owned_variables = []  # A list of _OwnedVariable objects.
    self._owned_variable_names = set()

  def add_variable(self, name, shape, dtype=None, initializer=None, **kwargs):
    """Create a new variable object to be saved with this `Checkpointable`.

    If the user has requested that this object or another `Checkpointable` which
    depends on this object be restored from a checkpoint (deferred loading
    before variable object creation), `initializer` may be ignored and the value
    from the checkpoint used instead.

    Args:
      name: A name for the variable. Must be unique within this object.
      shape: The shape of the variable.
      dtype: The data type of the variable.
      initializer: The initializer to use. Ignored if deferred loading has been
        requested.
      **kwargs: Passed to get_variable.

    Returns:
      The new variable object.

    Raises:
      ValueError: If the variable name is not unique.
    """
    if name in self._owned_variable_names:
      raise ValueError(
          ("A variable named '%s' already exists in this Checkpointable, but "
           "Checkpointable.add_variable called to create another with "
           "that name. Variable names must be unique within a Checkpointable "
           "object.") % (name,))
    if "getter" in kwargs:
      # Allow the getter to be overridden, typically because there is a need for
      # compatibility with some other variable creation mechanism. This should
      # be relatively uncommon in user code.
      getter = kwargs.pop("getter")
    else:
      getter = variable_scope.get_variable
    # TODO(allenl): handle deferred loading
    new_variable = getter(
        name=name, shape=shape, dtype=dtype, initializer=initializer, **kwargs)
    self._owned_variables.append(
        _OwnedVariable(name=name, variable=new_variable))
    self._owned_variable_names.add(name)
    return new_variable

  def track_checkpointable(self, checkpointable, name=None):
    """Declare a dependency on another `Checkpointable` object.

    Indicates that checkpoints for this object should include variables from
    `checkpointable`.

    Variables in a checkpoint are mapped to `Checkpointable`s based on names if
    provided when the checkpoint was written, but otherwise use the order those
    `Checkpointable`s were declared as dependencies. Both `name` arguments and
    the dependency declaration order should be deterministic.

    There are two sufficient conditions to avoid breaking existing checkpoints
    when modifying a class: (1) New dependencies must be declared after existing
    dependencies, and (2) dependencies which were previously declared may never
    be removed (a trivial placeholder with the same name may be used instead).

    Args:
      checkpointable: A `Checkpointable` which this object depends on.
      name: A local name for `checkpointable`, used for loading checkpoints into
        the correct objects. If provided, it must be unique within this
        `Checkpointable`. If None, dependency declaration order is used instead.

    Returns:
      `checkpointable`, for convenience when declaring a dependency and
      assigning to a member variable in one statement.

    Raises:
      RuntimeError: If __init__ was not called.
      TypeError: If `checkpointable` does not inherit from `Checkpointable`.
      ValueError: For invalid names.
    """
    if not hasattr(self, "_checkpoint_dependencies"):
      raise RuntimeError("Need to call Checkpointable.__init__ before calling "
                         "Checkpointable.track_checkpointable().")
    if not isinstance(checkpointable, Checkpointable):
      raise TypeError(
          ("Checkpointable.track_checkpointable() passed type %s, not a "
           "Checkpointable.") % (type(checkpointable),))
    if name is not None:
      if not _VALID_LOCAL_NAME.match(name):
        raise ValueError(
            ("Checkpointable names must match the regular expression '%s', but "
             "got an invalid name '%s' instead.") % (_VALID_LOCAL_NAME.pattern,
                                                     name))
      if name in self._dependency_names:
        raise ValueError(
            ("Called Checkpointable.track_checkpointable() with name='%s', but "
             "a Checkpointable with this name is already declared as a "
             "dependency. If provided, names must be unique.") % (name,))
      self._dependency_names.add(name)
    self._checkpoint_dependencies.append(
        _CheckpointableReference(
            name=name,
            ref=checkpointable,
            # TODO(allenl): Should this be exposed to allow users to stop
            # depending on things and still load checkpoints when not using
            # names?
            local_uid=len(self._checkpoint_dependencies)))
    return checkpointable

  @property
  def checkpoint_dependencies(self):
    """Other `Checkpointable` objects on which this object depends."""
    return self._checkpoint_dependencies


def _breadth_first_checkpointable_traversal(root_checkpointable):
  """Find shortest paths to all variables owned by dependencies of root."""
  bfs_sorted = []
  root_checkpointable_reference = _CheckpointableReference(
      name=None, local_uid=0, ref=root_checkpointable)
  to_visit = collections.deque([root_checkpointable_reference])
  path_to_root = {root_checkpointable_reference: ()}
  while to_visit:
    current_checkpointable = to_visit.popleft()
    bfs_sorted.append(current_checkpointable)
    for child_checkpointable in (
        current_checkpointable.ref.checkpoint_dependencies):
      if child_checkpointable not in path_to_root:
        path_to_root[child_checkpointable] = (
            path_to_root[current_checkpointable] + (child_checkpointable,))
        to_visit.append(child_checkpointable)
  return bfs_sorted, path_to_root


def _object_prefix_from_path(path_to_root):
  return "/".join((checkpointable.name if checkpointable.name else "_%d" % (
      checkpointable.local_uid,)) for checkpointable in path_to_root)


def _escape_variable_name(variable_name):
  # We need to support slashes in variable names for compatibility, since this
  # naming scheme is being patched in to things like Layer.add_variable where
  # slashes were previously accepted. We also want to use slashes to indicate
  # edges traversed to reach the variable, so we escape forward slashes in
  # variable names.
  return variable_name.replace("_S_", "_S_.").replace(r"/", r"_S__")


def _variable_naming_for_object(path_to_root):
  """Make a function for naming variables in an object."""
  # Name non-slot variables:
  #
  #   <path to node>/<local variable name>
  #
  # <path to node> is not necessarily unique, but this is fine since we also
  # save the graph of `Checkpointable`s with the checkpoint. Even if this path
  # no longer exists because of a change in the Python program, we can look up
  # the `Checkpointable` which owns the variable in the checkpoint's graph and
  # use another path if one still exists.

  object_prefix = _object_prefix_from_path(path_to_root)
  if object_prefix:
    object_prefix += "/"

  def _name_single_variable(owned_variable):
    """Names a variable within an object."""
    return object_prefix + _escape_variable_name(owned_variable.name)

  return _name_single_variable


def _slot_variable_naming_for_optimizer(optimizer, path_to_root):
  """Make a function for naming slot variables in an optimizer."""
  # Name slot variables:
  #
  #   <variable name>/<_OPTIMIZER_SLOTS_NAME>/<optimizer path>/<slot name>
  #
  # where <variable name> is exactly the checkpoint name used for the original
  # variable, including the path from the checkpoint root and the local name in
  # the object which owns it. Note that we only save slot variables if the
  # variable it's slotting for is also being saved.

  optimizer_identifier = "/%s/%s/" % (_OPTIMIZER_SLOTS_NAME,
                                      _object_prefix_from_path(path_to_root))

  def _name_slot_variable(variable_path, slot_name):
    """With an optimizer specified, name a slot variable."""

    if not _VALID_LOCAL_NAME.match(slot_name):
      # Slot variable names include the name of the slot. We need to
      # validate that part of the name to be sure that the checkpoint name
      # is a valid name scope name.
      raise ValueError(
          ("Could not save slot variables for optimizer %s, because its "
           "slot name has invalid characters (got '%s', was expecting it "
           "to match the regular expression '%s').") %
          (optimizer, slot_name, _VALID_LOCAL_NAME.pattern))

    return variable_path + optimizer_identifier + slot_name

  return _name_slot_variable


def _serialize_non_slot_variables(checkpointable_objects, path_to_root,
                                  object_graph_proto):
  """Name non-slot variables and add them to `object_graph_proto`."""
  named_variables = {}
  non_slot_variables = []
  checkpoint_node_ids = {}

  for checkpoint_id, checkpointable in enumerate(checkpointable_objects):
    checkpoint_node_ids[checkpointable] = checkpoint_id

  for checkpoint_id, checkpointable in enumerate(checkpointable_objects):
    naming_scheme = _variable_naming_for_object(path_to_root[checkpointable])
    object_proto = object_graph_proto.nodes.add()
    for owned_variable in checkpointable.ref._owned_variables:  # pylint: disable=protected-access
      variable_name = naming_scheme(owned_variable)
      named_variables[variable_name] = owned_variable.variable
      non_slot_variables.append((
          variable_name,  # The variable's full checkpoint name
          owned_variable,  # The variable's _OwnedVariable object
          checkpoint_id))  # The checkpoint ID of the node which owns this
      # variable.
      variable_proto = object_proto.variables.add()
      variable_proto.local_name = owned_variable.name
      # Figure out the name-based Saver's name for this variable.
      saver_dict = saver_lib.BaseSaverBuilder.OpListToDict(
          [owned_variable.variable], convert_variable_to_tensor=False)
      variable_full_name, = saver_dict.keys()
      variable_proto.full_name = variable_full_name

    for child in checkpointable.ref.checkpoint_dependencies:
      child_proto = object_proto.children.add()
      child_proto.node_id = checkpoint_node_ids[child]
      child_proto.local_uid = child.local_uid
      if child.name is not None:
        child_proto.local_name = child.name
  return named_variables, non_slot_variables


def _serialize_slot_variables(checkpointable_objects, path_to_root,
                              non_slot_variables, object_graph_proto):
  """Name slot variables and add them to `object_graph_proto`."""
  named_slot_variables = {}
  for optimizer_checkpoint_id, checkpointable_ref in enumerate(
      checkpointable_objects):
    if isinstance(checkpointable_ref.ref, optimizer_lib.Optimizer):
      optimizer_object_proto = object_graph_proto.nodes[optimizer_checkpoint_id]
      naming_scheme = _slot_variable_naming_for_optimizer(
          optimizer=checkpointable_ref.ref,
          path_to_root=path_to_root[checkpointable_ref])
      slot_names = checkpointable_ref.ref.get_slot_names()
      for (variable_path, owned_variable,
           original_node_checkpoint_id) in non_slot_variables:
        for slot_name in slot_names:
          slot_variable = checkpointable_ref.ref.get_slot(
              owned_variable.variable, slot_name)
          if slot_variable is not None:
            checkpoint_name = naming_scheme(
                variable_path=variable_path, slot_name=slot_name)
            named_slot_variables[checkpoint_name] = slot_variable
            slot_variable_proto = optimizer_object_proto.slot_variables.add()
            slot_variable_proto.slot_name = slot_name
            # Figure out the name-based Saver's name for this variable.
            saver_dict = saver_lib.BaseSaverBuilder.OpListToDict(
                [slot_variable], convert_variable_to_tensor=False)
            slot_variable_full_name, = saver_dict.keys()
            slot_variable_proto.full_name = slot_variable_full_name
            slot_variable_proto.original_variable_local_name = (
                owned_variable.name)
            slot_variable_proto.original_variable_node_id = (
                original_node_checkpoint_id)
  return named_slot_variables


# TODO(allenl): Convenience utility for saving multiple objects (i.e. construct
# a root Checkpointable if passed a list of Checkpointables).
def _serialize_object_graph(root_checkpointable):
  """Determine checkpoint keys for variables and build a serialized graph.

  Non-slot variables are keyed based on a shortest path from the root saveable
  to the object which owns the variable (i.e. the one which called
  `Checkpointable.add_variable` to create it).

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
  object_graph_proto = (
      checkpointable_object_graph_pb2.CheckpointableObjectGraph())

  # Gather non-slot variables.
  named_variables, non_slot_variables = _serialize_non_slot_variables(
      checkpointable_objects, path_to_root, object_graph_proto)

  # Gather slot variables which are associated with variables gathered above.
  named_slot_variables = _serialize_slot_variables(
      checkpointable_objects, path_to_root, non_slot_variables,
      object_graph_proto)

  named_variables.update(named_slot_variables)
  return named_variables, object_graph_proto
