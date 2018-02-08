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
import weakref

from tensorflow.contrib.eager.proto import checkpointable_object_graph_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import optimizer as optimizer_lib
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import slot_creator
from tensorflow.python.training import training

_CheckpointableReference = collections.namedtuple(
    "_CheckpointableReference",
    [
        # The local name if explicitly specified, else None.
        "name",
        # 1 for the first dependency, 2 for the next, ... Used for routing
        # checkpointed variables to their correct Checkpointables when "name" is
        # not set (see docstring of `track_checkpointable`).
        "local_uid",
        # The Checkpointable object being referenced.
        "ref"
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


def _assign_existing_variable(variable_to_restore, value_pointer):
  """Set a variable from a _ValuePointer object."""
  base_type = variable_to_restore.dtype.base_dtype
  with ops.colocate_with(variable_to_restore):
    # TODO(allenl): Handle partitioned variables
    value_to_restore, = io_ops.restore_v2(
        prefix=value_pointer.save_path,
        tensor_names=[value_pointer.checkpoint_key],
        shape_and_slices=[""],
        dtypes=[base_type],
        name="checkpoint_initializer")
    initializer_op = state_ops.assign(variable_to_restore, value_to_restore)
    variable_to_restore._initializer_op = initializer_op  # pylint:disable=protected-access
    if value_pointer.session is not None:
      value_pointer.session.run(initializer_op)


class Checkpointable(object):
  """Manages variables and dependencies on other objects.

  To make reliable checkpoints, all `Checkpointable`s on which this object
  depends must be registered in the constructor using `track_checkpointable` in
  a deterministic order, and if possible they should be named. Variables may be
  created using `add_variable` outside of the constructor and in any order, but
  only these variables will be saved.
  """

  def __init__(self):
    # Basically a less useful OrderedDict but without the reference cycles.
    # TODO(allenl): Switch this to OrderedDict once TensorFlow supports only
    # Python 3.6+.
    # A list of _CheckpointableReference objects.
    self._checkpoint_dependencies = []
    self._dependency_names = set()
    # Start numbering at 1, since an un-set protocol buffer integer is
    # indistinguishable from 0.
    self._next_unnamed_checkpoint_dependency_uid = 1
    self._owned_variables = {}  # local name -> variable object
    self._deferred_restorations = {}  # local name -> _VariableRestoration
                                      # object

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
    if name in self._owned_variables:
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
    deferred_restoration = self._deferred_restorations.pop(name, None)
    if deferred_restoration is not None:
      dtype = deferred_restoration.value_pointer.dtype
      base_type = dtype.base_dtype
      # TODO(allenl): Handle partitioned variables here too
      initializer, = io_ops.restore_v2(
          prefix=deferred_restoration.value_pointer.save_path,
          tensor_names=[deferred_restoration.value_pointer.checkpoint_key],
          shape_and_slices=[""],
          dtypes=[base_type],
          name="checkpoint_initializer")
      # We need to un-set the shape so get_variable doesn't complain, but we
      # also need to set the static shape information on the initializer if
      # possible so we don't get a variable with an unknown shape.
      initializer.set_shape(shape)
      # Un-set shape since we're using a constant initializer
      shape = None

    new_variable = getter(
        name=name, shape=shape, dtype=dtype, initializer=initializer, **kwargs)
    if deferred_restoration is not None:
      if deferred_restoration.value_pointer.session is not None:
        deferred_restoration.value_pointer.session.run(new_variable.initializer)
      for slot_restoration in deferred_restoration.slot_restorations:
        strong_ref = slot_restoration.optimizer_ref()
        if strong_ref is None:
          # If the optimizer object has been garbage collected, there's no need
          # to create the slot variable.
          continue
        strong_ref._process_slot_restoration(  # pylint: disable=protected-access
            slot_restoration, new_variable)
    self._owned_variables[name] = new_variable
    return new_variable

  def track_checkpointable(self, checkpointable, name=None):
    """Declare a dependency on another `Checkpointable` object.

    Indicates that checkpoints for this object should include variables from
    `checkpointable`.

    Variables in a checkpoint are mapped to `Checkpointable`s based on names if
    provided when the checkpoint was written, but otherwise use the order those
    `Checkpointable`s were declared as dependencies.

    There are three sufficient conditions to avoid breaking existing checkpoints
    when modifying a class: (1) New un-named dependencies must be declared after
    existing un-named dependencies, (2) un-named dependencies which were
    previously declared may never be removed (a trivial placeholder may be used
    instead if the dependency is no longer needed), and (3) names may not change
    (un-named dependencies may not later be named, named dependencies must keep
    the same name).

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
      local_uid = None
    else:
      # TODO(allenl): Should this be exposed to allow users to stop depending on
      # things and still load checkpoints when not using names?
      local_uid = self._next_unnamed_checkpoint_dependency_uid
      self._next_unnamed_checkpoint_dependency_uid += 1
    self._checkpoint_dependencies.append(
        _CheckpointableReference(
            name=name, ref=checkpointable, local_uid=local_uid))
    return checkpointable

  def _process_restoration(self, restoration):
    """Restore a variable and its slot variables (may be deferred)."""
    variable_to_restore = self._owned_variables.get(restoration.name, None)
    if variable_to_restore is not None:
      # This variable already exists, so just do an assignment for this and any
      # slot variables which depend on it.
      _assign_existing_variable(
          variable_to_restore, value_pointer=restoration.value_pointer)
      for slot_restoration in restoration.slot_restorations:
        strong_ref = slot_restoration.optimizer_ref()
        if strong_ref is None:
          continue
        strong_ref._process_slot_restoration(  # pylint: disable=protected-access
            slot_restoration, variable_to_restore)
    else:
      # Save this restoration for later. This intentionally overwrites any
      # previous deferred restorations, since that gives the same semantics as
      # direct assignment.
      self._deferred_restorations[restoration.name] = restoration

  def _process_slot_restoration(self, slot_restoration, variable):
    """Restore a slot variable's value (creating it if necessary)."""
    # TODO(allenl): Move this to Optimizer
    assert isinstance(self, optimizer_lib.Optimizer)
    named_slots = self._slot_dict(slot_restoration.slot_name)
    variable_key = optimizer_lib._var_key(variable)  # pylint: disable=protected-access
    existing_slot_variable = named_slots.get(variable_key, None)
    if existing_slot_variable is None:
      base_dtype = slot_restoration.value_pointer.dtype.base_dtype
      initializer, = io_ops.restore_v2(
          prefix=slot_restoration.value_pointer.save_path,
          tensor_names=[slot_restoration.value_pointer.checkpoint_key],
          shape_and_slices=[""],
          dtypes=[base_dtype],
          name="checkpoint_initializer")
      new_slot_variable = slot_creator.create_slot(variable, initializer,
                                                   slot_restoration.slot_name)
      if slot_restoration.value_pointer.session is not None:
        slot_restoration.value_pointer.session.run(
            new_slot_variable.initializer)
      named_slots[variable_key] = new_slot_variable
    else:
      _assign_existing_variable(
          existing_slot_variable, value_pointer=slot_restoration.value_pointer)

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

  def _name_single_variable(local_name):
    """Names a variable within an object."""
    return object_prefix + _escape_variable_name(local_name)

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
    for (local_name, owned_variable) in sorted(
        checkpointable.ref._owned_variables.items(),  # pylint: disable=protected-access
        key=lambda x: x[0]):
      variable_name = naming_scheme(local_name)
      named_variables[variable_name] = owned_variable
      non_slot_variables.append((
          variable_name,  # The variable's full checkpoint name
          owned_variable,  # The variable object
          local_name,  # The variable's local name
          checkpoint_id))  # The checkpoint ID of the node which owns this
      # variable.
      variable_proto = object_proto.variables.add()
      variable_proto.local_name = local_name
      variable_proto.checkpoint_key = variable_name
      # Figure out the name-based Saver's name for this variable.
      saver_dict = saver_lib.BaseSaverBuilder.OpListToDict(
          [owned_variable], convert_variable_to_tensor=False)
      variable_full_name, = saver_dict.keys()
      variable_proto.full_name = variable_full_name

    for child in checkpointable.ref.checkpoint_dependencies:
      child_proto = object_proto.children.add()
      child_proto.node_id = checkpoint_node_ids[child]
      if child.local_uid is not None:
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
      for (variable_path, original_variable, original_variable_local_name,
           original_node_checkpoint_id) in non_slot_variables:
        for slot_name in slot_names:
          slot_variable = checkpointable_ref.ref.get_slot(
              original_variable, slot_name)
          if slot_variable is not None:
            checkpoint_name = naming_scheme(
                variable_path=variable_path, slot_name=slot_name)
            named_slot_variables[checkpoint_name] = slot_variable
            slot_variable_proto = optimizer_object_proto.slot_variables.add()
            slot_variable_proto.slot_name = slot_name
            slot_variable_proto.checkpoint_key = checkpoint_name
            # Figure out the name-based Saver's name for this variable.
            saver_dict = saver_lib.BaseSaverBuilder.OpListToDict(
                [slot_variable], convert_variable_to_tensor=False)
            slot_variable_full_name, = saver_dict.keys()
            slot_variable_proto.full_name = slot_variable_full_name
            slot_variable_proto.original_variable_local_name = (
                original_variable_local_name)
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


def _set_reference(reference_proto_table, key, checkpointable, parent,
                   object_id_map):
  """Record a checkpoint<->object correspondence, with error checking.

  Args:
    reference_proto_table: Map from names or numbers to `ObjectReference` protos
      within the parent object.
    key: Either a numeric or string identifier for the reference.
    checkpointable: The object to record a correspondence for.
    parent: The parent Python object, for creating a useful error message.
    object_id_map: The map from `node_id` to Python object in which to record
      the reference.
  Returns:
    The `node_id` of the Object proto corresponding to the specified Python
    object.
  Raises:
    AssertionError: If another object is already bound to the `Object` proto.
  """
  reference_proto = reference_proto_table[key]
  set_reference = object_id_map.setdefault(reference_proto.node_id,
                                           checkpointable)
  if set_reference is not checkpointable:
    raise AssertionError(
        ("Unable to load the checkpoint into this object graph. Either "
         "the Checkpointable object references in the Python program "
         "have changed in an incompatible way, or the checkpoint was "
         "generated in an incompatible program.\n\nTwo checkpoint "
         "references (one being '%s' in %s) resolved to different "
         "objects (%s and %s).") % (key, parent, set_reference,
                                    checkpointable))
  return reference_proto.node_id


def _checkpoint_object_id_map(root_checkpointable, object_graph_proto):
  """Match a checkpointed object graph to a Python object graph.

  Args:
    root_checkpointable: A Checkpointable object.
    object_graph_proto: A CheckpointableObjectGraph protocol buffer representing
      a serialized object graph.
  Returns:
    A dictionary mapping from checkpoint node ids (indices into
    `object_graph_proto.nodes`) to `Checkpointable` objects which are
    dependencies of `root_checkpointable`.
  """
  node_list = object_graph_proto.nodes
  # Queue of (checkpointable object, node id)
  to_visit = collections.deque([(root_checkpointable, 0)])
  object_id_map = {0: root_checkpointable}
  seen = set()
  while to_visit:
    checkpointable, node_id = to_visit.popleft()
    object_proto = node_list[node_id]
    named_children = {}
    numbered_children = {}
    for child_reference in object_proto.children:
      if child_reference.local_name:
        named_children[child_reference.local_name] = child_reference
      else:
        if not child_reference.local_uid:
          raise AssertionError(
              ("The checkpointed object graph contains a reference with "
               "neither a name nor a number (corrupted?). The reference was "
               "from the node %s.") % (object_proto,))
        numbered_children[child_reference.local_uid] = child_reference

    for checkpointable_reference in checkpointable._checkpoint_dependencies:  # pylint: disable=protected-access
      if checkpointable_reference.name is not None:
        child_node_id = _set_reference(
            reference_proto_table=named_children,
            key=checkpointable_reference.name,
            checkpointable=checkpointable_reference.ref,
            parent=checkpointable,
            object_id_map=object_id_map)
      else:
        if checkpointable_reference.local_uid is None:
          raise AssertionError(
              ("A Checkpointable reference was created with no name and no "
               "number in %s.") % (checkpointable,))
        child_node_id = _set_reference(
            reference_proto_table=numbered_children,
            key=checkpointable_reference.local_uid,
            checkpointable=checkpointable_reference.ref,
            parent=checkpointable,
            object_id_map=object_id_map)
      if child_node_id not in seen:
        seen.add(child_node_id)
        to_visit.append((checkpointable_reference.ref, child_node_id))

  return object_id_map


_ValuePointer = collections.namedtuple(
    "_ValuePointer",
    [
        # Information needed to look up the value to restore.
        "save_path",
        "checkpoint_key",
        "dtype",
        # The session to use when restoring (None when executing eagerly)
        "session",
    ])

_SlotVariableRestoration = collections.namedtuple(
    "_SlotVariableRestoration",
    [
        # A weak reference to the Optimizer object
        "optimizer_ref",
        # The slot name
        "slot_name",
        # The _ValuePointer to use when restoring
        "value_pointer",
    ])

_VariableRestoration = collections.namedtuple(
    "_VariableRestoration",
    [
        # The variable's (local) name.
        "name",
        # _SlotVariableRestoration objects indicating slot variables which
        # should be created once this variable has been restored.
        "slot_restorations",
        # The _ValuePointer to use when restoring
        "value_pointer",
    ])


def _gather_restorations(object_graph_proto, save_path, object_id_map,
                         dtype_map, session):
  """Iterate over variables to restore, matching with Checkpointable objects."""
  variable_to_slot_restorations = {}
  for node_id, node in enumerate(object_graph_proto.nodes):
    for slot_variable in node.slot_variables:
      original_variable_key = (slot_variable.original_variable_node_id,
                               slot_variable.original_variable_local_name)
      variable_to_slot_restorations.setdefault(
          original_variable_key, []).append(
              _SlotVariableRestoration(
                  optimizer_ref=weakref.ref(object_id_map[node_id]),
                  slot_name=slot_variable.slot_name,
                  value_pointer=_ValuePointer(
                      save_path=save_path,
                      checkpoint_key=slot_variable.checkpoint_key,
                      dtype=dtype_map[slot_variable.checkpoint_key],
                      session=session)))

  for node_id, node in enumerate(object_graph_proto.nodes):
    for variable in node.variables:
      slots_key = (node_id, variable.local_name)
      variable_restore = _VariableRestoration(
          name=variable.local_name,
          slot_restorations=variable_to_slot_restorations.get(slots_key, []),
          value_pointer=_ValuePointer(
              save_path=save_path,
              checkpoint_key=variable.checkpoint_key,
              dtype=dtype_map[variable.checkpoint_key],
              session=session))
      yield variable_restore, object_id_map[node_id]


def save(file_prefix, root_checkpointable, global_step=None, session=None):
  """Save a training checkpoint.

  Args:
    file_prefix: A prefix to use for the checkpoint filenames
      (/path/to/directory/and_a_prefix). Names are generated based on this
      prefix and the global step, if provided.
    root_checkpointable: A Checkpointable object to save. The checkpoint
      includes variables created by this object and any Checkpointable objects
      it depends on.
    global_step: An integer variable or Tensor, used to number
      checkpoints. Typically this value is saved along with other variables in
      training checkpoints, which will happen automatically if it was created by
      `root_checkpointable` or one of its dependencies (via
      `Checkpointable.add_variable`).
    session: The session to evaluate variables in. Ignored when executing
      eagerly. If not provided when graph building, the default session is used.

  Returns:
    The full path to the checkpoint.

    Currently also returns the serialized object graph proto, but that will go
    away once it's saved with the checkpoint.
  """
  named_variables, serialized_graph = _serialize_object_graph(
      root_checkpointable)
  if context.in_graph_mode():
    if session is None:
      session = ops.get_default_session()
  else:
    session = None
  with ops.device("/device:CPU:0"):
    save_path = saver_lib.Saver(var_list=named_variables).save(
        sess=session,
        save_path=file_prefix,
        write_meta_graph=False,
        global_step=global_step)
  # TODO(allenl): Save the graph with the checkpoint, then returning it and
  # taking it as an argument to restore won't be necessary.
  return serialized_graph, save_path


# NOTE: Will be restore(file_prefix, root_checkpointable) once the object graph
# is saved with the checkpoint.
def restore(save_path, root_checkpointable, object_graph_proto, session=None):
  """Restore a training checkpoint.

  Restores the values of variables created with `Checkpointable.add_variable` in
  the dependency graph of `root_checkpointable`. Either assigns values
  immediately (if variables to restore have been created already), or defers
  restoration until the variables are created.

  When building a graph, restorations are executed in the default session if
  `session` is `None`. Variable initializers read checkpointed values.

  Args:
    save_path: The path to the checkpoint, as returned by `save` or
      `tf.train.latest_checkpoint`. If None (as when there is no latest
      checkpoint for `tf.train.latest_checkpoint` to return), does nothing.
    root_checkpointable: The root of the object graph to restore. Variables to
      restore need not have been created yet, but all dependencies on other
      Checkpointable objects should already be declared. Objects in the
      dependency graph are matched to objects in the checkpointed graph, and
      matching objects have their variables restored (or the checkpointed values
      saved for eventual restoration when the variable is created).
    object_graph_proto: (Temporary) the checkpointed object graph. This will
      eventually be saved with the checkpoint, and will not be part of the final
      API.
    session: The session to evaluate assignment ops in. Ignored when executing
      eagerly. If not provided when graph building, the default session is used.
  """
  if save_path is None:
    return
  object_id_map = _checkpoint_object_id_map(root_checkpointable,
                                            object_graph_proto)
  reader = training.NewCheckpointReader(save_path)
  dtype_map = reader.get_variable_to_dtype_map()
  if context.in_graph_mode():
    if session is None:
      session = ops.get_default_session()
  else:
    session = None
  for restoration, checkpointable in _gather_restorations(
      object_graph_proto, save_path, object_id_map, dtype_map, session=session):
    checkpointable._process_restoration(restoration)  # pylint: disable=protected-access
