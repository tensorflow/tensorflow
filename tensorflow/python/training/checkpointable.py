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

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_io_ops as io_ops
from tensorflow.python.util import nest

# A key indicating a variable's value in an object's checkpointed Tensors
# (Checkpointable._gather_saveables_for_checkpoint). If this is the only key and
# the object has no dependencies, then its value may be restored on object
# creation (avoiding double assignment when executing eagerly).
VARIABLE_VALUE_KEY = "VARIABLE_VALUE"

_CheckpointableReference = collections.namedtuple(
    "_CheckpointableReference",
    [
        # The local name for this dependency.
        "name",
        # The Checkpointable object being referenced.
        "ref"
    ])


class CheckpointInitialValue(ops.Tensor):
  """Tensor wrapper for managing update UIDs in `Variables`.

  When supplied as an initial value, objects of this type let a `Variable`
  (`Variable`, `ResourceVariable`, etc.) know the UID of the restore the initial
  value came from. This allows deferred restorations to be sequenced in the
  order the user specified them, and lets us fall back on assignment if an
  initial value is not set (e.g. due to a custom getter interfering).

  See comments in _add_variable_with_custom_getter for more information about
  how `CheckpointInitialValue` is used.
  """

  def __init__(self, checkpoint_position, shape=None):
    self.wrapped_value = checkpoint_position.value_tensors()[
        VARIABLE_VALUE_KEY]
    if shape:
      # We need to set the static shape information on the initializer if
      # possible so we don't get a variable with an unknown shape.
      self.wrapped_value.set_shape(shape)
    self._checkpoint_position = checkpoint_position

  @property
  def __class__(self):
    return (self.wrapped_value.__class__, CheckpointInitialValue)

  def __getattr__(self, attr):
    try:
      return getattr(self.wrapped_value, attr)
    except AttributeError:
      return self.__getattribute__(attr)

  @property
  def checkpoint_position(self):
    return self._checkpoint_position


class _CheckpointPosition(object):
  """Indicates a position within a `_Checkpoint`."""

  def __init__(self, checkpoint, proto_id):
    """Specify an object within a checkpoint.

    Args:
      checkpoint: A _Checkpoint object.
      proto_id: The index of this object in CheckpointableObjectGraph.nodes.
    """
    self._checkpoint = checkpoint
    self._proto_id = proto_id

  def restore(self, checkpointable):
    """Restore this value into `checkpointable`."""
    if self.bind_object(checkpointable):
      # This object's correspondence with a checkpointed object is new, so
      # process deferred restorations for it and its dependencies.
      restore_ops = checkpointable._restore_from_checkpoint_position(self)  # pylint: disable=protected-access
      if restore_ops:
        self._checkpoint.restore_ops.extend(restore_ops)

  def bind_object(self, checkpointable):
    """Set a checkpoint<->object correspondence and process slot variables.

    Args:
      checkpointable: The object to record a correspondence for.
    Returns:
      True if this is a new assignment, False if this object has already been
      mapped to a checkpointed `Object` proto.
    Raises:
      AssertionError: If another object is already bound to the `Object` proto.
    """
    checkpoint = self.checkpoint
    current_assignment = checkpoint.object_by_proto_id.get(self._proto_id, None)
    if current_assignment is None:
      checkpoint.object_by_proto_id[self._proto_id] = checkpointable
      for deferred_slot_restoration in (
          checkpoint.deferred_slot_restorations.pop(self._proto_id, ())):
        checkpointable._create_or_restore_slot_variable(  # pylint: disable=protected-access
            slot_variable_position=_CheckpointPosition(
                checkpoint=checkpoint,
                proto_id=deferred_slot_restoration.slot_variable_id),
            variable=deferred_slot_restoration.original_variable,
            slot_name=deferred_slot_restoration.slot_name)
      for slot_restoration in checkpoint.slot_restorations.pop(
          self._proto_id, ()):
        optimizer_object = checkpoint.object_by_proto_id.get(
            slot_restoration.optimizer_id, None)
        if optimizer_object is None:
          # The optimizer has not yet been created or tracked. Record in the
          # checkpoint that the slot variables need to be restored when it is.
          checkpoint.deferred_slot_restorations.setdefault(
              slot_restoration.optimizer_id, []).append(
                  _DeferredSlotVariableRestoration(
                      original_variable=checkpointable,
                      slot_variable_id=slot_restoration.slot_variable_id,
                      slot_name=slot_restoration.slot_name))
        else:
          optimizer_object._create_or_restore_slot_variable(  # pylint: disable=protected-access
              slot_variable_position=_CheckpointPosition(
                  checkpoint=checkpoint,
                  proto_id=slot_restoration.slot_variable_id),
              variable=checkpointable,
              slot_name=slot_restoration.slot_name)
      return True  # New assignment
    else:
      # The object was already mapped for this checkpoint load, which means
      # we don't need to do anything besides check that the mapping is
      # consistent (if the dependency DAG is not a tree then there are
      # multiple paths to the same object).
      if current_assignment is not checkpointable:
        raise AssertionError(
            ("Unable to load the checkpoint into this object graph. Either "
             "the Checkpointable object references in the Python program "
             "have changed in an incompatible way, or the checkpoint was "
             "generated in an incompatible program.\n\nTwo checkpoint "
             "references resolved to different objects (%s and %s).")
            % (current_assignment, checkpointable))
      return False  # Not a new assignment

  def is_simple_variable(self):
    """Determine whether this value is restorable with a Tensor initializer."""
    attributes = self.object_proto.attributes
    return (len(attributes) == 1
            and attributes[0].name == VARIABLE_VALUE_KEY
            and not self.object_proto.children)

  def value_tensors(self):
    """Create value `Tensor`s for this object's attributes.

    Does not require that the Python object has been created. Used for
    restore-on-create when executing eagerly.

    Returns:
      A dictionary mapping from object attribute names to `Tensor`s.
    """
    value_tensors = {}
    for serialized_tensor in self.object_proto.attributes:
      checkpoint_key = serialized_tensor.checkpoint_key
      dtype = self._checkpoint.dtype_map[checkpoint_key]
      base_type = dtype.base_dtype
      with ops.init_scope():
        value, = io_ops.restore_v2(
            prefix=self._checkpoint.save_path,
            tensor_names=[checkpoint_key],
            shape_and_slices=[""],
            dtypes=[base_type],
            name="%s_checkpoint_read" % (serialized_tensor.name,))
        value_tensors[serialized_tensor.name] = value
      return value_tensors

  def restore_ops(self):
    """Create or fetch restore ops for this object's attributes.

    Requires that the `Checkpointable` Python object has been bound to an object
    ID in the checkpoint.

    Returns:
      A list of operations when graph building, or an empty list when executing
      eagerly.
    """
    saveables = self.checkpointable._gather_saveables_for_checkpoint()  # pylint: disable=protected-access
    # Name saveables based on the name this object had when it was checkpointed.
    named_saveables = {}
    restore_ops = []
    in_graph_mode = context.in_graph_mode()
    for serialized_tensor in self.object_proto.attributes:
      saveable_object = saveables.get(serialized_tensor.name, None)
      if saveable_object is None:
        # Purposefully does not throw an exception if attributes have been added
        # or deleted. Stores unused attributes so an exception can be raised if
        # the user decides to check that everything in the checkpoint was
        # loaded.
        self._checkpoint.unused_attributes.setdefault(
            self.checkpointable, []).append(serialized_tensor.name)
        continue
      if in_graph_mode:
        existing_ops = self._checkpoint.restore_ops_by_name.get(
            serialized_tensor.name, None)
      else:
        existing_ops = None
      if existing_ops is None:
        named_saveables[serialized_tensor.checkpoint_key] = saveable_object
    if named_saveables:
      validated_saveables = (
          self._checkpoint.builder._ValidateAndSliceInputs(named_saveables))  # pylint: disable=protected-access
      validated_names = set(saveable.name for saveable in validated_saveables)
      if set(named_saveables.keys()) != validated_names:
        raise AssertionError(
            ("Saveable keys changed when validating. Got back %s, was "
             "expecting %s") % (named_saveables.keys(), validated_names))
      all_tensors = self._checkpoint.builder.bulk_restore(
          filename_tensor=self._checkpoint.save_path,
          saveables=validated_saveables, preferred_shard=-1,
          restore_sequentially=False)
      saveable_index = 0
      for saveable in validated_saveables:
        num_specs = len(saveable.specs)
        saveable_tensors = all_tensors[
            saveable_index:saveable_index + num_specs]
        saveable_index += num_specs
        restore_op = saveable.restore(saveable_tensors, restored_shapes=None)
        if in_graph_mode:
          assert saveable.name not in self._checkpoint.restore_ops_by_name
          self._checkpoint.restore_ops_by_name[saveable.name] = restore_op
          restore_ops.append(restore_op)
    return restore_ops

  @property
  def checkpoint(self):
    return self._checkpoint

  @property
  def checkpointable(self):
    return self._checkpoint.object_by_proto_id[self._proto_id]

  @property
  def object_proto(self):
    return self._checkpoint.object_graph_proto.nodes[self._proto_id]

  @property
  def restore_uid(self):
    return self._checkpoint.restore_uid

  def __repr__(self):
    return repr(self.object_proto)


_DeferredSlotVariableRestoration = collections.namedtuple(
    "_DeferredSlotVariableRestoration",
    [
        "original_variable",
        "slot_variable_id",
        "slot_name",
    ]
)

_SlotVariableRestoration = collections.namedtuple(
    "_SlotVariableRestoration",
    [
        # The checkpoint proto id of the optimizer object.
        "optimizer_id",
        # The checkpoint proto id of the slot variable.
        "slot_variable_id",
        "slot_name",
    ])


class CheckpointableBase(object):
  """Base class for `Checkpointable` objects without automatic dependencies.

  This class has no __setattr__ override for performance reasons. Dependencies
  must be added explicitly. Unless attribute assignment is performance-critical,
  use `Checkpointable` instead. Use `CheckpointableBase` for `isinstance`
  checks.
  """

  def _maybe_initialize_checkpointable(self):
    """Initialize dependency management.

    Not __init__, since most objects will forget to call it.
    """
    if hasattr(self, "_checkpoint_dependencies"):
      # __init__ already called. This check means that we don't need
      # Checkpointable.__init__() in the constructor of every TensorFlow object.
      return
    # A list of _CheckpointableReference objects.
    self._checkpoint_dependencies = []
    # Maps names -> Checkpointable objects
    self._dependency_names = {}
    # Restorations for other Checkpointable objects on which this object may
    # eventually depend.
    self._deferred_dependencies = {}  # local name -> _CheckpointPosition list
    # The UID of the highest assignment to this object. Used to ensure that the
    # last requested assignment determines the final value of an object.
    if hasattr(self, "_update_uid"):
      raise AssertionError(
          "Internal error: the object had an update UID set before its "
          "initialization code was run.")
    self._update_uid = -1

  def _add_variable_with_custom_getter(
      self, name, shape=None, dtype=dtypes.float32,
      initializer=None, getter=None, **kwargs_for_getter):
    """Restore-on-create for a variable be saved with this `Checkpointable`.

    If the user has requested that this object or another `Checkpointable` which
    depends on this object be restored from a checkpoint (deferred loading
    before variable object creation), `initializer` may be ignored and the value
    from the checkpoint used instead.

    Args:
      name: A name for the variable. Must be unique within this object.
      shape: The shape of the variable.
      dtype: The data type of the variable.

      initializer: The initializer to use. Ignored if there is a deferred
        restoration left over from a call to
        `_restore_from_checkpoint_position`.

      getter: The getter to wrap which actually fetches the variable.
      **kwargs_for_getter: Passed to the getter.

    Returns:
      The new variable object.

    Raises:
      ValueError: If the variable name is not unique.
    """
    self._maybe_initialize_checkpointable()
    if name in self._dependency_names:
      raise ValueError(
          ("A variable named '%s' already exists in this Checkpointable, but "
           "Checkpointable._add_variable called to create another with "
           "that name. Variable names must be unique within a Checkpointable "
           "object.") % (name,))
    if context.in_eager_mode():
      # If this is a variable with a single Tensor stored in the checkpoint, we
      # can set that value as an initializer rather than initializing and then
      # assigning (when executing eagerly). This call returns None if there is
      # nothing to restore.
      checkpoint_initializer = self._preload_simple_restoration(
          name=name, shape=shape)
    else:
      checkpoint_initializer = None
    if (checkpoint_initializer is not None
        and not (
            isinstance(initializer, CheckpointInitialValue)
            and initializer.restore_uid > checkpoint_initializer.restore_uid)):
      # If multiple Checkpointable objects are "creating" the same variable via
      # the magic of custom getters, the one with the highest restore UID (the
      # one called last) has to make the final initializer. If another custom
      # getter interrupts this process by overwriting the initializer, then
      # we'll catch that when we call _track_checkpointable. So this is "best
      # effort" to set the initializer with the highest restore UID.
      initializer = checkpoint_initializer
      shape = None

    new_variable = getter(
        name=name, shape=shape, dtype=dtype, initializer=initializer,
        **kwargs_for_getter)

    # If we set an initializer and the variable processed it, tracking will not
    # assign again. It will add this variable to our dependencies, and if there
    # is a non-trivial restoration queued, it will handle that. This also
    # handles slot variables.
    return self._track_checkpointable(new_variable, name=name)

  def _preload_simple_restoration(self, name, shape):
    """Return a dependency's value for restore-on-create.

    Note the restoration is not deleted; if for some reason preload is called
    and then not assigned to the variable (for example because a custom getter
    overrides the initializer), the assignment will still happen once the
    variable is tracked (determined based on checkpoint.restore_uid).

    Args:
      name: The object-local name of the dependency holding the variable's
        value.
      shape: The shape of the variable being loaded into.
    Returns:
      An callable for use as a variable's initializer/initial_value, or None if
      one should not be set (either because there was no variable with this name
      in the checkpoint or because it needs more complex deserialization). Any
      non-trivial deserialization will happen when the variable object is
      tracked.
    """
    deferred_dependencies_list = self._deferred_dependencies.get(name, ())
    if not deferred_dependencies_list:
      # Nothing to do; we don't have a restore for this dependency queued up.
      return
    for checkpoint_position in deferred_dependencies_list:
      if not checkpoint_position.is_simple_variable():
        # If _any_ pending restoration is too complicated to fit in an
        # initializer (because it has dependencies, or because there are
        # multiple Tensors to restore), bail and let the general tracking code
        # handle it.
        return None
    checkpoint_position = max(
        deferred_dependencies_list,
        key=lambda restore: restore.checkpoint.restore_uid)
    return CheckpointInitialValue(
        checkpoint_position=checkpoint_position, shape=shape)

  def _track_checkpointable(self, checkpointable, name, overwrite=False):
    """Declare a dependency on another `Checkpointable` object.

    Indicates that checkpoints for this object should include variables from
    `checkpointable`.

    Variables in a checkpoint are mapped to `Checkpointable`s based on the names
    provided when the checkpoint was written. To avoid breaking existing
    checkpoints when modifying a class, neither variable names nor dependency
    names (the names passed to `_track_checkpointable`) may change.

    Args:
      checkpointable: A `Checkpointable` which this object depends on.
      name: A local name for `checkpointable`, used for loading checkpoints into
        the correct objects.
      overwrite: Boolean, whether silently replacing dependencies is OK. Used
        for __setattr__, where throwing an error on attribute reassignment would
        be inappropriate.

    Returns:
      `checkpointable`, for convenience when declaring a dependency and
      assigning to a member variable in one statement.

    Raises:
      TypeError: If `checkpointable` does not inherit from `Checkpointable`.
      ValueError: If another object is already tracked by this name.
    """
    self._maybe_initialize_checkpointable()
    if not isinstance(checkpointable, CheckpointableBase):
      raise TypeError(
          ("Checkpointable._track_checkpointable() passed type %s, not a "
           "Checkpointable.") % (type(checkpointable),))
    new_reference = _CheckpointableReference(name=name, ref=checkpointable)
    if (name in self._dependency_names
        and self._dependency_names[name] is not checkpointable):
      if not overwrite:
        raise ValueError(
            ("Called Checkpointable._track_checkpointable() with name='%s', "
             "but a Checkpointable with this name is already declared as a "
             "dependency. Names must be unique (or overwrite=True).") % (name,))
      # This is a weird thing to do, but we're not going to stop people from
      # using __setattr__.
      for index, (old_name, _) in enumerate(self._checkpoint_dependencies):
        if name == old_name:
          self._checkpoint_dependencies[index] = new_reference
    else:
      self._checkpoint_dependencies.append(new_reference)

    self._dependency_names[name] = checkpointable
    deferred_dependency_list = self._deferred_dependencies.pop(name, None)
    if deferred_dependency_list is not None:
      for checkpoint_position in deferred_dependency_list:
        checkpoint_position.restore(checkpointable=checkpointable)
    return checkpointable

  def _restore_from_checkpoint_position(self, checkpoint_position):
    """Restore this object and its dependencies (may be deferred)."""
    # Attempt a breadth-first traversal, since presumably the user has more
    # control over shorter paths. If we don't have all of the dependencies at
    # this point, the end result is not breadth-first (since other deferred
    # traversals will happen later).
    visit_queue = collections.deque([checkpoint_position])
    restore_ops = []
    while visit_queue:
      current_position = visit_queue.popleft()
      restore_ops.extend(nest.flatten(
          current_position.checkpointable  # pylint: disable=protected-access
          ._single_restoration_from_checkpoint_position(
              checkpoint_position=current_position,
              visit_queue=visit_queue)))
    return restore_ops

  def _single_restoration_from_checkpoint_position(
      self, checkpoint_position, visit_queue):
    """Restore this object, and either queue its dependencies or defer them."""
    self._maybe_initialize_checkpointable()
    checkpoint = checkpoint_position.checkpoint
    # If the UID of this restore is lower than our current update UID, we don't
    # need to actually restore the object. However, we should pass the
    # restoration on to our dependencies.
    if checkpoint.restore_uid > self._update_uid:
      restore_ops = checkpoint_position.restore_ops()
      # TODO(allenl): Get a list of feeds for saving Python state
      self._update_uid = checkpoint.restore_uid
    else:
      restore_ops = ()
    for child in checkpoint_position.object_proto.children:
      child_position = _CheckpointPosition(
          checkpoint=checkpoint,
          proto_id=child.node_id)
      local_object = self._dependency_names.get(child.local_name, None)
      if local_object is None:
        # We don't yet have a dependency registered with this name. Save it
        # in case we do.
        self._deferred_dependencies.setdefault(child.local_name, []).append(
            child_position)
      else:
        if child_position.bind_object(checkpointable=local_object):
          # This object's correspondence is new, so dependencies need to be
          # visited. Delay doing it so that we get a breadth-first dependency
          # resolution order (shallowest paths first). The caller is responsible
          # for emptying visit_queue.
          visit_queue.append(child_position)
    return restore_ops

  def _gather_saveables_for_checkpoint(self):
    """Returns a dictionary of values to checkpoint with this object.

    Keys in the returned dictionary are local to this object and in a separate
    namespace from dependencies. Values may either be `SaveableObject`s or
    variables easily converted to `SaveableObject`s (as in `tf.train.Saver`'s
    `var_list` constructor argument).

    Returned values must be saved only by this object; if any value may be
    shared, it should instead be a dependency. For example, variable objects
    save their own values with the key `VARIABLE_VALUE_KEY`, but objects which
    reference variables simply add a dependency.
    """
    return {}


class Checkpointable(CheckpointableBase):
  """Manages dependencies on other objects.

  `Checkpointable` objects may have dependencies: other `Checkpointable` objects
  which should be saved if the object declaring the dependency is saved. A
  correctly saveable program has a dependency graph such that if changing a
  global variable affects an object (e.g. changes the behavior of any of its
  methods) then there is a chain of dependencies from the influenced object to
  the variable.

  Dependency edges have names, and are created implicitly when a
  `Checkpointable` object is assigned to an attribute of another
  `Checkpointable` object. For example:

  ```
  obj = Checkpointable()
  obj.v = ResourceVariable(0.)
  ```

  The `Checkpointable` object `obj` now has a dependency named "v" on a
  variable.

  `Checkpointable` objects may specify `Tensor`s to be saved and restored
  directly (e.g. a `Variable` indicating how to save itself) rather than through
  dependencies on other objects. See
  `Checkpointable._gather_saveables_for_checkpoint` for details.
  """

  def __setattr__(self, name, value):
    """Support self.foo = checkpointable syntax."""
    # Perform the attribute assignment, and potentially call other __setattr__
    # overrides such as that for tf.keras.Model.
    super(Checkpointable, self).__setattr__(name, value)
    if isinstance(value, CheckpointableBase):
      self._track_checkpointable(
          value, name=name,
          # Allow the user to switch the Checkpointable which is tracked by this
          # name, since assigning a new variable to an attribute has
          # historically been fine (e.g. Adam did this).
          # TODO(allenl): Should this be a warning once Checkpointable save/load
          # is usable?
          overwrite=True)
