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

import abc
import collections

import six

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_io_ops as io_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator

# Key where the object graph proto is saved in a TensorBundle
OBJECT_GRAPH_PROTO_KEY = "_CHECKPOINTABLE_OBJECT_GRAPH"

# A key indicating a variable's value in an object's checkpointed Tensors
# (Trackable._gather_saveables_for_checkpoint). If this is the only key and
# the object has no dependencies, then its value may be restored on object
# creation (avoiding double assignment when executing eagerly).
VARIABLE_VALUE_KEY = "VARIABLE_VALUE"
OBJECT_CONFIG_JSON_KEY = "OBJECT_CONFIG_JSON"

TrackableReference = collections.namedtuple(
    "TrackableReference",
    [
        # The local name for this dependency.
        "name",
        # The Trackable object being referenced.
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
    self.wrapped_value = checkpoint_position.value_tensors()[VARIABLE_VALUE_KEY]
    if shape:
      # We need to set the static shape information on the initializer if
      # possible so we don't get a variable with an unknown shape.
      self.wrapped_value.set_shape(shape)
    self._checkpoint_position = checkpoint_position

  def __getattr__(self, attr):
    try:
      return getattr(self.wrapped_value, attr)
    except AttributeError:
      return self.__getattribute__(attr)

  @property
  def checkpoint_position(self):
    return self._checkpoint_position


class NoRestoreSaveable(saveable_object.SaveableObject):
  """Embeds a tensor in a checkpoint with no restore ops."""

  def __init__(self, tensor, name, dtype=None, device=None):
    spec = saveable_object.SaveSpec(
        tensor, "", name, dtype=dtype, device=device)
    super(NoRestoreSaveable, self).__init__(tensor, [spec], name)

  def restore(self, restored_tensors, restored_shapes):
    return control_flow_ops.no_op()


@six.add_metaclass(abc.ABCMeta)
class PythonStateSaveable(saveable_object.SaveableObject):
  """An interface for saving/restoring volatile Python state."""

  @abc.abstractmethod
  def feed_dict_additions(self):
    """When running a graph, indicates fresh state to feed.

    Returns:
      A dictionary mapping `Tensor`s to current Python state.
    """
    pass

  @abc.abstractmethod
  def freeze(self):
    """Create a new `SaveableObject` which freezes current state as a constant.

    Used when executing eagerly to embed the current state as a constant, or
    when creating a static tf.compat.v1.train.Saver with the frozen current
    Python state.

    Returns:
      A `SaveableObject` which is not a `PythonStateSaveable` instance (i.e. has
      no Python state associated with it).
    """
    pass


class PythonStringStateSaveable(PythonStateSaveable):
  """Saves Python state in a checkpoint."""

  def __init__(self, name, state_callback, restore_callback=None):
    """Configure saving.

    Args:
      name: The checkpoint key to write to.
      state_callback: A function taking no arguments which returns a string.
        This function is run every time a checkpoint is written.
      restore_callback: A function taking a Python string, used to restore
        state. Optional; defaults to doing nothing, in which case it is ignored
        by status assertions such as assert_consumed().
    """
    self._has_trivial_state_callback = (restore_callback is None)

    def _state_callback_wrapper():
      with ops.init_scope():
        return state_callback()

    self._state_callback = _state_callback_wrapper
    self._restore_callback = restore_callback
    with ops.device("/cpu:0"):
      self._save_string = constant_op.constant("", dtype=dtypes.string)
    spec = saveable_object.SaveSpec(
        self._save_string, "", name, dtype=dtypes.string)
    super(PythonStringStateSaveable, self).__init__(self._save_string, [spec],
                                                    name)

  @property
  def optional_restore(self):
    """For values with no restore, relaxes assert_consumed()."""
    return self._has_trivial_state_callback

  def feed_dict_additions(self):
    """When running a graph, indicates fresh state to feed."""
    return {self._save_string: self._state_callback()}

  def freeze(self):
    """Create a frozen `SaveableObject` which saves the current state."""

    def _constant_state():
      return constant_op.constant(self._state_callback(), dtype=dtypes.string)

    return NoRestoreSaveable(
        tensor=_constant_state,
        dtype=dtypes.string,
        name=self.name,
        device="cpu:0")

  def python_restore(self, restored_strings):
    """Called to restore Python state."""
    if self._restore_callback:
      restored, = restored_strings
      self._restore_callback(restored)

  def restore(self, restored_tensors, restored_shapes):
    """Called to restore TensorFlow state (nothing to do)."""
    return control_flow_ops.no_op()


class CheckpointPosition(object):
  """Indicates a position within a `_CheckpointRestoreCoordinator`."""

  def __init__(self, checkpoint, proto_id):
    """Specify an object within a checkpoint.

    Args:
      checkpoint: A _CheckpointRestoreCoordinator object.
      proto_id: The index of this object in TrackableObjectGraph.nodes.
    """
    self._checkpoint = checkpoint
    self._proto_id = proto_id

  def restore(self, trackable):
    """Restore this value into `trackable`."""
    with ops.init_scope():
      if self.bind_object(trackable):
        # This object's correspondence with a checkpointed object is new, so
        # process deferred restorations for it and its dependencies.
        restore_ops = trackable._restore_from_checkpoint_position(self)  # pylint: disable=protected-access
        if restore_ops:
          self._checkpoint.new_restore_ops(restore_ops)

  def bind_object(self, trackable):
    """Set a checkpoint<->object correspondence and process slot variables.

    Args:
      trackable: The object to record a correspondence for.

    Returns:
      True if this is a new assignment, False if this object has already been
      mapped to a checkpointed `Object` proto.
    Raises:
      AssertionError: If another object is already bound to the `Object` proto.
    """
    checkpoint = self.checkpoint
    checkpoint.all_python_objects.add(trackable)
    current_assignment = checkpoint.object_by_proto_id.get(self._proto_id, None)
    checkpoint.matched_proto_ids.add(self._proto_id)
    if current_assignment is None:
      checkpoint.object_by_proto_id[self._proto_id] = trackable
      for deferred_slot_restoration in (
          checkpoint.deferred_slot_restorations.pop(self._proto_id, ())):
        trackable._create_or_restore_slot_variable(  # pylint: disable=protected-access
            slot_variable_position=CheckpointPosition(
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
                      original_variable=trackable,
                      slot_variable_id=slot_restoration.slot_variable_id,
                      slot_name=slot_restoration.slot_name))
        else:
          optimizer_object._create_or_restore_slot_variable(  # pylint: disable=protected-access
              slot_variable_position=CheckpointPosition(
                  checkpoint=checkpoint,
                  proto_id=slot_restoration.slot_variable_id),
              variable=trackable,
              slot_name=slot_restoration.slot_name)
      return True  # New assignment
    else:
      # The object was already mapped for this checkpoint load, which means
      # we don't need to do anything besides check that the mapping is
      # consistent (if the dependency DAG is not a tree then there are
      # multiple paths to the same object).
      if current_assignment is not trackable:
        logging.warning((
            "Inconsistent references when loading the checkpoint into this "
            "object graph. Either the Trackable object references in the "
            "Python program have changed in an incompatible way, or the "
            "checkpoint was generated in an incompatible program.\n\nTwo "
            "checkpoint references resolved to different objects (%s and %s)."),
                        current_assignment, trackable)
      return False  # Not a new assignment

  def is_simple_variable(self):
    """Determine whether this value is restorable with a Tensor initializer."""
    attributes = self.object_proto.attributes
    return (len(attributes) == 1 and
            attributes[0].name == VARIABLE_VALUE_KEY and
            not self.object_proto.children)

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
        with ops.device("/cpu:0"):
          # Run the restore itself on the CPU.
          value, = io_ops.restore_v2(
              prefix=self._checkpoint.save_path_tensor,
              tensor_names=[checkpoint_key],
              shape_and_slices=[""],
              dtypes=[base_type],
              name="%s_checkpoint_read" % (serialized_tensor.name,))
        # Copy the value to the current device if necessary.
        value_tensors[serialized_tensor.name] = array_ops.identity(value)
      return value_tensors

  def _gather_ops_or_named_saveables(self):
    """Looks up or creates SaveableObjects which don't have cached ops."""
    saveables = self.trackable._gather_saveables_for_checkpoint()  # pylint: disable=protected-access
    # Name saveables based on the name this object had when it was checkpointed.
    named_saveables = {}
    python_saveables = []
    existing_restore_ops = []
    for serialized_tensor in self.object_proto.attributes:
      if context.executing_eagerly():
        existing_op = None
      else:
        existing_op = self._checkpoint.restore_ops_by_name.get(
            serialized_tensor.checkpoint_key, None)
      if existing_op is not None:
        existing_restore_ops.append(existing_op)
        continue

      # Only if we don't have cached ops for this SaveableObject, we'll see if
      # the SaveableObject itself has been cached. If not, we'll make it, and
      # either way we'll extract new ops from it (or if it has Python state to
      # restore, we'll run that).
      saveables_cache = self._checkpoint.graph_view.saveables_cache
      if saveables_cache is None:
        # No SaveableObject caching when executing eagerly.
        saveable = None
      else:
        # If we've already created and cached a SaveableObject for this
        # attribute, we can re-use it to avoid re-creating some ops when graph
        # building.
        saveable_list = saveables_cache.get(self.trackable,
                                            {}).get(serialized_tensor.name,
                                                    (None,))
        if len(saveable_list) == 1:
          # Almost every attribute will have exactly one SaveableObject.
          saveable, = saveable_list
        else:
          # Don't use cached SaveableObjects for partitioned variables, which is
          # the only case where we'd have a list of SaveableObjects. Op caching
          # will catch them.
          saveable = None
      if saveable is not None:
        # The name of this attribute has changed, so we need to re-generate
        # the SaveableObject.
        if serialized_tensor.checkpoint_key not in saveable.name:
          saveable = None
          del saveables_cache[self.trackable]
          break
      if saveable is None:
        # If there was no cached SaveableObject, we should check if the Python
        # object has the attribute.
        saveable_factory = saveables.get(serialized_tensor.name, None)
        if saveable_factory is None:
          # Purposefully does not throw an exception if attributes have been
          # added or deleted. Stores unused attributes so an exception can be
          # raised if the user decides to check that everything in the
          # checkpoint was loaded.
          if not serialized_tensor.optional_restore:
            self._checkpoint.unused_attributes.setdefault(
                self._proto_id, []).append(serialized_tensor.name)
          continue
        if callable(saveable_factory):
          saveable = saveable_factory(name=serialized_tensor.checkpoint_key)
        else:
          saveable = saveable_factory
        if saveables_cache is not None:
          saveables_cache.setdefault(self.trackable,
                                     {})[serialized_tensor.name] = [saveable]
      if isinstance(saveable, PythonStateSaveable):
        python_saveables.append(saveable)
      else:
        named_saveables[serialized_tensor.checkpoint_key] = saveable
    return existing_restore_ops, named_saveables, python_saveables

  def restore_ops(self):
    """Create or fetch restore ops for this object's attributes.

    Requires that the `Trackable` Python object has been bound to an object
    ID in the checkpoint.

    Returns:
      A list of operations when graph building, or an empty list when executing
      eagerly.
    """
    (restore_ops, tensor_saveables,
     python_saveables) = self._gather_ops_or_named_saveables()
    restore_ops.extend(
        self._checkpoint.restore_saveables(tensor_saveables, python_saveables))
    return restore_ops

  @property
  def checkpoint(self):
    return self._checkpoint

  @property
  def trackable(self):
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
    "_DeferredSlotVariableRestoration", [
        "original_variable",
        "slot_variable_id",
        "slot_name",
    ])

_SlotVariableRestoration = collections.namedtuple(
    "_SlotVariableRestoration",
    [
        # The checkpoint proto id of the optimizer object.
        "optimizer_id",
        # The checkpoint proto id of the slot variable.
        "slot_variable_id",
        "slot_name",
    ])


def no_automatic_dependency_tracking(method):
  """Disables automatic dependency tracking on attribute assignment.

  Use to decorate any method of a Trackable object. Attribute assignment in
  that method will not add dependencies (also respected in Model). Harmless if
  used in a class which does not do automatic dependency tracking (which means
  it's safe to use in base classes which may have subclasses which also inherit
  from Trackable).

  Args:
    method: The method to decorate.

  Returns:
    A decorated method which sets and un-sets automatic dependency tracking for
    the object the method is called on (not thread safe).
  """

  def _method_wrapper(self, *args, **kwargs):
    previous_value = getattr(self, "_self_setattr_tracking", True)
    self._self_setattr_tracking = False  # pylint: disable=protected-access
    try:
      result = method(self, *args, **kwargs)
    finally:
      self._self_setattr_tracking = previous_value  # pylint: disable=protected-access
    return result

  return tf_decorator.make_decorator(
      target=method, decorator_func=_method_wrapper)


class Trackable(object):
  """Base class for `Trackable` objects without automatic dependencies.

  This class has no __setattr__ override for performance reasons. Dependencies
  must be added explicitly. Unless attribute assignment is performance-critical,
  use `AutoTrackable` instead. Use `Trackable` for `isinstance`
  checks.
  """

  # For compatibility with wrapt.ObjectProxy, attributes are all prefixed with
  # _self_. We have some properties to forward semi-public attributes to their
  # _self_ equivalents.

  @property
  def _setattr_tracking(self):
    if not hasattr(self, "_self_setattr_tracking"):
      self._self_setattr_tracking = True
    return self._self_setattr_tracking

  @_setattr_tracking.setter
  def _setattr_tracking(self, value):
    self._self_setattr_tracking = value

  @property
  def _update_uid(self):
    return self._self_update_uid

  @_update_uid.setter
  def _update_uid(self, value):
    self._self_update_uid = value

  @property
  def _unconditional_checkpoint_dependencies(self):
    return self._self_unconditional_checkpoint_dependencies

  @property
  def _unconditional_dependency_names(self):
    return self._self_unconditional_dependency_names

  @property
  def _name_based_restores(self):
    return self._self_name_based_restores

  # Trackable does not do automatic dependency tracking, but uses the
  # no_automatic_dependency_tracking decorator so it can avoid adding
  # dependencies if a subclass is Trackable / inherits from Model (both of
  # which have __setattr__ overrides).
  @no_automatic_dependency_tracking
  def _maybe_initialize_trackable(self):
    """Initialize dependency management.

    Not __init__, since most objects will forget to call it.
    """
    if hasattr(self, "_self_unconditional_checkpoint_dependencies"):
      # __init__ already called. This check means that we don't need
      # Trackable.__init__() in the constructor of every TensorFlow object.
      return
    # A list of TrackableReference objects. Some classes implementing
    # `Trackable`, notably `Optimizer`s, may override the
    # _checkpoint_dependencies property with conditional dependencies
    # (e.g. based on the current graph when saving).
    self._self_unconditional_checkpoint_dependencies = []
    # Maps names -> Trackable objects
    self._self_unconditional_dependency_names = {}
    # Restorations for other Trackable objects on which this object may
    # eventually depend. Maps local name -> CheckpointPosition list. Optimizers
    # tack on conditional dependencies, and so need separate management of
    # deferred dependencies too.
    self._self_unconditional_deferred_dependencies = {}
    # The UID of the highest assignment to this object. Used to ensure that the
    # last requested assignment determines the final value of an object.
    if hasattr(self, "_self_update_uid"):
      raise AssertionError(
          "Internal error: the object had an update UID set before its "
          "initialization code was run.")
    self._self_update_uid = -1
    # When executing eagerly, holds a collection of _NameBasedRestoreCoordinator
    # instances, which should be checked when creating variables or other
    # saveables. These are passed on recursively to all dependencies, since
    # unlike object-based checkpoint restores we don't know which subgraph is
    # being restored in advance. This mechanism is only necessary for
    # restore-on-create when executing eagerly, and so is unused when graph
    # building.
    self._self_name_based_restores = set()

  def _no_dependency(self, value):
    """If automatic dependency tracking is enabled, ignores `value`."""
    return value

  def _name_based_attribute_restore(self, checkpoint):
    """Restore the object's attributes from a name-based checkpoint."""
    self._self_name_based_restores.add(checkpoint)
    if self._self_update_uid < checkpoint.restore_uid:
      checkpoint.eager_restore(self)
      self._self_update_uid = checkpoint.restore_uid

  @property
  def _checkpoint_dependencies(self):
    """All dependencies of this object.

    May be overridden to include conditional dependencies.

    Returns:
      A list of `TrackableReference` objects indicating named
      `Trackable` dependencies which should be saved along with this
      object.
    """
    return self._self_unconditional_checkpoint_dependencies

  @property
  def _deferred_dependencies(self):
    """A dictionary with deferred dependencies.

    Stores restorations for other Trackable objects on which this object
    may eventually depend. May be overridden by sub-classes (e.g. Optimizers use
    conditional dependencies based the current graph, and so need separate
    management of deferred dependencies too).

    Returns:
      A dictionary mapping from local name to a list of CheckpointPosition
      objects.
    """
    return self._self_unconditional_deferred_dependencies

  def _lookup_dependency(self, name):
    """Look up a dependency by name.

    May be overridden to include conditional dependencies.

    Args:
      name: The local name of the dependency.

    Returns:
      A `Trackable` object, or `None` if no dependency by this name was
      found.
    """
    return self._self_unconditional_dependency_names.get(name, None)

  def _add_variable_with_custom_getter(self,
                                       name,
                                       shape=None,
                                       dtype=dtypes.float32,
                                       initializer=None,
                                       getter=None,
                                       overwrite=False,
                                       **kwargs_for_getter):
    """Restore-on-create for a variable be saved with this `Trackable`.

    If the user has requested that this object or another `Trackable` which
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
      overwrite: If True, disables unique name and type checks.
      **kwargs_for_getter: Passed to the getter.

    Returns:
      The new variable object.

    Raises:
      ValueError: If the variable name is not unique.
    """
    self._maybe_initialize_trackable()
    with ops.init_scope():
      if context.executing_eagerly():
        # If this is a variable with a single Tensor stored in the checkpoint,
        # we can set that value as an initializer rather than initializing and
        # then assigning (when executing eagerly). This call returns None if
        # there is nothing to restore.
        checkpoint_initializer = self._preload_simple_restoration(
            name=name, shape=shape)
      else:
        checkpoint_initializer = None
      if (checkpoint_initializer is not None and
          not (isinstance(initializer, CheckpointInitialValue) and
               (initializer.restore_uid > checkpoint_initializer.restore_uid))):
        # If multiple Trackable objects are "creating" the same variable
        # via the magic of custom getters, the one with the highest restore UID
        # (the one called last) has to make the final initializer. If another
        # custom getter interrupts this process by overwriting the initializer,
        # then we'll catch that when we call _track_trackable. So this is
        # "best effort" to set the initializer with the highest restore UID.
        initializer = checkpoint_initializer
        shape = None
    new_variable = getter(
        name=name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        **kwargs_for_getter)

    # If we set an initializer and the variable processed it, tracking will not
    # assign again. It will add this variable to our dependencies, and if there
    # is a non-trivial restoration queued, it will handle that. This also
    # handles slot variables.
    if not overwrite or isinstance(new_variable, Trackable):
      return self._track_trackable(new_variable, name=name, overwrite=overwrite)
    else:
      # TODO(allenl): Some variable types are not yet supported. Remove this
      # fallback once all get_variable() return types are Trackable.
      return new_variable

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

  def _track_trackable(self, trackable, name, overwrite=False):
    """Declare a dependency on another `Trackable` object.

    Indicates that checkpoints for this object should include variables from
    `trackable`.

    Variables in a checkpoint are mapped to `Trackable`s based on the names
    provided when the checkpoint was written. To avoid breaking existing
    checkpoints when modifying a class, neither variable names nor dependency
    names (the names passed to `_track_trackable`) may change.

    Args:
      trackable: A `Trackable` which this object depends on.
      name: A local name for `trackable`, used for loading checkpoints into the
        correct objects.
      overwrite: Boolean, whether silently replacing dependencies is OK. Used
        for __setattr__, where throwing an error on attribute reassignment would
        be inappropriate.

    Returns:
      `trackable`, for convenience when declaring a dependency and
      assigning to a member variable in one statement.

    Raises:
      TypeError: If `trackable` does not inherit from `Trackable`.
      ValueError: If another object is already tracked by this name.
    """
    self._maybe_initialize_trackable()
    if not isinstance(trackable, Trackable):
      raise TypeError(("Trackable._track_trackable() passed type %s, not a "
                       "Trackable.") % (type(trackable),))
    new_reference = TrackableReference(name=name, ref=trackable)
    current_object = self._lookup_dependency(name)
    if (current_object is not None and current_object is not trackable):
      if not overwrite:
        raise ValueError(
            ("Called Trackable._track_trackable() with name='%s', "
             "but a Trackable with this name is already declared as a "
             "dependency. Names must be unique (or overwrite=True).") % (name,))
      # This is a weird thing to do, but we're not going to stop people from
      # using __setattr__.
      for index, (old_name, _) in enumerate(
          self._self_unconditional_checkpoint_dependencies):
        if name == old_name:
          self._self_unconditional_checkpoint_dependencies[
              index] = new_reference
    elif current_object is None:
      self._self_unconditional_checkpoint_dependencies.append(new_reference)
      self._handle_deferred_dependencies(name=name, trackable=trackable)
    self._self_unconditional_dependency_names[name] = trackable
    return trackable

  def _handle_deferred_dependencies(self, name, trackable):
    """Pop and load any deferred checkpoint restores into `trackable`.

    This method does not add a new dependency on `trackable`, but it does
    check if any outstanding/deferred dependencies have been queued waiting for
    this dependency to be added (matched based on `name`). If so,
    `trackable` and its dependencies are restored. The restorations are
    considered fulfilled and so are deleted.

    `_track_trackable` is more appropriate for adding a
    normal/unconditional dependency, and includes handling for deferred
    restorations. This method allows objects such as `Optimizer` to use the same
    restoration logic while managing conditional dependencies themselves, by
    overriding `_checkpoint_dependencies` and `_lookup_dependency` to change the
    object's dependencies based on the context it is saved/restored in (a single
    optimizer instance can have state associated with multiple graphs).

    Args:
      name: The name of the dependency within this object (`self`), used to
        match `trackable` with values saved in a checkpoint.
      trackable: The Trackable object to restore (inheriting from `Trackable`).
    """
    self._maybe_initialize_trackable()
    trackable._maybe_initialize_trackable()  # pylint: disable=protected-access
    deferred_dependencies_list = self._deferred_dependencies.pop(name, ())
    for checkpoint_position in sorted(
        deferred_dependencies_list,
        key=lambda restore: restore.checkpoint.restore_uid,
        reverse=True):
      checkpoint_position.restore(trackable)

    # Pass on any name-based restores queued in this object.
    for name_based_restore in sorted(
        self._self_name_based_restores,
        key=lambda checkpoint: checkpoint.restore_uid,
        reverse=True):
      trackable._name_based_attribute_restore(name_based_restore)  # pylint: disable=protected-access

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
      restore_ops.extend(
          nest.flatten(current_position.trackable  # pylint: disable=protected-access
                       ._single_restoration_from_checkpoint_position(
                           checkpoint_position=current_position,
                           visit_queue=visit_queue)))
    return restore_ops

  def _single_restoration_from_checkpoint_position(self, checkpoint_position,
                                                   visit_queue):
    """Restore this object, and either queue its dependencies or defer them."""
    self._maybe_initialize_trackable()
    checkpoint = checkpoint_position.checkpoint
    # If the UID of this restore is lower than our current update UID, we don't
    # need to actually restore the object. However, we should pass the
    # restoration on to our dependencies.
    if checkpoint.restore_uid > self._self_update_uid:
      restore_ops = checkpoint_position.restore_ops()
      self._self_update_uid = checkpoint.restore_uid
    else:
      restore_ops = ()
    for child in checkpoint_position.object_proto.children:
      child_position = CheckpointPosition(
          checkpoint=checkpoint, proto_id=child.node_id)
      local_object = self._lookup_dependency(child.local_name)
      if local_object is None:
        # We don't yet have a dependency registered with this name. Save it
        # in case we do.
        self._deferred_dependencies.setdefault(child.local_name,
                                               []).append(child_position)
      else:
        if child_position.bind_object(trackable=local_object):
          # This object's correspondence is new, so dependencies need to be
          # visited. Delay doing it so that we get a breadth-first dependency
          # resolution order (shallowest paths first). The caller is responsible
          # for emptying visit_queue.
          visit_queue.append(child_position)
    return restore_ops

  def _gather_saveables_for_checkpoint(self):
    """Returns a dictionary of values to checkpoint with this object.

    Keys in the returned dictionary are local to this object and in a separate
    namespace from dependencies. Values may either be `SaveableObject` factories
    or variables easily converted to `SaveableObject`s (as in
    `tf.compat.v1.train.Saver`'s
    `var_list` constructor argument).

    `SaveableObjects` have a name set, which Trackable needs to generate
    itself. So rather than returning `SaveableObjects` directly, this method
    should return a dictionary of callables which take `name` arguments and
    return `SaveableObjects` with that name.

    If this object may also be passed to the global-name-based
    `tf.compat.v1.train.Saver`,
    the returned callables should have a default value for their name argument
    (i.e. be callable with no arguments).

    Returned values must be saved only by this object; if any value may be
    shared, it should instead be a dependency. For example, variable objects
    save their own values with the key `VARIABLE_VALUE_KEY`, but objects which
    reference variables simply add a dependency.

    Returns:
      The dictionary mapping attribute names to `SaveableObject` factories
      described above. For example:
      {VARIABLE_VALUE_KEY:
       lambda name="global_name_for_this_object":
       SaveableObject(name=name, ...)}
    """
    return {}

  def _list_functions_for_serialization(self):
    """Lists the functions of this trackable to serialize.

    Internal sub-classes can override this with specific logic. E.g.
    `AutoTrackable` provides an implementation that returns the `attr`
    that return functions.

    Returns:
        A dictionary mapping attribute names to `Function` or
        `ConcreteFunction`.
    """
    return dict()
