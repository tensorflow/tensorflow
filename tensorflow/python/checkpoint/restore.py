# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Logic for restoring checkpointed values for Trackables."""

import collections

from tensorflow.python.checkpoint import checkpoint_view
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops as io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base
from tensorflow.python.trackable import constants
from tensorflow.python.trackable import python_state
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.util import object_identity


class CheckpointPosition(object):
  """Indicates a position within a `_CheckpointRestoreCoordinator`."""

  __slots__ = ["_checkpoint", "_proto_id", "skip_restore"]

  def __init__(self, checkpoint, proto_id):
    """Specify an object within a checkpoint.

    Args:
      checkpoint: A _CheckpointRestoreCoordinator object.
      proto_id: The index of this object in TrackableObjectGraph.nodes.
    """
    self._checkpoint = checkpoint
    self._proto_id = proto_id
    # This may be set to True if the registered saver cannot be used with this
    # object.
    self.skip_restore = False

  def restore(self, trackable):
    """Restore this value into `trackable`."""
    with ops.init_scope():
      if self.bind_object(trackable):
        # This object's correspondence with a checkpointed object is new, so
        # process deferred restorations for it and its dependencies.
        restore_ops = self._restore_descendants()
        if restore_ops:
          self._checkpoint.new_restore_ops(restore_ops)

  def bind_object(self, trackable):
    """Set a checkpoint<->object correspondence.

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
      return True  # New assignment
    else:
      # The object was already mapped for this checkpoint load, which means
      # we don't need to do anything besides check that the mapping is
      # consistent (if the dependency DAG is not a tree then there are
      # multiple paths to the same object).
      if current_assignment is not trackable:
        logging.warning(
            "Inconsistent references when loading the checkpoint into this "
            "object graph. For example, in the saved checkpoint object, "
            "`model.layer.weight` and `model.layer_copy.weight` reference the "
            "same variable, while in the current object these are two different"
            " variables. The referenced variables are:"
            f"({current_assignment} and {trackable}).")
      return False  # Not a new assignment

  def is_simple_variable(self):
    """Determine whether this value is restorable with a Tensor initializer."""
    attributes = self.object_proto.attributes
    return (len(attributes) == 1 and
            attributes[0].name == constants.VARIABLE_VALUE_KEY and
            not self.object_proto.children)

  def value_tensors(self, shape_and_slices=None):
    """Create value `Tensor`s for this object's attributes.

    Does not require that the Python object has been created. Used for
    restore-on-create when executing eagerly.

    Args:
      shape_and_slices: A dict mapping from object attribute names to a shape
        and slice string that will be passed to a RestoreV2 op. If the dict is
        None or if an object attribute is not in the dict, the full tensor will
        be restored.

    Returns:
      A dictionary mapping from object attribute names to `Tensor`s.
    """
    value_tensors = {}
    for serialized_tensor in self.object_proto.attributes:
      checkpoint_key = serialized_tensor.checkpoint_key
      dtype = self._checkpoint.dtype_map[checkpoint_key]
      base_type = dtype.base_dtype
      io_device = self._checkpoint.options.experimental_io_device or "cpu:0"
      with ops.init_scope():
        with ops.device(io_device):
          # Run the restore itself on the io_device(CPU or specified).
          if (shape_and_slices is not None and
              serialized_tensor.name in shape_and_slices):
            shape_and_slice = shape_and_slices[serialized_tensor.name]
          else:
            shape_and_slice = ""
          value, = io_ops.restore_v2(
              prefix=self._checkpoint.save_path_tensor,
              tensor_names=[checkpoint_key],
              shape_and_slices=[shape_and_slice],
              dtypes=[base_type],
              name="%s_checkpoint_read" % (serialized_tensor.name,))
        # Copy the value to the current device if necessary.
        value_tensors[serialized_tensor.name] = array_ops.identity(value)
    return value_tensors

  def gather_ops_or_named_saveables(self):
    """Looks up or creates SaveableObjects which don't have cached ops.

    Returns:
      A tuple of (
          existing_restore_ops: list,
          named_saveables: dict,
          python_positions: list,
          registered_savers: dict)
    """
    # pylint:disable=g-import-not-at-top
    # There are circular dependencies between Trackable and SaveableObject,
    # so we must import it here.
    # TODO(b/224069573): Remove this code from Trackable.
    from tensorflow.python.training.saving import saveable_object_util
    # pylint:enable=g-import-not-at-top

    recorded_registered_saver = self.get_registered_saver_name()
    if not (self.object_proto.attributes or recorded_registered_saver):
      return [], {}, [], {}

    existing_restore_ops = []
    named_saveables = {}
    python_positions = []
    registered_savers = collections.defaultdict(dict)

    saveable_factories = saveable_object_util.saveable_objects_from_trackable(
        self.trackable)
    saver_name = registration.get_registered_saver_name(self.trackable)

    if recorded_registered_saver:
      if not self.skip_restore:
        name = self.object_proto.registered_saver.object_name
        registered_savers[recorded_registered_saver][name] = self.trackable
      # Else: Skip restoration of this Trackable. This skip only happens if the
      # registered saver has enabled `option_restore`. Otherwise, an error would
      # have been raised at `self.get_registered_saver_name()`.
    elif saver_name:
      # In this case, the checkpoint has a recorded serialized tensor but no
      # registered saver, while the Trackable loading the checkpoint has
      # migrated to the registered checkpoint functionality (TPUEmbedding is an
      # example of this).

      # Set the Trackable's object name to the first checkpoint key that is
      # stored in checkpoint. If there is a use case that requires the other
      # keys, then we can take another look at this.
      registered_savers[saver_name] = {
          self.object_proto.attributes[0].checkpoint_key: self.trackable
      }
    elif isinstance(self.trackable, python_state.PythonState):
      python_positions.append(self)
    elif saveable_factories.keys() == {
        trackable_utils.SERIALIZE_TO_TENSORS_NAME
    }:
      existing_restore_ops, named_saveables = (
          self._create_serialize_to_tensor_saveable(saveable_factories))
    elif saveable_factories:
      existing_restore_ops, named_saveables = (
          self._create_saveables_by_attribute_name(saveable_factories))
    else:
      # If no registered savers were found, then it means that one or more
      # serialized tensors were never used.
      for serialized_tensor in self.object_proto.attributes:
        self._checkpoint.unused_attributes.setdefault(
            self._proto_id, []).append(serialized_tensor.name)
    return (existing_restore_ops, named_saveables, python_positions,
            registered_savers)

  def _create_serialize_to_tensor_saveable(self, saveable_factories):
    """Creates a saveable using the _serialize_to_tensor method."""
    # Extract the saveable name from the checkpoint key. This will be used as
    # the cache key or the name to pass to the saveable factory.
    suffix = saveable_compat.get_saveable_name(self.trackable) or ""
    saveable_name = _extract_saveable_name(
        self.object_proto.attributes[0].checkpoint_key) + suffix

    # Try to find the cached saveable (only in graph mode).
    if not context.executing_eagerly():
      existing_op = self._checkpoint.restore_ops_by_name.get(
          saveable_name, None)
      if existing_op is not None:
        return [existing_op], {}

      saveables_cache = self._checkpoint.saveables_cache.setdefault(
          self.trackable, {})
      if saveable_name in saveables_cache:
        return [], {saveable_name: saveables_cache[saveable_name]}

    saveable = saveable_factories[trackable_utils.SERIALIZE_TO_TENSORS_NAME](
        name=saveable_name)
    if not context.executing_eagerly():
      saveables_cache[saveable_name] = saveable
    return [], {saveable_name: saveable}

  def _create_saveables_by_attribute_name(self, saveable_factories):
    """Creates or caches SaveableObjects by matching the attribute names.

    The attribute name keys in the `saveable_factories` is used to find the
    corresponding attribute in the object proto. Attributes contain checkpoint
    keys which are passed to the factory function to generate the
    SaveableObject.

    Args:
      saveable_factories: a dict mapping attribute name to a callable factory
        function that produces a SaveableObject.

    Returns:
      A tuple of (
          existing_restore_ops: list,
          named_saveables: dict)
    """
    # Name saveables based on the name this object had when it was checkpointed.
    named_saveables = {}
    existing_restore_ops = []

    # Forward compatibility code: when loading a future checkpoint, there may
    # be multiple SerializedTensors mapped to a single saveable.
    created_compat_names = set()

    for serialized_tensor in self.object_proto.attributes:
      if context.executing_eagerly():
        existing_op = None
      else:
        existing_op = self._checkpoint.restore_ops_by_name.get(
            serialized_tensor.checkpoint_key, None)
      if existing_op is not None:
        existing_restore_ops.append(existing_op)
        continue

      if any(serialized_tensor.name.startswith(name)
             for name in created_compat_names):
        continue  # Saveable has already been created for this tensor.

      # Only if we don't have cached ops for this SaveableObject, we'll see if
      # the SaveableObject itself has been cached. If not, we'll make it, and
      # either way we'll extract new ops from it (or if it has Python state to
      # restore, we'll run that).
      saveables_cache = self._checkpoint.saveables_cache
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
      if saveable is None:
        # If there was no cached SaveableObject, create one.
        # Use the name to check if the Python object has the same attribute.
        saveable = _get_saveable_from_factory(saveable_factories,
                                              serialized_tensor,
                                              created_compat_names)
        if saveable is None:
          # Purposefully does not throw an exception if attributes have been
          # added or deleted. Stores unused attributes so an exception can be
          # raised if the user decides to check that everything in the
          # checkpoint was loaded.
          self._checkpoint.unused_attributes.setdefault(
              self._proto_id, []).append(serialized_tensor.name)
          continue
        if saveables_cache is not None:
          saveables_cache.setdefault(self.trackable,
                                     {})[serialized_tensor.name] = [saveable]
      named_saveables[serialized_tensor.checkpoint_key] = saveable

    return existing_restore_ops, named_saveables

  def restore_ops(self):
    """Create or fetch restore ops for this object's attributes.

    Requires that the `Trackable` Python object has been bound to an object
    ID in the checkpoint.

    Returns:
      A list of operations when graph building, or an empty list when executing
      eagerly.
    """
    if self._has_registered_saver():
      raise ValueError("Unable to run individual checkpoint restore for objects"
                       " with registered savers.")
    (restore_ops, tensor_saveables, python_positions,
     _) = self.gather_ops_or_named_saveables()
    restore_ops.extend(
        self._checkpoint.restore_saveables(tensor_saveables, python_positions))
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
  def proto_id(self):
    return self._proto_id

  @property
  def restore_uid(self):
    return self._checkpoint.restore_uid

  def __repr__(self):
    return repr(self.object_proto)

  def value_shape(self):
    """The shape of the VARIABLE_VALUE tensor.

    Returns:
      If found a TensorShape object, otherwise None.
    """
    for serialized_tensor in self.object_proto.attributes:
      if serialized_tensor.name == constants.VARIABLE_VALUE_KEY:
        return self._checkpoint.shape_map[serialized_tensor.checkpoint_key]
    return None

  def _has_registered_saver(self):
    return bool(self.object_proto.registered_saver.name)

  def get_registered_saver_name(self):
    """Returns the registered saver name defined in the Checkpoint."""
    if self._has_registered_saver():
      saver_name = self.object_proto.registered_saver.name
      try:
        registration.validate_restore_function(self.trackable, saver_name)
      except ValueError as e:
        if registration.get_strict_predicate_restore(saver_name):
          raise e
        self.skip_restore = True
      return saver_name
    return None

  def create_slot_variable_position(self, optimizer_object, variable,
                                    slot_variable_id, slot_name):
    """Generates CheckpointPosition for a slot variable.

    Args:
      optimizer_object: Optimizer that owns the slot variable.
      variable: Variable associated with the slot variable.
      slot_variable_id: ID of the slot variable.
      slot_name: Name of the slot variable.

    Returns:
      If there is a slot variable in the `optimizer_object` that has not been
      bound to the checkpoint, this function returns a tuple of (
        new `CheckpointPosition` for the slot variable,
        the slot variable itself).
    """
    slot_variable_position = CheckpointPosition(
        checkpoint=self.checkpoint, proto_id=slot_variable_id)
    # pylint: disable=protected-access
    slot_variable = optimizer_object._create_or_restore_slot_variable(
        slot_variable_position=slot_variable_position,
        variable=variable,
        slot_name=slot_name)
    # pylint: enable=protected-access
    if (slot_variable is not None and
        slot_variable_position.bind_object(slot_variable)):
      return slot_variable_position, slot_variable
    else:
      return None, None

  def create_child_position(self, node_id):
    return CheckpointPosition(checkpoint=self.checkpoint, proto_id=node_id)

  def _restore_descendants(self):
    """Restore the bound Trackable and dependencies (may be deferred)."""
    # Attempt a breadth-first traversal, since presumably the user has more
    # control over shorter paths. If we don't have all of the dependencies at
    # this point, the end result is not breadth-first (since other deferred
    # traversals will happen later).

    # You may be wondering why elements in the `visit_queue` are tuples that
    # contains both CheckpointPositions and their Trackable. The reason is that
    # Optimizers will not keep a strong reference to slot vars for
    # ShardedVariables. The slot variable must be kept in memory until the
    # restore saveables have been created.
    visit_queue = collections.deque([(self, self.trackable)])
    restore_ops = []
    tensor_saveables = {}
    python_positions = []
    registered_savers = collections.defaultdict(dict)
    while visit_queue:
      current_position, _ = visit_queue.popleft()

      # Restore using the ops defined in a Saveable or registered function.
      (new_restore_ops, new_tensor_saveables, new_python_positions,
       new_registered_savers) = current_position._single_restore()  # pylint: disable=protected-access
      restore_ops.extend(new_restore_ops)
      tensor_saveables.update(new_tensor_saveables)
      python_positions.extend(new_python_positions)
      for saver_name, trackable_map in new_registered_savers.items():
        registered_savers[saver_name].update(trackable_map)

      # Pass the restoration to the dependencies.
      _queue_children_for_restoration(current_position, visit_queue)
      _queue_slot_variables(current_position, visit_queue)

    restore_ops.extend(
        current_position.checkpoint.restore_saveables(tensor_saveables,
                                                      python_positions,
                                                      registered_savers))
    return restore_ops

  def _single_restore(self):
    """Restores the trackable."""
    trackable = self.trackable
    trackable._maybe_initialize_trackable()  # pylint: disable=protected-access
    checkpoint = self.checkpoint
    # If the UID of this restore is lower than our current update UID, we don't
    # need to actually restore the object.
    if checkpoint.restore_uid > trackable._update_uid:  # pylint: disable=protected-access
      restore_ops, tensor_saveables, python_positions, registered_savers = (
          self.gather_ops_or_named_saveables())
      trackable._update_uid = checkpoint.restore_uid  # pylint: disable=protected-access
    else:
      restore_ops = ()
      tensor_saveables = {}
      python_positions = ()
      registered_savers = {}
    return restore_ops, tensor_saveables, python_positions, registered_savers


def restore_nodes(save_path, nodes_to_restore):
  """Restores nodes from a dict.

  Requires that the `Trackable` Python object has been bound to an object
  ID in the checkpoint.

  Args:
    save_path: a string represents path to the checkpoint.
    nodes_to_restore: a dict maps `node_id` to `trackable` to be restored.
  """
  if save_path is None:
    raise ValueError("save_path cannot be empty.")
  if not isinstance(nodes_to_restore, dict):
    raise ValueError(
        "Expecting a dictionary of node_id to Trackable for nodes_to_restore.")

  # pylint:disable=g-import-not-at-top
  # There are circular dependencies between Trackable and SaveableObject,
  # so we must import it here.
  from tensorflow.python.training.saving import saveable_object_util
  # pylint:enable=g-import-not-at-top

  ckpt_view = checkpoint_view.CheckpointView(save_path)
  ckpt_view_descendants = ckpt_view.descendants()
  for node_id, trackable in nodes_to_restore.items():
    # node_id does not have a corresponding Checkpoint value.
    if (node_id not in ckpt_view_descendants or
        ckpt_view._object_graph_proto.nodes[  # pylint: disable=protected-access
            node_id] is None):
      raise ValueError(
          f"The expected node_id: {node_id} to Trackable {trackable} to "
          "restore does not exist in the checkpoint.")
    # Trackable mapped to node_id to restore is empty.
    if trackable is None or not isinstance(trackable, base.Trackable):
      raise ValueError(
          f"Expecting a valid Trackable to node_id: {node_id} but got "
          f"trackable: {trackable}."
      )

  serialized_tensors = object_identity.ObjectIdentityDictionary()
  for node_id, current_trackable in nodes_to_restore.items():
    ckpt_contains_serialized_tensors = ckpt_view._object_graph_proto.nodes[  # pylint: disable=protected-access
        node_id].attributes
    node = ckpt_view._object_graph_proto.nodes[node_id]  # pylint: disable=protected-access
    trackable_has_serialize_to_tensor = saveable_object_util.trackable_has_serialize_to_tensor(
        current_trackable)
    if not trackable_has_serialize_to_tensor:
      if not node.attributes:
        if current_trackable._gather_saveables_for_checkpoint():  # pylint: disable=protected-access
          raise ValueError(
              f"Trackable {current_trackable} expects checkpointed values but "
              "checkpoint does not contain serialized tensors for node_id: "
              f"{node_id}.")
        else:
          continue
      object_names = object_identity.ObjectIdentityDictionary()
      object_names[current_trackable] = trackable_utils.extract_object_name(
          node.attributes[0].checkpoint_key)
      checkpoint_factory_map, _ = save_util_v1.get_checkpoint_factories_and_keys(
          object_names, None)
      saveable_objects = save_util_v1.generate_saveable_objects(
          checkpoint_factory_map)[0]
      if len(node.attributes) != len(saveable_objects):
        raise ValueError("Size for saveable_objects for Trackable: "
                         f"{len(saveable_objects)} did not match the size for "
                         "serialized_tensors for checkpoint: "
                         f"{len(node.attributes)}.")
      current_trackable = saveable_object_util.SaveableCompatibilityConverter(
          current_trackable, saveable_objects)

    serialized_tensors[
        current_trackable] = current_trackable._serialize_to_tensors()  # pylint: disable=protected-access
    trackable_expects_ckpted_value = bool(serialized_tensors[current_trackable])

    if trackable_expects_ckpted_value and not ckpt_contains_serialized_tensors:
      raise ValueError(
          f"Trackable {current_trackable} expects checkpointed values but "
          "checkpoint does not contain serialized tensors for node_id: "
          f"{node_id}.")

    if not trackable_expects_ckpted_value and ckpt_contains_serialized_tensors:
      raise ValueError(
          f"Trackable {current_trackable} does not expect checkpointed "
          "values but checkpoint contains serialized tensors: "
          f"{ckpt_contains_serialized_tensors} for node_id: {node_id}.")

    if len(node.attributes) != len(serialized_tensors[current_trackable]):
      raise ValueError("Size for serialized_tensors for Trackable: "
                       f"{len(serialized_tensors[current_trackable])} did not "
                       "match size for serialized_tensors for checkpoint: "
                       f"{len(node.attributes)}.")

    if not trackable_has_serialize_to_tensor:
      functional_saver.MultiDeviceSaver(serialized_tensors).restore(save_path)
    else:
      # Converts attribute.name to attribute.checkpoint_key since that's what
      # restore method is expecting. i.e., converts "a" to "/.ATTRIBUTES/a".
      serialized_tensors_renamed = object_identity.ObjectIdentityDictionary()
      serialized_tensors_renamed[current_trackable] = {}
      for attribute in node.attributes:
        name = attribute.name
        checkpoint_key = attribute.checkpoint_key
        serialized_tensors_renamed[current_trackable][
            checkpoint_key] = serialized_tensors[current_trackable][name]
      functional_saver.MultiDeviceSaver(serialized_tensors_renamed).restore(
          save_path)


def _queue_children_for_restoration(checkpoint_position, visit_queue):
  """Queues the restoration of trackable's children or defers them."""
  # pylint: disable=protected-access
  trackable = checkpoint_position.trackable
  for child in checkpoint_position.object_proto.children:
    child_position = checkpoint_position.create_child_position(child.node_id)
    local_object = trackable._lookup_dependency(child.local_name)
    child_proto = child_position.object_proto
    if local_object is None:
      # We don't yet have a dependency registered with this name. Save it
      # in case we do.
      if child_proto.HasField("has_checkpoint_values"):
        has_value = child_proto.has_checkpoint_values.value
      else:
        # If the field is not set, do a simple check to see if the dependency
        # has children and/or checkpointed values.
        has_value = bool(
            child_proto.children or child_proto.attributes or
            child_proto.slot_variables or
            child_proto.HasField("registered_saver"))
      if has_value:
        trackable._deferred_dependencies.setdefault(child.local_name,
                                                    []).append(child_position)
    else:
      if child_position.bind_object(trackable=local_object):
        # This object's correspondence is new, so dependencies need to be
        # visited. Delay doing it so that we get a breadth-first dependency
        # resolution order (shallowest paths first). The caller is responsible
        # for emptying visit_queue.
        visit_queue.append((child_position, local_object))


_DeferredSlotVariableRestoration = collections.namedtuple(
    "_DeferredSlotVariableRestoration", [
        "original_variable",
        "slot_variable_id",
        "slot_name",
    ])


def _queue_slot_variables(checkpoint_position, visit_queue):
  """Queues slot variables for restoration."""
  trackable = checkpoint_position.trackable
  checkpoint = checkpoint_position.checkpoint
  for deferred_slot_restoration in (checkpoint.deferred_slot_restorations.pop(
      checkpoint_position.proto_id, ())):
    slot_variable_position, slot_variable = (
        checkpoint_position.create_slot_variable_position(
            trackable, deferred_slot_restoration.original_variable,
            deferred_slot_restoration.slot_variable_id,
            deferred_slot_restoration.slot_name))
    if slot_variable_position is not None:
      visit_queue.append((slot_variable_position, slot_variable))
  for slot_restoration in checkpoint.slot_restorations.pop(
      checkpoint_position.proto_id, ()):
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

    # `optimizer_object` can be a `Checkpoint` when user only needs the
    # attributes the optimizer holds, such as `iterations`. In those cases,
    # it would not have the optimizer's `_create_or_restore_slot_variable`
    # method.
    elif hasattr(optimizer_object, "_create_or_restore_slot_variable"):
      slot_variable_position, slot_variable = (
          checkpoint_position.create_slot_variable_position(
              optimizer_object, trackable, slot_restoration.slot_variable_id,
              slot_restoration.slot_name))
      if slot_variable_position is not None:
        visit_queue.append((slot_variable_position, slot_variable))


def _extract_saveable_name(checkpoint_key):
  # Substring the checkpoint key to the end of the "{...}.ATTRIBUTES/"
  search_key = trackable_utils.OBJECT_ATTRIBUTES_NAME + "/"
  return checkpoint_key[:checkpoint_key.index(search_key) + len(search_key)]


def _get_saveable_from_factory(saveable_factories, serialized_tensor,
                               created_compat_names):
  """Returns the saveable generated from the factory method."""
  matched_factory = None

  # The `expected_factory_name` is used to find the right saveable factory,
  # while the `factory_input_name` is the value that is passed to the factory
  # method to instantiate the SaveableObject.
  expected_factory_name = serialized_tensor.name
  factory_input_name = serialized_tensor.checkpoint_key

  # Case 1: the name already exactly matches a key in saveable_factories.
  if expected_factory_name in saveable_factories:
    matched_factory = saveable_factories[expected_factory_name]

  # Case 2: (Forward compat) The serialized name is composed of
  # "factory_name" + "SUFFIX". Get the matching factory name.
  if matched_factory is None:

    for factory_name, factory in saveable_factories.items():
      if expected_factory_name.startswith(factory_name):
        if matched_factory is not None:
          # This condition is met in the extreme edge case where the object
          # returns two saveable factories with similar names. This is very
          # unlikely because there zero objects inside TensorFlow that use
          # more than one saveable factory.
          raise ValueError("Forward compatibility load error: Unable to load "
                           "checkpoint saved in future version of TensorFlow. "
                           "Please update your version of TensorFlow to the "
                           "version in which the checkpoint was saved.")

        matched_factory = factory
        factory_input_name = _extract_saveable_name(
            serialized_tensor.checkpoint_key) + factory_name
        created_compat_names.add(factory_name)

  if callable(matched_factory):
    return matched_factory(name=factory_input_name)
  return matched_factory
