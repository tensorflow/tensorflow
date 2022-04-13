"""Manages a graph of Trackable objects."""
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
import collections
import copy
import weakref

from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import registration
from tensorflow.python.training import optimizer as optimizer_v1
from tensorflow.python.training.saving import saveable_object as saveable_object_lib
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import trackable_utils
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export


# Factory and related info used to build a SaveableObject that saves a Trackable
# to checkpoint.
_CheckpointFactoryData = collections.namedtuple(
    "_CheckpointFactoryData", ["factory", "name", "checkpoint_key"])


def _serialize_slot_variables(trackable_objects, node_ids, object_names):
  """Gather and name slot variables."""
  non_slot_objects = list(trackable_objects)
  slot_variables = object_identity.ObjectIdentityDictionary()
  for trackable in non_slot_objects:
    if (isinstance(trackable, optimizer_v1.Optimizer)
        # TODO(b/110718070): Fix Keras imports.
        # Note: dir() is used rather than hasattr() here to avoid triggering
        # custom __getattr__ code, see b/152031870 for context.
        or "_create_or_restore_slot_variable" in dir(trackable)):
      slot_names = trackable.get_slot_names()
      for slot_name in slot_names:
        for original_variable_node_id, original_variable in enumerate(
            non_slot_objects):
          try:
            slot_variable = trackable.get_slot(
                original_variable, slot_name)
          except (AttributeError, KeyError):
            slot_variable = None
          if slot_variable is None:
            continue
          slot_variable._maybe_initialize_trackable()  # pylint: disable=protected-access
          if slot_variable._trackable_children():  # pylint: disable=protected-access
            # TODO(allenl): Gather dependencies of slot variables.
            raise NotImplementedError(
                "Currently only variables with no dependencies can be saved as "
                "slot variables. File a feature request if this limitation "
                "bothers you.")
          if slot_variable in node_ids:
            raise NotImplementedError(
                "A slot variable was re-used as a dependency of a Trackable "
                f"object: {slot_variable}. This is not currently allowed. "
                "File a feature request if this limitation bothers you.")
          checkpoint_name = trackable_utils.slot_variable_key(
              variable_path=object_names[original_variable],
              optimizer_path=object_names[trackable],
              slot_name=slot_name)
          object_names[slot_variable] = checkpoint_name
          slot_variable_node_id = len(trackable_objects)
          node_ids[slot_variable] = slot_variable_node_id
          trackable_objects.append(slot_variable)
          slot_variable_proto = (
              trackable_object_graph_pb2.TrackableObjectGraph
              .TrackableObject.SlotVariableReference(
                  slot_name=slot_name,
                  original_variable_node_id=original_variable_node_id,
                  slot_variable_node_id=slot_variable_node_id))
          slot_variables.setdefault(trackable, []).append(
              slot_variable_proto)
  return slot_variables


def _get_mapped_trackable(trackable, object_map):
  """Returns the mapped trackable if possible, otherwise returns trackable."""
  if object_map is None:
    return trackable
  else:
    return object_map.get(trackable, trackable)


def get_checkpoint_factories_and_keys(object_names, object_map=None):
  """Gets a map of saveable factories and corresponding checkpoint keys.

  Args:
    object_names: a dictionary that maps `Trackable` objects to auto-generated
      string names.
    object_map: a dictionary mapping `Trackable` to copied `Trackable` objects.
      The copied objects are generated from `Trackable._map_resources()` which
      copies the object into another graph. Generally only resource objects
      (e.g. Variables, Tables) will be in this map.
  Returns:
    A tuple of (
      Dictionary mapping trackable -> list of _CheckpointFactoryData,
      Dictionary mapping registered saver name -> {object name -> trackable})
  """
  checkpoint_factory_map = object_identity.ObjectIdentityDictionary()
  registered_savers = collections.defaultdict(dict)
  for trackable, object_name in object_names.items():
    # object_to_save is only used to retrieve the saving functionality. For keys
    # and other data, use the original `trackable`.
    object_to_save = _get_mapped_trackable(trackable, object_map)

    saver_name = registration.get_registered_saver_name(object_to_save)
    if saver_name:
      registered_savers[saver_name][object_name] = trackable
    else:
      checkpoint_factory_map[trackable] = []
      for name, saveable_factory in (
          saveable_object_util.saveable_objects_from_trackable(object_to_save)
          .items()):  # pylint: disable=protected-access
        checkpoint_key = trackable_utils.checkpoint_key(object_name, name)
        checkpoint_factory_map[trackable].append(_CheckpointFactoryData(
            factory=saveable_factory,
            name=name,
            checkpoint_key=checkpoint_key))
  return checkpoint_factory_map, registered_savers


def _add_attributes_to_object_graph_for_registered_savers(
    registered_savers, object_graph_proto, node_ids):
  """Fills the object graph proto with data about the registered savers."""
  for saver_name, trackables in registered_savers.items():
    for object_name, trackable in trackables.items():
      object_proto = object_graph_proto.nodes[node_ids[trackable]]
      object_proto.registered_saver.name = saver_name
      object_proto.registered_saver.object_name = object_name


@tf_export("__internal__.tracking.ObjectGraphView", v1=[])
class ObjectGraphView(object):
  """Gathers and serializes an object graph."""

  def __init__(self, root, saveables_cache=None, attached_dependencies=None):
    """Configure the graph view.

    Args:
      root: A `Trackable` object whose variables (including the variables
        of dependencies, recursively) should be saved. May be a weak reference.
      saveables_cache: A dictionary mapping `Trackable` objects ->
        attribute names -> SaveableObjects, used to avoid re-creating
        SaveableObjects when graph building.
      attached_dependencies: List of dependencies to attach to the root object.
        Used when saving a Checkpoint with a defined root object. To avoid
        reference cycles, this should use the WeakTrackableReference class.
    """
    # ObjectGraphView should never contain a strong reference to root, since it
    # may result in a cycle:
    #   root -> deferred dependencies -> CheckpointPosition
    #   -> CheckpointRestoreCoordinator -> ObjectGraphView -> root
    self._root_ref = (root if isinstance(root, weakref.ref)
                      else weakref.ref(root))
    self._saveables_cache = saveables_cache
    self._attached_dependencies = attached_dependencies

  def __deepcopy__(self, memo):
    # By default, weak references are not copied, which leads to surprising
    # deepcopy behavior. To fix, we first we copy the object itself, then we
    # make a weak reference to the copy.
    strong_root = self._root_ref()
    if strong_root is not None:
      strong_copy = copy.deepcopy(strong_root, memo)
      memo[id(self._root_ref)] = weakref.ref(strong_copy)
    # super() does not have a __deepcopy__, so we need to re-implement it
    copied = super().__new__(type(self))
    memo[id(self)] = copied
    for key, value in vars(self).items():
      setattr(copied, key, copy.deepcopy(value, memo))
    return copied

  def list_children(self, obj, save_type=base.SaveType.CHECKPOINT, **kwargs):
    """Returns all child trackables attached to obj.

    Args:
      obj: A `Trackable` object.
      save_type: A string, can be 'savedmodel' or 'checkpoint'.
      **kwargs: kwargs to use when retrieving the object's children.

    Returns:
      List of all children attached to the object.
    """
    # pylint: disable=protected-access
    obj._maybe_initialize_trackable()
    children = [base.TrackableReference(name, ref) for name, ref
                in obj._trackable_children(save_type, **kwargs).items()]
    # pylint: enable=protected-access

    # GraphView objects may define children of the root object that are not
    # actually attached, e.g. a Checkpoint object's save_counter.
    if obj is self.root and self._attached_dependencies:
      children.extend(self._attached_dependencies)
    return children

  @property
  def saveables_cache(self):
    """Maps Trackable objects -> attribute names -> list(SaveableObjects).

    Used to avoid re-creating SaveableObjects when graph building. None when
    executing eagerly.

    Returns:
      The cache (an object-identity dictionary), or None if caching is disabled.
    """
    return self._saveables_cache

  @property
  def attached_dependencies(self):
    """Returns list of dependencies that should be saved in the checkpoint.

    These dependencies are not tracked by root, but are in the checkpoint.
    This is defined when the user creates a Checkpoint with both root and kwargs
    set.

    Returns:
      A list of TrackableReferences.
    """
    return self._attached_dependencies

  @property
  def root(self):
    if isinstance(self._root_ref, weakref.ref):
      derefed = self._root_ref()
      assert derefed is not None
      return derefed
    else:
      return self._root_ref

  def _breadth_first_traversal(self):
    """Find shortest paths to all dependencies of self.root."""
    bfs_sorted = []
    to_visit = collections.deque([self.root])
    node_paths = object_identity.ObjectIdentityDictionary()
    node_paths[self.root] = ()
    while to_visit:
      current_trackable = to_visit.popleft()
      bfs_sorted.append(current_trackable)
      for name, dependency in self.list_children(current_trackable):
        if dependency not in node_paths:
          node_paths[dependency] = (
              node_paths[current_trackable] + (
                  base.TrackableReference(name, dependency),))
          to_visit.append(dependency)
    return bfs_sorted, node_paths

  def _add_attributes_to_object_graph(
      self, trackable_objects, object_graph_proto, node_ids, object_names,
      object_map, call_with_mapped_captures):
    """Create saveables/savers and corresponding protos in the object graph."""
    # The loop below creates TrackableObject protos in the TrackableObjectGraph,
    # which are filled in the `_add_attributes_to_object_graph_for_*` methods.
    for checkpoint_id, (trackable, unused_object_proto) in enumerate(
        zip(trackable_objects, object_graph_proto.nodes)):
      assert node_ids[trackable] == checkpoint_id

    checkpoint_factory_map, registered_savers = (
        get_checkpoint_factories_and_keys(object_names, object_map))

    # Add attributes, which describe what values are saved in checkpoint for
    # this trackable.
    _add_attributes_to_object_graph_for_registered_savers(
        registered_savers, object_graph_proto, node_ids)
    named_saveable_objects, feed_additions = (
        self._add_attributes_to_object_graph_for_saveable_objects(
            checkpoint_factory_map, object_graph_proto, node_ids, object_map,
            call_with_mapped_captures))
    return named_saveable_objects, feed_additions, registered_savers

  def _add_attributes_to_object_graph_for_saveable_objects(
      self, checkpoint_factory_map, object_graph_proto, node_ids, object_map,
      call_with_mapped_captures):
    """Create SaveableObjects and corresponding SerializedTensor protos."""
    named_saveable_objects = []
    if self._saveables_cache is None:
      # No SaveableObject caching. Either we're executing eagerly, or building a
      # static save which is specialized to the current Python state.
      feed_additions = None
    else:
      # If we are caching SaveableObjects, we need to build up a feed_dict with
      # functions computing volatile Python state to be saved with the
      # checkpoint.
      feed_additions = {}
    for trackable, factory_data_list in checkpoint_factory_map.items():
      object_proto = object_graph_proto.nodes[node_ids[trackable]]
      if self._saveables_cache is not None:
        object_to_save = _get_mapped_trackable(trackable, object_map)
        cached_attributes = self._saveables_cache.setdefault(object_to_save, {})
      else:
        cached_attributes = None

      for factory_data in factory_data_list:
        attribute = object_proto.attributes.add()
        attribute.name = name = factory_data.name
        attribute.checkpoint_key = key = factory_data.checkpoint_key
        saveable_factory = factory_data.factory

        # See if we can skip saving this checkpoint key.
        saveables = cached_attributes.get(name) if cached_attributes else None
        if saveables is not None:
          for saveable in saveables:
            if key not in saveable.name:
              # The checkpoint key for this SaveableObject is different. We
              # need to re-create it.
              saveables = None
              del cached_attributes[name]
              break

        if saveables is None:
          if callable(saveable_factory):
            maybe_saveable = saveable_object_util.create_saveable_object(
                saveable_factory, key, call_with_mapped_captures)
          else:
            maybe_saveable = saveable_factory
          if isinstance(maybe_saveable, saveable_object_lib.SaveableObject):
            saveables = (maybe_saveable,)
          else:
            # Figure out the name-based Saver's name for this variable. If it's
            # already a SaveableObject we'd just get the checkpoint key back, so
            # we leave full_name blank.
            saver_dict = saveable_object_util.op_list_to_dict(
                [maybe_saveable], convert_variable_to_tensor=False)
            full_name, = saver_dict.keys()
            saveables = tuple(saveable_object_util.saveable_objects_for_op(
                op=maybe_saveable, name=key))
            for saveable in saveables:
              saveable.full_name = full_name
          for saveable in saveables:
            if key not in saveable.name:
              raise AssertionError(
                  f"The object {trackable} produced a SaveableObject with name "
                  f"'{saveable.name}' for attribute '{name}'. Expected a name"
                  f" containing '{key}'.")
          if cached_attributes is not None:
            cached_attributes[name] = saveables

        for saveable in saveables:
          if hasattr(saveable, "full_name"):
            attribute.full_name = saveable.full_name
          if isinstance(saveable, base.PythonStateSaveable):
            if feed_additions is None:
              assert self._saveables_cache is None
              # If we're not caching saveables, then we're either executing
              # eagerly or building a static save/restore (e.g. for a
              # SavedModel). In either case, we should embed the current Python
              # state in the graph rather than relying on a feed dict.
              saveable = saveable.freeze()
            else:
              saveable_feed_dict = saveable.feed_dict_additions()
              for new_feed_key in saveable_feed_dict.keys():
                if new_feed_key in feed_additions:
                  raise AssertionError(
                      f"The object {trackable} tried to feed a value for the "
                      f"Tensor {new_feed_key} when saving, but another object "
                      "is already feeding a value.")
              feed_additions.update(saveable_feed_dict)
          named_saveable_objects.append(saveable)

    return named_saveable_objects, feed_additions

  def _add_checkpoint_values_check(self, trackable_objects, object_graph_proto):
    """Determines which objects have checkpoint values and saves to the proto.

    Args:
      trackable_objects: A list of all trackable objects.
      object_graph_proto: A `TrackableObjectGraph` proto.
    """
    # Trackable -> set of all trackables that depend on it (the "parents").
    # If a trackable has checkpoint values, then all of the parents can be
    # marked as having checkpoint values.
    parents = object_identity.ObjectIdentityDictionary()
    checkpointed_trackables = object_identity.ObjectIdentitySet()

    # First pass: build dictionary of parent objects and initial set of
    # checkpointed trackables.
    for trackable, object_proto in zip(trackable_objects,
                                       object_graph_proto.nodes):
      if (object_proto.attributes or object_proto.slot_variables or
          object_proto.HasField("registered_saver")):
        checkpointed_trackables.add(trackable)
      for child_proto in object_proto.children:
        child = trackable_objects[child_proto.node_id]
        if child not in parents:
          parents[child] = object_identity.ObjectIdentitySet()
        parents[child].add(trackable)

    # Second pass: add all connected parents to set of checkpointed trackables.
    to_visit = object_identity.ObjectIdentitySet()
    to_visit.update(checkpointed_trackables)

    while to_visit:
      trackable = to_visit.pop()
      if trackable not in parents:
        # Some trackables may not have parents (e.g. slot variables).
        continue
      current_parents = parents.pop(trackable)
      checkpointed_trackables.update(current_parents)
      for parent in current_parents:
        if parent in parents:
          to_visit.add(parent)

    for node_id, trackable in enumerate(trackable_objects):
      object_graph_proto.nodes[node_id].has_checkpoint_values.value = bool(
          trackable in checkpointed_trackables)

  def _fill_object_graph_proto(self, trackable_objects,
                               node_ids,
                               slot_variables,
                               object_graph_proto=None):
    """Name non-slot `Trackable`s and add them to `object_graph_proto`."""
    if object_graph_proto is None:
      object_graph_proto = (
          trackable_object_graph_pb2.TrackableObjectGraph())
    for checkpoint_id, trackable in enumerate(trackable_objects):
      assert node_ids[trackable] == checkpoint_id
      object_proto = object_graph_proto.nodes.add()
      object_proto.slot_variables.extend(slot_variables.get(trackable, ()))
      for child in self.list_children(trackable):
        child_proto = object_proto.children.add()
        child_proto.node_id = node_ids[child.ref]
        child_proto.local_name = child.name
    return object_graph_proto

  def _serialize_gathered_objects(self, trackable_objects, node_paths,
                                  object_map=None,
                                  call_with_mapped_captures=None):
    """Create SaveableObjects and protos for gathered objects."""
    object_names = object_identity.ObjectIdentityDictionary()
    for obj, path in node_paths.items():
      object_names[obj] = trackable_utils.object_path_to_string(path)
    node_ids = object_identity.ObjectIdentityDictionary()
    for node_id, node in enumerate(trackable_objects):
      node_ids[node] = node_id
    slot_variables = _serialize_slot_variables(
        trackable_objects=trackable_objects,
        node_ids=node_ids,
        object_names=object_names)
    object_graph_proto = self._fill_object_graph_proto(
        trackable_objects=trackable_objects,
        node_ids=node_ids,
        slot_variables=slot_variables)
    named_saveable_objects, feed_additions, registered_savers = (
        self._add_attributes_to_object_graph(
            trackable_objects=trackable_objects,
            object_graph_proto=object_graph_proto,
            node_ids=node_ids,
            object_names=object_names,
            object_map=object_map,
            call_with_mapped_captures=call_with_mapped_captures))
    # Gather all trackables that have checkpoint values or descendants with
    # checkpoint values, and add that info to the proto.
    self._add_checkpoint_values_check(trackable_objects, object_graph_proto)
    return (named_saveable_objects, object_graph_proto, feed_additions,
            registered_savers)

  def serialize_object_graph(self):
    """Determine checkpoint keys for variables and build a serialized graph.

    Non-slot variables are keyed based on a shortest path from the root saveable
    to the object which owns the variable (i.e. the one which called
    `Trackable._add_variable` to create it).

    Slot variables are keyed based on a shortest path to the variable being
    slotted for, a shortest path to their optimizer, and the slot name.

    Returns:
      A tuple of (named_variables, object_graph_proto, feed_additions):
        named_variables: A dictionary mapping names to variable objects.
        object_graph_proto: A TrackableObjectGraph protocol buffer
          containing the serialized object graph and variable references.
        feed_additions: A dictionary mapping from Tensors to values which should
          be fed when saving.

    Raises:
      ValueError: If there are invalid characters in an optimizer's slot names.
    """
    named_saveable_objects, object_graph_proto, feed_additions, _ = (
        self.serialize_object_graph_with_registered_savers())
    return named_saveable_objects, object_graph_proto, feed_additions

  def serialize_object_graph_with_registered_savers(self):
    """Determine checkpoint keys for variables and build a serialized graph."""
    trackable_objects, node_paths = self._breadth_first_traversal()
    return self._serialize_gathered_objects(
        trackable_objects, node_paths)

  def frozen_saveable_objects(self, object_map=None, to_graph=None,
                              call_with_mapped_captures=None):
    """Creates SaveableObjects with the current object graph frozen."""
    return self.frozen_saveables_and_savers(object_map, to_graph,
                                            call_with_mapped_captures)[0]

  def frozen_saveables_and_savers(self, object_map=None, to_graph=None,
                                  call_with_mapped_captures=None):
    """Generates SaveableObjects and registered savers in the frozen graph."""
    trackable_objects, node_paths = self._breadth_first_traversal()
    if to_graph:
      target_context = to_graph.as_default
    else:
      target_context = ops.NullContextmanager
    with target_context():
      named_saveable_objects, graph_proto, _, registered_savers = (
          self._serialize_gathered_objects(trackable_objects,
                                           node_paths,
                                           object_map,
                                           call_with_mapped_captures))
      with ops.device("/cpu:0"):
        object_graph_tensor = constant_op.constant(
            graph_proto.SerializeToString(), dtype=dtypes.string)
      named_saveable_objects.append(
          base.NoRestoreSaveable(
              tensor=object_graph_tensor,
              name=base.OBJECT_GRAPH_PROTO_KEY))
    return named_saveable_objects, registered_savers

  def objects_ids_and_slot_variables_and_paths(self):
    """Traverse the object graph and list all accessible objects.

    Looks for `Trackable` objects which are dependencies of
    `root_trackable`. Includes slot variables only if the variable they are
    slotting for and the optimizer are dependencies of `root_trackable`
    (i.e. if they would be saved with a checkpoint).

    Returns:
      A tuple of (trackable objects, paths from root for each object,
                  object -> node id, slot variables, object_names)
    """
    trackable_objects, node_paths = self._breadth_first_traversal()
    object_names = object_identity.ObjectIdentityDictionary()
    for obj, path in node_paths.items():
      object_names[obj] = trackable_utils.object_path_to_string(path)
    node_ids = object_identity.ObjectIdentityDictionary()
    for node_id, node in enumerate(trackable_objects):
      node_ids[node] = node_id
    slot_variables = _serialize_slot_variables(
        trackable_objects=trackable_objects,
        node_ids=node_ids,
        object_names=object_names)
    return (trackable_objects, node_paths, node_ids, slot_variables,
            object_names)

  def objects_ids_and_slot_variables(self):
    trackable_objects, _, node_ids, slot_variables, _ = (
        self.objects_ids_and_slot_variables_and_paths())
    return trackable_objects, node_ids, slot_variables

  def list_objects(self):
    """Traverse the object graph and list all accessible objects."""
    trackable_objects, _, _ = self.objects_ids_and_slot_variables()
    return trackable_objects
