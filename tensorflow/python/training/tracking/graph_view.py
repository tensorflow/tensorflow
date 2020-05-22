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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import weakref

from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer as optimizer_v1
from tensorflow.python.training.saving import saveable_object as saveable_object_lib
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import tracking
from tensorflow.python.util import object_identity


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
      (_escape_local_name(trackable.name)
       for trackable in path_to_root))


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
      naming_scheme = _slot_variable_naming_for_optimizer(
          optimizer_path=object_names[trackable])
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
          if slot_variable._checkpoint_dependencies:  # pylint: disable=protected-access
            # TODO(allenl): Gather dependencies of slot variables.
            raise NotImplementedError(
                "Currently only variables with no dependencies can be saved as "
                "slot variables. File a feature request if this limitation "
                "bothers you.")
          if slot_variable in node_ids:
            raise NotImplementedError(
                "A slot variable was re-used as a dependency of a "
                "Trackable object. This is not currently allowed. File a "
                "feature request if this limitation bothers you.")
          checkpoint_name = naming_scheme(
              variable_path=object_names[original_variable],
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


class ObjectGraphView(object):
  """Gathers and serializes an object graph."""

  def __init__(self, root, saveables_cache=None):
    """Configure the graph view.

    Args:
      root: A `Trackable` object whose variables (including the variables
        of dependencies, recursively) should be saved. May be a weak reference.
      saveables_cache: A dictionary mapping `Trackable` objects ->
        attribute names -> SaveableObjects, used to avoid re-creating
        SaveableObjects when graph building.
    """
    self._root_ref = root
    self._saveables_cache = saveables_cache

  def list_dependencies(self, obj):
    # pylint: disable=protected-access
    obj._maybe_initialize_trackable()
    return obj._checkpoint_dependencies
    # pylint: enable=protected-access

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
    path_to_root = object_identity.ObjectIdentityDictionary()
    path_to_root[self.root] = ()
    while to_visit:
      current_trackable = to_visit.popleft()
      if isinstance(current_trackable, tracking.NotTrackable):
        raise NotImplementedError(
            ("The object %s does not support object-based saving. File a "
             "feature request if this limitation bothers you. In the meantime, "
             "you can remove the dependency on this object and save everything "
             "else.")
            % (current_trackable,))
      bfs_sorted.append(current_trackable)
      for name, dependency in self.list_dependencies(current_trackable):
        if dependency not in path_to_root:
          path_to_root[dependency] = (
              path_to_root[current_trackable] + (
                  base.TrackableReference(name, dependency),))
          to_visit.append(dependency)
    return bfs_sorted, path_to_root

  def _add_attributes_to_object_graph(
      self, trackable_objects, object_graph_proto, node_ids, object_names,
      object_map):
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
    for checkpoint_id, (trackable, object_proto) in enumerate(
        zip(trackable_objects, object_graph_proto.nodes)):
      assert node_ids[trackable] == checkpoint_id
      object_name = object_names[trackable]
      if object_map is None:
        object_to_save = trackable
      else:
        object_to_save = object_map.get(trackable, trackable)
      if self._saveables_cache is not None:
        cached_attributes = self._saveables_cache.setdefault(object_to_save, {})
      else:
        cached_attributes = None

      for name, saveable_factory in (
          object_to_save._gather_saveables_for_checkpoint().items()):  # pylint: disable=protected-access
        attribute = object_proto.attributes.add()
        attribute.name = name
        attribute.checkpoint_key = "%s/%s/%s" % (
            object_name, _OBJECT_ATTRIBUTES_NAME, _escape_local_name(name))
        if cached_attributes is None:
          saveables = None
        else:
          saveables = cached_attributes.get(name, None)
          if saveables is not None:
            for saveable in saveables:
              if attribute.checkpoint_key not in saveable.name:
                # The checkpoint key for this SaveableObject is different. We
                # need to re-create it.
                saveables = None
                del cached_attributes[name]
                break
        if saveables is None:
          if callable(saveable_factory):
            maybe_saveable = saveable_factory(name=attribute.checkpoint_key)
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
                op=maybe_saveable, name=attribute.checkpoint_key))
            for saveable in saveables:
              saveable.full_name = full_name
          for saveable in saveables:
            if attribute.checkpoint_key not in saveable.name:
              raise AssertionError(
                  ("The object %s produced a SaveableObject with name '%s' for "
                   "attribute '%s'. Expected a name containing '%s'.")
                  % (trackable, name, saveable.name,
                     attribute.checkpoint_key))
          if cached_attributes is not None:
            cached_attributes[name] = saveables

        optional_restore = None
        for saveable in saveables:
          if optional_restore is None:
            optional_restore = saveable.optional_restore
          else:
            optional_restore = optional_restore and saveable.optional_restore

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
                      ("The object %s tried to feed a value for the Tensor %s "
                       "when saving, but another object is already feeding a "
                       "value.")
                      % (trackable, new_feed_key))
              feed_additions.update(saveable_feed_dict)
          named_saveable_objects.append(saveable)
        if optional_restore is None:
          optional_restore = False
        attribute.optional_restore = optional_restore

    return named_saveable_objects, feed_additions

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
      for child in self.list_dependencies(trackable):
        child_proto = object_proto.children.add()
        child_proto.node_id = node_ids[child.ref]
        child_proto.local_name = child.name
    return object_graph_proto

  def _serialize_gathered_objects(self, trackable_objects, path_to_root,
                                  object_map=None):
    """Create SaveableObjects and protos for gathered objects."""
    object_names = object_identity.ObjectIdentityDictionary()
    for obj, path in path_to_root.items():
      object_names[obj] = _object_prefix_from_path(path)
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
    named_saveable_objects, feed_additions = (
        self._add_attributes_to_object_graph(
            trackable_objects=trackable_objects,
            object_graph_proto=object_graph_proto,
            node_ids=node_ids,
            object_names=object_names,
            object_map=object_map))
    return named_saveable_objects, object_graph_proto, feed_additions

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
    trackable_objects, path_to_root = self._breadth_first_traversal()
    return self._serialize_gathered_objects(
        trackable_objects, path_to_root)

  def frozen_saveable_objects(self, object_map=None, to_graph=None):
    """Creates SaveableObjects with the current object graph frozen."""
    trackable_objects, path_to_root = self._breadth_first_traversal()
    if to_graph:
      target_context = to_graph.as_default
    else:
      target_context = ops.NullContextmanager
    with target_context():
      named_saveable_objects, graph_proto, _ = self._serialize_gathered_objects(
          trackable_objects,
          path_to_root,
          object_map)
      with ops.device("/cpu:0"):
        object_graph_tensor = constant_op.constant(
            graph_proto.SerializeToString(), dtype=dtypes.string)
      named_saveable_objects.append(
          base.NoRestoreSaveable(
              tensor=object_graph_tensor,
              name=base.OBJECT_GRAPH_PROTO_KEY))
    return named_saveable_objects

  def objects_ids_and_slot_variables(self):
    """Traverse the object graph and list all accessible objects.

    Looks for `Trackable` objects which are dependencies of
    `root_trackable`. Includes slot variables only if the variable they are
    slotting for and the optimizer are dependencies of `root_trackable`
    (i.e. if they would be saved with a checkpoint).

    Returns:
      A tuple of (trackable objects, object -> node id, slot variables)
    """
    trackable_objects, path_to_root = self._breadth_first_traversal()
    object_names = object_identity.ObjectIdentityDictionary()
    for obj, path in path_to_root.items():
      object_names[obj] = _object_prefix_from_path(path)
    node_ids = object_identity.ObjectIdentityDictionary()
    for node_id, node in enumerate(trackable_objects):
      node_ids[node] = node_id
    slot_variables = _serialize_slot_variables(
        trackable_objects=trackable_objects,
        node_ids=node_ids,
        object_names=object_names)
    return trackable_objects, node_ids, slot_variables

  def list_objects(self):
    """Traverse the object graph and list all accessible objects."""
    trackable_objects, _, _ = self.objects_ids_and_slot_variables()
    return trackable_objects
