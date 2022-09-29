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
"""Extracts tensors for checkpointing while updating a TrackableObjectGraph.

This is labelled "v1" because the methods here use SaveableObject, which will
soon be deprecated.
"""

import collections

from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.checkpoint import util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base
from tensorflow.python.trackable import python_state
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object as saveable_object_lib
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import object_identity

# Factory and related info used to build a SaveableObject that saves a Trackable
# to checkpoint.
_CheckpointFactoryData = collections.namedtuple(
    "_CheckpointFactoryData", ["factory", "name", "checkpoint_key"])


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
  unmapped_registered_savers = collections.defaultdict(dict)
  for trackable, object_name in object_names.items():
    # object_to_save is only used to retrieve the saving functionality. For keys
    # and other data, use the original `trackable`.
    object_to_save = util.get_mapped_trackable(trackable, object_map)

    saver_name = registration.get_registered_saver_name(object_to_save)
    if saver_name:
      # Add the original trackable instead of `object_to_save` to the returned
      # dict because the original is needed for writing the object proto.
      unmapped_registered_savers[saver_name][object_name] = trackable
    else:
      checkpoint_factory_map[trackable] = []
      for name, saveable_factory in (
          saveable_object_util.saveable_objects_from_trackable(
              object_to_save).items()):  # pylint: disable=protected-access
        # Retrieve the legacy saveable name (for compatibility purposes during
        # SaveableObject deprecation)

        key_suffix = saveable_compat.get_saveable_name(object_to_save) or name
        checkpoint_key = trackable_utils.checkpoint_key(object_name, key_suffix)

        if not saveable_compat.force_checkpoint_conversion_enabled():
          # Make sure the set the name as the legacy saveable name if there
          # is one (only when checkpoint conversion is diabled)
          name = key_suffix

        checkpoint_factory_map[trackable].append(
            _CheckpointFactoryData(
                factory=saveable_factory,
                name=name,
                checkpoint_key=checkpoint_key))
  return checkpoint_factory_map, unmapped_registered_savers


def _add_attributes_to_object_graph(trackable_objects, object_graph_proto,
                                    node_ids, object_names, object_map,
                                    call_with_mapped_captures, saveables_cache):
  """Create saveables/savers and corresponding protos in the object graph."""
  # The loop below creates TrackableObject protos in the TrackableObjectGraph,
  # which are filled in the `_add_attributes_to_object_graph_for_*` methods.
  for checkpoint_id, (trackable, unused_object_proto) in enumerate(
      zip(trackable_objects, object_graph_proto.nodes)):
    assert node_ids[trackable] == checkpoint_id

  checkpoint_factory_map, unmapped_registered_savers = (
      get_checkpoint_factories_and_keys(object_names, object_map))

  # Add attributes, which describe what values are saved in checkpoint for
  # this trackable.
  registered_savers = _add_attributes_to_object_graph_for_registered_savers(
      unmapped_registered_savers, object_graph_proto, node_ids, object_map)
  named_saveable_objects, feed_additions = (
      add_attributes_to_object_graph_for_saveable_objects(
          checkpoint_factory_map, object_graph_proto, node_ids, object_map,
          call_with_mapped_captures, saveables_cache))
  return named_saveable_objects, feed_additions, registered_savers


def _add_attributes_to_object_graph_for_registered_savers(
    unmapped_registered_savers, object_graph_proto, node_ids, object_map):
  """Fills the object graph proto with data about the registered savers."""
  registered_savers = collections.defaultdict(dict)
  for saver_name, trackables in unmapped_registered_savers.items():
    for object_name, trackable in trackables.items():
      object_proto = object_graph_proto.nodes[node_ids[trackable]]
      object_proto.registered_saver.name = saver_name
      object_proto.registered_saver.object_name = object_name

      object_to_save = util.get_mapped_trackable(trackable, object_map)
      registered_savers[saver_name][object_name] = object_to_save
  return registered_savers


def add_attributes_to_object_graph_for_saveable_objects(
    checkpoint_factory_map, object_graph_proto, node_ids, object_map,
    call_with_mapped_captures, saveables_cache):
  """Create SaveableObjects and corresponding SerializedTensor protos."""
  named_saveable_objects = []
  if saveables_cache is None:
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
    object_to_save = util.get_mapped_trackable(trackable, object_map)
    if saveables_cache is not None:
      cached_attributes = saveables_cache.setdefault(object_to_save, {})
    else:
      cached_attributes = None

    for factory_data in factory_data_list:
      name = factory_data.name
      key = factory_data.checkpoint_key
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
              name, key, saveable_factory, call_with_mapped_captures)
        else:
          maybe_saveable = saveable_factory
        if isinstance(maybe_saveable, saveable_object_lib.SaveableObject):
          saveables = (maybe_saveable,)
        else:
          saveables = tuple(
              saveable_object_util.saveable_objects_for_op(
                  op=maybe_saveable, name=key))
        for saveable in saveables:
          if key not in saveable.name:
            raise AssertionError(
                f"The object {trackable} produced a SaveableObject with name "
                f"'{saveable.name}' for attribute '{name}'. Expected a name"
                f" containing '{key}'.")
        if cached_attributes is not None:
          cached_attributes[name] = saveables

      if isinstance(object_to_save, python_state.PythonState):
        assert len(saveables) == 1
        saveable = saveables[0]

        if feed_additions is None:
          assert saveables_cache is None
          # If we're not caching saveables, then we're either executing
          # eagerly or building a static save/restore (e.g. for a
          # SavedModel). In either case, we should embed the current Python
          # state in the graph rather than relying on a feed dict.
          saveables = (saveable.freeze(),)
        else:
          feed_additions.update(saveable.feed_dict_additions())
      named_saveable_objects.extend(saveables)

      # Update the object proto.
      # For updated Trackables that override serialize_to_tensors, add an
      # attribute for each tensor that is serialized.
      # For Trackables that have SaveableObjects or a legacy saveable name,
      # add a single attribute to the proto.
      if (isinstance(saveables[0], saveable_object_util.TrackableSaveable) and
          (saveable_compat.force_checkpoint_conversion_enabled() or
           saveable_compat.get_saveable_name(object_to_save) is None)):
        for local_name, local_key in (
            saveables[0].get_proto_names_and_checkpoint_keys()):
          object_proto.attributes.add(
              name=local_name,
              checkpoint_key=local_key,
              full_name=util.get_full_name(object_to_save))
      else:
        object_proto.attributes.add(
            name=name,
            checkpoint_key=key,
            full_name=util.get_full_name(object_to_save))

  return named_saveable_objects, feed_additions


def _fill_object_graph_proto(graph_view,
                             trackable_objects,
                             node_ids,
                             slot_variables):
  """Name non-slot `Trackable`s and add them to `object_graph_proto`."""
  object_graph_proto = trackable_object_graph_pb2.TrackableObjectGraph()
  for checkpoint_id, trackable in enumerate(trackable_objects):
    assert node_ids[trackable] == checkpoint_id
    object_proto = object_graph_proto.nodes.add(
        slot_variables=slot_variables.get(trackable, ())
    )
    for child in graph_view.list_children(trackable):
      object_proto.children.add(
          node_id=node_ids[child.ref],
          local_name=child.name)
  return object_graph_proto


def serialize_gathered_objects(graph_view,
                               object_map=None,
                               call_with_mapped_captures=None,
                               saveables_cache=None):
  """Create SaveableObjects and protos for gathered objects."""
  trackable_objects, node_paths = graph_view.breadth_first_traversal()
  object_names = object_identity.ObjectIdentityDictionary()
  for obj, path in node_paths.items():
    object_names[obj] = trackable_utils.object_path_to_string(path)
  node_ids = object_identity.ObjectIdentityDictionary()
  for node_id, node in enumerate(trackable_objects):
    node_ids[node] = node_id
  slot_variables = util.serialize_slot_variables(
      trackable_objects=trackable_objects,
      node_ids=node_ids,
      object_names=object_names)
  object_graph_proto = _fill_object_graph_proto(
      graph_view=graph_view,
      trackable_objects=trackable_objects,
      node_ids=node_ids,
      slot_variables=slot_variables)
  named_saveable_objects, feed_additions, registered_savers = (
      _add_attributes_to_object_graph(
          trackable_objects=trackable_objects,
          object_graph_proto=object_graph_proto,
          node_ids=node_ids,
          object_names=object_names,
          object_map=object_map,
          call_with_mapped_captures=call_with_mapped_captures,
          saveables_cache=saveables_cache))
  # Gather all trackables that have checkpoint values or descendants with
  # checkpoint values, and add that info to the proto.
  util.add_checkpoint_values_check(object_graph_proto)
  return (named_saveable_objects, object_graph_proto, feed_additions,
          registered_savers)


def serialize_object_graph_with_registered_savers(graph_view, saveables_cache):
  """Determine checkpoint keys for variables and build a serialized graph."""
  return serialize_gathered_objects(graph_view, saveables_cache=saveables_cache)


def frozen_saveables_and_savers(graph_view,
                                object_map=None,
                                to_graph=None,
                                call_with_mapped_captures=None,
                                saveables_cache=None):
  """Generates SaveableObjects and registered savers in the frozen graph."""
  if to_graph:
    target_context = to_graph.as_default
  else:
    target_context = ops.NullContextmanager
  with target_context():
    named_saveable_objects, graph_proto, _, registered_savers = (
        serialize_gathered_objects(graph_view, object_map,
                                   call_with_mapped_captures, saveables_cache))
    with ops.device("/cpu:0"):
      object_graph_tensor = constant_op.constant(
          graph_proto.SerializeToString(), dtype=dtypes.string)
    named_saveable_objects.append(
        base.NoRestoreSaveable(
            tensor=object_graph_tensor, name=base.OBJECT_GRAPH_PROTO_KEY))
  return named_saveable_objects, registered_savers
