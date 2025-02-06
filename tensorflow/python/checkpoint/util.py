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
"""Utilities for extracting and writing checkpoint info`."""

from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.util import object_identity


def serialize_slot_variables(trackable_objects, node_ids, object_names):
  """Gather and name slot variables."""
  non_slot_objects = list(trackable_objects)
  slot_variables = object_identity.ObjectIdentityDictionary()
  for trackable in non_slot_objects:
        # TODO(b/110718070): Fix Keras imports.
        # Note: dir() is used rather than hasattr() here to avoid triggering
        # custom __getattr__ code, see b/152031870 for context.
    if "get_slot_names" in dir(trackable):
      slot_names = trackable.get_slot_names()
      for slot_name in slot_names:
        for original_variable_node_id, original_variable in enumerate(
            non_slot_objects):
          try:
            slot_variable = trackable.get_slot(original_variable, slot_name)
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
              trackable_object_graph_pb2.TrackableObjectGraph.TrackableObject
              .SlotVariableReference(
                  slot_name=slot_name,
                  original_variable_node_id=original_variable_node_id,
                  slot_variable_node_id=slot_variable_node_id))
          slot_variables.setdefault(trackable, []).append(slot_variable_proto)
  return slot_variables


def get_mapped_trackable(trackable, object_map):
  """Returns the mapped trackable if possible, otherwise returns trackable."""
  if object_map is None:
    return trackable
  else:
    return object_map.get(trackable, trackable)


def get_full_name(var):
  """Gets the full name of variable for name-based checkpoint compatibility."""
  # pylint: disable=protected-access
  if (not (isinstance(var, variables.Variable) or
           # Some objects do not subclass Variable but still act as one.
           resource_variable_ops.is_resource_variable(var))):
    return ""

  if getattr(var, "_save_slice_info", None) is not None:
    # Use getattr because `var._save_slice_info` may be set as `None`.
    return var._save_slice_info.full_name
  else:
    return var._shared_name
  # pylint: enable=protected-access


def add_checkpoint_values_check(object_graph_proto):
  """Determines which objects have checkpoint values and save this to the proto.

  Args:
    object_graph_proto: A `TrackableObjectGraph` proto.
  """
  # Trackable -> set of all trackables that depend on it (the "parents").
  # If a trackable has checkpoint values, then all of the parents can be
  # marked as having checkpoint values.
  parents = {}
  checkpointed_trackables = object_identity.ObjectIdentitySet()

  # First pass: build dictionary of parent objects and initial set of
  # checkpointed trackables.
  checkpointed_trackables = set()
  for node_id, object_proto in enumerate(object_graph_proto.nodes):
    if (object_proto.attributes or object_proto.slot_variables or
        object_proto.HasField("registered_saver")):
      checkpointed_trackables.add(node_id)
    for child_proto in object_proto.children:
      child = child_proto.node_id
      if child not in parents:
        parents[child] = set()
      parents[child].add(node_id)

  # Second pass: add all connected parents to set of checkpointed trackables.
  to_visit = set()
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

  for node_id, object_proto in enumerate(object_graph_proto.nodes):
    object_proto.has_checkpoint_values.value = bool(
        node_id in checkpointed_trackables)


def objects_ids_and_slot_variables_and_paths(graph_view,
                                             skip_slot_variables=False):
  """Traverse the object graph and list all accessible objects.

  Looks for `Trackable` objects which are dependencies of
  `root_trackable`. Includes slot variables only if the variable they are
  slotting for and the optimizer are dependencies of `root_trackable`
  (i.e. if they would be saved with a checkpoint).

  Args:
    graph_view: A GraphView object.
    skip_slot_variables: If True does not return trackables for slot variable.
      Default False.

  Returns:
    A tuple of (trackable objects, paths from root for each object,
                object -> node id, slot variables, object_names)
  """
  trackable_objects, node_paths = graph_view.breadth_first_traversal()
  object_names = object_identity.ObjectIdentityDictionary()
  for obj, path in node_paths.items():
    object_names[obj] = trackable_utils.object_path_to_string(path)
  node_ids = object_identity.ObjectIdentityDictionary()
  for node_id, node in enumerate(trackable_objects):
    node_ids[node] = node_id
  if skip_slot_variables:
    slot_variables = object_identity.ObjectIdentityDictionary()
  else:
    slot_variables = serialize_slot_variables(
        trackable_objects=trackable_objects,
        node_ids=node_ids,
        object_names=object_names,
    )
  return (trackable_objects, node_paths, node_ids, slot_variables, object_names)


def list_objects(graph_view, skip_slot_variables=False):
  """Traverse the object graph and list all accessible objects."""
  trackable_objects = objects_ids_and_slot_variables_and_paths(
      graph_view, skip_slot_variables
  )[0]
  return trackable_objects
