# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Utility methods for the trackable dependencies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


def pretty_print_node_path(path):
  if not path:
    return "root object"
  else:
    return "root." + ".".join([p.name for p in path])


class CyclicDependencyError(Exception):

  def __init__(self, leftover_dependency_map):
    """Creates a CyclicDependencyException."""
    # Leftover edges that were not able to be topologically sorted.
    self.leftover_dependency_map = leftover_dependency_map
    super(CyclicDependencyError, self).__init__()


def order_by_dependency(dependency_map):
  """Topologically sorts the keys of a map so that dependencies appear first.

  Uses Kahn's algorithm:
  https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm

  Args:
    dependency_map: a dict mapping values to a list of dependencies (other keys
      in the map). All keys and dependencies must be hashable types.

  Returns:
    A sorted array of keys from dependency_map.

  Raises:
    CyclicDependencyError: if there is a cycle in the graph.
    ValueError: If there are values in the dependency map that are not keys in
      the map.
  """
  # Maps trackables -> trackables that depend on them. These are the edges used
  # in Kahn's algorithm.
  reverse_dependency_map = collections.defaultdict(set)
  for x, deps in dependency_map.items():
    for dep in deps:
      reverse_dependency_map[dep].add(x)

  # Validate that all values in the dependency map are also keys.
  unknown_keys = reverse_dependency_map.keys() - dependency_map.keys()
  if unknown_keys:
    raise ValueError("Found values in the dependency map which are not keys: "
                     f"{unknown_keys}")

  # Generate the list sorted by objects without dependencies -> dependencies.
  # The returned list will reverse this.
  reversed_dependency_arr = []

  # Prefill `to_visit` with all nodes that do not have other objects depending
  # on them.
  to_visit = [x for x in dependency_map if x not in reverse_dependency_map]

  while to_visit:
    x = to_visit.pop(0)
    reversed_dependency_arr.append(x)
    for dep in set(dependency_map[x]):
      edges = reverse_dependency_map[dep]
      edges.remove(x)
      if not edges:
        to_visit.append(dep)
        reverse_dependency_map.pop(dep)

  if reverse_dependency_map:
    leftover_dependency_map = collections.defaultdict(list)
    for dep, xs in reverse_dependency_map.items():
      for x in xs:
        leftover_dependency_map[x].append(dep)
    raise CyclicDependencyError(leftover_dependency_map)

  return reversed(reversed_dependency_arr)


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
OBJECT_ATTRIBUTES_NAME = _ESCAPE_CHAR + "ATTRIBUTES"

# A constant string that is used to reference the save and restore functions of
#  Trackable objects that define `_serialize_to_tensors` and
# `_restore_from_tensors`. This is written as the key in the
# `SavedObject.saveable_objects<string, SaveableObject>` map in the SavedModel.
SERIALIZE_TO_TENSORS_NAME = _ESCAPE_CHAR + "TENSORS"


def escape_local_name(name):
  # We need to support slashes in local names for compatibility, since this
  # naming scheme is being patched in to things like Layer.add_variable where
  # slashes were previously accepted. We also want to use slashes to indicate
  # edges traversed to reach the variable, so we escape forward slashes in
  # names.
  return (name.replace(_ESCAPE_CHAR, _ESCAPE_CHAR + _ESCAPE_CHAR).replace(
      r"/", _ESCAPE_CHAR + "S"))


def object_path_to_string(node_path_arr):
  """Converts a list of nodes to a string."""
  return "/".join(
      (escape_local_name(trackable.name) for trackable in node_path_arr))


def checkpoint_key(object_path, local_name):
  """Returns the checkpoint key for a local attribute of an object."""
  key_suffix = escape_local_name(local_name)
  if local_name == SERIALIZE_TO_TENSORS_NAME:
    # In the case that Trackable uses the _serialize_to_tensor API for defining
    # tensors to save to the checkpoint, the suffix should be the key(s)
    # returned by `_serialize_to_tensor`. The suffix used here is empty.
    key_suffix = ""

  return f"{object_path}/{OBJECT_ATTRIBUTES_NAME}/{key_suffix}"


def slot_variable_key(variable_path, optimizer_path, slot_name):
  """Returns checkpoint key for a slot variable."""
  # Name slot variables:
  #
  #   <variable name>/<_OPTIMIZER_SLOTS_NAME>/<optimizer path>/<slot name>
  #
  # where <variable name> is exactly the checkpoint name used for the original
  # variable, including the path from the checkpoint root and the local name in
  # the object which owns it. Note that we only save slot variables if the
  # variable it's slotting for is also being saved.

  return (f"{variable_path}/{_OPTIMIZER_SLOTS_NAME}/{optimizer_path}/"
          f"{escape_local_name(slot_name)}")
