"""Manages a Checkpoint View."""
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


from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import trackable_view
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base
from tensorflow.python.training import py_checkpoint_reader


class CheckpointView(object):
  """Gathers and serializes a checkpoint view."""

  def __init__(self, checkpoint_path):
    """Configure the trackable view.

    Args:
      checkpoint_path: The path to the checkpoint.

    Raises:
      ValueError: If an object graph was not found in the checkpoint.
    """
    reader = py_checkpoint_reader.NewCheckpointReader(checkpoint_path)
    try:
      object_graph_string = reader.get_tensor(base.OBJECT_GRAPH_PROTO_KEY)
    except errors_impl.NotFoundError as not_found_error:
      raise ValueError(
          f"The specified checkpoint \"{checkpoint_path}\" does not appear to be "
          "object-based (saved with TF2) since it is missing the key "
          f"\"{base.OBJECT_GRAPH_PROTO_KEY}\". Likely it was created with the "
          "TF1 name-based saver and does not contain an object dependency graph."
      ) from not_found_error
    object_graph_proto = (trackable_object_graph_pb2.TrackableObjectGraph())
    object_graph_proto.ParseFromString(object_graph_string)
    self._object_graph_proto = object_graph_proto

  def children(self, node_id):
    """Returns all child trackables attached to obj.

    Args:
      node_id: Id of the node to return its children.

    Returns:
      Dictionary of all children attached to the object with name to node_id.
    """
    return {
        child.local_name: child.node_id
        for child in self._object_graph_proto.nodes[node_id].children
    }

  def descendants(self):
    """Returns a list of all node_ids from ObjectGraphProto."""
    all_nodes = []
    to_visit = collections.deque([0])
    all_nodes.append(0)
    while to_visit:
      node_id = to_visit.popleft()
      obj = self._object_graph_proto.nodes[node_id]
      for child in obj.children:
        if child.node_id not in all_nodes:
          all_nodes.append(child.node_id)
          to_visit.append(child.node_id)
    return all_nodes

  def match(self, trackable_object):
    """Returns all matching trackables between CheckpointView and Trackable.

    Args:
      trackable_object: `Trackable` root.

    Returns:
      Dictionary containing all overlapping trackables that maps `node_id` to
      `Trackable`.
    """
    if not isinstance(trackable_object, base.Trackable):
      raise ValueError(f"Expected a Trackable, got {trackable_object} of type "
                       "{type(trackable_object)}.")

    overlapping_nodes = {}
    # Root node is always matched.
    overlapping_nodes[0] = trackable_object

    # Queue of tuples of node_id and trackable.
    to_visit = collections.deque([(0, trackable_object)])
    visited = set()
    view = trackable_view.TrackableView(trackable_object)
    while to_visit:
      current_node_id, current_trackable = to_visit.popleft()
      trackable_children = view.children(current_trackable)
      for child_name, child_node_id in self.children(current_node_id).items():
        if child_node_id in visited or child_node_id == 0:
          continue
        if child_name in trackable_children:
          current_assignment = overlapping_nodes.get(child_node_id)
          if current_assignment is None:
            overlapping_nodes[child_node_id] = trackable_children[child_name]
            to_visit.append((child_node_id, trackable_children[child_name]))
          else:
            # The object was already mapped for this checkpoint load, which
            # means we don't need to do anything besides check that the mapping
            # is consistent (if the dependency DAG is not a tree then there are
            # multiple paths to the same object).
            if current_assignment is not trackable_children[child_name]:
              logging.warning(
                  "Inconsistent references when matching the checkpoint into "
                  "this object graph. The referenced objects are: "
                  f"({current_assignment} and "
                  f"{trackable_children[child_name]}).")
      visited.add(current_node_id)
    return overlapping_nodes
