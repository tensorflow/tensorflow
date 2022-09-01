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
from tensorflow.python.util.tf_export import tf_export


@tf_export("train.CheckpointView", v1=[])
class CheckpointView(object):
  """Gathers and serializes a checkpoint view.

  This is for loading specific portions of a module from a
  checkpoint, and be able to compare two modules by matching components.

  Example usage:

  >>> class SimpleModule(tf.Module):
  ...   def __init__(self, name=None):
  ...     super().__init__(name=name)
  ...     self.a_var = tf.Variable(5.0)
  ...     self.b_var = tf.Variable(4.0)
  ...     self.vars = [tf.Variable(1.0), tf.Variable(2.0)]

  >>> root = SimpleModule(name="root")
  >>> root.leaf = SimpleModule(name="leaf")
  >>> ckpt = tf.train.Checkpoint(root)
  >>> save_path = ckpt.save('/tmp/tf_ckpts')
  >>> checkpoint_view = tf.train.CheckpointView(save_path)

  Pass `node_id=0` to `tf.train.CheckpointView.children()` to get the dictionary
  of all children directly linked to the checkpoint root.

  >>> for name, node_id in checkpoint_view.children(0).items():
  ...   print(f"- name: '{name}', node_id: {node_id}")
  - name: 'a_var', node_id: 1
  - name: 'b_var', node_id: 2
  - name: 'vars', node_id: 3
  - name: 'leaf', node_id: 4
  - name: 'root', node_id: 0
  - name: 'save_counter', node_id: 5

  """

  def __init__(self, save_path):
    """Configure the checkpoint view.

    Args:
      save_path: The path to the checkpoint.

    Raises:
      ValueError: If the save_path does not lead to a TF2 checkpoint.
    """

    reader = py_checkpoint_reader.NewCheckpointReader(save_path)
    try:
      object_graph_string = reader.get_tensor(base.OBJECT_GRAPH_PROTO_KEY)
    except errors_impl.NotFoundError as not_found_error:
      raise ValueError(
          f"The specified checkpoint \"{save_path}\" does not appear to be "
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
    """Returns a list of trackables by node_id attached to obj."""

    return list(self._descendants_with_paths().keys())

  def _descendants_with_paths(self):
    """Returns a dict of descendants by node_id and paths to node.

    The names returned by this private method are subject to change.
    """

    all_nodes_with_paths = {}
    to_visit = collections.deque([0])
    # node_id:0 will always be "root".
    all_nodes_with_paths[0] = "root"
    path = all_nodes_with_paths.get(0)
    while to_visit:
      node_id = to_visit.popleft()
      obj = self._object_graph_proto.nodes[node_id]
      for child in obj.children:
        if child.node_id == 0 or child.node_id in all_nodes_with_paths.keys():
          continue
        path = all_nodes_with_paths.get(node_id)
        if child.node_id not in all_nodes_with_paths.keys():
          to_visit.append(child.node_id)
        all_nodes_with_paths[child.node_id] = path + "." + child.local_name
    return all_nodes_with_paths

  def match(self, obj):
    """Returns all matching trackables between CheckpointView and Trackable.

    Matching trackables represents trackables with the same name and position in
    graph.

    Args:
      obj: `Trackable` root.

    Returns:
      Dictionary containing all overlapping trackables that maps `node_id` to
      `Trackable`.

    Example usage:

    >>> class SimpleModule(tf.Module):
    ...   def __init__(self, name=None):
    ...     super().__init__(name=name)
    ...     self.a_var = tf.Variable(5.0)
    ...     self.b_var = tf.Variable(4.0)
    ...     self.vars = [tf.Variable(1.0), tf.Variable(2.0)]

    >>> root = SimpleModule(name="root")
    >>> leaf = root.leaf = SimpleModule(name="leaf")
    >>> leaf.leaf3 = tf.Variable(6.0, name="leaf3")
    >>> leaf.leaf4 = tf.Variable(7.0, name="leaf4")
    >>> ckpt = tf.train.Checkpoint(root)
    >>> save_path = ckpt.save('/tmp/tf_ckpts')
    >>> checkpoint_view = tf.train.CheckpointView(save_path)

    >>> root2 = SimpleModule(name="root")
    >>> leaf2 = root2.leaf2 = SimpleModule(name="leaf2")
    >>> leaf2.leaf3 = tf.Variable(6.0)
    >>> leaf2.leaf4 = tf.Variable(7.0)

    Pass `node_id=0` to `tf.train.CheckpointView.children()` to get the
    dictionary of all children directly linked to the checkpoint root.

    >>> checkpoint_view_match = checkpoint_view.match(root2).items()
    >>> for item in checkpoint_view_match:
    ...   print(item)
    (0, ...)
    (1, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=5.0>)
    (2, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=4.0>)
    (3, ListWrapper([<tf.Variable 'Variable:0' shape=() dtype=float32,
    numpy=1.0>, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>]))
    (6, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>)
    (7, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>)

    """
    if not isinstance(obj, base.Trackable):
      raise ValueError(f"Expected a Trackable, got {obj} of type {type(obj)}.")

    overlapping_nodes = {}
    # Root node is always matched.
    overlapping_nodes[0] = obj

    # Queue of tuples of node_id and trackable.
    to_visit = collections.deque([(0, obj)])
    visited = set()
    view = trackable_view.TrackableView(obj)
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
