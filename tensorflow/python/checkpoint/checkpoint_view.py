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

from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.framework import errors_impl
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
