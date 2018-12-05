# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Import a checkpointable object from a SavedModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import saved_object_graph_pb2
from tensorflow.python.training.checkpointable import tracking
from tensorflow.python.util import compat


def _recreate_object_graph(object_graph_proto):
  """Recreates Python objects from an ObjectGraph proto."""
  objects = []
  for _ in object_graph_proto.nodes:
    # TODO(allenl): re-create variables and other types
    objects.append(tracking.Checkpointable())
  for obj, object_proto in zip(objects, object_graph_proto.nodes):
    for reference in object_proto.children:
      setattr(obj, reference.local_name, objects[reference.node_id])
  return objects[0]


def load(export_dir):
  """Load a SavedModel from `export_dir`."""
  object_graph_filename = os.path.join(
      compat.as_bytes(export_dir),
      compat.as_bytes(constants.EXTRA_ASSETS_DIRECTORY),
      compat.as_bytes("object_graph.pb"))
  if file_io.file_exists(object_graph_filename):
    # If there is an object graph associated with the SavedModel, we'll create a
    # root object from that.
    object_graph_string = file_io.FileIO(object_graph_filename, "rb").read()
    object_graph_proto = (
        saved_object_graph_pb2.SavedObjectGraph())
    object_graph_proto.ParseFromString(object_graph_string)
    root = _recreate_object_graph(object_graph_proto)
  else:
    raise NotImplementedError(
        "Currently only SavedModels exported with `tf.saved_model.save` may be "
        "imported. Other SavedModels may eventually be supported via load().")
  # TODO(allenl): load functions from the SavedModel into the eager context
  return root
