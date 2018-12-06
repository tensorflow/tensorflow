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
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.training.checkpointable import tracking
from tensorflow.python.util import compat


class _Loader(object):
  """Helper class to load an object-based SavedModel."""

  def __init__(self, proto, export_dir):
    self._proto = proto
    self._export_dir = export_dir
    self._load_all()

  def _load_all(self):
    self._nodes = [self._recreate(proto) for proto in self._proto.nodes]
    # After creating the objects, construct the edges between the objects.
    for obj, object_proto in zip(self._nodes, self._proto.nodes):
      for reference in object_proto.children:
        setattr(obj, reference.local_name, self._nodes[reference.node_id])

  def get(self, node_id):
    return self._nodes[node_id]

  def _recreate(self, proto):
    factory = {
        "user_object": lambda: self._recreate_user_object(proto.user_object),
        "asset": lambda: self._recreate_asset(proto.asset),
    }
    kind = proto.WhichOneof("kind")
    if kind not in factory:
      raise ValueError("Unknown SavedObject type: %r" % kind)
    return factory[kind]()

  def _recreate_user_object(self, proto):
    del proto
    return tracking.Checkpointable()

  def _recreate_asset(self, proto):
    filename = os.path.join(
        saved_model_utils.get_assets_dir(self._export_dir),
        proto.relative_filename)
    return tracking.TrackableAsset(filename)


def _load_saved_object_graph_proto(filename):
  with file_io.FileIO(filename, "rb") as f:
    contents = f.read()
    return saved_object_graph_pb2.SavedObjectGraph.FromString(contents)


def load(export_dir):
  """Load a SavedModel from `export_dir`."""
  object_graph_filename = os.path.join(
      compat.as_bytes(export_dir),
      compat.as_bytes(constants.EXTRA_ASSETS_DIRECTORY),
      compat.as_bytes("object_graph.pb"))
  if file_io.file_exists(object_graph_filename):
    proto = _load_saved_object_graph_proto(object_graph_filename)
    loader = _Loader(proto, export_dir)
    root = loader.get(0)
  else:
    raise NotImplementedError(
        "Currently only SavedModels exported with `tf.saved_model.save` may be "
        "imported. Other SavedModels may eventually be supported via load().")
  # TODO(allenl): load functions from the SavedModel into the eager context
  return root
