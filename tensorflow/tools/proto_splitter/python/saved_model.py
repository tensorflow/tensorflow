# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""SavedModel Splitter."""

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.tools.proto_splitter import constants
from tensorflow.tools.proto_splitter import split
from tensorflow.tools.proto_splitter import split_graph_def


class SavedModelSplitter(split.ComposableSplitter):
  """Splits a SavedModel proto into chunks of size < 2GB."""

  def build_chunks(self):
    if not isinstance(self._proto, saved_model_pb2.SavedModel):
      raise TypeError(
          "SavedModelSplitter can only split SavedModel protos. "
          f"Got {type(self._proto)}."
      )

    if self._proto.ByteSize() >= constants.max_size():
      graph_def = self._proto.meta_graphs[0].graph_def
      graph_def_fields = ["meta_graphs", 0, "graph_def"]
      split_graph_def.GraphDefSplitter(
          self._proto.meta_graphs[0].graph_def,
          parent_splitter=self,
          fields_in_parent=graph_def_fields,
      ).build_chunks()

    # Check if the proto size is still larger than the max size.
    if self._proto.ByteSize() >= constants.max_size():
      # Create a chunk for the GraphDef, and ensure the GraphDef is merged in
      # first by adding it at index 1. The 0th chunk is the SavedModel itself.
      self.add_chunk(graph_def, graph_def_fields, index=1)
      self._proto.meta_graphs[0].ClearField("graph_def")
