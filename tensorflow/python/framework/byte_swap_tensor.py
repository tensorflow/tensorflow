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

"""Utilities for byte swapping the tensor content."""

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import dtypes

# Based on tensor_bundle/byte_swap.cc
byte_swappable = [
    dtypes.float16,
    dtypes.float32,
    dtypes.float64,
    dtypes.bfloat16,
    dtypes.complex64,
    dtypes.complex128,
    dtypes.uint16,
    dtypes.uint32,
    dtypes.uint64,
    dtypes.int16,
    dtypes.int32,
    dtypes.int64,
    dtypes.qint16,
    dtypes.quint16,
    dtypes.qint32,
]


def byte_swap_tensor_content(tensor, from_endiness, to_endiness):
  """Byte swaps.

  Args:
    tensor: Target tensor to change endiness.
    from_endiness: The original endianness format. "big" or "little"
    to_endiness: The target endianness format. "big" or "little"
  """
  if tensor.dtype in byte_swappable:
    tshape = tensor.tensor_shape.dim
    tensor_bytes = tensor.tensor_content
    if tensor_bytes:
      tensor_size = 1
      for sz in tshape:
        if sz.size != 0:
          tensor_size *= sz.size
      chunksize = len(tensor_bytes) // tensor_size
      # Split tensor_data into chunks for byte swapping.
      to_swap = [
          tensor_bytes[i : i + chunksize]
          for i in range(0, len(tensor_bytes), chunksize)
      ]
      # Swap and replace tensor_content.
      tensor.tensor_content = b"".join(
          [
              int.from_bytes(byteswap, from_endiness).to_bytes(
                  chunksize, to_endiness
              )
              for byteswap in to_swap
          ]
      )


def swap_tensor_content_in_graph_function(
    graph_def, from_endiness, to_endiness
):
  """Fix endiness of tensor contents.

  Args:
    graph_def: Target graph_def to change endiness.
    from_endiness: The original endianness format. "big" or "little"
    to_endiness: The target endianness format. "big" or "little"
  """
  if isinstance(graph_def, meta_graph_pb2.MetaGraphDef):
    functions = graph_def.graph_def.library.function
  elif isinstance(graph_def, graph_pb2.GraphDef):
    functions = graph_def.library.function
  else:
    return
  for function in functions:
    node_def = function.node_def
    for node in node_def:
      if node.op == "Const":
        tensor = node.attr["value"].tensor
        byte_swap_tensor_content(tensor, from_endiness, to_endiness)


def swap_tensor_content_in_graph_node(graph_def, from_endiness, to_endiness):
  for node in graph_def.node:
    if node.op == "Const":
      tensor = node.attr["value"].tensor
      byte_swap_tensor_content(tensor, from_endiness, to_endiness)
