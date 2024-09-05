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
"""GraphDef splitter."""

from collections.abc import Sequence
import itertools
from typing import Optional, Type

from google.protobuf import message
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.tools.proto_splitter import constants
from tensorflow.tools.proto_splitter import split
from tensorflow.tools.proto_splitter import util

_CONST_OP = "Const"


class GraphDefSplitter(split.ComposableSplitter):
  """Implements proto splitter for GraphDef.

  This Splitter will modify the passed in proto in place.
  """

  def build_chunks(self):
    """Splits a GraphDef proto into smaller chunks."""
    proto = self._proto
    if not isinstance(proto, graph_pb2.GraphDef):
      raise TypeError("Can only split GraphDef type protos.")

    proto_size = proto.ByteSize()
    if proto_size < constants.max_size():
      return

    # Split `GraphDef.node`
    node_splitter = RepeatedMessageSplitter(
        proto,
        "node",
        [ConstantNodeDefSplitter, LargeMessageSplitter],
        parent_splitter=self,
        fields_in_parent=[],
    )

    # Split `GraphDef.library.function`
    function_splitter = RepeatedMessageSplitter(
        proto.library,
        ["function"],
        [FunctionDefSplitter],
        parent_splitter=self,
        fields_in_parent=["library"],
    )

    library_size = proto.library.ByteSize()
    approx_node_size = proto_size - library_size

    if library_size > approx_node_size:
      library_size -= function_splitter.build_chunks()
      if library_size + approx_node_size > constants.max_size():
        approx_node_size -= node_splitter.build_chunks()

    else:
      approx_node_size -= node_splitter.build_chunks()
      if library_size + approx_node_size > constants.max_size():
        library_size -= function_splitter.build_chunks()

    if proto.ByteSize() > constants.max_size():
      # Since there are chunks with the "library" field tag, insert this
      # chunk before the other chunks at index 1 (index 0 is reserved for the
      # base chunk).
      self.add_chunk(proto.library, ["library"], 1)
      proto.ClearField("library")

_KEEP_TENSOR_PROTO_FIELDS = ("dtype", "tensor_shape", "version_number")


def chunk_constant_value(node: node_def_pb2.NodeDef, size: int):
  """Extracts and clears the constant value from a NodeDef.

  Args:
    node: NodeDef with const value to extract.
    size: Size of NodeDef (for error reporting).

  Returns:
    Bytes representation of the Constant tensor content.
  """
  if node.op == _CONST_OP:
    tensor_proto = node.attr["value"].tensor
    if tensor_proto.tensor_content:
      b = tensor_proto.tensor_content
    else:
      # The raw tensor value could be stored in one of the "xxx_val" attributes.
      # Extract it here, and convert to bytes.
      b = tensor_util.MakeNdarray(tensor_proto).tobytes()

    # Keep the TensorProto's dtype, tensor_shape, and version_number fields,
    # but clear the raw tensor content / "xxx_val" attributes.
    kept_attributes = {
        key: getattr(tensor_proto, key) for key in _KEEP_TENSOR_PROTO_FIELDS
    }
    tensor_proto.Clear()
    for field, val in kept_attributes.items():
      if isinstance(val, message.Message):
        getattr(tensor_proto, field).MergeFrom(val)
      else:
        setattr(tensor_proto, field, val)

    return b
  else:
    attributes_and_sizes = ", ".join(
        [
            f"{key}: {util.format_bytes(val.ByteSize())}"
            for key, val in node.attr.items()
        ]
    )
    raise ValueError(
        "Unable to split GraphDef because at least one of the nodes "
        "individually exceeds the max size of "
        f"{util.format_bytes(constants.max_size())}. "
        "Currently only Const nodes can be further split."
        "\nNode info:"
        f"\n\tsize: {util.format_bytes(size)}"
        f"\n\tname: {node.name}"
        f"\n\top: {node.op}"
        f"\n\tinputs: {node.input}"
        f"\n\top: {node.op}"
        f"\n\tdevice: {node.device}"
        f"\n\tattr (and sizes): {attributes_and_sizes}"
    )


def _split_repeated_field(
    proto: message.Message,
    new_proto: message.Message,
    fields: util.FieldTypes,
    start_index: int,
    end_index: Optional[int] = None,
) -> None:
  """Generic function for copying a repeated field from one proto to another."""
  util.get_field(new_proto, fields)[0].MergeFrom(
      util.get_field(proto, fields)[0][start_index:end_index]
  )


_ABOVE_MAX_SIZE = lambda x: x > constants.max_size()
_GREEDY_SPLIT = lambda x: x > constants.max_size() // 3
_ALWAYS_SPLIT = lambda x: True


class SplitBasedOnSize(split.ComposableSplitter):
  """A Splitter that's based on the size of the input proto."""

  __slots__ = ("fn", "proto_size")

  def __init__(self, proto, proto_size, **kwargs):
    """Initializer."""
    self.proto_size = proto_size
    super().__init__(proto, **kwargs)

  def build_chunks(self) -> int:
    """Splits the proto, and returns the size of the chunks created."""
    return 0


class RepeatedMessageSplitter(split.ComposableSplitter):
  """Splits a repeated message field on a proto."""

  __slots__ = ("repeated_field", "message_splitters")

  def __init__(
      self,
      proto,
      repeated_field: util.FieldTypes,
      message_splitters: Sequence[Type[SplitBasedOnSize]],
      **kwargs,
  ):
    """Initializer."""
    super().__init__(proto, **kwargs)
    if not isinstance(repeated_field, list):
      repeated_field = [repeated_field]
    self.repeated_field = repeated_field
    self.message_splitters = message_splitters

  def build_chunks(self) -> int:
    """Splits the proto, and returns the size of the chunks created."""
    proto = self._proto

    total_size_diff = 0

    field, field_desc = util.get_field(proto, self.repeated_field)
    if not util.is_repeated(field_desc) and field_desc.message_type:
      raise ValueError(
          "RepeatedMessageSplitter can only be used on repeated fields. "
          f"Got proto={type(proto)}, field='{field_desc.name}'"
      )

    # List of indices at which to split the repeated field. For example, [3, 5]
    # means that the field list is split into: [:3], [3:5], [5:]
    repeated_msg_split = []
    # Should be the same length as the list above. Contains new protos to hold
    # the elements that are split from the original proto.
    # From the [3, 5] example above, the messages in this list contain nodes
    # [3:5] and [5:]
    repeated_msg_graphs = []
    # Track the total size of the current node split.
    total_size = 0

    # Linearly iterate through all nodes. It may be possible to optimize this
    # further by making best guesses as to where to split the nodes, since
    # most nodes (aside from constants) are relatively small.
    for n, ele in enumerate(field):
      size = ele.ByteSize()

      for splitter_cls in self.message_splitters:
        splitter = splitter_cls(
            ele,
            size,
            parent_splitter=self,
            fields_in_parent=self.repeated_field + [n],
        )
        size_diff = splitter.build_chunks()
        total_size_diff += size_diff
        size -= size_diff

      # Create a new GraphDef chunk if the current list of nodes is too large.
      if total_size + size >= constants.max_size():
        new_msg = type(self._proto)()
        repeated_msg_split.append(n)
        repeated_msg_graphs.append(new_msg)
        self.add_chunk(new_msg, [])

        if len(repeated_msg_split) >= 1:
          total_size_diff += total_size

        total_size = 0

      total_size += size

    if repeated_msg_split:
      # Finish writing repeated chunks.
      start = repeated_msg_split[0]
      for end, msg in zip(
          itertools.chain.from_iterable([repeated_msg_split[1:], [None]]),
          repeated_msg_graphs,
      ):
        _split_repeated_field(proto, msg, self.repeated_field, start, end)
        start = end
      del field[repeated_msg_split[0] :]

    return total_size_diff


class ConstantNodeDefSplitter(SplitBasedOnSize):
  """Extracts constant value from a `Const` NodeDef."""

  def build_chunks(self) -> int:
    """Splits a NodeDef proto, and returns the size of the chunks created."""
    if _ABOVE_MAX_SIZE(self.proto_size):
      constant_bytes = chunk_constant_value(self._proto, self.proto_size)
      self.add_chunk(
          constant_bytes,
          ["attr", "value", "tensor", "tensor_content"],
      )
      return len(constant_bytes)
    return 0


class LargeMessageSplitter(SplitBasedOnSize):
  """Splits a message into a separaet chunk if its over a certain size."""

  __slots__ = ("size_check",)

  def __init__(self, proto, proto_size, size_check=_GREEDY_SPLIT, **kwargs):
    """Initializer."""

    self.size_check = size_check
    super().__init__(proto, proto_size, **kwargs)

  def build_chunks(self) -> int:
    """Creates a chunk for the entire proto and returns the original size."""
    if self.size_check(self.proto_size):
      new_proto = type(self._proto)()
      new_proto.MergeFrom(self._proto)
      self._proto.Clear()
      self.add_chunk(new_proto, [])
      return self.proto_size
    return 0


class FunctionDefSplitter(SplitBasedOnSize):
  """Splits the FunctionDef message type."""

  def build_chunks(self) -> int:
    """Splits the proto, and returns the size of the chunks created."""
    size_diff = 0

    # First check if the entire FunctionDef can be split into a separate chunk.
    # We do this before the `RepeatedMessageSplitter`, which is costly because
    # it iterates through every `node_def`.
    if _GREEDY_SPLIT(self.proto_size) and not _ABOVE_MAX_SIZE(self.proto_size):
      size_diff += LargeMessageSplitter(
          self._proto,
          self.proto_size,
          parent_splitter=self,
          fields_in_parent=[],
      ).build_chunks()

    if _ABOVE_MAX_SIZE(self.proto_size):
      # Split FunctionDefLibrary.function.node_def
      size_diff += RepeatedMessageSplitter(
          self._proto,
          "node_def",
          [ConstantNodeDefSplitter, LargeMessageSplitter],
          parent_splitter=self,
          fields_in_parent=[],
      ).build_chunks()
    return size_diff
