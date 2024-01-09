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
"""Utilities for Proto Splitter modules."""

from collections.abc import Sequence
from typing import Any, Optional, Union

from google.protobuf import descriptor
from google.protobuf import message
from tensorflow.tools.proto_splitter import chunk_pb2

_BYTE_UNITS = [(1, "B"), (1 << 10, "KiB"), (1 << 20, "MiB"), (1 << 30, "GiB")]


def format_bytes(b: int) -> str:
  """Formats bytes into a human-readable string."""
  for i in range(1, len(_BYTE_UNITS)):
    if b < _BYTE_UNITS[i][0]:
      n = f"{b / _BYTE_UNITS[i-1][0]:.2f}"
      units = _BYTE_UNITS[i - 1][1]
      break
  else:
    n = f"{b / _BYTE_UNITS[-1][0]:.2f}"
    units = _BYTE_UNITS[-1][1]
  n = n.rstrip("0").rstrip(".")
  return f"{n}{units}"


FieldTypes = Union[str, int, bool, Sequence[Union[str, int, bool]]]


def get_field(
    proto: message.Message, fields: FieldTypes
) -> tuple[Any, Optional[descriptor.FieldDescriptor]]:
  """Returns the field and field descriptor from the proto.

  Args:
    proto: Parent proto of any message type.
    fields: List of string/int/map key fields, e.g. ["nodes", "attr", "value"]
      can represent `proto.nodes.attr["value"]`.

  Returns:
    Tuple of (
      Field in the proto or `None` if none are found,
      Field descriptor
    )
  """
  field_proto = proto
  field_desc = None
  for field_proto, field_desc, _, _ in _walk_fields(proto, fields):
    pass
  return field_proto, field_desc


def get_field_tag(
    proto: message.Message, fields: FieldTypes
) -> Sequence[chunk_pb2.FieldIndex]:
  """Generates FieldIndex proto for a nested field within a proto.

  Args:
    proto: Parent proto of any message type.
    fields: List of string/int/map key fields, e.g. ["nodes", "attr", "value"]
      can represent `proto.nodes.attr["value"]`.

  Returns:
    A list of FieldIndex protos with the same length as `fields`.
  """
  field_tags = []
  for _, field_desc, map_key, list_index in _walk_fields(proto, fields):
    field_tags.append(chunk_pb2.FieldIndex(field=field_desc.number))
    if map_key is not None:
      key_type = field_desc.message_type.fields_by_name["key"].type
      field_tags.append(
          chunk_pb2.FieldIndex(map_key=_map_key_proto(key_type, map_key))
      )
    elif list_index is not None:
      field_tags.append(chunk_pb2.FieldIndex(index=list_index))
  return field_tags


def _walk_fields(proto: message.Message, fields: FieldTypes):
  """Yields fields in a proto.

  Args:
    proto: Parent proto of any message type.
    fields: List of string/int/map key fields, e.g. ["nodes", "attr", "value"]
      can represent `proto.nodes.attr["value"]`.

  Yields:
    Tuple of (
      Field in the proto or `None` if none are found,
      Field descriptor,
      Key into this map field (or None),
      Index into this repeated field (or None))
  """
  if not isinstance(fields, list):
    fields = [fields]

  field_proto = proto
  parent_desc = proto.DESCRIPTOR
  i = 0
  while i < len(fields):
    field = fields[i]
    field_desc = None
    map_key = None
    index = None

    if parent_desc is None:
      raise ValueError(
          f"Unable to find fields: {fields} in proto of type {type(proto)}."
      )

    if isinstance(field, int):
      try:
        field_desc = parent_desc.fields_by_number[field]
      except KeyError:
        raise KeyError(  # pylint:disable=raise-missing-from
            f"Unable to find field number {field} in {parent_desc.full_name}. "
            f"Valid field numbers: {parent_desc.fields_by_number.keys()}"
        )
    elif isinstance(field, str):
      try:
        field_desc = parent_desc.fields_by_name[field]
      except KeyError:
        raise KeyError(  # pylint:disable=raise-missing-from
            f"Unable to find field '{field}' in {parent_desc.full_name}. "
            f"Valid field names: {parent_desc.fields_by_name.keys()}"
        )
    else:  # bool (only expected as map key)
      raise TypeError("Unexpected bool found in field list.")

    i += 1
    parent_desc = field_desc.message_type
    if field_proto is not None:
      field_proto = getattr(field_proto, field_desc.name)

    # Handle special fields types (map key and list index).
    if _is_map(parent_desc) and i < len(fields):
      # Next field is the map key.
      map_key = fields[i]

      try:
        field_proto = field_proto[map_key] if field_proto is not None else None
      except KeyError:
        field_proto = None
      i += 1

      if i < len(fields):
        # The next field must be from the Value Message.
        value_desc = parent_desc.fields_by_name["value"]
        assert value_desc.message_type is not None
        parent_desc = value_desc.message_type

    elif is_repeated(field_desc) and i < len(fields):
      # The next field is the index within the list.
      index = fields[i]
      try:
        field_proto = field_proto[index] if field_proto is not None else None
      except IndexError:
        field_proto = None
      i += 1

    yield field_proto, field_desc, map_key, index


def _is_map(desc: descriptor.Descriptor) -> bool:
  return desc.GetOptions().map_entry if desc is not None else False


def is_repeated(field_desc: descriptor.FieldDescriptor) -> bool:
  return field_desc.label == descriptor.FieldDescriptor.LABEL_REPEATED


_FIELD_DESC = descriptor.FieldDescriptor
_MAP_KEY = {
    _FIELD_DESC.TYPE_STRING: lambda key: chunk_pb2.FieldIndex.MapKey(s=key),
    _FIELD_DESC.TYPE_BOOL: lambda key: chunk_pb2.FieldIndex.MapKey(boolean=key),
    _FIELD_DESC.TYPE_UINT32: lambda key: chunk_pb2.FieldIndex.MapKey(ui32=key),
    _FIELD_DESC.TYPE_UINT64: lambda key: chunk_pb2.FieldIndex.MapKey(ui64=key),
    _FIELD_DESC.TYPE_INT32: lambda key: chunk_pb2.FieldIndex.MapKey(i32=key),
    _FIELD_DESC.TYPE_INT64: lambda key: chunk_pb2.FieldIndex.MapKey(i64=key),
}


def _map_key_proto(key_type, key):
  """Returns MapKey proto for a key of key_type."""
  return _MAP_KEY[key_type](key)
