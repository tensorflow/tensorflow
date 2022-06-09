# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Utils for serializing and deserializing TraceTypes."""

import abc
from typing import Type

from google.protobuf import message
from tensorflow.core.function.trace_type import serialization_pb2

SerializedTraceType = serialization_pb2.SerializedTraceType

PROTO_CLASS_TO_PY_CLASS = {}


class Serializable(abc.ABC):
  """TraceTypes implementing this additional interface are portable."""

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    if cls.type_proto() in PROTO_CLASS_TO_PY_CLASS:
      raise ValueError(
          "Existing Python class " +
          PROTO_CLASS_TO_PY_CLASS[cls.type_proto()].__name__ + " already has " +
          cls.type_proto().__name__ +
          " as its associated proto representation. Please ensure " +
          cls.__name__ + " has a unique proto representation.")

    PROTO_CLASS_TO_PY_CLASS[cls.type_proto()] = cls

  @classmethod
  @abc.abstractmethod
  def type_proto(cls) -> Type[message.Message]:
    """Returns the unique type of proto associated with this class."""
    raise NotImplementedError

  @classmethod
  @abc.abstractmethod
  def from_proto(cls, proto: message.Message) -> "Serializable":
    """Returns an instance based on a proto generated from to_proto()."""
    raise NotImplementedError

  @abc.abstractmethod
  def to_proto(self) -> message.Message:
    """Returns a proto of type type_proto() representing this instance."""
    raise NotImplementedError


def serialize(to_serialize: Serializable) -> SerializedTraceType:
  """Converts Serializable to a proto SerializedTraceType."""

  if not isinstance(to_serialize, Serializable):
    raise ValueError("Can not serialize ",
                     type(to_serialize).__name__,
                     " since it is not Serializable. For object ",
                     to_serialize)
  actual_proto = to_serialize.to_proto()

  if not isinstance(actual_proto, to_serialize.type_proto()):
    raise ValueError(
        type(to_serialize).__name__ +
        " returned different type of proto than specified by type_proto()")

  serialized = SerializedTraceType()
  serialized.representation.Pack(actual_proto)
  return serialized


def deserialize(proto: SerializedTraceType) -> Serializable:
  """Converts a proto SerializedTraceType to instance of Serializable."""
  for proto_class in PROTO_CLASS_TO_PY_CLASS:
    if proto.representation.Is(proto_class.DESCRIPTOR):
      actual_proto = proto_class()
      proto.representation.Unpack(actual_proto)
      return PROTO_CLASS_TO_PY_CLASS[proto_class].from_proto(actual_proto)

  raise ValueError(
      "Can not deserialize proto of url: ", proto.representation.type_url,
      " since no matching Python class could be found. For value ",
      proto.representation.value)
