# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""TraceType implementations for common Python types."""

from typing import Any, Hashable, Optional, Sequence, Type
from typing import Dict as PythonDict
from typing import Tuple as PythonTuple
import weakref

from tensorflow.core.function.trace_type import default_types_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.types import trace


class Literal(trace.TraceType, serialization.Serializable):
  """Represents a Literal type like bool, int or string."""

  def __init__(self, value: Any):
    self.value = value
    self._value_hash = hash(value)

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    return self == other

  def most_specific_common_supertype(
      self, types: Sequence[trace.TraceType]) -> Optional["Literal"]:
    return self if all(self == other for other in types) else None

  @classmethod
  def experimental_type_proto(cls) -> Type[default_types_pb2.SerializedLiteral]:
    return default_types_pb2.SerializedLiteral

  @classmethod
  def experimental_from_proto(
      cls, proto: default_types_pb2.SerializedLiteral) -> "Literal":
    if proto.HasField("bool_value"):
      return Literal(proto.bool_value)

    if proto.HasField("int_value"):
      return Literal(proto.int_value)

    if proto.HasField("float_value"):
      return Literal(proto.float_value)

    if proto.HasField("str_value"):
      return Literal(proto.str_value)

    if proto.HasField("none_value"):
      return Literal(None)

    raise ValueError("Malformed Literal proto can not be deserialized")

  def experimental_as_proto(self) -> default_types_pb2.SerializedLiteral:
    if isinstance(self.value, bool):
      return default_types_pb2.SerializedLiteral(bool_value=self.value)

    if isinstance(self.value, int):
      return default_types_pb2.SerializedLiteral(int_value=self.value)

    if isinstance(self.value, float):
      return default_types_pb2.SerializedLiteral(float_value=self.value)

    if isinstance(self.value, str):
      return default_types_pb2.SerializedLiteral(str_value=self.value)

    if self.value is None:
      return default_types_pb2.SerializedLiteral(
          none_value=default_types_pb2.SerializedLiteral.NoneValue())

    raise ValueError("Can not serialize Literal of type " +
                     type(self.value).__name__)

  def _placeholder_value(self, placeholder_context) -> Any:
    return self.value

  def __eq__(self, other) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    return isinstance(other, Literal) and self.value == other.value

  def __hash__(self) -> int:
    return self._value_hash

  def __repr__(self):
    return f"{self.__class__.__name__}(value={self.value!r})"


class Weakref(trace.TraceType):
  """Represents weakref of an arbitrary Python object.

  When a function argument is a custom class, instead of making a copy of it
  just for the sake of function cache, a weakref is instead kept to save memory.
  """

  def __init__(self, ref: weakref.ReferenceType):
    self._ref = ref
    self._ref_hash = hash(ref)

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    return self == other

  def most_specific_common_supertype(
      self, types: Sequence[trace.TraceType]) -> Optional["Weakref"]:
    return self if all(self == other for other in types) else None

  def _placeholder_value(self, placeholder_context) -> Any:
    return self._ref()

  def __eq__(self, other):
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not isinstance(other, Weakref):
      return False

    if self._ref() is None or other._ref() is None:
      return False

    if self._ref() is other._ref():
      return True

    return self._ref == other._ref

  def __hash__(self):
    return self._ref_hash

  def __repr__(self):
    return f"{self.__class__.__name__}(ref={self._ref!r})"


class Tuple(trace.TraceType, serialization.Serializable):
  """Represents a tuple of TraceType objects."""

  def __init__(self, *components: trace.TraceType):
    self.components = components

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    if (not isinstance(other, Tuple) or
        len(self.components) != len(other.components)):
      return False

    return all(
        self_component.is_subtype_of(other_component) for self_component,
        other_component in zip(self.components, other.components))

  def most_specific_common_supertype(
      self, others: Sequence[trace.TraceType]) -> Optional["Tuple"]:
    """See base class."""
    if not all(
        isinstance(other, Tuple) and
        len(self.components) == len(other.components) for other in others):
      return None

    supertyped_components = []
    for i, component in enumerate(self.components):
      supertyped_component = component.most_specific_common_supertype(
          [other.components[i] for other in others])
      if supertyped_component is None:
        return None
      supertyped_components.append(supertyped_component)

    return Tuple(*supertyped_components)

  @classmethod
  def experimental_type_proto(cls) -> Type[default_types_pb2.SerializedTuple]:
    return default_types_pb2.SerializedTuple

  @classmethod
  def experimental_from_proto(
      cls, proto: default_types_pb2.SerializedTuple) -> "Tuple":
    return Tuple(*[serialization.deserialize(c) for c in proto.components])

  def experimental_as_proto(self) -> default_types_pb2.SerializedTuple:
    return default_types_pb2.SerializedTuple(
        components=[serialization.serialize(c) for c in self.components])

  def _placeholder_value(self, placeholder_context) -> Any:
    components = [
        component._placeholder_value(placeholder_context)  # pylint: disable=protected-access
        for component in self.components
    ]
    return tuple(components)

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not isinstance(other, Tuple):
      return False

    return self.components == other.components

  def __hash__(self) -> int:
    return hash(self.components)

  def __repr__(self):
    return f"Tuple(components={self.components!r})"


class List(trace.TraceType, serialization.Serializable):
  """Represents a list of TraceType objects."""

  def __init__(self, *components: trace.TraceType):
    self.components_tuple = Tuple(*components)

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    if not isinstance(other, List):
      return False

    return self.components_tuple.is_subtype_of(other.components_tuple)

  def most_specific_common_supertype(
      self, others: Sequence[trace.TraceType]) -> Optional["Tuple"]:
    """See base class."""
    if not all(isinstance(other, List) for other in others):
      return None

    supertyped_components_tuple = self.components_tuple.most_specific_common_supertype(
        [other.components_tuple for other in others])

    if supertyped_components_tuple is None:
      return None

    return List(*supertyped_components_tuple.components)

  @classmethod
  def experimental_type_proto(cls) -> Type[default_types_pb2.SerializedList]:
    return default_types_pb2.SerializedList

  @classmethod
  def experimental_from_proto(
      cls, proto: default_types_pb2.SerializedList) -> "List":
    return List(
        *Tuple.experimental_from_proto(proto.components_tuple).components)

  def experimental_as_proto(self) -> default_types_pb2.SerializedList:
    return default_types_pb2.SerializedList(
        components_tuple=self.components_tuple.experimental_as_proto())

  def _placeholder_value(self, placeholder_context) -> Any:
    return list(self.components_tuple._placeholder_value(placeholder_context))  # pylint: disable=protected-access

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not isinstance(other, List):
      return False

    return self.components_tuple == other.components_tuple

  def __hash__(self) -> int:
    return hash(self.components_tuple)

  def __repr__(self):
    return f"List(components={self.components_tuple.components!r})"


class NamedTuple(trace.TraceType, serialization.Serializable):
  """Represents a NamedTuple of TraceType objects."""

  def __init__(self,
               type_name: str,
               attribute_names: PythonTuple[str],
               attributes: PythonTuple[trace.TraceType],
               placeholder_type: Optional[Type[Any]] = None):
    self.type_name = type_name
    self.attribute_names = attribute_names
    self.attributes = Tuple(*attributes)
    self._placeholder_type = placeholder_type

  @classmethod
  def from_type_and_attributes(
      cls, named_tuple_type: Any,
      attributes: PythonTuple[trace.TraceType]) -> "NamedTuple":
    return NamedTuple(named_tuple_type.__name__, named_tuple_type._fields,
                      attributes, named_tuple_type)

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    if not isinstance(other, NamedTuple):
      return False

    return (self.type_name == other.type_name and
            self.attribute_names == other.attribute_names and
            self.attributes.is_subtype_of(other.attributes))

  def most_specific_common_supertype(
      self, others: Sequence[trace.TraceType]) -> Optional["NamedTuple"]:
    """See base class."""
    if not all(
        isinstance(other, NamedTuple) and self.type_name == other.type_name and
        self.attribute_names == other.attribute_names for other in others):
      return None

    supertyped_attributes = self.attributes.most_specific_common_supertype(
        [other.attributes for other in others])

    if supertyped_attributes is None:
      return None

    return NamedTuple(self.type_name, self.attribute_names,
                      supertyped_attributes.components, self._placeholder_type)

  @classmethod
  def experimental_type_proto(
      cls) -> Type[default_types_pb2.SerializedNamedTuple]:
    return default_types_pb2.SerializedNamedTuple

  @classmethod
  def experimental_from_proto(
      cls, proto: default_types_pb2.SerializedNamedTuple) -> "NamedTuple":
    return NamedTuple(
        proto.type_name, tuple(proto.attribute_names),
        Tuple.experimental_from_proto(proto.attributes).components)

  def experimental_as_proto(self) -> default_types_pb2.SerializedNamedTuple:
    return default_types_pb2.SerializedNamedTuple(
        type_name=self.type_name,
        attribute_names=list(self.attribute_names),
        attributes=self.attributes.experimental_as_proto())

  def _placeholder_value(self, placeholder_context) -> Any:
    if self._placeholder_type is None:
      # We don't need to trace after serialization so it is not needed but we
      # can generate a placeholder type using the description if ever needed.
      raise ValueError("Can not generate placeholder value for NamedTuple with"
                       " unspecified placeholder_type. Note: placeholder_type "
                       "is lost during serialization.")
    attribute_placeholders = [
        attribute._placeholder_value(placeholder_context)  # pylint: disable=protected-access
        for attribute in self.attributes.components
    ]
    return self._placeholder_type(*attribute_placeholders)

  def __hash__(self) -> int:
    return hash((self.type_name, self.attribute_names, self.attributes))

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not isinstance(other, NamedTuple):
      return False

    return (self.type_name == other.type_name and
            self.attribute_names == other.attribute_names and
            self.attributes == other.attributes)

  def __repr__(self):
    return (f"NamedTuple(type_name={self.type_name}, "
            f"attribute_names={self.attribute_names}, "
            f"attributes={self.attributes.components})")


class Attrs(trace.TraceType):
  """Represents a class annotated by attr.s."""

  def __init__(self,
               type_name: str,
               attribute_names: PythonTuple[str],
               attributes: PythonTuple[trace.TraceType],
               placeholder_type: Optional[Type[Any]] = None):
    self.named_attributes = NamedTuple(type_name, attribute_names, attributes)
    self._placeholder_type = placeholder_type

  @classmethod
  def from_type_and_attributes(
      cls, attrs_type: Any,
      attributes: PythonTuple[trace.TraceType]) -> "Attrs":
    return Attrs(attrs_type.__name__,
                 tuple(attr.name for attr in attrs_type.__attrs_attrs__),
                 attributes, attrs_type)

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    if not isinstance(other, Attrs):
      return False

    return self.named_attributes.is_subtype_of(other.named_attributes)

  def most_specific_common_supertype(
      self, others: Sequence[trace.TraceType]) -> Optional["Attrs"]:
    """See base class."""
    if not all(isinstance(other, Attrs) for other in others):
      return None

    supertyped_attributes = self.named_attributes.most_specific_common_supertype(
        [other.named_attributes for other in others])

    if supertyped_attributes is None:
      return None

    return Attrs(self.named_attributes.type_name,
                 self.named_attributes.attribute_names,
                 supertyped_attributes.attributes.components,
                 self._placeholder_type)

  @classmethod
  def experimental_type_proto(cls) -> Type[default_types_pb2.SerializedAttrs]:
    return default_types_pb2.SerializedAttrs

  @classmethod
  def experimental_from_proto(
      cls, proto: default_types_pb2.SerializedAttrs) -> "Attrs":
    return Attrs(
        proto.named_attributes.type_name,
        tuple(proto.named_attributes.attribute_names),
        Tuple.experimental_from_proto(
            proto.named_attributes.attributes).components)

  def experimental_as_proto(self) -> default_types_pb2.SerializedAttrs:
    return default_types_pb2.SerializedAttrs(
        named_attributes=self.named_attributes.experimental_as_proto())

  def _placeholder_value(self, placeholder_context) -> Any:
    if self._placeholder_type is None:
      # We don't need to trace after serialization so it is not needed but we
      # can generate a placeholder type using the description if ever needed.
      raise ValueError("Can not generate placeholder value for Attrs with"
                       " unspecified placeholder_type. Note: placeholder_type "
                       "is lost during serialization.")
    attribute_placeholders = [
        attribute._placeholder_value(placeholder_context)  # pylint: disable=protected-access
        for attribute in self.named_attributes.attributes.components
    ]
    return self._placeholder_type(*attribute_placeholders)

  def __hash__(self) -> int:
    return hash(self.named_attributes)

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not isinstance(other, Attrs):
      return False

    return self.named_attributes == other.named_attributes

  def __repr__(self):
    return (f"Attrs(type_name={self.named_attributes.type_name}, "
            f"attribute_names={self.named_attributes.attribute_names}, "
            f"attributes={self.named_attributes.attributes.components})")


class Dict(trace.TraceType, serialization.Serializable):
  """Represents a dictionary of TraceType objects.

  Attributes:
    mapping: A mapping from keys to corresponding TraceTypes of the dict values.
  """

  def __init__(self, mapping: PythonDict[Hashable, trace.TraceType]):
    self.mapping = mapping

  def _has_same_structure(self, other):
    if not isinstance(other, Dict):
      return False

    return self.mapping.keys() == other.mapping.keys()

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    """See base class."""
    if not self._has_same_structure(other):
      return False

    # We need all keys to be present because there can be logic relying on
    # their existence or lack thereof and hence can not guarantee subtype based
    # on a subset or superset of keys.
    # Only the tracing code can explicitly check for key dependencies and inform
    # that decision.
    return all(self.mapping[key].is_subtype_of(other.mapping[key])
               for key in self.mapping)

  def most_specific_common_supertype(
      self, types: Sequence[trace.TraceType]) -> Optional["Dict"]:
    """See base class."""
    if not all(self._has_same_structure(other) for other in types):
      return None

    new_mapping = {}
    for key in self.mapping.keys():
      common = self.mapping[key].most_specific_common_supertype(
          [other.mapping[key] for other in types])
      if common is None:
        return None
      else:
        new_mapping[key] = common

    return Dict(new_mapping)

  @classmethod
  def experimental_type_proto(cls) -> Type[default_types_pb2.SerializedDict]:
    return default_types_pb2.SerializedDict

  @classmethod
  def experimental_from_proto(
      cls, proto: default_types_pb2.SerializedDict) -> "Dict":
    return Dict({
        Literal.experimental_from_proto(k).value: serialization.deserialize(v)
        for k, v in zip(proto.keys, proto.values)
    })

  def experimental_as_proto(self) -> default_types_pb2.SerializedDict:
    return default_types_pb2.SerializedDict(
        keys=[Literal(k).experimental_as_proto() for k in self.mapping.keys()],
        values=[serialization.serialize(v) for v in self.mapping.values()])

  def _placeholder_value(self, placeholder_context) -> Any:
    return {
        key: value._placeholder_value(placeholder_context)  # pylint: disable=protected-access
        for key, value in self.mapping.items()
    }

  def __eq__(self, other) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not isinstance(other, Dict):
      return False

    return self.mapping == other.mapping

  def __hash__(self) -> int:
    return hash(frozenset(self.mapping.keys()))

  def __repr__(self):
    return f"{self.__class__.__name__}(mapping={self.mapping!r})"

serialization.register_serializable(Literal)
serialization.register_serializable(Tuple)
serialization.register_serializable(List)
serialization.register_serializable(NamedTuple)
serialization.register_serializable(Attrs)
serialization.register_serializable(Dict)
