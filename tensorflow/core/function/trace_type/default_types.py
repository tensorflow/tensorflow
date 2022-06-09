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
    self._value = value
    self._value_hash = hash(value)

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    return self == other

  def most_specific_common_supertype(
      self, types: Sequence[trace.TraceType]) -> Optional["Literal"]:
    return self if all(self == other for other in types) else None

  @classmethod
  def type_proto(cls) -> type[default_types_pb2.SerializedLiteral]:
    return default_types_pb2.SerializedLiteral

  @classmethod
  def from_proto(cls, proto: default_types_pb2.SerializedLiteral) -> "Literal":
    if proto.HasField("bool_value"):
      return Literal(proto.bool_value)

    if proto.HasField("int_value"):
      return Literal(proto.int_value)

    if proto.HasField("float_value"):
      return Literal(proto.float_value)

    if proto.HasField("str_value"):
      return Literal(proto.str_value)

    raise ValueError("Malformed Literal proto can not be deserialized")

  def to_proto(self) -> default_types_pb2.SerializedLiteral:
    if isinstance(self._value, bool):
      return default_types_pb2.SerializedLiteral(bool_value=self._value)

    if isinstance(self._value, int):
      return default_types_pb2.SerializedLiteral(int_value=self._value)

    if isinstance(self._value, float):
      return default_types_pb2.SerializedLiteral(float_value=self._value)

    if isinstance(self._value, str):
      return default_types_pb2.SerializedLiteral(str_value=self._value)

    raise ValueError("Can not serialize Literal of type " +
                     type(self._value).__name__)

  def _placeholder_value(self) -> Any:
    return self._value

  def __eq__(self, other) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    return isinstance(other, Literal) and self._value == other._value

  def __hash__(self) -> int:
    return self._value_hash

  def __repr__(self):
    return f"{self.__class__.__name__}(value={self._value!r})"


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

  def _placeholder_value(self) -> Any:
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


class OrderedCollection(trace.TraceType):
  """Represents an ordered collection of TraceType objects.

  Attributes:
    collection_type: Python type for the collection (list, tuple etc.)
    components: A corresponding sequence of TraceTypes to the values in the
      collection.
  """

  def __init__(self, collection_type: Type[Any],
               components: PythonTuple[trace.TraceType]):
    self.collection_type = collection_type
    self.components = components

  def _shallow_equal(self, other):
    return (isinstance(other, OrderedCollection) and
            self.collection_type == other.collection_type and
            len(self.components) == len(other.components))

  def _supertype_components(
      self, others: Sequence["OrderedCollection"]
  ) -> Optional[Sequence[trace.TraceType]]:
    """Helper that generates a list of per-component supertypes or None."""
    new_components = []
    for i, component in enumerate(self.components):
      common = component.most_specific_common_supertype(
          [other.components[i] for other in others])
      if common is None:
        return None
      else:
        new_components.append(common)
    return new_components

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    """See base class."""
    if not self._shallow_equal(other):
      return False

    return all(
        self_component.is_subtype_of(other_component) for self_component,
        other_component in zip(self.components, other.components))

  def __eq__(self, other) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not self._shallow_equal(other):
      return False

    return self.components == other.components

  def __hash__(self) -> int:
    return hash((self.collection_type, self.components))

  def __repr__(self):
    return (f"{self.__class__.__name__}(collection_type="
            f"{self.collection_type!r}, components={self.components!r})")


class List(OrderedCollection):
  """Represents a list of TraceType objects."""

  def __init__(self, *components: trace.TraceType):
    super().__init__(list, components)

  def most_specific_common_supertype(
      self, types: Sequence[trace.TraceType]) -> Optional["List"]:
    """See base class."""
    if not all(self._shallow_equal(other) for other in types):
      return None

    new_components = self._supertype_components(types)

    return None if new_components is None else List(*new_components)

  def _placeholder_value(self) -> Any:
    components = [
        component._placeholder_value()  # pylint: disable=protected-access
        for component in self.components
    ]
    return list(components)


class Tuple(OrderedCollection):
  """Represents a tuple of TraceType objects."""

  def __init__(self, *components: trace.TraceType):
    super().__init__(tuple, components)

  def most_specific_common_supertype(
      self, types: Sequence[trace.TraceType]) -> Optional["Tuple"]:
    """See base class."""
    if not all(self._shallow_equal(other) for other in types):
      return None

    new_components = self._supertype_components(types)

    return None if new_components is None else Tuple(*new_components)

  def _placeholder_value(self) -> Any:
    components = [
        component._placeholder_value()  # pylint: disable=protected-access
        for component in self.components
    ]
    return tuple(components)


class NamedTuple(OrderedCollection):
  """Represents a NamedTuple of TraceType objects."""

  def __init__(self, collection_type: Type[object],
               attributes: PythonTuple[trace.TraceType]):
    super().__init__(collection_type, attributes)

  def most_specific_common_supertype(
      self, types: Sequence[trace.TraceType]) -> Optional["NamedTuple"]:
    """See base class."""
    if not all(self._shallow_equal(other) for other in types):
      return None

    new_components = self._supertype_components(types)

    return None if new_components is None else type(self)(self.collection_type,
                                                          tuple(new_components))

  def _placeholder_value(self) -> Any:
    components = [
        component._placeholder_value()  # pylint: disable=protected-access
        for component in self.components
    ]
    return self.collection_type(*components)


class Attrs(NamedTuple):
  """Represents a class annotated by attr.s.

  Each attr.s class has a fixed, ordered set of attributes. Therefore, we only
  need to consider the class type and the underlying attributes. Extra
  metadata including attribute names can be ignored.
  """


class Dict(trace.TraceType):
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

  def _placeholder_value(self) -> Any:
    return {
        key: value._placeholder_value()  # pylint: disable=protected-access
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


class Reference(trace.TraceType):
  """Represents a resource with an identifier.

  Resource identifiers are useful to denote identical resources, that is,
  resources which are known at compilation time to point to the same thing.
  This information is useful in automatic control dependencies for instance,
  where ops using the same resource don't run concurrently.
  """

  def __init__(self, base: trace.TraceType, identifier: Hashable):
    self.base = base
    self.identifier = identifier

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    if isinstance(other, Reference) and self.identifier == other.identifier:
      return self.base.is_subtype_of(other.base)
    return False

  def most_specific_common_supertype(
      self, types: Sequence[trace.TraceType]) -> Optional["Reference"]:
    if all(
        isinstance(other, Reference) and self.identifier == other.identifier
        for other in types):
      base_supertype = self.base.most_specific_common_supertype(
          [other.base for other in types])
      if base_supertype is not None:
        return Reference(base_supertype, self.identifier)
    return None

  def _placeholder_value(self) -> Any:
    return self.base._placeholder_value()  # pylint: disable=protected-access

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    return isinstance(
        other, Reference
    ) and self.identifier == other.identifier and self.base == other.base

  def __hash__(self) -> int:
    return hash((self.identifier, self.base))

  def __repr__(self):
    return (f"{self.__class__.__name__}(base={self.base!r}, "
            f"identifier={self.identifier!r})")
