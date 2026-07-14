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

import collections
import math
import numbers
from typing import Any, Dict as PythonDict, Hashable, List as PythonList, Optional, Sequence, Tuple as PythonTuple, Type
import weakref

from tensorflow.core.function.trace_type import default_types_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace

# Register the TraceType of Tensor (aka TensorSpec) to avoid cyclic dependency.
TENSOR = None


def register_tensor_type(tensor_type):
  global TENSOR
  if not TENSOR:
    TENSOR = tensor_type
  else:
    raise AssertionError("Tensor type is already registered.")

NanMarker = object()


def is_nan(x):
  """Checks if given value is a Python NaN."""
  if not isinstance(x, numbers.Number):
    return False

  if isinstance(x, complex):
    return math.isnan(x.real) or math.isnan(x.imag)
  else:
    return math.isnan(x)


class Literal(trace.TraceType, serialization.Serializable):
  """Represents a Literal type like bool, int or string."""

  def __init__(self, value: Any):
    # We match nan values against each other even though Python doesn't.
    if is_nan(value):
      value = NanMarker

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

  def placeholder_value(self, placeholder_context) -> Any:
    # TODO(b/263505796): Remove this check when a range's placeholder output
    # is expected to be a range and not a list.
    if isinstance(self.value, range):
      return list(self.value)

    if self.value is NanMarker:
      return float("nan")

    return self.value

  def cast(self, value: Any, casting_context: Any) -> Any:
    if self.value is NanMarker and is_nan(value):
      return value

    if value == self.value:
      return value
    else:
      raise ValueError(f"Can not cast {value!r} to {self!r}")

  def __eq__(self, other) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    return isinstance(other, Literal) and self.value == other.value

  def __hash__(self) -> int:
    return self._value_hash

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}[{self.value!r}]"


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

  def placeholder_value(self, placeholder_context) -> Any:
    return self._ref()

  def cast(self, value, _):
    if value is self._ref() or value == self._ref():
      return value

    # We unwrap objects when generating the TraceType so we allow matching now.
    while hasattr(value, "__wrapped__"):
      value = value.__wrapped__
      if value is self._ref():
        return value

    raise ValueError(f"Can not cast {value!r} to {self!r}")

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

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}[{self._ref!r}])"


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

  def placeholder_value(self, placeholder_context) -> Any:
    components = [
        component.placeholder_value(placeholder_context)
        for component in self.components
    ]
    return tuple(components)

  def to_tensors(self, value) -> Any:
    assert isinstance(value, tuple)
    flattened_values = []
    for comp_value, comp_type in zip(value, self.components):
      flattened_values.extend(comp_type.to_tensors(comp_value))
    return flattened_values

  def from_tensors(self, tensors) -> Any:
    return tuple(c.from_tensors(tensors) for c in self.components)

  def flatten(self) -> PythonList[trace.TraceType]:
    flattened_types = []
    for component in self.components:
      flattened_types.extend(component.flatten())
    return flattened_types

  def cast(self, value: Any, casting_context) -> Any:
    assert isinstance(value, tuple), f"Can not cast {value!r} to tuple type."
    assert len(value) == len(
        self.components
    ), f"Expected {value} to have length of {len(self.components)}"

    casted_values, was_casted = util.cast_and_return_whether_casted(
        self.components, value, casting_context
    )
    if was_casted:
      return tuple(casted_values)
    else:
      return value

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not isinstance(other, Tuple):
      return False

    return self.components == other.components

  def __hash__(self) -> int:
    return hash(self.components)

  def __repr__(self) -> str:
    return f"Tuple[{', '.join(map(repr, self.components))}]"


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

    supertyped_components_tuple = (
        self.components_tuple.most_specific_common_supertype(
            [other.components_tuple for other in others]
        )
    )

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

  def placeholder_value(self, placeholder_context) -> Any:
    return list(self.components_tuple.placeholder_value(placeholder_context))

  def to_tensors(self, value):
    assert isinstance(value, list)
    return self.components_tuple.to_tensors(tuple(value))

  def from_tensors(self, tensors) -> Any:
    return list(self.components_tuple.from_tensors(tensors))

  def flatten(self) -> PythonList[trace.TraceType]:
    return self.components_tuple.flatten()

  def cast(self, value: Any, casting_context) -> Any:
    assert isinstance(value, list), f"Can not cast {value!r} to list type."
    assert len(value) == len(
        self.components_tuple.components
    ), f"Expected {value} to have length of {len(self.components_tuple)}"

    casted_values, was_casted = util.cast_and_return_whether_casted(
        self.components_tuple.components, value, casting_context
    )
    if was_casted:
      return list(casted_values)
    else:
      return value

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not isinstance(other, List):
      return False

    return self.components_tuple == other.components_tuple

  def __hash__(self) -> int:
    return hash(self.components_tuple)

  def __repr__(self) -> str:
    return f"List[{', '.join(map(repr, self.components_tuple.components))}]"


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

  def placeholder_value(self, placeholder_context) -> Any:
    if self._placeholder_type is None:
      # We don't need to trace after serialization so it is not needed but we
      # can generate a placeholder type using the description if ever needed.
      raise ValueError("Can not generate placeholder value for NamedTuple with"
                       " unspecified placeholder_type. Note: placeholder_type "
                       "is lost during serialization.")
    attribute_placeholders = [
        attribute.placeholder_value(placeholder_context)
        for attribute in self.attributes.components
    ]
    return self._placeholder_type(*attribute_placeholders)

  def to_tensors(self, value: Any):
    assert util.is_namedtuple(value)
    flattened_values = []
    for attribute_name, attribute_type in zip(
        self.attribute_names, self.attributes.components):
      attribute_value = getattr(value, attribute_name)
      flattened_values.extend(attribute_type.to_tensors(attribute_value))
    return flattened_values

  def from_tensors(self, tensors) -> Any:
    if self._placeholder_type is None:
      raise ValueError("Packing serialized NamedTuples is not supported.")

    return self._placeholder_type(
        *[c.from_tensors(tensors) for c in self.attributes.components]
    )

  def flatten(self) -> PythonList[trace.TraceType]:
    flattened_types = []

    for component in self.attributes.components:
      flattened_types.extend(component.flatten())

    return flattened_types

  def cast(self, value: Any, casting_context) -> Any:
    # Value must have same attributes with the TraceType
    assert util.is_namedtuple(
        value
    ), f"Cannot cast {value!r} to type {self._placeholder_type!r}."
    value_dict = value._asdict()
    assert set(value_dict.keys()) == set(
        self.attribute_names
    ), f"{value!r} has different attributes with the TraceType {self!r}"

    casted_values, was_casted = util.cast_and_return_whether_casted(
        self.attributes.components,
        [getattr(value, name) for name in self.attribute_names],
        casting_context,
    )
    if was_casted:
      return self._placeholder_type(*casted_values)
    else:
      return value

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

  def __repr__(self) -> str:
    paired = [
        f"[{n!r}, {c!r}]"
        for n, c in zip(self.attribute_names, self.attributes.components)
    ]
    return f"{self.type_name}[{', '.join(paired)}]"


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

    supertyped_attributes = (
        self.named_attributes.most_specific_common_supertype(
            [other.named_attributes for other in others]
        )
    )

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

  def placeholder_value(self, placeholder_context) -> Any:
    if self._placeholder_type is None:
      # We don't need to trace after serialization so it is not needed but we
      # can generate a placeholder type using the description if ever needed.
      raise ValueError("Can not generate placeholder value for Attrs with"
                       " unspecified placeholder_type. Note: placeholder_type "
                       "is lost during serialization.")
    attribute_placeholders = [
        attribute.placeholder_value(placeholder_context)
        for attribute in self.named_attributes.attributes.components
    ]
    return self._placeholder_type(*attribute_placeholders)

  def to_tensors(self, value: Any):
    assert util.is_attrs(value)
    flattened_values = []
    for attribute_name, attribute_type in zip(
        self.named_attributes.attribute_names,
        self.named_attributes.attributes.components):
      attribute_value = getattr(value, attribute_name)
      flattened_values.extend(attribute_type.to_tensors(attribute_value))
    return flattened_values

  def from_tensors(self, tensors):
    if self._placeholder_type is None:
      raise ValueError("Packing serialized NamedTuples is not supported.")

    return self._placeholder_type(
        *[
            c.from_tensors(tensors)
            for c in self.named_attributes.attributes.components
        ]
    )

  def flatten(self) -> PythonList[trace.TraceType]:
    flattened_types = []

    for component in self.named_attributes.attributes.components:
      flattened_types.extend(component.flatten())

    return flattened_types

  def cast(self, value: Any, casting_context) -> Any:
    assert util.is_attrs(value)

    attr_names = self.named_attributes.attribute_names
    casted_values, was_casted = util.cast_and_return_whether_casted(
        self.named_attributes.attributes.components,
        [getattr(value, name) for name in attr_names],
        casting_context,
    )

    if was_casted:
      return self._placeholder_type(*casted_values)
    else:
      return value

  def __hash__(self) -> int:
    return hash(self.named_attributes)

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not isinstance(other, Attrs):
      return False

    return self.named_attributes == other.named_attributes

  def __repr__(self) -> str:
    name_component_zip = zip(
        self.named_attributes.attribute_names,
        self.named_attributes.attributes.components,
    )
    paired = [f"[{n!r}, {c!r}]" for n, c in name_component_zip]
    return f"{self.named_attributes.type_name}[{', '.join(paired)}]"


class Dict(trace.TraceType, serialization.Serializable):
  """Represents a dictionary of TraceType objects.

  Attributes:
    mapping: A mapping from keys to corresponding TraceTypes of the dict values.
  """

  def __init__(self,
               mapping: PythonDict[Hashable, trace.TraceType],
               placeholder_type: Optional[Type[Any]] = None):
    self.mapping = mapping
    self._placeholder_type = placeholder_type

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

    return Dict(new_mapping, self._placeholder_type)

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

  def placeholder_value(self, placeholder_context) -> Any:
    if self._placeholder_type is None:
      raise ValueError("Can not generate placeholder value for Dict with"
                       " unspecified placeholder_type. Note: placeholder_type "
                       "is lost during serialization.")
    attribute_placeholders = [
        (key, value.placeholder_value(placeholder_context))
        for key, value in self.mapping.items()
    ]
    if self._placeholder_type is collections.defaultdict:
      return dict(attribute_placeholders)
    return self._placeholder_type(attribute_placeholders)

  def to_tensors(self, value: Any):
    assert isinstance(value, collections.abc.Mapping)
    flattened_values = []
    for key in sorted(self.mapping.keys()):
      comp_value, comp_type = value[key], self.mapping[key]
      flattened_values.extend(comp_type.to_tensors(comp_value))
    return flattened_values

  def from_tensors(self, tensors):
    if self._placeholder_type is None:
      raise ValueError("Packing serialized Dict is not supported.")

    sorted_traversal = {
        key: self.mapping[key].from_tensors(tensors)
        for key in sorted(self.mapping)
    }

    if self._placeholder_type is collections.defaultdict:
      return {key: sorted_traversal[key] for key in self.mapping}

    return self._placeholder_type(
        (key, sorted_traversal[key]) for key in self.mapping
    )

  def flatten(self) -> PythonList[trace.TraceType]:
    flattened_types = []

    for key in sorted(self.mapping.keys()):
      flattened_types.extend(self.mapping[key].flatten())

    return flattened_types

  def cast(self, value: Any, casting_context) -> Any:
    # Value must have same keys with the TraceType
    assert isinstance(
        value, collections.abc.Mapping
    ), f"Can not cast {value!r} to a Dict type."
    assert set(value.keys()) == set(
        self.mapping.keys()
    ), f"{value!r} has different keys with the TraceType {self!r}."

    casted_values, was_casted = util.cast_and_return_whether_casted(
        self.mapping.values(),
        [value[k] for k in self.mapping.keys()],
        casting_context,
    )

    if was_casted:
      return self._placeholder_type(
          **{k: v for k, v in zip(self.mapping.keys(), casted_values)}
      )
    else:
      return value

  def __eq__(self, other) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not isinstance(other, Dict):
      return False

    return self.mapping == other.mapping

  def __hash__(self) -> int:
    return hash(frozenset(self.mapping.keys()))

  def __repr__(self) -> str:
    paired = [f"[{n!r}, {t!r}]" for n, t in self.mapping.items()]
    return f"{self.__class__.__name__}[{', '.join(paired)}]"


serialization.register_serializable(Literal)
serialization.register_serializable(Tuple)
serialization.register_serializable(List)
serialization.register_serializable(NamedTuple)
serialization.register_serializable(Attrs)
serialization.register_serializable(Dict)
