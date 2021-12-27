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

from typing import Dict as PythonDict
from typing import Hashable, Optional, Sequence, Type
from typing import Tuple as PythonTuple

from tensorflow.python.types import trace


class Generic(trace.TraceType):
  """Represents an arbitrary Python object."""

  def __init__(self, obj):
    self._object = obj
    self._object_hash = hash(obj)

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    return self == other

  def most_specific_common_supertype(
      self, types: Sequence[trace.TraceType]) -> Optional[trace.TraceType]:
    if not types:
      raise ValueError(f"`types` must be a non-empty sequence, got{types}")

    return None

  def __eq__(self, other) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    return isinstance(other, Generic) and self._object == other._object

  def __hash__(self) -> int:
    return self._object_hash

  def __repr__(self):
    return f"{self.__class__.__name__}(obj={self._object!r})"


class Weakref(Generic):
  """Represents weakref of an arbitrary Python object.

  When a function argument is a custom class, instead of making a copy of it
  just for the sake of function cache, a weakref is instead kept to save memory.
  """

  def __eq__(self, other):
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not isinstance(other, Weakref):
      return False

    if self._object() is None or other._object() is None:
      return False

    if self._object() is other._object():
      return True

    return self._object == other._object

  def __hash__(self):
    return self._object_hash


class OrderedCollection(trace.TraceType):
  """Represents an ordered collection of TraceType objects.

  Attributes:
    components: A corresponding sequence of TraceTypes to the values in the
      collection.
  """

  def __init__(self, *components: trace.TraceType):
    self.components = components

  def _has_same_structure(self, other):
    if not isinstance(other, type(self)):
      return False

    if len(self.components) != len(other.components):
      return False

    return True

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    """See base class."""
    if not self._has_same_structure(other):
      return False

    if not all([
        component.is_subtype_of(other.components[i])
        for i, component in enumerate(self.components)
    ]):
      return False

    return True

  def most_specific_common_supertype(self, types: Sequence[trace.TraceType]):
    """See base class."""
    if not types:
      raise ValueError(f"`types` must be a non-empty sequence, got{types}")

    if not all(self._has_same_structure(other) for other in types):
      return None

    new_components = []
    for i, component in enumerate(self.components):
      common = component.most_specific_common_supertype(
          [other.components[i] for other in types])
      if common is None:
        return None
      else:
        new_components.append(common)

    return type(self)(*new_components)

  def __eq__(self, other) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not self._has_same_structure(other):
      return False

    return self.components == other.components

  def __hash__(self) -> int:
    return hash(self.components)

  def __repr__(self):
    return f"{self.__class__.__name__}(components={self.components!r})"


class List(OrderedCollection):
  pass


class Tuple(OrderedCollection):
  pass


class Attrs(OrderedCollection):
  """Represents a class annotated by attr.s.

  Each attr.s class has a fixed, ordered set of attributes. Therefore, we only
  need to consider the class type and the underlying attributes. Extra
  metadata including attribute names can be ignored.
  """

  def __init__(self, classtype: Type[object],
               attributes: PythonTuple[trace.TraceType]):
    super().__init__(Generic(classtype), *attributes)


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

  def most_specific_common_supertype(self, types: Sequence[trace.TraceType]):
    """See base class."""

    if not types:
      raise ValueError(f"`types` must be a non-empty sequence, got{types}")

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
