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
"""TraceType implementations for classes thatimplement the CustomNestProtocol."""

from typing import Any, Iterator, List as PythonList, Optional, Sequence, Tuple as PythonTuple, Type

from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace
from tensorflow.python.util import custom_nest_protocol


class CustomNestTraceType(trace.TraceType):
  """Represents the TraceType of a class implmenting the CustomNestProtocol."""

  def __init__(
      self,
      value_type: Type[Any],
      metadata: Any,
      components: PythonTuple[trace.TraceType],
  ):
    if not issubclass(value_type, custom_nest_protocol.CustomNestProtocol):
      raise ValueError(f"{value_type!r} does not implement CustomNestProtocol.")
    self.value_type = value_type
    self.metadata = metadata
    self.components = components

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    if not self._is_same_trace_type(other):
      return False
    for c_self, c_other in zip(self.components, other.components):  # pytype: disable=attribute-error
      if not c_self.is_subtype_of(c_other):
        return False
    return True

  def most_specific_common_supertype(
      self, others: Sequence[trace.TraceType]
  ) -> Optional["CustomNestTraceType"]:
    for other in others:
      if not self._is_same_trace_type(other):
        return None

    others_components = [other.components for other in others]  # pytype: disable=attribute-error
    supertyped_components = tuple(
        self_component.most_specific_common_supertype(others_component)
        for self_component, *others_component in zip(
            self.components, *others_components
        )
    )
    return CustomNestTraceType(
        self.value_type, self.metadata, supertyped_components
    )

  def __eq__(self, other: trace.TraceType) -> bool:
    return (
        isinstance(other, CustomNestTraceType)
        and self.value_type == other.value_type
        and self.metadata == other.metadata
        and self.components == other.components
    )

  def __hash__(self) -> int:
    # The hash computation doesn't use self.metadata, so unhashable metadata can
    # be used. The `self.__eq__` method is used instead to differentiate between
    # two objects with the same components but different metadata.
    return hash((self.value_type, self.components))

  def __repr__(self) -> str:
    return (
        f"{self.__class__.__name__} [metadata={self.metadata!r}, "
        f"components={self.components!r}]"
    )

  def placeholder_value(self, placeholder_context: Any) -> Any:
    components_placeholder_value = tuple(
        c.placeholder_value(placeholder_context) for c in self.components
    )
    return self.value_type.__tf_unflatten__(
        self.metadata, components_placeholder_value
    )

  def to_tensors(self, value: Any) -> PythonList[Any]:
    if not isinstance(value, self.value_type):
      raise TypeError(f"{value!r} is not of type {self.value_type}.")
    _, value_components = value.__tf_flatten__()
    flattened_values = []
    for value_comp, type_comp in zip(value_components, self.components):
      flattened_values.extend(type_comp.to_tensors(value_comp))
    return flattened_values

  def from_tensors(self, tensors: Iterator[Any]) -> Any:
    return self.value_type.__tf_unflatten__(
        self.metadata, tuple(c.from_tensors(tensors) for c in self.components)
    )

  def flatten(self) -> PythonList[trace.TraceType]:
    flat_list = []
    for c in self.components:
      flat_list.extend(c.flatten())
    return flat_list

  def cast(self, value: Any, casting_context: Any) -> Any:
    if not isinstance(value, self.value_type):
      raise TypeError(f"[{value!r}] is not of type {self.value_type}.")
    value_metadata, value_components = value.__tf_flatten__()
    if self.metadata != value_metadata:
      raise ValueError(
          f"Metadata mismatch: [{self.metadata!r}] != [{value_metadata!r}]."
      )
    if len(self.components) != len(value_components):
      raise ValueError(
          f"Lengths of components mismatch: {len(self.components)} != "
          f"{len(value_components)}."
      )

    casted_value_components, was_casted = util.cast_and_return_whether_casted(
        self.components, value_components, casting_context
    )
    if was_casted:
      return self.value_type.__tf_unflatten__(
          self.metadata, casted_value_components
      )
    else:
      return value

  def _is_same_trace_type(self, other: trace.TraceType) -> bool:
    return (
        isinstance(other, CustomNestTraceType)
        and self.value_type == other.value_type
        and self.metadata == other.metadata
        and len(self.components) == len(other.components)
    )
