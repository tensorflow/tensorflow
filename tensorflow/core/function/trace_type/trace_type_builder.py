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
"""Utitiles for Cache Key generation based on Function Trace Type."""

import collections.abc
from typing import Any, Callable, Hashable, Optional, Dict
import weakref

from tensorflow.core.function.trace_type import default_types
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace


class WeakrefDeletionObserver:
  """An observer for the event of deleting a weakref.

  This allows users of FunctionTraceType to be notified when an instance which
  depends on a weakref becomes invalid by the deletion of the weakref. In
  particular, tf.function caches can use this mechanism to clear the cache of
  keys that are no longer valid.

  We use the observer pattern and not just basic callbacks because the keys
  are typically created before they are used by the cache.
  """

  def __init__(self):
    self._triggered = False
    self._callables = []

  def add_listener(self, on_delete: Callable[[], None]):
    if self._triggered:
      on_delete()
    else:
      self._callables.append(on_delete)

  def weakref_deleted(self):
    self._triggered = True
    for c in self._callables:
      c()

  def __call__(self, _):
    """Call handler for convenience of use with weakref."""
    self.weakref_deleted()


class InternalTracingContext(trace.TracingContext):
  """Container for variables and flags shared across TraceType generation."""

  def __init__(self, is_legacy_signature: bool = False):
    self._deletion_observer = WeakrefDeletionObserver()
    self._global_to_local_id = {}
    self._alias_id_to_placeholder = {}
    self._spec_id_to_handledata = {}
    self._is_legacy_signature = is_legacy_signature

  def alias_global_id(self, global_id: Hashable) -> Hashable:
    if global_id not in self._global_to_local_id:
      self._global_to_local_id[global_id] = len(self._global_to_local_id)

    return self._global_to_local_id[global_id]

  def add_placeholder(self, alias_id: Hashable, variable) -> None:
    self._alias_id_to_placeholder[alias_id] = variable

  def get_placeholder_mapping(self) -> Dict[Hashable, Any]:
    return self._alias_id_to_placeholder

  def add_handledata(self, spec_id: Hashable, handledata: Any) -> None:
    self._spec_id_to_handledata[spec_id] = handledata

  def get_handledata_mapping(self) -> Dict[Hashable, Any]:
    return self._spec_id_to_handledata

  @property
  def deletion_observer(self) -> WeakrefDeletionObserver:
    """Returns a functor which invalidates the current key when called."""
    return self._deletion_observer

  @property
  def is_legacy_signature(self) -> bool:
    """If the value is from a legacy signature representation.

    Legacy signature representations include tf.function.input_signature and
    ConcreteFunction.structured_input_signature.
    """
    return self._is_legacy_signature


class InternalPlaceholderContext(trace.PlaceholderContext):
  """Container with mappings shared across TraceTypes for placeholder values."""

  def __init__(self,
               context_graph=None,
               placeholder_mapping=None,
               handledata_mapping=None,
               unnest_only=False):
    self._alias_id_to_placeholder = placeholder_mapping or {}
    self._spec_id_to_handledata = handledata_mapping or {}
    self._naming_scope = None
    self._context_graph = context_graph
    self._unnest_only = unnest_only

  def has_placeholder(self, alias_id: Hashable) -> bool:
    return alias_id in self._alias_id_to_placeholder

  def get_placeholder(self, alias_id: Hashable) -> Hashable:
    if not self.has_placeholder(alias_id):
      raise KeyError(f"alias_id: {alias_id} not found in this instance of "
                     "placeholder context.")
    return self._alias_id_to_placeholder[alias_id]

  def add_placeholder(self, alias_id: Hashable, placeholder: Hashable) -> None:
    if alias_id in self._alias_id_to_placeholder:
      raise KeyError(f"alias id: {alias_id} is already stored in this "
                     "instance of placeholder context.")
    self._alias_id_to_placeholder[alias_id] = placeholder

  def has_handledata(self, spec_id: Hashable) -> bool:
    return spec_id in self._spec_id_to_handledata

  def get_handledata(self, spec_id: Hashable) -> Any:
    if not self.has_handledata(spec_id):
      raise KeyError("Could not find handle data for TraceType with "
                     f"id: {spec_id} in this instance of placeholder context.")
    return self._spec_id_to_handledata[spec_id]

  def update_naming_scope(self, naming_scope: Optional[str]) -> None:
    self._naming_scope = naming_scope

  @property
  def naming_scope(self) -> Optional[str]:
    return self._naming_scope

  @property
  def context_graph(self):
    return self._context_graph

  @property
  def unnest_only(self) -> bool:
    return self._unnest_only


class InternalCastContext(trace.CastContext):
  """Default casting behaviors."""


def from_value(value: Any,
               context: trace.TracingContext = None) -> trace.TraceType:
  """Returns a TraceType corresponding to the value based on the context.

  Args:
    value: The value to generate a TraceType for.
    context: The TracingContext to be shared during protocol calls.

  Returns:
    A TraceType object representing the given value.
  """

  if context is None:
    context = InternalTracingContext()

  if context.is_legacy_signature and isinstance(value, trace.TraceType):
    return value
  elif isinstance(value, trace.SupportsTracingProtocol):
    generated_type = value.__tf_tracing_type__(context)
    if not isinstance(generated_type, trace.TraceType):
      raise TypeError(
          "Expected an instance of TraceType for Tracing Protocol call to " +
          str(value) + " but got " + str(generated_type))
    return generated_type

  if hasattr(value, "__wrapped__"):
    return from_value(value.__wrapped__, context)

  if isinstance(value, list):
    return default_types.List(*(from_value(c, context) for c in value))

  if isinstance(value, tuple):
    if util.is_namedtuple(value):
      named_tuple_type = type(value)
      return default_types.NamedTuple.from_type_and_attributes(
          named_tuple_type, tuple(from_value(c, context) for c in value))
    else:
      return default_types.Tuple(*(from_value(c, context) for c in value))

  if isinstance(value, collections.abc.Mapping):
    mapping_type = type(value)
    return default_types.Dict(
        {k: from_value(value[k], context) for k in value}, mapping_type)

  if util.is_attrs(value):
    return default_types.Attrs.from_type_and_attributes(
        type(value),
        tuple(
            from_value(getattr(value, a.name), context)
            for a in value.__attrs_attrs__))

  try:
    ref = weakref.ref(value, context.deletion_observer)
    if ref is None:
      raise TypeError(
          f"Deleted objects are not valid tf.function arguments, Got {value!r}")
    else:
      return default_types.Weakref(ref)
  except TypeError:
    try:
      return default_types.Literal(value)
    except:
      raise TypeError(  # pylint: disable=raise-missing-from
          f"Could not generate a generic TraceType for {value!r}."
          f"Please verify that it is immutable/hashable. Otheriwse, consider "
          f"implementing the Tracing Protocol for it.")
