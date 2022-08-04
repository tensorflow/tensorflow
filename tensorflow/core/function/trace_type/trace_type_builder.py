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
from typing import Any, Callable, Hashable
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

  def __init__(self):
    self._deletion_observer = WeakrefDeletionObserver()
    self._global_to_local_id = {}

  # TODO(b/202772221): Consider dropping after alias pattern matching is
  # supported.
  def make_reference_type(self, base_type: trace.TraceType,
                          local_id: Hashable) -> trace.TraceType:
    if local_id not in self._global_to_local_id:
      self._global_to_local_id[local_id] = len(self._global_to_local_id)

    return default_types.Reference(base_type,
                                   self._global_to_local_id[local_id])

  @property
  def deletion_observer(self):
    """Returns a functor which invalidates the current key when called."""
    return self._deletion_observer


def from_object(obj: Any,
                context: trace.TracingContext = None) -> trace.TraceType:
  """Returns a TraceType corresponding to the object based on the context.

  Args:
    obj: The object to generate a TraceType for.
    context: The TracingContext to be shared during protocol calls.

  Returns:
    A TraceType object representing the given object.
  """

  if context is None:
    context = InternalTracingContext()

  if isinstance(obj, trace.SupportsTracingProtocol):
    return obj.__tf_tracing_type__(context)

  if hasattr(obj, "__wrapped__"):
    return from_object(obj.__wrapped__, context)

  if isinstance(obj, list):
    return default_types.List(*(from_object(c, context) for c in obj))

  if isinstance(obj, tuple):
    if util.is_namedtuple(obj):
      named_tuple_type = type(obj)
      return default_types.NamedTuple.from_type_and_attributes(
          named_tuple_type, tuple(from_object(c, context) for c in obj))
    else:
      return default_types.Tuple(*(from_object(c, context) for c in obj))

  if isinstance(obj, collections.abc.Mapping):
    return default_types.Dict({k: from_object(obj[k], context) for k in obj})

  if util.is_attrs(obj):
    return default_types.Attrs.from_type_and_attributes(
        type(obj),
        tuple(
            from_object(getattr(obj, a.name), context)
            for a in obj.__attrs_attrs__))

  try:
    ref = weakref.ref(obj, context.deletion_observer)
    if ref is None:
      raise TypeError(
          f"Deleted objects are not valid tf.function arguments, Got {obj!r}")
    else:
      return default_types.Weakref(ref)
  except TypeError:
    try:
      return default_types.Literal(obj)
    except:
      raise TypeError(
          f"Python object could not be represented through the generic tracing "
          f"type. Consider implementing the Tracing Protocol for it: {obj!r}")
