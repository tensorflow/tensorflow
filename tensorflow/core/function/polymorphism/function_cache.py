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
"""Cache to manage concrete functions and their signatures."""

import collections
from typing import Optional, Sequence, Any, NamedTuple

from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import type_dispatch
from tensorflow.python.types import trace

# TODO(b/182990542): Enable and remove flag when stable.
DELETE_WITH_WEAKREF = False


class FunctionContext(NamedTuple):
  """Contains information regarding tf.function execution context."""
  context: Any


class FunctionCacheKey(trace.TraceType):
  """The unique key associated with a concrete function.

  Attributes:
    function_signature: A TraceType corresponding to the function arguments.
    call_context: The FunctionContext for when the function_signature was
      generated.
  """

  def __init__(self, function_signature: trace.TraceType,
               call_context: FunctionContext):
    self.function_signature = function_signature
    self.call_context = call_context

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    if not isinstance(other, FunctionCacheKey):
      return False

    if self.call_context != other.call_context:
      return False

    return self.function_signature.is_subtype_of(other.function_signature)

  def most_specific_common_supertype(
      self, others: Sequence[trace.TraceType]) -> Optional["FunctionCacheKey"]:
    if not all(
        isinstance(other, FunctionCacheKey) and
        self.call_context == other.call_context for other in others):
      return None

    common = self.function_signature.most_specific_common_supertype(
        [other.function_signature for other in others])

    if common is None:
      return None

    return FunctionCacheKey(common, self.call_context)

  def _placeholder_value(self) -> Any:
    """Value used for tracing a function signature with this TraceType."""
    return self.function_signature._placeholder_value()  # pylint: disable=protected-access

  def __hash__(self) -> int:
    return hash((self.call_context, self.function_signature))

  def __eq__(self, other) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not isinstance(other, FunctionCacheKey):
      return False

    return (self.call_context == other.call_context and
            self.function_signature == other.function_signature)

  def __repr__(self) -> str:
    return (
        f"{type(self).__name__}(function_signature={repr(self.function_signature)},"
        f" call_context={repr(self.call_context)})")


# TODO(fmuham): Rename to FunctionLibrary.
class FunctionCache:
  """A container for managing concrete functions."""

  __slots__ = [
      "_primary", "_dispatch_table", "_garbage_collectors"
  ]

  def __init__(self):
    # The primary cache, mapping FunctionCacheKey to a concrete function.
    self._primary = collections.OrderedDict()

    # Maps a FunctionCacheKey K to a FunctionCacheKey V such that it is safe
    # to dispatch K to the concrete function of V that exists in _primary.
    # Used to lookup posible concrete functions when K is not in _primary.
    self._dispatch_table = type_dispatch.TypeDispatchTable()

  # Note: Instead of returning any viable function, we can return the most
  # specfic one by maintaining trees of traces where children are more specific
  # traces of their parents.
  def lookup(self, key: FunctionCacheKey, use_function_subtyping: bool):
    """Looks up a concrete function based on the key."""
    if not use_function_subtyping:
      return self._primary.get(key, None)

    dispatch_key = self._dispatch_table.dispatch(key)
    if dispatch_key is not None:
      return self._primary[dispatch_key]

    return None

  def delete(self, key: FunctionCacheKey):
    """Deletes a concrete function given the key it was added with."""
    if key not in self._primary:
      return False

    del self._primary[key]
    self._dispatch_table.delete(key)

    return True

  def add(self, key: FunctionCacheKey,
          deletion_observer: trace_type.WeakrefDeletionObserver,
          concrete):
    """Adds a new concrete function alongside its key.

    Args:
      key: A FunctionCacheKey object corresponding to the provided `concrete`.
      deletion_observer: A WeakrefDeletionObserver object for the `key`.
      concrete: The concrete function to be added to the cache.
    """
    self._primary[key] = concrete
    self._dispatch_table.add_target(key)
    deletion_observer.add_listener(
        lambda: self.delete(key) if DELETE_WITH_WEAKREF else None)

  def generalize(self, key: FunctionCacheKey) -> FunctionCacheKey:
    return self._dispatch_table.try_generalizing_trace_type(key)  # pylint: disable=protected-access

  # TODO(b/205971333): Remove this function.
  def clear(self):
    """Removes all concrete functions from the cache."""
    self._primary.clear()
    self._dispatch_table.clear()

  def values(self):
    """Returns a list of all `ConcreteFunction` instances held by this cache."""
    return list(self._primary.values())
