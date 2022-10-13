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
from typing import Any, Dict, Hashable, NamedTuple, Optional, Sequence

from tensorflow.core.function import trace_type
from tensorflow.core.function.function_type import function_type as function_type_lib
from tensorflow.core.function.polymorphism import type_dispatch
from tensorflow.python.types import trace

# TODO(b/182990542): Enable and remove flag when stable.
DELETE_WITH_WEAKREF = False


class FunctionContext(NamedTuple):
  """Contains information regarding tf.function execution context."""
  context: Any


# TODO(fmuham): Move into FunctionType.
class CaptureSnapshot(trace.TraceType):
  """Store tf.function captures to accommodate its specific tracing logic.

  Captures are stored in mapping format, but its tracing logic is different from
  Python dict. When comparing types of two normal Python dicts in function
  argumenst, their keys are required to be the same. When comparing types for
  captures, keys can be different. This is because tf.function maintains a full
  list of captures and only a subset is active for each ConcreteFunction.
  But before dispatch, which captures are active is unknown, so all caputres are
  evaluated for comparison. Please also check `is_subtype_of` method.

  Attributes:
    mapping: A mapping from keys to corresponding TraceTypes of the dict values.
  """

  def __init__(self, mapping: Dict[Hashable, trace.TraceType]):
    self.mapping = mapping

  def _contain_all_keys_of(self, other):
    for key in other.mapping:
      if key not in self.mapping:
        return False
    return True

  def is_subtype_of(self, query: "CaptureSnapshot") -> bool:
    """This method is used to check if `self` is a subtype of query.

    Typically, self represents an existing snapshot for a ConcreteFunction, and
    the query is a snapshot from all captures with runtime values. Keys in the
    query should be a superset of self.
    This method differs from default_types.Dict as this CaptureSnapshot doesn't
    require a full match of keys.

      For example:

      a = CaptureSnapshot({'x'=1, 'y'=2})
      b = CaptureSnapshot({'x'=1, 'y'=2, 'z'=3})
      assert not a.is_subtype_of(b)
      assert b.is_subtype_of(a)

    Args:
      query: A CaptureSnapshot instance that represents the current runtime
        values of all captures.

    Returns:
      A bool value represents the result.
    """
    if not isinstance(query, CaptureSnapshot):
      return False

    if not self._contain_all_keys_of(query):
      return False
    return all(self.mapping[key].is_subtype_of(item)
               for key, item in query.mapping.items())

  def most_specific_common_supertype(
      self, types: Sequence[trace.TraceType]) -> Optional["CaptureSnapshot"]:
    """See base class."""
    common_keys = set(self.mapping.keys())
    for other in types:
      common_keys = common_keys.intersection(other.mapping.keys())
    new_mapping = {}
    for key in common_keys:
      common = self.mapping[key].most_specific_common_supertype(
          [other.mapping[key] for other in types])
      if common is None:
        return None
      else:
        new_mapping[key] = common
    return CaptureSnapshot(new_mapping)

  def _placeholder_value(self) -> Any:
    return {
        key: value._placeholder_value()  # pylint: disable=protected-access
        for key, value in self.mapping.items()
    }

  def __eq__(self, other: "CaptureSnapshot") -> bool:
    if not isinstance(other, CaptureSnapshot):
      return False

    return self.mapping == other.mapping

  def __hash__(self) -> int:
    return hash(frozenset(self.mapping.keys()))


# TODO(fmuham): Remove inheritance from TraceType.
class FunctionCacheKey(trace.TraceType):
  """The unique key associated with a concrete function.

  Attributes:
    function_type: A FunctionType corresponding to the function arguments.
    captures_signature: A CaptureSnapshot corresponding to the function
      captures.
    call_context: The FunctionContext for when the function was called.
  """

  def __init__(self, function_type: function_type_lib.FunctionType,
               captures_signature: CaptureSnapshot,
               call_context: FunctionContext):
    self.function_type = function_type
    self.captures_signature = captures_signature
    self.call_context = call_context

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    if not isinstance(other, FunctionCacheKey):
      return False

    if self.call_context != other.call_context:
      return False

    return (self.function_type.is_supertype_of(other.function_type) and
            self.captures_signature.is_subtype_of(other.captures_signature))

  def most_specific_common_supertype(
      self, others: Sequence[trace.TraceType]) -> Optional["FunctionCacheKey"]:
    if not all(
        isinstance(other, FunctionCacheKey) and
        self.call_context == other.call_context for other in others):
      return None

    function_type_common = self.function_type.most_specific_common_subtype(
        [other.function_type for other in others])

    if function_type_common is None:
      return None

    captures_common = self.captures_signature.most_specific_common_supertype(
        [other.captures_signature for other in others])

    return FunctionCacheKey(function_type_common, captures_common,
                            self.call_context)

  def _placeholder_value(self) -> Any:
    """Value used for tracing a function signature with this TraceType."""
    return {
        "args": self.function_type.placeholder_arguments().args[0],
        "captures": self.captures_signature._placeholder_value()  # pylint: disable=protected-access
    }

  def __hash__(self) -> int:
    return hash(
        (self.call_context, self.function_type, self.captures_signature))

  def __eq__(self, other) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not isinstance(other, FunctionCacheKey):
      return False

    return (self.call_context == other.call_context and
            self.function_type == other.function_type and
            self.captures_signature == other.captures_signature)

  def __repr__(self) -> str:
    return (f"{type(self).__name__}(function_type={repr(self.function_type)},"
            f"(captures_signature={repr(self.captures_signature)},"
            f" call_context={repr(self.call_context)})")


# TODO(fmuham): Rename to FunctionLibrary.
class FunctionCache:
  """A container for managing concrete functions."""

  __slots__ = ["_primary", "_dispatch_table", "_garbage_collectors"]

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
          deletion_observer: trace_type.WeakrefDeletionObserver, concrete: ...):
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
