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
"""Polymorphic Type Dispatch."""

import collections
from typing import Optional, Iterable

from tensorflow.python.types import trace

# The maximum number of dispatch lookups to cache.
_MAX_DISPATCH_CACHE = 1024


class TypeDispatchTable:
  """Type dispatch table implementation.

  A type dispatch table is a list, L, of target types. Given a request type, R,
  the table selects a target type, T, according to the following dispatch rules:
    1. R == T or R is subtype of T
    2. There does not exist O in L such that R is subtype of O and O is a
       subtype of T (in other words, T is the closest to R, within list L).
    3. If the above two rules are satisfied by multiple targets, the earliest
       inserted one is chosen.
  """

  def __init__(self):
    """Creates a TypeDispatchTable object."""
    # Holds all inserted types as keys mapping to None.
    # (Using OrderedDict as a set for determinism)
    self._dispatch_table = collections.OrderedDict()

    # LRU cache for dispatch results.
    # Maps request types to target types (see class description).
    # Does not contain exact matches, i.e, if cache[a] is b then a is not b.
    self._dispatch_cache = collections.OrderedDict()

  def add_target(self, target: trace.TraceType) -> None:
    """Adds a new target type."""
    self._dispatch_table[target] = None
    for request in self._dispatch_cache:
      if target.is_subtype_of(self._dispatch_cache[request]):
        self._dispatch_cache[request] = target

  @property
  def targets(self) -> Iterable[trace.TraceType]:
    """Returns an iterable to all targets in the table."""
    return self._dispatch_table.keys()

  def delete(self, target: trace.TraceType) -> None:
    """Deletes a target in the table if it exists."""
    if target in self._dispatch_table:
      del self._dispatch_table[target]
      for request in list(self._dispatch_cache.keys()):
        if self._dispatch_cache[request] == target:
          del self._dispatch_cache[request]

  # TODO(b/205971333): remove once FunctionCache 'clear' is removed.
  def clear(self) -> None:
    """Deletes all targets in the table."""
    self._dispatch_table.clear()
    self._dispatch_cache.clear()

  def dispatch(self, request: trace.TraceType) -> Optional[trace.TraceType]:
    """Returns the deepest subtype target if it exists in the table."""
    # For known exact matches.
    if request in self._dispatch_table:
      return request

    # For known non-exact matches.
    # (self._dispatch cache does not contain exact matches)
    if request in self._dispatch_cache:
      # Move to the front of LRU cache.
      result = self._dispatch_cache.pop(request)
      self._dispatch_cache[request] = result
      return result

    most_specific_subtype = None
    for other in self._dispatch_table:
      if request.is_subtype_of(other):
        if most_specific_subtype is None or other.is_subtype_of(
            most_specific_subtype):
          most_specific_subtype = other

    self._cache_dispatch(request, most_specific_subtype)
    return most_specific_subtype

  def _cache_dispatch(self, request, target):
    """Caches the dispatch lookup result for a target."""
    if target is not None:
      # LRU Cache removes oldest item
      if len(self._dispatch_cache) > _MAX_DISPATCH_CACHE:
        self._dispatch_cache.popitem(last=False)
      self._dispatch_cache[request] = target

  def try_generalizing_trace_type(self,
                                  target: trace.TraceType) -> trace.TraceType:
    """Returns a generalized subtype of the one given.

    This heuristic aims to reduce the number of future traces by computing a
    type that represents more general function inputs.

    The original "experimental_relax_shapes" heuristic identified a known type
    which shared a common supertype with the current unknown type and then
    traced with that common supertype. However, the notion of "common supertype"
    was only limited to shapes. This heuristic extends that to TraceType.

    Returns `target` if a common supertype can not be found.

    Args:
      target: The TraceType to generalize
    """
    relaxed = target
    for other in self._dispatch_table:
      supertype = relaxed.most_specific_common_supertype([other])
      if supertype is not None:
        relaxed = supertype
    return relaxed
