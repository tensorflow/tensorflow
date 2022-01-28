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
from typing import Optional, List

from tensorflow.python.types import trace

# The maximum number of dispatch lookups to cache.
_MAX_DISPATCH_CACHE = 1024


class TypeDispatchTable:
  """Type dispatch table implementation.

  A type dispatch table is a list, L, of target types. Given an input type, I,
  the table selects a target type, T, according to the following dispatch rules:
    1. I == T or I is more specific than T
    2. ∄O∈L such that I is more specific than O and O is more specific than T
    3. If the above two rules are satisfied by multiple targets, the earliest
       inserted one is chosen.

  This table is intended for use with function signatures and so it assumes that
  the types are contravariant. In other words, a supertype is more specific and
  hence it is acceptable to use a subtype where a supertype is specified.
  """

  def __init__(self):
    """Creates a TypeDispatchTable object."""
    self._is_more_specific_fn = lambda a, b: a != b and b.is_subtype_of(a)

    # Holds all inserted types as keys. (Using OrderedDict for determinism)
    self._targets = collections.OrderedDict()

    # LRU Cache for dispatch results to avoid expensive lookups.
    self._dispatch_cache = collections.OrderedDict()

  def add_target(self, target: trace.TraceType) -> None:
    """Adds a new target type."""
    self._targets[target] = None
    self._dispatch_cache.clear()

  def all_targets(self) -> List[trace.TraceType]:
    """Returns all targets in the table."""
    return list(self._targets.keys())

  def delete(self, target: trace.TraceType) -> None:
    """Deletes a target in the table if it exists."""
    if target in self._targets:
      del self._targets[target]
      self._dispatch_cache.clear()

  def clear(self) -> None:
    """Deletes all targets in the table."""
    self._targets.clear()
    self._dispatch_cache.clear()

  def contains(self, target: trace.TraceType) -> bool:
    """Returns True if the exact target exists in the table."""
    return target in self._targets

  def dispatch(self, target: trace.TraceType) -> Optional[trace.TraceType]:
    """Returns the most specific target if it exists in the table."""
    if target in self._targets:
      return target

    if target in self._dispatch_cache:
      # Move to the front of LRU cache.
      result = self._dispatch_cache.pop(target)
      self._dispatch_cache[target] = result
      return result

    most_specific = None
    for other in self._targets:
      if self._is_more_specific_fn(target, other):
        if most_specific is None or self._is_more_specific_fn(
            other, most_specific):
          most_specific = other

    if most_specific is not None:
      # LRU Cache removes oldest item
      if len(self._dispatch_cache) > _MAX_DISPATCH_CACHE:
        self._dispatch_cache.popitem(last=False)

      self._dispatch_cache[target] = most_specific

    return most_specific

