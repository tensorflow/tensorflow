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
from typing import Any, NamedTuple, Optional

from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.function.polymorphism import type_dispatch


class FunctionContext(NamedTuple):
  """Contains information regarding tf.function execution context."""
  context: Any


class FunctionCache:
  """A container for managing concrete functions."""

  __slots__ = ["_primary", "_dispatch_dict", "_garbage_collectors"]

  def __init__(self):
    # Maps (FunctionContext, FunctionType) to a concrete function.
    self._primary = collections.OrderedDict()

    # Maps FunctionContext to a TypeDispatchTable containing FunctionTypes of
    # that particular context.
    self._dispatch_dict = {}

  def lookup(self, context: FunctionContext,
             function_type: function_type_lib.FunctionType) -> Optional[Any]:
    """Looks up a concrete function based on the context and type."""
    if context in self._dispatch_dict:
      dispatch_type = self._dispatch_dict[context].dispatch(function_type)
      if dispatch_type:
        return self._primary[(context, dispatch_type)]

    return None

  def delete(self, context: FunctionContext,
             function_type: function_type_lib.FunctionType) -> bool:
    """Deletes a concrete function given the context and type."""
    if (context, function_type) not in self._primary:
      return False

    del self._primary[(context, function_type)]
    self._dispatch_dict[context].delete(function_type)

    return True

  def add(self, context: FunctionContext,
          function_type: function_type_lib.FunctionType,
          concrete_fn: Any):
    """Adds a new concrete function alongside its key.

    Args:
      context: A FunctionContext representing the current context.
      function_type: A FunctionType representing concrete_fn signature.
      concrete_fn: The concrete function to be added to the cache.
    """
    self._primary[(context, function_type)] = concrete_fn
    if context not in self._dispatch_dict:
      self._dispatch_dict[context] = type_dispatch.TypeDispatchTable()

    self._dispatch_dict[context].add_target(function_type)

  def generalize(
      self, context: FunctionContext,
      function_type: function_type_lib.FunctionType
  ) -> function_type_lib.FunctionType:
    """Try to generalize a FunctionType within a FunctionContext."""
    if context in self._dispatch_dict:
      return self._dispatch_dict[context].try_generalizing_function_type(
          function_type)
    else:
      return function_type

  # TODO(b/205971333): Remove this function.
  def clear(self):
    """Removes all concrete functions from the cache."""
    self._primary.clear()
    self._dispatch_dict.clear()

  def values(self):
    """Returns a list of all `ConcreteFunction` instances held by this cache."""
    return list(self._primary.values())
