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
from typing import Optional, Sequence, Tuple, Any

from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import type_dispatch
from tensorflow.python.eager import context
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.saved_model import save_context
from tensorflow.python.types import trace
from tensorflow.python.util import memory

# TODO(b/182990542): Enable and remove flag when stable.
DELETE_WITH_WEAKREF = False

ExecutionContext = collections.namedtuple("ExecutionContext", [
    "parent_graph",
    "device_functions",
    "colocation_stack",
    "in_cross_replica_context",
    "variable_policy",
    "xla_context_id",
])


class FunctionCacheKey(trace.TraceType):
  """The unique key associated with a concrete function.

  Attributes:
    function_signature: A TraceType corresponding to the function arguments.
    call_context: The ExecutionContext for when the function_signature was
      generated.
  """

  def __init__(self, function_signature: trace.TraceType,
               call_context: ExecutionContext):
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

    self._garbage_collectors = [
        _FunctionGarbageCollector(self._primary),
    ]

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


class _FunctionGarbageCollector(object):
  """Cleans up cycles when a defun goes out of scope."""

  __slots__ = ["_cache"]

  def __init__(self, cache):
    self._cache = cache

  def __del__(self):
    if func_graph_module is None or memory is None:
      return
    try:
      while self._cache:
        self._cache.popitem()
      memory.dismantle_ordered_dict(self._cache)
    except:  # pylint: disable=bare-except
      pass


def make_cache_key(
    args,
    include_tensor_ranks_only: bool = False
) -> Tuple[FunctionCacheKey, trace_type.WeakrefDeletionObserver]:
  """Computes the cache key given the function arguments."""
  signature_context = trace_type.SignatureContext(
      include_tensor_ranks_only)
  function_signature = trace_type.make_function_signature(
      args, signature_context)
  return FunctionCacheKey(
      function_signature,
      _make_execution_context()), signature_context.deletion_observer


def _make_execution_context() -> ExecutionContext:
  """Generates an ExecutionContext based on current contextual info."""
  ctx = context.context()

  # Don't need to open an init_scope if the _cache_key call is in eager mode
  # already.
  executing_eagerly = ctx.executing_eagerly()
  parent_graph = None
  xla_context_id = 0
  if not executing_eagerly:
    # We want to force function retracing for each different
    # XLAControlFlowContext, so add `xla_context_id` to the cache key.
    xla_context = _enclosing_xla_context()
    if xla_context is not None and xla_context.RequiresUniqueFunctionRetracing(
    ):
      xla_context_id = id(xla_context)

    with ops.init_scope():
      # The graph, or whether we're executing eagerly, should be a part of the
      # cache key so we don't improperly capture tensors such as variables.
      executing_eagerly = ctx.executing_eagerly()
      parent_graph = None if executing_eagerly else ops.get_default_graph()

  # pylint: disable=protected-access
  default_graph = ops.get_default_graph()
  # TODO(b/117617952): The current distribution strategy will affect graph
  # building (e.g. accessing different variables from different devices) and
  # so requires retracing for each device.
  strategy_stack = default_graph._distribution_strategy_stack
  uses_distribution_strategy = (
      strategy_stack and
      strategy_stack[-1].strategy.extended._retrace_functions_for_each_device)
  if executing_eagerly:
    colocation_stack = ()
    if uses_distribution_strategy:
      device_functions = (pydev.merge_device(ctx.device_name),)
    else:
      device_functions = ()
  else:
    colocation_stack = tuple(default_graph._colocation_stack.peek_objs())
    if (uses_distribution_strategy or
        func_graph_module.device_stack_has_callable(
            default_graph._device_function_stack)):
      # Putting the device in the cache key ensures that call-site device
      # annotations are respected.
      device_functions = tuple(default_graph._device_functions_outer_to_inner)
    else:
      device_functions = ()

  in_cross_replica_context = False
  try:
    in_cross_replica_context = (strategy_stack[-1].replica_context is None)  # pylint: disable=protected-access
  except (AttributeError, IndexError):
    pass

  if save_context.in_save_context():
    variable_policy = (
        save_context.get_save_options().experimental_variable_policy)
  else:
    variable_policy = None

  return ExecutionContext(parent_graph, device_functions, colocation_stack,
                          in_cross_replica_context, variable_policy,
                          xla_context_id)


def _enclosing_xla_context():
  """Returns the XLAControlFlowContext, which exists inside a tpu.rewrite()."""
  graph = ops.get_default_graph()
  while graph is not None:
    # pylint: disable=protected-access
    context_ = graph._get_control_flow_context()
    # pylint: enable=protected-access
    while context_ is not None:
      if isinstance(context_, control_flow_ops.XLAControlFlowContext):
        return context_
      context_ = context_.outer_context
    # This may be a FuncGraph due to defuns or v2 control flow. We need to
    # find the original graph with the XLAControlFlowContext.
    graph = getattr(graph, "outer_graph", None)
  return None
