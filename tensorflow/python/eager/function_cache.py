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
from typing import Sequence, Optional

from tensorflow.python.eager import context
from tensorflow.python.eager import function_trace_type
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.saved_model import save_context
from tensorflow.python.types import trace
from tensorflow.python.util import memory

# A temporary flag. Turning this on will allow tf.function to aggressively avoid
# retracing ResourceVariable inputs. This feature will change tf.function's
# Variable tracing behavior, hence we want to limit the potential blockers that
# are not detected by Global TAP.
# TODO(jiaweix): remove this flag and related args (b/198782192)
_ENCODE_VARIABLES_BY_RESOURCE_ID = True
# TODO(b/201533914): Remove this flag and related args
USE_FULL_TRACE_TYPE = True

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
    arg_spec: A TraceType corresponding to the function arguments.
    call_context: The ExecutionContext for when the arg_spec was generated.
  """

  def __init__(self, arg_spec: trace.TraceType, call_context: ExecutionContext):
    self.arg_spec = arg_spec
    self.call_context = call_context

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    if not isinstance(other, FunctionCacheKey):
      return False

    if self.call_context != other.call_context:
      return False

    # Functions are contravariant.
    return other.arg_spec.is_subtype_of(self.arg_spec)

  def most_specific_common_supertype(
      self, others: Sequence[trace.TraceType]) -> Optional["FunctionCacheKey"]:
    raise NotImplementedError(
        "Requires TraceType to include most_specific_common_subtype")

  def most_specific_common_subtype(
      self, others: Sequence[trace.TraceType]) -> Optional["FunctionCacheKey"]:
    if not all(
        isinstance(other, FunctionCacheKey) and
        self.call_context == other.call_context for other in others):
      return None

    common = self.arg_spec.most_specific_common_supertype(
        [other.arg_spec for other in others])

    if common is None:
      return None

    return FunctionCacheKey(common, self.call_context)

  def __hash__(self) -> int:
    return hash((self.call_context, self.arg_spec))

  def __eq__(self, other) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not isinstance(other, FunctionCacheKey):
      return False

    return (self.call_context == other.call_context and
            self.arg_spec == other.arg_spec)

  def __repr__(self) -> str:
    return (f"{type(self).__name__}(arg_spec={repr(self.arg_spec)},"
            f" call_context={repr(self.call_context)})")


class FunctionCache:
  """A container for managing concrete functions."""

  __slots__ = [
      "_missed", "_primary", "_dispatch_cache", "arg_relaxed_specs",
      "arg_relaxed", "_garbage_collectors"
  ]

  def __init__(self):
    # The set of functions that have been missed; entries are ExecutionContext.
    self._missed = set()
    # The primary cache, mapping FunctionCacheKey to a concrete function.
    self._primary = collections.OrderedDict()

    # Maps a FunctionCacheKey K to a FunctionCacheKey V such that it is safe
    # to dispatch K to the concrete function of V that exists in _primary.
    # Used to lookup posible concrete functions when K is not in _primary.
    self._dispatch_cache = collections.OrderedDict()

    # TODO(b/202430155): Incorporate relaxation logic inside FunctionCache.
    # A cache key lookup, mapping a cache key generated without shape info to a
    # flat list of `TypeSpec`s with relaxed shapes (one for each flattened
    # argument). Arguments that are not Tensors or `CompositeTensor`s contain a
    # `None` for the corresponding relaxed spec.
    self.arg_relaxed_specs = collections.OrderedDict()
    # The secondary cache, mapping a cache key generated without shape info to a
    # function.
    self.arg_relaxed = collections.OrderedDict()
    # All OrderedDicts require manual garbage collection.

    self._garbage_collectors = [
        _FunctionGarbageCollector(self._primary),
        _FunctionGarbageCollector(self._dispatch_cache),
        _FunctionGarbageCollector(self.arg_relaxed),
        _FunctionGarbageCollector(self.arg_relaxed_specs)]

  # Note: Instead of returning any viable function, we can return the most
  # specfic one by maintaining trees of traces where children are more specific
  # traces of their parents.
  def lookup(self, key: FunctionCacheKey, use_function_subtyping: bool):
    """Looks up a concrete function based on the key."""
    if not use_function_subtyping:
      return self._primary.get(key, None)

    if key in self._primary:
      return self._primary[key]

    if key in self._dispatch_cache:
      return self._primary[self._dispatch_cache[key]]

    for known_key in self._primary:
      if known_key.is_subtype_of(key):
        self._dispatch_cache[key] = known_key
        return self._primary[known_key]

    return None

  # NOTE: We can optimize deletion to O(1) by maintaining a reverse dict of
  # self._dispatch_cache.
  def delete(self, key: FunctionCacheKey):
    """Deletes a concrete function given the key it was added with."""
    del self._primary[key]

    for dispatched_key in self._dispatch_cache:
      if self._dispatch_cache[dispatched_key] == key:
        del self._dispatch_cache[dispatched_key]

  def add(self, key: FunctionCacheKey, concrete):
    """Adds a new concrete function alongside its key."""
    self._primary[key] = concrete

  def clear(self):
    """Removes all concrete functions from the cache."""
    self._primary.clear()
    self._dispatch_cache.clear()
    self.arg_relaxed_specs.clear()
    self.arg_relaxed.clear()

  def values(self):
    """Returns a list of all `ConcreteFunction` instances held by this cache."""
    # We need to simultaneously make sure our returned concrete functions are
    # unique *and* make sure they are returned in a deterministic order for
    # serialization.
    #
    # TODO(b/174215821): It's likely that we ultimately would just prefer to
    # choose the most specific concrete function shape given a set of
    # arguments. If and when that is implemented, this logic can be revisited.
    primary_functions = set(self._primary.values())
    return list(self._primary.values()) + [
        v for v in self.arg_relaxed.values() if v not in primary_functions]

  def has_call_context(self, call_context: ExecutionContext) -> bool:
    """Checks if an ExcutionContext was observed."""
    return call_context in self._missed

  def add_call_context(self, call_context: ExecutionContext) -> None:
    """Adds a new ExcutionContext observation."""
    self._missed.add(call_context)


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


def make_cache_key(args,
                   include_tensor_ranks_only: bool = False) -> FunctionCacheKey:
  """Computes the cache key given the function arguments."""
  arg_spec = function_trace_type.get_arg_spec(
      args, include_tensor_ranks_only, _ENCODE_VARIABLES_BY_RESOURCE_ID,
      USE_FULL_TRACE_TYPE)
  return FunctionCacheKey(arg_spec, _make_execution_context())


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
      strategy_stack[-1].strategy.extended._retrace_functions_for_each_device
  )
  if executing_eagerly:
    colocation_stack = ()
    if uses_distribution_strategy:
      device_functions = (pydev.merge_device(ctx.device_name),)
    else:
      device_functions = ()
  else:
    colocation_stack = tuple(default_graph._colocation_stack.peek_objs())
    if (uses_distribution_strategy
        or func_graph_module.device_stack_has_callable(
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
