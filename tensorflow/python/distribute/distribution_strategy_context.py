# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Utility to get tf.distribute.Strategy related contexts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export


# There is a circular dependency between this and `distribute` module. So we
# load it lazily to workaround this.
distribute_lib = LazyLoader(
    "distribute_lib", globals(),
    "tensorflow.python.distribute.distribute_lib")

# ------------------------------------------------------------------------------
# Internal API for setting the current thread mode as being either in a
# replica or cross-replica context for a particular tf.distribute.Strategy.


class _ThreadMode(object):

  def __init__(self, dist, cross, replica):
    self.strategy = dist
    self.cross_replica_context = cross
    self.replica_context = replica


class _CrossReplicaThreadMode(_ThreadMode):

  def __init__(self, strategy):
    _ThreadMode.__init__(self, strategy, strategy, None)


class _InReplicaThreadMode(_ThreadMode):

  def __init__(self, replica_ctx):
    _ThreadMode.__init__(self, replica_ctx.strategy, None, replica_ctx)


def _push_per_thread_mode(context):
  ops.get_default_graph()._distribution_strategy_stack.append(context)  # pylint: disable=protected-access


def _pop_per_thread_mode():
  ops.get_default_graph()._distribution_strategy_stack.pop(-1)  # pylint: disable=protected-access


class _DefaultReplicaThreadMode(_ThreadMode):
  """Type of default value returned by `_get_per_thread_mode()`.

  Used when the thread-local stack is empty.
  """

  def __init__(self):
    _ThreadMode.__init__(self, _get_default_strategy(), None,
                         _get_default_replica_context())


def _get_per_thread_mode():
  try:
    return ops.get_default_graph()._distribution_strategy_stack[-1]  # pylint: disable=protected-access
  except (AttributeError, IndexError):
    return _get_default_replica_mode()


# ------------------------------------------------------------------------------
# Public API for accessing the current thread mode


@tf_export("distribute.get_replica_context")
def get_replica_context():
  """Returns the current `tf.distribute.ReplicaContext` or `None`.

  Returns `None` if in a cross-replica context.

  Note that execution:

  1. starts in the default (single-replica) replica context (this function
     will return the default `ReplicaContext` object);
  2. switches to cross-replica context (in which case this will return
     `None`) when entering a `with tf.distribute.Strategy.scope():` block;
  3. switches to a (non-default) replica context inside
     `extended.call_for_each_replica(fn, ...)`;
  4. if `fn` calls `get_replica_context().merge_call(merge_fn, ...)`, then
     inside `merge_fn` you are back in the cross-replica context (and again
     this function will return `None`).

  Note that you can also go directly from step 1 to 4 to switch to a
  cross-replica context for the default `tf.distribute.Strategy`. You may
  also switch from the cross-replica context of 4 to a replica context by
  calling `extended.call_for_each_replica()`, jumping back to step 3.

  Most `tf.distribute.Strategy` methods may only be executed in
  a cross-replica context, in a replica context you should use the
  `ReplicaContext` API instead.

  Returns:
    The current `ReplicaContext` object when in a replica context scope,
    else `None`.

    Within a particular block, exactly one of these two things will be true:

    * `get_replica_context()` returns non-`None`, or
    * `tf.distribute.is_cross_replica_context()` returns True.
  """
  return _get_per_thread_mode().replica_context


def get_cross_replica_context():
  """Returns the current tf.distribute.Strategy if in a cross-replica context.

  DEPRECATED: Please use `in_cross_replica_context()` and
  `get_strategy()` instead.

  Note that execution:

  1. starts in the default (single-replica) replica context;
  2. switches to cross-replica context when entering a
     `with tf.distribute.Strategy.scope():` block;
  3. switches to a (non-default) replica context inside
     `call_for_each_replica(fn, ...)`;
  4. if `fn` calls `get_replica_context()->merge_call(merge_fn, ...)`, then
     inside `merge_fn` you are back in the cross-replica context.

  Note that you can also go directly from step 1 to 4 to switch to a
  cross-replica context for the default `tf.distribute.Strategy`. You may
  also switch from the cross-replica context of 4 to a replica context by
  calling `call_for_each_replica()`, jumping back to step 3.

  Most `tf.distribute.Strategy` methods may only be executed in
  a cross-replica context.

  Returns:
    Returns the current `tf.distribute.Strategy` object in a cross-replica
    context, or `None`.

    Exactly one of `get_replica_context()` and `get_cross_replica_context()`
    will return `None` in a particular block.
  """
  return _get_per_thread_mode().cross_replica_context


@tf_export("distribute.in_cross_replica_context")
def in_cross_replica_context():
  """Returns True if in a cross-replica context.

  See `tf.distribute.get_replica_context` for details.

  Returns:
    True if in a cross-replica context (`get_replica_context()` returns
    `None`), or False if in a replica context (`get_replica_context()` returns
    non-`None`).
  """
  return _get_per_thread_mode().cross_replica_context is not None


@tf_export("distribute.get_strategy")
def get_strategy():
  """Returns the current `tf.distribute.Strategy` object.

  Typically only used in a cross-replica context:

  ```
  if tf.distribute.in_cross_replica_context():
    strategy = tf.distribute.get_strategy()
    ...
  ```

  Returns:
    A `tf.distribute.Strategy` object. Inside a `with strategy.scope()` block,
    it returns `strategy`, otherwise it returns the default (single-replica)
    `tf.distribute.Strategy` object.
  """
  return _get_per_thread_mode().strategy


@tf_export("distribute.has_strategy")
def has_strategy():
  """Return if there is a current non-default `tf.distribute.Strategy`.

  Returns:
    True if inside a `with strategy.scope():`.
  """
  return get_strategy() is not _get_default_strategy()


def get_strategy_and_replica_context():
  per_thread_mode = _get_per_thread_mode()
  return (per_thread_mode.strategy, per_thread_mode.replica_context)


@tf_export("distribute.experimental_set_strategy")
def experimental_set_strategy(strategy):
  """Set a `tf.distribute.Strategy` as current without `with strategy.scope()`.

  ```
  tf.distribute.experimental_set_strategy(strategy1)
  f()
  tf.distribute.experimental_set_strategy(strategy2)
  g()
  tf.distribute.experimental_set_strategy(None)
  h()
  ```

  is equivalent to:

  ```
  with strategy1.scope():
    f()
  with strategy2.scope():
    g()
  h()
  ```

  In general, you should use the `with strategy.scope():` API, but this
  alternative may be convenient in notebooks where you would have to put
  each cell in a `with strategy.scope():` block.

  Note: This should only be called outside of any TensorFlow scope to
  avoid improper nesting.

  Args:
    strategy: A `tf.distribute.Strategy` object or None.

  Raises:
    RuntimeError: If called inside a `with strategy.scope():`.
  """
  old_scope = ops.get_default_graph()._global_distribute_strategy_scope  # pylint: disable=protected-access
  if old_scope is not None:
    old_scope.__exit__(None, None, None)
    ops.get_default_graph()._global_distribute_strategy_scope = None  # pylint: disable=protected-access
  if has_strategy():
    raise RuntimeError(
        "Must not be called inside a `tf.distribute.Strategy` scope.")
  if strategy is not None:
    new_scope = strategy.scope()
    new_scope.__enter__()
    ops.get_default_graph()._global_distribute_strategy_scope = new_scope  # pylint: disable=protected-access


# ------------------------------------------------------------------------------
# Defaults that are used when no tf.distribute.Strategy is explicitly created.
# We create them lazily in a function so that we can workaround the circular
# dependency on distribute_lib. See lazy loader at the top of this file.

_defaults = {
    "strategy": None,
    "replica_context": None,
    "replica_mode": None
}


def _get_default_strategy():
  if _defaults["strategy"] is None:
    _defaults["strategy"] = distribute_lib._DefaultDistributionStrategy()  # pylint: disable=protected-access
  return _defaults["strategy"]


def _get_default_replica_context():
  if _defaults["replica_context"] is None:
    _defaults["replica_context"] = distribute_lib.ReplicaContext(
        _get_default_strategy(), replica_id_in_sync_group=0)
  return _defaults["replica_context"]


def _get_default_replica_mode():
  if _defaults["replica_mode"] is None:
    _defaults["replica_mode"] = _DefaultReplicaThreadMode()
  return _defaults["replica_mode"]


# Aliases for compatibility with old names.
get_distribution_strategy = get_strategy
has_distribution_strategy = has_strategy
