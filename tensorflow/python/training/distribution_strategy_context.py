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
"""Utility to get distribution strategy related contexts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.util.lazy_loader import LazyLoader


# There is a circular dependency between this and `distribute` module. So we
# load it lazily to workaround this.
distribute_lib = LazyLoader(
    "distribute_lib", globals(),
    "tensorflow.python.training.distribute")

# ------------------------------------------------------------------------------
# Internal API for setting the current thread mode as being either in a
# tower or cross-tower context for a particular distribution strategy.


class _ThreadMode(object):

  def __init__(self, dist, cross, tower):
    self.distribution_strategy = dist
    self.cross_tower_context = cross
    self.tower_context = tower


class _CrossTowerThreadMode(_ThreadMode):

  def __init__(self, distribution_strategy):
    _ThreadMode.__init__(
        self, distribution_strategy, distribution_strategy, None)


class _InTowerThreadMode(_ThreadMode):

  def __init__(self, tower_ctx):
    _ThreadMode.__init__(
        self, tower_ctx.distribution_strategy, None, tower_ctx)


def _push_per_thread_mode(context):
  ops.get_default_graph()._distribution_strategy_stack.append(context)  # pylint: disable=protected-access


def _pop_per_thread_mode():
  ops.get_default_graph()._distribution_strategy_stack.pop(-1)  # pylint: disable=protected-access


class _DefaultTowerThreadMode(_ThreadMode):
  """Type of default value returned by `_get_per_thread_mode()`.

  Used when the thread-local stack is empty.
  """

  def __init__(self):
    _ThreadMode.__init__(self, _get_default_distribution_strategy(), None,
                         _get_default_tower_context())


def _get_per_thread_mode():
  try:
    return ops.get_default_graph()._distribution_strategy_stack[-1]  # pylint: disable=protected-access
  except (AttributeError, IndexError):
    return _get_default_tower_mode()


# ------------------------------------------------------------------------------
# Public API for accessing the current thread mode


def get_tower_context():
  """Returns the current TowerContext or None if in a cross-tower context.

  Note that execution:

  1. starts in the default (single-tower) tower context (this function
     will return the default TowerContext object);
  2. switches to cross-tower context (in which case this will return
     None) when entering a `with DistributionStrategy.scope():` block;
  3. switches to a (non-default) tower context inside
     `call_for_each_tower(fn, ...)`;
  4. if `fn` calls `get_tower_context()->merge_call(merge_fn, ...)`, then
     inside `merge_fn` you are back in the cross-tower context (and again
     this function will return None).

  Note that you can also go directly from step 1 to 4 to switch to a
  cross-tower context for the default `DistributionStrategy`. You may
  also switch from the cross-tower context of 4 to a tower context by
  calling `call_for_each_tower()`, jumping back to step 3.

  Most `DistributionStrategy` methods may only be executed in
  a cross-tower context, in a tower context you should use the
  `TowerContext` API instead.

  Returns:
    The current `TowerContext` object when in a tower context scope, else None.

    Exactly one of `get_tower_context()` and `get_cross_tower_context()`
    will return None in a particular block.
  """
  return _get_per_thread_mode().tower_context


def get_cross_tower_context():
  """Returns the current DistributionStrategy if in a cross-tower context.

  Note that execution:

  1. starts in the default (single-tower) tower context;
  2. switches to cross-tower context when entering a
     `with DistributionStrategy.scope():` block;
  3. switches to a (non-default) tower context inside
     `call_for_each_tower(fn, ...)`;
  4. if `fn` calls `get_tower_context()->merge_call(merge_fn, ...)`, then
     inside `merge_fn` you are back in the cross-tower context.

  Note that you can also go directly from step 1 to 4 to switch to a
  cross-tower context for the default `DistributionStrategy`. You may
  also switch from the cross-tower context of 4 to a tower context by
  calling `call_for_each_tower()`, jumping back to step 3.

  Most `DistributionStrategy` methods may only be executed in
  a cross-tower context.

  Returns:
    Returns the current `DistributionStrategy` object in a cross-tower
    context, or None.

    Exactly one of `get_tower_context()` and `get_cross_tower_context()`
    will return None in a particular block.
  """
  return _get_per_thread_mode().cross_tower_context


def get_distribution_strategy():
  """Returns the current `DistributionStrategy` object.

  Prefer to use `get_tower_context()` or `get_cross_tower_context()`
  instead when possible.

  Returns:
    A `DistributionStrategy` object. Inside a
    `with distribution_strategy.scope()` block, it returns
    `distribution_strategy`, otherwise it returns the default
    (single-tower) `DistributionStrategy` object.
  """
  return _get_per_thread_mode().distribution_strategy


def has_distribution_strategy():
  """Return if there is a current non-default `DistributionStrategy`.

  Returns:
    True if inside a `with distribution_strategy.scope():`.
  """
  return get_distribution_strategy() is not _get_default_distribution_strategy()


# ------------------------------------------------------------------------------
# Defaults that are used when no distribution strategy is explicitly created.
# We create them lazily in a function so that we can workaround the circular
# dependency on distribute_lib. See lazy loader at the top of this file.

_defaults = {
    "distribution_strategy": None,
    "tower_context": None,
    "tower_mode": None
}


def _get_default_distribution_strategy():
  if _defaults["distribution_strategy"] is None:
    _defaults["distribution_strategy"] = (
        distribute_lib._DefaultDistributionStrategy())  # pylint: disable=protected-access
  return _defaults["distribution_strategy"]


def _get_default_tower_context():
  if _defaults["tower_context"] is None:
    _defaults["tower_context"] = distribute_lib.TowerContext(
        _get_default_distribution_strategy(), tower_id=0)
  return _defaults["tower_context"]


def _get_default_tower_mode():
  if _defaults["tower_mode"] is None:
    _defaults["tower_mode"] = _DefaultTowerThreadMode()
  return _defaults["tower_mode"]
