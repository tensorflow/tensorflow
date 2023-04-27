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
# pylint: disable=line-too-long
"""Library for running a computation across multiple devices.

The intent of this library is that you can write an algorithm in a stylized way
and it will be usable with a variety of different `tf.distribute.Strategy`
implementations. Each descendant will implement a different strategy for
distributing the algorithm across multiple devices/machines.  Furthermore, these
changes can be hidden inside the specific layers and other library classes that
need special treatment to run in a distributed setting, so that most users'
model definition code can run unchanged. The `tf.distribute.Strategy` API works
the same way with eager and graph execution.

*Guides*

* [TensorFlow v2.x](https://www.tensorflow.org/guide/distributed_training)
* [TensorFlow v1.x](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/distribute_strategy.ipynb)

*Tutorials*

* [Distributed Training Tutorials](https://www.tensorflow.org/tutorials/distribute/)

  The tutorials cover how to use `tf.distribute.Strategy` to do distributed
  training with native Keras APIs, custom training loops,
  and Estimator APIs. They also cover how to save/load model when using
  `tf.distribute.Strategy`.

*Glossary*

* _Data parallelism_ is where we run multiple copies of the model
  on different slices of the input data. This is in contrast to
  _model parallelism_ where we divide up a single copy of a model
  across multiple devices.
  Note: we only support data parallelism for now, but
  hope to add support for model parallelism in the future.
* A _device_ is a CPU or accelerator (e.g. GPUs, TPUs) on some machine that
  TensorFlow can run operations on (see e.g. `tf.device`). You may have multiple
  devices on a single machine, or be connected to devices on multiple
  machines. Devices used to run computations are called _worker devices_.
  Devices used to store variables are _parameter devices_. For some strategies,
  such as `tf.distribute.MirroredStrategy`, the worker and parameter devices
  will be the same (see mirrored variables below). For others they will be
  different. For example, `tf.distribute.experimental.CentralStorageStrategy`
  puts the variables on a single device (which may be a worker device or may be
  the CPU), and `tf.distribute.experimental.ParameterServerStrategy` puts the
  variables on separate machines called _parameter servers_ (see below).
* A _replica_ is one copy of the model, running on one slice of the
  input data. Right now each replica is executed on its own
  worker device, but once we add support for model parallelism
  a replica may span multiple worker devices.
* A _host_ is the CPU device on a machine with worker devices, typically
  used for running input pipelines.
* A _worker_ is defined to be the physical machine(s) containing the physical
  devices (e.g. GPUs, TPUs) on which the replicated computation is executed. A
  worker may contain one or more replicas, but contains at least one
  replica. Typically one worker will correspond to one machine, but in the case
  of very large models with model parallelism, one worker may span multiple
  machines. We typically run one input pipeline per worker, feeding all the
  replicas on that worker.
* _Synchronous_, or more commonly _sync_, training is where the updates from
  each replica are aggregated together before updating the model variables. This
  is in contrast to _asynchronous_, or _async_ training, where each replica
  updates the model variables independently. You may also have replicas
  partitioned into groups which are in sync within each group but async between
  groups.
* _Parameter servers_: These are machines that hold a single copy of
  parameters/variables, used by some strategies (right now just
  `tf.distribute.experimental.ParameterServerStrategy`). All replicas that want
  to operate on a variable retrieve it at the beginning of a step and send an
  update to be applied at the end of the step. These can in principle support
  either sync or async training, but right now we only have support for async
  training with parameter servers. Compare to
  `tf.distribute.experimental.CentralStorageStrategy`, which puts all variables
  on a single device on the same machine (and does sync training), and
  `tf.distribute.MirroredStrategy`, which mirrors variables to multiple devices
  (see below).

* _Replica context_ vs. _Cross-replica context_ vs _Update context_

  A _replica context_ applies
  when you execute the computation function that was called with `strategy.run`.
  Conceptually, you're in replica context when executing the computation
  function that is being replicated.

  An _update context_ is entered in a `tf.distribute.StrategyExtended.update`
  call.

  An _cross-replica context_ is entered when you enter a `strategy.scope`. This
  is useful for calling `tf.distribute.Strategy` methods which operate across
  the replicas (like `reduce_to()`). By default you start in a _replica context_
  (the "default single _replica context_") and then some methods can switch you
  back and forth.

* _Distributed value_: Distributed value is represented by the base class
  `tf.distribute.DistributedValues`. `tf.distribute.DistributedValues` is useful
  to represent values on multiple devices, and it contains a map from replica id
  to values. Two representative types of `tf.distribute.DistributedValues`
  are `tf.types.experimental.PerReplica` and `tf.types.experimental.Mirrored`
  values.

  `PerReplica` values exist on the worker devices, with a different value for
  each replica. They are produced by iterating through a distributed dataset
  returned by `tf.distribute.Strategy.experimental_distribute_dataset` and
  `tf.distribute.Strategy.distribute_datasets_from_function`. They are also the
  typical result returned by `tf.distribute.Strategy.run`.

  `Mirrored` values are like `PerReplica` values, except we know that the value
  on all replicas are the same. `Mirrored` values are kept synchronized by the
  distribution strategy in use, while `PerReplica` values are left
  unsynchronized. `Mirrored` values typically represent model weights. We can
  safely read a `Mirrored` value in a cross-replica context by using the value
  on any replica, while PerReplica values can only be read within a replica
  context.

* _Unwrapping_ and _merging_: Consider calling a function `fn` on multiple
  replicas, like `strategy.run(fn, args=[w])` with an
  argument `w` that is a `tf.distribute.DistributedValues`. This means `w` will
  have a map taking replica id `0` to `w0`, replica id `1` to `w1`, etc.
  `strategy.run()` unwraps `w` before calling `fn`, so it calls `fn(w0)` on
  device `d0`, `fn(w1)` on device `d1`, etc.  It then merges the return
  values from `fn()`, which leads to one common object if the returned values
  are the same object from every replica, or a `DistributedValues` object
  otherwise.

* _Reductions_ and _all-reduce_: A _reduction_ is a method of aggregating
  multiple values into one value, like "sum" or "mean". If a strategy is doing
  sync training, we will perform a reduction on the gradients to a parameter
  from all replicas before applying the update. _All-reduce_ is an algorithm for
  performing a reduction on values from multiple devices and making the result
  available on all of those devices.

* _Mirrored variables_: These are variables that are created on multiple
  devices, where we keep the variables in sync by applying the same
  updates to every copy. Mirrored variables are created with
  `tf.Variable(...synchronization=tf.VariableSynchronization.ON_WRITE...)`.
  Normally they are only used in synchronous training.

* _SyncOnRead variables_

  _SyncOnRead variables_ are created by
  `tf.Variable(...synchronization=tf.VariableSynchronization.ON_READ...)`, and
  they are created on multiple devices. In replica context, each
  component variable on the local replica can perform reads and writes without
  synchronization with each other. When the
  _SyncOnRead variable_ is read in cross-replica context, the values from
  component variables are aggregated and returned.

  _SyncOnRead variables_ bring a lot of custom configuration difficulty to the
  underlying logic, so we do not encourage users to instantiate and use
  _SyncOnRead variable_ on their own. We have mainly used _SyncOnRead
  variables_ for use cases such as batch norm and metrics. For performance
  reasons, we often don't need to keep these statistics in sync every step and
  they can be accumulated on each replica independently. The only time we want
  to sync them is reporting or checkpointing, which typically happens in
  cross-replica context. _SyncOnRead variables_ are also often used by advanced
  users who want to control when variable values are aggregated. For example,
  users sometimes want to maintain gradients independently on each replica for a
  couple of steps without aggregation.

* _Distribute-aware layers_

  Layers are generally called in a replica context, except when defining a
  Keras functional model. `tf.distribute.in_cross_replica_context` will let you
  determine which case you are in. If in a replica context,
  the `tf.distribute.get_replica_context` function will return the default
  replica context outside a strategy scope, `None` within a strategy scope, and
  a `tf.distribute.ReplicaContext` object inside a strategy scope and within a
  `tf.distribute.Strategy.run` function. The `ReplicaContext` object has an
  `all_reduce` method for aggregating across all replicas.


Note that we provide a default version of `tf.distribute.Strategy` that is
used when no other strategy is in scope, that provides the same API with
reasonable default behavior.
"""
# pylint: enable=line-too-long

import collections
import contextlib
import copy
import enum  # pylint: disable=g-bad-import-order
import functools
import threading
import weakref

import six

from tensorflow.python import tf2
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context as eager_context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import tape
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import distribute as ds_types
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls

# ------------------------------------------------------------------------------
# Context tracking whether in a strategy.update() or .update_non_slot() call.


_update_replica_id = threading.local()


def get_update_replica_id():
  """Get the current device if in a `tf.distribute.Strategy.update()` call."""
  try:
    return _update_replica_id.current
  except AttributeError:
    return None


class UpdateContext(object):
  """Context manager when you are in `update()` or `update_non_slot()`."""

  __slots__ = ["_replica_id", "_old_replica_id"]

  def __init__(self, replica_id):
    self._replica_id = replica_id
    self._old_replica_id = None

  def __enter__(self):
    self._old_replica_id = get_update_replica_id()
    _update_replica_id.current = self._replica_id

  def __exit__(self, exception_type, exception_value, traceback):
    del exception_type, exception_value, traceback
    _update_replica_id.current = self._old_replica_id


# ------------------------------------------------------------------------------
# Internal API for validating the current thread mode


def _require_cross_replica_or_default_context_extended(extended,
                                                       error_message=None):
  """Verify in cross-replica context."""
  context = _get_per_thread_mode()
  cross_replica = context.cross_replica_context
  if cross_replica is not None and cross_replica.extended is extended:
    return
  if context is _get_default_replica_mode():
    return
  strategy = extended._container_strategy()  # pylint: disable=protected-access
  # We have an error to report, figure out the right message.
  if context.strategy is not strategy:
    _wrong_strategy_scope(strategy, context)
  assert cross_replica is None
  if not error_message:
    error_message = ("Method requires being in cross-replica context, use "
                     "get_replica_context().merge_call()")
  raise RuntimeError(error_message)


def _wrong_strategy_scope(strategy, context):
  # Figure out the right error message.
  if not has_strategy():
    raise RuntimeError(
        'Need to be inside "with strategy.scope()" for %s' %
        (strategy,))
  else:
    raise RuntimeError(
        "Mixing different tf.distribute.Strategy objects: %s is not %s" %
        (context.strategy, strategy))


def require_replica_context(replica_ctx):
  """Verify in `replica_ctx` replica context."""
  context = _get_per_thread_mode()
  if context.replica_context is replica_ctx: return
  # We have an error to report, figure out the right message.
  if context.replica_context is None:
    raise RuntimeError("Need to be inside `call_for_each_replica()`")
  if context.strategy is replica_ctx.strategy:
    # Two different ReplicaContexts with the same tf.distribute.Strategy.
    raise RuntimeError("Mismatching ReplicaContext.")
  raise RuntimeError(
      "Mismatching tf.distribute.Strategy objects: %s is not %s." %
      (context.strategy, replica_ctx.strategy))


def _require_strategy_scope_strategy(strategy):
  """Verify in a `strategy.scope()` in this thread."""
  context = _get_per_thread_mode()
  if context.strategy is strategy: return
  _wrong_strategy_scope(strategy, context)


def _require_strategy_scope_extended(extended):
  """Verify in a `distribution_strategy.scope()` in this thread."""
  context = _get_per_thread_mode()
  if context.strategy.extended is extended: return
  # Report error.
  strategy = extended._container_strategy()  # pylint: disable=protected-access
  _wrong_strategy_scope(strategy, context)


_creating_default_strategy_singleton = False

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


_variable_sync_on_read_context = threading.local()


@tf_export("__internal__.distribute.variable_sync_on_read_context", v1=[])
@contextlib.contextmanager
def variable_sync_on_read_context():
  """A context that forces SyncOnReadVariable to aggregate upon reading.

  This context is useful if one wants to read the aggregated value out of a
  SyncOnReadVariable in replica context. By default the aggregation is turned
  off per the definition of SyncOnReadVariable.

  When reading a SyncOnReadVariable in cross-replica context, aggregation is
  always turned on so there is no need for such context.

  By reading a SyncOnReadVariable, we mean:
    1. Convert the variable to a tensor using `convert_to_tensor`.
    2. Calling `variable.value()` or `variable.read_value()`.

  Example usage:

  ```
  strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])
  with strategy.scope():
    v = tf.Variable(1.0, synchronization=tf.VariableSynchronization.ON_READ,
      aggregation=tf.VariableAggregation.SUM)

  def replica_fn():
    return v + 10.0

  non_aggregated = strategy.run(replica_fn)
  print(non_aggregated) # PerReplica: {0: 11.0, 1: 11.0}

  def replica_fn():
    with variable_sync_on_read_context():
      return v + 10.0

  aggregated = strategy.run(replica_fn)
  print(aggregated) # PerReplica: {0: 12.0, 1: 12.0}
  ```

  Yields:
    Context manager for aggregating SyncOnReadVariable upon reading.
  """
  try:
    _variable_sync_on_read_context.entered = True
    yield
  finally:
    _variable_sync_on_read_context.entered = False


def in_variable_sync_on_read_context():
  try:
    return _variable_sync_on_read_context.entered
  except AttributeError:
    return False

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
  3. switches to a (non-default) replica context inside `strategy.run(fn, ...)`;
  4. if `fn` calls `get_replica_context().merge_call(merge_fn, ...)`, then
     inside `merge_fn` you are back in the cross-replica context (and again
     this function will return `None`).

  Most `tf.distribute.Strategy` methods may only be executed in
  a cross-replica context, in a replica context you should use the
  API of the `tf.distribute.ReplicaContext` object returned by this
  method instead.

  ```
  assert tf.distribute.get_replica_context() is not None  # default
  with strategy.scope():
    assert tf.distribute.get_replica_context() is None

    def f():
      replica_context = tf.distribute.get_replica_context()  # for strategy
      assert replica_context is not None
      tf.print("Replica id: ", replica_context.replica_id_in_sync_group,
               " of ", replica_context.num_replicas_in_sync)

    strategy.run(f)
  ```

  Returns:
    The current `tf.distribute.ReplicaContext` object when in a replica context
    scope, else `None`.

    Within a particular block, exactly one of these two things will be true:

    * `get_replica_context()` returns non-`None`, or
    * `tf.distribute.is_cross_replica_context()` returns True.
  """
  return _get_per_thread_mode().replica_context


def get_cross_replica_context():
  """Returns the current tf.distribute.Strategy if in a cross-replica context.

  DEPRECATED: Please use `in_cross_replica_context()` and
  `get_strategy()` instead.

  Returns:
    Returns the current `tf.distribute.Strategy` object in a cross-replica
    context, or `None`.

    Exactly one of `get_replica_context()` and `get_cross_replica_context()`
    will return `None` in a particular block.
  """
  return _get_per_thread_mode().cross_replica_context


@tf_export("distribute.in_cross_replica_context")
def in_cross_replica_context():
  """Returns `True` if in a cross-replica context.

  See `tf.distribute.get_replica_context` for details.

  ```
  assert not tf.distribute.in_cross_replica_context()
  with strategy.scope():
    assert tf.distribute.in_cross_replica_context()

    def f():
      assert not tf.distribute.in_cross_replica_context()

    strategy.run(f)
  ```

  Returns:
    `True` if in a cross-replica context (`get_replica_context()` returns
    `None`), or `False` if in a replica context (`get_replica_context()` returns
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

  ```
  assert not tf.distribute.has_strategy()
  with strategy.scope():
    assert tf.distribute.has_strategy()
  ```

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
# Internal helpers.


@contextlib.contextmanager
def enter_or_assert_strategy(strategy):
  if has_strategy():
    _assert_strategy(strategy)
    yield
  else:
    with strategy.scope():
      yield


# ------------------------------------------------------------------------------
# Defaults that are used when no tf.distribute.Strategy is explicitly created.
# We create them lazily in a function so that we can workaround the circular
# dependency on distribute_lib. See lazy loader at the top of this file.

_defaults = {
    "strategy": None,
    "replica_context": None,
    "replica_mode": None
}
# Note: These need to be different locks since _get_default_replica_context
# calls _get_default_strategy inside its lock, and them using the same lock
# can lead to deadlock.
_default_strategy_lock = threading.Lock()
_default_replica_context_lock = threading.Lock()
_default_replica_mode_lock = threading.Lock()


def _assert_strategy(strategy):
  if not has_strategy():
    raise RuntimeError('Need to be inside "with strategy.scope()" for %s' %
                       (strategy,))
  current_strategy = get_strategy()
  if current_strategy is not strategy:
    raise RuntimeError(
        "Mixing different tf.distribute.Strategy objects: %s is not %s" %
        (current_strategy, strategy))


def _get_default_strategy():
  if _defaults["strategy"] is None:
    # Avoid race condition causing two defaults to be created
    with _default_strategy_lock:
      if _defaults["strategy"] is None:
        # pylint: disable=protected-access
        # Make sure distribute_lib module is loaded by accessing some member.
        global _creating_default_strategy_singleton
        _creating_default_strategy_singleton = True
        if tf2.enabled():
          _defaults["strategy"] = _DefaultDistributionStrategy()
        else:
          _defaults["strategy"] = (
              _DefaultDistributionStrategyV1())
        _creating_default_strategy_singleton = False
        # pylint: enable=protected-access
  return _defaults["strategy"]


def _get_default_replica_context():
  if _defaults["replica_context"] is None:
    # Avoid race condition causing two defaults to be created
    with _default_replica_context_lock:
      if _defaults["replica_context"] is None:
        # pylint: disable=protected-access
        _defaults["replica_context"] = _DefaultReplicaContext(
            _get_default_strategy(), replica_id_in_sync_group=0)
        # pylint: enable=protected-access
  return _defaults["replica_context"]


def _get_default_replica_mode():
  if _defaults["replica_mode"] is None:
    # Avoid race condition causing two defaults to be created
    with _default_replica_mode_lock:
      if _defaults["replica_mode"] is None:
        _defaults["replica_mode"] = _DefaultReplicaThreadMode()
  return _defaults["replica_mode"]


# Aliases for compatibility with old names.
get_distribution_strategy = get_strategy
has_distribution_strategy = has_strategy


# ------------------------------------------------------------------------------
# Internal context managers used to implement the DistributionStrategy
# base class


class _CurrentDistributionContext(object):
  """Context manager setting the current `tf.distribute.Strategy`.

  Also: overrides the variable creator and optionally the current device.
  """

  def __init__(self,
               strategy,
               var_creator_scope,
               var_scope=None,
               resource_creator_scope=None,
               default_device=None):
    self._context = _CrossReplicaThreadMode(  # pylint: disable=protected-access
        strategy)
    self._var_creator_scope = var_creator_scope
    self._var_scope = var_scope
    self._resource_creator_scope = resource_creator_scope
    if default_device:
      self._device_scope = ops.device(default_device)
    else:
      self._device_scope = None
    self._same_scope_again_count = 0

  def __enter__(self):
    # Allow this scope to be entered if this strategy is already in scope.
    if has_strategy():
      _require_cross_replica_or_default_context_extended(
          self._context.strategy.extended)
      self._same_scope_again_count += 1
    else:
      _push_per_thread_mode(self._context)
      if self._var_scope:
        self._var_scope.__enter__()
      self._var_creator_scope.__enter__()
      if self._resource_creator_scope:
        nest.map_structure(lambda scope: scope.__enter__(),
                           self._resource_creator_scope)
      if self._device_scope:
        self._device_scope.__enter__()
    return self._context.strategy

  def __exit__(self, exception_type, exception_value, traceback):
    if self._same_scope_again_count > 0:
      self._same_scope_again_count -= 1
      return
    if self._device_scope:
      try:
        self._device_scope.__exit__(exception_type, exception_value, traceback)
      except RuntimeError as e:
        six.raise_from(
            RuntimeError("Device scope nesting error: move call to "
                         "tf.distribute.set_strategy() out of `with` scope."),
            e)

    try:
      self._var_creator_scope.__exit__(
          exception_type, exception_value, traceback)
    except RuntimeError as e:
      six.raise_from(
          RuntimeError("Variable creator scope nesting error: move call to "
                       "tf.distribute.set_strategy() out of `with` scope."),
          e)

    if self._resource_creator_scope:
      try:
        if isinstance(self._resource_creator_scope, list):
          reversed_resource_creator_scope = self._resource_creator_scope[::-1]
          nest.map_structure(
              lambda scope: scope.__exit__(exception_type, exception_value,  # pylint:disable=g-long-lambda
                                           traceback),
              reversed_resource_creator_scope)

        else:
          self._resource_creator_scope.__exit__(exception_type, exception_value,
                                                traceback)
      except RuntimeError as e:
        six.raise_from(
            RuntimeError("Resource creator scope nesting error: move call "
                         "to tf.distribute.set_strategy() out of `with` "
                         "scope."), e)

    if self._var_scope:
      try:
        self._var_scope.__exit__(exception_type, exception_value, traceback)
      except RuntimeError as e:
        six.raise_from(
            RuntimeError("Variable scope nesting error: move call to "
                         "tf.distribute.set_strategy() out of `with` scope."),
            e)
    _pop_per_thread_mode()


# TODO(yuefengz): add more replication modes.
@tf_export("distribute.InputReplicationMode")
class InputReplicationMode(enum.Enum):
  """Replication mode for input function.

  * `PER_WORKER`: The input function will be called on each worker
    independently, creating as many input pipelines as number of workers.
    Replicas will dequeue from the local Dataset on their worker.
    `tf.distribute.Strategy` doesn't manage any state sharing between such
    separate input pipelines.
  * `PER_REPLICA`: The input function will be called on each replica separately.
    `tf.distribute.Strategy` doesn't manage any state sharing between such
    separate input pipelines.
  """
  PER_WORKER = "PER_WORKER"
  PER_REPLICA = "PER_REPLICA"


@tf_export("distribute.InputContext")
class InputContext(object):
  """A class wrapping information needed by an input function.

  This is a context class that is passed to the user's input function and
  contains information about the compute replicas and input pipelines. The
  number of compute replicas (in sync training) helps compute the local batch
  size from the desired global batch size for each replica. The input pipeline
  information can be used to return a different subset of the input in each
  replica (for e.g. shard the input pipeline, use a different input
  source etc).
  """

  __slots__ = [
      "_num_input_pipelines", "_input_pipeline_id", "_num_replicas_in_sync"
  ]

  def __init__(self,
               num_input_pipelines=1,
               input_pipeline_id=0,
               num_replicas_in_sync=1):
    """Initializes an InputContext object.

    Args:
      num_input_pipelines: the number of input pipelines in a cluster.
      input_pipeline_id: the current input pipeline id, should be an int in
        [0,`num_input_pipelines`).
      num_replicas_in_sync: the number of replicas that are in sync.
    """
    self._num_input_pipelines = num_input_pipelines
    self._input_pipeline_id = input_pipeline_id
    self._num_replicas_in_sync = num_replicas_in_sync

  @property
  def num_replicas_in_sync(self):
    """Returns the number of compute replicas in sync."""
    return self._num_replicas_in_sync

  @property
  def input_pipeline_id(self):
    """Returns the input pipeline ID."""
    return self._input_pipeline_id

  @property
  def num_input_pipelines(self):
    """Returns the number of input pipelines."""
    return self._num_input_pipelines

  def get_per_replica_batch_size(self, global_batch_size):
    """Returns the per-replica batch size.

    Args:
      global_batch_size: the global batch size which should be divisible by
        `num_replicas_in_sync`.

    Returns:
      the per-replica batch size.

    Raises:
      ValueError: if `global_batch_size` not divisible by
        `num_replicas_in_sync`.
    """
    if global_batch_size % self._num_replicas_in_sync != 0:
      raise ValueError("The `global_batch_size` %r is not divisible by "
                       "`num_replicas_in_sync` %r " %
                       (global_batch_size, self._num_replicas_in_sync))
    return global_batch_size // self._num_replicas_in_sync

  def __str__(self):
    return "tf.distribute.InputContext(input pipeline id {}, total: {})".format(
        self.input_pipeline_id, self.num_input_pipelines)


@tf_export("distribute.experimental.ValueContext", v1=[])
class ValueContext(object):
  """A class wrapping information needed by a distribute function.

  This is a context class that is passed to the `value_fn` in
  `strategy.experimental_distribute_values_from_function` and contains
  information about the compute replicas. The `num_replicas_in_sync` and
  `replica_id` can be used to customize the value on each replica.

  Example usage:

  1.  Directly constructed.

      >>> def value_fn(context):
      ...   return context.replica_id_in_sync_group/context.num_replicas_in_sync
      >>> context = tf.distribute.experimental.ValueContext(
      ...   replica_id_in_sync_group=2, num_replicas_in_sync=4)
      >>> per_replica_value = value_fn(context)
      >>> per_replica_value
      0.5

  2.  Passed in by `experimental_distribute_values_from_function`.  {: value=2}

      >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
      >>> def value_fn(value_context):
      ...   return value_context.num_replicas_in_sync
      >>> distributed_values = (
      ...      strategy.experimental_distribute_values_from_function(
      ...        value_fn))
      >>> local_result = strategy.experimental_local_results(distributed_values)
      >>> local_result
      (2, 2)

  """

  __slots__ = ["_replica_id_in_sync_group", "_num_replicas_in_sync"]

  def __init__(self,
               replica_id_in_sync_group=0,
               num_replicas_in_sync=1):
    """Initializes an ValueContext object.

    Args:
      replica_id_in_sync_group: the current replica_id, should be an int in
        [0,`num_replicas_in_sync`).
      num_replicas_in_sync: the number of replicas that are in sync.
    """
    self._replica_id_in_sync_group = replica_id_in_sync_group
    self._num_replicas_in_sync = num_replicas_in_sync

  @property
  def num_replicas_in_sync(self):
    """Returns the number of compute replicas in sync."""
    return self._num_replicas_in_sync

  @property
  def replica_id_in_sync_group(self):
    """Returns the replica ID."""
    return self._replica_id_in_sync_group

  def __str__(self):
    return (("tf.distribute.ValueContext(replica id {}, "
             " total replicas in sync: ""{})")
            .format(self.replica_id_in_sync_group, self.num_replicas_in_sync))


@tf_export("distribute.RunOptions")
class RunOptions(
    collections.namedtuple("RunOptions", [
        "experimental_enable_dynamic_batch_size",
        "experimental_bucketizing_dynamic_shape",
        "experimental_xla_options",
    ])):
  """Run options for `strategy.run`.

  This can be used to hold some strategy specific configs.

  Attributes:
    experimental_enable_dynamic_batch_size: Boolean. Only applies to
      TPUStrategy. Default to True. If True, TPUStrategy will enable dynamic
      padder to support dynamic batch size for the inputs. Otherwise only static
      shape inputs are allowed.
    experimental_bucketizing_dynamic_shape: Boolean. Only applies to
      TPUStrategy. Default to False. If True, TPUStrategy will automatic
      bucketize inputs passed into `run` if the input shape is
      dynamic. This is a performance optimization to reduce XLA recompilation,
      which should not have impact on correctness.
    experimental_xla_options: A `tf.tpu.XLAOptions` instance. Only applies to
      TPUStrategy. Controls the XLA compiling options on TPUs. Default to None.
  """

  def __new__(cls,
              experimental_enable_dynamic_batch_size=True,
              experimental_bucketizing_dynamic_shape=False,
              experimental_xla_options=None):
    return super(RunOptions,
                 cls).__new__(cls, experimental_enable_dynamic_batch_size,
                              experimental_bucketizing_dynamic_shape,
                              experimental_xla_options)


@tf_export("distribute.InputOptions", v1=[])
class InputOptions(
    collections.namedtuple("InputOptions", [
        "experimental_fetch_to_device",
        "experimental_replication_mode",
        "experimental_place_dataset_on_device",
        "experimental_per_replica_buffer_size",
    ])):
  """Run options for `experimental_distribute_dataset(s_from_function)`.

  This can be used to hold some strategy specific configs.

  ```python
  # Setup TPUStrategy
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  strategy = tf.distribute.TPUStrategy(resolver)

  dataset = tf.data.Dataset.range(16)
  distributed_dataset_on_host = (
      strategy.experimental_distribute_dataset(
          dataset,
          tf.distribute.InputOptions(
              experimental_replication_mode=
              experimental_replication_mode.PER_WORKER,
              experimental_place_dataset_on_device=False,
              experimental_per_replica_buffer_size=1)))
  ```

  Attributes:
    experimental_fetch_to_device: Boolean. If True, dataset
      elements will be prefetched to accelerator device memory. When False,
      dataset elements are prefetched to host device memory. Must be False when
      using TPUEmbedding API. experimental_fetch_to_device can only be used
      with experimental_replication_mode=PER_WORKER. Default behavior is same as
      setting it to True.
    experimental_replication_mode: Replication mode for the input function.
      Currently, the InputReplicationMode.PER_REPLICA is only supported with
      tf.distribute.MirroredStrategy.
      experimental_distribute_datasets_from_function.
      The default value is InputReplicationMode.PER_WORKER.
    experimental_place_dataset_on_device: Boolean. Default to False. When True,
      dataset will be placed on the device, otherwise it will remain on the
      host. experimental_place_dataset_on_device=True can only be used with
      experimental_replication_mode=PER_REPLICA
    experimental_per_replica_buffer_size: Integer. Default to 1. Indicates the
      prefetch buffer size in the replica device memory. Users can set it
      to 0 to completely disable prefetching behavior, or a number greater than
      1 to enable larger buffer size. Note that this option is still
      valid with `experimental_fetch_to_device=False`.
  """

  def __new__(cls,
              experimental_fetch_to_device=None,
              experimental_replication_mode=InputReplicationMode.PER_WORKER,
              experimental_place_dataset_on_device=False,
              experimental_per_replica_buffer_size=1):
    if experimental_fetch_to_device is None:
      experimental_fetch_to_device = True

    return super(InputOptions,
                 cls).__new__(cls, experimental_fetch_to_device,
                              experimental_replication_mode,
                              experimental_place_dataset_on_device,
                              experimental_per_replica_buffer_size)

# ------------------------------------------------------------------------------
# Base classes for all distribution strategies.


# Base class for v1 Strategy and v2 Strategy classes. For API's specific to
# v1/v2 Strategy, add to implementing classes of StrategyBase.
# pylint: disable=line-too-long
class StrategyBase(object):
  """A state & compute distribution policy on a list of devices.

  See [the guide](https://www.tensorflow.org/guide/distributed_training)
  for overview and examples. See `tf.distribute.StrategyExtended` and
  [`tf.distribute`](https://www.tensorflow.org/api_docs/python/tf/distribute)
  for a glossary of concepts mentioned on this page such as "per-replica",
  _replica_, and _reduce_.

  In short:

  * To use it with Keras `compile`/`fit`,
    [please
    read](https://www.tensorflow.org/guide/distributed_training#using_tfdistributestrategy_with_keras).
  * You may pass descendant of `tf.distribute.Strategy` to
    `tf.estimator.RunConfig` to specify how a `tf.estimator.Estimator`
    should distribute its computation. See
    [guide](https://www.tensorflow.org/guide/distributed_training#using_tfdistributestrategy_with_estimator_limited_support).
  * Otherwise, use `tf.distribute.Strategy.scope` to specify that a
    strategy should be used when building an executing your model.
    (This puts you in the "cross-replica context" for this strategy, which
    means the strategy is put in control of things like variable placement.)
  * If you are writing a custom training loop, you will need to call a few more
    methods,
    [see the
    guide](https://www.tensorflow.org/guide/distributed_training#using_tfdistributestrategy_with_custom_training_loops):

      * Start by creating a `tf.data.Dataset` normally.
      * Use `tf.distribute.Strategy.experimental_distribute_dataset` to convert
        a `tf.data.Dataset` to something that produces "per-replica" values.
        If you want to manually specify how the dataset should be partitioned
        across replicas, use
        `tf.distribute.Strategy.distribute_datasets_from_function`
        instead.
      * Use `tf.distribute.Strategy.run` to run a function
        once per replica, taking values that may be "per-replica" (e.g.
        from a `tf.distribute.DistributedDataset` object) and returning
        "per-replica" values.
        This function is executed in "replica context", which means each
        operation is performed separately on each replica.
      * Finally use a method (such as `tf.distribute.Strategy.reduce`) to
        convert the resulting "per-replica" values into ordinary `Tensor`s.

  A custom training loop can be as simple as:

  ```
  with my_strategy.scope():
    @tf.function
    def distribute_train_epoch(dataset):
      def replica_fn(input):
        # process input and return result
        return result

      total_result = 0
      for x in dataset:
        per_replica_result = my_strategy.run(replica_fn, args=(x,))
        total_result += my_strategy.reduce(tf.distribute.ReduceOp.SUM,
                                           per_replica_result, axis=None)
      return total_result

    dist_dataset = my_strategy.experimental_distribute_dataset(dataset)
    for _ in range(EPOCHS):
      train_result = distribute_train_epoch(dist_dataset)
  ```

  This takes an ordinary `dataset` and `replica_fn` and runs it
  distributed using a particular `tf.distribute.Strategy` named
  `my_strategy` above. Any variables created in `replica_fn` are created
  using `my_strategy`'s policy, and library functions called by
  `replica_fn` can use the `get_replica_context()` API to implement
  distributed-specific behavior.

  You can use the `reduce` API to aggregate results across replicas and use
  this as a return value from one iteration over a
  `tf.distribute.DistributedDataset`. Or
  you can use `tf.keras.metrics` (such as loss, accuracy, etc.) to
  accumulate metrics across steps in a given epoch.

  See the
  [custom training loop
  tutorial](https://www.tensorflow.org/tutorials/distribute/custom_training)
  for a more detailed example.

  Note: `tf.distribute.Strategy` currently does not support TensorFlow's
  partitioned variables (where a single variable is split across multiple
  devices) at this time.
  """
  # pylint: enable=line-too-long

  # TODO(joshl): Partitioned computations, state; sharding
  # TODO(joshl): Model parallelism: "replicas" with multiple devices; shuffling

  def __init__(self, extended):
    self._extended = extended

    # Flag that is used to indicate whether distribution strategy is used with
    # Estimator. This is required for backward compatibility of loss scaling
    # when using v1 optimizer with estimator.
    self._scale_loss_for_estimator = False

    if not hasattr(extended, "_retrace_functions_for_each_device"):
      # pylint: disable=protected-access
      # `extended._retrace_functions_for_each_device` dictates
      # whether the same function will be retraced when it is called on
      # different devices.
      try:
        extended._retrace_functions_for_each_device = (
            len(extended.worker_devices) > 1)
        distribution_strategy_replica_gauge.get_cell("num_replicas").set(
            self.num_replicas_in_sync)
      except:  # pylint: disable=bare-except
        # Default for the case where extended.worker_devices can't return
        # a sensible value.
        extended._retrace_functions_for_each_device = True

    # Below are the dicts of axis(int) -> `tf.function`.
    self._mean_reduce_helper_fns = {}
    self._reduce_sum_fns = {}

    # Whether this strategy is designed to work with `ClusterCoordinator`.
    self._should_use_with_coordinator = False

  @property
  def extended(self):
    """`tf.distribute.StrategyExtended` with additional methods."""
    return self._extended

  @tf_contextlib.contextmanager
  def _scale_loss_for_estimator_enabled(self):
    """Scope which sets a flag used for scaling losses in optimizer.

    Yields:
      `_scale_loss_for_estimator_enabled` is a context manager with a
      side effect, but doesn't return a value.
    """
    self._scale_loss_for_estimator = True
    try:
      yield
    finally:
      self._scale_loss_for_estimator = False

  # pylint: disable=line-too-long
  def scope(self):
    """Context manager to make the strategy current and distribute variables.

    This method returns a context manager, and is used as follows:

    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> # Variable created inside scope:
    >>> with strategy.scope():
    ...   mirrored_variable = tf.Variable(1.)
    >>> mirrored_variable
    MirroredVariable:{
      0: <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>,
      1: <tf.Variable 'Variable/replica_1:0' shape=() dtype=float32, numpy=1.0>
    }
    >>> # Variable created outside scope:
    >>> regular_variable = tf.Variable(1.)
    >>> regular_variable
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>

    _What happens when Strategy.scope is entered?_

    * `strategy` is installed in the global context as the "current" strategy.
      Inside this scope, `tf.distribute.get_strategy()` will now return this
      strategy. Outside this scope, it returns the default no-op strategy.
    * Entering the scope also enters the "cross-replica context". See
      `tf.distribute.StrategyExtended` for an explanation on cross-replica and
      replica contexts.
    * Variable creation inside `scope` is intercepted by the strategy. Each
      strategy defines how it wants to affect the variable creation. Sync
      strategies like `MirroredStrategy`, `TPUStrategy` and
      `MultiWorkerMiroredStrategy` create variables replicated on each replica,
      whereas `ParameterServerStrategy` creates variables on the parameter
      servers. This is done using a custom `tf.variable_creator_scope`.
    * In some strategies, a default device scope may also be entered: in
      `MultiWorkerMiroredStrategy`, a default device scope of "/CPU:0" is
      entered on each worker.

    Note: Entering a scope does not automatically distribute a computation, except
      in the case of high level training framework like keras `model.fit`. If
      you're not using `model.fit`, you
      need to use `strategy.run` API to explicitly distribute that computation.
      See an example in the [custom training loop tutorial](https://www.tensorflow.org/tutorials/distribute/custom_training).


    _What should be in scope and what should be outside?_

    There are a number of requirements on what needs to happen inside the scope.
    However, in places where we have information about which strategy is in use,
    we often enter the scope for the user, so they don't have to do it
    explicitly (i.e. calling those either inside or outside the scope is OK).

    * Anything that creates variables that should be distributed variables
      must be called in a `strategy.scope`. This can be accomplished either by
      directly calling the variable creating function within the scope context,
      or by relying on another API like `strategy.run` or `keras.Model.fit` to
      automatically enter it for you. Any variable that is created outside scope
      will not be distributed and may have performance implications. Some common
      objects that create variables in TF are Models, Optimizers, Metrics. Such
      objects should always be initialized in the scope, and any functions
      that may lazily create variables (e.g., `Model.__call__()`, tracing a
      `tf.function`, etc.) should similarly be called within scope. Another
      source of variable creation can be a checkpoint restore - when variables
      are created lazily. Note that any variable created inside a strategy
      captures the strategy information. So reading and writing to these
      variables outside the `strategy.scope` can also work seamlessly, without
      the user having to enter the scope.
    * Some strategy APIs (such as `strategy.run` and `strategy.reduce`) which
      require to be in a strategy's scope, enter the scope automatically, which
      means when using those APIs you don't need to explicitly enter the scope
      yourself.
    * When a `tf.keras.Model` is created inside a `strategy.scope`, the Model
      object captures the scope information. When high level training framework
      methods such as `model.compile`, `model.fit`, etc. are then called, the
      captured scope will be automatically entered, and the associated strategy
      will be used to distribute the training etc. See a detailed example in
      [distributed keras tutorial](https://www.tensorflow.org/tutorials/distribute/keras).
      WARNING: Simply calling `model(..)` does not automatically enter the
      captured scope -- only high level training framework APIs support this
      behavior: `model.compile`, `model.fit`, `model.evaluate`, `model.predict`
      and `model.save` can all be called inside or outside the scope.
    * The following can be either inside or outside the scope:
        * Creating the input datasets
        * Defining `tf.function`s that represent your training step
        * Saving APIs such as `tf.saved_model.save`. Loading creates variables,
          so that should go inside the scope if you want to train the model in a
          distributed way.
        * Checkpoint saving. As mentioned above - `checkpoint.restore` may
          sometimes need to be inside scope if it creates variables.

    Returns:
      A context manager.
    """
    return self._extended._scope(self)  # pylint: disable=protected-access
  # pylint: enable=line-too-long

  @doc_controls.do_not_doc_inheritable  # DEPRECATED, moving to `extended`
  @deprecated(None, "use extended.colocate_vars_with() instead.")
  def colocate_vars_with(self, colocate_with_variable):
    """DEPRECATED: use extended.colocate_vars_with() instead."""
    return self._extended.colocate_vars_with(colocate_with_variable)

  @doc_controls.do_not_generate_docs  # DEPRECATED: TF 1.x only
  def make_dataset_iterator(self, dataset):
    """DEPRECATED TF 1.x ONLY."""
    return self._extended._make_dataset_iterator(dataset)  # pylint: disable=protected-access

  @doc_controls.do_not_generate_docs  # DEPRECATED: TF 1.x only
  def make_input_fn_iterator(self,
                             input_fn,
                             replication_mode=InputReplicationMode.PER_WORKER):
    """DEPRECATED TF 1.x ONLY."""
    if replication_mode != InputReplicationMode.PER_WORKER:
      raise ValueError(
          "Input replication mode not supported: %r" % replication_mode)
    with self.scope():
      return self.extended._make_input_fn_iterator(  # pylint: disable=protected-access
          input_fn, replication_mode=replication_mode)

  @doc_controls.do_not_generate_docs  # DEPRECATED: TF 1.x only
  @deprecated(None, "use run() instead")
  def experimental_run(self, fn, input_iterator=None):
    """DEPRECATED TF 1.x ONLY."""
    with self.scope():
      args = (input_iterator.get_next(),) if input_iterator is not None else ()
    return self.run(fn, args=args)

  def experimental_distribute_dataset(self, dataset, options=None):
    # pylint: disable=line-too-long
    """Creates `tf.distribute.DistributedDataset` from `tf.data.Dataset`.

    The returned `tf.distribute.DistributedDataset` can be iterated over
    similar to regular datasets.
    NOTE: The user cannot add any more transformations to a
    `tf.distribute.DistributedDataset`. You can only create an iterator or
    examine the `tf.TypeSpec` of the data generated by it. See API docs of
    `tf.distribute.DistributedDataset` to learn more.

    The following is an example:

    >>> global_batch_size = 2
    >>> # Passing the devices is optional.
    ... strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])
    >>> # Create a dataset
    ... dataset = tf.data.Dataset.range(4).batch(global_batch_size)
    >>> # Distribute that dataset
    ... dist_dataset = strategy.experimental_distribute_dataset(dataset)
    >>> @tf.function
    ... def replica_fn(input):
    ...   return input*2
    >>> result = []
    >>> # Iterate over the `tf.distribute.DistributedDataset`
    ... for x in dist_dataset:
    ...   # process dataset elements
    ...   result.append(strategy.run(replica_fn, args=(x,)))
    >>> print(result)
    [PerReplica:{
      0: <tf.Tensor: shape=(1,), dtype=int64, numpy=array([0])>,
      1: <tf.Tensor: shape=(1,), dtype=int64, numpy=array([2])>
    }, PerReplica:{
      0: <tf.Tensor: shape=(1,), dtype=int64, numpy=array([4])>,
      1: <tf.Tensor: shape=(1,), dtype=int64, numpy=array([6])>
    }]


    Three key actions happening under the hood of this method are batching,
    sharding, and prefetching.

    In the code snippet above, `dataset` is batched by `global_batch_size`, and
    calling `experimental_distribute_dataset` on it rebatches `dataset` to a
    new batch size that is equal to the global batch size divided by the number
    of replicas in sync. We iterate through it using a Pythonic for loop.
    `x` is a `tf.distribute.DistributedValues` containing data for all replicas,
    and each replica gets data of the new batch size.
    `tf.distribute.Strategy.run` will take care of feeding the right per-replica
    data in `x` to the right `replica_fn` executed on each replica.

    Sharding contains autosharding across multiple workers and within every
    worker. First, in multi-worker distributed training (i.e. when you use
    `tf.distribute.experimental.MultiWorkerMirroredStrategy`
    or `tf.distribute.TPUStrategy`), autosharding a dataset over a set of
    workers means that each worker is assigned a subset of the entire dataset
    (if the right `tf.data.experimental.AutoShardPolicy` is set). This is to
    ensure that at each step, a global batch size of non-overlapping dataset
    elements will be processed by each worker. Autosharding has a couple of
    different options that can be specified using
    `tf.data.experimental.DistributeOptions`. Then, sharding within each worker
    means the method will split the data among all the worker devices (if more
    than one a present). This will happen regardless of multi-worker
    autosharding.

    Note: for autosharding across multiple workers, the default mode is
    `tf.data.experimental.AutoShardPolicy.AUTO`. This mode
    will attempt to shard the input dataset by files if the dataset is
    being created out of reader datasets (e.g. `tf.data.TFRecordDataset`,
    `tf.data.TextLineDataset`, etc.) or otherwise shard the dataset by data,
    where each of the workers will read the entire dataset and only process the
    shard assigned to it. However, if you have less than one input file per
    worker, we suggest that you disable dataset autosharding across workers by
    setting the `tf.data.experimental.DistributeOptions.auto_shard_policy` to be
    `tf.data.experimental.AutoShardPolicy.OFF`.

    By default, this method adds a prefetch transformation at the end of the
    user provided `tf.data.Dataset` instance. The argument to the prefetch
    transformation which is `buffer_size` is equal to the number of replicas in
    sync.

    If the above batch splitting and dataset sharding logic is undesirable,
    please use
    `tf.distribute.Strategy.distribute_datasets_from_function`
    instead, which does not do any automatic batching or sharding for you.

    Note: If you are using TPUStrategy, the order in which the data is processed
    by the workers when using
    `tf.distribute.Strategy.experimental_distribute_dataset` or
    `tf.distribute.Strategy.distribute_datasets_from_function` is
    not guaranteed. This is typically required if you are using
    `tf.distribute` to scale prediction. You can however insert an index for
    each element in the batch and order outputs accordingly. Refer to [this
    snippet](https://www.tensorflow.org/tutorials/distribute/input#caveats)
    for an example of how to order outputs.

    Note: Stateful dataset transformations are currently not supported with
    `tf.distribute.experimental_distribute_dataset` or
    `tf.distribute.distribute_datasets_from_function`. Any stateful
    ops that the dataset may have are currently ignored. For example, if your
    dataset has a `map_fn` that uses `tf.random.uniform` to rotate an image,
    then you have a dataset graph that depends on state (i.e the random seed) on
    the local machine where the python process is being executed.

    For a tutorial on more usage and properties of this method, refer to the
    [tutorial on distributed input](https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategyexperimental_distribute_dataset).
    If you are interested in last partial batch handling, read [this section](https://www.tensorflow.org/tutorials/distribute/input#partial_batches).

    Args:
      dataset: `tf.data.Dataset` that will be sharded across all replicas using
        the rules stated above.
      options: `tf.distribute.InputOptions` used to control options on how this
        dataset is distributed.

    Returns:
      A `tf.distribute.DistributedDataset`.
    """
    distribution_strategy_input_api_counter.get_cell(
        self.__class__.__name__, "distribute_dataset").increase_by(1)
    # pylint: enable=line-too-long
    return self._extended._experimental_distribute_dataset(dataset, options)  # pylint: disable=protected-access

  def distribute_datasets_from_function(self, dataset_fn, options=None):
    # pylint: disable=line-too-long
    """Distributes `tf.data.Dataset` instances created by calls to `dataset_fn`.

    The argument `dataset_fn` that users pass in is an input function that has a
    `tf.distribute.InputContext` argument and returns a `tf.data.Dataset`
    instance. It is expected that the returned dataset from `dataset_fn` is
    already batched by per-replica batch size (i.e. global batch size divided by
    the number of replicas in sync) and sharded.
    `tf.distribute.Strategy.distribute_datasets_from_function` does
    not batch or shard the `tf.data.Dataset` instance
    returned from the input function. `dataset_fn` will be called on the CPU
    device of each of the workers and each generates a dataset where every
    replica on that worker will dequeue one batch of inputs (i.e. if a worker
    has two replicas, two batches will be dequeued from the `Dataset` every
    step).

    This method can be used for several purposes. First, it allows you to
    specify your own batching and sharding logic. (In contrast,
    `tf.distribute.experimental_distribute_dataset` does batching and sharding
    for you.) For example, where
    `experimental_distribute_dataset` is unable to shard the input files, this
    method might be used to manually shard the dataset (avoiding the slow
    fallback behavior in `experimental_distribute_dataset`). In cases where the
    dataset is infinite, this sharding can be done by creating dataset replicas
    that differ only in their random seed.

    The `dataset_fn` should take an `tf.distribute.InputContext` instance where
    information about batching and input replication can be accessed.

    You can use `element_spec` property of the
    `tf.distribute.DistributedDataset` returned by this API to query the
    `tf.TypeSpec` of the elements returned by the iterator. This can be used to
    set the `input_signature` property of a `tf.function`. Follow
    `tf.distribute.DistributedDataset.element_spec` to see an example.

    IMPORTANT: The `tf.data.Dataset` returned by `dataset_fn` should have a
    per-replica batch size, unlike `experimental_distribute_dataset`, which uses
    the global batch size. This may be computed using
    `input_context.get_per_replica_batch_size`.

    Note: If you are using TPUStrategy, the order in which the data is processed
    by the workers when using
    `tf.distribute.Strategy.experimental_distribute_dataset` or
    `tf.distribute.Strategy.distribute_datasets_from_function` is
    not guaranteed. This is typically required if you are using
    `tf.distribute` to scale prediction. You can however insert an index for
    each element in the batch and order outputs accordingly. Refer to [this
    snippet](https://www.tensorflow.org/tutorials/distribute/input#caveats)
    for an example of how to order outputs.

    Note: Stateful dataset transformations are currently not supported with
    `tf.distribute.experimental_distribute_dataset` or
    `tf.distribute.distribute_datasets_from_function`. Any stateful
    ops that the dataset may have are currently ignored. For example, if your
    dataset has a `map_fn` that uses `tf.random.uniform` to rotate an image,
    then you have a dataset graph that depends on state (i.e the random seed) on
    the local machine where the python process is being executed.

    For a tutorial on more usage and properties of this method, refer to the
    [tutorial on distributed input](https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategyexperimental_distribute_datasets_from_function)).
    If you are interested in last partial batch handling, read [this section](https://www.tensorflow.org/tutorials/distribute/input#partial_batches).

    Args:
      dataset_fn: A function taking a `tf.distribute.InputContext` instance and
        returning a `tf.data.Dataset`.
      options: `tf.distribute.InputOptions` used to control options on how this
        dataset is distributed.

    Returns:
      A `tf.distribute.DistributedDataset`.
    """
    distribution_strategy_input_api_counter.get_cell(
        self.__class__.__name__,
        "distribute_datasets_from_function").increase_by(1)
    # pylint: enable=line-too-long
    return self._extended._distribute_datasets_from_function(  # pylint: disable=protected-access
        dataset_fn, options)

  # TODO(b/162776748): Remove deprecated symbol.
  @doc_controls.do_not_doc_inheritable
  @deprecation.deprecated(None, "rename to distribute_datasets_from_function")
  def experimental_distribute_datasets_from_function(self,
                                                     dataset_fn,
                                                     options=None):
    return self.distribute_datasets_from_function(dataset_fn, options)

  def run(self, fn, args=(), kwargs=None, options=None):
    """Invokes `fn` on each replica, with the given arguments.

    This method is the primary way to distribute your computation with a
    tf.distribute object. It invokes `fn` on each replica. If `args` or `kwargs`
    have `tf.distribute.DistributedValues`, such as those produced by a
    `tf.distribute.DistributedDataset` from
    `tf.distribute.Strategy.experimental_distribute_dataset` or
    `tf.distribute.Strategy.distribute_datasets_from_function`,
    when `fn` is executed on a particular replica, it will be executed with the
    component of `tf.distribute.DistributedValues` that correspond to that
    replica.

    `fn` is invoked under a replica context. `fn` may call
    `tf.distribute.get_replica_context()` to access members such as
    `all_reduce`. Please see the module-level docstring of tf.distribute for the
    concept of replica context.

    All arguments in `args` or `kwargs` can be a nested structure of tensors,
    e.g. a list of tensors, in which case `args` and `kwargs` will be passed to
    the `fn` invoked on each replica. Or `args` or `kwargs` can be
    `tf.distribute.DistributedValues` containing tensors or composite tensors,
    i.e. `tf.compat.v1.TensorInfo.CompositeTensor`, in which case each `fn` call
    will get the component of a `tf.distribute.DistributedValues` corresponding
    to its replica. Note that arbitrary Python values that are not of the types
    above are not supported.

    IMPORTANT: Depending on the implementation of `tf.distribute.Strategy` and
    whether eager execution is enabled, `fn` may be called one or more times. If
    `fn` is annotated with `tf.function` or `tf.distribute.Strategy.run` is
    called inside a `tf.function` (eager execution is disabled inside a
    `tf.function` by default), `fn` is called once per replica to generate a
    Tensorflow graph, which will then be reused for execution with new inputs.
    Otherwise, if eager execution is enabled, `fn` will be called once per
    replica every step just like regular python code.

    Example usage:

    1.  Constant tensor input.

        >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
        >>> tensor_input = tf.constant(3.0)
        >>> @tf.function
        ... def replica_fn(input):
        ...   return input*2.0
        >>> result = strategy.run(replica_fn, args=(tensor_input,))
        >>> result
        PerReplica:{
          0: <tf.Tensor: shape=(), dtype=float32, numpy=6.0>,
          1: <tf.Tensor: shape=(), dtype=float32, numpy=6.0>
        }

    2.  DistributedValues input.  {: value=2}

        >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
        >>> @tf.function
        ... def run():
        ...   def value_fn(value_context):
        ...     return value_context.num_replicas_in_sync
        ...   distributed_values = (
        ...     strategy.experimental_distribute_values_from_function(
        ...       value_fn))
        ...   def replica_fn2(input):
        ...     return input*2
        ...   return strategy.run(replica_fn2, args=(distributed_values,))
        >>> result = run()
        >>> result
        <tf.Tensor: shape=(), dtype=int32, numpy=4>

    3.  Use `tf.distribute.ReplicaContext` to allreduce values. {: value=3}

        >>> strategy = tf.distribute.MirroredStrategy(["gpu:0", "gpu:1"])
        >>> @tf.function
        ... def run():
        ...    def value_fn(value_context):
        ...      return tf.constant(value_context.replica_id_in_sync_group)
        ...    distributed_values = (
        ...        strategy.experimental_distribute_values_from_function(
        ...            value_fn))
        ...    def replica_fn(input):
        ...      return tf.distribute.get_replica_context().all_reduce(
        ...          "sum", input)
        ...    return strategy.run(replica_fn, args=(distributed_values,))
        >>> result = run()
        >>> result
        PerReplica:{
          0: <tf.Tensor: shape=(), dtype=int32, numpy=1>,
          1: <tf.Tensor: shape=(), dtype=int32, numpy=1>
        }

    Args:
      fn: The function to run on each replica.
      args: Optional positional arguments to `fn`. Its element can be a tensor,
        a nested structure of tensors or a `tf.distribute.DistributedValues`.
      kwargs: Optional keyword arguments to `fn`. Its element can be a tensor,
        a nested structure of tensors or a `tf.distribute.DistributedValues`.
      options: An optional instance of `tf.distribute.RunOptions` specifying
        the options to run `fn`.

    Returns:
      Merged return value of `fn` across replicas. The structure of the return
      value is the same as the return value from `fn`. Each element in the
      structure can either be `tf.distribute.DistributedValues`, `Tensor`
      objects, or `Tensor`s (for example, if running on a single replica).
    """
    del options

    if not isinstance(args, (list, tuple)):
      raise ValueError(
          "positional args must be a list or tuple, got {}".format(type(args)))

    with self.scope():
      # tf.distribute supports Eager functions, so AutoGraph should not be
      # applied when the caller is also in Eager mode.
      fn = autograph.tf_convert(
          fn, autograph_ctx.control_status_ctx(), convert_by_default=False)
      return self._extended.call_for_each_replica(fn, args=args, kwargs=kwargs)

  def reduce(self, reduce_op, value, axis):
    """Reduce `value` across replicas and return result on current device.

    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> def step_fn():
    ...   i = tf.distribute.get_replica_context().replica_id_in_sync_group
    ...   return tf.identity(i)
    >>>
    >>> per_replica_result = strategy.run(step_fn)
    >>> total = strategy.reduce("SUM", per_replica_result, axis=None)
    >>> total
    <tf.Tensor: shape=(), dtype=int32, numpy=1>

    To see how this would look with multiple replicas, consider the same
    example with MirroredStrategy with 2 GPUs:

    ```python
    strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])
    def step_fn():
      i = tf.distribute.get_replica_context().replica_id_in_sync_group
      return tf.identity(i)

    per_replica_result = strategy.run(step_fn)
    # Check devices on which per replica result is:
    strategy.experimental_local_results(per_replica_result)[0].device
    # /job:localhost/replica:0/task:0/device:GPU:0
    strategy.experimental_local_results(per_replica_result)[1].device
    # /job:localhost/replica:0/task:0/device:GPU:1

    total = strategy.reduce("SUM", per_replica_result, axis=None)
    # Check device on which reduced result is:
    total.device
    # /job:localhost/replica:0/task:0/device:CPU:0

    ```

    This API is typically used for aggregating the results returned from
    different replicas, for reporting etc. For example, loss computed from
    different replicas can be averaged using this API before printing.

    Note: The result is copied to the "current" device - which would typically
    be the CPU of the worker on which the program is running. For `TPUStrategy`,
    it is the first TPU host. For multi client `MultiWorkerMirroredStrategy`,
    this is CPU of each worker.

    There are a number of different tf.distribute APIs for reducing values
    across replicas:
    * `tf.distribute.ReplicaContext.all_reduce`: This differs from
    `Strategy.reduce` in that it is for replica context and does
    not copy the results to the host device. `all_reduce` should be typically
    used for reductions inside the training step such as gradients.
    * `tf.distribute.StrategyExtended.reduce_to` and
    `tf.distribute.StrategyExtended.batch_reduce_to`: These APIs are more
    advanced versions of `Strategy.reduce` as they allow customizing the
    destination of the result. They are also called in cross replica context.

    _What should axis be?_

    Given a per-replica value returned by `run`, say a
    per-example loss, the batch will be divided across all the replicas.  This
    function allows you to aggregate across replicas and optionally also across
    batch elements by specifying the axis parameter accordingly.

    For example, if you have a global batch size of 8 and 2
    replicas, values for examples `[0, 1, 2, 3]` will be on replica 0 and
    `[4, 5, 6, 7]` will be on replica 1. With `axis=None`, `reduce` will
    aggregate only across replicas, returning `[0+4, 1+5, 2+6, 3+7]`.
    This is useful when each replica is computing a scalar or some other value
    that doesn't have a "batch" dimension (like a gradient or loss).
    ```
    strategy.reduce("sum", per_replica_result, axis=None)
    ```

    Sometimes, you will want to aggregate across both the global batch _and_
    all replicas. You can get this behavior by specifying the batch
    dimension as the `axis`, typically `axis=0`. In this case it would return a
    scalar `0+1+2+3+4+5+6+7`.
    ```
    strategy.reduce("sum", per_replica_result, axis=0)
    ```

    If there is a last partial batch, you will need to specify an axis so
    that the resulting shape is consistent across replicas. So if the last
    batch has size 6 and it is divided into [0, 1, 2, 3] and [4, 5], you
    would get a shape mismatch unless you specify `axis=0`. If you specify
    `tf.distribute.ReduceOp.MEAN`, using `axis=0` will use the correct
    denominator of 6. Contrast this with computing `reduce_mean` to get a
    scalar value on each replica and this function to average those means,
    which will weigh some values `1/8` and others `1/4`.

    Args:
      reduce_op: a `tf.distribute.ReduceOp` value specifying how values should
        be combined. Allows using string representation of the enum such as
        "SUM", "MEAN".
      value: a `tf.distribute.DistributedValues` instance, e.g. returned by
        `Strategy.run`, to be combined into a single tensor. It can also be a
        regular tensor when used with `OneDeviceStrategy` or default strategy.
      axis: specifies the dimension to reduce along within each
        replica's tensor. Should typically be set to the batch dimension, or
        `None` to only reduce across replicas (e.g. if the tensor has no batch
        dimension).

    Returns:
      A `Tensor`.
    """
    # TODO(joshl): support `value` being a nest.
    _require_cross_replica_or_default_context_extended(self._extended)
    if isinstance(reduce_op, six.string_types):
      reduce_op = reduce_util.ReduceOp(reduce_op.upper())
    if axis is None:
      return self._extended._reduce(reduce_op, value)  # pylint: disable=protected-access
    if reduce_op == reduce_util.ReduceOp.SUM:

      def reduce_sum(v):
        return math_ops.reduce_sum(v, axis=axis)

      if eager_context.executing_eagerly():
        # As some strategies (e.g. TPUStrategy) doesn't support pure eager
        # execution, wrap the `reduce_sum_fn` with a `tf.function` so it can be
        # run from eager mode. Cache the tf.function by `axis` to avoid the
        # same function to be traced again.
        if axis not in self._reduce_sum_fns:
          self._reduce_sum_fns[axis] = def_function.function(reduce_sum)
        value = self.run(self._reduce_sum_fns[axis], args=(value,))
      else:
        value = self.run(reduce_sum, args=(value,))

      return self._extended._reduce(reduce_op, value)  # pylint: disable=protected-access
    if reduce_op != reduce_util.ReduceOp.MEAN:
      raise TypeError("Expected `reduce_op` to be a `tf.distribute.ReduceOp`, "
                      "not: %r" % reduce_op)

    def mean_reduce_helper(v, axes=axis):
      """Computes the numerator and denominator on each replica."""
      numer = math_ops.reduce_sum(v, axis=axes)
      def dimension(axis):
        if v.shape.rank is not None:
          # Note(joshl): We support axis < 0 to be consistent with the
          # tf.math.reduce_* operations.
          if axis < 0:
            if axis + v.shape.rank < 0:
              raise ValueError(
                  "`axis` = %r out of range for `value` with rank %d" %
                  (axis, v.shape.rank))
            axis += v.shape.rank
          elif axis >= v.shape.rank:
            raise ValueError(
                "`axis` = %r out of range for `value` with rank %d" %
                (axis, v.shape.rank))
          # TF v2 returns `None` for unknown dimensions and an integer for
          # known dimension, whereas TF v1 returns tensor_shape.Dimension(None)
          # or tensor_shape.Dimension(integer). `dimension_value` hides this
          # difference, always returning `None` or an integer.
          dim = tensor_shape.dimension_value(v.shape[axis])
          if dim is not None:
            # By returning a python value in the static shape case, we can
            # maybe get a fast path for reducing the denominator.
            # TODO(b/151871486): Remove array_ops.identity after we fallback to
            # simple reduction if inputs are all on CPU.
            return array_ops.identity(
                constant_op.constant(dim, dtype=dtypes.int64))
        elif axis < 0:
          axis = axis + array_ops.rank(v)
        # TODO(b/151871486): Remove array_ops.identity after we fallback to
        # simple reduction if inputs are all on CPU.
        return array_ops.identity(
            array_ops.shape_v2(v, out_type=dtypes.int64)[axis])
      if isinstance(axis, six.integer_types):
        denom = dimension(axis)
      elif isinstance(axis, (tuple, list)):
        denom = math_ops.reduce_prod([dimension(a) for a in axes])
      else:
        raise TypeError(
            "Expected `axis` to be an integer, tuple or list not: %r" % axis)
      # TODO(joshl): Should we cast denom to v.dtype here instead of after the
      # reduce is complete?
      return numer, denom

    if eager_context.executing_eagerly():
      # As some strategies (e.g. TPUStrategy) doesn't support pure eager
      # execution, wrap the `mean_reduce_helper` with a `tf.function` so it can
      # be run from eager mode. Cache the tf.function by `axis` to avoid the
      # same function to be traced again.
      if axis not in self._mean_reduce_helper_fns:
        self._mean_reduce_helper_fns[axis] = def_function.function(
            mean_reduce_helper)
      numer, denom = self.run(self._mean_reduce_helper_fns[axis], args=(value,))
    else:
      numer, denom = self.run(mean_reduce_helper, args=(value,))

    # TODO(joshl): Should batch reduce here instead of doing two.
    numer = self._extended._reduce(reduce_util.ReduceOp.SUM, numer)  # pylint: disable=protected-access
    denom = self._extended._reduce(reduce_util.ReduceOp.SUM, denom)  # pylint: disable=protected-access
    denom = math_ops.cast(denom, numer.dtype)
    return math_ops.truediv(numer, denom)

  @doc_controls.do_not_doc_inheritable  # DEPRECATED
  @deprecated(None, "use `experimental_local_results` instead.")
  def unwrap(self, value):
    """Returns the list of all local per-replica values contained in `value`.

    DEPRECATED: Please use `experimental_local_results` instead.

    Note: This only returns values on the workers initiated by this client.
    When using a `tf.distribute.Strategy` like
    `tf.distribute.experimental.MultiWorkerMirroredStrategy`, each worker
    will be its own client, and this function will only return values
    computed on that worker.

    Args:
      value: A value returned by `experimental_run()`,
        `extended.call_for_each_replica()`, or a variable created in `scope`.

    Returns:
      A tuple of values contained in `value`. If `value` represents a single
      value, this returns `(value,).`
    """
    return self._extended._local_results(value)  # pylint: disable=protected-access

  def experimental_local_results(self, value):
    """Returns the list of all local per-replica values contained in `value`.

    Note: This only returns values on the worker initiated by this client.
    When using a `tf.distribute.Strategy` like
    `tf.distribute.experimental.MultiWorkerMirroredStrategy`, each worker
    will be its own client, and this function will only return values
    computed on that worker.

    Args:
      value: A value returned by `experimental_run()`, `run(), or a variable
      created in `scope`.

    Returns:
      A tuple of values contained in `value` where ith element corresponds to
      ith replica. If `value` represents a single value, this returns
      `(value,).`
    """
    return self._extended._local_results(value)  # pylint: disable=protected-access

  @doc_controls.do_not_doc_inheritable  # DEPRECATED: TF v1.x only
  def group(self, value, name=None):
    """Shortcut for `tf.group(self.experimental_local_results(value))`."""
    return self._extended._group(value, name)  # pylint: disable=protected-access

  @property
  def num_replicas_in_sync(self):
    """Returns number of replicas over which gradients are aggregated."""
    return self._extended._num_replicas_in_sync  # pylint: disable=protected-access

  @doc_controls.do_not_doc_inheritable  # DEPRECATED: see doc string
  @deprecated(None, "use `update_config_proto` instead.")
  def configure(self,
                session_config=None,
                cluster_spec=None,
                task_type=None,
                task_id=None):
    # pylint: disable=g-doc-return-or-yield,g-doc-args
    """DEPRECATED: use `update_config_proto` instead.

    Configures the strategy class.

    DEPRECATED: This method's functionality has been split into the strategy
    constructor and `update_config_proto`. In the future, we will allow passing
    cluster and config_proto to the constructor to configure the strategy. And
    `update_config_proto` can be used to update the config_proto based on the
    specific strategy.
    """
    return self._extended._configure(  # pylint: disable=protected-access
        session_config, cluster_spec, task_type, task_id)

  @doc_controls.do_not_generate_docs  # DEPRECATED
  def update_config_proto(self, config_proto):
    """DEPRECATED TF 1.x ONLY."""
    return self._extended._update_config_proto(config_proto)  # pylint: disable=protected-access

  def __deepcopy__(self, memo):
    # First do a regular deepcopy of `self`.
    cls = self.__class__
    result = cls.__new__(cls)
    memo[id(self)] = result
    for k, v in self.__dict__.items():
      setattr(result, k, copy.deepcopy(v, memo))
    # One little fix-up: we want `result._extended` to reference `result`
    # instead of `self`.
    result._extended._container_strategy_weakref = weakref.ref(result)  # pylint: disable=protected-access
    return result

  def __copy__(self):
    raise RuntimeError("Must only deepcopy DistributionStrategy.")

  @property
  def cluster_resolver(self):
    """Returns the cluster resolver associated with this strategy.

    In general, when using a multi-worker `tf.distribute` strategy such as
    `tf.distribute.experimental.MultiWorkerMirroredStrategy` or
    `tf.distribute.TPUStrategy()`, there is a
    `tf.distribute.cluster_resolver.ClusterResolver` associated with the
    strategy used, and such an instance is returned by this property.

    Strategies that intend to have an associated
    `tf.distribute.cluster_resolver.ClusterResolver` must set the
    relevant attribute, or override this property; otherwise, `None` is returned
    by default. Those strategies should also provide information regarding what
    is returned by this property.

    Single-worker strategies usually do not have a
    `tf.distribute.cluster_resolver.ClusterResolver`, and in those cases this
    property will return `None`.

    The `tf.distribute.cluster_resolver.ClusterResolver` may be useful when the
    user needs to access information such as the cluster spec, task type or task
    id. For example,

    ```python

    os.environ['TF_CONFIG'] = json.dumps({
      'cluster': {
          'worker': ["localhost:12345", "localhost:23456"],
          'ps': ["localhost:34567"]
      },
      'task': {'type': 'worker', 'index': 0}
    })

    # This implicitly uses TF_CONFIG for the cluster and current task info.
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    ...

    if strategy.cluster_resolver.task_type == 'worker':
      # Perform something that's only applicable on workers. Since we set this
      # as a worker above, this block will run on this particular instance.
    elif strategy.cluster_resolver.task_type == 'ps':
      # Perform something that's only applicable on parameter servers. Since we
      # set this as a worker above, this block will not run on this particular
      # instance.
    ```

    For more information, please see
    `tf.distribute.cluster_resolver.ClusterResolver`'s API docstring.

    Returns:
      The cluster resolver associated with this strategy. Returns `None` if a
      cluster resolver is not applicable or available in this strategy.
    """
    if hasattr(self.extended, "_cluster_resolver"):
      return self.extended._cluster_resolver  # pylint: disable=protected-access
    return None


@tf_export("distribute.Strategy", v1=[])  # pylint: disable=g-missing-docstring
class Strategy(StrategyBase):

  __doc__ = StrategyBase.__doc__

  def experimental_distribute_values_from_function(self, value_fn):
    """Generates `tf.distribute.DistributedValues` from `value_fn`.

    This function is to generate `tf.distribute.DistributedValues` to pass
    into `run`, `reduce`, or other methods that take
    distributed values when not using datasets.

    Args:
      value_fn: The function to run to generate values. It is called for
        each replica with `tf.distribute.ValueContext` as the sole argument. It
        must return a Tensor or a type that can be converted to a Tensor.
    Returns:
      A `tf.distribute.DistributedValues` containing a value for each replica.

    Example usage:

    1.  Return constant value per replica:

        >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
        >>> def value_fn(ctx):
        ...   return tf.constant(1.)
        >>> distributed_values = (
        ...     strategy.experimental_distribute_values_from_function(
        ...        value_fn))
        >>> local_result = strategy.experimental_local_results(
        ...     distributed_values)
        >>> local_result
        (<tf.Tensor: shape=(), dtype=float32, numpy=1.0>,
        <tf.Tensor: shape=(), dtype=float32, numpy=1.0>)

    2.  Distribute values in array based on replica_id: {: value=2}

        >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
        >>> array_value = np.array([3., 2., 1.])
        >>> def value_fn(ctx):
        ...   return array_value[ctx.replica_id_in_sync_group]
        >>> distributed_values = (
        ...     strategy.experimental_distribute_values_from_function(
        ...         value_fn))
        >>> local_result = strategy.experimental_local_results(
        ...     distributed_values)
        >>> local_result
        (3.0, 2.0)

    3.  Specify values using num_replicas_in_sync:  {: value=3}

        >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
        >>> def value_fn(ctx):
        ...   return ctx.num_replicas_in_sync
        >>> distributed_values = (
        ...     strategy.experimental_distribute_values_from_function(
        ...         value_fn))
        >>> local_result = strategy.experimental_local_results(
        ...     distributed_values)
        >>> local_result
        (2, 2)

    4.  Place values on devices and distribute: {: value=4}

        ```
        strategy = tf.distribute.TPUStrategy()
        worker_devices = strategy.extended.worker_devices
        multiple_values = []
        for i in range(strategy.num_replicas_in_sync):
          with tf.device(worker_devices[i]):
            multiple_values.append(tf.constant(1.0))

        def value_fn(ctx):
          return multiple_values[ctx.replica_id_in_sync_group]

        distributed_values = strategy.
          experimental_distribute_values_from_function(
          value_fn)
        ```

    """
    return self._extended._experimental_distribute_values_from_function(  # pylint: disable=protected-access
        value_fn)

  def gather(self, value, axis):
    # pylint: disable=line-too-long, protected-access
    """Gather `value` across replicas along `axis` to the current device.

    Given a `tf.distribute.DistributedValues` or `tf.Tensor`-like
    object `value`, this API gathers and concatenates `value` across replicas
    along the `axis`-th dimension. The result is copied to the "current" device,
    which would typically be the CPU of the worker on which the program is
    running. For `tf.distribute.TPUStrategy`, it is the first TPU host. For
    multi-client `tf.distribute.MultiWorkerMirroredStrategy`, this is the CPU of
    each worker.

    This API can only be called in the cross-replica context. For a counterpart
    in the replica context, see `tf.distribute.ReplicaContext.all_gather`.

    Note: For all strategies except `tf.distribute.TPUStrategy`, the input
    `value` on different replicas must have the same rank, and their shapes must
    be the same in all dimensions except the `axis`-th dimension. In other
    words, their shapes cannot be different in a dimension `d` where `d` does
    not equal to the `axis` argument. For example, given a
    `tf.distribute.DistributedValues` with component tensors of shape
    `(1, 2, 3)` and `(1, 3, 3)` on two replicas, you can call
    `gather(..., axis=1, ...)` on it, but not `gather(..., axis=0, ...)` or
    `gather(..., axis=2, ...)`. However, for `tf.distribute.TPUStrategy.gather`,
    all tensors must have exactly the same rank and same shape.

    Note: Given a `tf.distribute.DistributedValues` `value`, its component
    tensors must have a non-zero rank. Otherwise, consider using
    `tf.expand_dims` before gathering them.

    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> # A DistributedValues with component tensor of shape (2, 1) on each replica
    ... distributed_values = strategy.experimental_distribute_values_from_function(lambda _: tf.identity(tf.constant([[1], [2]])))
    >>> @tf.function
    ... def run():
    ...   return strategy.gather(distributed_values, axis=0)
    >>> run()
    <tf.Tensor: shape=(4, 1), dtype=int32, numpy=
    array([[1],
           [2],
           [1],
           [2]], dtype=int32)>


    Consider the following example for more combinations:

    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1", "GPU:2", "GPU:3"])
    >>> single_tensor = tf.reshape(tf.range(6), shape=(1,2,3))
    >>> distributed_values = strategy.experimental_distribute_values_from_function(lambda _: tf.identity(single_tensor))
    >>> @tf.function
    ... def run(axis):
    ...   return strategy.gather(distributed_values, axis=axis)
    >>> axis=0
    >>> run(axis)
    <tf.Tensor: shape=(4, 2, 3), dtype=int32, numpy=
    array([[[0, 1, 2],
            [3, 4, 5]],
           [[0, 1, 2],
            [3, 4, 5]],
           [[0, 1, 2],
            [3, 4, 5]],
           [[0, 1, 2],
            [3, 4, 5]]], dtype=int32)>
    >>> axis=1
    >>> run(axis)
    <tf.Tensor: shape=(1, 8, 3), dtype=int32, numpy=
    array([[[0, 1, 2],
            [3, 4, 5],
            [0, 1, 2],
            [3, 4, 5],
            [0, 1, 2],
            [3, 4, 5],
            [0, 1, 2],
            [3, 4, 5]]], dtype=int32)>
    >>> axis=2
    >>> run(axis)
    <tf.Tensor: shape=(1, 2, 12), dtype=int32, numpy=
    array([[[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            [3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5]]], dtype=int32)>


    Args:
      value: a `tf.distribute.DistributedValues` instance, e.g. returned by
        `Strategy.run`, to be combined into a single tensor. It can also be a
        regular tensor when used with `tf.distribute.OneDeviceStrategy` or the
        default strategy. The tensors that constitute the DistributedValues
        can only be dense tensors with non-zero rank, NOT a `tf.IndexedSlices`.
      axis: 0-D int32 Tensor. Dimension along which to gather. Must be in the
        range [0, rank(value)).

    Returns:
       A `Tensor` that's the concatenation of `value` across replicas along
       `axis` dimension.
    """
    # pylint: enable=line-too-long
    error_message = ("tf.distribute.Strategy.gather method requires "
                     "cross-replica context, use "
                     "get_replica_context().all_gather() instead")
    _require_cross_replica_or_default_context_extended(self._extended,
                                                       error_message)
    dst = device_util.current(
    ) or self._extended._default_device or "/device:CPU:0"
    if isinstance(value, indexed_slices.IndexedSlices):
      raise NotImplementedError("gather does not support IndexedSlices")
    return self._extended._local_results(
        self._extended._gather_to(value, dst, axis))[0]


# TF v1.x version has additional deprecated APIs
@tf_export(v1=["distribute.Strategy"])
class StrategyV1(StrategyBase):
  """A list of devices with a state & compute distribution policy.

  See [the guide](https://www.tensorflow.org/guide/distribute_strategy)
  for overview and examples.

  Note: Not all `tf.distribute.Strategy` implementations currently support
  TensorFlow's partitioned variables (where a single variable is split across
  multiple devices) at this time.
  """

  def make_dataset_iterator(self, dataset):
    """Makes an iterator for input provided via `dataset`.

    DEPRECATED: This method is not available in TF 2.x.

    Data from the given dataset will be distributed evenly across all the
    compute replicas. We will assume that the input dataset is batched by the
    global batch size. With this assumption, we will make a best effort to
    divide each batch across all the replicas (one or more workers).
    If this effort fails, an error will be thrown, and the user should instead
    use `make_input_fn_iterator` which provides more control to the user, and
    does not try to divide a batch across replicas.

    The user could also use `make_input_fn_iterator` if they want to
    customize which input is fed to which replica/worker etc.

    Args:
      dataset: `tf.data.Dataset` that will be distributed evenly across all
        replicas.

    Returns:
      An `tf.distribute.InputIterator` which returns inputs for each step of the
      computation.  User should call `initialize` on the returned iterator.
    """
    return self._extended._make_dataset_iterator(dataset)  # pylint: disable=protected-access

  def make_input_fn_iterator(self,  # pylint: disable=useless-super-delegation
                             input_fn,
                             replication_mode=InputReplicationMode.PER_WORKER):
    """Returns an iterator split across replicas created from an input function.

    DEPRECATED: This method is not available in TF 2.x.

    The `input_fn` should take an `tf.distribute.InputContext` object where
    information about batching and input sharding can be accessed:

    ```
    def input_fn(input_context):
      batch_size = input_context.get_per_replica_batch_size(global_batch_size)
      d = tf.data.Dataset.from_tensors([[1.]]).repeat().batch(batch_size)
      return d.shard(input_context.num_input_pipelines,
                     input_context.input_pipeline_id)
    with strategy.scope():
      iterator = strategy.make_input_fn_iterator(input_fn)
      replica_results = strategy.experimental_run(replica_fn, iterator)
    ```

    The `tf.data.Dataset` returned by `input_fn` should have a per-replica
    batch size, which may be computed using
    `input_context.get_per_replica_batch_size`.

    Args:
      input_fn: A function taking a `tf.distribute.InputContext` object and
        returning a `tf.data.Dataset`.
      replication_mode: an enum value of `tf.distribute.InputReplicationMode`.
        Only `PER_WORKER` is supported currently, which means there will be
        a single call to `input_fn` per worker. Replicas will dequeue from the
        local `tf.data.Dataset` on their worker.

    Returns:
      An iterator object that should first be `.initialize()`-ed. It may then
      either be passed to `strategy.experimental_run()` or you can
      `iterator.get_next()` to get the next value to pass to
      `strategy.extended.call_for_each_replica()`.
    """
    return super(StrategyV1, self).make_input_fn_iterator(
        input_fn, replication_mode)

  def experimental_make_numpy_dataset(self, numpy_input, session=None):
    """Makes a tf.data.Dataset for input provided via a numpy array.

    This avoids adding `numpy_input` as a large constant in the graph,
    and copies the data to the machine or machines that will be processing
    the input.

    Note that you will likely need to use
    tf.distribute.Strategy.experimental_distribute_dataset
    with the returned dataset to further distribute it with the strategy.

    Example:
    ```
    numpy_input = np.ones([10], dtype=np.float32)
    dataset = strategy.experimental_make_numpy_dataset(numpy_input)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    ```

    Args:
      numpy_input: A nest of NumPy input arrays that will be converted into a
      dataset. Note that lists of Numpy arrays are stacked, as that is normal
      `tf.data.Dataset` behavior.
      session: (TensorFlow v1.x graph execution only) A session used for
        initialization.

    Returns:
      A `tf.data.Dataset` representing `numpy_input`.
    """
    return self.extended.experimental_make_numpy_dataset(
        numpy_input, session=session)

  @deprecated(
      None,
      "This method is not available in TF 2.x. Please switch to using `run` instead."
  )
  def experimental_run(self, fn, input_iterator=None):  # pylint: disable=useless-super-delegation
    """Runs ops in `fn` on each replica, with inputs from `input_iterator`.

    DEPRECATED: This method is not available in TF 2.x. Please switch
    to using `run` instead.

    When eager execution is enabled, executes ops specified by `fn` on each
    replica. Otherwise, builds a graph to execute the ops on each replica.

    Each replica will take a single, different input from the inputs provided by
    one `get_next` call on the input iterator.

    `fn` may call `tf.distribute.get_replica_context()` to access members such
    as `replica_id_in_sync_group`.

    IMPORTANT: Depending on the `tf.distribute.Strategy` implementation being
    used, and whether eager execution is enabled, `fn` may be called one or more
    times (once for each replica).

    Args:
      fn: The function to run. The inputs to the function must match the outputs
        of `input_iterator.get_next()`. The output must be a `tf.nest` of
        `Tensor`s.
      input_iterator: (Optional) input iterator from which the inputs are taken.

    Returns:
      Merged return value of `fn` across replicas. The structure of the return
      value is the same as the return value from `fn`. Each element in the
      structure can either be `PerReplica` (if the values are unsynchronized),
      `Mirrored` (if the values are kept in sync), or `Tensor` (if running on a
      single replica).
    """
    return super(StrategyV1, self).experimental_run(
        fn, input_iterator)

  def reduce(self, reduce_op, value, axis=None):
    return super(StrategyV1, self).reduce(reduce_op, value, axis)

  reduce.__doc__ = StrategyBase.reduce.__doc__

  def update_config_proto(self, config_proto):
    """Returns a copy of `config_proto` modified for use with this strategy.

    DEPRECATED: This method is not available in TF 2.x.

    The updated config has something needed to run a strategy, e.g.
    configuration to run collective ops, or device filters to improve
    distributed training performance.

    Args:
      config_proto: a `tf.ConfigProto` object.

    Returns:
      The updated copy of the `config_proto`.
    """
    return self._extended._update_config_proto(config_proto)  # pylint: disable=protected-access


# NOTE(joshl): For any strategy that needs to support tf.compat.v1,
# instead descend from StrategyExtendedV1.
@tf_export("distribute.StrategyExtended", v1=[])
class StrategyExtendedV2(object):
  """Additional APIs for algorithms that need to be distribution-aware.

  Note: For most usage of `tf.distribute.Strategy`, there should be no need to
  call these methods, since TensorFlow libraries (such as optimizers) already
  call these methods when needed on your behalf.


  Some common use cases of functions on this page:

  * _Locality_

  `tf.distribute.DistributedValues` can have the same _locality_ as a
  _distributed variable_, which leads to a mirrored value residing on the same
  devices as the variable (as opposed to the compute devices). Such values may
  be passed to a call to `tf.distribute.StrategyExtended.update` to update the
  value of a variable. You may use
  `tf.distribute.StrategyExtended.colocate_vars_with` to give a variable the
  same locality as another variable. You may convert a "PerReplica" value to a
  variable's locality by using `tf.distribute.StrategyExtended.reduce_to` or
  `tf.distribute.StrategyExtended.batch_reduce_to`.

  * _How to update a distributed variable_

  A distributed variable is variables created on multiple devices. As discussed
  in the [glossary](https://www.tensorflow.org/api_docs/python/tf/distribute),
  mirrored variable and SyncOnRead variable are two examples. The standard
  pattern for updating distributed variables is to:

  1. In your function passed to `tf.distribute.Strategy.run`,
     compute a list of (update, variable) pairs. For example, the update might
     be a gradient of the loss with respect to the variable.
  2. Switch to cross-replica mode by calling
     `tf.distribute.get_replica_context().merge_call()` with the updates and
     variables as arguments.
  3. Call
     `tf.distribute.StrategyExtended.reduce_to(VariableAggregation.SUM, t, v)`
     (for one variable) or `tf.distribute.StrategyExtended.batch_reduce_to`
     (for a list of variables) to sum the updates.
  4. Call `tf.distribute.StrategyExtended.update(v)` for each variable to update
     its value.

  Steps 2 through 4 are done automatically by class
  `tf.keras.optimizers.Optimizer` if you call its
  `tf.keras.optimizers.Optimizer.apply_gradients` method in a replica context.

  In fact, a higher-level solution to update a distributed variable is by
  calling `assign` on the variable as you would do to a regular `tf.Variable`.
  You can call the method in both _replica context_ and _cross-replica context_.
  For a _mirrored variable_, calling `assign` in _replica context_ requires you
  to specify the `aggregation` type in the variable constructor. In that case,
  the context switching and sync described in steps 2 through 4 are handled for
  you. If you call `assign` on _mirrored variable_ in _cross-replica context_,
  you can only assign a single value or assign values from another mirrored
  variable or a mirrored `tf.distribute.DistributedValues`. For a _SyncOnRead
  variable_, in _replica context_, you can simply call `assign` on it and no
  aggregation happens under the hood. In _cross-replica context_, you can only
  assign a single value to a SyncOnRead variable. One example case is restoring
  from a checkpoint: if the `aggregation` type of the variable is
  `tf.VariableAggregation.SUM`, it is assumed that replica values were added
  before checkpointing, so at the time of restoring, the value is divided by
  the number of replicas and then assigned to each replica; if the `aggregation`
  type is `tf.VariableAggregation.MEAN`, the value is assigned to each replica
  directly.

  """

  def __init__(self, container_strategy):
    self._container_strategy_weakref = weakref.ref(container_strategy)
    self._default_device = None
    # This property is used to determine if we should set drop_remainder=True
    # when creating Datasets from numpy array inputs.
    self._require_static_shapes = False

  def _resource_creator_scope(self):
    """Returns one or a list of ops.resource_creator_scope for some Strategy."""
    return None

  def _container_strategy(self):
    """Get the containing `tf.distribute.Strategy`.

    This should not generally be needed except when creating a new
    `ReplicaContext` and to validate that the caller is in the correct
    `scope()`.

    Returns:
      The `tf.distribute.Strategy` such that `strategy.extended` is `self`.
    """
    container_strategy = self._container_strategy_weakref()
    assert container_strategy is not None
    return container_strategy

  def _scope(self, strategy):
    """Implementation of tf.distribute.Strategy.scope()."""

    def creator_with_resource_vars(next_creator, **kwargs):
      """Variable creator to use in `_CurrentDistributionContext`."""
      if ops.inside_function():
        if_graph_building = "graph_building"
      else:
        if_graph_building = "not_graph_building"

      with monitoring.MonitoredTimer(distributed_variable_creation_time_counter.get_cell(strategy.__class__.__name__, if_graph_building)):
        _require_strategy_scope_extended(self)
        kwargs["use_resource"] = True
        kwargs["distribute_strategy"] = strategy

        # Unwrap `initial_value` if it is a `CheckpointInitialValue` to avoid
        # dereferencing a `Tensor` that is without a `name`. We still need to
        # propagate the metadata it's holding.
        if isinstance(kwargs["initial_value"], trackable.CheckpointInitialValue):
          checkpoint_restore_uid = kwargs[
              "initial_value"].checkpoint_position.restore_uid
          kwargs["initial_value"] = kwargs["initial_value"].wrapped_value
        elif isinstance(kwargs["initial_value"],
                        trackable.CheckpointInitialValueCallable):
          checkpoint_restore_uid = kwargs[
              "initial_value"].checkpoint_position.restore_uid
        elif (isinstance(kwargs["initial_value"], functools.partial) and
              isinstance(kwargs["initial_value"].func,
                         trackable.CheckpointInitialValueCallable)):
          # Some libraries (e.g, Keras) create partial function out of initializer
          # to bind shape/dtype, for example:
          #  initial_val = functools.partial(initializer, shape, dtype=dtype)
          # Therefore to get the restore_uid we need to examine the "func" of
          # the partial function.
          checkpoint_restore_uid = kwargs[
              "initial_value"].func.checkpoint_position.restore_uid
        else:
          checkpoint_restore_uid = None

        created = self._create_variable(next_creator, **kwargs)

        if checkpoint_restore_uid is not None:
          # pylint: disable=protected-access
          # Let the checkpointing infrastructure know that the variable was
          # already restored so it doesn't waste memory loading the value again.
          # In this case of CheckpointInitialValueCallable this may already be
          # done by the final variable creator, but it doesn't hurt to do it
          # again.
          created._maybe_initialize_trackable()
          created._update_uid = checkpoint_restore_uid
          # pylint: enable=protected-access
        return created

    def distributed_getter(getter, *args, **kwargs):
      if not self._allow_variable_partition():
        if kwargs.pop("partitioner", None) is not None:
          tf_logging.log_first_n(
              tf_logging.WARN, "Partitioned variables are disabled when using "
              "current tf.distribute.Strategy.", 1)
      return getter(*args, **kwargs)

    return _CurrentDistributionContext(
        strategy,
        variable_scope.variable_creator_scope(creator_with_resource_vars),
        variable_scope.variable_scope(
            variable_scope.get_variable_scope(),
            custom_getter=distributed_getter),
        strategy.extended._resource_creator_scope(),  # pylint: disable=protected-access
        self._default_device)

  def _allow_variable_partition(self):
    return False

  def _create_variable(self, next_creator, **kwargs):
    # Note: should support "colocate_with" argument.
    raise NotImplementedError("must be implemented in descendants")

  def variable_created_in_scope(self, v):
    """Tests whether `v` was created while this strategy scope was active.

    Variables created inside the strategy scope are "owned" by it:

    >>> strategy = tf.distribute.MirroredStrategy()
    >>> with strategy.scope():
    ...   v = tf.Variable(1.)
    >>> strategy.extended.variable_created_in_scope(v)
    True

    Variables created outside the strategy are not owned by it:

    >>> strategy = tf.distribute.MirroredStrategy()
    >>> v = tf.Variable(1.)
    >>> strategy.extended.variable_created_in_scope(v)
    False

    Args:
      v: A `tf.Variable` instance.

    Returns:
      True if `v` was created inside the scope, False if not.
    """
    return v._distribute_strategy == self._container_strategy_weakref()  # pylint: disable=protected-access

  def colocate_vars_with(self, colocate_with_variable):
    """Scope that controls which devices variables will be created on.

    No operations should be added to the graph inside this scope, it
    should only be used when creating variables (some implementations
    work by changing variable creation, others work by using a
    tf.compat.v1.colocate_with() scope).

    This may only be used inside `self.scope()`.

    Example usage:

    ```
    with strategy.scope():
      var1 = tf.Variable(...)
      with strategy.extended.colocate_vars_with(var1):
        # var2 and var3 will be created on the same device(s) as var1
        var2 = tf.Variable(...)
        var3 = tf.Variable(...)

      def fn(v1, v2, v3):
        # operates on v1 from var1, v2 from var2, and v3 from var3

      # `fn` runs on every device `var1` is on, `var2` and `var3` will be there
      # too.
      strategy.extended.update(var1, fn, args=(var2, var3))
    ```

    Args:
      colocate_with_variable: A variable created in this strategy's `scope()`.
        Variables created while in the returned context manager will be on the
        same set of devices as `colocate_with_variable`.

    Returns:
      A context manager.
    """

    def create_colocated_variable(next_creator, **kwargs):
      _require_strategy_scope_extended(self)
      kwargs["use_resource"] = True
      kwargs["colocate_with"] = colocate_with_variable
      return next_creator(**kwargs)

    _require_strategy_scope_extended(self)
    self._validate_colocate_with_variable(colocate_with_variable)
    return variable_scope.variable_creator_scope(create_colocated_variable)

  def _validate_colocate_with_variable(self, colocate_with_variable):
    """Validate `colocate_with_variable` argument to `colocate_vars_with`."""
    pass

  def _make_dataset_iterator(self, dataset):
    raise NotImplementedError("must be implemented in descendants")

  def _make_input_fn_iterator(self, input_fn, replication_mode):
    raise NotImplementedError("must be implemented in descendants")

  def _experimental_distribute_dataset(self, dataset, options):
    raise NotImplementedError("must be implemented in descendants")

  def _distribute_datasets_from_function(self, dataset_fn, options):
    raise NotImplementedError("must be implemented in descendants")

  def _experimental_distribute_values_from_function(self, value_fn):
    raise NotImplementedError("must be implemented in descendants")

  def _reduce(self, reduce_op, value):
    # Default implementation until we have an implementation for each strategy.
    dst = device_util.current() or self._default_device or "/device:CPU:0"
    return self._local_results(self.reduce_to(reduce_op, value, dst))[0]

  def reduce_to(self, reduce_op, value, destinations, options=None):
    """Combine (via e.g. sum or mean) values across replicas.

    `reduce_to` aggregates `tf.distribute.DistributedValues` and distributed
    variables. It supports both dense values and `tf.IndexedSlices`.

    This API currently can only be called in cross-replica context. Other
    variants to reduce values across replicas are:
    * `tf.distribute.StrategyExtended.batch_reduce_to`: the batch version of
      this API.
    * `tf.distribute.ReplicaContext.all_reduce`: the counterpart of this API
      in replica context. It supports both batched and non-batched all-reduce.
    * `tf.distribute.Strategy.reduce`: a more convenient method to reduce
      to the host in cross-replica context.

    `destinations` specifies where to reduce the value to, e.g. "GPU:0". You can
    also pass in a `Tensor`, and the destinations will be the device of that
    tensor. For all-reduce, pass the same to `value` and `destinations`.

    It can be used in `tf.distribute.ReplicaContext.merge_call` to write code
    that works for all `tf.distribute.Strategy`.

    @tf.function
    def step_fn(var):

      def merge_fn(strategy, value, var):
        # All-reduce the value. Note that `value` here is a
        # `tf.distribute.DistributedValues`.
        reduced = strategy.extended.reduce_to(tf.distribute.ReduceOp.SUM,
            value, destinations=var)
        strategy.extended.update(var, lambda var, value: var.assign(value),
            args=(reduced,))

      value = tf.identity(1.)
      tf.distribute.get_replica_context().merge_call(merge_fn,
        args=(value, var))

    def run(strategy):
      with strategy.scope():
        v = tf.Variable(0.)
        strategy.run(step_fn, args=(v,))
        return v

    run(tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"]))
    MirroredVariable:{
      0: <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>,
      1: <tf.Variable 'Variable/replica_1:0' shape=() dtype=float32, numpy=2.0>
    }
    run(tf.distribute.experimental.CentralStorageStrategy(
        compute_devices=["GPU:0", "GPU:1"], parameter_device="CPU:0"))
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>
    run(tf.distribute.OneDeviceStrategy("GPU:0"))
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>

    Args:
      reduce_op: a `tf.distribute.ReduceOp` value specifying how values should
        be combined. Allows using string representation of the enum such as
        "SUM", "MEAN".
      value: a `tf.distribute.DistributedValues`, or a `tf.Tensor` like object.
      destinations: a `tf.distribute.DistributedValues`, a `tf.Variable`, a
        `tf.Tensor` alike object, or a device string. It specifies the devices
        to reduce to. To perform an all-reduce, pass the same to `value` and
        `destinations`. Note that if it's a `tf.Variable`, the value is reduced
        to the devices of that variable, and this method doesn't update the
        variable.
      options: a `tf.distribute.experimental.CommunicationOptions`. Options to
        perform collective operations. This overrides the default options if the
        `tf.distribute.Strategy` takes one in the constructor. See
        `tf.distribute.experimental.CommunicationOptions` for details of the
        options.

    Returns:
      A tensor or value reduced to `destinations`.
    """
    if options is None:
      options = collective_util.Options()
    _require_cross_replica_or_default_context_extended(self)
    assert not isinstance(destinations, (list, tuple))
    assert not isinstance(reduce_op, variable_scope.VariableAggregation)
    if isinstance(reduce_op, six.string_types):
      reduce_op = reduce_util.ReduceOp(reduce_op.upper())
    assert (reduce_op == reduce_util.ReduceOp.SUM or
            reduce_op == reduce_util.ReduceOp.MEAN)
    return self._reduce_to(reduce_op, value, destinations, options)

  def _reduce_to(self, reduce_op, value, destinations, options):
    raise NotImplementedError("must be implemented in descendants")

  def batch_reduce_to(self, reduce_op, value_destination_pairs, options=None):
    """Combine multiple `reduce_to` calls into one for faster execution.

    Similar to `reduce_to`, but accepts a list of (value, destinations) pairs.
    It's more efficient than reduce each value separately.

    This API currently can only be called in cross-replica context. Other
    variants to reduce values across replicas are:
    * `tf.distribute.StrategyExtended.reduce_to`: the non-batch version of
      this API.
    * `tf.distribute.ReplicaContext.all_reduce`: the counterpart of this API
      in replica context. It supports both batched and non-batched all-reduce.
    * `tf.distribute.Strategy.reduce`: a more convenient method to reduce
      to the host in cross-replica context.

    See `reduce_to` for more information.

    @tf.function
    def step_fn(var):

      def merge_fn(strategy, value, var):
        # All-reduce the value. Note that `value` here is a
        # `tf.distribute.DistributedValues`.
        reduced = strategy.extended.batch_reduce_to(
            tf.distribute.ReduceOp.SUM, [(value, var)])[0]
        strategy.extended.update(var, lambda var, value: var.assign(value),
            args=(reduced,))

      value = tf.identity(1.)
      tf.distribute.get_replica_context().merge_call(merge_fn,
        args=(value, var))

    def run(strategy):
      with strategy.scope():
        v = tf.Variable(0.)
        strategy.run(step_fn, args=(v,))
        return v

    run(tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"]))
    MirroredVariable:{
      0: <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>,
      1: <tf.Variable 'Variable/replica_1:0' shape=() dtype=float32, numpy=2.0>
    }
    run(tf.distribute.experimental.CentralStorageStrategy(
        compute_devices=["GPU:0", "GPU:1"], parameter_device="CPU:0"))
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>
    run(tf.distribute.OneDeviceStrategy("GPU:0"))
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>

    Args:
      reduce_op: a `tf.distribute.ReduceOp` value specifying how values should
        be combined. Allows using string representation of the enum such as
        "SUM", "MEAN".
      value_destination_pairs: a sequence of (value, destinations) pairs. See
        `tf.distribute.Strategy.reduce_to` for descriptions.
      options: a `tf.distribute.experimental.CommunicationOptions`. Options to
        perform collective operations. This overrides the default options if the
        `tf.distribute.Strategy` takes one in the constructor. See
        `tf.distribute.experimental.CommunicationOptions` for details of the
        options.

    Returns:
      A list of reduced values, one per pair in `value_destination_pairs`.
    """
    if options is None:
      options = collective_util.Options()
    _require_cross_replica_or_default_context_extended(self)
    assert not isinstance(reduce_op, variable_scope.VariableAggregation)
    if isinstance(reduce_op, six.string_types):
      reduce_op = reduce_util.ReduceOp(reduce_op.upper())
    return self._batch_reduce_to(reduce_op, value_destination_pairs, options)

  def _batch_reduce_to(self, reduce_op, value_destination_pairs, options):
    return [
        self.reduce_to(reduce_op, t, destinations=v, options=options)
        for t, v in value_destination_pairs
    ]

  def _replica_ctx_all_reduce(self, reduce_op, value, options=None):
    """All-reduce `value` across all replicas so that all get the final result.

    If `value` is a nested structure of tensors, all-reduces of these tensors
    will be batched when possible. `options` can be set to hint the batching
    behavior.

    This API must be called in a replica context.

    Args:
      reduce_op: A `tf.distribute.ReduceOp` value specifying how values should
        be combined.
      value: Value to be reduced. A tensor or a nested structure of tensors.
      options: A `tf.distribute.experimental.CommunicationOptions`. Options to
        perform collective operations. This overrides the default options if the
        `tf.distribute.Strategy` takes one in the constructor.

    Returns:
      A tensor or a nested strucutre of tensors with the reduced values. The
      structure is the same as `value`.
    """
    if options is None:
      options = collective_util.Options()
    replica_context = get_replica_context()
    assert replica_context, (
        "`StrategyExtended._replica_ctx_all_reduce` must be called in"
        " a replica context")

    def merge_fn(_, flat_value):
      return self.batch_reduce_to(reduce_op, [(v, v) for v in flat_value],
                                  options)

    reduced = replica_context.merge_call(merge_fn, args=(nest.flatten(value),))
    return nest.pack_sequence_as(value, reduced)

  def _replica_ctx_update(self, var, fn, args=(), kwargs=None, group=True):
    """Run `fn` with `args` and `kwargs` to update `var`."""
    # This method is called by ReplicaContext.update. Strategies who'd like to
    # remove merge_call in this path should override this method.
    replica_context = get_replica_context()
    if not replica_context:
      raise ValueError("`StrategyExtended._replica_ctx_update` must be called "
                       "in a replica context.")

    def merge_fn(_, *merged_args, **merged_kwargs):
      return self.update(var, fn, merged_args, merged_kwargs, group=group)

    return replica_context.merge_call(merge_fn, args=args, kwargs=kwargs)

  def _gather_to(self, value, destinations, axis, options=None):
    """Gather `value` across replicas along axis-th dimension to `destinations`.

    `gather_to` gathers `tf.distribute.DistributedValues` or `tf.Tensor`-like
    object, along `axis`-th dimension. It supports only dense tensors but NOT
    sparse tensor. This API can only be called in cross-replica context.

    Args:
      value: a `tf.distribute.DistributedValues`, or a `tf.Tensor` like object.
      destinations: a `tf.distribute.DistributedValues`, a `tf.Variable`, a
        `tf.Tensor` alike object, or a device string. It specifies the devices
        to reduce to. To perform an all-gather, pass the same to `value` and
        `destinations`. Note that if it's a `tf.Variable`, the value is reduced
        to the devices of that variable, and this method doesn't update the
        variable.
      axis: 0-D int32 Tensor. Dimension along which to gather. Must be in the
        range [0, rank(value)).
      options: a `tf.distribute.experimental.CommunicationOptions`. Options to
        perform collective operations. This overrides the default options if the
        `tf.distribute.Strategy` takes one in the constructor. See
        `tf.distribute.experimental.CommunicationOptions` for details of the
        options.

    Returns:
      A tensor or value gathered to `destinations`.
    """
    _require_cross_replica_or_default_context_extended(self)
    assert not isinstance(destinations, (list, tuple))
    if options is None:
      options = collective_util.Options()
    return self._gather_to_implementation(value, destinations, axis, options)

  def _gather_to_implementation(self, value, destinations, axis, options):
    raise NotImplementedError("_gather_to must be implemented in descendants")

  def _batch_gather_to(self, value_destination_pairs, axis, options=None):
    _require_cross_replica_or_default_context_extended(self)
    if options is None:
      options = collective_util.Options()
    return [
        self._gather_to(t, destinations=v, axis=axis, options=options)
        for t, v in value_destination_pairs
    ]

  def update(self, var, fn, args=(), kwargs=None, group=True):
    """Run `fn` to update `var` using inputs mirrored to the same devices.

    `tf.distribute.StrategyExtended.update` takes a distributed variable `var`
    to be updated, an update function `fn`, and `args` and `kwargs` for `fn`. It
    applies `fn` to each component variable of `var` and passes corresponding
    values from `args` and `kwargs`. Neither `args` nor `kwargs` may contain
    per-replica values. If they contain mirrored values, they will be unwrapped
    before calling `fn`. For example, `fn` can be `assign_add` and `args` can be
    a mirrored DistributedValues where each component contains the value to be
    added to this mirrored variable `var`. Calling `update` will call
    `assign_add` on each component variable of `var` with the corresponding
    tensor value on that device.

    Example usage:

    ```python
    strategy = tf.distribute.MirroredStrategy(['GPU:0', 'GPU:1']) # With 2
    devices
    with strategy.scope():
      v = tf.Variable(5.0, aggregation=tf.VariableAggregation.SUM)
    def update_fn(v):
      return v.assign(1.0)
    result = strategy.extended.update(v, update_fn)
    # result is
    # Mirrored:{
    #  0: tf.Tensor(1.0, shape=(), dtype=float32),
    #  1: tf.Tensor(1.0, shape=(), dtype=float32)
    # }
    ```

    If `var` is mirrored across multiple devices, then this method implements
    logic as following:

    ```python
    results = {}
    for device, v in var:
      with tf.device(device):
        # args and kwargs will be unwrapped if they are mirrored.
        results[device] = fn(v, *args, **kwargs)
    return merged(results)
    ```

    Otherwise, this method returns `fn(var, *args, **kwargs)` colocated with
    `var`.

    Args:
      var: Variable, possibly mirrored to multiple devices, to operate on.
      fn: Function to call. Should take the variable as the first argument.
      args: Tuple or list. Additional positional arguments to pass to `fn()`.
      kwargs: Dict with keyword arguments to pass to `fn()`.
      group: Boolean. Defaults to True. If False, the return value will be
        unwrapped.

    Returns:
      By default, the merged return value of `fn` across all replicas.  The
      merged result has dependencies to make sure that if it is evaluated at
      all, the side effects (updates) will happen on every replica. If instead
      "group=False" is specified, this function will return a nest of lists
      where each list has an element per replica, and the caller is responsible
      for ensuring all elements are executed.
    """
    # TODO(b/178944108): Update the documentation to relfect the fact that
    # `update` can be called in a replica context.
    if kwargs is None:
      kwargs = {}
    replica_context = get_replica_context()
    # pylint: disable=protected-access
    if (replica_context is None or replica_context is
        _get_default_replica_context()):
      fn = autograph.tf_convert(
          fn, autograph_ctx.control_status_ctx(), convert_by_default=False)
      with self._container_strategy().scope():
        return self._update(var, fn, args, kwargs, group)
    else:
      return self._replica_ctx_update(
          var, fn, args=args, kwargs=kwargs, group=group)

  def _update(self, var, fn, args, kwargs, group):
    raise NotImplementedError("must be implemented in descendants")

  def _local_results(self, val):
    """Returns local results per replica as a tuple."""
    if isinstance(val, ds_types.DistributedValues):
      return val._values  # pylint: disable=protected-access

    if nest.is_nested(val):
      replica_values = []

      def get_values(x, index):
        if isinstance(x, ds_types.DistributedValues):
          return x._values[index]  # pylint: disable=protected-access
        return x

      for i in range(len(self.worker_devices)):
        replica_values.append(
            nest.map_structure(
                lambda x: get_values(x, i),  # pylint: disable=cell-var-from-loop
                val))
      return tuple(replica_values)
    return (val,)

  def value_container(self, value):
    """Returns the container that this per-replica `value` belongs to.

    Args:
      value: A value returned by `run()` or a variable created in `scope()`.

    Returns:
      A container that `value` belongs to.
      If value does not belong to any container (including the case of
      container having been destroyed), returns the value itself.
      `value in experimental_local_results(value_container(value))` will
      always be true.
    """
    raise NotImplementedError("must be implemented in descendants")

  def _group(self, value, name=None):
    """Implementation of `group`."""
    value = nest.flatten(self._local_results(value))

    if len(value) != 1 or name is not None:
      return control_flow_ops.group(value, name=name)
    # Special handling for the common case of one op.
    v, = value
    if hasattr(v, "op"):
      v = v.op
    return v

  @property
  def experimental_require_static_shapes(self):
    """Returns `True` if static shape is required; `False` otherwise."""
    return self._require_static_shapes

  @property
  def _num_replicas_in_sync(self):
    """Returns number of replicas over which gradients are aggregated."""
    raise NotImplementedError("must be implemented in descendants")

  @property
  def worker_devices(self):
    """Returns the tuple of all devices used to for compute replica execution.
    """
    # TODO(joshl): More docstring
    raise NotImplementedError("must be implemented in descendants")

  @property
  def parameter_devices(self):
    """Returns the tuple of all devices used to place variables."""
    # TODO(joshl): More docstring
    raise NotImplementedError("must be implemented in descendants")

  def _configure(self,
                 session_config=None,
                 cluster_spec=None,
                 task_type=None,
                 task_id=None):
    """Configures the strategy class."""
    del session_config, cluster_spec, task_type, task_id

  def _update_config_proto(self, config_proto):
    return copy.deepcopy(config_proto)

  def _in_multi_worker_mode(self):
    """Whether this strategy indicates working in multi-worker settings.

    Multi-worker training refers to the setup where the training is
    distributed across multiple workers, as opposed to the case where
    only a local process performs the training. This function is
    used by higher-level APIs such as Keras' `model.fit()` to infer
    for example whether or not a distribute coordinator should be run,
    and thus TensorFlow servers should be started for communication
    with other servers in the cluster, or whether or not saving/restoring
    checkpoints is relevant for preemption fault tolerance.

    Subclasses should override this to provide whether the strategy is
    currently in multi-worker setup.

    Experimental. Signature and implementation are subject to change.
    """
    raise NotImplementedError("must be implemented in descendants")


@tf_export(v1=["distribute.StrategyExtended"])  # pylint: disable=missing-docstring
class StrategyExtendedV1(StrategyExtendedV2):

  __doc__ = StrategyExtendedV2.__doc__

  def experimental_make_numpy_dataset(self, numpy_input, session=None):
    """Makes a dataset for input provided via a numpy array.

    This avoids adding `numpy_input` as a large constant in the graph,
    and copies the data to the machine or machines that will be processing
    the input.

    Args:
      numpy_input: A nest of NumPy input arrays that will be distributed evenly
        across all replicas. Note that lists of Numpy arrays are stacked, as
        that is normal `tf.data.Dataset` behavior.
      session: (TensorFlow v1.x graph execution only) A session used for
        initialization.

    Returns:
      A `tf.data.Dataset` representing `numpy_input`.
    """
    _require_cross_replica_or_default_context_extended(self)
    return self._experimental_make_numpy_dataset(numpy_input, session=session)

  def _experimental_make_numpy_dataset(self, numpy_input, session):
    raise NotImplementedError("must be implemented in descendants")

  def broadcast_to(self, tensor, destinations):
    """Mirror a tensor on one device to all worker devices.

    Args:
      tensor: A Tensor value to broadcast.
      destinations: A mirrored variable or device string specifying the
        destination devices to copy `tensor` to.

    Returns:
      A value mirrored to `destinations` devices.
    """
    assert destinations is not None  # from old strategy.broadcast()
    # TODO(joshl): More docstring
    _require_cross_replica_or_default_context_extended(self)
    assert not isinstance(destinations, (list, tuple))
    return self._broadcast_to(tensor, destinations)

  def _broadcast_to(self, tensor, destinations):
    raise NotImplementedError("must be implemented in descendants")

  @deprecated(None, "please use `run` instead.")
  def experimental_run_steps_on_iterator(self,
                                         fn,
                                         iterator,
                                         iterations=1,
                                         initial_loop_values=None):
    """DEPRECATED: please use `run` instead.

    Run `fn` with input from `iterator` for `iterations` times.

    This method can be used to run a step function for training a number of
    times using input from a dataset.

    Args:
      fn: function to run using this distribution strategy. The function must
        have the following signature: `def fn(context, inputs)`. `context` is an
          instance of `MultiStepContext` that will be passed when `fn` is run.
          `context` can be used to specify the outputs to be returned from `fn`
          by calling `context.set_last_step_output`. It can also be used to
          capture non tensor outputs by `context.set_non_tensor_output`. See
          `MultiStepContext` documentation for more information. `inputs` will
          have same type/structure as `iterator.get_next()`. Typically, `fn`
          will use `call_for_each_replica` method of the strategy to distribute
          the computation over multiple replicas.
      iterator: Iterator of a dataset that represents the input for `fn`. The
        caller is responsible for initializing the iterator as needed.
      iterations: (Optional) Number of iterations that `fn` should be run.
        Defaults to 1.
      initial_loop_values: (Optional) Initial values to be passed into the
        loop that runs `fn`. Defaults to `None`. # TODO(priyag): Remove
          initial_loop_values argument when we have a mechanism to infer the
          outputs of `fn`.

    Returns:
      Returns the `MultiStepContext` object which has the following properties,
      among other things:
        - run_op: An op that runs `fn` `iterations` times.
        - last_step_outputs: A dictionary containing tensors set using
        `context.set_last_step_output`. Evaluating this returns the value of
        the tensors after the last iteration.
        - non_tensor_outputs: A dictionary containing anything that was set by
          `fn` by calling `context.set_non_tensor_output`.
    """
    _require_cross_replica_or_default_context_extended(self)
    with self._container_strategy().scope():
      return self._experimental_run_steps_on_iterator(fn, iterator, iterations,
                                                      initial_loop_values)

  def _experimental_run_steps_on_iterator(self, fn, iterator, iterations,
                                          initial_loop_values):
    raise NotImplementedError("must be implemented in descendants")

  def call_for_each_replica(self, fn, args=(), kwargs=None):
    """Run `fn` once per replica.

    `fn` may call `tf.get_replica_context()` to access methods such as
    `replica_id_in_sync_group` and `merge_call()`.

    `merge_call()` is used to communicate between the replicas and
    re-enter the cross-replica context. All replicas pause their execution
    having encountered a `merge_call()` call. After that the
    `merge_fn`-function is executed. Its results are then unwrapped and
    given back to each replica call. After that execution resumes until
    `fn` is complete or encounters another `merge_call()`.  Example:

    ```python
    # Called once in "cross-replica" context.
    def merge_fn(distribution, three_plus_replica_id):
      # sum the values across replicas
      return sum(distribution.experimental_local_results(three_plus_replica_id))

    # Called once per replica in `distribution`, in a "replica" context.
    def fn(three):
      replica_ctx = tf.get_replica_context()
      v = three + replica_ctx.replica_id_in_sync_group
      # Computes the sum of the `v` values across all replicas.
      s = replica_ctx.merge_call(merge_fn, args=(v,))
      return s + v

    with distribution.scope():
      # in "cross-replica" context
      ...
      merged_results = distribution.run(fn, args=[3])
      # merged_results has the values from every replica execution of `fn`.
      # This statement prints a list:
      print(distribution.experimental_local_results(merged_results))
    ```

    Args:
      fn: function to run (will be run once per replica).
      args: Tuple or list with positional arguments for `fn`.
      kwargs: Dict with keyword arguments for `fn`.

    Returns:
      Merged return value of `fn` across all replicas.
    """
    _require_cross_replica_or_default_context_extended(self)
    if kwargs is None:
      kwargs = {}
    with self._container_strategy().scope():
      return self._call_for_each_replica(fn, args, kwargs)

  def _call_for_each_replica(self, fn, args, kwargs):
    raise NotImplementedError("must be implemented in descendants")

  def read_var(self, v):
    """Reads the value of a variable.

    Returns the aggregate value of a replica-local variable, or the
    (read-only) value of any other variable.

    Args:
      v: A variable allocated within the scope of this `tf.distribute.Strategy`.

    Returns:
      A tensor representing the value of `v`, aggregated across replicas if
      necessary.
    """
    raise NotImplementedError("must be implemented in descendants")

  def update_non_slot(
      self, colocate_with, fn, args=(), kwargs=None, group=True):
    """Runs `fn(*args, **kwargs)` on `colocate_with` devices.

    Used to update non-slot variables.

    DEPRECATED: TF 1.x ONLY.

    Args:
      colocate_with: Devices returned by `non_slot_devices()`.
      fn: Function to execute.
      args: Tuple or list. Positional arguments to pass to `fn()`.
      kwargs: Dict with keyword arguments to pass to `fn()`.
      group: Boolean. Defaults to True. If False, the return value will be
        unwrapped.

    Returns:
      Return value of `fn`, possibly merged across devices.
    """
    _require_cross_replica_or_default_context_extended(self)
    if kwargs is None:
      kwargs = {}
    fn = autograph.tf_convert(
        fn, autograph_ctx.control_status_ctx(), convert_by_default=False)
    with self._container_strategy().scope():
      return self._update_non_slot(colocate_with, fn, args, kwargs, group)

  def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
    raise NotImplementedError("must be implemented in descendants")

  def non_slot_devices(self, var_list):
    """Device(s) for non-slot variables.

    DEPRECATED: TF 1.x ONLY.

    This method returns non-slot devices where non-slot variables are placed.
    Users can create non-slot variables on these devices by using a block:

    ```python
    with tf.distribute.StrategyExtended.colocate_vars_with(tf.distribute.StrategyExtended.non_slot_devices(...)):
      ...
    ```

    Args:
      var_list: The list of variables being optimized, needed with the
        default `tf.distribute.Strategy`.
    Returns:
      A sequence of devices for non-slot variables.
    """
    raise NotImplementedError("must be implemented in descendants")

  def _use_merge_call(self):
    """Whether to use merge-calls inside the distributed strategy."""
    return True

  @property
  def experimental_between_graph(self):
    """Whether the strategy uses between-graph replication or not.

      This is expected to return a constant value that will not be changed
      throughout its life cycle.
    """
    raise NotImplementedError("must be implemented in descendants")

  @property
  def experimental_should_init(self):
    """Whether initialization is needed."""
    raise NotImplementedError("must be implemented in descendants")

  @property
  def should_checkpoint(self):
    """Whether checkpointing is needed."""
    raise NotImplementedError("must be implemented in descendants")

  @property
  def should_save_summary(self):
    """Whether saving summaries is needed."""
    raise NotImplementedError("must be implemented in descendants")


# A note about the difference between the context managers
# `ReplicaContext` (defined here) and `_CurrentDistributionContext`
# (defined above) used by `tf.distribute.Strategy.scope()`:
#
# * a ReplicaContext is only present during a `run()`
#   call (except during a `merge_run` call) and in such a scope it
#   will be returned by calls to `get_replica_context()`.  Implementers of new
#   Strategy descendants will frequently also need to
#   define a descendant of ReplicaContext, and are responsible for
#   entering and exiting this context.
#
# * Strategy.scope() sets up a variable_creator scope that
#   changes variable creation calls (e.g. to make mirrored
#   variables). This is intended as an outer scope that users enter once
#   around their model creation and graph definition. There is no
#   anticipated need to define descendants of _CurrentDistributionContext.
#   It sets the current Strategy for purposes of
#   `get_strategy()` and `has_strategy()`
#   and switches the thread mode to a "cross-replica context".
class ReplicaContextBase(object):
  """A class with a collection of APIs that can be called in a replica context.

  You can use `tf.distribute.get_replica_context` to get an instance of
  `ReplicaContext`, which can only be called inside the function passed to
  `tf.distribute.Strategy.run`.

  >>> strategy = tf.distribute.MirroredStrategy(['GPU:0', 'GPU:1'])
  >>> def func():
  ...   replica_context = tf.distribute.get_replica_context()
  ...   return replica_context.replica_id_in_sync_group
  >>> strategy.run(func)
  PerReplica:{
    0: <tf.Tensor: shape=(), dtype=int32, numpy=0>,
    1: <tf.Tensor: shape=(), dtype=int32, numpy=1>
  }
  """

  def __init__(self, strategy, replica_id_in_sync_group):
    """Creates a ReplicaContext.

    Args:
      strategy: A `tf.distribute.Strategy`.
      replica_id_in_sync_group: An integer, a `Tensor` or None. Prefer an
        integer whenever possible to avoid issues with nested `tf.function`. It
        accepts a `Tensor` only to be compatible with `tpu.replicate`.
    """
    self._strategy = strategy
    self._thread_context = _InReplicaThreadMode(  # pylint: disable=protected-access
        self)
    if not (replica_id_in_sync_group is None or
            tensor_util.is_tf_type(replica_id_in_sync_group) or
            isinstance(replica_id_in_sync_group, int)):
      raise ValueError(
          "replica_id_in_sync_group can only be an integer, a Tensor or None.")
    self._replica_id_in_sync_group = replica_id_in_sync_group
    # We need this check because TPUContext extends from ReplicaContext and
    # does not pass a strategy object since it is used by TPUEstimator.
    if strategy:
      self._local_replica_id = strategy.extended._get_local_replica_id(
          replica_id_in_sync_group)
    self._summary_recording_distribution_strategy = None

  @doc_controls.do_not_generate_docs
  def __enter__(self):
    _push_per_thread_mode(self._thread_context)

    def replica_id_is_zero():
      return math_ops.equal(self.replica_id_in_sync_group,
                            constant_op.constant(0))

    summary_state = summary_ops_v2._summary_state  # pylint: disable=protected-access
    self._summary_recording_distribution_strategy = (
        summary_state.is_recording_distribution_strategy)
    summary_state.is_recording_distribution_strategy = replica_id_is_zero

  @doc_controls.do_not_generate_docs
  def __exit__(self, exception_type, exception_value, traceback):
    summary_state = summary_ops_v2._summary_state  # pylint: disable=protected-access
    summary_state.is_recording_distribution_strategy = (
        self._summary_recording_distribution_strategy)
    _pop_per_thread_mode()

  def merge_call(self, merge_fn, args=(), kwargs=None):
    """Merge args across replicas and run `merge_fn` in a cross-replica context.

    This allows communication and coordination when there are multiple calls
    to the step_fn triggered by a call to `strategy.run(step_fn, ...)`.

    See `tf.distribute.Strategy.run` for an explanation.

    If not inside a distributed scope, this is equivalent to:

    ```
    strategy = tf.distribute.get_strategy()
    with cross-replica-context(strategy):
      return merge_fn(strategy, *args, **kwargs)
    ```

    Args:
      merge_fn: Function that joins arguments from threads that are given as
        PerReplica. It accepts `tf.distribute.Strategy` object as
        the first argument.
      args: List or tuple with positional per-thread arguments for `merge_fn`.
      kwargs: Dict with keyword per-thread arguments for `merge_fn`.

    Returns:
      The return value of `merge_fn`, except for `PerReplica` values which are
      unpacked.
    """
    require_replica_context(self)
    if kwargs is None:
      kwargs = {}

    merge_fn = autograph.tf_convert(
        merge_fn, autograph_ctx.control_status_ctx(), convert_by_default=False)
    return self._merge_call(merge_fn, args, kwargs)

  def _merge_call(self, merge_fn, args, kwargs):
    """Default implementation for single replica."""
    _push_per_thread_mode(  # thread-local, so not needed with multiple threads
        _CrossReplicaThreadMode(self._strategy))  # pylint: disable=protected-access
    try:
      return merge_fn(self._strategy, *args, **kwargs)
    finally:
      _pop_per_thread_mode()

  @property
  def num_replicas_in_sync(self):
    """Returns number of replicas that are kept in sync."""
    return self._strategy.num_replicas_in_sync

  @property
  def replica_id_in_sync_group(self):
    """Returns the id of the replica.

    This identifies the replica among all replicas that are kept in sync. The
    value of the replica id can range from 0 to
    `tf.distribute.ReplicaContext.num_replicas_in_sync` - 1.

    NOTE: This is not guaranteed to be the same ID as the XLA replica ID use
    for low-level operations such as collective_permute.

    Returns:
      a `Tensor`.
    """
    # It's important to prefer making the Tensor at call time whenever possible.
    # Keeping Tensors in global states doesn't work well with nested
    # tf.function, since it's possible that the tensor is generated in one func
    # graph, and gets captured by another, which will result in a subtle "An op
    # outside of the function building code is being passed a Graph tensor"
    # error. Making the tensor at call time to ensure it is the same graph where
    # it's used. However to be compatible with tpu.replicate(),
    # self._replica_id_in_sync_group can also be a Tensor.
    if tensor_util.is_tf_type(self._replica_id_in_sync_group):
      return self._replica_id_in_sync_group
    return constant_op.constant(
        self._replica_id_in_sync_group,
        dtypes.int32,
        name="replica_id_in_sync_group")

  @property
  def _replica_id(self):
    """This is the local replica id in a given sync group."""
    return self._local_replica_id

  @property
  def strategy(self):
    """The current `tf.distribute.Strategy` object."""
    return self._strategy

  @property
  @deprecation.deprecated(None, "Please avoid relying on devices property.")
  def devices(self):
    """Returns the devices this replica is to be executed on, as a tuple of strings.

    NOTE: For `tf.distribute.MirroredStrategy` and
    `tf.distribute.experimental.MultiWorkerMirroredStrategy`, this returns a
    nested
    list of device strings, e.g, [["GPU:0"]].
    """
    require_replica_context(self)
    return (device_util.current(),)

  def all_reduce(self, reduce_op, value, options=None):
    """All-reduces `value` across all replicas.

    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> def step_fn():
    ...   ctx = tf.distribute.get_replica_context()
    ...   value = tf.identity(1.)
    ...   return ctx.all_reduce(tf.distribute.ReduceOp.SUM, value)
    >>> strategy.experimental_local_results(strategy.run(step_fn))
    (<tf.Tensor: shape=(), dtype=float32, numpy=2.0>,
     <tf.Tensor: shape=(), dtype=float32, numpy=2.0>)

    It supports batched operations. You can pass a list of values and it
    attempts to batch them when possible. You can also specify `options`
    to indicate the desired batching behavior, e.g. batch the values into
    multiple packs so that they can better overlap with computations.

    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> def step_fn():
    ...   ctx = tf.distribute.get_replica_context()
    ...   value1 = tf.identity(1.)
    ...   value2 = tf.identity(2.)
    ...   return ctx.all_reduce(tf.distribute.ReduceOp.SUM, [value1, value2])
    >>> strategy.experimental_local_results(strategy.run(step_fn))
    ([<tf.Tensor: shape=(), dtype=float32, numpy=2.0>,
    <tf.Tensor: shape=(), dtype=float32, numpy=4.0>],
    [<tf.Tensor: shape=(), dtype=float32, numpy=2.0>,
    <tf.Tensor: shape=(), dtype=float32, numpy=4.0>])

    Note that all replicas need to participate in the all-reduce, otherwise this
    operation hangs. Note that if there're multiple all-reduces, they need to
    execute in the same order on all replicas. Dispatching all-reduce based on
    conditions is usually error-prone.

    Known limitation: if `value` contains `tf.IndexedSlices`, attempting to
    compute gradient w.r.t `value` would result in an error.

    This API currently can only be called in the replica context. Other
    variants to reduce values across replicas are:
    * `tf.distribute.StrategyExtended.reduce_to`: the reduce and all-reduce API
      in the cross-replica context.
    * `tf.distribute.StrategyExtended.batch_reduce_to`: the batched reduce and
      all-reduce API in the cross-replica context.
    * `tf.distribute.Strategy.reduce`: a more convenient method to reduce
      to the host in cross-replica context.

    Args:
      reduce_op: a `tf.distribute.ReduceOp` value specifying how values should
        be combined. Allows using string representation of the enum such as
        "SUM", "MEAN".
      value: a potentially nested structure of `tf.Tensor` or `tf.IndexedSlices` which
        `tf.nest.flatten` accepts. The structure and the shapes of `value` need to be
        same on all replicas.
      options: a `tf.distribute.experimental.CommunicationOptions`. Options to
        perform collective operations. This overrides the default options if the
        `tf.distribute.Strategy` takes one in the constructor. See
        `tf.distribute.experimental.CommunicationOptions` for details of the
        options.

    Returns:
       A nested structure of `tf.Tensor` with the reduced values. The structure
       is the same as `value`.
    """
    flattened_value = nest.flatten(value)
    has_indexed_slices = False

    for v in flattened_value:
      if isinstance(v, indexed_slices.IndexedSlices):
        has_indexed_slices = True

    if isinstance(reduce_op, six.string_types):
      reduce_op = reduce_util.ReduceOp(reduce_op.upper())
    if options is None:
      options = collective_util.Options()

    def batch_all_reduce(strategy, *value_flat):
      return strategy.extended.batch_reduce_to(
          reduce_op, [(v, _batch_reduce_destination(v)) for v in value_flat],
          options)

    # Due to the use of `capture_call_time_value` in collective ops, we have
    # to maintain two branches: one w/ merge_call and one w/o. Details can be
    # found in b/184009754.
    if self._strategy.extended._use_merge_call():  # pylint: disable=protected-access
      # TODO(cjfj): Work out why `batch_reduce` doesn't return the correct grad.
      if has_indexed_slices:
        return nest.pack_sequence_as(
            value,
            self.merge_call(batch_all_reduce, args=flattened_value))

      @custom_gradient.custom_gradient
      def grad_wrapper(*xs):
        ys = self.merge_call(batch_all_reduce, args=xs)
        # The gradient of an all-sum is itself an all-sum (all-mean, likewise).
        return ys, lambda *dy_s: self.all_reduce(reduce_op, dy_s)
      return nest.pack_sequence_as(value, grad_wrapper(*flattened_value))
    else:
      if has_indexed_slices:
        return nest.pack_sequence_as(
            value,
            self._strategy.extended._replica_ctx_all_reduce(  # pylint: disable=protected-access
                reduce_op, flattened_value, options))

      @custom_gradient.custom_gradient
      def grad_wrapper(*xs):
        ys = self._strategy.extended._replica_ctx_all_reduce(  # pylint: disable=protected-access
            reduce_op, xs, options)
        # The gradient of an all-sum is itself an all-sum (all-mean, likewise).
        return ys, lambda *dy_s: self.all_reduce(reduce_op, dy_s)

      return nest.pack_sequence_as(value, grad_wrapper(*flattened_value))

  # TODO(joshl): Implement `start_all_reduce(method, t)` for efficient
  # all-reduce. It would return a function returning the result of reducing `t`
  # across all replicas. The caller would wait to call this function until they
  # needed the reduce result, allowing an efficient implementation:
  # * With eager execution, the reduction could be performed asynchronously
  #   in the background, not blocking until the result was needed.
  # * When constructing a graph, it could batch up all reduction requests up
  #   to that point that the first result is needed. Most likely this can be
  #   implemented in terms of `merge_call()` and `batch_reduce_to()`.


@tf_export("distribute.ReplicaContext", v1=[])
class ReplicaContext(ReplicaContextBase):

  __doc__ = ReplicaContextBase.__doc__

  def all_gather(self, value, axis, options=None):
    """All-gathers `value` across all replicas along `axis`.

    Note: An `all_gather` method can only be called in replica context. For
    a cross-replica context counterpart, see `tf.distribute.Strategy.gather`.
    All replicas need to participate in the all-gather, otherwise this
    operation hangs. So if `all_gather` is called in any replica, it must be
    called in all replicas.

    Note: If there are multiple `all_gather` calls, they need to be executed in
    the same order on all replicas. Dispatching `all_gather` based on conditions
    is usually error-prone.

    For all strategies except `tf.distribute.TPUStrategy`, the input
    `value` on different replicas must have the same rank, and their shapes must
    be the same in all dimensions except the `axis`-th dimension. In other
    words, their shapes cannot be different in a dimension `d` where `d` does
    not equal to the `axis` argument. For example, given a
    `tf.distribute.DistributedValues` with component tensors of shape
    `(1, 2, 3)` and `(1, 3, 3)` on two replicas, you can call
    `all_gather(..., axis=1, ...)` on it, but not `all_gather(..., axis=0, ...)`
    or `all_gather(..., axis=2, ...)`. However, with
    `tf.distribute.TPUStrategy`, all tensors must have exactly the same rank and
    same shape.

    Note: The input `value` must have a non-zero rank. Otherwise, consider using
    `tf.expand_dims` before gathering them.

    You can pass in a single tensor to all-gather:

    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> @tf.function
    ... def gather_value():
    ...   ctx = tf.distribute.get_replica_context()
    ...   local_value = tf.constant([1, 2, 3])
    ...   return ctx.all_gather(local_value, axis=0)
    >>> result = strategy.run(gather_value)
    >>> result
    PerReplica:{
      0: <tf.Tensor: shape=(6,), dtype=int32, numpy=array([1, 2, 3, 1, 2, 3], dtype=int32)>,
      1: <tf.Tensor: shape=(6,), dtype=int32, numpy=array([1, 2, 3, 1, 2, 3], dtype=int32)>
    }
    >>> strategy.experimental_local_results(result)
    (<tf.Tensor: shape=(6,), dtype=int32, numpy=array([1, 2, 3, 1, 2, 3],
    dtype=int32)>,
    <tf.Tensor: shape=(6,), dtype=int32, numpy=array([1, 2, 3, 1, 2, 3],
    dtype=int32)>)


    You can also pass in a nested structure of tensors to all-gather, say, a
    list:

    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> @tf.function
    ... def gather_nest():
    ...   ctx = tf.distribute.get_replica_context()
    ...   value_1 = tf.constant([1, 2, 3])
    ...   value_2 = tf.constant([[1, 2], [3, 4]])
    ...   # all_gather a nest of `tf.distribute.DistributedValues`
    ...   return ctx.all_gather([value_1, value_2], axis=0)
    >>> result = strategy.run(gather_nest)
    >>> result
    [PerReplica:{
      0: <tf.Tensor: shape=(6,), dtype=int32, numpy=array([1, 2, 3, 1, 2, 3], dtype=int32)>,
      1: <tf.Tensor: shape=(6,), dtype=int32, numpy=array([1, 2, 3, 1, 2, 3], dtype=int32)>
    }, PerReplica:{
      0: <tf.Tensor: shape=(4, 2), dtype=int32, numpy=
    array([[1, 2],
           [3, 4],
           [1, 2],
           [3, 4]], dtype=int32)>,
      1: <tf.Tensor: shape=(4, 2), dtype=int32, numpy=
    array([[1, 2],
           [3, 4],
           [1, 2],
           [3, 4]], dtype=int32)>
    }]
    >>> strategy.experimental_local_results(result)
    ([<tf.Tensor: shape=(6,), dtype=int32, numpy=array([1, 2, 3, 1, 2, 3], dtype=int32)>,
    <tf.Tensor: shape=(4, 2), dtype=int32, numpy=
    array([[1, 2],
           [3, 4],
           [1, 2],
           [3, 4]], dtype=int32)>],
           [<tf.Tensor: shape=(6,), dtype=int32, numpy=array([1, 2, 3, 1, 2, 3], dtype=int32)>,
           <tf.Tensor: shape=(4, 2), dtype=int32, numpy=
    array([[1, 2],
           [3, 4],
           [1, 2],
           [3, 4]], dtype=int32)>])


    What if you are all-gathering tensors with different shapes on different
    replicas? Consider the following example with two replicas, where you have
    `value` as a nested structure consisting of two items to all-gather, `a` and
    `b`.

    * On Replica 0, `value` is `{'a': [0], 'b': [[0, 1]]}`.
    * On Replica 1, `value` is `{'a': [1], 'b': [[2, 3], [4, 5]]}`.
    * Result for `all_gather` with `axis=0` (on each of the replicas) is:

      ```
      {'a': [1, 2], 'b': [[0, 1], [2, 3], [4, 5]]}
      ```

    Args:
      value: a nested structure of `tf.Tensor` which `tf.nest.flatten` accepts,
        or a `tf.distribute.DistributedValues` instance. The structure of the
        `tf.Tensor` need to be same on all replicas. The underlying tensor
        constructs can only be dense tensors with non-zero rank, NOT
        `tf.IndexedSlices`.
      axis: 0-D int32 Tensor. Dimension along which to gather.
      options: a `tf.distribute.experimental.CommunicationOptions`. Options to
        perform collective operations. This overrides the default options if the
        `tf.distribute.Strategy` takes one in the constructor. See
        `tf.distribute.experimental.CommunicationOptions` for details of the
        options.

    Returns:
       A nested structure of `tf.Tensor` with the gathered values. The structure
       is the same as `value`.
    """
    for v in nest.flatten(value):
      if isinstance(v, indexed_slices.IndexedSlices):
        raise NotImplementedError("all_gather does not support IndexedSlices")

    if options is None:
      options = collective_util.Options()

    def batch_all_gather(strategy, *value_flat):
      return strategy.extended._batch_gather_to(  # pylint: disable=protected-access
          [(v, _batch_reduce_destination(v)) for v in value_flat], axis,
          options)

    @custom_gradient.custom_gradient
    def grad_wrapper(*xs):
      ys = self.merge_call(batch_all_gather, args=xs)

      def grad(*dy_s):
        grads = self.all_reduce(reduce_util.ReduceOp.SUM, dy_s)
        new_grads = []
        for i, grad in enumerate(grads):
          input_shape = array_ops.shape(xs[i])
          axis_dim = array_ops.reshape(input_shape[axis], [1])
          with ops.control_dependencies([array_ops.identity(grads)]):
            d = self.all_gather(axis_dim, axis=0)
            begin_dim = math_ops.reduce_sum(d[:self.replica_id_in_sync_group])
            end_dim = begin_dim + array_ops.shape(xs[i])[axis]
            new_grad = array_ops.gather(
                grad, axis=axis, indices=math_ops.range(begin_dim, end_dim))
            new_grads.append(new_grad)
        return new_grads

      return ys, grad

    return nest.pack_sequence_as(value, grad_wrapper(*nest.flatten(value)))

  def _update(self, var, fn, args=(), kwargs=None, group=True):
    """Run `fn` to update `var` with `args` and `kwargs` in replica context.

    `tf.distribute.ReplicaContext.update` takes a (distributed) variable `var`
    to be updated, an update function `fn`, and `args` and `kwargs` for `fn`.
    `fn` applies to each component variable of `var` with corresponding input
    values from `args` and `kwargs`.

    Example usage:

    >>> strategy = tf.distribute.MirroredStrategy(['GPU:0', 'GPU:1']) # 2 replicas
    >>> with strategy.scope():
    ...   distributed_variable = tf.Variable(5.0)
    >>> distributed_variable
    MirroredVariable:{
      0: <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=5.0>,
      1: <tf.Variable 'Variable/replica_1:0' shape=() dtype=float32, numpy=5.0>
    }
    >>> def replica_fn(v):
    ...   value = tf.identity(1.0)
    ...   replica_context = tf.distribute.get_replica_context()
    ...   update_fn = lambda var, value: var.assign(value)
    ...   replica_context._update(v, update_fn, args=(value,))
    >>> strategy.run(replica_fn, args=(distributed_variable,))
    >>> distributed_variable
    MirroredVariable:{
      0: <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>,
      1: <tf.Variable 'Variable/replica_1:0' shape=() dtype=float32, numpy=1.0>
    }

    This API must be called in a replica context.

    Note that if `var` is a MirroredVariable (i.e., the type of variable created
    under the scope of a synchronous strategy, and is synchronized on-write, see
    `tf.VariableSynchronization` for more information) and `args`/`kwargs`
    contains different values for different replicas, `var` will be dangerously
    out of synchronization. Thus we recommend using `variable.assign(value)` as
    long as you can, which under the hood aggregates the updates and guarantees
    the synchronization. The case where you actually want this API instead of
    `variable.assign(value)` is that before assigning `value` to the `variable`,
    you'd like to conduct some pre-`assign` computation colocated with the
    variable devices (i.e. where variables reside, for MirroredStrategy they are
    the same as the compute device, for ParameterServerStrategy they refer to
    parameter servers). E.g.,

    ```python
    strategy = tf.distribute.MirroredStrategy(['GPU:0', 'GPU:1']) # 2 replicas
    with strategy.scope():
      v = tf.Variable(5.0, aggregation=tf.VariableAggregation.SUM)
    def replica_fn(inputs):
      value = computation(inputs)
      replica_context = tf.distribute.get_replica_context()
      reduced_value = replica_context.all_reduce(value)

      def update_fn(var, value):
        # this computation will colocate with `var`'s device
        updated_value = post_reduce_pre_update_computation(value)
        var.assign(value)

      replica_context._update(v, update_fn, args=(reduced_value,))

    strategy.run(replica_fn, args=(inputs,))
    ```

    This code snippet is consistent across all strategies. If you directly
    compute and use `assign` in the replica context instead of wrapping it with
    `update`, for strategies with fewer variable devices than compute devices
    (e.g., parameter server strategy, usually), the
    `post_reduce_pre_update_computation` will happen
    N==number_of_compute_devices times which is less performant.


    Args:
      var: Variable, possibly distributed to multiple devices, to operate on.
      fn: Function to call. Should take the variable as the first argument.
      args: Tuple or list. Additional positional arguments to pass to `fn()`.
      kwargs: Dict with keyword arguments to pass to `fn()`.
      group: Boolean. Defaults to True. Most strategies enter a merge_call to
      conduct update in cross-replica context, and group=True guarantees updates
      on all replicas is executed.

    Returns:
      The return value of `fn` for the local replica.
    """
    if kwargs is None:
      kwargs = {}
    return self._strategy.extended._replica_ctx_update(var, fn, args=args, kwargs=kwargs, group=group)  # pylint: disable=protected-access


@tf_export(v1=["distribute.ReplicaContext"])
class ReplicaContextV1(ReplicaContextBase):
  __doc__ = ReplicaContextBase.__doc__


def _batch_reduce_destination(x):
  """Returns the destinations for batch all-reduce."""
  if isinstance(x, ops.Tensor):
    # If this is a one device strategy.
    return x.device
  else:
    return x
# ------------------------------------------------------------------------------


class _DefaultDistributionStrategyV1(StrategyV1):
  """Default `tf.distribute.Strategy` if none is explicitly selected."""

  def __init__(self):
    if not _creating_default_strategy_singleton:
      raise RuntimeError("Should only create a single instance of "
                         "_DefaultDistributionStrategy")
    super(_DefaultDistributionStrategyV1,
          self).__init__(_DefaultDistributionExtended(self))

  def __deepcopy__(self, memo):
    del memo
    raise RuntimeError("Should only create a single instance of "
                       "_DefaultDistributionStrategy")


class _DefaultDistributionStrategy(Strategy):
  """Default `tf.distribute.Strategy` if none is explicitly selected."""

  def __init__(self):
    if not _creating_default_strategy_singleton:
      raise RuntimeError("Should only create a single instance of "
                         "_DefaultDistributionStrategy")
    super(_DefaultDistributionStrategy, self).__init__(
        _DefaultDistributionExtended(self))

  def __deepcopy__(self, memo):
    del memo
    raise RuntimeError("Should only create a single instance of "
                       "_DefaultDistributionStrategy")


class _DefaultDistributionContext(object):
  """Context manager setting the default `tf.distribute.Strategy`."""

  __slots__ = ["_var_creator_scope", "_strategy", "_nested_count"]

  def __init__(self, strategy):

    def creator(next_creator, **kwargs):
      _require_strategy_scope_strategy(strategy)
      return next_creator(**kwargs)

    self._var_creator_scope = variable_scope.variable_creator_scope(creator)
    self._strategy = strategy
    self._nested_count = 0

  def __enter__(self):
    # Allow this scope to be entered if this strategy is already in scope.
    if has_strategy():
      raise RuntimeError("Must not nest tf.distribute.Strategy scopes.")
    if self._nested_count == 0:
      self._var_creator_scope.__enter__()
    self._nested_count += 1
    return self._strategy

  def __exit__(self, exception_type, exception_value, traceback):
    self._nested_count -= 1
    if self._nested_count == 0:
      try:
        self._var_creator_scope.__exit__(
            exception_type, exception_value, traceback)
      except RuntimeError as e:
        six.raise_from(
            RuntimeError("Variable creator scope nesting error: move call to "
                         "tf.distribute.set_strategy() out of `with` scope."),
            e)


class _DefaultDistributionExtended(StrategyExtendedV1):
  """Implementation of _DefaultDistributionStrategy."""

  def __init__(self, container_strategy):
    super(_DefaultDistributionExtended, self).__init__(container_strategy)
    self._retrace_functions_for_each_device = False

  def _scope(self, strategy):
    """Context manager setting a variable creator and `self` as current."""
    return _DefaultDistributionContext(strategy)

  def colocate_vars_with(self, colocate_with_variable):
    """Does not require `self.scope`."""
    _require_strategy_scope_extended(self)
    return ops.colocate_with(colocate_with_variable)

  def variable_created_in_scope(self, v):
    return v._distribute_strategy is None  # pylint: disable=protected-access

  def _experimental_distribute_dataset(self, dataset, options):
    return dataset

  def _distribute_datasets_from_function(self, dataset_fn, options):
    return dataset_fn(InputContext())

  def _experimental_distribute_values_from_function(self, value_fn):
    return value_fn(ValueContext())

  def _make_dataset_iterator(self, dataset):
    return _DefaultDistributionExtended.DefaultInputIterator(dataset)

  def _make_input_fn_iterator(self,
                              input_fn,
                              replication_mode=InputReplicationMode.PER_WORKER):
    dataset = input_fn(InputContext())
    return _DefaultDistributionExtended.DefaultInputIterator(dataset)

  def _experimental_make_numpy_dataset(self, numpy_input, session):
    numpy_flat = nest.flatten(numpy_input)
    vars_flat = tuple(
        variable_v1.VariableV1(array_ops.zeros(i.shape, i.dtype),
                               trainable=False, use_resource=True)
        for i in numpy_flat
    )
    for v, i in zip(vars_flat, numpy_flat):
      numpy_dataset.init_var_from_numpy(v, i, session)
    vars_nested = nest.pack_sequence_as(numpy_input, vars_flat)
    return dataset_ops.Dataset.from_tensor_slices(vars_nested)

  def _broadcast_to(self, tensor, destinations):
    if destinations is None:
      return tensor
    else:
      raise NotImplementedError("TODO")

  def _call_for_each_replica(self, fn, args, kwargs):
    with ReplicaContext(self._container_strategy(), replica_id_in_sync_group=0):
      return fn(*args, **kwargs)

  def _reduce_to(self, reduce_op, value, destinations, options):
    # TODO(joshl): Use destinations?
    del reduce_op, destinations, options
    return value

  def _gather_to_implementation(self, value, destinations, axis, options):
    del destinations, axis, options
    return value

  def _update(self, var, fn, args, kwargs, group):
    # The implementations of _update() and _update_non_slot() are identical
    # except _update() passes `var` as the first argument to `fn()`.
    return self._update_non_slot(var, fn, (var,) + tuple(args), kwargs, group)

  def _update_non_slot(self, colocate_with, fn, args, kwargs, should_group):
    # TODO(joshl): Figure out what we should be passing to UpdateContext()
    # once that value is used for something.
    with UpdateContext(colocate_with):
      result = fn(*args, **kwargs)
      if should_group:
        return result
      else:
        return nest.map_structure(self._local_results, result)

  def read_var(self, replica_local_var):
    return array_ops.identity(replica_local_var)

  def _local_results(self, distributed_value):
    return (distributed_value,)

  def value_container(self, value):
    return value

  @property
  def _num_replicas_in_sync(self):
    return 1

  @property
  def worker_devices(self):
    raise RuntimeError("worker_devices() method unsupported by default "
                       "tf.distribute.Strategy.")

  @property
  def parameter_devices(self):
    raise RuntimeError("parameter_devices() method unsupported by default "
                       "tf.distribute.Strategy.")

  def non_slot_devices(self, var_list):
    return min(var_list, key=lambda x: x.name)

  def _in_multi_worker_mode(self):
    """Whether this strategy indicates working in multi-worker settings."""
    # Default strategy doesn't indicate multi-worker training.
    return False

  @property
  def should_checkpoint(self):
    return True

  @property
  def should_save_summary(self):
    return True

  def _get_local_replica_id(self, replica_id_in_sync_group):
    return replica_id_in_sync_group

  def _get_replica_id_in_sync_group(self, replica_id):
    return replica_id

  # TODO(priyag): This should inherit from `InputIterator`, once dependency
  # issues have been resolved.
  class DefaultInputIterator(object):
    """Default implementation of `InputIterator` for default strategy."""

    def __init__(self, dataset):
      self._dataset = dataset
      if eager_context.executing_eagerly():
        self._iterator = dataset_ops.make_one_shot_iterator(dataset)
      else:
        self._iterator = dataset_ops.make_initializable_iterator(dataset)

    def get_next(self):
      return self._iterator.get_next()

    def get_next_as_optional(self):
      return self._iterator.get_next_as_optional()

    @deprecated(None, "Use the iterator's `initializer` property instead.")
    def initialize(self):
      """Initialize underlying iterators.

      Returns:
        A list of any initializer ops that should be run.
      """
      if eager_context.executing_eagerly():
        self._iterator = self._dataset.make_one_shot_iterator()
        return []
      else:
        return [self._iterator.initializer]

    @property
    def initializer(self):
      """Returns a list of ops that initialize the iterator."""
      return self.initialize()

  # TODO(priyag): Delete this once all strategies use global batch size.
  @property
  def _global_batch_size(self):
    """Global and per-replica batching are equivalent for this strategy."""
    return True


class _DefaultReplicaContext(ReplicaContext):
  """ReplicaContext for _DefaultDistributionStrategy."""

  @property
  def replica_id_in_sync_group(self):
    # Return 0 instead of a constant tensor to avoid creating a new node for
    # users who don't use distribution strategy.
    return 0


# ------------------------------------------------------------------------------
# We haven't yet implemented deserialization for DistributedVariables.
# So here we catch any attempts to deserialize variables
# when using distribution strategies.
# pylint: disable=protected-access
_original_from_proto = ref_variable._from_proto_fn


def _from_proto_fn(v, import_scope=None):
  if has_strategy():
    raise NotImplementedError(
        "Deserialization of variables is not yet supported when using a "
        "tf.distribute.Strategy.")
  else:
    return _original_from_proto(v, import_scope=import_scope)

ref_variable._from_proto_fn = _from_proto_fn
# pylint: enable=protected-access


def get_local_results_or_value_container(variable):
  strategy, context = get_strategy_and_replica_context()
  if context:
    return [strategy.extended.value_container(variable)]
  else:
    return strategy.experimental_local_results(variable)


tape.register_watched_variable_resolver(get_local_results_or_value_container)


# ------------------------------------------------------------------------------
# Metrics to track which distribution strategy is being called
distribution_strategy_gauge = monitoring.StringGauge(
    "/tensorflow/api/distribution_strategy",
    "Gauge to track the type of distribution strategy used.", "TFVersion")
distribution_strategy_replica_gauge = monitoring.IntGauge(
    "/tensorflow/api/distribution_strategy/replica",
    "Gauge to track the number of replica each distribution strategy used.",
    "CountType")
distribution_strategy_input_api_counter = monitoring.Counter(
    "/tensorflow/api/distribution_strategy/input_api",
    "Counter to track the usage of the input APIs", "strategy", "api")
distributed_variable_creation_time_counter = monitoring.Counter(
    "/tensorflow/api/distribution_strategy/distributed_variable_creation_time_usecs",
    "Time to create distributed variables (us).", "strategy", "if_graph_building")
