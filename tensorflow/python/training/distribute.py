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
"""Class DistributionStrategy, TowerContext, and supporting APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import device_util
from tensorflow.python.util import nest


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


_per_thread_mode = threading.local()


def _push_per_thread_mode(context):
  if not hasattr(_per_thread_mode, "stack"):
    _per_thread_mode.stack = []
  _per_thread_mode.stack.append(context)


def _pop_per_thread_mode():
  _per_thread_mode.stack.pop(-1)


class _DefaultTowerThreadMode(_ThreadMode):
  """Type of default value returned by `_get_per_thread_mode()`.

  Used when the thread-local stack is empty.
  """

  def __init__(self):
    # _default_distribution_strategy and _default_tower_context are
    # defined at the bottom of this file.
    _ThreadMode.__init__(
        self, _default_distribution_strategy, None, _default_tower_context)


def _get_per_thread_mode():
  try:
    return _per_thread_mode.stack[-1]
  except (AttributeError, IndexError):
    # _default_tower_mode is defined at the bottom of this file.
    return _default_tower_mode


# ------------------------------------------------------------------------------
# Context tracking whether in a distribution.update() or .update_non_slot()
# call.


_update_device = threading.local()


def get_update_device():
  """Get the current device if in a `DistributionStrategy.update()` call."""
  try:
    return _update_device.current
  except AttributeError:
    return None


class UpdateContext(object):
  """Context manager when you are in `update()` or `update_non_slot()`."""

  def __init__(self, device):
    self._device = device
    self._old_device = None

  def __enter__(self):
    self._old_device = get_update_device()
    _update_device.current = self._device

  def __exit__(self, exception_type, exception_value, traceback):
    del exception_type, exception_value, traceback
    _update_device.current = self._old_device


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
  return get_distribution_strategy() is not _default_distribution_strategy


# ------------------------------------------------------------------------------
# Public utility functions.


def get_loss_reduction():
  """Reduce `aggregation` corresponding to the last loss reduction."""
  loss_reduction = ops.get_default_graph()._last_loss_reduction  # pylint: disable=protected-access
  if loss_reduction == losses_impl.Reduction.SUM:
    return variable_scope.VariableAggregation.SUM
  return variable_scope.VariableAggregation.MEAN


# ------------------------------------------------------------------------------
# Internal API for validating the current thread mode


def _require_cross_tower_context(distribution_strategy):
  """Verify in cross-tower context for `distribution_strategy`."""
  context = _get_per_thread_mode()
  if context.cross_tower_context is distribution_strategy: return
  # We have an error to report, figure out the right message.
  if context.distribution_strategy is not distribution_strategy:
    if context.distribution_strategy is _default_distribution_strategy:
      raise RuntimeError(
          'Need to be inside "with distribution_strategy.scope()" for %s' %
          (distribution_strategy,))
    else:
      raise RuntimeError(
          "Mixing different DistributionStrategy objects: %s is not %s" %
          (context.distribution_strategy, distribution_strategy))
  assert context.cross_tower_context is None
  raise RuntimeError("Method requires being in cross-tower context, use "
                     "get_tower_context().merge_call()")


def require_tower_context(tower_ctx):
  """Verify in `tower_ctx` tower context."""
  context = _get_per_thread_mode()
  if context.tower_context is tower_ctx: return
  # We have an error to report, figure out the right message.
  if context.tower_context is None:
    raise RuntimeError("Need to be inside `call_for_each_tower()`")
  if context.distribution_strategy is tower_ctx.distribution_strategy:
    # Two different TowerContexts with the same DistributionStrategy.
    raise RuntimeError("Mismatching tower context.")
  raise RuntimeError(
      "Mismatching DistributionStrategy objects: %s is not %s." %
      (context.distribution_strategy, tower_ctx.distribution_strategy))


def _require_distribution_strategy_scope(distribution_strategy):
  """Verify in a `distribution_strategy.scope()` in this thread."""
  context = _get_per_thread_mode()
  if context.distribution_strategy is distribution_strategy: return
  # We have an error to report, figure out the right message.
  if context.distribution_strategy is _default_distribution_strategy:
    raise RuntimeError(
        'Need to be inside "with distribution_strategy.scope()" for %s' %
        (distribution_strategy,))
  else:
    raise RuntimeError(
        "Mixing different DistributionStrategy objects: %s is not %s" %
        (context.distribution_strategy, distribution_strategy))


# ------------------------------------------------------------------------------
# Internal context managers used to implement the DistributionStrategy
# base class


class _CurrentDistributionContext(object):
  """Context manager for setting the `DistributionStrategy` and var creator."""

  def __init__(self,
               distribution_strategy,
               var_creator_scope,
               var_scope=None,
               default_device=None):
    self._context = _CrossTowerThreadMode(distribution_strategy)
    self._var_creator_scope = var_creator_scope
    self._var_scope = var_scope
    if default_device:
      self._device_scope = ops.device(default_device)
    else:
      self._device_scope = None

  def __enter__(self):
    _push_per_thread_mode(self._context)
    if self._var_scope:
      self._var_scope.__enter__()
    self._var_creator_scope.__enter__()
    if self._device_scope:
      self._device_scope.__enter__()
    return self._context.distribution_strategy

  def __exit__(self, exception_type, exception_value, traceback):
    if self._device_scope:
      self._device_scope.__exit__(exception_type, exception_value, traceback)
    self._var_creator_scope.__exit__(exception_type, exception_value, traceback)
    if self._var_scope:
      self._var_scope.__exit__(exception_type, exception_value, traceback)
    _pop_per_thread_mode()


class _SameScopeAgainContext(object):
  """Trivial context manager when you are already in `scope()`."""

  def __init__(self, distribution_strategy):
    self._distribution_strategy = distribution_strategy

  def __enter__(self):
    return self._distribution_strategy

  def __exit__(self, exception_type, exception_value, traceback):
    del exception_type, exception_value, traceback


# ------------------------------------------------------------------------------
# Base classes for all distribution strategies.


class DistributionStrategy(object):
  """A list of devices with a state & compute distribution policy.

  The intent is that you can write an algorithm in a stylized way and
  it will be usable with a variety of different `DistributionStrategy`
  implementations. Each descendant will implement a different strategy
  for distributing the algorithm across multiple devices/machines.
  Furthermore, these changes can be hidden inside the specific layers
  and other library classes that need special treatment to run in a
  distributed setting, so that most users' model definition code can
  run unchanged. The `DistributionStrategy` API works the same way
  with eager and graph execution.

  First let's introduce a few high-level concepts:

  * _Data parallelism_ is where we run multiple copies of the model
    on different slices of the input data. This is in contrast to
    _model parallelism_ where we divide up a single copy of a model
    across multiple devices.
    Note: we only support data parallelism for now, but
    hope to add support for model parallelism in the future.
  * A _tower_ is one copy of the model, running on one slice of the
    input data.
  * _Synchronous_, or more commonly _sync_, training is where the
    updates from each tower are aggregated together before updating
    the model variables. This is in contrast to _asynchronous_, or
    _async_ training, where each tower updates the model variables
    independently.
  * Furthermore you might run your computation on multiple devices
    on one machine (or "host"), or on multiple machines/hosts.
    If you are running on multiple machines, you might have a
    single master host that drives computation across all of them,
    or you might have multiple clients driving the computation
    asynchronously.

  To distribute an algorithm, we might use some of these ingredients:

  * Parameter servers: These are hosts that hold a single copy of
    parameters/variables. All towers that want to operate on a variable
    retrieve it at the beginning of a step and send an update to be
    applied at the end of the step. Can support either sync or async
    training.
  * Mirrored variables: These are variables that are copied to multiple
    devices, where we keep the copies in sync by applying the same
    updates to every copy. Normally would only be used with sync training.
  * Reductions and Allreduce: A _reduction_ is some method of
    aggregating multiple values into one value, like "sum" or
    "mean". If doing sync training, we will perform a reduction on the
    gradients to a parameter from all towers before applying the
    update. Allreduce is an algorithm for performing a reduction on
    values from multiple devices and making the result available on
    all of those devices.
  * In the future we will have support for TensorFlow's partitioned
    variables, where a single variable is split across multiple
    devices.

  We have then a few approaches we want to support:
  * Code written (as if) with no knowledge of class `DistributionStrategy`.
    This code should work as before, even if some of the layers, etc.
    used by that code are written to be distribution-aware. This is done
    by having a default `DistributionStrategy` that gives ordinary behavior,
    and by default being in a single tower context.
  * Ordinary model code that you want to run using a specific
    `DistributionStrategy`. This can be as simple as:

    ```
    with my_distribution.scope():
      iterator = my_distribution.distribute_dataset(
          dataset).make_one_shot_iterator()
      tower_train_ops = my_distribution.call_for_each_tower(
          tower_fn, iterator.get_next())
      train_op = tf.group(my_distribution.unwrap(tower_train_ops))
    ```

    This takes an ordinary `dataset` and `tower_fn` and runs it
    distributed using a particular `DistributionStrategy` in
    `my_distribution`. Any variables created in `tower_fn` are created
    using `my_distribution`'s policy, and library functions called by
    `tower_fn` can use the `get_tower_context()` API to get enhanced
    behavior in this case.

    You can also create an initializable iterator instead of a one-shot
    iterator. In that case, you will need to ensure that you initialize the
    iterator before calling get_next.
    ```
    iterator = my_distribution.distribute_dataset(
        dataset).make_initializable_iterator())
    session.run(iterator.initializer)
    ```

  * If you want to write a distributed algorithm, you may use any of
    the `DistributionStrategy` APIs inside a
    `with my_distribution.scope():` block of code.

  Lower-level concepts:

  * Wrapped values: In order to represent values parallel across devices
    (either towers or the devices associated with a particular value), we
    wrap them in a "PerDevice" or "Mirrored" object that contains a map
    from device to values. "PerDevice" is used when the value may be
    different across devices, and "Mirrored" when the value are the same.
  * Unwrapping and merging: Consider calling a function `fn` on
    multiple devices, like `call_for_each_tower(fn, w)` with an
    argument `w` that is a wrapped value. This means `w` will have a
    map taking tower device `d0` to `w0`, tower device `d1` to `w1`,
    etc. `call_for_each_tower()` unwraps `w` before calling `fn`, so
    it calls `fn(w0)` on `d0`, `fn(w1)` on `d1`, etc.  It then merges
    the return values from `fn()`, which can possibly result in
    wrapped values. For example, let's say `fn()` returns a tuple with
    three components: `(x, a, v0)` from tower 0, `(x, b, v1)` on tower 1,
    etc. If the first component is the same object `x` from every
    tower, then the first component of the merged result will also be
    `x`. If the second component is different (`a`, `b`, ...)  from
    each tower, then the merged value will have a wrapped map from
    tower device to the different values. If the third component is
    the members of a mirrored variable (`v` maps `d0` to `v0`, `d1` to
    `v1`, etc.), then the merged result will be that mirrored variable
    (`v`).
  * Tower context vs. Cross-tower context: _tower context_ is when we
    are in some function that is being called once for each tower.
    Otherwise we are in cross-tower context, which is useful for
    calling `DistributionStrategy` methods which operate across the
    towers (like `reduce()`). By default you start in a tower context
    (the default "single tower context") and then some methods can
    switch you back and forth, as described below.
  * Worker devices vs. parameter devices: Most tower computations will
    happen on worker devices. Since we don't yet support model
    parallelism, there will be one worker device per tower. When using
    parameter servers (see above), the set of devices holding
    variables may be different, otherwise the parameter devices might
    match the worker devices.
  * Non-slot devices are some subset of the parameter devices where we
    put all the non-slot variables. We need to ensure that all
    non-slot variables are allocated on the same device, or mirrored
    across the same set of devices. If you have some variable you want
    to colocate all the non-slot variables with, you can use
    `colocate_vars_with()` to get the remaining non-slot variables on
    the same device.  Otherwise you can use `non_slot_devices()` to
    pick a consistent set of devices to pass to both
    `colocate_vars_with()` and `update_non_slot()`.

  When using a `DistributionStrategy`, we have a new type dimension
  called _locality_ that says what values are compatible with which
  APIs:

  * T: different value for each tower (e.g. a PerDevice-wrapped value).
  * M: value is "mirrored" across towers, i.e. there are copies with the
    same value on each tower (e.g. a Mirrored-wrapped value).
  * V(`v`): value is "mirrored" across all the devices which have a
    copy of variable `v` (also a Mirrored-wrapped value, but over
    parameter devices instead of worker devices).
  * N: value is "mirrored" across all the "non-slot" devices

  Rules for methods with respect to locality and single-tower vs.
  cross-tower context:

  * `with d.scope()`: default single-tower context -> cross-tower context for
    `d`
  * `with d.colocate_vars_with(v)`: in tower/cross-tower context, variables
    will be created with locality V(`v`). That is, if we write
    `with d.colocate_vars_with(v1): v2 = tf.get_variable(...)`, then
    `v2` will have locality V(`v1`), i.e. locality V(`v2`) will equal
    V(`v1`).
  * `with d.colocate_vars_with(d.non_slot_devices(...))`: in
    tower/cross-tower context, variables will be created with locality N
  * `v = tf.get_variable(...)`: in tower/cross-tower context, creates
    a variable (which by definition will have locality V(`v`), though
    will match another locality if inside a `colocate_vars_with`
    scope).
  * `d.distribute_dataset(dataset).make_one_shot_iterator()`: in cross-tower
    context, produces an iterator with locality T
  * `d.broadcast(t)`: in cross-tower context, produces a value with locality M
  * `d.broadcast(t, v)`: in cross-tower context, produces a value with
    locality V(`v`)
  * `d.call_for_each_tower(fn, ...)`: in cross-tower context, runs
    `fn()` in a tower context (and so may call `get_tower_context()` and
    use its API, including `merge_call()` to get back to cross-tower
    context), once for each tower. May use values with locality T or
    M, and any variable.
  * `d.reduce(m, t)`: in cross-tower context, accepts t with locality T
    and produces a value with locality M.
  * `d.reduce(m, t, v)`: in cross-tower context, accepts t with
    locality T and produces a value with locality V(`v`).
  * `d.batch_reduce(m, [(t, v)]): see `d.reduce()`
  * `d.update(v, fn, ...)`: in cross-tower context, runs `fn()` once
    for each device `v` is copied to, all inputs should have locality
    V(`v`), output will have locality V(`v`) as well.
  * `d.update_non_slot(d.non_slot_devices(), fn)`: in cross-tower
    context, like `d.update()` except with locality N.
  * `d.read_var(v)`: Gets the (read-only) value of the variable `v` (on
    the device determined by the current device scope), aggregating
    across towers for tower-local variables. Frequently, this will be
    done automatically when using `v` in an expression or fetching it in
    a cross-tower context, but this function can be used to force that
    conversion happens at a particular point in time (for example, to
    add the result of the conversion to a graph collection).

  The standard pattern for updating variables is to:

  1. Wrap your input dataset in `d.distribute_dataset()` and create an iterator.
  2. Define each tower `d.call_for_each_tower()` up to the point of
     getting a list of gradient, variable pairs.
  3. Call `d.reduce(VariableAggregation.SUM, t, v)` or `d.batch_reduce()` to sum
     the gradients (with locality T) into values with locality V(`v`).
  4. Call `d.update(v)` for each variable to update its value.

  Steps 3 and 4 are done automatically by class `Optimizer` if you call
  its `apply_gradients` method in a tower context. Otherwise you can
  manually call its `_distributed_apply` method in a cross-tower context.

  Another thing you might want to do in the middle of your tower function
  is an all-reduce of some intermediate value, using `d.reduce()` or
  `d.batch_reduce()` without supplying a variable as the destination.

  Layers should expect to be called in a tower context, and can use
  the `get_tower_context()` function to get a `TowerContext` object.  The
  `TowerContext` object has a `merge_call()` method for entering
  cross-tower context where you can use `reduce()` (or
  `batch_reduce()`) and then optionally `update()` to update state.

  You may use this API whether or not a `DistributionStrategy` is
  being used, since there is a default implementation of
  `TowerContext` and `DistributionStrategy`. Or you can use the
  `get_tower_context().is_single_tower` property to run different code
  in the distributed vs. single tower cases.
  """

  # TODO(josh11b): Raise an exception if variable partitioning requested before
  #   we add support.
  # TODO(josh11b): Also `parameter_device_index` property?
  # TODO(josh11b): `map()`
  # TODO(josh11b): ClusterSpec/ClusterResolver
  # TODO(josh11b): Partitioned computations, state; sharding
  # TODO(josh11b): Model parallelism: "towers" with multiple devices; shuffling
  # TODO(josh11b): List of towers with their worker and parameter devices
  #   (where the parameter devices may overlap in the ps case).

  def __init__(self):
    self._default_device = None

  def scope(self):
    """Returns a context manager selecting this DistributionStrategy as current.

    Inside a `with distribution_strategy.scope():` code block, this thread
    will use a variable creator set by `distribution_strategy`, and will
    enter its "cross-tower context".

    Returns:
      A context manager.
    """
    if has_distribution_strategy():
      _require_cross_tower_context(self)
      return _SameScopeAgainContext(self)

    def creator_with_resource_vars(*args, **kwargs):
      _require_distribution_strategy_scope(self)
      kwargs["use_resource"] = True
      return self._create_variable(*args, **kwargs)

    def disable_partitioned_variables(getter, *args, **kwargs):
      if kwargs.pop("partitioner", None) is not None:
        tf_logging.log_first_n(
            tf_logging.WARN, "Partitioned variables are disabled when using "
            "DistributionStrategy.", 1)
      return getter(*args, **kwargs)

    return _CurrentDistributionContext(
        self, variable_scope.variable_creator_scope(creator_with_resource_vars),
        variable_scope.variable_scope(
            variable_scope.get_variable_scope(),
            custom_getter=disable_partitioned_variables),
        self._default_device)

  def _create_variable(self, next_creator, *args, **kwargs):
    # Note: should support "colocate_with" argument.
    raise NotImplementedError("must be implemented in descendants")

  def tower_local_var_scope(self, aggregation):
    """Inside this scope, new variables will not be mirrored.

    There will still be one component variable per tower, but there is
    no requirement that they stay in sync. Instead, when saving them
    or calling `read_var()`, we use the value that results when
    calling `reduce()` on all the towers' variables.

    Note: tower-local implies not trainable. Instead, it is expected
    that each tower will directly update (using `assign_add()` or
    whatever) its local variable instance but only the aggregated
    value (accessible using `read_var()`) will be exported from the
    model. When it is acceptable to only aggregate on export, we
    greatly reduce communication overhead by using tower-local
    variables.

    Note: All component variables will be initialized to the same
    value, using the initialization expression from the first tower.
    The values will match even if the initialization expression uses
    random numbers.

    Args:
      aggregation: Indicates how a variable will be aggregated. Accepted values
        are @{tf.VariableAggregation.SUM}, @{tf.VariableAggregation.MEAN}.

    Returns:
      A context manager.
    """
    # TODO(psv): Remove this after adding support for synchronization and
    # aggregation parameters in get_variable() and mirrored strategy.
    def create_tower_local_variable(next_creator, *args, **kwargs):
      _require_distribution_strategy_scope(self)
      kwargs["use_resource"] = True

      # Set synchronization to be ON_READ for tower local variables.
      kwargs["synchronization"] = variable_scope.VariableSynchronization.ON_READ
      kwargs["aggregation"] = aggregation
      return next_creator(*args, **kwargs)

    _require_distribution_strategy_scope(self)
    return variable_scope.variable_creator_scope(create_tower_local_variable)

  def read_var(self, v):
    """Reads the value of a variable.

    Returns the aggregate value of a tower-local variable, or the
    (read-only) value of any other variable.

    Args:
      v: A variable allocated within the scope of this `DistributionStrategy`.

    Returns:
      A tensor representing the value of `v`, aggregated across towers if
      necessary.
    """
    raise NotImplementedError("must be implemented in descendants")

  def colocate_vars_with(self, colocate_with_variable):
    """Scope that controls which devices variables will be created on.

    No operations should be added to the graph inside this scope, it
    should only be used when creating variables (some implementations
    work by changing variable creation, others work by using a
    tf.colocate_with() scope).

    This may only be used inside `self.scope()`.

    Example usage:

    ```
    with distribution_strategy.scope():
      var1 = tf.get_variable(...)
      with distribution_strategy.colocate_vars_with(v1):
        # var2 and var3 will be created on the same device(s) as var1
        var2 = tf.get_variable(...)
        var3 = tf.get_variable(...)

      def fn(v1, v2, v3):
        # operates on v1 from var1, v2 from var2, and v3 from var3

      # `fn` runs on every device `v1` is on, `v2` and `v3` will be there too.
      distribution_strategy.update(v1, fn, v2, v3)
    ```

    Args:
      colocate_with_variable: A created in `self.scope()`. Variables created
        while in the returned context manager will be on the same set of
        devices as `colocate_with_variable`.

    Returns:
      A context manager.
    """
    def create_colocated_variable(next_creator, *args, **kwargs):
      _require_distribution_strategy_scope(self)
      kwargs["use_resource"] = True
      kwargs["colocate_with"] = colocate_with_variable
      return next_creator(*args, **kwargs)

    _require_distribution_strategy_scope(self)
    return variable_scope.variable_creator_scope(create_colocated_variable)

  def _call_dataset_fn(self, dataset_fn):
    result = dataset_fn()
    if not isinstance(result, dataset_ops.Dataset):
      raise ValueError(
          "dataset_fn() must return a tf.data.Dataset when using a "
          "DistributionStrategy.")
    return result

  # TODO(josh11b): `PerDeviceDataset` currently only implements a few methods of
  # Dataset API such as make_one_shot_iterator and make_initializable_iterator.
  # Extend to implement more functionality of datasets.
  def distribute_dataset(self, dataset_fn):
    """Return a `dataset` split across all towers.

    Suitable for providing input to for `call_for_each_tower()` by creating an
    iterator:

    ```
    def dataset_fn():
      return tf.data.Dataset.from_tensors([[1.]]).repeat()
    with distribution_strategy.scope():
      distributed_dataset = distribution_strategy.distribute_dataset(dataset_fn)
      iterator = distributed_dataset.make_one_shot_iterator()
      tower_results = distribution_strategy.call_for_each_tower(
          tower_fn, iterator.get_next())
    ```

    Args:
      dataset_fn: A function that returns a `tf.data.Dataset`.

    Returns:
      A `PerDeviceDataset` that will produce data for each tower.
    """
    raise NotImplementedError("must be implemented in descendants")

  def broadcast(self, tensor, destinations=None):
    """Mirror a tensor on one device to all worker devices.

    Args:
      tensor: A Tensor value to broadcast.
      destinations: An optional mirrored variable, device string, or
        list of device strings, specifying the destination devices
        to copy `tensor` to. Defaults to `self.worker_devices`.

    Returns:
      A value mirrored to `destinations` devices.
    """
    # TODO(josh11b): More docstring
    _require_cross_tower_context(self)
    return self._broadcast(tensor, destinations)

  def _broadcast(self, tensor, destinations):
    raise NotImplementedError("must be implemented in descendants")

  def call_for_each_tower(self, fn, *args, **kwargs):
    """Run `fn` once per tower.

    `fn` may call `tf.get_tower_context()` to access methods such as
    `tower_id()` and `merge_call()`.

    `merge_call()` is used to communicate between the towers and
    re-enter the cross-tower context. All towers pause their execution
    having encountered a `merge_call()` call. After that the
    `merge_fn`-function is executed. Its results are then unwrapped and
    given back to each tower call. After that execution resumes until
    `fn` is complete or encounters another `merge_call()`.  Example:

    ```python
    # Called once in "cross-tower" context.
    def merge_fn(distribution, three_plus_tower_id):
      # sum the values across towers
      return sum(distribution.unwrap(three_plus_tower_id))

    # Called once per tower in `distribution`, in a "tower" context.
    def fn(three):
      tower_ctx = tf.get_tower_context()
      v = three + tower_ctx.tower_id
      # Computes the sum of the `v` values across all towers.
      s = tower_ctx.merge_call(merge_fn, v)
      return s + v

    with distribution.scope():
      # in "cross-tower" context
      ...
      merged_results = distribution.call_for_each_tower(fn, 3)
      # merged_results has the values from every tower execution of `fn`.
      print(distribution.unwrap(merged_results))  # Prints a list
    ```

    Args:
      fn: function to run (will be run once per tower).
      *args: positional arguments for `fn`
      **kwargs: keyword arguments for `fn`.
          `"run_concurrently"`: Boolean indicating whether executions of `fn`
             can be run concurrently (under eager execution only), defaults to
             `True`.

    Returns:
      Merged return value of `fn` across all towers.
    """
    _require_cross_tower_context(self)
    return self._call_for_each_tower(fn, *args, **kwargs)

  def _call_for_each_tower(self, fn, *args, **kwargs):
    raise NotImplementedError("must be implemented in descendants")

  def reduce(self, aggregation, value, destinations=None):
    """Combine (via e.g. sum or mean) values across towers.

    Args:
      aggregation: Indicates how a variable will be aggregated. Accepted values
        are @{tf.VariableAggregation.SUM}, @{tf.VariableAggregation.MEAN}.
      value: A per-device value with one value per tower.
      destinations: An optional mirrored variable, a device string,
        list of device strings. The return value will be copied to all
        destination devices (or all the devices where the mirrored
        variable resides). If `None` or unspecified, the destinations
        will match the devices `value` resides on.

    Returns:
      A value mirrored to `destinations`.
    """
    # TODO(josh11b): More docstring
    # TODO(josh11b): Return an unwrapped value if colocate_with is a
    # single device.
    _require_cross_tower_context(self)
    assert aggregation in [
        variable_scope.VariableAggregation.SUM,
        variable_scope.VariableAggregation.MEAN
    ]
    return self._reduce(aggregation, value, destinations)

  def _reduce(self, aggregation, value, destinations):
    raise NotImplementedError("must be implemented in descendants")

  def batch_reduce(self, aggregation, value_destination_pairs):
    """Combine multiple `reduce` calls into one for faster execution.

    Args:
      aggregation: Indicates how a variable will be aggregated. Accepted values
        are @{tf.VariableAggregation.SUM}, @{tf.VariableAggregation.MEAN}.
      value_destination_pairs: A sequence of (value, destinations)
        pairs. See `reduce()` for a description.

    Returns:
      A list of mirrored values, one per pair in `value_destination_pairs`.
    """
    # TODO(josh11b): More docstring
    _require_cross_tower_context(self)
    assert aggregation in [
        variable_scope.VariableAggregation.SUM,
        variable_scope.VariableAggregation.MEAN
    ]
    return self._batch_reduce(aggregation, value_destination_pairs)

  def _batch_reduce(self, aggregation, value_destination_pairs):
    return [
        self.reduce(aggregation, t, destinations=v)
        for t, v in value_destination_pairs
    ]

  def update(self, var, fn, *args, **kwargs):
    """Run `fn` to update `var` using inputs mirrored to the same devices.

    If `var` is mirrored across multiple devices, then this implements
    logic like:

    ```
    results = {}
    for device, v in var:
      with tf.device(device):
        # *args and **kwargs will be unwrapped if they are mirrored.
        results[device] = fn(v, *args, **kwargs)
    return merged(results)
    ```

    Otherwise this returns `fn(var, *args, **kwargs)` colocated with `var`.'

    Neither *args nor **kwargs may contain per-device values.
    If they contain mirrored values, they will be unwrapped before
    calling `fn`.

    Args:
      var: Variable, possibly mirrored to multiple devices, to operate on.
      fn: Function to call. Should take the variable as the first argument.
      *args: Additional positional arguments to pass to `fn()`.
      **kwargs: Keyword arguments to pass to `fn()`.

    Returns:
      Merged return value of `fn` across all towers.
    """
    _require_cross_tower_context(self)
    return self._update(var, fn, *args, **kwargs)

  def _update(self, var, fn, *args, **kwargs):
    raise NotImplementedError("must be implemented in descendants")

  def update_non_slot(self, colocate_with, fn, *args, **kwargs):
    """Runs `fn(*args, **kwargs)` on `colocate_with` devices.

    Args:
      colocate_with: The return value of `non_slot_devices()`.
      fn: Function to execute.
      *args: Positional arguments to pass to `fn()`.
      **kwargs: Keyword arguments to pass to `fn()`.

    Returns:
      Return value of `fn`, possibly merged across devices.
    """
    _require_cross_tower_context(self)
    return self._update_non_slot(colocate_with, fn, *args, **kwargs)

  def _update_non_slot(self, colocate_with, fn, *args, **kwargs):
    raise NotImplementedError("must be implemented in descendants")

  def unwrap(self, value):
    """Returns the list of all per-device values contained in `value`.

    Args:
      value: A value returned by `call_for_each_tower()` or a variable
        created in `scope()`.

    Returns:
      A list of values contained in `value`. If `value` represents a single
      value, this returns `[value].`
    """
    _require_cross_tower_context(self)
    return self._unwrap(value)

  def _unwrap(self, distributed_value):
    raise NotImplementedError("must be implemented in descendants")

  def group(self, value, name=None):
    """Shortcut for `tf.group(distribution.unwrap(value))`."""
    value = nest.flatten(self.unwrap(value))

    if len(value) != 1 or name is not None:
      return control_flow_ops.group(value, name=name)
    # Special handling for the common case of one op.
    v, = value
    if hasattr(v, "op"):
      v = v.op
    return v

  @property
  def is_single_tower(self):
    """Returns whether there is a single tower or multiple.

    Returns:
      A boolean. If `True`, `call_for_each_tower(fn)` will only call `fn` once.
      If `False`, `call_for_each_tower(fn)` may call `fn` multiple times.
    """
    raise NotImplementedError("must be implemented in descendants")

  @property
  def num_towers(self):
    """Returns number of towers, for purposes of averaging across towers."""
    raise NotImplementedError("must be implemented in descendants")

  @property
  def worker_devices(self):
    """Returns the list of devices used to run `call_for_each_tower()` calls."""
    # TODO(josh11b): More docstring
    raise NotImplementedError("must be implemented in descendants")

  @property
  def parameter_devices(self):
    """Returns the list of devices used for variable and `update` placement."""
    # TODO(josh11b): More docstring
    raise NotImplementedError("must be implemented in descendants")

  def non_slot_devices(self, var_list):
    """Device(s) for non-slot variables.

    Create variables on these devices in a
    `with colocate_vars_with(non_slot_devices(...)):` block.
    Update those using `update_non_slot()`.

    Args:
      var_list: The list of variables being optimized, needed with the
        default `DistributionStrategy`.
    """
    raise NotImplementedError("must be implemented in descendants")

  @property
  def worker_device_index(self):
    """An object mapping worker device to an id.

    This might be passed as an argument to `call_for_each_tower()`, as in:

    ```
    with distribution_strategy.scope():

      def fn(device_id):
        # device_id is an integer. `fn` is being executed on device:
        #    distribution_strategy.worker_devices[device_id].

      distribution_strategy.call_for_each_tower(
          fn, distribution_strategy.worker_device_index)
    ```

    Returns:
      An index object, or the integer 0 if there is only a single tower.
    """
    _require_cross_tower_context(self)
    return self._worker_device_index()

  def _worker_device_index(self):
    raise NotImplementedError("must be implemented in descendants")

  def configure(self, session_config=None):
    """Find the best configuration given a tensorflow session config."""
    del session_config


# A note about the difference between the context managers
# `TowerContext` (defined here) and `_CurrentDistributionContext`
# (defined above) used by `DistributionStrategy.scope()`:
#
# * a TowerContext is only present during a `call_for_each_tower()`
#   call (except during a `merge_run` call) and in such a scope it
#   will be returned by calls to `get_tower_context()`.  Implementers of new
#   DistributionStrategy descendants will frequently also need to
#   define a descendant of TowerContext, and are responsible for
#   entering and exiting this context.
#
# * DistributionStrategy.scope() sets up a variable_creator scope that
#   changes variable creation calls (e.g. to make mirrored
#   variables). This is intended as an outer scope that users enter once
#   around their model creation and graph definition. There is no
#   anticipated need to define descendants of _CurrentDistributionContext.
#   It sets the current DistributionStrategy for purposes of
#   `get_distribution_strategy()` and `has_distribution_strategy()`
#   and switches the thread mode to a "cross-tower context".
class TowerContext(object):
  """DistributionStrategy API inside a `call_for_each_tower()` call."""

  def __init__(self, distribution_strategy, tower_id):
    self._distribution_strategy = distribution_strategy
    self._thread_context = _InTowerThreadMode(self)
    self._tower_id = tower_id

  def __enter__(self):
    _push_per_thread_mode(self._thread_context)

  def __exit__(self, exception_type, exception_value, traceback):
    _pop_per_thread_mode()

  def merge_call(self, merge_fn, *args, **kwargs):
    """Merge args across towers and run `merge_fn` in a cross-tower context.

    This allows communication and coordination when there are multiple calls
    to a model function triggered by a call to
    `distribution.call_for_each_tower(model_fn, ...)`.

    See `MirroredDistribution.call_for_each_tower()` for an explanation.

    Otherwise, this is equivalent to:

    ```
    distribution = get_distribution_strategy()
    with cross-tower-context(distribution):
      return merge_fn(distribution, *args, **kwargs)
    ```

    Args:
      merge_fn: function that joins arguments from threads that are given as
        PerDevice. It accepts `DistributionStrategy` object as the first
        argument.
      *args: positional per-thread arguments for `merge_fn`
      **kwargs: keyword per-thread arguments for `merge_fn`.

    Returns:
      The return value of `merge_fn`, except for `PerDevice` values which are
      unpacked.
    """
    require_tower_context(self)
    return self._merge_call(merge_fn, *args, **kwargs)

  def _merge_call(self, merge_fn, *args, **kwargs):
    """Default implementation for single tower."""
    _push_per_thread_mode(  # thread-local, so not needed with multiple threads
        _CrossTowerThreadMode(self._distribution_strategy))
    try:
      return merge_fn(self._distribution_strategy, *args, **kwargs)
    finally:
      _pop_per_thread_mode()

  def tower_local_var_scope(self, aggregation):
    """Alias for distribution_strategy.tower_local_var_scope()."""
    return self._distribution_strategy.tower_local_var_scope(aggregation)

  @property
  def is_single_tower(self):
    """Returns whether there is a single tower or multiple."""
    require_tower_context(self)
    return self._distribution_strategy.is_single_tower

  @property
  def num_towers(self):
    """Returns number of towers, for purposes of averaging across towers."""
    return self._distribution_strategy.num_towers

  @property
  def tower_id(self):
    """Which tower is being defined, a number from 0 to `num_towers - 1`."""
    require_tower_context(self)
    return self._tower_id

  @property
  def distribution_strategy(self):
    """The current `DistributionStrategy` object."""
    return self._distribution_strategy

  @property
  def device(self):
    """The device this tower is to be executed on, as a string."""
    require_tower_context(self)
    return device_util.current()

  # TODO(josh11b): Implement `start_all_reduce(method, t)` that returns
  # a function returning the result of reducing `t` across all
  # towers. Most likely can be implemented in terms of `merge_call()`
  # and `batch_reduce()`.

# ------------------------------------------------------------------------------


class _DefaultDistributionStrategy(DistributionStrategy):
  """Default `DistributionStrategy` if none is explicitly selected."""

  def scope(self):
    """Context manager setting a variable creator and `self` as current."""
    if has_distribution_strategy():
      raise RuntimeError("Must not nest DistributionStrategy scopes.")

    def creator(next_creator, *args, **kwargs):
      _require_distribution_strategy_scope(self)
      return next_creator(*args, **kwargs)

    return _CurrentDistributionContext(
        self, variable_scope.variable_creator_scope(creator))

  def tower_local_var_scope(self, aggregation):
    """Does not set to resource variables."""
    def create_tower_local_variable(next_creator, *args, **kwargs):
      _require_distribution_strategy_scope(self)
      kwargs["trainable"] = False
      return next_creator(*args, **kwargs)

    _require_distribution_strategy_scope(self)
    return variable_scope.variable_creator_scope(create_tower_local_variable)

  def colocate_vars_with(self, colocate_with_variable):
    """Does not require `self.scope`."""
    _require_distribution_strategy_scope(self)
    return ops.colocate_with(colocate_with_variable)

  def distribute_dataset(self, dataset_fn):
    return self._call_dataset_fn(dataset_fn)

  def _broadcast(self, tensor, destinations):
    if destinations is None:
      return tensor
    else:
      raise NotImplementedError("TODO")

  def _call_for_each_tower(self, fn, *args, **kwargs):
    # We don't run `fn` in multiple threads in _DefaultDistributionStrategy.
    kwargs.pop("run_concurrently", None)
    with TowerContext(self, tower_id=0):
      return fn(*args, **kwargs)

  def _reduce(self, aggregation, value, destinations):
    # TODO(josh11b): Use destinations?
    del aggregation, destinations
    return value

  def _update(self, var, fn, *args, **kwargs):
    # TODO(josh11b): Figure out what we should be passing to UpdateContext()
    # once that value is used for something.
    with ops.colocate_with(var), UpdateContext(var):
      return fn(var, *args, **kwargs)

  def _update_non_slot(self, colocate_with, fn, *args, **kwargs):
    # TODO(josh11b): Figure out what we should be passing to UpdateContext()
    # once that value is used for something.
    with ops.colocate_with(colocate_with), UpdateContext(colocate_with):
      return fn(*args, **kwargs)

  def read_var(self, tower_local_var):
    return array_ops.identity(tower_local_var)

  def _unwrap(self, distributed_value):
    return [distributed_value]

  @property
  def is_single_tower(self):
    return True

  @property
  def num_towers(self):
    return 1

  @property
  def worker_devices(self):
    raise RuntimeError(
        "worker_devices() method unsupported by _DefaultDistributionStrategy.")

  @property
  def parameter_devices(self):
    raise RuntimeError("parameter_devices() method unsupported by "
                       "_DefaultDistributionStrategy.")

  def non_slot_devices(self, var_list):
    return min(var_list, key=lambda x: x.name)

  def _worker_device_index(self):
    raise RuntimeError("worker_device_index() method unsupported by "
                       "_DefaultDistributionStrategy.")

# ------------------------------------------------------------------------------
# Common operations


def increment_var(v, amount=1):
  """`v += amount`, distributed-aware version."""
  def update(vu):
    if isinstance(vu, resource_variable_ops.ResourceVariable):
      return vu.assign_add(amount, read_value=False)
    else:
      return state_ops.assign_add(vu, amount)

  def merge_fn(dist, vm):
    return dist.group(dist.update(vm, update))

  tower_context = get_tower_context()
  return tower_context.merge_call(merge_fn, v)


# ------------------------------------------------------------------------------
# Singletons

_default_distribution_strategy = _DefaultDistributionStrategy()
_default_tower_context = TowerContext(
    _default_distribution_strategy, tower_id=0)
_default_tower_mode = _DefaultTowerThreadMode()


# ------------------------------------------------------------------------------
# We haven't yet implemented deserialization for DistributedVariables.
# So here we catch any attempts to deserialize variables
# when using distribution strategies.
# pylint: disable=protected-access
_original_from_proto = resource_variable_ops._from_proto_fn


def _from_proto_fn(v, import_scope=None):
  if has_distribution_strategy():
    raise NotImplementedError(
        "Deserialization of variables is not yet supported when using"
        "distributed strategies.")
  else:
    return _original_from_proto(v, import_scope=import_scope)

resource_variable_ops._from_proto_fn = _from_proto_fn
# pylint: enable=protected-access
