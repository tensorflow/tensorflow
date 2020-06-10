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

See the guide for overview and examples:
[TensorFlow v2.x](https://www.tensorflow.org/guide/distributed_training),
[TensorFlow v1.x](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/distribute_strategy.ipynb).

The intent of this library is that you can write an algorithm in a stylized way
and it will be usable with a variety of different `tf.distribute.Strategy`
implementations. Each descendant will implement a different strategy for
distributing the algorithm across multiple devices/machines.  Furthermore, these
changes can be hidden inside the specific layers and other library classes that
need special treatment to run in a distributed setting, so that most users'
model definition code can run unchanged. The `tf.distribute.Strategy` API works
the same way with eager and graph execution.

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
  different.  For example, `tf.distribute.experimental.CentralStorageStrategy`
  puts the variables on a single device (which may be a worker device or may be
  the CPU), and `tf.distribute.experimental.ParameterServerStrategy` puts the
  variables on separate machines called parameter servers (see below).
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
  update to be applied at the end of the step. These can in priniciple support
  either sync or async training, but right now we only have support for async
  training with parameter servers. Compare to
  `tf.distribute.experimental.CentralStorageStrategy`, which puts all variables
  on a single device on the same machine (and does sync training), and
  `tf.distribute.MirroredStrategy`, which mirrors variables to multiple devices
  (see below).
* _Mirrored variables_: These are variables that are copied to multiple
  devices, where we keep the copies in sync by applying the same
  updates to every copy. Normally would only be used with sync training.
* Reductions and all-reduce: A _reduction_ is some method of aggregating
  multiple values into one value, like "sum" or "mean". If a strategy is doing
  sync training, we will perform a reduction on the gradients to a parameter
  from all replicas before applying the update. _All-reduce_ is an algorithm for
  performing a reduction on values from multiple devices and making the result
  available on all of those devices.

Note that we provide a default version of `tf.distribute.Strategy` that is
used when no other strategy is in scope, that provides the same API with
reasonable default behavior.
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import enum  # pylint: disable=g-bad-import-order
import threading
import weakref

import six

from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context as eager_context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import loss_reduction
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.tracking import base as trackable
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
# Public utility functions.


@tf_export(v1=["distribute.get_loss_reduction"])
def get_loss_reduction():
  """`tf.distribute.ReduceOp` corresponding to the last loss reduction.

  This is used to decide whether loss should be scaled in optimizer (used only
  for estimator + v1 optimizer use case).

  Returns:
    `tf.distribute.ReduceOp` corresponding to the last loss reduction for
    estimator and v1 optimizer use case. `tf.distribute.ReduceOp.SUM` otherwise.
  """
  if not distribution_strategy_context.get_strategy()._scale_loss_for_estimator:  # pylint: disable=protected-access
    # If we are not in Estimator context then return 'SUM'. We do not need to
    # scale loss in the optimizer.
    return reduce_util.ReduceOp.SUM
  last_reduction = ops.get_default_graph()._last_loss_reduction  # pylint: disable=protected-access
  if (last_reduction == losses_impl.Reduction.SUM or
      last_reduction == loss_reduction.ReductionV2.SUM):
    return reduce_util.ReduceOp.SUM
  return reduce_util.ReduceOp.MEAN


# ------------------------------------------------------------------------------
# Internal API for validating the current thread mode


def _require_cross_replica_or_default_context_extended(extended):
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
  raise RuntimeError("Method requires being in cross-replica context, use "
                     "get_replica_context().merge_call()")


def _wrong_strategy_scope(strategy, context):
  # Figure out the right error message.
  if not distribution_strategy_context.has_strategy():
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
               default_device=None):
    self._context = distribution_strategy_context._CrossReplicaThreadMode(  # pylint: disable=protected-access
        strategy)
    self._var_creator_scope = var_creator_scope
    self._var_scope = var_scope
    if default_device:
      self._device_scope = ops.device(default_device)
    else:
      self._device_scope = None
    self._same_scope_again_count = 0

  def __enter__(self):
    # Allow this scope to be entered if this strategy is already in scope.
    if distribution_strategy_context.has_strategy():
      _require_cross_replica_or_default_context_extended(
          self._context.strategy.extended)
      self._same_scope_again_count += 1
    else:
      _push_per_thread_mode(self._context)
      if self._var_scope:
        self._var_scope.__enter__()
      self._var_creator_scope.__enter__()
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
  """
  PER_WORKER = "PER_WORKER"


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

  1. Directly constructed.

  >>> def value_fn(context):
  ...   return context.replica_id_in_sync_group/context.num_replicas_in_sync
  >>> context = tf.distribute.experimental.ValueContext(
  ...   replica_id_in_sync_group=2, num_replicas_in_sync=4)
  >>> per_replica_value = value_fn(context)
  >>> per_replica_value
  0.5

  2. Passed in by `experimental_distribute_values_from_function`.

  >>> strategy = tf.distribute.MirroredStrategy()
  >>> def value_fn(value_context):
  ...   return value_context.num_replicas_in_sync
  >>> distributed_values = (
  ...      strategy.experimental_distribute_values_from_function(
  ...        value_fn))
  >>> local_result = strategy.experimental_local_results(distributed_values)
  >>> local_result
  (1,)

  """

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
  """

  def __new__(cls,
              experimental_enable_dynamic_batch_size=True,
              experimental_bucketizing_dynamic_shape=False):
    return super(RunOptions,
                 cls).__new__(cls, experimental_enable_dynamic_batch_size,
                              experimental_bucketizing_dynamic_shape)

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
  for a glossory of concepts mentioned on this page such as "per-replica",
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

      * Start by either creating a `tf.data.Dataset` normally or using
        `tf.distribute.experimental_make_numpy_dataset` to make a dataset out of
        a `numpy` array.
      * Use `tf.distribute.Strategy.experimental_distribute_dataset` to convert
        a `tf.data.Dataset` to something that produces "per-replica" values.
        If you want to manually specify how the dataset should be partitioned
        across replicas, use
        `tf.distribute.Strategy.experimental_distribute_datasets_from_function`
        instead.
      * Use `tf.distribute.Strategy.run` to run a function
        once per replica, taking values that may be "per-replica" (e.g.
        from a distributed dataset) and returning "per-replica" values.
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
  this as a return value from one iteration over the distributed dataset. Or
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

  # TODO(josh11b): Partitioned computations, state; sharding
  # TODO(josh11b): Model parallelism: "replicas" with multiple devices; shuffling

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

  def scope(self):
    """Returns a context manager selecting this Strategy as current.

    Inside a `with strategy.scope():` code block, this thread
    will use a variable creator set by `strategy`, and will
    enter its "cross-replica context".

    Returns:
      A context manager.
    """
    return self._extended._scope(self)  # pylint: disable=protected-access

  @doc_controls.do_not_doc_inheritable  # DEPRECATED, moving to `extended`
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

  @deprecation.deprecated(
      "2020-09-30", "Please use tf.data.Dataset.from_tensor_slices instead")
  def experimental_make_numpy_dataset(self, numpy_input):
    """Makes a `tf.data.Dataset` from a numpy array.

    This avoids adding `numpy_input` as a large constant in the graph,
    and copies the data to the machine or machines that will be processing
    the input.

    Note that you will likely need to use `experimental_distribute_dataset`
    with the returned dataset to further distribute it with the strategy.

    Example:

    >>> strategy = tf.distribute.MirroredStrategy()
    >>> numpy_input = np.ones([10], dtype=np.float32)
    >>> dataset = strategy.experimental_make_numpy_dataset(numpy_input)
    >>> dataset
    <TensorSliceDataset shapes: (), types: tf.float32>
    >>> dataset = dataset.batch(2)
    >>> dist_dataset = strategy.experimental_distribute_dataset(dataset)

    Args:
      numpy_input: a nest of NumPy input arrays that will be converted into a
        dataset. Note that the NumPy arrays are stacked, as that is normal
        `tf.data.Dataset` behavior.

    Returns:
      A `tf.data.Dataset` representing `numpy_input`.
    """
    return self.extended.experimental_make_numpy_dataset(
        numpy_input, session=None)

  @doc_controls.do_not_generate_docs  # DEPRECATED: TF 1.x only
  def experimental_run(self, fn, input_iterator=None):
    """DEPRECATED TF 1.x ONLY."""
    with self.scope():
      args = (input_iterator.get_next(),) if input_iterator is not None else ()
    return self.run(fn, args=args)

  def experimental_distribute_dataset(self, dataset):
    """Distributes a tf.data.Dataset instance provided via `dataset`.

    The returned distributed dataset can be iterated over similar to how
    regular datasets can.
    NOTE: Currently, the user cannot add any more transformations to a
    distributed dataset.

    The following is an example:

    ```python
    strategy = tf.distribute.MirroredStrategy()

    # Create a dataset
    dataset = dataset_ops.Dataset.TFRecordDataset([
      "/a/1.tfr", "/a/2.tfr", "/a/3.tfr", "/a/4.tfr"])

    # Distribute that dataset
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    # Iterate over the distributed dataset
    for x in dist_dataset:
      # process dataset elements
      strategy.run(replica_fn, args=(x,))
    ```

    In the code snippet above, the dataset `dist_dataset` is batched by
    GLOBAL_BATCH_SIZE, and we iterate through it using `for x in dist_dataset`,
    where x is one batch of data of GLOBAL_BATCH_SIZE containing N batches of
    data of per-replica batch size, corresponding to N replicas.
    `tf.distribute.Strategy.run` will take care of feeding
    the right per-replica batch to the right `replica_fn` execution on each
    replica.

    In a multi-worker setting, we will first attempt to distribute the dataset
    by attempting to detect whether the dataset is being created out of
    ReaderDatasets (e.g. TFRecordDataset, TextLineDataset, etc.) and if so,
    attempting to shard the input files. Note that there has to be at least one
    input file per worker. If you have less than one input file per worker, we
    suggest that you should disable distributing your dataset using the method
    below.

    If that attempt is unsuccessful (e.g. the dataset is created from a
    Dataset.range), we will shard the dataset evenly at the end by appending a
    `.shard` operation to the end of the processing pipeline. This will cause
    the entire preprocessing pipeline for all the data to be run on every
    worker, and each worker will do redundant work. We will print a warning
    if this method of sharding is selected.

    You can disable dataset sharding across workers using the
    `auto_shard_policy` option in `tf.data.experimental.DistributeOptions`.

    Within each worker, we will also split the data among all the worker
    devices (if more than one a present), and this will happen even if
    multi-worker sharding is disabled using the method above.

    If the above batch splitting and dataset sharding logic is undesirable,
    please use `experimental_distribute_datasets_from_function` instead, which
    does not do any automatic splitting or sharding.

    You can also use the `element_spec` property of the distributed dataset
    returned by this API to query the `tf.TypeSpec` of the elements returned
    by the iterator. This can be used to set the `input_signature` property
    of a `tf.function`.

    ```python
    strategy = tf.distribute.MirroredStrategy()

    # Create a dataset
    dataset = dataset_ops.Dataset.TFRecordDataset([
      "/a/1.tfr", "/a/2.tfr", "/a/3.tfr", "/a/4.tfr"])

    # Distribute that dataset
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    @tf.function(input_signature=[dist_dataset.element_spec])
    def train_step(inputs):
      # train model with inputs
      return

    # Iterate over the distributed dataset
    for x in dist_dataset:
      # process dataset elements
      strategy.run(train_step, args=(x,))
    ```

    Args:
      dataset: `tf.data.Dataset` that will be sharded across all replicas using
        the rules stated above.

    Returns:
      A "distributed `Dataset`", which acts like a `tf.data.Dataset` except
      it produces "per-replica" values.
    """
    return self._extended._experimental_distribute_dataset(dataset)  # pylint: disable=protected-access

  def experimental_distribute_datasets_from_function(self, dataset_fn):
    """Distributes `tf.data.Dataset` instances created by calls to `dataset_fn`.

    `dataset_fn` will be called once for each worker in the strategy. Each
    replica on that worker will dequeue one batch of inputs from the local
    `Dataset` (i.e. if a worker has two replicas, two batches will be dequeued
    from the `Dataset` every step).

    This method can be used for several purposes. For example, where
    `experimental_distribute_dataset` is unable to shard the input files, this
    method might be used to manually shard the dataset (avoiding the slow
    fallback behavior in `experimental_distribute_dataset`). In cases where the
    dataset is infinite, this sharding can be done by creating dataset replicas
    that differ only in their random seed.
    `experimental_distribute_dataset` may also sometimes fail to split the
    batch across replicas on a worker. In that case, this method can be used
    where that limitation does not exist.

    The `dataset_fn` should take an `tf.distribute.InputContext` instance where
    information about batching and input replication can be accessed.

    You can also use the `element_spec` property of the distributed dataset
    returned by this API to query the `tf.TypeSpec` of the elements returned
    by the iterator. This can be used to set the `input_signature` property
    of a `tf.function`.

    >>> global_batch_size = 8
    >>> def dataset_fn(input_context):
    ...   batch_size = input_context.get_per_replica_batch_size(
    ...                    global_batch_size)
    ...   d = tf.data.Dataset.from_tensors([[1.]]).repeat().batch(batch_size)
    ...   return d.shard(
    ...       input_context.num_input_pipelines,
    ...       input_context.input_pipeline_id)

    >>> strategy = tf.distribute.MirroredStrategy()
    >>> ds = strategy.experimental_distribute_datasets_from_function(dataset_fn)

    >>> def train(ds):
    ...   @tf.function(input_signature=[ds.element_spec])
    ...   def step_fn(inputs):
    ...     # train the model with inputs
    ...     return inputs

    ...   for batch in ds:
    ...     replica_results = strategy.run(replica_fn, args=(batch,))
    >>> train(ds)

    IMPORTANT: The `tf.data.Dataset` returned by `dataset_fn` should have a
    per-replica batch size, unlike `experimental_distribute_dataset`, which uses
    the global batch size.  This may be computed using
    `input_context.get_per_replica_batch_size`.

    Args:
      dataset_fn: A function taking a `tf.distribute.InputContext` instance and
        returning a `tf.data.Dataset`.

    Returns:
      A "distributed `Dataset`", which acts like a `tf.data.Dataset` except
      it produces "per-replica" values.
    """
    return self._extended._experimental_distribute_datasets_from_function(  # pylint: disable=protected-access
        dataset_fn)

  def run(self, fn, args=(), kwargs=None, options=None):
    """Run `fn` on each replica, with the given arguments.

    Executes ops specified by `fn` on each replica. If `args` or `kwargs` have
    `tf.distribute.DistributedValues`, such as those produced by a
    "distributed `Dataset`" or `experimental_distribute_values_from_function`
    when `fn` is executed on a particular replica, it will be executed with the
    component of `tf.distribute.DistributedValues` that correspond to that
    replica.

    `fn` may call `tf.distribute.get_replica_context()` to access members such
    as `all_reduce`.

    All arguments in `args` or `kwargs` should either be nest of tensors or
    `tf.distribute.DistributedValues` containing tensors or composite tensors.

    IMPORTANT: Depending on the implementation of `tf.distribute.Strategy` and
    whether eager execution is enabled, `fn` may be called one or more times. If
    `fn` is annotated with `tf.function` or `tf.distribute.Strategy.run` is
    called inside a `tf.function`, eager execution is disabled and `fn` is
    called once (or once per replica, if you are using MirroredStrategy) to
    generate a Tensorflow graph, which will then be reused for execution with
    new inputs. Otherwise, if eager execution is enabled, `fn` will be called
    every step just like regular python code.

    Example usage:

    1. Constant tensor input.

    >>> strategy = tf.distribute.MirroredStrategy()
    >>> tensor_input = tf.constant(3.0)
    >>> @tf.function
    ... def replica_fn(input):
    ...   return input*2.0
    >>> result = strategy.run(replica_fn, args=(tensor_input,))
    >>> result
    <tf.Tensor: shape=(), dtype=float32, numpy=6.0>

    2. DistributedValues input.

    >>> strategy = tf.distribute.MirroredStrategy()
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
    <tf.Tensor: shape=(), dtype=int32, numpy=2>

    Args:
      fn: The function to run. The output must be a `tf.nest` of `Tensor`s.
      args: (Optional) Positional arguments to `fn`.
      kwargs: (Optional) Keyword arguments to `fn`.
      options: (Optional) An instance of `tf.distribute.RunOptions` specifying
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
      # applied when when the caller is also in Eager mode.
      fn = autograph.tf_convert(
          fn, autograph_ctx.control_status_ctx(), convert_by_default=False)
      return self._extended.call_for_each_replica(fn, args=args, kwargs=kwargs)

  # TODO(b/151224785): Remove deprecated alias.
  @doc_controls.do_not_doc_inheritable  # DEPRECATED
  @deprecation.deprecated(None, "renamed to `run`")
  def experimental_run_v2(self, fn, args=(), kwargs=None, options=None):
    return self.run(fn, args=args, kwargs=kwargs, options=options)

  def reduce(self, reduce_op, value, axis):
    """Reduce `value` across replicas.

    Given a per-replica value returned by `run`, say a
    per-example loss, the batch will be divided across all the replicas.  This
    function allows you to aggregate across replicas and optionally also across
    batch elements.  For example, if you have a global batch size of 8 and 2
    replicas, values for examples `[0, 1, 2, 3]` will be on replica 0 and
    `[4, 5, 6, 7]` will be on replica 1. By default, `reduce` will just
    aggregate across replicas, returning `[0+4, 1+5, 2+6, 3+7]`. This is useful
    when each replica is computing a scalar or some other value that doesn't
    have a "batch" dimension (like a gradient). More often you will want to
    aggregate across the global batch, which you can get by specifying the batch
    dimension as the `axis`, typically `axis=0`. In this case it would return a
    scalar `0+1+2+3+4+5+6+7`.

    If there is a last partial batch, you will need to specify an axis so
    that the resulting shape is consistent across replicas. So if the last
    batch has size 6 and it is divided into [0, 1, 2, 3] and [4, 5], you
    would get a shape mismatch unless you specify `axis=0`. If you specify
    `tf.distribute.ReduceOp.MEAN`, using `axis=0` will use the correct
    denominator of 6. Contrast this with computing `reduce_mean` to get a
    scalar value on each replica and this function to average those means,
    which will weigh some values `1/8` and others `1/4`.

    Args:
      reduce_op: A `tf.distribute.ReduceOp` value specifying how values should
        be combined.
      value: A "per replica" value, e.g. returned by `run` to
        be combined into a single tensor.
      axis: Specifies the dimension to reduce along within each
        replica's tensor. Should typically be set to the batch dimension, or
        `None` to only reduce across replicas (e.g. if the tensor has no batch
        dimension).

    Returns:
      A `Tensor`.
    """
    # TODO(josh11b): support `value` being a nest.
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

          def reduce_sum_fn(v):
            return self.run(reduce_sum, args=(v,))

          self._reduce_sum_fns[axis] = def_function.function(reduce_sum_fn)
        value = self._reduce_sum_fns[axis](value)
      else:
        value = self.run(reduce_sum, args=(value,))

      return self._extended._reduce(reduce_op, value)  # pylint: disable=protected-access
    if reduce_op != reduce_util.ReduceOp.MEAN:
      raise TypeError("Expected `reduce_op` to be a `tf.distribute.ReduceOp`, "
                      "not: %r" % reduce_op)
    # TODO(josh11b): Support list/tuple and tensor axis values.
    if not isinstance(axis, six.integer_types):
      raise TypeError("Expected `axis` to be an integer not: %r" % axis)

    def mean_reduce_helper(v, axis=axis):
      """Computes the numerator and denominator on each replica."""
      numer = math_ops.reduce_sum(v, axis=axis)
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
          return numer, array_ops.identity(
              constant_op.constant(dim, dtype=dtypes.int64))
      elif axis < 0:
        axis = axis + array_ops.rank(v)
      # TODO(b/151871486): Remove array_ops.identity after we fallback to simple
      # reduction if inputs are all on CPU.
      denom = array_ops.identity(
          array_ops.shape_v2(v, out_type=dtypes.int64)[axis])
      # TODO(josh11b): Should we cast denom to v.dtype here instead of after the
      # reduce is complete?
      return numer, denom

    if eager_context.executing_eagerly():
      # As some strategies (e.g. TPUStrategy) doesn't support pure eager
      # execution, wrap the `mean_reduce_helper` with a `tf.function` so it can
      # be run from eager mode. Cache the tf.function by `axis` to avoid the
      # same function to be traced again.
      if axis not in self._mean_reduce_helper_fns:

        def mean_reduce_fn(v):
          return self.run(mean_reduce_helper, args=(v,))

        self._mean_reduce_helper_fns[axis] = def_function.function(
            mean_reduce_fn)
      numer, denom = self._mean_reduce_helper_fns[axis](value)
    else:
      numer, denom = self.run(mean_reduce_helper, args=(value,))

    # TODO(josh11b): Should batch reduce here instead of doing two.
    numer = self._extended._reduce(reduce_util.ReduceOp.SUM, numer)  # pylint: disable=protected-access
    denom = self._extended._reduce(reduce_util.ReduceOp.SUM, denom)  # pylint: disable=protected-access
    denom = math_ops.cast(denom, numer.dtype)
    return math_ops.truediv(numer, denom)

  @doc_controls.do_not_doc_inheritable  # DEPRECATED
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
      value: A value returned by `experimental_run()`, `run()`,
        `extended.call_for_each_replica()`, or a variable created in `scope`.

    Returns:
      A tuple of values contained in `value`. If `value` represents a single
      value, this returns `(value,).`
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


@tf_export("distribute.Strategy", v1=[])  # pylint: disable=g-missing-docstring
class Strategy(StrategyBase):

  __doc__ = StrategyBase.__doc__

  def experimental_assign_to_logical_device(self, tensor, logical_device_id):
    """Adds annotation that `tensor` will be assigned to a logical device.

    NOTE: This API is only supported in TPUStrategy for now.
    This adds an annotation to `tensor` specifying that operations on
    `tensor` will be invoked on logical core device id `logical_device_id`.
    When model parallelism is used, the default behavior is that all ops
    are placed on zero-th logical device.

    ```python

    # Initializing TPU system with 2 logical devices and 4 replicas.
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(
        topology,
        computation_shape=[1, 1, 2],
        num_replicas=4)
    strategy = tf.distribute.experimental.TPUStrategy(
        resolver, device_assignment=device_assignment)
    iterator = iter(inputs)

    @tf.function()
    def step_fn(inputs):
      output = tf.add(inputs, inputs)

      // Add operation will be executed on logical device 0.
      output = strategy.experimental_assign_to_logical_device(output, 0)
      return output

    strategy.run(step_fn, args=(next(iterator),))
    ```

    Args:
      tensor: Input tensor to annotate.
      logical_device_id: Id of the logical core to which the tensor will be
        assigned.

    Raises:
      ValueError: The logical device id presented is not consistent with total
      number of partitions specified by the device assignment.

    Returns:
      Annotated tensor with idential value as `tensor`.
    """
    return self._extended._experimental_assign_to_logical_device(  # pylint: disable=protected-access
        tensor, logical_device_id)

  def experimental_split_to_logical_devices(self, tensor, partition_dimensions):
    """Adds annotation that `tensor` will be split across logical devices.

    NOTE: This API is only supported in TPUStrategy for now.
    This adds an annotation to tensor `tensor` specifying that operations on
    `tensor` will be be split among multiple logical devices. Tensor `tensor`
    will be split across dimensions specified by `partition_dimensions`.
    The dimensions of `tensor` must be divisible by corresponding value in
    `partition_dimensions`.

    For example, for system with 8 logical devices, if `tensor` is an image
    tensor with shape (batch_size, width, height, channel) and
    `partition_dimensions` is [1, 2, 4, 1], then `tensor` will be split
    2 in width dimension and 4 way in height dimension and the split
    tensor values will be fed into 8 logical devices.

    ```python
    # Initializing TPU system with 8 logical devices and 1 replica.
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(
        topology,
        computation_shape=[2, 2, 2],
        num_replicas=1)
    strategy = tf.distribute.experimental.TPUStrategy(
        resolver, device_assignment=device_assignment)

    iterator = iter(inputs)

    @tf.function()
    def step_fn(inputs):
      inputs = strategy.experimental_split_to_logical_devices(
        inputs, [1, 2, 4, 1])

      // model() function will be executed on 8 logical devices with `inputs`
      // split 2 * 4  ways.
      output = model(inputs)
      return output

    strategy.run(step_fn, args=(next(iterator),))
    ```
    Args:
      tensor: Input tensor to annotate.
      partition_dimensions: An unnested list of integers with the size equal to
        rank of `tensor` specifying how `tensor` will be partitioned. The
        product of all elements in `partition_dimensions` must be equal to the
        total number of logical devices per replica.

    Raises:
      ValueError: 1) If the size of partition_dimensions does not equal to rank
        of `tensor` or 2) if product of elements of `partition_dimensions` does
        not match the number of logical devices per replica defined by the
        implementing DistributionStrategy's device specification or
        3) if a known size of `tensor` is not divisible by corresponding
        value in `partition_dimensions`.

    Returns:
      Annotated tensor with idential value as `tensor`.
    """
    return self._extended._experimental_split_to_logical_devices(  # pylint: disable=protected-access
        tensor, partition_dimensions)

  def experimental_replicate_to_logical_devices(self, tensor):
    """Adds annotation that `tensor` will be replicated to all logical devices.

    NOTE: This API is only supported in TPUStrategy for now.
    This adds an annotation to tensor `tensor` specifying that operations on
    `tensor` will be invoked on all logical devices.

    ```python
    # Initializing TPU system with 2 logical devices and 4 replicas.
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(
        topology,
        computation_shape=[1, 1, 2],
        num_replicas=4)
    strategy = tf.distribute.experimental.TPUStrategy(
        resolver, device_assignment=device_assignment)

    iterator = iter(inputs)

    @tf.function()
    def step_fn(inputs):
      images, labels = inputs
      images = strategy.experimental_split_to_logical_devices(
        inputs, [1, 2, 4, 1])

      // model() function will be executed on 8 logical devices with `inputs`
      // split 2 * 4  ways.
      output = model(inputs)

      // For loss calculation, all logical devices share the same logits
      // and labels.
      labels = strategy.experimental_replicate_to_logical_devices(labels)
      output = strategy.experimental_replicate_to_logical_devices(output)
      loss = loss_fn(labels, output)

      return loss

    strategy.run(step_fn, args=(next(iterator),))
    ```
    Args:
      tensor: Input tensor to annotate.

    Returns:
      Annotated tensor with idential value as `tensor`.
    """
    return self._extended._experimental_replicate_to_logical_devices(tensor)  # pylint: disable=protected-access

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

    1. Return constant value per replica:

    >>> strategy = tf.distribute.MirroredStrategy()
    >>> def value_fn(ctx):
    ...   return tf.constant(1.)
    >>> distributed_values = (
    ...      strategy.experimental_distribute_values_from_function(
    ...        value_fn))
    >>> local_result = strategy.experimental_local_results(distributed_values)
    >>> local_result
    (<tf.Tensor: shape=(), dtype=float32, numpy=1.0>,)

    2. Distribute values in array based on replica_id:

    >>> strategy = tf.distribute.MirroredStrategy()
    >>> array_value = np.array([3., 2., 1.])
    >>> def value_fn(ctx):
    ...   return array_value[ctx.replica_id_in_sync_group]
    >>> distributed_values = (
    ...      strategy.experimental_distribute_values_from_function(
    ...        value_fn))
    >>> local_result = strategy.experimental_local_results(distributed_values)
    >>> local_result
    (3.0,)

    3. Specify values using num_replicas_in_sync:

    >>> strategy = tf.distribute.MirroredStrategy()
    >>> def value_fn(ctx):
    ...   return ctx.num_replicas_in_sync
    >>> distributed_values = (
    ...      strategy.experimental_distribute_values_from_function(
    ...        value_fn))
    >>> local_result = strategy.experimental_local_results(distributed_values)
    >>> local_result
    (1,)

    4. Place values on devices and distribute:

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


# NOTE(josh11b): For any strategy that needs to support tf.compat.v1,
# instead descend from StrategyExtendedV1.
@tf_export("distribute.StrategyExtended", v1=[])
class StrategyExtendedV2(object):
  """Additional APIs for algorithms that need to be distribution-aware.

  Note: For most usage of `tf.distribute.Strategy`, there should be no need to
  call these methods, since TensorFlow libraries (such as optimizers) already
  call these methods when needed on your behalf.

  Lower-level concepts:

  * Wrapped values: In order to represent values parallel across devices
    (either replicas or the devices associated with a particular value), we
    wrap them in a "PerReplica" or "Mirrored" object that contains a map
    from replica id to values. "PerReplica" is used when the value may be
    different across replicas, and "Mirrored" when the value are the same.
  * Unwrapping and merging: Consider calling a function `fn` on multiple
    replicas, like `run(fn, args=[w])` with an
    argument `w` that is a wrapped value. This means `w` will have a map taking
    replica id `0` to `w0`, replica id `1` to `w1`, etc.
    `run()` unwraps `w` before calling `fn`, so
    it calls `fn(w0)` on `d0`, `fn(w1)` on `d1`, etc.  It then merges the return
    values from `fn()`, which can possibly result in wrapped values. For
    example, let's say `fn()` returns a tuple with three components: `(x, a,
    v0)` from replica 0, `(x, b, v1)` on replica 1, etc. If the first component
    is the same object `x` from every replica, then the first component of the
    merged result will also be `x`. If the second component is different (`a`,
    `b`, ...)  from each replica, then the merged value will have a wrapped map
    from replica device to the different values. If the third component is the
    members of a mirrored variable (`v` maps `d0` to `v0`, `d1` to `v1`, etc.),
    then the merged result will be that mirrored variable (`v`).
  * Worker devices vs. parameter devices: Most replica computations will
    happen on worker devices. Since we don't yet support model
    parallelism, there will be one worker device per replica. When using
    parameter servers or central storage, the set of devices holding
    variables may be different, otherwise the parameter devices might
    match the worker devices.

  *Replica context vs. Cross-replica context*

  A _replica context_ applies when we are in some function that is being called
  once for each replica.  Otherwise we are in cross-replica context, which is
  useful for calling `tf.distribute.Strategy` methods which operate across the
  replicas (like `reduce_to()`). By default you start in a replica context
  (the "default single replica context") and then some methods can switch you
  back and forth. There is a third mode you can be in called _update context_
  used when updating variables.

  * `tf.distribute.Strategy.scope`: enters cross-replica context when
    no other strategy is in scope.
  * `tf.distribute.Strategy.run`: calls a function in
    replica context.
  * `tf.distribute.ReplicaContext.merge_call`: transitions from replica
    context to cross-replica context.
  * `tf.distribute.StrategyExtended.update`: calls a function in an update
    context from a cross-replica context.

  In a replica context, you may freely read the values of variables, but
  you may only update their value if they specify a way to aggregate the
  update using the `aggregation` parameter in the variable's constructor.
  In a cross-replica context, you may read or write variables (writes may
  need to be broadcast to all copies of the variable if it is mirrored).

  *Sync on read variables*

  In some cases, such as a metric, we want to accumulate a bunch of updates on
  each replica independently and only aggregate when reading. This can be a big
  performance win when the value is read only rarely (maybe the value is only
  read at the end of an epoch or when checkpointing).  These are variables
  created by passing `synchronization=ON_READ` to the variable's constructor
  (and some value for `aggregation`).

  The strategy may choose to put the variable on multiple devices, like mirrored
  variables, but unlike mirrored variables we don't synchronize the updates to
  them to make sure they have the same value. Instead, the synchronization is
  performed when reading in cross-replica context.  In a replica context, reads
  and writes are performed on the local copy (we allow reads so you can write
  code like `v = 0.9*v + 0.1*update`).  We don't allow operations like
  `v.assign_add` in a cross-replica context for sync on read variables; right
  now we don't have a use case for such updates and depending on the aggregation
  mode such updates may not be sensible.

  *Locality*

  Depending on how a value is produced, it will have a type that will determine
  how it may be used.

  "Per-replica" values exist on the worker devices, with a different value for
  each replica. They are produced by iterating through a "distributed `Dataset`"
  returned by `tf.distribute.Strategy.experimental_distribute_dataset` and
  `tf.distribute.Strategy.experimental_distribute_datasets_from_function`.  They
  are also the typical result returned by
  `tf.distribute.Strategy.run`. You typically can't use a
  per-replica value directly in a cross-replica context, without first resolving
  how to aggregate the values across replicas, for instance by using
  `tf.distribute.Strategy.reduce`.

  "Mirrored" values are like per-replica values, except we know that the value
  on all replicas are the same. We can safely read a mirrored value in a
  cross-replica context by using the value on any replica. You can convert
  a per-replica value into a mirrored value by using
  `tf.distribute.ReplicaContext.all_reduce`.

  Values can also have the same locality as a variable, which is a mirrored
  value but residing on the same devices as the variable (as opposed to the
  compute devices). Such values may be passed to a call to
  `tf.distribute.StrategyExtended.update` to update the value of a variable.
  You may use `tf.distribute.StrategyExtended.colocate_vars_with` to give a
  variable the same locality as another variable. This is useful, for example,
  for "slot" variables used by an optimizer for keeping track of statistics
  used to update a primary/model variable. You may convert a per-replica
  value to a variable's locality by using
  `tf.distribute.StrategyExtended.reduce_to` or
  `tf.distribute.StrategyExtended.batch_reduce_to`.

  In addition to slot variables which should be colocated with their primary
  variables, optimizers also define non-slot variables. These can be things like
  "number of step updates performed" or "beta1^t" and "beta2^t".  Each strategy
  has some policy for which devices those variables should be copied too, called
  the "non-slot devices" (some subset of the parameter devices). We require that
  all non-slot variables are allocated on the same device, or mirrored across
  the same set of devices. You can use
  `tf.distribute.StrategyExtended.non_slot_devices` to pick a consistent set of
  devices to pass to both `tf.distribute.StrategyExtended.colocate_vars_with`
  and `tf.distribute.StrategyExtended.update_non_slot`.

  *How to update a variable*

  The standard pattern for updating variables is to:

  1. In your function passed to `tf.distribute.Strategy.run`,
     compute a list of (update, variable) pairs. For example, the update might
     be a the gradient of the loss with respect to the variable.
  2. Switch to cross-replica mode by calling
     `tf.distribute.get_replica_context().merge_call()` with the updates and
     variables as arguments.
  3. Call
     `tf.distribute.StrategyExtended.reduce_to(VariableAggregation.SUM, t, v)`
     (for one variable) or `tf.distribute.StrategyExtended.batch_reduce_to`
     (for a list of variables) to sum the updates.
     and broadcast the result to the variable's devices.
  4. Call `tf.distribute.StrategyExtended.update(v)` for each variable to update
     its value.

  Steps 2 through 4 are done automatically by class
  `tf.keras.optimizers.Optimizer` if you call its
  `tf.keras.optimizers.Optimizer.apply_gradients` method in a replica context.
  They are also done automatically if you call an `assign*` method on a (non
  sync-on-read) variable that was constructed with an aggregation method (which
  is used to determine the reduction used in step 3).

  *Distribute-aware layers*

  Layers are generally called in a replica context, except when defining a
  functional model. `tf.distribute.in_cross_replica_context` will let you
  determine which case you are in. If in a replica context,
  the `tf.distribute.get_replica_context` function will return a
  `tf.distribute.ReplicaContext` object. The `ReplicaContext` object has an
  `all_reduce` method for aggregating across all replicas. Alternatively, you
  can update variables following steps 2-4 above.

  Note: For new `tf.distribute.Strategy` implementations, please put all logic
  in a subclass of `tf.distribute.StrategyExtended`. The only code needed for
  the `tf.distribute.Strategy` subclass is for instantiating your subclass of
  `tf.distribute.StrategyExtended` in the `__init__` method.
  """

  def __init__(self, container_strategy):
    self._container_strategy_weakref = weakref.ref(container_strategy)
    self._default_device = None
    # This property is used to determine if we should set drop_remainder=True
    # when creating Datasets from numpy array inputs.
    self._require_static_shapes = False

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
      else:
        checkpoint_restore_uid = None

      created = self._create_variable(next_creator, **kwargs)

      if checkpoint_restore_uid is not None:
        # pylint: disable=protected-access
        # Let the checkpointing infrastructure know that the variable was
        # already restored so it doesn't waste memory loading the value again.
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
            custom_getter=distributed_getter), self._default_device)

  def _allow_variable_partition(self):
    return False

  def _create_variable(self, next_creator, **kwargs):
    # Note: should support "colocate_with" argument.
    raise NotImplementedError("must be implemented in descendants")

  def variable_created_in_scope(self, v):
    """Tests whether `v` was created while this strategy scope was active.

    Variables created inside the strategy scope are "owned" by it:

    ```python
    strategy = tf.distribute.StrategyExtended()
    with strategy.scope():
      v = tf.Variable(1.)
    strategy.variable_created_in_scope(v)
    True
    ```

    Variables created outside the strategy are not owned by it:

    ```python
    v = tf.Variable(1.)
    strategy.variable_created_in_scope(v)
    False
    ```

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

  def _experimental_assign_to_logical_device(self, tensor, logical_device_id):
    raise NotImplementedError("This method should be overriden by "
                              "sub-classes which support model parallelism.")

  def _experimental_split_to_logical_devices(self, tensor,
                                             partition_dimensions):
    raise NotImplementedError("This method should be overriden by "
                              "sub-classes which support model parallelism.")

  def _experimental_replicate_to_logical_devices(self, tensor):
    raise NotImplementedError("This method should be overriden by "
                              "sub-classes which support model parallelism.")

  def _make_dataset_iterator(self, dataset):
    raise NotImplementedError("must be implemented in descendants")

  def _make_input_fn_iterator(self, input_fn, replication_mode):
    raise NotImplementedError("must be implemented in descendants")

  def _experimental_distribute_dataset(self, dataset):
    raise NotImplementedError("must be implemented in descendants")

  def _experimental_distribute_datasets_from_function(self, dataset_fn):
    raise NotImplementedError("must be implemented in descendants")

  def _experimental_distribute_values_from_function(self, value_fn):
    raise NotImplementedError("must be implemented in descendants")

  def _reduce(self, reduce_op, value):
    # Default implementation until we have an implementation for each strategy.
    dst = device_util.current() or self._default_device or "/device:CPU:0"
    return self._local_results(self.reduce_to(reduce_op, value, dst))[0]

  def reduce_to(self, reduce_op, value, destinations, experimental_hints=None):
    """Combine (via e.g. sum or mean) values across replicas.

    Args:
      reduce_op: Reduction type, an instance of `tf.distribute.ReduceOp` enum.
      value: A per-replica value with one value per replica.
      destinations: A mirrored variable, a per-replica tensor, or a device
        string. The return value will be copied to all destination devices (or
        all the devices where the `destinations` value resides). To perform an
        all-reduction, pass `value` to `destinations`.
      experimental_hints: A `tf.distrbute.experimental.CollectiveHints`. Hints
        to perform collective operations.

    Returns:
      A tensor or value mirrored to `destinations`.
    """
    # TODO(josh11b): More docstring
    _require_cross_replica_or_default_context_extended(self)
    assert not isinstance(destinations, (list, tuple))
    assert not isinstance(reduce_op, variable_scope.VariableAggregation)
    if isinstance(reduce_op, six.string_types):
      reduce_op = reduce_util.ReduceOp(reduce_op.upper())
    assert (reduce_op == reduce_util.ReduceOp.SUM or
            reduce_op == reduce_util.ReduceOp.MEAN)
    if experimental_hints is None:
      experimental_hints = collective_util.Hints()
    return self._reduce_to(reduce_op, value, destinations, experimental_hints)

  def _reduce_to(self, reduce_op, value, destinations, experimental_hints):
    raise NotImplementedError("must be implemented in descendants")

  def batch_reduce_to(self,
                      reduce_op,
                      value_destination_pairs,
                      experimental_hints=None):
    """Combine multiple `reduce_to` calls into one for faster execution.

    Args:
      reduce_op: Reduction type, an instance of `tf.distribute.ReduceOp` enum.
      value_destination_pairs: A sequence of (value, destinations) pairs. See
        `reduce_to()` for a description.
      experimental_hints: A `tf.distrbute.experimental.CollectiveHints`. Hints
        to perform collective operations.

    Returns:
      A list of mirrored values, one per pair in `value_destination_pairs`.
    """
    # TODO(josh11b): More docstring
    _require_cross_replica_or_default_context_extended(self)
    assert not isinstance(reduce_op, variable_scope.VariableAggregation)
    if isinstance(reduce_op, six.string_types):
      reduce_op = reduce_util.ReduceOp(reduce_op.upper())
    if experimental_hints is None:
      experimental_hints = collective_util.Hints()
    return self._batch_reduce_to(reduce_op, value_destination_pairs,
                                 experimental_hints)

  def _batch_reduce_to(self, reduce_op, value_destination_pairs,
                       experimental_hints):
    return [
        self.reduce_to(
            reduce_op, t, destinations=v, experimental_hints=experimental_hints)
        for t, v in value_destination_pairs
    ]

  def update(self, var, fn, args=(), kwargs=None, group=True):
    """Run `fn` to update `var` using inputs mirrored to the same devices.

    If `var` is mirrored across multiple devices, then this implements
    logic like:

    ```
    results = {}
    for device, v in var:
      with tf.device(device):
        # args and kwargs will be unwrapped if they are mirrored.
        results[device] = fn(v, *args, **kwargs)
    return merged(results)
    ```

    Otherwise this returns `fn(var, *args, **kwargs)` colocated with `var`.

    Neither `args` nor `kwargs` may contain per-replica values.
    If they contain mirrored values, they will be unwrapped before
    calling `fn`.

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
    _require_cross_replica_or_default_context_extended(self)
    if kwargs is None:
      kwargs = {}
    fn = autograph.tf_convert(
        fn, autograph_ctx.control_status_ctx(), convert_by_default=False)
    with self._container_strategy().scope():
      return self._update(var, fn, args, kwargs, group)

  def _update(self, var, fn, args, kwargs, group):
    raise NotImplementedError("must be implemented in descendants")

  def update_non_slot(
      self, colocate_with, fn, args=(), kwargs=None, group=True):
    """Runs `fn(*args, **kwargs)` on `colocate_with` devices.

    Args:
      colocate_with: The return value of `non_slot_devices()`.
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

  def _local_results(self, distributed_value):
    raise NotImplementedError("must be implemented in descendants")

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
    # TODO(josh11b): More docstring
    raise NotImplementedError("must be implemented in descendants")

  @property
  def parameter_devices(self):
    """Returns the tuple of all devices used to place variables."""
    # TODO(josh11b): More docstring
    raise NotImplementedError("must be implemented in descendants")

  def non_slot_devices(self, var_list):
    """Device(s) for non-slot variables.

    Create variables on these devices in a
    `with colocate_vars_with(non_slot_devices(...)):` block.
    Update those using `update_non_slot()`.

    Args:
      var_list: The list of variables being optimized, needed with the
        default `tf.distribute.Strategy`.
    Returns:
      A sequence of devices for non-slot variables.
    """
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
    used by higher-level apis such as Keras' `model.fit()` to infer
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
    # TODO(josh11b): More docstring
    _require_cross_replica_or_default_context_extended(self)
    assert not isinstance(destinations, (list, tuple))
    return self._broadcast_to(tensor, destinations)

  def _broadcast_to(self, tensor, destinations):
    raise NotImplementedError("must be implemented in descendants")

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
@tf_export("distribute.ReplicaContext")
class ReplicaContext(object):
  """`tf.distribute.Strategy` API when in a replica context.

  You can use `tf.distribute.get_replica_context` to get an instance of
  `ReplicaContext`. This should be inside your replicated step function, such
  as in a `tf.distribute.Strategy.run` call.
  """

  def __init__(self, strategy, replica_id_in_sync_group):
    self._strategy = strategy
    self._thread_context = distribution_strategy_context._InReplicaThreadMode(  # pylint: disable=protected-access
        self)
    self._replica_id_in_sync_group = replica_id_in_sync_group
    self._summary_recording_distribution_strategy = None

  def __enter__(self):
    _push_per_thread_mode(self._thread_context)

    def replica_id_is_zero():
      return math_ops.equal(self._replica_id_in_sync_group,
                            constant_op.constant(0))

    summary_state = summary_ops_v2._summary_state  # pylint: disable=protected-access
    self._summary_recording_distribution_strategy = (
        summary_state.is_recording_distribution_strategy)
    summary_state.is_recording_distribution_strategy = replica_id_is_zero

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
        distribution_strategy_context._CrossReplicaThreadMode(self._strategy))  # pylint: disable=protected-access
    try:
      return merge_fn(self._strategy, *args, **kwargs)
    finally:
      _pop_per_thread_mode()

  @property
  def num_replicas_in_sync(self):
    """Returns number of replicas over which gradients are aggregated."""
    return self._strategy.num_replicas_in_sync

  @property
  def replica_id_in_sync_group(self):
    """Returns the id of the replica being defined.

    This identifies the replica that is part of a sync group. Currently we
    assume that all sync groups contain the same number of replicas. The value
    of the replica id can range from 0 to `num_replica_in_sync` - 1.

    NOTE: This is not guaranteed to be the same ID as the XLA replica ID use
    for low-level operations such as collective_permute.
    """
    require_replica_context(self)
    return self._replica_id_in_sync_group

  @property
  def strategy(self):
    """The current `tf.distribute.Strategy` object."""
    return self._strategy

  @property
  def devices(self):
    """The devices this replica is to be executed on, as a tuple of strings."""
    require_replica_context(self)
    return (device_util.current(),)

  def all_reduce(self, reduce_op, value, experimental_hints=None):
    """All-reduces the given `value Tensor` nest across replicas.

    If `all_reduce` is called in any replica, it must be called in all replicas.
    The nested structure and `Tensor` shapes must be identical in all replicas.

    IMPORTANT: The ordering of communications must be identical in all replicas.

    Example with two replicas:
      Replica 0 `value`: {'a': 1, 'b': [40, 1]}
      Replica 1 `value`: {'a': 3, 'b': [ 2, 98]}

      If `reduce_op` == `SUM`:
        Result (on all replicas): {'a': 4, 'b': [42, 99]}

      If `reduce_op` == `MEAN`:
        Result (on all replicas): {'a': 2, 'b': [21, 49.5]}

    Args:
      reduce_op: Reduction type, an instance of `tf.distribute.ReduceOp` enum.
      value: The nested structure of `Tensor`s to all-reduce. The structure must
        be compatible with `tf.nest`.
      experimental_hints: A `tf.distrbute.experimental.CollectiveHints`. Hints
        to perform collective operations.

    Returns:
       A `Tensor` nest with the reduced `value`s from each replica.
    """
    if isinstance(reduce_op, six.string_types):
      reduce_op = reduce_util.ReduceOp(reduce_op.upper())
    if experimental_hints is None:
      experimental_hints = collective_util.Hints()

    def batch_all_reduce(strategy, *value_flat):
      return strategy.extended.batch_reduce_to(
          reduce_op, [(v, _batch_reduce_destination(v)) for v in value_flat],
          experimental_hints)

    if reduce_op in [reduce_util.ReduceOp.SUM, reduce_util.ReduceOp.MEAN]:
      # TODO(cjfj): Work out why `batch_reduce` doesn't return the correct grad.
      @custom_gradient.custom_gradient
      def grad_wrapper(*xs):
        ys = self.merge_call(batch_all_reduce, args=xs)
        # The gradient of an all-sum is itself an all-sum (all-mean, likewise).
        return ys, lambda *dy_s: self.all_reduce(reduce_op, dy_s)
      return nest.pack_sequence_as(value, grad_wrapper(*nest.flatten(value)))
    else:
      # TODO(cjfj): Implement gradients for other reductions.
      reduced = nest.pack_sequence_as(
          value, self.merge_call(batch_all_reduce, args=nest.flatten(value)))
      return nest.map_structure(array_ops.prevent_gradient, reduced)

  # TODO(josh11b): Implement `start_all_reduce(method, t)` for efficient
  # all-reduce. It would return a function returning the result of reducing `t`
  # across all replicas. The caller would wait to call this function until they
  # needed the reduce result, allowing an efficient implementation:
  # * With eager execution, the reduction could be performed asynchronously
  #   in the background, not blocking until the result was needed.
  # * When constructing a graph, it could batch up all reduction requests up
  #   to that point that the first result is needed. Most likely this can be
  #   implemented in terms of `merge_call()` and `batch_reduce_to()`.


def _batch_reduce_destination(x):
  """Returns the destinations for batch all-reduce."""
  if isinstance(x, ops.Tensor):
    # If this is a one device strategy.
    return x.device
  else:
    return x


# ------------------------------------------------------------------------------


_creating_default_strategy_singleton = False


class _DefaultDistributionStrategy(StrategyV1):
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

  def __init__(self, strategy):

    def creator(next_creator, **kwargs):
      _require_strategy_scope_strategy(strategy)
      return next_creator(**kwargs)

    self._var_creator_scope = variable_scope.variable_creator_scope(creator)
    self._strategy = strategy
    self._nested_count = 0

  def __enter__(self):
    # Allow this scope to be entered if this strategy is already in scope.
    if distribution_strategy_context.has_strategy():
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

  def _experimental_distribute_dataset(self, dataset):
    return dataset

  def _experimental_distribute_datasets_from_function(self, dataset_fn):
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
        variable_scope.variable(array_ops.zeros(i.shape, i.dtype),
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
    with ReplicaContext(
        self._container_strategy(),
        replica_id_in_sync_group=constant_op.constant(0, dtypes.int32)):
      return fn(*args, **kwargs)

  def _reduce_to(self, reduce_op, value, destinations, experimental_hints):
    # TODO(josh11b): Use destinations?
    del reduce_op, destinations, experimental_hints
    return value

  def _update(self, var, fn, args, kwargs, group):
    # The implementations of _update() and _update_non_slot() are identical
    # except _update() passes `var` as the first argument to `fn()`.
    return self._update_non_slot(var, fn, (var,) + tuple(args), kwargs, group)

  def _update_non_slot(self, colocate_with, fn, args, kwargs, should_group):
    # TODO(josh11b): Figure out what we should be passing to UpdateContext()
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


# ------------------------------------------------------------------------------
# We haven't yet implemented deserialization for DistributedVariables.
# So here we catch any attempts to deserialize variables
# when using distribution strategies.
# pylint: disable=protected-access
_original_from_proto = resource_variable_ops._from_proto_fn


def _from_proto_fn(v, import_scope=None):
  if distribution_strategy_context.has_strategy():
    raise NotImplementedError(
        "Deserialization of variables is not yet supported when using a "
        "tf.distribute.Strategy.")
  else:
    return _original_from_proto(v, import_scope=import_scope)

resource_variable_ops._from_proto_fn = _from_proto_fn
# pylint: enable=protected-access


#-------------------------------------------------------------------------------
# Shorthand for some methods from distribution_strategy_context.
_push_per_thread_mode = distribution_strategy_context._push_per_thread_mode  # pylint: disable=protected-access
_get_per_thread_mode = distribution_strategy_context._get_per_thread_mode  # pylint: disable=protected-access
_pop_per_thread_mode = distribution_strategy_context._pop_per_thread_mode  # pylint: disable=protected-access
_get_default_replica_mode = (
    distribution_strategy_context._get_default_replica_mode)  # pylint: disable=protected-access


# ------------------------------------------------------------------------------
# Metrics to track which distribution strategy is being called
distribution_strategy_gauge = monitoring.StringGauge(
    "/tensorflow/api/distribution_strategy",
    "Gauge to track the type of distribution strategy used.", "TFVersion")
distribution_strategy_replica_gauge = monitoring.IntGauge(
    "/tensorflow/api/distribution_strategy/replica",
    "Gauge to track the number of replica each distribution strategy used.",
    "CountType")
