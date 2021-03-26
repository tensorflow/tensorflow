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
"""A tf.distribute.Strategy for running on a single device."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


# TODO(josh11b): Do we wrap values in types to generate errors if you are
# doing something that won't work with other DistributionStrategy
# implementations?


@tf_export("distribute.OneDeviceStrategy", v1=[])
class OneDeviceStrategy(distribute_lib.Strategy):
  """A distribution strategy for running on a single device.

  Using this strategy will place any variables created in its scope on the
  specified device. Input distributed through this strategy will be
  prefetched to the specified device. Moreover, any functions called via
  `strategy.run` will also be placed on the specified device
  as well.

  Typical usage of this strategy could be testing your code with the
  tf.distribute.Strategy API before switching to other strategies which
  actually distribute to multiple devices/machines.

  For example:
  ```
  strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

  with strategy.scope():
    v = tf.Variable(1.0)
    print(v.device)  # /job:localhost/replica:0/task:0/device:GPU:0

  def step_fn(x):
    return x * 2

  result = 0
  for i in range(10):
    result += strategy.run(step_fn, args=(i,))
  print(result)  # 90
  ```
  """

  def __init__(self, device):
    """Creates a `OneDeviceStrategy`.

    Args:
      device: Device string identifier for the device on which the variables
        should be placed. See class docs for more details on how the device is
        used. Examples: "/cpu:0", "/gpu:0", "/device:CPU:0", "/device:GPU:0"
    """
    super(OneDeviceStrategy, self).__init__(OneDeviceExtended(self, device))
    distribute_lib.distribution_strategy_gauge.get_cell("V2").set(
        "OneDeviceStrategy")

  def experimental_distribute_dataset(self, dataset, options=None):  # pylint: disable=useless-super-delegation
    """Distributes a tf.data.Dataset instance provided via dataset.

    In this case, there is only one device, so this is only a thin wrapper
    around the input dataset. It will, however, prefetch the input data to the
    specified device. The returned distributed dataset can be iterated over
    similar to how regular datasets can.

    NOTE: Currently, the user cannot add any more transformations to a
    distributed dataset.

    Example:
    ```
    strategy = tf.distribute.OneDeviceStrategy()
    dataset = tf.data.Dataset.range(10).batch(2)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    for x in dist_dataset:
      print(x)  # [0, 1], [2, 3],...
    ```
    Args:
      dataset: `tf.data.Dataset` to be prefetched to device.
      options: `tf.distribute.InputOptions` used to control options on how this
        dataset is distributed.
    Returns:
      A "distributed `Dataset`" that the caller can iterate over.
    """
    return super(OneDeviceStrategy, self).experimental_distribute_dataset(
        dataset, options)

  def distribute_datasets_from_function(
      self,
      dataset_fn,  # pylint: disable=useless-super-delegation
      options=None):
    """Distributes `tf.data.Dataset` instances created by calls to `dataset_fn`.

    `dataset_fn` will be called once for each worker in the strategy. In this
    case, we only have one worker and one device so `dataset_fn` is called
    once.

    The `dataset_fn` should take an `tf.distribute.InputContext` instance where
    information about batching and input replication can be accessed:

    ```
    def dataset_fn(input_context):
      batch_size = input_context.get_per_replica_batch_size(global_batch_size)
      d = tf.data.Dataset.from_tensors([[1.]]).repeat().batch(batch_size)
      return d.shard(
          input_context.num_input_pipelines, input_context.input_pipeline_id)

    inputs = strategy.distribute_datasets_from_function(dataset_fn)

    for batch in inputs:
      replica_results = strategy.run(replica_fn, args=(batch,))
    ```

    IMPORTANT: The `tf.data.Dataset` returned by `dataset_fn` should have a
    per-replica batch size, unlike `experimental_distribute_dataset`, which uses
    the global batch size.  This may be computed using
    `input_context.get_per_replica_batch_size`.

    Args:
      dataset_fn: A function taking a `tf.distribute.InputContext` instance and
        returning a `tf.data.Dataset`.
      options: `tf.distribute.InputOptions` used to control options on how this
        dataset is distributed.

    Returns:
      A "distributed `Dataset`", which the caller can iterate over like regular
      datasets.
    """
    return super(OneDeviceStrategy,
                 self).distribute_datasets_from_function(dataset_fn, options)

  def experimental_local_results(self, value):  # pylint: disable=useless-super-delegation
    """Returns the list of all local per-replica values contained in `value`.

    In `OneDeviceStrategy`, the `value` is always expected to be a single
    value, so the result is just the value in a tuple.

    Args:
      value: A value returned by `experimental_run()`, `run()`,
        `extended.call_for_each_replica()`, or a variable created in `scope`.

    Returns:
      A tuple of values contained in `value`. If `value` represents a single
      value, this returns `(value,).`
    """
    return super(OneDeviceStrategy, self).experimental_local_results(value)

  def run(self, fn, args=(), kwargs=None, options=None):  # pylint: disable=useless-super-delegation
    """Run `fn` on each replica, with the given arguments.

    In `OneDeviceStrategy`, `fn` is simply called within a device scope for the
    given device, with the provided arguments.

    Args:
      fn: The function to run. The output must be a `tf.nest` of `Tensor`s.
      args: (Optional) Positional arguments to `fn`.
      kwargs: (Optional) Keyword arguments to `fn`.
      options: (Optional) An instance of `tf.distribute.RunOptions` specifying
        the options to run `fn`.

    Returns:
      Return value from running `fn`.
    """
    return super(OneDeviceStrategy, self).run(fn, args, kwargs, options)

  def reduce(self, reduce_op, value, axis):  # pylint: disable=useless-super-delegation
    """Reduce `value` across replicas.

    In `OneDeviceStrategy`, there is only one replica, so if axis=None, value
    is simply returned. If axis is specified as something other than None,
    such as axis=0, value is reduced along that axis and returned.

    Example:
    ```
    t = tf.range(10)

    result = strategy.reduce(tf.distribute.ReduceOp.SUM, t, axis=None).numpy()
    # result: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    result = strategy.reduce(tf.distribute.ReduceOp.SUM, t, axis=0).numpy()
    # result: 45
    ```

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
    return super(OneDeviceStrategy, self).reduce(reduce_op, value, axis)

  def scope(self):  # pylint: disable=useless-super-delegation
    """Returns a context manager selecting this Strategy as current.

    Inside a `with strategy.scope():` code block, this thread
    will use a variable creator set by `strategy`, and will
    enter its "cross-replica context".

    In `OneDeviceStrategy`, all variables created inside `strategy.scope()`
    will be on `device` specified at strategy construction time.
    See example in the docs for this class.

    Returns:
      A context manager to use for creating variables with this strategy.
    """
    return super(OneDeviceStrategy, self).scope()


@tf_export(v1=["distribute.OneDeviceStrategy"])  # pylint: disable=empty-docstring
class OneDeviceStrategyV1(distribute_lib.StrategyV1):

  __doc__ = OneDeviceStrategy.__doc__.replace(
      "For example:\n  ```",
      "For example:\n  ```\n  tf.enable_eager_execution()")

  def __init__(self, device):
    super(OneDeviceStrategyV1, self).__init__(OneDeviceExtended(self, device))
    distribute_lib.distribution_strategy_gauge.get_cell("V1").set(
        "OneDeviceStrategy")
  __init__.__doc__ = OneDeviceStrategy.__init__.__doc__


# TODO(josh11b): Switch to V2 after callers have been updated to only V2 APIs.
class OneDeviceExtended(distribute_lib.StrategyExtendedV1):
  """Implementation of OneDeviceStrategy."""

  def __init__(self, container_strategy, device):
    super(OneDeviceExtended, self).__init__(container_strategy)
    self._device = device_util.resolve(device)
    self._input_device = device_util.get_host_for_device(self._device)

  def _input_workers_with_options(self, options=None):
    if not options or options.experimental_fetch_to_device:
      return input_lib.InputWorkers([(self._input_device, (self._device,))])
    else:
      return input_lib.InputWorkers([(self._input_device,
                                      (self._input_device,))])

  @property
  def _input_workers(self):
    return self._input_workers_with_options()

  def _create_variable(self, next_creator, **kwargs):
    colocate_with = kwargs.pop("colocate_with", None)
    if colocate_with is None:
      with ops.device(self._device):
        return next_creator(**kwargs)
    elif isinstance(colocate_with, numpy_dataset.SingleDevice):
      with ops.device(colocate_with.device):
        return next_creator(**kwargs)
    else:
      with ops.colocate_with(colocate_with):
        return next_creator(**kwargs)

  def _validate_colocate_with_variable(self, colocate_with_variable):
    distribute_utils.validate_colocate(colocate_with_variable, self)

  def _make_dataset_iterator(self, dataset):
    """Make iterator from dataset without splitting the batch."""
    # Note that split_batch_by argument is not passed because it is always 1 in
    # this strategy, and adding it adds unnecessary overhead to the dataset.
    return input_lib.DatasetIterator(dataset, self._input_workers,
                                     self._container_strategy())

  def _make_input_fn_iterator(
      self,
      input_fn,
      replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
    return input_lib.InputFunctionIterator(input_fn, self._input_workers,
                                           [distribute_lib.InputContext()],
                                           self._container_strategy())

  def _experimental_make_numpy_dataset(self, numpy_input, session):
    return numpy_dataset.one_host_numpy_dataset(
        numpy_input, numpy_dataset.SingleDevice(self._input_device), session)

  def _broadcast_to(self, tensor, destinations):
    del destinations
    return tensor

  def _experimental_distribute_dataset(self, dataset, options):
    # Note that split_batch_by argument is not passed because it is always 1 in
    # this strategy, and adding it adds unnecessary overhead to the dataset.
    if (options and options.experimental_replication_mode ==
        distribute_lib.InputReplicationMode.PER_REPLICA):
      raise NotImplementedError(
          "InputReplicationMode.PER_REPLICA "
          "is only supported in  "
          "`experimental_distribute_datasets_from_function`."
      )
    return input_lib.get_distributed_dataset(
        dataset,
        self._input_workers_with_options(options),
        self._container_strategy(),
        options=options)

  def _distribute_datasets_from_function(self, dataset_fn, options):
    if (options and options.experimental_replication_mode ==
        distribute_lib.InputReplicationMode.PER_REPLICA):
      raise NotImplementedError(
          "InputReplicationMode.PER_REPLICA "
          "is only supported in "
          "`experimental_distribute_datasets_from_function` "
          "of tf.distribute.MirroredStrategy")
    return input_lib.get_distributed_datasets_from_function(
        dataset_fn,
        self._input_workers_with_options(options),
        [distribute_lib.InputContext()],
        self._container_strategy(),
        options=options)

  def _experimental_distribute_values_from_function(self, value_fn):
    # TODO(b/137795644): This should return a PerReplica value but other
    # methods like run in OneDeviceStrategy need to be modified
    # to do the same.
    return value_fn(distribute_lib.ValueContext())

  # TODO(priyag): Deal with OutOfRange errors  once b/111349762 is fixed.
  def _experimental_run_steps_on_iterator(self, fn, iterator, iterations,
                                          initial_loop_values=None):
    if initial_loop_values is None:
      initial_loop_values = {}
    initial_loop_values = nest.flatten(initial_loop_values)

    ctx = input_lib.MultiStepContext()
    def body(i, *args):
      """A wrapper around `fn` to create the while loop body."""
      del args
      fn_result = fn(ctx, iterator.get_next())
      flat_last_step_outputs = nest.flatten(ctx.last_step_outputs)
      with ops.control_dependencies([fn_result]):
        return [i + 1] + flat_last_step_outputs

    # We capture the control_flow_context at this point, before we run `fn`
    # inside a while_loop. This is useful in cases where we might need to exit
    # these contexts and get back to the outer context to do some things, for
    # e.g. create an op which should be evaluated only once at the end of the
    # loop on the host. One such usage is in creating metrics' value op.
    self._outer_control_flow_context = (
        ops.get_default_graph()._get_control_flow_context())  # pylint: disable=protected-access

    # TODO(priyag): Use max_iterations instead of an explicit counter.
    cond = lambda i, *args: i < iterations
    i = constant_op.constant(0)
    loop_result = control_flow_ops.while_loop(
        cond, body, [i] + initial_loop_values, name="",
        parallel_iterations=1, back_prop=False, swap_memory=False,
        return_same_structure=True)
    del self._outer_control_flow_context

    ctx.run_op = control_flow_ops.group(loop_result)

    # Convert the last_step_outputs from a list to the original dict structure
    # of last_step_outputs.
    last_step_tensor_outputs = loop_result[1:]
    last_step_tensor_outputs_dict = nest.pack_sequence_as(
        ctx.last_step_outputs, last_step_tensor_outputs)

    ctx._set_last_step_outputs(last_step_tensor_outputs_dict)  # pylint: disable=protected-access
    return ctx

  def _call_for_each_replica(self, fn, args, kwargs):
    strategy = self._container_strategy()
    with ops.device(self._device), _OneDeviceReplicaContext(strategy):
      return fn(*args, **kwargs)

  def _reduce_to(self, reduce_op, value, destinations, options):
    del reduce_op, destinations, options
    return value

  def _gather_to_implementation(self, value, destinations, axis, options):
    del destinations, axis, options
    return value

  def _update(self, var, fn, args, kwargs, group):
    # The implementations of _update() and _update_non_slot() are identical
    # except _update() passes `var` as the first argument to `fn()`.
    return self._update_non_slot(var, fn, (var,) + tuple(args), kwargs, group)

  def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
    del colocate_with
    with ops.device(self._device), distribute_lib.UpdateContext(self._device):
      result = fn(*args, **kwargs)
      if group:
        return result
      else:
        return nest.map_structure(self._local_results, result)

  def read_var(self, replica_local_var):
    """Read the aggregate value of a replica-local variable."""
    return array_ops.identity(replica_local_var)

  def _local_results(self, value):
    return (value,)

  def value_container(self, value):
    return value

  def _in_multi_worker_mode(self):
    """Whether this strategy indicates working in multi-worker settings."""
    return False

  @property
  def _num_replicas_in_sync(self):
    return 1

  @property
  def worker_devices(self):
    return (self._device,)

  @property
  def parameter_devices(self):
    return (self._device,)

  def non_slot_devices(self, var_list):
    del var_list
    return (self._device,)

  @property
  def experimental_should_init(self):
    return True

  @property
  def experimental_between_graph(self):
    return False

  @property
  def should_checkpoint(self):
    return True

  @property
  def should_save_summary(self):
    return True

  # TODO(priyag): Delete this once all strategies use global batch size.
  @property
  def _global_batch_size(self):
    """Global and per-replica batching are equivalent for OneDeviceStrategy."""
    return True

  @property
  def _support_per_replica_values(self):
    return False

  def _get_local_replica_id(self, replica_id_in_sync_group):
    return replica_id_in_sync_group


class _OneDeviceReplicaContext(distribute_lib.ReplicaContext):
  """ReplicaContext for OneDeviceStrategy."""

  def __init__(self, strategy):
    distribute_lib.ReplicaContext.__init__(
        self, strategy, replica_id_in_sync_group=0)

  @property
  def devices(self):
    return self._strategy.extended.worker_devices
