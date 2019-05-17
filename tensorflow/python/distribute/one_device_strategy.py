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
"""Class OneDeviceStrategy implementing DistributionStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import values
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


# TODO(josh11b): Replace asserts in this file with if ...: raise ...

# TODO(josh11b): Do we wrap values in types to generate errors if you are
# doing something that won't work with other DistributionStrategy
# implementations?


@tf_export("distribute.OneDeviceStrategy", v1=[])
class OneDeviceStrategy(distribute_lib.Strategy):
  """A distribution strategy for running on a single device."""

  def __init__(self, device):
    super(OneDeviceStrategy, self).__init__(OneDeviceExtended(self, device))


@tf_export(v1=["distribute.OneDeviceStrategy"])
class OneDeviceStrategyV1(distribute_lib.StrategyV1):
  """A distribution strategy for running on a single device."""

  def __init__(self, device):
    super(OneDeviceStrategyV1, self).__init__(OneDeviceExtended(self, device))


# TODO(josh11b): Switch to V2 after callers have been updated to only V2 APIs.
class OneDeviceExtended(distribute_lib.StrategyExtendedV1):
  """Implementation of OneDeviceStrategy."""

  def __init__(self, container_strategy, device):
    super(OneDeviceExtended, self).__init__(container_strategy)
    self._device = device_util.canonicalize(device)
    suffix_loc = self._device.rfind("/")
    self._input_device = self._device[:suffix_loc] + "/device:CPU:0"
    worker_device_pairs = [(self._input_device, [self._device])]
    device_map = values.SingleDeviceMap(device)
    self._input_workers = input_lib.InputWorkers(
        device_map, worker_device_pairs)

  def _create_variable(self, next_creator, *args, **kwargs):
    colocate_with = kwargs.pop("colocate_with", None)
    if colocate_with is None:
      with ops.device(self._device):
        return next_creator(*args, **kwargs)
    elif isinstance(colocate_with, numpy_dataset.SingleDevice):
      with ops.device(colocate_with.device):
        return next_creator(*args, **kwargs)
    else:
      with ops.colocate_with(colocate_with):
        return next_creator(*args, **kwargs)

  def _validate_colocate_with_variable(self, colocate_with_variable):
    values.validate_colocate(colocate_with_variable, self)

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

  def _experimental_distribute_dataset(self, dataset):
    # Note that split_batch_by argument is not passed because it is always 1 in
    # this strategy, and adding it adds unnecessary overhead to the dataset.
    return input_lib.get_distributed_dataset(dataset, self._input_workers,
                                             self._container_strategy())

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

  def _reduce_to(self, reduce_op, value, destinations):
    del reduce_op, destinations
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


class _OneDeviceReplicaContext(distribute_lib.ReplicaContext):
  """ReplicaContext for OneDeviceStrategy."""

  def __init__(self, strategy):
    zero = constant_op.constant(0, dtypes.int32)
    distribute_lib.ReplicaContext.__init__(
        self, strategy, replica_id_in_sync_group=zero)

  @property
  def devices(self):
    return self._strategy.extended.worker_devices
