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

import six

from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import values
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import nest


# TODO(josh11b): Replace asserts in this file with if ...: raise ...


class OneDeviceStrategy(distribute_lib.DistributionStrategy):
  """A distribution strategy for running on a single device."""
  # TODO(josh11b): Do we wrap values in types to generate errors if you are
  # doing something that won't work with other DistributionStrategy
  # implementations?

  def __init__(self, device):
    super(OneDeviceStrategy, self).__init__(OneDeviceExtended(self, device))


class OneDeviceExtended(distribute_lib.DistributionStrategyExtended):
  """Implementation of OneDeviceStrategy."""

  def __init__(self, container_strategy, device):
    super(OneDeviceExtended, self).__init__(container_strategy)
    self._device = device
    self._default_device = device
    worker = device_util.canonicalize("/device:CPU:0")
    worker_device_pairs = [(worker, [self._device])]
    device_map = values.SingleDeviceMap(device)
    self._input_workers = values.InputWorkers(device_map, worker_device_pairs)

  def _create_variable(self, next_creator, *args, **kwargs):
    colocate_with = kwargs.pop("colocate_with", None)
    if colocate_with is None:
      with ops.device(self._device):
        return next_creator(*args, **kwargs)
    if isinstance(colocate_with, six.string_types):
      with ops.device(colocate_with):
        return next_creator(*args, **kwargs)
    if (isinstance(colocate_with, (list, tuple)) and len(colocate_with) == 1 and
        isinstance(colocate_with[0], six.string_types)):
      with ops.device(colocate_with[0]):
        return next_creator(*args, **kwargs)
    with ops.colocate_with(colocate_with):
      return next_creator(*args, **kwargs)

  def _make_dataset_iterator(self, dataset):
    """Make iterator from dataset without splitting the batch."""
    return values.DatasetIterator(dataset, self._input_workers)

  def _distribute_dataset(self, dataset_fn):
    return values.PerReplicaDataset(
        self._call_dataset_fn(dataset_fn), self._input_workers, 0)

  def _make_input_fn_iterator(
      self,
      input_fn,
      replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
    return values.InputFunctionIterator(
        input_fn, self._input_workers, [distribute_lib.InputContext()])

  def _broadcast_to(self, tensor, destinations):
    del destinations
    return tensor

  # TODO(priyag): Deal with OutOfRange errors  once b/111349762 is fixed.
  def _experimental_run_steps_on_iterator(self, fn, iterator, iterations,
                                          initial_loop_values=None):
    if initial_loop_values is None:
      initial_loop_values = {}
    initial_loop_values = nest.flatten(initial_loop_values)

    ctx = values.MultiStepContext()
    def body(i, *args):
      """A wrapper around `fn` to create the while loop body."""
      del args
      fn_inputs = iterator.get_next()
      if not isinstance(fn_inputs, tuple):
        fn_inputs = (fn_inputs,)
      fn_result = fn(ctx, fn_inputs)
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
        return nest.map_structure(self._unwrap, result)

  def read_var(self, replica_local_var):
    """Read the aggregate value of a replica-local variable."""
    return array_ops.identity(replica_local_var)

  def _unwrap(self, value):
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
    return True


class _OneDeviceReplicaContext(distribute_lib.ReplicaContext):
  """ReplicaContext for OneDeviceStrategy."""

  def __init__(self, distribution_strategy):
    distribute_lib.ReplicaContext.__init__(
        self,
        distribution_strategy,
        replica_id_in_sync_group=constant_op.constant(0, dtypes.int32))

  @property
  def devices(self):
    return self._distribution_strategy.extended.worker_devices
