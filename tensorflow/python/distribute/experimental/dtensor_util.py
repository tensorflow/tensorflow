# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for strategies that are backed by DTensor."""

from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import input_util
from tensorflow.dtensor.python import layout
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2


# Default dimension name used for the mesh created when user provide a list
# of devices. For mirrored strategy, it should be a 1D mesh with batch dim only.
DEFAULT_BATCH_MESH_DIM_NAME = "batch"


class DTensorDistributedValue(values_lib.DistributedValues):
  """DistributedValue backed by a DTensor instance.

  This class is useful to align the interface between DTensor and tf.distribute.
  Most of the tf.distribute API will accept/return DistributedValue, whereas
  DTensor low level API will only accept DTensor instance. In order to avoid
  the conversion back and forth between DistributedValue and DTensor, we
  introduce this class so that it can work with both side.
  """

  def __init__(self, dtensor):
    if context.executing_eagerly():
      if not d_api.is_dtensor(dtensor):
        raise ValueError("The DTensorDistributedValue can only be built with "
                         f"DTensor instance, got {type(dtensor)}")
      super().__init__(d_api.unpack(dtensor))
    else:
      # We can't unpack the dtensor instance for now due to graph context.
      # We will treat the dtensor instance as one global instance and let it
      # return as a global replica instance.
      # TODO(feyu): Support unpack in the graph context.
      super().__init__([dtensor,])
    self._dtensor = dtensor

  def get_dtensor(self):
    return self._dtensor

  @property
  def values(self):
    # Note that this method exists so that it match the interface for PerReplica
    # The public API in `tf.types.experimental.distributed.PerReplica` doesn't
    # define any methods.
    return self._values


def _dtensor_distributed_value_to_tensor(
    var, dtype=None, name=None, as_ref=False):
  del name
  dtensor = var.get_dtensor()
  if dtype is not None and not dtype.is_compatible_with(dtensor.dtype):
    raise ValueError(
        "Incompatible type conversion requested to type {!r} for variable "
        "of type {!r}".format(dtype.name, dtensor.dtype.name))
  if as_ref:
    raise NotImplementedError(
        "PerReplica doesn't support being used as a reference.")
  return dtensor


# Register a conversion function to provide a useful error message when users
# try to use PerReplica values in the wrong contexts
tensor_conversion_registry.register_tensor_conversion_function(
    DTensorDistributedValue, _dtensor_distributed_value_to_tensor)


class DTensorReplicaContext(distribute_lib.ReplicaContext):
  """ReplicaContext for strategy that is backed by DTensor.

  Since the DTensor is operated in the global context, most of the methods from
  existing strategy ReplicaContext is not applicable since they need to access
  local values. For now most of the methods in this class will raise explicit
  error to user, and we will add more support for local values in future.
  """
  _UNSUPPORTED_ERROR_MSG = (
      "Strategy that is backed by DTensor is run with a global context, and "
      "doesn't support operations for local context, like any call to merge/"
      "gather/reduce or local replica ID. Please use any strategy that is not "
      "backed by DTensor")

  def __init__(self, strategy):
    # Since DTensor strategy only runs in a global context, and we can't have
    # a local replica ID in the sync group. For now we pass None to parent, and
    # raise an explicit error when it is accessed.
    super().__init__(strategy, replica_id_in_sync_group=None)

  def __enter__(self):
    # This is a copy of parent class, without any check about whether the
    # current replica is the first one (since DTensor only has one).
    distribute_lib._push_per_thread_mode(self._thread_context)  # # pylint: disable=protected-access

    summary_state = summary_ops_v2._summary_state  # pylint: disable=protected-access
    self._summary_recording_distribution_strategy = (
        summary_state.is_recording_distribution_strategy)
    summary_state.is_recording_distribution_strategy = True

  @property
  def replica_id_in_sync_group(self):
    # Since there is only one global context for DTensor, we always return a
    # constant value here. This value is needed by the RNG which try to generate
    # different seed for different replica.
    return 0

  @property
  def _replica_id(self):
    raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)

  def merge_call(self, merge_fn, args=(), kwargs=None):
    raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)

  def all_reduce(self, reduce_op, value, options=None):
    raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)

  def all_gather(self, value, axis, options=None):
    raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)

  def _update(self, var, fn, args=(), kwargs=None, group=True):
    raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)


def initialize_accelerator_system_once(device_type):
  # Initialize the GPU/TPU before creating the mesh.
  # Note that this method will also trigger the creation of the pairing
  # virtual host CPUs, which is needed by dataset and checkpoint.
  if not accelerator_util.is_initialized():
    # TODO(feyu): Add a method in accelerator_util to check the initialized
    # mesh device types.
    accelerator_util.initialize_accelerator_system(
        device_type,
        experimental_reset_context=True)


def convert_inputs_to_dtensor(inputs, mesh):
  """Convert any input types to DTensor instance."""
  if isinstance(inputs, DTensorDistributedValue):
    return inputs.get_dtensor()
  elif isinstance(inputs, values_lib.DistributedValues):
    return convert_per_replica_to_dtensor(inputs, mesh)
  elif isinstance(inputs, input_util._DTensorIterator):   # pylint: disable=protected-access
    return inputs
  elif tensor_util.is_tensor(inputs):
    if context.executing_eagerly():
      if d_api.is_dtensor(inputs):
        return inputs
      else:
        # For a non-dtensor input in eager context, we could choose to replica
        # them into per-replica and then pack them into dtensor. However, this
        # will cause an eager/graph discrepancy since we can't do this check in
        # the graph context. For now, we will ask user to provide a distributed
        # value for inputs.
        _raise_unsupported_input_type_error(inputs)
    else:
      # For graph context, since we can't check if they are dtensor or not. We
      # will assume the value is already distributed. This is a critical use
      # case for keras, where all the inputs are pre-distributed via strategy,
      # and the train function execute within graph context.
      return inputs
  else:
    # For any other types.
    _raise_unsupported_input_type_error(inputs)


def _raise_unsupported_input_type_error(inputs):
  raise ValueError("Unsupported input types for MirroredStrategy. "
                   "Please use `strategy.distribute_dataset` or "
                   "`strategy.distribute_values_from_function` to "
                   f"distribute inputs. Received input type: {type(inputs)}")


def is_distributed_value(value):
  return isinstance(
      value, values_lib.DistributedValues) or d_api.is_dtensor(value)


def convert_per_replica_to_dtensor(per_replica_value, mesh):
  """Convert a PerReplica result to a DTensor instance.

  Args:
    per_replica_value: A PerReplica instance whose value will be converted
      to DTensor.
    mesh: The mesh used for layout creation.

  Returns:
    A DTensor instance that packed from per_replica_value with batch sharded
      layout.
  """
  values = per_replica_value.values
  if isinstance(values[0], (float, int)):
    rank = 0
  else:
    rank = len(values[0].shape)

  if rank == 0:
    result = []
    # dtensor.pack requires each component to have same rank as the packed
    # result. When the individual value is scalar, it needs to be expanded into
    # 1D tensor.
    for v in values:
      result.append(array_ops.expand_dims_v2(v, axis=0))
    rank += 1
  else:
    result = list(values)   # dtensor.pack requires a list as input.

  # TODO(scottzhu): Note that the result tensor could be a partial value and
  # not always batch shard or fully replicaed. See
  # http://screenshot/6ERkXyX95KqftCw as an example.
  batch_layout = layout.Layout.batch_sharded(
      mesh, batch_dim=DEFAULT_BATCH_MESH_DIM_NAME, rank=rank)

  return d_api.pack(result, batch_layout)


def dtensor_reduce(strategy, reduce_op, value, axis):
  """Implement dtensor based strategy.reduce()."""
  # Due to the limitation of using scalar in DTensor (e.g. the rank 0 tensor
  # loss the batch shard information), we need to override the default
  # reduce in addition to the strategy.extend._reduce_to()
  # Most of the logic here is a mimic of the parent class, except for how
  # mean and sum are calculated in a global context.
  distribute_lib._require_cross_replica_or_default_context_extended(  # pylint: disable=protected-access
      strategy.extended)
  if isinstance(reduce_op, str):
    reduce_op = reduce_util.ReduceOp(reduce_op.upper())

  distributed_input = is_distributed_value(value)
  if not distributed_input and axis is None:
    # For any value that isn't distributed and doesn't need a reduction within
    # the replica.
    destinations = (device_util.current() or
                    strategy.extended._default_device or  # pylint: disable=protected-access
                    "/device:CPU:0")
    devices = cross_device_ops_lib.get_devices_from(destinations)
    with ops.device(devices[0]):
      return array_ops.identity(
          cross_device_ops_lib.reduce_non_distributed_value(
              reduce_op, value, destinations, strategy.num_replicas_in_sync))

  value = convert_inputs_to_dtensor(value, strategy._mesh)  # pylint: disable=protected-access
  # At this point, the value is a DTensor instance now.
  # There will be a final reduction step cross replica. In order to maintain
  # the shape of each local replica, we need to add a new dim to the front.
  # E.g. 2 replica with local shape as (4, 5, 6), the global tensor shape
  # should be (8, 5, 6), we will reshape into (2, 4, 5, 6) and then do a
  # reduction on axis 0.
  if reduce_op == reduce_util.ReduceOp.MEAN:
    reduce_op = math_ops.reduce_mean
  else:
    reduce_op = math_ops.reduce_sum

  # TODO(scottzhu): Make sure we handle dynamic/uneven shape in future.
  if d_api.fetch_layout(value).is_fully_replicated():
    # In case of fully mirrored dtensor, we only need to do one reduce, and
    # don't need to care about any per-replica logic.
    if axis is not None:
      value = reduce_op(value, axis=axis)
  else:
    new_shape = [strategy.num_replicas_in_sync, -1]
    if len(value.shape) > 1:
      new_shape.extend(array_ops.shape(value)[1:])
    value = array_ops.reshape(value, new_shape)
    if axis is not None:
      # we do a reduce_sum/mean within each of the replica when axis is not
      # None. Add 1 to the axis since there is a new dim added by reshape in
      # front.
      value = reduce_op(value, axis=axis + 1)
    value = reduce_op(value, axis=0)

  # Note that we return a DTensor instance here, which should have the same
  # value as the original MirroredStrategy, but with a different type. User
  # might want a tf.Tensor for the status quo.
  return value
