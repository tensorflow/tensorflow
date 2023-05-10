# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Implement a MirroredStrategy based on the DTensor low level API.

This is an experiment to validate the viability of the DTensor API, and expose
any potential feature gaps between the current API and the need.
"""

from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import config as d_config
from tensorflow.dtensor.python import mesh_util
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute.experimental import dtensor_strategy_extended
from tensorflow.python.distribute.experimental import dtensor_util
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


class MirroredStrategy(distribute_lib.Strategy):
  """Synchronous training across multiple replicas on one machine.

  This strategy is typically used for training on one machine with multiple
  accelerators (GPUs/TPUs).

  For example, a variable created under a `MirroredStrategy` is a distributed
  variable with layout replicated on each dimension. The variables will be
  placed on the `mesh` that is specified in the __init__.
  """

  def __init__(self, mesh=None, devices=None, cross_device_ops=None):
    """Synchronous training across multiple replicas on one machine.

    Args:
      mesh: optional DTensor mesh for the computation. Note that either `mesh`
        or `devices` should be provided, and not both. The mesh should be 1D,
        and will be used to split the input data among that dimension.
      devices: a list of device strings, such as ['/gpu:0', '/gpu:1']. If both
        `mesh` and `devices` are None, all the available GPU/TPU will be used.
        If no accelerators are found, CPU is used.
      cross_device_ops: optional, a descendant of `CrossDeviceOps`. The value is
        ignored at the moment, and support will be added later.
    """
    self._validate_init_args(mesh, devices)
    if not mesh:
      mesh = self._build_mesh_from_device_list(devices)

    extended = dtensor_strategy_extended.DTensorStrategyExtended(
        container_strategy=self, mesh=mesh)
    super().__init__(extended)
    self._mesh = mesh
    self._devices = devices

  @classmethod
  def _validate_init_args(cls, mesh, devices):
    if mesh and devices:
      raise ValueError('Mesh and devices can not be provided at the same time. '
                       f'received mesh = {mesh}, devices = {devices}')

    # For mirrored strategy, the mesh should be 1D, and only contains a batch
    # dimension, we will use that dimension to shard the inputs.
    if mesh and len(mesh.shape()) != 1:
      raise ValueError('The mesh for MirroredStrategy must be 1D, received: '
                       f'{len(mesh.shape())}D')

  @classmethod
  def _build_mesh_from_device_list(cls, devices):
    if devices:
      device_type = tf_device.DeviceSpec.from_string(devices[0]).device_type
      dtensor_util.initialize_accelerator_system_once(device_type)
      mesh = mesh_util.create_mesh(
          mesh_dims=[(dtensor_util.DEFAULT_BATCH_MESH_DIM_NAME, len(devices))],
          devices=devices)
    else:
      # Trying to detect if there is any GPU/TPUs attached.
      device_type = d_config.preferred_device_type()
      devices = d_config.local_devices(device_type)
      dtensor_util.initialize_accelerator_system_once(device_type)
      mesh = mesh_util.create_mesh(
          mesh_dims=[(dtensor_util.DEFAULT_BATCH_MESH_DIM_NAME, len(devices))],
          device_type=device_type)
    return mesh

  def reduce(self, reduce_op, value, axis):
    # Due to the limitation of using scalar in DTensor (e.g. the rank 0 tensor
    # loss the batch shard information), we need to override the default
    # reduce in addition to the strategy.extend._reduce_to()
    # Most of the logic here is a mimic of the parent class, except for how
    # mean and sum are calculated in a global context.
    distribute_lib._require_cross_replica_or_default_context_extended(  # pylint: disable=protected-access
        self.extended)
    if isinstance(reduce_op, str):
      reduce_op = reduce_util.ReduceOp(reduce_op.upper())

    distributed_input = dtensor_util.is_distributed_value(value)
    if not distributed_input and axis is None:
      # For any value that isn't distributed and doesn't need a reduction within
      # the replica.
      destinations = (device_util.current() or
                      self.extended._default_device or  # pylint: disable=protected-access
                      '/device:CPU:0')
      devices = cross_device_ops_lib.get_devices_from(destinations)
      with ops.device(devices[0]):
        return array_ops.identity(
            cross_device_ops_lib.reduce_non_distributed_value(
                reduce_op, value, destinations, self.num_replicas_in_sync))

    value = dtensor_util.convert_inputs_to_dtensor(value, self._mesh)
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
      new_shape = [self.num_replicas_in_sync, -1]
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
