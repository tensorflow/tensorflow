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
import functools

from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import config as d_config
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import input_util
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python import mesh_util
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.distribute.experimental import dtensor_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest

# Default dimension name used for the mesh created when user provide a list
# of devices. For mirrored strategy, it should be a 1D mesh with batch dim only.
_DEFAULT_BATCH_MESH_DIM_NAME = 'batch'


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

    extended = MirroredExtended(container_strategy=self, mesh=mesh)
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
      mesh = mesh_util.create_mesh(
          mesh_dims=[(_DEFAULT_BATCH_MESH_DIM_NAME, len(devices))],
          devices=devices)
    else:
      # Trying to detect if there is any GPU/TPUs attached.
      device_type = d_config.preferred_device_type()
      devices = d_config.local_devices(device_type)
      mesh = mesh_util.create_mesh(
          mesh_dims=[(_DEFAULT_BATCH_MESH_DIM_NAME, len(devices))],
          device_type=device_type)
    return mesh


class MirroredExtended(distribute_lib.StrategyExtendedV2):
  """Strategy extension contains the concrete logic for variable creation."""

  def __init__(self, container_strategy, mesh):
    super().__init__(container_strategy)
    self._mesh = mesh

  def _create_variable(self, next_creator, **kwargs):
    # Make sure the pop the `use_resource` which is not supported by the
    # base tf.Variable. The `use_resource` is added by
    # creator_with_resource_vars in distribute_lib.py
    kwargs.pop('use_resource', None)

    # Ignore the colocate_with for the mirrored strategy. Each of the device
    # will get same copy of variable in the DTensor's case.
    # `colocate_with` is added when user call:
    # strategy.extended.colocate_vars_with(variable)
    kwargs.pop('colocate_with', None)

    # Make sure to call DVariable initializer under the scope so that it will
    # have the proper replicated layout. The initial_value is multi-typed,
    # eg it can be a tensor, or a python/numpy type, or a callable that
    # produce tensor/python/numpy types. In all those cases, we need to wrap
    # them invoke convert_to_tensor() under the scope so that the proper
    # layout can be assigned.

    # TODO(scottzhu): The layout information should be injected via kwargs, or
    # lazily set later.
    initial_value = kwargs.pop('initial_value')
    def new_initial_value():
      if callable(initial_value):
        init_var = ops.convert_to_tensor(initial_value())
      else:
        init_var = ops.convert_to_tensor(initial_value)
      rank = init_var.shape.rank
      return d_api.copy_to_mesh(
          init_var, layout.Layout.replicated(self._mesh, rank))

    return d_variable.DVariable(new_initial_value, **kwargs)

  @property
  def _num_replicas_in_sync(self):
    # The mesh should be 1D with batch sharding only.
    # In the model parallel case, it should only return the size of
    # batch dimension.
    return self._mesh.size

  def value_container(self, value):
    return value

  @property
  def worker_devices(self):
    # Note that we return the local device here since this is a single worker
    # setting, and the local devices will be all the devices in the current
    # mesh. In the multi-worker mirrored strategy, this value should be
    # expanded to the global device list.
    return tuple(self._mesh.local_devices())

  @property
  def parameter_devices(self):
    # Same as the worker_devices.
    return self.worker_devices

  def _in_multi_worker_mode(self):
    # This method is mostly used in the input relate context and high level API.
    # In the single client mesh DTensor context, this is False.
    return False

  def _experimental_distribute_dataset(self, dataset, options):
    # Strategy always assume the user input data is a batched dataset for
    # experimental_distribute_dataset().
    # TODO(yuefengz): Add check for whether a dataset is batched for all
    # strategies.

    # TODO(b/265198795): Support dataset already batched to global batch size.
    # Since DTensorDataset doesn't support batched dataset that is already
    # batched global batch size, it only supports dataset that is batched to
    # local batch size, we need to infer the batch size, and unbatch the dataset
    # until the b/265198795 is resolved.
    batch_size = distribute.compute_batch_size(dataset)

    # There are multiple case that the batch is not static, eg partial batch,
    # or uneven batch, in all those case, it will return -1.
    if batch_size.numpy() < 0:
      # When we don't have a static batch size.
      raise ValueError('DTensor strategy requires a static batch size for now.'
                       'The dynamic batch size will be supported in future')
    # Unbatch the dataset for now since the DTensorDataset has some limitation
    # about the local batch size as well as the mesh size.
    dataset = dataset.unbatch()

    def _create_batch_layout(tensor_spec):
      # For unbatched dataset, the new layout need to have +1 rank for
      # the batched result.
      rank = len(tensor_spec.shape) + 1
      return layout.Layout.batch_sharded(
          self._mesh, batch_dim=_DEFAULT_BATCH_MESH_DIM_NAME, rank=rank)

    layouts = nest.map_structure(_create_batch_layout, dataset.element_spec)

    return input_util.DTensorDataset(
        dataset=dataset,
        mesh=self._mesh,
        layouts=layouts,
        global_batch_size=batch_size,
        dataset_already_batched=False,
        batch_dim=_DEFAULT_BATCH_MESH_DIM_NAME,
        # TODO(scottzhu): Add prefetch support by inspecting the input dataset.
        prefetch=None,
        tf_data_service_config=None
    )

  def _make_dataset_iterator(self, dataset):
    raise NotImplementedError(
        'Strategy.make_dataset_iterator() is deprecated, and only available '
        'in the V1 API.')

  def _make_input_fn_iterator(self, input_fn, replication_mode):
    raise NotImplementedError(
        'Strategy.make_input_fn_iterator() is deprecated, and only available '
        'in the V1 API.')

  def _distribute_datasets_from_function(self, dataset_fn, options):
    # TODO(scottzhu): Implement the logic for options in future
    del options
    # Single worker for now, this will change when deal with different input
    # options or multiple workers.
    input_context = distribute_lib.InputContext(
        num_input_pipelines=1,
        input_pipeline_id=0,
        num_replicas_in_sync=self._num_replicas_in_sync
    )
    dataset = dataset_fn(input_context)

    # Note that the dataset should already batched to local per-relica batch
    def _create_batch_layout(tensor_spec):
      rank = len(tensor_spec.shape)
      return layout.Layout.batch_sharded(
          self._mesh, batch_dim=_DEFAULT_BATCH_MESH_DIM_NAME, rank=rank)

    layouts = nest.map_structure(_create_batch_layout, dataset.element_spec)

    batch_size = distribute.compute_batch_size(dataset)
    # There are multiple case that the batch is not static, eg partial batch,
    # or uneven batch, in all those case, it will return -1.
    if batch_size.numpy() < 0:
      # When we don't have a static batch size.
      raise ValueError('DTensor strategy requires a static batch size for now.'
                       'The dynamic batch size will be supported in future')
    global_batch_size = batch_size.numpy() * self._num_replicas_in_sync

    return input_util.DTensorDataset(
        dataset=dataset,
        mesh=self._mesh,
        layouts=layouts,
        global_batch_size=global_batch_size,
        dataset_already_batched=True,
        batch_dim=_DEFAULT_BATCH_MESH_DIM_NAME,
        # TODO(scottzhu): Add prefetch support by inspecting the input dataset.
        prefetch=None,
        tf_data_service_config=None
    )

  def _experimental_distribute_values_from_function(self, value_fn):
    per_replica_values = []
    for replica_id in range(self._num_replicas_in_sync):
      per_replica_values.append(value_fn(
          distribute_lib.ValueContext(replica_id,
                                      self._num_replicas_in_sync)))
    # Instead of using the DistributeVariable, return a DTensor instead since
    # the run() will expect a DTensor instance.
    result = distribute_utils.regroup(per_replica_values, always_wrap=True)
    map_fn = functools.partial(_convert_per_replica_to_dtensor, mesh=self._mesh)
    return nest.map_structure(map_fn, result)

  def call_for_each_replica(self, fn, args, kwargs):
    """Run `fn` once per replica.

    This is a method that expected by the strategy base class in its `run()`.

    Args:
      fn: function to run (will be run once per replica).
      args: Tuple or list with positional arguments for `fn`.
      kwargs: Dict with keyword arguments for `fn`.

    Returns:
      Merged return value of `fn` across all replicas.
    """
    # Comparing to the existing MirroredStrategy, which will run the fn on
    # each of the replica with individual thread, the DTensor will just run
    # the fn once with the DTensor inputs, and the distribution will be handled
    # by the DTensor.

    distribute_lib._require_cross_replica_or_default_context_extended(self)   # pylint: disable=protected-access
    if kwargs is None:
      kwargs = {}

    # For any value that is not DTensor, eg normal tf.Tensor or
    # DistributedValues, we need to convert them into DTensor.
    map_fn = functools.partial(_convert_inputs_to_dtensor, mesh=self._mesh)
    d_args = nest.map_structure(map_fn, args)
    d_kwargs = nest.map_structure(map_fn, kwargs)

    with self._container_strategy().scope():
      # TODO(scottzhu): Add support for get_replica_context() within the fn.
      dtensor_result = fn(*d_args, **d_kwargs)

    return nest.map_structure(
        dtensor_util.DTensorDistributedValue,
        dtensor_result)


def _convert_inputs_to_dtensor(inputs, mesh):
  """Convert any input types to DTensor instance."""
  if d_api.is_dtensor(inputs):
    return inputs
  elif isinstance(inputs, dtensor_util.DTensorDistributedValue):
    return inputs.get_dtensor()
  elif isinstance(inputs, values_lib.DistributedValues):
    return _convert_per_replica_to_dtensor(inputs, mesh)
  else:
    # For the rest of the types, we will convert it to dtensor.
    # Any of the inputs will be replicate to all the devices.
    rank = len(inputs.shape)
    replicated_layout = layout.Layout.replicated(mesh, rank=rank)
    return d_api.copy_to_mesh(inputs, replicated_layout)


def _convert_per_replica_to_dtensor(per_replica_value, mesh):
  """Convert a PreReplica result to a DTensor instance.

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
      mesh, batch_dim=_DEFAULT_BATCH_MESH_DIM_NAME, rank=rank)

  return d_api.pack(result, batch_layout)
