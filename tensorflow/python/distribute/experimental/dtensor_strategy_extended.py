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
"""Implement a StrategyExtended based on the DTensor low level API."""

import functools

from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import config as d_config
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import input_util
from tensorflow.dtensor.python import layout
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute.experimental import dtensor_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest


class DTensorStrategyExtended(distribute_lib.StrategyExtendedV2):
  """Strategy extension that support both single and multi worker strategy."""
  # Note that the unit test for this class is via the strategy interface.

  def __init__(self, container_strategy, mesh):
    super().__init__(container_strategy)
    self._mesh = mesh
    self._num_clients = d_config.num_clients()
    self._client_id = d_config.client_id()

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

    # Ignore expected_shape, which is from the v1 Variable. Keras was somehow
    # using the v1 Variable, but didn't specify that value particularly.
    kwargs.pop('expected_shape', None)

    # Make sure to call DVariable initializer under the scope so that it will
    # have the proper replicated layout. The initial_value is multi-typed,
    # eg it can be a tensor, or a python/numpy type, or a callable that
    # produce tensor/python/numpy types. In all those cases, we need to wrap
    # them invoke convert_to_tensor() under the scope so that the proper
    # layout can be assigned.

    # TODO(scottzhu): The layout information should be injected via kwargs, or
    # lazily set later.
    initial_value = kwargs.pop('initial_value')
    dtype = kwargs.get('dtype', None)
    def new_initial_value():
      if callable(initial_value):
        init_var = ops.convert_to_tensor(initial_value(), dtype=dtype)
      else:
        init_var = ops.convert_to_tensor(initial_value, dtype=dtype)
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
    # Note that in either single worker (MirroredStrategy) or multi worker (
    # MultiWorkerMirroredStrategy), worker_devices refers to the local worker
    # devices.
    return tuple(self._mesh.local_devices())

  @property
  def parameter_devices(self):
    # Same as the worker_devices.
    return self.worker_devices

  def _in_multi_worker_mode(self):
    return d_config.num_clients() > 1

  def _get_local_replica_id(self, replica_id_in_sync_group):
    return replica_id_in_sync_group

  def _default_device_scope(self):
    return d_api.default_mesh(self._mesh)

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
          self._mesh, batch_dim=dtensor_util.DEFAULT_BATCH_MESH_DIM_NAME,
          rank=rank)

    layouts = nest.map_structure(_create_batch_layout, dataset.element_spec)

    return input_util.DTensorDataset(
        dataset=dataset,
        mesh=self._mesh,
        layouts=layouts,
        global_batch_size=batch_size,
        dataset_already_batched=False,
        batch_dim=dtensor_util.DEFAULT_BATCH_MESH_DIM_NAME,
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
    input_context = distribute_lib.InputContext(
        num_input_pipelines=self._num_clients,
        input_pipeline_id=self._client_id,
        num_replicas_in_sync=self._num_replicas_in_sync
    )
    dataset = dataset_fn(input_context)

    # Note that the dataset should already batched to local per-relica batch
    def _create_batch_layout(tensor_spec):
      rank = len(tensor_spec.shape)
      return layout.Layout.batch_sharded(
          self._mesh, batch_dim=dtensor_util.DEFAULT_BATCH_MESH_DIM_NAME,
          rank=rank)

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
        batch_dim=dtensor_util.DEFAULT_BATCH_MESH_DIM_NAME,
        # TODO(scottzhu): Add prefetch support by inspecting the input dataset.
        prefetch=None,
        tf_data_service_config=None
    )

  def _experimental_distribute_values_from_function(self, value_fn):
    per_replica_values = []
    # Note that in the multi-worker setting, this function only return the
    # slide of DistributedValue for the current worker.
    for i in range(self._mesh.num_local_devices()):
      # In the case of 2 worker with 2 local devices on each worker,
      # worker 0 will get 0 and 1 for replica_id.
      # worker 1 will get 2 and 3 for replica_id.
      replica_id = d_config.client_id() * self._mesh.num_local_devices() + i
      per_replica_values.append(value_fn(
          distribute_lib.ValueContext(replica_id,
                                      self._num_replicas_in_sync)))
    # Instead of using the DistributeVariable, return a DTensor instead since
    # the run() will expect a DTensor instance.
    result = distribute_utils.regroup(per_replica_values, always_wrap=True)
    map_fn = functools.partial(dtensor_util.convert_per_replica_to_dtensor,
                               mesh=self._mesh)
    return nest.map_structure(map_fn, result)

  def call_for_each_replica(self, fn, args=(), kwargs=None):
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
    map_fn = functools.partial(dtensor_util.convert_inputs_to_dtensor,
                               mesh=self._mesh)
    d_args = nest.map_structure(map_fn, args)
    d_kwargs = nest.map_structure(map_fn, kwargs)

    with self._container_strategy().scope():
      with dtensor_util.DTensorReplicaContext(self._container_strategy()):
        dtensor_result = fn(*d_args, **d_kwargs)

    return nest.map_structure(
        dtensor_util.DTensorDistributedValue,
        dtensor_result)

  def _gather_to_implementation(self, value, destinations, axis, options):
    if isinstance(value, dtensor_util.DTensorDistributedValue):
      value = value.get_dtensor()
    if not d_api.is_dtensor(value):
      # This is the current behavior for mirrored strategy, should we raise an
      # error for unsupported types?
      return value

    # Unpack the dtensor components and gather the tensors on the axis
    components = d_api.unpack(value)
    return array_ops.concat(components, axis=axis)

  def _use_merge_call(self):
    # This is method for V1 StrategyExtended by still used by
    # tf.__internal__.distribute.strategy_supports_no_merge_call
    return False
