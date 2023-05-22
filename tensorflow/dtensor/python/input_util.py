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
"""APIs to deal with input datasets efficiently in DTensor.

When using tf.data with DTensor, the `DTensorDataset` API can be used to
efficiently handle loading the input data and correctly packing it to the
corresponding devices. This API is intended to work with unbatched data and can
be used for both data and model parallel setups.

Example usage:

>>> # 1-D mesh with 4 devices
>>> mesh = dtensor.Mesh(dim_names=['batch'], ...)
>>> layout = dtensor.Layout.batch_sharded(mesh, 'batch', rank=1)
>>> dataset = tf.data.Dataset.range(256)
>>> d_dataset = dtensor.DTensorDataset(
...     dataset=dataset,
...     global_batch_size=16,
...     mesh=mesh,
...     layouts=layout,
...     batch_dim='batch')
>>> d_iter = iter(d_dataset)
>>> # Each batch is a length 16 tensor sharded across 4 devices
>>> batch_0_dtensor = next(d_iter)
>>> batch_0_dtensor
<tf.Tensor: shape=(16,),
            dtype=int64,
            value={"CPU:0": [ 0  1  2  4],
                   "CPU:1": [ 5  6  7  8],
                   "CPU:2": [ 9 10 11 12],
                   "CPU:3": [13 14 15 16]}>
>>> batch_1_dtensor = next(d_iter)
>>> batch_1_dtensor
<tf.Tensor: shape=(16,),
            dtype=int64,
            value={"CPU:0": [17 18 19 20],
                   "CPU:1": [21 22 23 24],
                   "CPU:2": [25 26 27 28],
                   "CPU:3": [29 30 31 32]}>

For multi-client setups, `DTensorDataset` interacts with tf.data service to
correctly distribute the dataset among the participating clients. DTensor works
with tf.data service in co-located mode where each worker is running alongside
the DTensor client (the Tensorflow Python process). The `TFDataServiceConfig`
dataclass can be filled with information about the tf.data service cluster, and
passed to `DTensorDataset` to enable distribution.
"""

import dataclasses
import operator

from typing import Any, List, Optional, Sequence, Tuple

from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


@dataclasses.dataclass
class TFDataServiceConfig:
  """Specifies the tf.data service configuration to use.

  Attributes:
    dispatcher_address: a string specifying the address of the tf.data service
      dispatcher server.
    job_name: a non-empty string identifying the shared job that will be created
      on tf.data service to process this dataset.
  """
  dispatcher_address: str
  job_name: str


# TODO(b/223275517): Add support for get_next_as_optional().
class _DTensorIterator(iterator_ops.OwnedIterator):
  """An iterator for a tf.data.Dataset distributed using DTensor.

  DTensorIterator encapsulates multiple underlying dataset iterators. It handles
  retrieving the tensors to be placed on each underlying device and then uses
  the 'pack' operation to create and return a DTensor. Thus users need only
  interact with a single DTensorIterator to automatically distribute dataset
  tensors onto devices.
  """

  def __init__(
      self,
      dtensor_components: Tuple[ops.Tensor],
      global_element_spec: tensor_spec.TensorSpec,
      layouts: Any):
    """Initializes a distributed iterator for DTensor datasets.

    This iterator encapsulates tf.data iterators for the underlying devices, and
    treats it as a packed DTensor of iterator resource tensors.

    Args:
      dtensor_components: a tuple containing the underlying iterator resources
        packed into a DTensor. This is expected to be a tuple with a single
        element.
      global_element_spec: the underlying dataset's element spec from a global
        view.
      layouts: a structure of DTensor layouts to be applied to the elements
        returned by the underlying iterators. This can be a single layout or
        (possibly nested) tuples or dictionaries of layouts, and the structure
        must match the structure of the iterator elements.
    """
    # dtensor_components is expected to be a single-element tuple.
    [self._iterator_resource_dtensor] = dtensor_components
    self._global_element_spec = global_element_spec
    self._layouts = layouts
    self._layouts_str = nest.map_structure(
        lambda layout: layout.to_string(), layouts)

    super().__init__(
        components=dtensor_components, element_spec=global_element_spec)

  def __next__(self):
    try:
      # IteratorGetNext will return a DTensor on the host, so move it to the
      # device mesh. If the dataset layouts are on the host mesh itself, this
      # is handled by DTensor as a no-op.
      host_elem = self._next_internal()
      device_elem = nest.map_structure(
          api.copy_to_mesh, host_elem, self._layouts)
      context.async_wait()
      return device_elem
    except errors.OutOfRangeError as e:
      # Match TF2 eager executor behavior by raising StopIteration when iterator
      # is out of range.
      if context.executing_eagerly():
        raise StopIteration from e
      else:
        raise e

  @property
  def _type_spec(self):
    return _DTensorIteratorSpec(self._global_element_spec, self._layouts_str)


class _DTensorIteratorSpec(iterator_ops.IteratorSpec):
  """Type specification for `_DTensorIterator`."""

  __slots__ = ['_global_element_spec', '_layouts_str']

  def __init__(
      self, global_element_spec: tensor_spec.TensorSpec, layouts_str: Any):
    super().__init__(global_element_spec)
    self._global_element_spec = global_element_spec
    self._layouts_str = layouts_str

  @property
  def value_type(self):
    return _DTensorIterator

  def _serialize(self):
    return (self._global_element_spec, self._layouts_str)

  @property
  def _component_specs(self):
    return (tensor_spec.TensorSpec([], dtypes.resource),)

  def _to_components(self, value):
    return (value._iterator_resource_dtensor,)  # pylint: disable=protected-access

  def _from_components(self, components):
    layouts = nest.map_structure(
        layout_lib.Layout.from_string, self._layouts_str)
    return _DTensorIterator(
        dtensor_components=components,
        global_element_spec=self._global_element_spec,
        layouts=layouts)

  @classmethod
  def from_value(cls, value):
    return cls(value._global_element_spec, value._layouts_str)  # pylint: disable=protected-access


def _validate_input(flattened_layouts: Sequence[layout_lib.Layout],
                    flattened_elem_spec: Sequence[tensor_spec.TensorSpec],
                    dataset_already_batched: bool):
  """Checks that the dataset's layouts and element specs are compatible.

  Args:
    flattened_layouts: the flattened list of layouts used to distribute the
      dataset.
    flattened_elem_spec: the flattened list of element specs used in the
      dataset's components.
    dataset_already_batched: whether the dataset to be validated is already
      batched.

  Raises:
    ValueError: if the dataset's inputs are incompatible.
  """
  if not flattened_elem_spec:
    raise ValueError(
        'Expected input element spec of at least one element, was empty.')

  first_elem_shape = flattened_elem_spec[0].shape

  for layout, elem_spec in zip(flattened_layouts, flattened_elem_spec):
    if elem_spec.shape.rank is None:
      raise ValueError(
          'Dataset element shape must have a valid rank, got spec %s.' %
          elem_spec)

    # Check that layout's rank matches the element's rank. If dataset is not yet
    # batched, then the layout's rank must be one greater than the element's
    # rank.
    expected_rank = elem_spec.shape.rank
    if not dataset_already_batched:
      expected_rank += 1
    if layout.rank != expected_rank:
      raise ValueError(
          ('Expected layout with rank %d for element spec %s, got layout %s. '
           'Check that the dataset is not batched before passing to '
           'DTensorDataset.') %
          (expected_rank, elem_spec, layout.sharding_specs))

    if dataset_already_batched:
      # Check that the batch dimension size of all dataset elements match.
      batch_dim_size = first_elem_shape.as_list()[0]
      if batch_dim_size is None:
        raise ValueError(
            ('Size of batch dimension of element spec %s is None. Ensure '
             'drop_remainder=True when batching the dataset.') % elem_spec)

      if elem_spec.shape.as_list()[0] != batch_dim_size:
        raise ValueError(
            ('Size of batch dimension of element spec %s does not match '
             'expected size %d.') % (elem_spec, batch_dim_size))


def _shard_counts(layout: layout_lib.Layout,
                  batch_dim: Optional[str] = None) -> List[int]:
  """Computes a list of the number of shards in each dimension of the layout.

  The shard counts are used to slice each dataset element. The batch dimension's
  count is overridden to 1 since we only consider how many shards to make
  locally (within each local replica). Sharding across clients is handled by
  either tf.data.Dataset's shard transformation (in the single-client case) or
  tf.data service's distribute function (in the multi-client case).

  Args:
    layout: the layout to compute the shard counts for.
    batch_dim: the name of the batch dimension of the layout, if present.

  Returns:
    A list of shard counts, one element per dimension of the layout.
  """
  shard_counts = []
  for spec in layout.sharding_specs:
    if spec in (batch_dim, layout_lib.UNSHARDED):
      shard_counts.append(1)
    else:
      shard_counts.append(layout.mesh.dim_size(spec))
  return shard_counts


def _index_matrix(layout: layout_lib.Layout,
                  elem_spec: tensor_spec.TensorSpec) -> ops.Tensor:
  """Computes a utility matrix to derive device-based slice offsets.

  This function builds a matrix of shape `[mesh.rank, layout.rank]` for each
  dataset element. This matrix can be used to slice the DTensor components
  returned by the iterator according to the local device that component is to be
  placed on. This can be done by multiplying the device offsets of shape
  `[1, mesh.rank]` with this index matrix to get a `[1, layout.rank]` shape
  tensor containing the slice offsets.

  Note: the index on the batch dim is always 0 since sharding on the batch
  dimension is handled by either tf.data.Dataset's shard transformation (in the
  single-client case) or tf.data service's distribute function (in the
  multi-client case). If there is no sharding on the batch dimension (or any
  other dimension), the slice index remains 0.

  Args:
    layout: the layout of the dataset element.
    elem_spec: the spec of the dataset element.

  Returns:
    The index matrix as a tensor.
  """
  matrix = []
  for dim in layout.mesh.dim_names:
    row = [0]
    for layout_idx, spec in enumerate(layout.sharding_specs[1:]):
      if spec == layout_lib.UNSHARDED or spec != dim:
        row.append(0)
      else:
        row.append(elem_spec.shape[layout_idx] // layout.mesh.dim_size(dim))
    matrix.append(row)

  return constant_op.constant(matrix, dtype=dtypes.int32)


def _pack_iterator_resource_dtensor(
    datasets: List[Tuple[int, data_types.DatasetV2]],
    layouts: Any,
    mesh: layout_lib.Mesh,
    num_local_devices_per_replica: int):
  """Creates a DTensor iterator resource for the per-replica datasets.

  Given a list of replica ID to tf.data.Dataset mappings, this function creates
  iterators for each device and then packs the underlying iterator resource
  tensors into a single DTensor. This resource tensor is used by the
  IteratorGetNext op to retrieve the next element in the dataset.

  Args:
    datasets: a list of tuples of each unique local replica ID to the dataset
      object whose elements will be placed on the devices corresponding to that
      replica.
    layouts: a structure of DTensor layouts to be applied to the elements
      returned by the underlying iterators. This can be a single layout or
      (possibly nested) tuples or dictionaries of layouts, and the structure
      must match the structure of the iterator elements.
    mesh: the DTensor mesh to place the iterator batches on.
    num_local_devices_per_replica: the number of devices in each data-parallel
      replica.

  Returns:
    A DTensor of the underlying iterator resource tensors.
  """
  host_mesh_devices = mesh.host_mesh().local_devices()
  device_idx = 0

  iterators = []
  for _, dataset in datasets:
    for idx in range(num_local_devices_per_replica):
      with ops.device_v2(host_mesh_devices[device_idx]):
        device_dataset = dataset.shard(
            num_shards=num_local_devices_per_replica, index=idx)
        iterators.append(iter(device_dataset))
      device_idx += 1

  if device_idx != len(host_mesh_devices):
    raise ValueError(
        'The `datasets` argument does not have the correct number of'
        f' underlying datasets, found {device_idx} but expected'
        f' {len(host_mesh_devices)}.')

  host_layouts = nest.map_structure(
      lambda l: layout_lib.Layout(l.sharding_specs, mesh.host_mesh()), layouts)

  # Pack the iterator resource tensors into a replicated 0-dimensional DTensor
  # and set the element layouts.
  iterator_resources = [it._iterator_resource for it in iterators]  # pylint: disable=protected-access
  d_iterator_resource = api.pack(
      iterator_resources,
      layout_lib.Layout.replicated(mesh=mesh.host_mesh(), rank=0))
  api._dtensor_device().set_iterator_element_layouts(  # pylint: disable=protected-access
      d_iterator_resource, nest.flatten(host_layouts))

  return d_iterator_resource


@tf_export('experimental.dtensor.DTensorDataset', v1=[])
class DTensorDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A dataset of DTensors.

  DTensorDataset encapsulates a `tf.data.Dataset` whose elements are
  automatically packed and returned as DTensors based on a given mesh and
  layouts.
  """

  def __init__(self,
               dataset: data_types.DatasetV2,
               *,
               mesh: layout_lib.Mesh,
               layouts: Any,
               global_batch_size: int,
               dataset_already_batched: bool = False,
               batch_dim: Optional[str] = None,
               prefetch: Optional[int] = None,
               tf_data_service_config: Optional[TFDataServiceConfig] = None):
    """Creates a DTensorDataset.

    DTensorDataset automatically handles distribution of the dataset elements to
    each client's devices. It can be used to create an iterator that returns
    DTensors of the input data on each iteration.

    DTensorDataset works best with unbatched datasets. It takes the mesh and the
    provided layouts to automatically calculate how to batch the input locally
    for each replica.

    If the provided dataset is already batched according to the per-replica
    batch size, then `dataset_already_batched` must be set and DTensorDataset
    will check that the batch size is consistent with the intended
    `global_batch_size` using the layout information. Each replica receives a
    separate slice of the global batch, thus the per-replica batch size can be
    computed as the global batch size divided by the number of model replicas.
    For a DTensor mesh, the number of replicas is equal to the size of the
    mesh's batch dimension.

    TODO(b/223275517): add support for input datasets that are already batched
    to the global batch size.

    Args:
      dataset: a `tf.data.Dataset` object.
      mesh: the DTensor mesh to place the dataset batches on.
      layouts: a structure of DTensor layouts to be applied to the input dataset
        values. This can be a single layout or (possibly nested) tuples or
        dictionaries of layouts, and the structure must match the structure of
        the dataset. Either all or none of the layouts should be sharded on the
        batch dimension; having only a subset of layouts batch sharded will not
        work and raises a ValueError.
      global_batch_size: the desired global batch size.
      dataset_already_batched: must be set only if the dataset is already
        batched to the per-replica batch size. The batched dataset must have
        `drop_remainder=True` set since DTensor requires static shapes for
        slicing the input tensors.
      batch_dim: the mesh dimension on which the input's batch dimension is
        sharded. Set to None if the input layouts do not shard on the batch
        dimension.
      prefetch: number of batches to prefetch using Dataset.prefetch.
      tf_data_service_config: if operating in multi-client mode, this config
        specifies the tf.data service configuration to use.

    Raises:
      ValueError: on any of the following situations,
        1. if the structures and ranks of layouts and the dataset do not match.
        2. if the shapes in the dataset's spec are not fully defined.
        3. if batch_dim is specified and all layouts are not batch-sharded.
        4. if per_replica_batch_size is specified for an already batched Dataset
           but it does not match the expected per-replica size based on the
           provided mesh.
      TypeError: if type of structures of layouts and the dataset do not match.
    """
    super().__init__(dataset, dataset_ops.to_variant(dataset))

    # TODO(b/271162918): fix multi-client use case.
    if tf_data_service_config is not None:
      raise NotImplementedError(
          'Multi-client DTensorDataset is currently not supported.'
          ' Check b/271162918.')

    self._mesh = mesh
    self._layouts = layouts
    self._batch_dim = batch_dim
    self._prefetch = prefetch
    self._tf_data_service_config = tf_data_service_config

    nest.assert_same_structure(dataset.element_spec, layouts)
    flattened_layouts = nest.flatten(layouts)
    flattened_elem_spec = nest.flatten(dataset.element_spec)

    if batch_dim:
      num_global_replicas = mesh.dim_size(batch_dim)
      self._local_replica_ids = list(
          dict.fromkeys(
              [loc[batch_dim] for loc in mesh.local_device_locations()]))

      for layout in flattened_layouts:
        if batch_dim != layout.sharding_specs[0]:
          raise ValueError(
              ('batch_dim %s was specified but at least one layout did not '
               'contain it: %s') % (batch_dim, layout))
    else:
      # Only one replica since there is no sharding on the batch dimension.
      num_global_replicas = 1
      self._local_replica_ids = [0]

    # Validate layout and element spec compatibility, and raise ValueError if
    # invalid.
    _validate_input(
        flattened_layouts,
        flattened_elem_spec,
        dataset_already_batched=dataset_already_batched)

    expected_batch_size = global_batch_size // num_global_replicas
    if not dataset_already_batched:
      self._batched_dataset = dataset.batch(
          expected_batch_size, drop_remainder=True)
    else:
      per_replica_batch_size = flattened_elem_spec[0].shape.as_list()[0]
      if per_replica_batch_size != expected_batch_size:
        raise ValueError(
            ('per_replica_batch_size does not matched expected size based on '
             'the mesh, got %d but expected %d.') %
            (per_replica_batch_size, expected_batch_size))
      self._batched_dataset = dataset

    # Construct a global element spec of the dataset.
    flattened_global_elem_spec = []
    batch_tensor_shape = tensor_shape.as_shape([global_batch_size])
    for elem_spec in nest.flatten(self._batched_dataset.element_spec):
      new_elem_spec = tensor_spec.TensorSpec(
          shape=operator.concat(batch_tensor_shape, elem_spec.shape[1:]),
          dtype=elem_spec.dtype,
          name=elem_spec.name)
      flattened_global_elem_spec.append(new_elem_spec)
    self._global_element_spec = nest.pack_sequence_as(
        dataset.element_spec, flattened_global_elem_spec)

    num_global_devices_per_replica = config.num_global_devices(
        mesh.device_type()) // num_global_replicas
    self._num_local_replicas = len(self._local_replica_ids)
    self._num_local_devices_per_replica = mesh.num_local_devices(
    ) // self._num_local_replicas
    # The number of clients each replica is split over.
    self._num_clients_per_replica = (
        num_global_devices_per_replica // self._num_local_devices_per_replica)
    # In the case where a replica is split across multiple clients, an offset
    # needs to be added to the index used by the partitioning logic such that
    # the local devices on that client can be correctly matched to slices of the
    # input tensor(s). If replicas are wholly contained within a client, then
    # this offset is always 0.
    self._partition_offset = (config.client_id() % self._num_clients_per_replica
                             ) * self._num_local_devices_per_replica

    # Helper data structures used in partitioning the dataset tensors.
    self._all_shard_counts = [
        _shard_counts(layout, batch_dim) for layout in flattened_layouts
    ]
    self._index_matrices = [
        _index_matrix(layout, elem_spec)
        for layout, elem_spec in zip(flattened_layouts, flattened_elem_spec)
    ]

  def __iter__(self):
    datasets: List[Tuple[int, data_types.DatasetV2]] = []

    # Start with the batched the dataset.
    local_dataset = self._batched_dataset

    if self._batch_dim is not None:
      if self._num_clients_per_replica > 1:
        # If a replica is split over multiple clients then each batch needs to
        # be repeated before distribution as many times as there are clients
        # corresponding to that replica.
        local_dataset = self._repeat_batch(local_dataset,
                                           self._num_clients_per_replica)
        sharding_policy = data_service_ops.ShardingPolicy.DATA
      else:
        # Replicas are unique to each client, so FILE based sharding can be used
        # which is more performant since each worker does not need to read the
        # entire dataset.
        sharding_policy = data_service_ops.ShardingPolicy.FILE
    else:
      # No batch dimension sharding specified so disable dataset sharding during
      # the distribute step.
      sharding_policy = data_service_ops.ShardingPolicy.OFF

    # Apply distribution here (if specified) so all remaining transformations
    # are executed locally.
    if self._tf_data_service_config is not None:
      local_dataset = local_dataset.apply(
          data_service_ops.distribute(
              processing_mode=sharding_policy,
              service=self._tf_data_service_config.dispatcher_address,
              job_name=f'{self._tf_data_service_config.job_name}_{config.client_id()}',
              target_workers='LOCAL'))

    for local_replica_idx, replica_id in enumerate(self._local_replica_ids):
      # Select the shard for the corresponding replica.
      dataset = local_dataset.shard(self._num_local_replicas, local_replica_idx)

      # Repeat each batch for each local device in the replica.
      dataset = self._repeat_batch(dataset, self._num_local_devices_per_replica)

      # Slice each shard further for all non-batch dim shards. If there is no
      # non-batch dim sharding, this slice is essentially a no-op.
      dataset = self._partition(dataset)

      # Apply prefetch as the last step. Since each batch is repeated, the
      # number of elements to prefetch has to be scaled by the same size.
      if self._prefetch is not None:
        dataset = dataset.prefetch(
            self._prefetch * self._num_local_devices_per_replica)

      datasets.append((replica_id, dataset))

    # Convert the datasets into iterators placed on the host.
    d_iterator_resource = _pack_iterator_resource_dtensor(
        datasets=datasets,
        layouts=self._layouts,
        mesh=self._mesh,
        num_local_devices_per_replica=self._num_local_devices_per_replica)

    return _DTensorIterator(
        dtensor_components=(d_iterator_resource,),
        global_element_spec=self._global_element_spec,
        layouts=self._layouts)

  def _repeat_batch(self, dataset, repeats):
    def repeat(*x):
      return dataset_ops.DatasetV2.from_tensors(x).repeat(repeats)

    return dataset.flat_map(repeat)

  def _partition(self, dataset):
    """Slices each dataset element on any sharded non-batch dimension."""

    # TODO(b/223275517): decouple from self and make testable.
    def slice_batch(index, batch):
      flattened_batch = nest.flatten(batch)
      flattened_output = []

      norm_index = math_ops.cast(
          index % self._num_local_devices_per_replica, dtype=dtypes.int32)
      norm_index += self._partition_offset
      coords = self._mesh.coords(norm_index)
      coords = array_ops.reshape(coords, (1, -1))

      for element, shard_counts, idx_matrix in zip(flattened_batch,
                                                   self._all_shard_counts,
                                                   self._index_matrices):
        indexes = math_ops.matmul(coords, idx_matrix)
        start = array_ops.reshape(indexes, (-1,))
        size = array_ops.shape_v2(
            element, out_type=dtypes.int32) // shard_counts
        flattened_output.append(
            array_ops.slice(element, begin=start, size=size))

      return nest.pack_sequence_as(batch, flattened_output)

    enumerated_dataset = dataset.enumerate()
    partitioned_dataset = enumerated_dataset.map(slice_batch)
    return partitioned_dataset

  @property
  def element_spec(self):
    return self._global_element_spec
