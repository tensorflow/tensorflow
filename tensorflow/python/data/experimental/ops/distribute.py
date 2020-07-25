# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Distribution Strategy-related dataset transformations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.experimental.ops.distribute_options import ExternalStatePolicy
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops


class _AutoShardDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that shards the `Dataset` automatically.

  This dataset takes in an existing dataset and tries to automatically figure
  out how to shard the dataset in a multi-worker scenario. Currently, it uses
  Grappler to walk up the dataset graph until it finds a reader dataset (e.g.
  CSVDataset, TFRecordDataset), then inserts a ShardDataset op before that node
  so that each worker only sees some files.

  Args:
    num_workers: Total number of workers to shard this dataset across.
    index: The current worker index (out of the total number of workers) this
      dataset is for.

  Raises:
    NotFoundError: If we cannot find a suitable reader dataset to begin
      automatically sharding the dataset.
  """

  def __init__(self, input_dataset, num_workers, index):
    self._input_dataset = input_dataset

    self._element_spec = input_dataset.element_spec
    variant_tensor = ged_ops.auto_shard_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        num_workers=num_workers,
        index=index,
        auto_shard_policy=int(
            input_dataset.options().experimental_distribute.auto_shard_policy),
        **self._flat_structure)
    super(_AutoShardDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._element_spec


def _AutoShardDatasetV1(input_dataset, num_workers, index):  # pylint: disable=invalid-name
  return dataset_ops.DatasetV1Adapter(
      _AutoShardDataset(input_dataset, num_workers, index))


class _RebatchDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that rebatches elements from its input into new batch sizes.

  `_RebatchDataset(input_dataset, batch_sizes)` is functionally equivalent to
  `input_dataset.unbatch().batch(N)`, where the value of N cycles through the
  `batch_sizes` input list. The elements produced by this dataset have the same
  rank as the elements of the input dataset.

  For example:

  ```python
  ds = tf.data.Dataset.range(8)
  ds = ds.batch(4)
  ds = _RebatchDataset(ds, batch_sizes=[2, 1, 1])
  for elem in ds:
    print(elem)
  >> [0, 1], [2], [3], [4, 5], [6], [7]

  ds = tf.data.Dataset.range(16)
  ds = ds.batch(4)
  ds = _RebatchDataset(ds, batch_sizes=[6])
  for elem in ds:
    print(elem)
  >> [0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15]
  ```
  """

  def __init__(self, input_dataset, batch_sizes, drop_remainder=False):
    """Creates a _RebatchDataset.

    Args:
      input_dataset: `Dataset` to rebatch.
      batch_sizes: A `tf.int64` scalar or vector, representing the size of
        batches to produce. If this argument is a vector, these values are
        cycled through in order.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in the case it has fewer than
        `batch_sizes[cycle_index] elements; the default behavior is not to drop
        the smaller batch.
    """
    self._input_dataset = input_dataset
    self._batch_sizes = ops.convert_to_tensor(
        batch_sizes, dtype=dtypes.int64, name="batch_sizes")
    self._drop_remainder = ops.convert_to_tensor(
        drop_remainder, dtype=dtypes.bool, name="drop_remainder")
    new_batch_dim = self._compute_static_batch_dim()

    # pylint: disable=protected-access
    self._element_spec = nest.map_structure(
        lambda ts: ts._unbatch()._batch(new_batch_dim),
        dataset_ops.get_structure(input_dataset))
    # pylint: enable=protected-access

    input_dataset = dataset_ops.normalize_to_dense(input_dataset)
    variant_tensor = ged_ops.rebatch_dataset_v2(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        batch_sizes=batch_sizes,
        drop_remainder=drop_remainder,
        **self._flat_structure)
    super(_RebatchDataset, self).__init__(input_dataset, variant_tensor)

  def _compute_static_batch_dim(self):
    """Computes the static batch dimension of a dataset if it can be determined.

    Given the _RebatchDataset parameters, determines the batch dimension of this
    dataset statically. Returns None if this cannot be determined or is
    variable.

    Returns:
      An integer representing the batch dimension of the dataset. If it cannot
      be determined statically, returns None.

    Raises:
      ValueError: The batch_sizes parameter is malformed, input_dataset is
      not batched, or input_dataset batch sizes are incompatible with each
      other.
    """
    new_batch_dim = tensor_util.constant_value(self._batch_sizes)
    if new_batch_dim is None:
      return None

    if isinstance(new_batch_dim, np.ndarray):
      if len(new_batch_dim.shape) == 1:
        if np.all(new_batch_dim == new_batch_dim[0]):
          new_batch_dim = new_batch_dim[0]
        else:
          return None
      elif len(new_batch_dim.shape) > 1:
        raise ValueError("Expected batch_sizes to be a scalar or vector.")

    if self._may_form_partial_batches(new_batch_dim):
      return None

    return new_batch_dim

  def _may_form_partial_batches(self, desired_batch_size):
    """Returns whether this dataset may form partial batches."""
    if tensor_util.constant_value(self._drop_remainder):
      return False

    def get_batch_dim(type_spec):
      shape = type_spec._to_legacy_output_shapes()  # pylint: disable=protected-access
      if not isinstance(shape, tensor_shape.TensorShape):
        return None
      if shape.rank is None:
        return None
      if len(shape) < 1:
        raise ValueError("Expected a dataset whose elements have rank >= 1 "
                         "but found a dataset whose elements are scalars. "
                         "You can fix the issue by adding the `batch` "
                         "transformation to the dataset.")
      return shape.dims[0].value

    input_batch_dims = [
        get_batch_dim(ts)
        for ts in nest.flatten(dataset_ops.get_structure(self._input_dataset))
    ]
    known_input_batch_dims = [d for d in input_batch_dims if d is not None]

    if not known_input_batch_dims:
      return True

    known_input_batch_dims = np.asarray(known_input_batch_dims)
    if not np.all(known_input_batch_dims == known_input_batch_dims[0]):
      raise ValueError(
          "Batch dimensions of input dataset are not compatible.")

    return known_input_batch_dims[0] % desired_batch_size != 0

  @property
  def element_spec(self):
    return self._element_spec


class _LegacyRebatchDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that divides its input batches into `num_replicas` sub-batches.

  For each batch in the input dataset, _LegacyRebatchDataset will produce
  `num_replicas` smaller batches whose sizes add up to the original batch size.

  For example:

  ```python
  ds = tf.data.Dataset.range(8)
  ds = ds.batch(4)
  ds = _LegacyRebatchDataset(ds, num_replicas=3)
  for elem in ds:
    print(elem)
  >> [0, 1], [2, 3], [], [4, 5], [6, 7], []
  ```
  """

  def __init__(self, input_dataset, num_replicas):
    """Creates a _LegacyRebatchDataset.

    Args:
      input_dataset: `Dataset` to rebatch.
      num_replicas: A `tf.int64` scalar, representing the number of sub-batches
        to split each batch from `input_dataset` into.
    """

    def recalculate_batch_size(type_spec):
      """Recalculates the output_shape after dividing it by num_replicas."""
      output_shape = type_spec._to_legacy_output_shapes()  # pylint: disable=protected-access
      if not isinstance(output_shape, tensor_shape.TensorShape):
        return None

      # If the output shape is unknown, we set the batch dimension to unknown.
      if output_shape.rank is None:
        return None

      if len(output_shape) < 1:
        raise ValueError("Expected a dataset whose elements have rank >= 1 "
                         "but found a dataset whose elements are scalars. "
                         "You can fix the issue by adding the `batch` "
                         "transformation to the dataset.")
      output_dims = [d.value for d in output_shape.dims]

      if output_dims[0] is not None and output_dims[0] % num_replicas == 0:
        return output_dims[0] // num_replicas

      # Set the batch dimension to unknown. If the global batch size does not
      # divide num_replicas evenly, the minibatches may have different sizes.
      return None

    def rebatch(type_spec):
      # pylint: disable=protected-access
      batch_size = recalculate_batch_size(type_spec)
      return type_spec._unbatch()._batch(batch_size)
      # pylint: enable=protected-access

    self._element_spec = nest.map_structure(
        rebatch, dataset_ops.get_structure(input_dataset))
    input_dataset = dataset_ops.normalize_to_dense(input_dataset)
    variant_tensor = ged_ops.rebatch_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        num_replicas=num_replicas,
        **self._flat_structure)
    super(_LegacyRebatchDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._element_spec


class _RemoteDataset(dataset_ops.DatasetSource):
  """Creates a dataset on a given `device` given a graph def."""

  def __init__(self, graph_def, device, element_spec):
    self._elem_spec = element_spec
    with ops.device(device):
      variant_tensor = ged_ops.dataset_from_graph(graph_def)
    super(_RemoteDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._elem_spec


def replicate(dataset, devices):
  """A transformation that replicates `dataset` onto a list of devices.

  Args:
    dataset: A `tf.data.Dataset` object.
    devices: A list of devices to replicate the dataset on.

  Returns:
    A dictionary mapping device name to a dataset on that device.
  """
  if not isinstance(dataset, dataset_ops.DatasetV2):
    raise TypeError("`dataset` must be a `tf.data.Dataset` object.")

  # pylint: disable=protected-access
  dataset_device = dataset._variant_tensor.device

  datasets = {}
  if len(devices) == 1 and devices[0] == dataset_device:
    datasets[devices[0]] = dataset
    return datasets

  with ops.colocate_with(dataset._variant_tensor):
    dataset = dataset._apply_options()
    policy = dataset.options().experimental_external_state_policy
    if policy is None:
      policy = ExternalStatePolicy.WARN
    graph_def = dataset._as_serialized_graph(
        strip_device_assignment=True,
        external_state_policy=policy)
  for device in devices:
    ds = _RemoteDataset(graph_def, device, dataset.element_spec)
    datasets[device] = ds
  return datasets


def compute_batch_size(dataset):
  """An operation that returns the batch size of the dataset.

  This op tries to infer the batch size statically by walking up the dataset
  tree from the final dataset node and returning the batch size of the first
  batching dataset (such as from .batch() and .padded_batch()) that it
  encounters. This differs from using the `element_spec` of a dataset in that it
  does not account for partial batches.

  This operation may fail if it encounters contradictory batch sizes (for
  example, if the dataset is created by zipping together two datasets with
  different batch sizes), if there are no explicit batching transformations, or
  if there are operations downstream from the batching transformation that may
  modify its batch size. In these cases, it returns a -1.

  Args:
    dataset: A `tf.data.Dataset` object.

  Returns:
    A `tf.int64` Tensor representing the batch size of the dataset sans partial
    batches. If this cannot be inferred statically, the value of this tensor
    will be -1.
  """

  def get_static_batch_dim(output_shape):
    if output_shape.rank is None:
      return None
    return output_shape.dims[0].value

  batch_dims = [
      get_static_batch_dim(ts._to_legacy_output_shapes())  # pylint: disable=protected-access
      for ts in nest.flatten(dataset_ops.get_structure(dataset))
  ]

  if all(d is not None for d in batch_dims):

    if all(d == batch_dims[0] for d in batch_dims):
      # If all batch dimensions are known and equal, return that directly.
      batch_dim = batch_dims[0]
    else:
      # If all batch dimensions are known but not all equal, return -1.
      batch_dim = -1

    return constant_op.constant(
        batch_dim, dtype=dtypes.int64, name="static_batch_size")

  # If any batch dimensions are unknown, use compute_batch_size op.
  return ged_ops.compute_batch_size(dataset._variant_tensor)  # pylint: disable=protected-access


_AutoShardDatasetV1.__doc__ = _AutoShardDataset.__doc__
