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

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import rebatch_op
from tensorflow.python.data.ops.options import ExternalStatePolicy
from tensorflow.python.data.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util.tf_export import tf_export

# TODO(b/246022798): Migrate all intenral uses of `_RebatchDataset` to
# `tf.data.Dataset.rebatch` and remove the below symbol forward declaration.
# Symbols forwarded for legacy access through distribute.py. These forwarded
# symbols can be removed once all internal uses are updated.
_RebatchDataset = rebatch_op.RebatchDataset

SHARD_HINT = -1
tf_export("data.experimental.SHARD_HINT").export_constant(
    __name__, "SHARD_HINT")


class _AutoShardDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that shards the `Dataset` automatically.

  This dataset takes in an existing dataset and tries to automatically figure
  out how to shard the dataset in a multi-worker scenario using graph rewrites.

  If the AutoShardPolicy is set to FILE, it walks up the dataset graph until
  it finds a reader dataset, then inserts a ShardDataset op before that node
  so that each worker only sees some files.

  If the AutoShardPolicy is set to DATA, it inserts a ShardDataset op at the
  end of the input pipeline, before any terminal PrefetchDataset if there is
  one. Additionally, if there is a RebatchDatasetV2 in the input pipeline, it
  is written to legacy RebatchDataset for correctness reasons, since
  RebatchDatasetV2 is incompatible with data sharding.

  If the AutoShardPolicy is set to AUTO, it tries to do file-based sharding.
  If it cannot find a reader dataset, it falls back to doing data-based
  sharding.

  If the AutoShardPolicy is set to OFF, it does nothing.

  Attributes:
    num_workers: Total number of workers to shard this dataset across.
    index: The current worker index (out of the total number of workers) this
      dataset is for.
    num_replicas: The total number of replicas across all workers. This is used
      only when sharding by data (either DATA or AUTO) in order to rewrite
      RebatchDatasetV2 to RebatchDataset.

  Raises:
    NotFoundError: If we cannot find a suitable reader dataset to begin
      automatically sharding the dataset.
  """

  def __init__(self, input_dataset, num_workers, index, num_replicas=None):
    self._input_dataset = input_dataset

    self._element_spec = input_dataset.element_spec
    variant_tensor = ged_ops.auto_shard_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        num_workers=num_workers,
        index=index,
        auto_shard_policy=int(
            input_dataset.options().experimental_distribute.auto_shard_policy),
        num_replicas=num_replicas,
        **self._flat_structure)
    super(_AutoShardDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._element_spec


def _AutoShardDatasetV1(input_dataset, num_workers, index, num_replicas=None):  # pylint: disable=invalid-name
  return dataset_ops.DatasetV1Adapter(
      _AutoShardDataset(input_dataset, num_workers, index, num_replicas))


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
        raise ValueError(
            "Invalid `input_dataset`. Expected a dataset whose elements "
            "have rank >= 1 but found a dataset whose elements are scalars. "
            "Fix the issue by adding the `batch` transformation to the "
            "dataset.")
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

    # auto_shard rewrite assumes that there's normalize_to_dense before
    # rebatch_dataset.
    # LINT.IfChange
    input_dataset = dataset_ops.normalize_to_dense(input_dataset)
    variant_tensor = ged_ops.rebatch_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        num_replicas=num_replicas,
        **self._flat_structure)
    # LINT.ThenChange(//tensorflow/core/grappler/optimizers/data/auto_shard.cc)
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
    raise TypeError(
        f"Invalid `dataset`. Expected a `tf.data.Dataset` object but "
        f"got {type(dataset)}.")

  # pylint: disable=protected-access
  dataset_device = dataset._variant_tensor.device

  datasets = {}
  if len(devices) == 1 and devices[0] == dataset_device:
    datasets[devices[0]] = dataset
    return datasets

  with ops.colocate_with(dataset._variant_tensor):
    dataset = dataset._apply_debug_options()
    graph_def = dataset._as_serialized_graph(
        strip_device_assignment=True,
        external_state_policy=ExternalStatePolicy.WARN)
  for device in devices:
    ds = _RemoteDataset(graph_def, device, dataset.element_spec)
    datasets[device] = ds
  return datasets


def batch_sizes_for_worker(global_batch_size, num_workers,
                           num_replicas_per_worker, worker_index):
  """Determines how to rebatch a dataset for the given worker.

  Given the global batch size, number of workers, number of replicas per worker,
  and worker index, returns the correct batch sizes for rebatching a dataset
  on worker `worker_index` of `num_workers`, such that each global step (across
  all workers and replicas) will consume global_batch_size elements. The
  returned value should be passed as the `batch_sizes` input parameter to
  `tf.data.experimental.rebatch()`. The returned batch sizes meet the following
  constraints:

  Let G = global_batch_size, W = num_workers, R = num_replicas_per_worker
  (A) for any worker, len(batch_sizes) = W * R
  (B) for any worker, sum(batch_sizes) == G
  (C) for any global step (i.e. R iterations on each worker), the sum of batches
      consumed by replicas across all workers is G.
  (D) any two batch sizes of any two replicas differs by at most one.

  For example, suppose we have G = 7, W = 2, R = 2, and suppose we have two
  files which each contain 7 elements:

  ```python
  # WORKER 0
  batch_sizes_0 = batch_sizes_for_worker(global_batch_size=global_batch_size,
                                         num_workers=2,
                                         num_replicas_per_worker=2,
                                         worker_index=0)
  print(batch_sizes_0)
  >> [2, 2, 2, 1]

  dataset_0 = tf.data.Dataset.from_tensor_slices(["file_a", "file_b"])
  dataset_0 = dataset_0.shard(num_shards, index=0)
  dataset_0 = dataset_0.batch(7)
  dataset_0 = dataset_0.apply(tf.data.experimental.rebatch(batch_sizes_0))
  for elem in dataset_0:
    print(elem)
  >> [[A0, A1], [A2, A3], [A4, A5], [A6]]

  # WORKER 1
  batch_sizes_1 = batch_sizes_for_worker(global_batch_size=global_batch_size,
                                         num_workers=2,
                                         num_replicas_per_worker=2,
                                         worker_index=1)
  print(batch_sizes_1)
  >> [2, 1, 2, 2]

  dataset_1 = tf.data.Dataset.from_tensor_slices(["file_a", "file_b"])
  dataset_1 = dataset_1.shard(num_shards, index=1)
  dataset_1 = dataset_1.batch(7)
  dataset_1 = dataset_1.apply(tf.data.experimental.rebatch(batch_sizes_1))
  for elem in dataset_1:
    print(elem)
  >> [[B0, B1], [B2], [B3, B4], [B5, B6]]
  ```

  The above example will produce the following elements:

  Step 1:
    Worker 0 Replica 0: [A0, A1]
    Worker 0 Replica 1: [A2, A3]
    Worker 1 Replica 0: [B0, B1]
    Worker 1 Replica 1: [B2]
  Total batch size = 7

  Step 2:
    Worker 0 Replica 0: [A4, A5]
    Worker 0 Replica 1: [A6]
    Worker 1 Replica 0: [B3, B4]
    Worker 1 Replica 1: [B5, B6]
  Total batch size = 7

  Args:
    global_batch_size: A `tf.int64` scalar, representing the global batch size.
    num_workers: An integer representing the number of workers the dataset will
      be distributed across.
    num_replicas_per_worker: An integer representing the number of replicas per
      worker. All workers are assumed to have the same number of replicas.
    worker_index: An integer index of the worker to be rebatched.

  Returns:
    A `tf.int64` vector, representing the batch sizes to rebatch the dataset
    into.
  """
  # Constraint (A)
  num_subbatches = num_workers * num_replicas_per_worker

  offset = worker_index * num_replicas_per_worker

  const_value = tensor_util.constant_value(global_batch_size)
  if const_value is not None:
    # Use the constant global batch size for further calculations
    global_batch_size = const_value

  # Let N = W * R. Constraint (B) and (D) jointly mean that the iterations
  # should have batch size either floor(B/N) or ceil(B/N). Namely, of the N
  # subbatches a batch is split into, B - N * floor(B/N) of them will have size
  # ceil(B/N), and the rest will have size floor(B/N).
  floor = global_batch_size // num_subbatches
  num_ceil = global_batch_size - (num_subbatches * floor)

  # For worker 0, we assign the first num_ceil subbatches to have size
  # ceil(B/N), and the remainder to have size floor(B/N). The other workers will
  # each be offset by R * worker_index in order to meet constraint (C).
  if const_value is not None:
    # If the global batch size is a known constant value, we return a constant
    # tensor directly instead of manipulating it with TF ops. This allows for
    # better downstream shape inference.
    worker_0 = [floor + 1] * num_ceil + [floor] * (num_subbatches - num_ceil)
    return ops.convert_to_tensor(
        worker_0[offset:] + worker_0[:offset],
        dtype=dtypes.int64,
        name="batch_sizes")

  worker_0 = array_ops.ones(num_subbatches, dtype=dtypes.int64)
  worker_0 = floor * worker_0 + array_ops.concat([
      array_ops.ones(num_ceil, dtype=dtypes.int64),
      array_ops.zeros(num_subbatches - num_ceil, dtype=dtypes.int64)
  ],
                                                 axis=0)

  return array_ops.concat([worker_0[offset:], worker_0[:offset]], axis=0)


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

  def get_static_batch_dim(type_spec):
    try:
      output_shape = type_spec._to_legacy_output_shapes()  # pylint: disable=protected-access
    except NotImplementedError:
      return None
    if not isinstance(output_shape, tensor_shape.TensorShape):
      return None
    if output_shape.rank is None:
      return None
    return output_shape.dims[0].value

  batch_dims = [
      get_static_batch_dim(type_spec)
      for type_spec in nest.flatten(dataset_ops.get_structure(dataset))
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
