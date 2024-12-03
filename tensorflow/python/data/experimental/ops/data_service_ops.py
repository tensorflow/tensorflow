# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Python API for executing a tf.data.Dataset using a tf.data service."""

import enum
import functools
from typing import Callable

from tensorflow.core.protobuf import data_service_pb2
from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import compression_ops
from tensorflow.python.data.experimental.service import _pywrap_server_lib
from tensorflow.python.data.experimental.service import _pywrap_utils_exp
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.ops.options import AutoShardPolicy
from tensorflow.python.data.ops.options import ExternalStatePolicy
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export

COMPRESSION_AUTO = "AUTO"
COMPRESSION_NONE = None
COMPRESSION_SNAPPY = "SNAPPY"
_PARALLEL_EPOCHS = "parallel_epochs"
_DISTRIBUTED_EPOCH = "distributed_epoch"


@tf_export("data.experimental.service.ShardingPolicy")
class ShardingPolicy(enum.IntEnum):
  """Specifies how to shard data among tf.data service workers.

  OFF: No sharding will be performed. Each worker produces the entire dataset
  without any sharding. With this mode, the best practice is to shuffle the
  dataset nondeterministically so that workers process the dataset in different
  orders. If workers are restarted or join the cluster mid-job, they will begin
  processing the dataset from the beginning.

  DYNAMIC: The input dataset is dynamically split among workers at runtime. Each
  worker gets the next split when it reads data from the dispatcher. Data is
  produced non-deterministically in this mode. Dynamic sharding works well with
  varying-sized tf.data service clusters, e.g., when you need to auto-scale your
  workers. Dynamic sharding provides at-most once visitation guarantees. No
  examples will be repeated, but some may be missed if a tf.data service worker
  gets restarted while processing a file.

  The following are static sharding policies. The semantics are similar to
  `tf.data.experimental.AutoShardPolicy`. These policies require:
  * The tf.data service cluster is configured with a fixed list of workers
    in DispatcherConfig.
  * Each client only reads from the local tf.data service worker.

  If a worker is restarted while performing static sharding, the worker will
  begin processing its shard again from the beginning.

  FILE: Shards by input files (i.e. each worker will get a fixed set of files to
  process). When this option is selected, make sure that there is at least as
  many files as workers. If there are fewer input files than workers, a runtime
  error will be raised.

  DATA: Shards by elements produced by the dataset. Each worker will process the
  whole dataset and discard the portion that is not for itself. Note that for
  this mode to correctly partition the dataset elements, the dataset needs to
  produce elements in a deterministic order.

  FILE_OR_DATA: Attempts FILE-based sharding, falling back to DATA-based
  sharding on failure.

  HINT: Looks for the presence of `shard(SHARD_HINT, ...)` which is treated as a
  placeholder to replace with `shard(num_workers, worker_index)`.
  """

  # LINT.IfChange(tf_data_service_sharding_policy)
  OFF = 0
  DYNAMIC = 1
  FILE = 2
  DATA = 3
  FILE_OR_DATA = 4
  HINT = 5
  # LINT.ThenChange()

  def _to_proto(self) -> data_service_pb2.ProcessingModeDef.ShardingPolicy:
    """Converts the policy to ProcessingModeDef proto enum."""

    if self == ShardingPolicy.OFF:
      return data_service_pb2.ProcessingModeDef.OFF
    if self == ShardingPolicy.DYNAMIC:
      return data_service_pb2.ProcessingModeDef.DYNAMIC
    if self == ShardingPolicy.FILE:
      return data_service_pb2.ProcessingModeDef.FILE
    if self == ShardingPolicy.DATA:
      return data_service_pb2.ProcessingModeDef.DATA
    if self == ShardingPolicy.FILE_OR_DATA:
      return data_service_pb2.ProcessingModeDef.FILE_OR_DATA
    if self == ShardingPolicy.HINT:
      return data_service_pb2.ProcessingModeDef.HINT
    raise ValueError(f"Unable to convert sharding policy {self!r} to proto.")


@tf_export("data.experimental.service.CrossTrainerCache")
class CrossTrainerCache:
  """Options related to the tf.data service cross trainer cache.

  This is used to enable cross-trainer cache when distributing a dataset. For
  example:

  ```
  dataset = dataset.apply(tf.data.experimental.service.distribute(
      processing_mode=tf.data.experimental.service.ShardingPolicy.OFF,
      service=FLAGS.tf_data_service_address,
      job_name="job",
      cross_trainer_cache=data_service_ops.CrossTrainerCache(
          trainer_id=trainer_id())))
  ```

  For more details, refer to
  https://www.tensorflow.org/api_docs/python/tf/data/experimental/service#sharing_tfdata_service_with_concurrent_trainers.
  """

  def __init__(self, trainer_id):
    """Constructs a CrossTrainerCache.

    Args:
      trainer_id: Each training job has a unique ID. Once a job has consumed
      data, the data remains in the cache and is re-used by jobs with different
      `trainer_id`s. Requests with the same `trainer_id` do not re-use data.

    Raises:
      ValueError if `trainer_id` is empty.
    """
    if not trainer_id:
      raise ValueError(
          "tf.data service cross-trainer cache requires a non-empty trainer ID."
      )
    self.trainer_id = trainer_id

  def _to_proto(self) -> data_service_pb2.CrossTrainerCacheOptions:
    return data_service_pb2.CrossTrainerCacheOptions(trainer_id=self.trainer_id)


def _get_validated_sharding_policy(processing_mode) -> ShardingPolicy:
  """Validates `processing_mode` and converts it to ShardingPolicy."""

  if isinstance(processing_mode, ShardingPolicy):
    return processing_mode
  if processing_mode == _PARALLEL_EPOCHS:
    return ShardingPolicy.OFF
  if processing_mode == _DISTRIBUTED_EPOCH:
    return ShardingPolicy.DYNAMIC

  raise ValueError("tf.data service processing mode should be a "
                   "`tf.data.experimental.service.ShardingPolicy`, "
                   "`\"parallel_epochs\"`, or `\"distributed_epoch\"`. Got "
                   f"{processing_mode!r}.")


def _validate_job_name(job_name) -> None:
  if job_name is None:
    return
  if not isinstance(job_name, str):
    raise ValueError("`job_name` must be a string, but `job_name` was of type "
                     f"{type(job_name)}. job_name={job_name}")
  if not job_name:
    raise ValueError("`job_name` must not be empty")


def _validate_compression(compression) -> None:
  valid_compressions = [
      COMPRESSION_AUTO,
      COMPRESSION_NONE,
      COMPRESSION_SNAPPY,
  ]
  if compression not in valid_compressions:
    raise ValueError(f"Invalid `compression` argument: {compression}. "
                     f"Must be one of {valid_compressions}.")


def _get_compression_proto(
    compression) -> data_service_pb2.DataServiceMetadata.Compression:
  if compression == COMPRESSION_AUTO:
    return data_service_pb2.DataServiceMetadata.COMPRESSION_SNAPPY
  if compression == COMPRESSION_SNAPPY:
    return data_service_pb2.DataServiceMetadata.COMPRESSION_FORCED_SNAPPY
  if compression == COMPRESSION_NONE:
    return data_service_pb2.DataServiceMetadata.COMPRESSION_OFF
  raise ValueError(f"Invalid `compression` argument: {compression}. "
                   f"Must be one of {[COMPRESSION_AUTO, COMPRESSION_NONE]}.")


def _to_tensor(dataset_id) -> tensor.Tensor:
  """Converts `dataset_id` to Tensor."""

  if isinstance(dataset_id, tensor.Tensor):
    return dataset_id
  if isinstance(dataset_id, str) or isinstance(dataset_id, bytes):
    return ops.convert_to_tensor(
        dataset_id, dtype=dtypes.string, name="dataset_id")
  return ops.convert_to_tensor(
      dataset_id, dtype=dtypes.int64, name="dataset_id")


def _to_string(dataset_id) -> str:
  """Converts `dataset_id` to string."""

  if isinstance(dataset_id, tensor.Tensor):
    return (dataset_id if dataset_id.dtype == dtypes.string else
            string_ops.as_string(dataset_id))
  return (dataset_id.decode()
          if isinstance(dataset_id, bytes) else str(dataset_id))


class _DataServiceDatasetV2(dataset_ops.DatasetSource):
  """A `Dataset` that reads elements from the tf.data service."""

  def __init__(self,
               dataset_id,
               processing_mode,
               address,
               element_spec,
               protocol,
               data_transfer_protocol,
               job_name=None,
               consumer_index=None,
               num_consumers=None,
               max_outstanding_requests=None,
               task_refresh_interval_hint_ms=None,
               cross_trainer_cache=None,
               target_workers="AUTO"):
    """Constructs a _DataServiceDatasetV2.

    Args:
      dataset_id: The dataset id for the dataset to read from.
      processing_mode: A `tf.data.experimental.service.ShardingPolicy`
        specifying how to shard the dataset among tf.data workers. See
        `tf.data.experimental.service.ShardingPolicy` for details. For backwards
        compatibility, `processing_mode` may also be set to the strings
        `"parallel_epochs"` or `"distributed_epoch"`, which are respectively
        equivalent to `ShardingPolicy.OFF` and `ShardingPolicy.DYNAMIC`.
      address: The tf.data service address, e.g. "localhost:5000".
      element_spec: The dataset element spec for the dataset to read from.
      protocol: The protocol to use for communicating with the tf.data service,
        e.g. "grpc".
      data_transfer_protocol: (Optional.) The protocol to use for transferring
        data with the tf.data service. If not provided, a protocol is determined
        at runtime.
      job_name: (Optional.) The name of the job. If provided, it must be a
        non-empty string or Tensor. This argument makes it possible for multiple
        datasets to share the same job. The default behavior is that the dataset
        creates anonymous, exclusively owned jobs.
      consumer_index: (Optional.) The index of the consumer in the range from
        `0` to `num_consumers`. Must be specified alongside `num_consumers`.
        When specified, consumers will read from the job in a strict round-robin
        order, instead of the default first-come-first-served order.
      num_consumers: (Optional.) The number of consumers which will consume from
        the job. Must be specified alongside `consumer_index`. When specified,
        consumers will read from the job in a strict round-robin order, instead
        of the default first-come-first-served order. When `num_consumers` is
        specified, the dataset must have infinite cardinality to prevent a
        producer from running out of data early and causing consumers to go out
        of sync.
      max_outstanding_requests: (Optional.) A limit on how many elements may be
        requested at the same time. You can use this option to control the
        amount of memory used, since `distribute` won't use more than
        `element_size` * `max_outstanding_requests` of memory.
      task_refresh_interval_hint_ms: (Optional.) A hint for how often to query
        the dispatcher for task changes.
      cross_trainer_cache: (Optional.) If a `CrossTrainerCache` object is
        provided, dataset iteration will be shared across concurrently running
        trainers. See
        https://www.tensorflow.org/api_docs/python/tf/data/experimental/service#sharing_tfdata_service_with_concurrent_trainers
          for details.
      target_workers: (Optional.) Which workers to read from. If `"AUTO"`,
        tf.data runtime decides which workers to read from. If `"ANY"`, reads
        from any tf.data service workers. If `"LOCAL"`, only reads from local
        in-processs tf.data service workers. `"AUTO"` works well for most cases,
        while users can specify other targets. For example, `"LOCAL"` helps
        avoid RPCs and data copy if every TF worker colocates with a tf.data
        service worker. Consumers of a shared job must use the same
        `target_workers`. Defaults to `"AUTO"`.
    """
    if consumer_index is None != num_consumers is None:
      raise ValueError(
          "Must either set both `consumer_index` and `num_consumers`, "
          "or neither. ",
          f"consumer_index={consumer_index}, num_consumers={num_consumers}")
    if num_consumers is not None and job_name is None:
      raise ValueError("`job_name` must be set when setting `num_consumers`. "
                       f"num_consumers was set to {num_consumers}.")

    processing_mode_def = data_service_pb2.ProcessingModeDef(
        sharding_policy=_get_validated_sharding_policy(
            processing_mode)._to_proto())
    if job_name is None:
      job_name = ""
    if max_outstanding_requests is None:
      max_outstanding_requests = dataset_ops.AUTOTUNE
    if task_refresh_interval_hint_ms is None:
      task_refresh_interval_hint_ms = dataset_ops.AUTOTUNE

    self._dataset_id = _to_tensor(dataset_id)
    self._processing_mode = ops.convert_to_tensor(
        processing_mode_def.SerializeToString(),
        dtype=dtypes.string,
        name="processing_mode")
    self._address = ops.convert_to_tensor(
        address, dtype=dtypes.string, name="address")
    self._protocol = ops.convert_to_tensor(
        protocol, dtype=dtypes.string, name="protocol")
    self._job_name = ops.convert_to_tensor(
        job_name, dtype=dtypes.string, name="job_name")
    self._consumer_index = ops.convert_to_tensor(
        -1 if consumer_index is None else consumer_index,
        dtype=dtypes.int64,
        name="consumer_index")
    self._num_consumers = ops.convert_to_tensor(
        -1 if num_consumers is None else num_consumers,
        dtype=dtypes.int64,
        name="num_consumers")
    self._max_outstanding_requests = ops.convert_to_tensor(
        max_outstanding_requests,
        dtype=dtypes.int64,
        name="max_outstanding_requests")
    self._element_spec = element_spec
    uncompress_func = structured_function.StructuredFunctionWrapper(
        lambda x: compression_ops.uncompress(x, output_spec=element_spec),
        transformation_name="DataServiceDataset.uncompress()",
        input_structure=tensor.TensorSpec(shape=(), dtype=dtypes.variant))
    cross_trainer_cache_options = (
        cross_trainer_cache._to_proto().SerializeToString()
        if cross_trainer_cache else None)

    compat_kwargs = {}
    if data_transfer_protocol is not None:
      compat_kwargs["data_transfer_protocol"] = data_transfer_protocol

    # If `uncompress` is `True`, the dataset will query the servers to find
    # out the actual compression used. It is always set to `True` the first
    # time the graph is built, and set to false when serializing, so we will
    # uncompress at most once.
    uncompress = True
    variant_tensor = gen_experimental_dataset_ops.data_service_dataset_v4(
        dataset_id=self._dataset_id,
        processing_mode=self._processing_mode,
        address=self._address,
        protocol=self._protocol,
        job_name=self._job_name,
        consumer_index=self._consumer_index,
        num_consumers=self._num_consumers,
        max_outstanding_requests=self._max_outstanding_requests,
        task_refresh_interval_hint_ms=task_refresh_interval_hint_ms,
        iteration_counter=(
            gen_experimental_dataset_ops.dummy_iteration_counter()),
        target_workers=target_workers,
        uncompress=uncompress,
        uncompress_fn=uncompress_func.function,
        cross_trainer_cache_options=cross_trainer_cache_options,
        **compat_kwargs,
        **self._flat_structure)
    super(_DataServiceDatasetV2, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._element_spec


class _DataServiceDatasetV1(dataset_ops.DatasetV1Adapter):
  """A `Dataset` that executes its input through the tf.data service."""

  @functools.wraps(_DataServiceDatasetV2.__init__)
  def __init__(self, dataset_id, processing_mode, address, element_spec,
               protocol, data_transfer_protocol, job_name, consumer_index,
               num_consumers, max_outstanding_requests,
               task_refresh_interval_hint_ms, cross_trainer_cache,
               target_workers):

    self._wrapped = _DataServiceDatasetV2(
        dataset_id=dataset_id,
        processing_mode=processing_mode,
        address=address,
        element_spec=element_spec,
        protocol=protocol,
        data_transfer_protocol=data_transfer_protocol,
        job_name=job_name,
        consumer_index=consumer_index,
        num_consumers=num_consumers,
        max_outstanding_requests=max_outstanding_requests,
        task_refresh_interval_hint_ms=task_refresh_interval_hint_ms,
        cross_trainer_cache=cross_trainer_cache,
        target_workers=target_workers)
    super(_DataServiceDatasetV1, self).__init__(self._wrapped)


if tf2.enabled():
  _DataServiceDataset = _DataServiceDatasetV2
else:
  _DataServiceDataset = _DataServiceDatasetV1


def _parse_service(service) -> tuple[str, str]:
  """Converts a tf.data service string into a (protocol, address) tuple.

  Args:
    service: A string in the format "protocol://address" or just "address". If
      the string is only an address, the default protocol will be used.

  Returns:
    The (protocol, address) tuple
  """
  if not isinstance(service, str):
    raise ValueError("`service` must be a string, but `service` was of type "
                     f"{type(service)}. service={service}")
  if not service:
    raise ValueError("`service` must not be empty")
  parts = service.split("://")
  if len(parts) == 2:
    protocol, address = parts
  elif len(parts) == 1:
    address = parts[0]
    protocol = _pywrap_utils_exp.TF_DATA_DefaultProtocol()
  else:
    raise ValueError("Malformed `service` string has multiple '://': "
                     f"{service}.")
  # TODO(aaudibert): Considering validating reachability of address here.
  return (protocol, address)


def _distribute(
    processing_mode,
    service,
    job_name=None,
    consumer_index=None,
    num_consumers=None,
    max_outstanding_requests=None,
    task_refresh_interval_hint_ms=None,
    data_transfer_protocol=None,
    compression="AUTO",
    cross_trainer_cache=None,
    target_workers="AUTO",
) -> Callable[dataset_ops.Dataset, dataset_ops.Dataset]:
  """A transformation that moves dataset processing to the tf.data service.

  This transformation is similar to `distribute`, but supports additional
  parameters which we do not yet want to add to the public Python API.

  Args:
    processing_mode: A `tf.data.experimental.service.ShardingPolicy` specifying
      how to shard the dataset among tf.data workers. See
      `tf.data.experimental.service.ShardingPolicy` for details. For backwards
      compatibility, `processing_mode` may also be set to the strings
      `"parallel_epochs"` or `"distributed_epoch"`, which are respectively
      equivalent to `ShardingPolicy.OFF` and `ShardingPolicy.DYNAMIC`.
    service: A string or a tuple indicating how to connect to the tf.data
      service. If it's a string, it should be in the format
      `[<protocol>://]<address>`, where `<address>` identifies the dispatcher
      address and `<protocol>` can optionally be used to override the default
      protocol to use. If it's a tuple, it should be (protocol, address).
    job_name: (Optional.) The name of the job. If provided, it must be a
      non-empty string. This argument makes it possible for multiple datasets to
      share the same job. The default behavior is that the dataset creates
      anonymous, exclusively owned jobs.
    consumer_index: (Optional.) The index of the consumer in the range from `0`
      to `num_consumers`. Must be specified alongside `num_consumers`. When
      specified, consumers will read from the job in a strict round-robin order,
      instead of the default first-come-first-served order.
    num_consumers: (Optional.) The number of consumers which will consume from
      the job. Must be specified alongside `consumer_index`. When specified,
      consumers will read from the job in a strict round-robin order, instead of
      the default first-come-first-served order. When `num_consumers` is
      specified, the dataset must have infinite cardinality to prevent a
      producer from running out of data early and causing consumers to go out of
      sync.
    max_outstanding_requests: (Optional.) A limit on how many elements may be
      requested at the same time. You can use this option to control the amount
      of memory used, since `distribute` won't use more than `element_size` *
      `max_outstanding_requests` of memory.
    task_refresh_interval_hint_ms: (Optional.) A hint for how often to query the
      dispatcher for task changes.
    data_transfer_protocol: (Optional.) The protocol to use for transferring
      data with the tf.data service. If not provided, a protocol is determined
      at runtime.
    compression: How to compress the dataset's elements before transferring them
      over the network. "AUTO" leaves the decision of how to compress up to the
      tf.data service runtime. `None` indicates not to compress. "SNAPPY" forces
      snappy compression.
    cross_trainer_cache: (Optional.) If a `CrossTrainerCache` object is
      provided, dataset iteration will be shared across concurrently running
      trainers. See
      https://www.tensorflow.org/api_docs/python/tf/data/experimental/service#sharing_tfdata_service_with_concurrent_trainers
        for details.
    target_workers: (Optional.) Which workers to read from. If `"AUTO"`, tf.data
      runtime decides which workers to read from. If `"ANY"`, reads from any
      tf.data service workers. If `"LOCAL"`, only reads from local in-processs
      tf.data service workers. `"AUTO"` works well for most cases, while users
      can specify other targets. For example, `"LOCAL"` helps avoid RPCs and
      data copy if every TF worker colocates with a tf.data service worker.
      Consumers of a shared job must use the same `target_workers`. Defaults to
      `"AUTO"`.

  Returns:
    Dataset: A `Dataset` of the elements produced by the data service.
  """
  processing_mode = _get_validated_sharding_policy(processing_mode)
  _validate_compression(compression)

  def _apply_fn(dataset) -> dataset_ops.Dataset:  # pylint: disable=missing-docstring
    dataset_id = _register_dataset(service, dataset, compression=compression)
    return _from_dataset_id(
        processing_mode,
        service,
        dataset_id,
        dataset.element_spec,
        job_name=job_name,
        consumer_index=consumer_index,
        num_consumers=num_consumers,
        max_outstanding_requests=max_outstanding_requests,
        task_refresh_interval_hint_ms=task_refresh_interval_hint_ms,
        data_transfer_protocol=data_transfer_protocol,
        cross_trainer_cache=cross_trainer_cache,
        target_workers=target_workers)

  return _apply_fn


@tf_export("data.experimental.service.distribute")
def distribute(
    processing_mode,
    service,
    job_name=None,
    consumer_index=None,
    num_consumers=None,
    max_outstanding_requests=None,
    data_transfer_protocol=None,
    compression="AUTO",
    cross_trainer_cache=None,
    target_workers="AUTO",
) -> Callable[dataset_ops.Dataset, dataset_ops.Dataset]:
  """A transformation that moves dataset processing to the tf.data service.

  When you iterate over a dataset containing the `distribute` transformation,
  the tf.data service creates a "job" which produces data for the dataset
  iteration.

  The tf.data service uses a cluster of workers to prepare data for training
  your model.
  The `processing_mode` argument to `tf.data.experimental.service.distribute`
  describes how to leverage multiple workers to process the input dataset.
  Currently, there are two processing modes to choose from: "distributed_epoch"
  and "parallel_epochs".

  "distributed_epoch" means that the dataset will be split across all tf.data
  service workers.
  The dispatcher produces "splits" for the dataset and sends them to workers for
  further processing. For example, if a dataset begins with a list of filenames,
  the dispatcher will iterate through the filenames and send the filenames to
  tf.data workers, which will perform the rest of the dataset transformations on
  those files. "distributed_epoch" is useful when your model needs to see each
  element of the dataset exactly once, or if it needs to see the data in a
  generally-sequential order. "distributed_epoch" only works for datasets with
  splittable sources, such as `Dataset.from_tensor_slices`,
  `Dataset.list_files`, or `Dataset.range`.

  "parallel_epochs" means that the entire input dataset will be processed
  independently by each of the tf.data service workers.
  For this reason, it is important to shuffle data (e.g. filenames)
  non-deterministically, so that each worker will process the elements of the
  dataset in a different order. "parallel_epochs" can be used to distribute
  datasets that aren't splittable.

  With two workers, "parallel_epochs" will produce every element of the dataset
  twice:

  >>> dispatcher = tf.data.experimental.service.DispatchServer()
  >>> dispatcher_address = dispatcher.target.split("://")[1]
  >>> # Start two workers
  >>> workers = [
  ...     tf.data.experimental.service.WorkerServer(
  ...         tf.data.experimental.service.WorkerConfig(
  ...             dispatcher_address=dispatcher_address)) for _ in range(2)
  ... ]
  >>> dataset = tf.data.Dataset.range(10)
  >>> dataset = dataset.apply(tf.data.experimental.service.distribute(
  ...     processing_mode="parallel_epochs", service=dispatcher.target))
  >>> sorted([a.item() for a in dataset.as_numpy_iterator()])
  [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]

  "distributed_epoch", on the other hand, will still produce each element once:

  >>> dispatcher = tf.data.experimental.service.DispatchServer()
  >>> dispatcher_address = dispatcher.target.split("://")[1]
  >>> workers = [
  ...     tf.data.experimental.service.WorkerServer(
  ...         tf.data.experimental.service.WorkerConfig(
  ...             dispatcher_address=dispatcher_address)) for _ in range(2)
  ... ]
  >>> dataset = tf.data.Dataset.range(10)
  >>> dataset = dataset.apply(tf.data.experimental.service.distribute(
  ...     processing_mode="distributed_epoch", service=dispatcher.target))
  >>> sorted([a.item() for a in dataset.as_numpy_iterator()])
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  When using `apply(tf.data.experimental.service.distribute(...))`, the dataset
  before the `apply` transformation executes within the tf.data service, while
  the operations after `apply` happen within the local process.

  >>> dispatcher = tf.data.experimental.service.DispatchServer()
  >>> dispatcher_address = dispatcher.target.split("://")[1]
  >>> workers = [
  ...     tf.data.experimental.service.WorkerServer(
  ...         tf.data.experimental.service.WorkerConfig(
  ...             dispatcher_address=dispatcher_address)) for _ in range(2)
  ... ]
  >>> dataset = tf.data.Dataset.range(5)
  >>> dataset = dataset.map(lambda x: x*x)
  >>> dataset = dataset.apply(
  ...    tf.data.experimental.service.distribute("parallel_epochs",
  ...                                            dispatcher.target))
  >>> dataset = dataset.map(lambda x: x+1)
  >>> sorted([a.item() for a in dataset.as_numpy_iterator()])
  [1, 1, 2, 2, 5, 5, 10, 10, 17, 17]

  In the above example, the dataset operations (before applying the `distribute`
  function on the elements) will be executed on the tf.data workers,
  and the elements are provided over RPC. The remaining transformations
  (after the call to `distribute`) will be executed locally. The dispatcher
  and the workers will bind to usused free ports (which are chosen at random),
  in order to communicate with each other. However, to bind them to specific
  ports, the `port` parameter can be passed.

  The `job_name` argument allows jobs to be shared across multiple
  datasets. Instead of each dataset creating its own job, all
  datasets with the same `job_name` will consume from the same job. A new job
  will be created for each iteration of the dataset (with each repetition of
  `Dataset.repeat` counting as a new iteration). Suppose the `DispatchServer`
  is serving on `localhost:5000` and two training workers (in either a single
  client or multi-client setup) iterate over the below dataset, and there is a
  single tf.data worker:

  ```
  range5_dataset = tf.data.Dataset.range(5)
  dataset = range5_dataset.apply(tf.data.experimental.service.distribute(
      "parallel_epochs", "localhost:5000", job_name="my_job_name"))
  for iteration in range(3):
    print(list(dataset))
  ```

  The elements of each job will be split between the two processes, with
  elements being consumed by the processes on a first-come first-served basis.
  One possible result is that process 1 prints

  ```
  [0, 2, 4]
  [0, 1, 3]
  [1]
  ```

  and process 2 prints

  ```
  [1, 3]
  [2, 4]
  [0, 2, 3, 4]
  ```

  Job names must not be re-used across different training jobs within the
  lifetime of the tf.data service. In general, the tf.data service is expected
  to live for the duration of a single training job.
  To use the tf.data service with multiple training jobs, make sure to use
  different job names to avoid conflicts. For example, suppose a training job
  calls `distribute` with `job_name="job"` and reads until end of input. If
  another independent job connects to the same tf.data service and tries to read
  from `job_name="job"`, it will immediately receive end of input, without
  getting any data.

  **Coordinated data read**

  By default, when multiple consumers read from the same job, they receive data
  on a first-come first-served basis. In some use cases, it is advantageous to
  coordinate the consumers. At each step, consumers read data from the same
  worker.

  For example, the tf.data service can be used to coordinate example sizes
  across a cluster during synchronous training, so that during each step all
  replicas train on similar-sized elements. To achieve this, define a dataset
  which generates rounds of `num_consumers` consecutive similar-sized batches,
  then enable coordinated reads by setting `consumer_index` and `num_consumers`.

  NOTE: To keep consumers in sync, round robin data consumption requires that
  the dataset have infinite cardinality. You can get this by adding `.repeat()`
  at the end of the dataset definition.

  **Keras and Distribution Strategies**

  The dataset produced by the `distribute` transformation can be passed to
  Keras' `Model.fit` or Distribution Strategy's
  `tf.distribute.Strategy.experimental_distribute_dataset` like any other
  `tf.data.Dataset`. We recommend setting a `job_name` on the call to
  `distribute` so that if there are multiple workers, they read data from the
  same job. Note that the autosharding normally performed by
  `experimental_distribute_dataset` will be disabled when setting a `job_name`,
  since sharing the job already results in splitting data across the workers.
  When using a shared job, data will be dynamically balanced across workers, so
  that they reach end of input about the same time. This results in better
  worker utilization than with autosharding, where each worker processes an
  independent set of files, and some workers may run out of data earlier than
  others.

  Args:
    processing_mode: A `tf.data.experimental.service.ShardingPolicy` specifying
      how to shard the dataset among tf.data workers. See
      `tf.data.experimental.service.ShardingPolicy` for details. For backwards
      compatibility, `processing_mode` may also be set to the strings
      `"parallel_epochs"` or `"distributed_epoch"`, which are respectively
      equivalent to `ShardingPolicy.OFF` and `ShardingPolicy.DYNAMIC`.
    service: A string or a tuple indicating how to connect to the tf.data
      service. If it's a string, it should be in the format
      `[<protocol>://]<address>`, where `<address>` identifies the dispatcher
      address and `<protocol>` can optionally be used to override the default
      protocol to use. If it's a tuple, it should be (protocol, address).
    job_name: (Optional.) The name of the job. If provided, it must be a
      non-empty string. This argument makes it possible for multiple datasets to
      share the same job. The default behavior is that the dataset creates
      anonymous, exclusively owned jobs.
    consumer_index: (Optional.) The index of the consumer in the range from `0`
      to `num_consumers`. Must be specified alongside `num_consumers`. When
      specified, consumers will read from the job in a strict round-robin order,
      instead of the default first-come-first-served order.
    num_consumers: (Optional.) The number of consumers which will consume from
      the job. Must be specified alongside `consumer_index`. When specified,
      consumers will read from the job in a strict round-robin order, instead of
      the default first-come-first-served order. When `num_consumers` is
      specified, the dataset must have infinite cardinality to prevent a
      producer from running out of data early and causing consumers to go out of
      sync.
    max_outstanding_requests: (Optional.) A limit on how many elements may be
      requested at the same time. You can use this option to control the amount
      of memory used, since `distribute` won't use more than `element_size` *
      `max_outstanding_requests` of memory.
    data_transfer_protocol: (Optional.) The protocol to use for transferring
      data with the tf.data service. If not provided, a protocol is determined
      at runtime.
    compression: How to compress the dataset's elements before transferring them
      over the network. "AUTO" leaves the decision of how to compress up to the
      tf.data service runtime. `None` indicates not to compress. "SNAPPY" forces
      the use of snappy compression.
    cross_trainer_cache: (Optional.) If a `CrossTrainerCache` object is
      provided, dataset iteration will be shared across concurrently running
      trainers. See
      https://www.tensorflow.org/api_docs/python/tf/data/experimental/service#sharing_tfdata_service_with_concurrent_trainers
        for details.
    target_workers: (Optional.) Which workers to read from. If `"AUTO"`, tf.data
      runtime decides which workers to read from. If `"ANY"`, reads from any
      tf.data service workers. If `"LOCAL"`, only reads from local in-processs
      tf.data service workers. `"AUTO"` works well for most cases, while users
      can specify other targets. For example, `"LOCAL"` helps avoid RPCs and
      data copy if every TF worker colocates with a tf.data service worker.
      Consumers of a shared job must use the same `target_workers`. Defaults to
      `"AUTO"`.

  Returns:
    Dataset: A `Dataset` of the elements produced by the data service.
  """
  _validate_job_name(job_name)
  return _distribute(
      processing_mode=processing_mode,
      service=service,
      job_name=job_name,
      consumer_index=consumer_index,
      num_consumers=num_consumers,
      max_outstanding_requests=max_outstanding_requests,
      data_transfer_protocol=data_transfer_protocol,
      compression=compression,
      cross_trainer_cache=cross_trainer_cache,
      target_workers=target_workers)


def _register_dataset(
    service, dataset, compression, dataset_id=None) -> tensor.Tensor:
  """Registers a dataset with the tf.data service.

  This transformation is similar to `register_dataset`, but supports additional
  parameters which we do not yet want to add to the public Python API.

  Args:
    service: A string or a tuple indicating how to connect to the tf.data
      service. If it's a string, it should be in the format
      `[<protocol>://]<address>`, where `<address>` identifies the dispatcher
      address and `<protocol>` can optionally be used to override the default
      protocol to use. If it's a tuple, it should be (protocol, address).
    dataset: A `tf.data.Dataset` to register with the tf.data service.
    compression: How to compress the dataset's elements before transferring them
      over the network. "AUTO" leaves the decision of how to compress up to the
      tf.data service runtime. `None` indicates not to compress. "SNAPPY" forces
      the use of snappy compression.
    dataset_id: (Optional.) By default, tf.data service generates a unique
      (string) ID for each registered dataset. If a `dataset_id` is provided, it
      will use the specified ID. If a dataset with a matching ID already exists,
      no new dataset is registered. This is useful if multiple training jobs
      want to (re)use the same dataset for training. In this case, they can
      register the dataset with the same dataset ID.

  Returns:
    A scalar string tensor representing the dataset ID.
  """
  _validate_compression(compression)

  if isinstance(service, tuple):
    protocol, address = service
  else:
    protocol, address = _parse_service(service)
  external_state_policy = dataset.options().experimental_external_state_policy
  if external_state_policy is None:
    external_state_policy = ExternalStatePolicy.WARN

  encoded_spec = None
  if context.executing_eagerly():
    encoded_spec = nested_structure_coder.encode_structure(
        dataset.element_spec).SerializeToString()

  if (
      compression == COMPRESSION_AUTO
      or compression == COMPRESSION_SNAPPY
  ):
    dataset = dataset.map(
        lambda *x: compression_ops.compress(x),
        num_parallel_calls=dataset_ops.AUTOTUNE)
  dataset = dataset._apply_debug_options()  # pylint: disable=protected-access

  metadata = data_service_pb2.DataServiceMetadata(
      element_spec=encoded_spec,
      compression=_get_compression_proto(compression))

  return gen_experimental_dataset_ops.register_dataset_v2(
      dataset._variant_tensor,  # pylint: disable=protected-access
      address=address,
      protocol=protocol,
      external_state_policy=external_state_policy.value,
      requested_dataset_id=dataset_id,
      metadata=metadata.SerializeToString())


@tf_export("data.experimental.service.register_dataset")
def register_dataset(
    service, dataset, compression="AUTO", dataset_id=None) -> tensor.Tensor:
  """Registers a dataset with the tf.data service.

  `register_dataset` registers a dataset with the tf.data service so that
  datasets can be created later with
  `tf.data.experimental.service.from_dataset_id`. This is useful when the
  dataset
  is registered by one process, then used in another process. When the same
  process is both registering and reading from the dataset, it is simpler to use
  `tf.data.experimental.service.distribute` instead.

  If the dataset is already registered with the tf.data service,
  `register_dataset` returns the already-registered dataset's id.

  >>> dispatcher = tf.data.experimental.service.DispatchServer()
  >>> dispatcher_address = dispatcher.target.split("://")[1]
  >>> worker = tf.data.experimental.service.WorkerServer(
  ...     tf.data.experimental.service.WorkerConfig(
  ...         dispatcher_address=dispatcher_address))
  >>> dataset = tf.data.Dataset.range(10)
  >>> dataset_id = tf.data.experimental.service.register_dataset(
  ...     dispatcher.target, dataset)
  >>> dataset = tf.data.experimental.service.from_dataset_id(
  ...     processing_mode="parallel_epochs",
  ...     service=dispatcher.target,
  ...     dataset_id=dataset_id,
  ...     element_spec=dataset.element_spec)
  >>> [a.item() for a in dataset.as_numpy_iterator()]
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  Args:
    service: A string or a tuple indicating how to connect to the tf.data
      service. If it's a string, it should be in the format
      `[<protocol>://]<address>`, where `<address>` identifies the dispatcher
        address and `<protocol>` can optionally be used to override the default
        protocol to use. If it's a tuple, it should be (protocol, address).
    dataset: A `tf.data.Dataset` to register with the tf.data service.
    compression: (Optional.) How to compress the dataset's elements before
      transferring them over the network. "AUTO" leaves the decision of how to
      compress up to the tf.data service runtime. `None` indicates not to
      compress.
    dataset_id: (Optional.) By default, tf.data service generates a unique
      (string) ID for each registered dataset. If a `dataset_id` is provided, it
      will use the specified ID. If a dataset with a matching ID already exists,
      no new dataset is registered. This is useful if multiple training jobs
      want to (re)use the same dataset for training. In this case, they can
      register the dataset with the same dataset ID.

  Returns:
    A scalar string tensor representing the dataset ID.
  """
  return _register_dataset(service, dataset, compression, dataset_id)


def _from_dataset_id(processing_mode,
                     service,
                     dataset_id,
                     element_spec,
                     job_name=None,
                     consumer_index=None,
                     num_consumers=None,
                     max_outstanding_requests=None,
                     task_refresh_interval_hint_ms=None,
                     data_transfer_protocol=None,
                     cross_trainer_cache=None,
                     target_workers="AUTO") -> dataset_ops.Dataset:
  """Creates a dataset which reads data from the tf.data service.

  This transformation is similar to `from_dataset_id`, but supports additional
  parameters which we do not yet want to add to the public Python API.

  Args:
    processing_mode: A `tf.data.experimental.service.ShardingPolicy` specifying
      how to shard the dataset among tf.data workers. See
      `tf.data.experimental.service.ShardingPolicy` for details. For backwards
      compatibility, `processing_mode` may also be set to the strings
      `"parallel_epochs"` or `"distributed_epoch"`, which are respectively
      equivalent to `ShardingPolicy.OFF` and `ShardingPolicy.DYNAMIC`.
    service: A string or a tuple indicating how to connect to the tf.data
      service. If it's a string, it should be in the format
      `[<protocol>://]<address>`, where `<address>` identifies the dispatcher
      address and `<protocol>` can optionally be used to override the default
      protocol to use. If it's a tuple, it should be (protocol, address).
    dataset_id: The id of the dataset to read from. This id is returned by
      `register_dataset` when the dataset is registered with the tf.data
      service.
    element_spec: A nested structure of `tf.TypeSpec`s representing the type of
      elements produced by the dataset. This argument is only required inside a
      tf.function. Use `tf.data.Dataset.element_spec` to get the element spec
      for a given dataset.
    job_name: (Optional.) The name of the job. If provided, it must be a
      non-empty string or tensor. This argument makes it possible for multiple
      datasets to share the same job. The default behavior is that the dataset
      creates anonymous, exclusively owned jobs.
    consumer_index: (Optional.) The index of the consumer in the range from `0`
      to `num_consumers`. Must be specified alongside `num_consumers`. When
      specified, consumers will read from the job in a strict round-robin order,
      instead of the default first-come-first-served order.
    num_consumers: (Optional.) The number of consumers which will consume from
      the job. Must be specified alongside `consumer_index`. When specified,
      consumers will read from the job in a strict round-robin order, instead of
      the default first-come-first-served order. When `num_consumers` is
      specified, the dataset must have infinite cardinality to prevent a
      producer from running out of data early and causing consumers to go out of
      sync.
    max_outstanding_requests: (Optional.) A limit on how many elements may be
      requested at the same time. You can use this option to control the amount
      of memory used, since `distribute` won't use more than `element_size` *
      `max_outstanding_requests` of memory.
    task_refresh_interval_hint_ms: (Optional.) A hint for how often to query the
      dispatcher for task changes.
    data_transfer_protocol: (Optional.) The protocol to use for transferring
      data with the tf.data service. If not provided, a protocol is determined
      at runtime.
    cross_trainer_cache: (Optional.) If a `CrossTrainerCache` object is
      provided, dataset iteration will be shared across concurrently running
      trainers. See
      https://www.tensorflow.org/api_docs/python/tf/data/experimental/service#sharing_tfdata_service_with_concurrent_trainers
        for details.
    target_workers: (Optional.) Which workers to read from. If `"AUTO"`, tf.data
      runtime decides which workers to read from. If `"ANY"`, reads from any
      tf.data service workers. If `"LOCAL"`, only reads from local in-processs
      tf.data service workers. `"AUTO"` works well for most cases, while users
      can specify other targets. For example, `"LOCAL"` helps avoid RPCs and
      data copy if every TF worker colocates with a tf.data service worker.
      Consumers of a shared job must use the same `target_workers`. Defaults to
      `"AUTO"`.

  Returns:
    A `tf.data.Dataset` which reads from the tf.data service.
  """
  def _get_element_spec():
    """Fetches the element spec from the server."""
    data_service_metadata = None
    dataset_id_val = tensor_util.constant_value(dataset_id)
    try:
      data_service_metadata = (
          _pywrap_server_lib.TF_DATA_GetDataServiceMetadataByID(
              dataset_id_val, address, protocol
          )
      )
    except NotImplementedError as err:
      raise ValueError(
          "The tf.data service is running an earlier version of TensorFlow "
          "that requires specifying `element_spec` as an argument to "
          "`from_dataset_id`. Please either supply an element spec or update "
          "the tf.data service to the latest version.") from err
    except RuntimeError:
      # This error results from dataset ID not found. A more appropriate error
      # will be raised when the dataset is created.
      pass

    if not data_service_metadata or not data_service_metadata.element_spec:
      dataset_id_val = tensor_util.constant_value(dataset_id)
      raise ValueError(
          f"Failed to fetch element spec for dataset id {dataset_id_val} from "
          "tf.data service. If the dataset was registered in graph mode or "
          "inside a tf.function, the `element_spec` must be specified as an "
          "argument to `from_dataset_id`.")

    struct_pb = nested_structure_coder.struct_pb2.StructuredValue()
    struct_pb.ParseFromString(data_service_metadata.element_spec)
    return nested_structure_coder.decode_proto(struct_pb)

  processing_mode = _get_validated_sharding_policy(processing_mode)
  if isinstance(service, tuple):
    protocol, address = service
  else:
    protocol, address = _parse_service(service)
  if job_name is not None:
    if not isinstance(job_name, str) and not isinstance(
        job_name, tensor.Tensor):
      raise ValueError(
          "`job_name` must be a string or Tensor, but `job_name` was of type "
          f"{type(job_name)}. job_name={job_name}.")

  if not element_spec:
    if not context.executing_eagerly():
      raise ValueError(
          "In graph mode `element_spec` must be provided manually.")
    element_spec = _get_element_spec()

  dataset = _DataServiceDataset(
      dataset_id=dataset_id,
      processing_mode=processing_mode,
      address=address,
      element_spec=element_spec,
      protocol=protocol,
      data_transfer_protocol=data_transfer_protocol,
      job_name=job_name,
      consumer_index=consumer_index,
      num_consumers=num_consumers,
      max_outstanding_requests=max_outstanding_requests,
      task_refresh_interval_hint_ms=task_refresh_interval_hint_ms,
      cross_trainer_cache=cross_trainer_cache,
      target_workers=target_workers)

  # Disable autosharding for shared jobs.
  if job_name is not None:
    options = options_lib.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.OFF
    dataset = dataset.with_options(options)
  return dataset


@tf_export("data.experimental.service.from_dataset_id")
def from_dataset_id(processing_mode,
                    service,
                    dataset_id,
                    element_spec=None,
                    job_name=None,
                    consumer_index=None,
                    num_consumers=None,
                    max_outstanding_requests=None,
                    data_transfer_protocol=None,
                    cross_trainer_cache=None,
                    target_workers="AUTO") -> dataset_ops.Dataset:
  """Creates a dataset which reads data from the tf.data service.

  This is useful when the dataset is registered by one process, then used in
  another process. When the same process is both registering and reading from
  the dataset, it is simpler to use `tf.data.experimental.service.distribute`
  instead.

  Before using `from_dataset_id`, the dataset must have been registered with the
  tf.data service using `tf.data.experimental.service.register_dataset`.
  `register_dataset` returns a dataset id for the registered dataset. That is
  the `dataset_id` which should be passed to `from_dataset_id`.

  The `element_spec` argument indicates the `tf.TypeSpec`s for the elements
  produced by the dataset. Currently `element_spec` must be explicitly
  specified, and match the dataset registered under `dataset_id`. `element_spec`
  defaults to `None` so that in the future we can support automatically
  discovering the `element_spec` by querying the tf.data service.

  `tf.data.experimental.service.distribute` is a convenience method which
  combines `register_dataset` and `from_dataset_id` into a dataset
  transformation.
  See the documentation for `tf.data.experimental.service.distribute` for more
  detail about how `from_dataset_id` works.

  >>> dispatcher = tf.data.experimental.service.DispatchServer()
  >>> dispatcher_address = dispatcher.target.split("://")[1]
  >>> worker = tf.data.experimental.service.WorkerServer(
  ...     tf.data.experimental.service.WorkerConfig(
  ...         dispatcher_address=dispatcher_address))
  >>> dataset = tf.data.Dataset.range(10)
  >>> dataset_id = tf.data.experimental.service.register_dataset(
  ...     dispatcher.target, dataset)
  >>> dataset = tf.data.experimental.service.from_dataset_id(
  ...     processing_mode="parallel_epochs",
  ...     service=dispatcher.target,
  ...     dataset_id=dataset_id,
  ...     element_spec=dataset.element_spec)
  >>> [a.item() for a in dataset.as_numpy_iterator()]
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  Args:
    processing_mode: A `tf.data.experimental.service.ShardingPolicy` specifying
      how to shard the dataset among tf.data workers. See
      `tf.data.experimental.service.ShardingPolicy` for details. For backwards
      compatibility, `processing_mode` may also be set to the strings
      `"parallel_epochs"` or `"distributed_epoch"`, which are respectively
      equivalent to `ShardingPolicy.OFF` and `ShardingPolicy.DYNAMIC`.
    service: A string or a tuple indicating how to connect to the tf.data
      service. If it's a string, it should be in the format
      `[<protocol>://]<address>`, where `<address>` identifies the dispatcher
      address and `<protocol>` can optionally be used to override the default
      protocol to use. If it's a tuple, it should be (protocol, address).
    dataset_id: The id of the dataset to read from. This id is returned by
      `register_dataset` when the dataset is registered with the tf.data
      service.
    element_spec: A nested structure of `tf.TypeSpec`s representing the type of
      elements produced by the dataset. This argument is only required inside a
      tf.function. Use `tf.data.Dataset.element_spec` to get the element spec
      for a given dataset.
    job_name: (Optional.) The name of the job. If provided, it must be a
      non-empty string. This argument makes it possible for multiple datasets to
      share the same job. The default behavior is that the dataset creates
      anonymous, exclusively owned jobs.
    consumer_index: (Optional.) The index of the consumer in the range from `0`
      to `num_consumers`. Must be specified alongside `num_consumers`. When
      specified, consumers will read from the job in a strict round-robin order,
      instead of the default first-come-first-served order.
    num_consumers: (Optional.) The number of consumers which will consume from
      the job. Must be specified alongside `consumer_index`. When specified,
      consumers will read from the job in a strict round-robin order, instead of
      the default first-come-first-served order. When `num_consumers` is
      specified, the dataset must have infinite cardinality to prevent a
      producer from running out of data early and causing consumers to go out of
      sync.
    max_outstanding_requests: (Optional.) A limit on how many elements may be
      requested at the same time. You can use this option to control the amount
      of memory used, since `distribute` won't use more than `element_size` *
      `max_outstanding_requests` of memory.
    data_transfer_protocol: (Optional.) The protocol to use for transferring
      data with the tf.data service. If not provided, a protocol is determined
      at runtime.
    cross_trainer_cache: (Optional.) If a `CrossTrainerCache` object is
      provided, dataset iteration will be shared across concurrently running
      trainers. See
      https://www.tensorflow.org/api_docs/python/tf/data/experimental/service#sharing_tfdata_service_with_concurrent_trainers
        for details.
    target_workers: (Optional.) Which workers to read from. If `"AUTO"`, tf.data
      runtime decides which workers to read from. If `"ANY"`, reads from any
      tf.data service workers. If `"LOCAL"`, only reads from local in-processs
      tf.data service workers. `"AUTO"` works well for most cases, while users
      can specify other targets. For example, `"LOCAL"` helps avoid RPCs and
      data copy if every TF worker colocates with a tf.data service worker.
      Consumers of a shared job must use the same `target_workers`. Defaults to
      `"AUTO"`.

  Returns:
    A `tf.data.Dataset` which reads from the tf.data service.
  """
  _validate_job_name(job_name)
  if job_name is not None:
    job_name = string_ops.string_join(
        ["dataset_id=", _to_string(dataset_id), job_name], "/")

  return _from_dataset_id(
      processing_mode=processing_mode,
      service=service,
      dataset_id=dataset_id,
      element_spec=element_spec,
      job_name=job_name,
      consumer_index=consumer_index,
      num_consumers=num_consumers,
      max_outstanding_requests=max_outstanding_requests,
      data_transfer_protocol=data_transfer_protocol,
      cross_trainer_cache=cross_trainer_cache,
      target_workers=target_workers)
