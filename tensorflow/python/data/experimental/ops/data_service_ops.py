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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import six

from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import compression_ops
from tensorflow.python.data.experimental.ops.distribute_options import AutoShardPolicy
from tensorflow.python.data.experimental.ops.distribute_options import ExternalStatePolicy
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.util.tf_export import tf_export


class ProcessingMode(object):
  """tf.data service processing modes."""

  PARALLEL_EPOCHS = "parallel_epochs"
  DISTRIBUTED_EPOCH = "distributed_epoch"

  @staticmethod
  def validate(mode):
    """Raises a ValueError if the given object is not a valid processing mode."""
    valid_modes = [
        ProcessingMode.PARALLEL_EPOCHS, ProcessingMode.DISTRIBUTED_EPOCH
    ]
    if mode not in valid_modes:
      raise ValueError(
          "{0} is not a valid processing mode. Valid modes: {1}".format(
              mode, valid_modes))


class _DataServiceDatasetV2(dataset_ops.DatasetSource):
  """A `Dataset` that reads elements from the tf.data service."""

  def __init__(self,
               dataset_id,
               processing_mode,
               address,
               protocol,
               job_name=None,
               max_outstanding_requests=None,
               task_refresh_interval_hint_ms=None):
    """Constructs a _DataServiceDatasetV2.

    Args:
      dataset_id: The dataset id for the dataset to read from.
      processing_mode: A string specifying the policy for how data should be
        processed by tf.data workers. Can be either "parallel_epochs" to have
        each tf.data worker process a copy of the dataset, or
        "distributed_epoch" to split a single iteration of the dataset across
        all the workers.
      address: The tf.data service address, e.g. "localhost:5000".
      protocol: The protocol to use for communicating with the tf.data service,
        e.g. "grpc".
      job_name: (Optional.) The name of the job. This argument makes it possible
        for multiple datasets to share the same job. The default behavior is
        that the dataset creates anonymous, exclusively owned jobs.
      max_outstanding_requests: (Optional.) A limit on how many elements may be
        requested at the same time. You can use this option to control the
        amount of memory used, since `distribute` won't use more than
        `element_size` * `max_outstanding_requests` of memory.
      task_refresh_interval_hint_ms: (Optional.) A hint for how often to query
        the dispatcher for task changes.
    """

    if job_name is None:
      job_name = ""
    if max_outstanding_requests is None:
      max_outstanding_requests = dataset_ops.AUTOTUNE
    if task_refresh_interval_hint_ms is None:
      task_refresh_interval_hint_ms = dataset_ops.AUTOTUNE

    self._dataset_id = ops.convert_to_tensor(
        dataset_id, dtype=dtypes.int64, name="dataset_id")
    self._processing_mode = ops.convert_to_tensor(
        processing_mode, dtype=dtypes.string, name="processing_mode")
    self._address = ops.convert_to_tensor(
        address, dtype=dtypes.string, name="address")
    self._protocol = ops.convert_to_tensor(
        protocol, dtype=dtypes.string, name="protocol")
    self._job_name = ops.convert_to_tensor(
        job_name, dtype=dtypes.string, name="job_name")
    self._max_outstanding_requests = ops.convert_to_tensor(
        max_outstanding_requests,
        dtype=dtypes.int64,
        name="max_outstanding_requests")
    # Datasets executed by the tf.data service produce compressed elements
    # represented by scalar DT_VARIANTs.
    self._element_spec = tensor_spec.TensorSpec(shape=(), dtype=dtypes.variant)

    variant_tensor = gen_experimental_dataset_ops.data_service_dataset(
        dataset_id=self._dataset_id,
        processing_mode=self._processing_mode,
        address=self._address,
        protocol=self._protocol,
        job_name=self._job_name,
        max_outstanding_requests=self._max_outstanding_requests,
        task_refresh_interval_hint_ms=task_refresh_interval_hint_ms,
        iteration_counter=gen_experimental_dataset_ops.dummy_iteration_counter(
        ),
        **self._flat_structure)
    super(_DataServiceDatasetV2, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._element_spec


class _DataServiceDatasetV1(dataset_ops.DatasetV1Adapter):
  """A `Dataset` that executes its input through the tf.data service."""

  @functools.wraps(_DataServiceDatasetV2.__init__)
  def __init__(self, dataset_id, processing_mode, address, protocol, job_name,
               max_outstanding_requests, task_refresh_interval_hint_ms):

    self._wrapped = _DataServiceDatasetV2(
        dataset_id=dataset_id,
        processing_mode=processing_mode,
        address=address,
        protocol=protocol,
        job_name=job_name,
        max_outstanding_requests=max_outstanding_requests,
        task_refresh_interval_hint_ms=task_refresh_interval_hint_ms)
    super(_DataServiceDatasetV1, self).__init__(self._wrapped)


if tf2.enabled():
  _DataServiceDataset = _DataServiceDatasetV2
else:
  _DataServiceDataset = _DataServiceDatasetV1


def _parse_service(service):
  """Parses a tf.data service string into a (protocol, address) tuple.

  Args:
    service: A string in the format "protocol://address".

  Returns:
    The parsed (protocol, address) tuple
  """
  if not isinstance(service, six.string_types):
    raise ValueError(
        "service must be a string, but service was of type {0}. service={1}"
        .format(type(service), service))
  if not service:
    raise ValueError("service must not be empty")
  parts = service.split("://")
  if len(parts) == 1:
    raise ValueError("service string %s does not begin with a protocol. "
                     "The service should be in the format "
                     "<protocol>://<address>, e.g. grpc://localhost:5000" %
                     service)
  if len(parts) > 2:
    raise ValueError("malformed service string has multiple '://': %s" %
                     service)
  return parts


def _from_dataset_id(processing_mode,
                     service,
                     dataset_id,
                     element_spec,
                     job_name=None,
                     max_outstanding_requests=None,
                     task_refresh_interval_hint_ms=None):
  """Creates a dataset which reads data from the tf.data service.

  This transformation is similar to `from_dataset_id`, but supports additional
  parameters which we do not yet want to add to the public Python API.

  Args:
    processing_mode: A string specifying the policy for how data should be
      processed by tf.data workers. Can be either "parallel_epochs" to have
      each tf.data worker process a copy of the dataset, or
      "distributed_epoch" to split a single iteration of the dataset across
      all the workers.
    service: A string indicating how to connect to the tf.data service. The
      string should be in the format "<protocol>://<address>", e.g.
      "grpc://localhost:5000".
    dataset_id: The id of the dataset to read from. This id is returned by
      `register_dataset` when the dataset is registered with the tf.data
      service.
    element_spec: A nested structure of `tf.TypeSpec`s representing the type of
      elements produced by the dataset. Use `tf.data.Dataset.element_spec` to
      see the element spec for a given dataset.
    job_name: (Optional.) The name of the job. This argument makes it possible
      for multiple datasets to share the same job. The default behavior is that
      the dataset creates anonymous, exclusively owned jobs.
    max_outstanding_requests: (Optional.) A limit on how many elements may be
      requested at the same time. You can use this option to control the amount
      of memory used, since `distribute` won't use more than `element_size` *
      `max_outstanding_requests` of memory.
    task_refresh_interval_hint_ms: (Optional.) A hint for how often to query the
      dispatcher for task changes.

  Returns:
    A `tf.data.Dataset` which reads from the tf.data service.
  """
  ProcessingMode.validate(processing_mode)
  if job_name is not None:
    if not isinstance(job_name, six.string_types):
      raise ValueError("job_name must be a string, but job_name was of type "
                       "{0}. job_name={1}".format(type(job_name), job_name))
    if not job_name:
      raise ValueError("job_name must not be empty")
  if element_spec is None:
    raise ValueError("element_spec must not be None")
  protocol, address = _parse_service(service)

  dataset = _DataServiceDataset(
      dataset_id=dataset_id,
      processing_mode=processing_mode,
      address=address,
      protocol=protocol,
      job_name=job_name,
      max_outstanding_requests=max_outstanding_requests,
      task_refresh_interval_hint_ms=task_refresh_interval_hint_ms)
  dataset = dataset.map(
      lambda x: compression_ops.uncompress(x, output_spec=element_spec),
      num_parallel_calls=dataset_ops.AUTOTUNE)

  # Disable autosharding for shared jobs.
  if job_name:
    options = dataset_ops.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.OFF
    dataset = dataset.with_options(options)
  return dataset


def _distribute(processing_mode,
                service,
                job_name=None,
                max_outstanding_requests=None,
                task_refresh_interval_hint_ms=None):
  """A transformation that moves dataset processing to the tf.data service.

  This transformation is similar to `distribute`, but supports additional
  parameters which we do not yet want to add to the public Python API.

  Args:
    processing_mode: A string specifying the policy for how data should be
      processed by tf.data workers. Can be either "parallel_epochs" to have
      each tf.data worker process a copy of the dataset, or
      "distributed_epoch" to split a single iteration of the dataset across
      all the workers.
    service: A string indicating how to connect to the tf.data service. The
      string should be in the format "<protocol>://<address>", e.g.
      "grpc://localhost:5000".
    job_name: (Optional.) The name of the job. This argument makes it possible
      for multiple datasets to share the same job. The default behavior is that
      the dataset creates anonymous, exclusively owned jobs.
    max_outstanding_requests: (Optional.) A limit on how many elements may be
      requested at the same time. You can use this option to control the amount
      of memory used, since `distribute` won't use more than `element_size` *
      `max_outstanding_requests` of memory.
    task_refresh_interval_hint_ms: (Optional.) A hint for how often to query the
      dispatcher for task changes.

  Returns:
    Dataset: A `Dataset` of the elements produced by the data service.
  """
  ProcessingMode.validate(processing_mode)

  def _apply_fn(dataset):  # pylint: disable=missing-docstring
    dataset_id = register_dataset(service, dataset)
    return _from_dataset_id(
        processing_mode,
        service,
        dataset_id,
        dataset.element_spec,
        job_name=job_name,
        max_outstanding_requests=max_outstanding_requests,
        task_refresh_interval_hint_ms=task_refresh_interval_hint_ms)

  return _apply_fn


@tf_export("data.experimental.service.distribute")
def distribute(processing_mode,
               service,
               job_name=None,
               max_outstanding_requests=None):
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
  >>> print(sorted(list(dataset.as_numpy_iterator())))
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
  >>> print(sorted(list(dataset.as_numpy_iterator())))
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
  >>> print(sorted(list(dataset.as_numpy_iterator())))
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
      "parallel_epochs", "grpc://localhost:5000", job_name="my_job_name"))
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
    processing_mode: A string specifying the policy for how data should be
      processed by tf.data workers. Can be either "parallel_epochs" to have
      each tf.data worker process a copy of the dataset, or
      "distributed_epoch" to split a single iteration of the dataset across
      all the workers.
    service: A string indicating how to connect to the tf.data service. The
      string should be in the format "protocol://address", e.g.
      "grpc://localhost:5000".
    job_name: (Optional.) The name of the job. This argument makes it possible
      for multiple datasets to share the same job. The default behavior is that
      the dataset creates anonymous, exclusively owned jobs.
    max_outstanding_requests: (Optional.) A limit on how many elements may be
      requested at the same time. You can use this option to control the amount
      of memory used, since `distribute` won't use more than `element_size` *
      `max_outstanding_requests` of memory.

  Returns:
    Dataset: A `Dataset` of the elements produced by the data service.
  """
  return _distribute(
      processing_mode=processing_mode,
      service=service,
      job_name=job_name,
      max_outstanding_requests=max_outstanding_requests)


@tf_export("data.experimental.service.register_dataset")
def register_dataset(service, dataset):
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
  >>> print(list(dataset.as_numpy_iterator()))
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  Args:
    service: A string indicating how to connect to the tf.data service. The
      string should be in the format "protocol://address", e.g.
      "grpc://localhost:5000".
    dataset: A `tf.data.Dataset` to register with the tf.data service.

  Returns:
    A scalar int64 tensor of the registered dataset's id.
  """
  protocol, address = _parse_service(service)
  external_state_policy = dataset.options().experimental_external_state_policy
  if external_state_policy is None:
    external_state_policy = ExternalStatePolicy.WARN

  # Compress the dataset elements to reduce the amount of data that needs to
  # be sent over the network.
  dataset = dataset.map(
      lambda *x: compression_ops.compress(x),
      num_parallel_calls=dataset_ops.AUTOTUNE)
  dataset = dataset.prefetch(dataset_ops.AUTOTUNE)
  # Apply options so that the dataset executed in the tf.data service will
  # be optimized and support autotuning.
  dataset = dataset._apply_options()  # pylint: disable=protected-access

  dataset_id = gen_experimental_dataset_ops.register_dataset(
      dataset._variant_tensor,  # pylint: disable=protected-access
      address=address,
      protocol=protocol,
      external_state_policy=external_state_policy.value)

  return dataset_id


@tf_export("data.experimental.service.from_dataset_id")
def from_dataset_id(processing_mode,
                    service,
                    dataset_id,
                    element_spec=None,
                    job_name=None,
                    max_outstanding_requests=None):
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
  >>> print(list(dataset.as_numpy_iterator()))
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  Args:
    processing_mode: A string specifying the policy for how data should be
      processed by tf.data workers. Can be either "parallel_epochs" to have
      each tf.data worker process a copy of the dataset, or
      "distributed_epoch" to split a single iteration of the dataset across
      all the workers.
    service: A string indicating how to connect to the tf.data service. The
      string should be in the format "protocol://address", e.g.
      "grpc://localhost:5000".
    dataset_id: The id of the dataset to read from. This id is returned by
      `register_dataset` when the dataset is registered with the tf.data
      service.
    element_spec: A nested structure of `tf.TypeSpec`s representing the type of
      elements produced by the dataset. Use `tf.data.Dataset.element_spec` to
      see the element spec for a given dataset.
    job_name: (Optional.) The name of the job. This argument makes it possible
      for multiple datasets to share the same job. The default behavior is that
      the dataset creates anonymous, exclusively owned jobs.
    max_outstanding_requests: (Optional.) A limit on how many elements may be
      requested at the same time. You can use this option to control the amount
      of memory used, since `distribute` won't use more than `element_size` *
      `max_outstanding_requests` of memory.

  Returns:
    A `tf.data.Dataset` which reads from the tf.data service.
  """
  return _from_dataset_id(
      processing_mode=processing_mode,
      service=service,
      dataset_id=dataset_id,
      element_spec=element_spec,
      job_name=job_name,
      max_outstanding_requests=max_outstanding_requests)
