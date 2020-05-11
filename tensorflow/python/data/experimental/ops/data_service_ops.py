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
from tensorflow.python.data.experimental.ops.distribute_options import ExternalStatePolicy
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops


class ProcessingMode(object):
  PARALLEL_EPOCHS = "parallel_epochs"

  @staticmethod
  def validate(mode):
    """Raises a ValueError if the given object is not a valid processing mode."""
    valid_modes = [ProcessingMode.PARALLEL_EPOCHS]
    if mode not in valid_modes:
      raise ValueError(
          "{0} is not a valid processing mode. Valid modes: {1}".format(
              mode, valid_modes))


class _DataServiceDatasetV2(dataset_ops.DatasetSource):
  """A `Dataset` that reads elements from the tf.data service."""

  def __init__(self,
               input_dataset,
               dataset_id,
               processing_mode,
               address,
               protocol,
               job_name=None,
               max_outstanding_requests=None,
               task_refresh_interval_hint_ms=None):
    """Constructs a _DataServiceDatasetV2.

    Args:
      input_dataset: The input dataset, which should be registered with the
        tf.data service under `dataset_id`.
      dataset_id: The dataset id for the dataset to read from.
      processing_mode: A string specifying the policy for how data should be
        processed by tf.data workers. Currently, the only supported value is
        "parallel_epochs".
      address: The tf.data service address, e.g. "localhost:5000".
      protocol: The protocol to use for communicating with the tf.data service,
        e.g. "grpc".
      job_name: (Optional.) The name of the job. This argument makes it
        possible for multiple datasets to share the same job. The default
        behavior is that the dataset creates anonymous, exclusively owned jobs.
      max_outstanding_requests: (Optional.) A limit on how many elements may be
        requested at the same time. You can use this option to control the
        amount of memory used, since `distribute` won't use more than
        `element_size` * `max_outstanding_requests` of memory.
      task_refresh_interval_hint_ms: (Optional.) A hint for how often to query
        the master for task changes.
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
    self._element_spec = input_dataset.element_spec

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
  def __init__(self, input_dataset, dataset_id, processing_mode, address,
               protocol, job_name, max_outstanding_requests,
               task_refresh_interval_hint_ms):

    self._wrapped = _DataServiceDatasetV2(
        input_dataset=input_dataset,
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
      processed by tf.data workers. Currently, the only supported value is
      "parallel_epochs".
    service: A string indicating how to connect to the tf.data service. The
      string should be in the format <protocol>://<address>, e.g.
      grpc://localhost:5000.
    job_name: (Optional.) The name of the job. This argument makes it
      possible for multiple datasets to share the same job. The default behavior
      is that the dataset creates anonymous, exclusively owned jobs.
    max_outstanding_requests: (Optional.) A limit on how many elements may be
      requested at the same time. You can use this option to control the amount
      of memory used, since `distribute` won't use more than `element_size` *
      `max_outstanding_requests` of memory.
    task_refresh_interval_hint_ms: (Optional.) A hint for how often to query the
      master for task changes.

  Returns:
    Dataset: A `Dataset` of the elements produced by the data service.
  """
  ProcessingMode.validate(processing_mode)
  if job_name is not None:
    if not isinstance(job_name, six.string_types):
      raise ValueError("job_name must be a string, but job_name was of type "
                       "{0}. job_name={1}".format(type(job_name), job_name))
    if not job_name:
      raise ValueError("job_name must not be empty")
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
  protocol, address = parts
  address = ops.convert_to_tensor(address, dtype=dtypes.string, name="address")
  protocol = ops.convert_to_tensor(
      protocol, dtype=dtypes.string, name="protocol")

  def _apply_fn(dataset):
    external_state_policy = dataset.options().experimental_external_state_policy
    if external_state_policy is None:
      external_state_policy = ExternalStatePolicy.WARN
    dataset_id = gen_experimental_dataset_ops.register_dataset(
        dataset._variant_tensor,  # pylint: disable=protected-access
        address=address,
        protocol=protocol,
        external_state_policy=external_state_policy.value)
    return _DataServiceDataset(
        input_dataset=dataset,
        dataset_id=dataset_id,
        processing_mode=processing_mode,
        address=address,
        protocol=protocol,
        job_name=job_name,
        max_outstanding_requests=max_outstanding_requests,
        task_refresh_interval_hint_ms=task_refresh_interval_hint_ms)

  return _apply_fn


def distribute(processing_mode,
               service,
               job_name=None,
               max_outstanding_requests=None):
  """A transformation that moves dataset processing to the tf.data service.

  When you iterate over a dataset containing the `distribute` transformation,
  the tf.data service creates a "job" which produces data for the dataset
  iteration.

  The `processing_mode` argument controls what data is produced by a tf.data
  service job. Currently, the only supported mode is "parallel_epochs".

  processing_mode="parallel_epochs" means that multiple tf.data workers will
  iterate through the dataset in parallel, each producing all elements of the
  dataset. For example, if the dataset contains {0, 1, 2}, every tf.data worker
  used for execution will produce {0, 1, 2}. If there are 3 workers, the job
  will produce the elements {0, 0, 0, 1, 1, 1, 2, 2, 2} (though not necessarily
  in that order). To account for this, it is recommended to randomly shuffle
  your dataset, so that different tf.data workers will iterate through the
  dataset in different orders.

  In the future, there will be additional processing modes. For example,
  a "one_epoch" mode which partitions the dataset across the tf.data
  workers, so that the consumers see each element of the dataset only once.

  ```
  dataset = tf.data.Dataset.range(5)
  dataset = dataset.map(lambda x: x*x)
  dataset = dataset.apply(
      tf.data.experimental.service.distribute("parallel_epochs",
                                              "grpc://dataservice:5000"))
  dataset = dataset.map(lambda x: x+1)

  for element in dataset:
    print(element)  # prints { 1, 2, 5, 10, 17 }
  ```

  In the above example, the first two lines (before the call to `distribute`)
  will be executed on tf.data workers, and the elements provided over
  RPC. The remaining transformations (after the call to `distribute`) will be
  executed locally.

  The `job_name` argument allows jobs to be shared across multiple
  datasets. Instead of each dataset creating its own job, all datasets with the
  same `job_name` will consume from the same job. A new job will
  be created for each iteration of the dataset (with each repetition of
  `Dataset.repeat` counting as a new iteration). The following example
  demonstrates shared iteration, with the assumption that the tf.data service is
  running with a single worker.

  ```
  range5_dataset = tf.data.Dataset.range(5)
  dataset1 = range5_dataset.apply(tf.data.experimental.service.distribute(
      "parallel_epochs", "my_job_name", "grpc://dataservice:5000"))
  dataset2 = range5_dataset.apply(tf.data.experimental.service.distribute(
      "parallel_epochs", "my_job_name", "grpc://dataservice:5000"))
  iter_1_1 = iter(dataset1)
  iter_1_2 = iter(dataset1)
  iter_2_1 = iter(dataset2)
  iter_2_2 = iter(dataset2)
  print(next(iter_1_1))  # Prints "0"
  # iter_1_2 consumes from the same job as iter_1_1
  print(next(iter_1_2))  # Prints "1"
  # iter_2_1 consumes from a new job
  print(next(iter_2_1))  # Prints "0"
  # iter_2_2 consumes from the same job as iter_2_1
  print(next(iter_2_2))  # Prints "1"
  ```

  Args:
    processing_mode: A string specifying the policy for how data should be
      processed by tf.data workers. Currently, the only supported value is
      "parallel_epochs".
    service: A string indicating how to connect to the tf.data service. The
      string should be in the format <protocol>://<address>, e.g.
      grpc://localhost:5000.
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
