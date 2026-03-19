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
"""Distributed saving of a dataset to disk."""

from typing import Optional

from tensorflow.core.protobuf import snapshot_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops
# TODO(b/238903802): Use TypeSpec serialization methods directly.
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.distributed_save")
def distributed_save(
    dataset: dataset_ops.Dataset,
    path: str,
    data_service_address: str,
    compression: str = "AUTO",
) -> Optional[ops.OperationType]:
  """Initiates the process of saving a dataset to disk using tf.data service.

  The op uses tf.data service
  (https://www.tensorflow.org/api_docs/python/tf/data/experimental/service) to
  write a dataset snapshot. Returns immediately after submitting the request.
  Does not wait for the snapshot to be finished. Requires that the tf.data
  service run a fixed number of worker replicas.

  To load the snapshot, users may optionally pass `wait=True` to
  `tf.data.Dataset.load` so it can read snapshots as they are being written.

  Example usage:

  >>> import os
  >>> import tempfile

  >>> # Runs tf.data service.
  >>> tempdir = tempfile.gettempdir()
  >>> dispatcher = tf.data.experimental.service.DispatchServer(
  ...     tf.data.experimental.service.DispatcherConfig(
  ...         fault_tolerant_mode=True,
  ...         work_dir=os.path.join(tempdir, "work_dir")))
  >>> dispatcher_address = dispatcher.target.split("://")[1]
  >>> worker = tf.data.experimental.service.WorkerServer(
  ...     tf.data.experimental.service.WorkerConfig(
  ...         dispatcher_address=dispatcher_address))

  >>> # Writes dataset snapshot.
  >>> path = os.path.join(tempdir, "dataset_snapshot")
  >>> dataset = tf.data.Dataset.range(1)
  >>> tf.data.experimental.distributed_save(dataset, path, dispatcher_address)

  >>> # Loads a dataset snapshot.
  >>> loaded_dataset = tf.data.Dataset.load(path, wait=True)
  >>> for elem in loaded_dataset:
  ...   print(elem)
  tf.Tensor(0, shape=(), dtype=int64)

  Args:
    dataset: The `tf.data.Dataset` to save.
    path: The directory path to save the dataset. Requires that:
      - The directory does not exist and will create the directory.
      - The file system supports atomic move (rename).
    data_service_address: tf.data service dispatcher address.
    compression: (Optional.) Whether and how to compress the `dataset` snapshot.
      If `"AUTO"`, the tf.data runtime decides which algorithm to use. If
      `"GZIP"` or `"SNAPPY"`, that specific algorithm is used.  If `None`, the
      `dataset` snapshot is not compressed.

  Returns:
    An operation which when executed performs the distributed save.

  Raises:
    ValueError: If `dispatcher_address` is invalid.
    tf.errors.AlreadyExistsError: If the snapshot has already started or has
      finished.
    tf.errors.FailedPreconditionError: If the file system does not support
      atomic move (rename).
    tf.errors.InvalidArgumentError: If tf.data service is not running in the
      fault tolerant mode.
  """
  if not isinstance(data_service_address, str):
    raise ValueError("`data_service_address` must be a string, but is a "
                     f"{type(data_service_address)} ({data_service_address}")
  if not data_service_address:
    raise ValueError("`data_service_address` must not be empty")

  metadata = snapshot_pb2.DistributedSnapshotMetadata(
      element_spec=nested_structure_coder.encode_structure(
          dataset.element_spec).SerializeToString(),
      compression=compression)

  return gen_experimental_dataset_ops.distributed_save(
      dataset._variant_tensor,  # pylint: disable=protected-access
      directory=path,
      address=data_service_address,
      metadata=metadata.SerializeToString())
