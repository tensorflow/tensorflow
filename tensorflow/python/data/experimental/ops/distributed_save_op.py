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

from tensorflow.core.protobuf import snapshot_pb2
from tensorflow.python.ops import gen_experimental_dataset_ops
# TODO(b/238903802): Use TypeSpec serialization methods directly.
from tensorflow.python.saved_model import nested_structure_coder


# TODO(b/250921378): Add example to docstring and export to TF API.
def distributed_save(dataset,
                     directory,
                     dispatcher_address,
                     compression="AUTO"):
  """Initiates the process of distributedly saving a dataset to disk.

  Args:
    dataset: The `tf.data.Dataset` to save.
    directory: A string indicating the directory to which to save `dataset`.
    dispatcher_address: A string indicating the address of the dispatcher for
      the tf.data service instance used to save `dataset`.
    compression: (Optional.) A string indicating whether and how to compress the
      `dataset` materialization.  If `"AUTO"`, the tf.data runtime decides which
      algorithm to use.  If `"GZIP"` or `"SNAPPY"`, that specific algorithm is
      used.  If `None`, the `dataset` materialization is not compressed.

  Returns:
    `None`.

  Raises:
    ValueError: If `dispatcher_address` is invalid.
  """
  if not isinstance(dispatcher_address, str):
    raise ValueError("`dispatcher_address` must be a string, but is a "
                     f"{type(dispatcher_address)} ({dispatcher_address}")
  if not dispatcher_address:
    raise ValueError("`dispatcher_address` must not be empty")

  metadata = snapshot_pb2.DistributedSnapshotMetadata(
      element_spec=nested_structure_coder.encode_structure(
          dataset.element_spec).SerializeToString(),
      compression=compression,
  )

  return gen_experimental_dataset_ops.distributed_save(
      dataset._variant_tensor,  # pylint: disable=protected-access
      directory=directory,
      address=dispatcher_address,
      metadata=metadata.SerializeToString(),
  )
