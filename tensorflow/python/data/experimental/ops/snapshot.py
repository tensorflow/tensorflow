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
"""Dataset snapshot and related functionality."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

COMPRESSION_GZIP = "GZIP"
COMPRESSION_SNAPPY = "SNAPPY"
COMPRESSION_NONE = None


class _LegacySnapshotDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A Dataset that captures a snapshot or reads from a snapshot."""

  def __init__(self,
               input_dataset,
               path,
               compression=None,
               reader_path_prefix=None,
               writer_path_prefix=None,
               shard_size_bytes=None,
               pending_snapshot_expiry_seconds=None,
               num_reader_threads=None,
               reader_buffer_size=None,
               num_writer_threads=None,
               writer_buffer_size=None,
               shuffle_on_read=None,
               shuffle_seed=None,
               mode=None,
               snapshot_name=None):

    self._compression = compression if compression is not None else ""
    self._reader_path_prefix = (
        reader_path_prefix if reader_path_prefix is not None else "")
    self._writer_path_prefix = (
        writer_path_prefix if writer_path_prefix is not None else "")
    self._shard_size_bytes = (
        shard_size_bytes if shard_size_bytes is not None else -1)
    self._pending_snapshot_expiry_seconds = (
        pending_snapshot_expiry_seconds
        if pending_snapshot_expiry_seconds is not None else -1)
    self._num_reader_threads = (
        num_reader_threads if num_reader_threads is not None else -1)
    self._reader_buffer_size = (
        reader_buffer_size if reader_buffer_size is not None else -1)
    self._num_writer_threads = (
        num_writer_threads if num_writer_threads is not None else -1)
    self._writer_buffer_size = (
        writer_buffer_size if writer_buffer_size is not None else -1)
    self._shuffle_on_read = (
        shuffle_on_read if shuffle_on_read is not None else False)
    self._mode = (mode if mode is not None else "auto")
    self._snapshot_name = (snapshot_name if snapshot_name is not None else "")

    self._seed, self._seed2 = random_seed.get_seed(shuffle_seed)

    self._input_dataset = input_dataset
    self._path = ops.convert_to_tensor(path, dtype=dtypes.string, name="path")

    variant_tensor = ged_ops.snapshot_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        path=self._path,
        compression=self._compression,
        reader_path_prefix=self._reader_path_prefix,
        writer_path_prefix=self._writer_path_prefix,
        shard_size_bytes=self._shard_size_bytes,
        pending_snapshot_expiry_seconds=self._pending_snapshot_expiry_seconds,
        num_reader_threads=self._num_reader_threads,
        reader_buffer_size=self._reader_buffer_size,
        num_writer_threads=self._num_writer_threads,
        writer_buffer_size=self._writer_buffer_size,
        shuffle_on_read=self._shuffle_on_read,
        seed=self._seed,
        seed2=self._seed2,
        mode=self._mode,
        snapshot_name=self._snapshot_name,
        **self._flat_structure)

    super(_LegacySnapshotDataset, self).__init__(input_dataset, variant_tensor)


@deprecation.deprecated(
    None, "Use `tf.data.experimental.snapshot(...)` instead.")
def legacy_snapshot(path,
                    compression=None,
                    reader_path_prefix=None,
                    writer_path_prefix=None,
                    shard_size_bytes=None,
                    pending_snapshot_expiry_seconds=None,
                    num_reader_threads=None,
                    reader_buffer_size=None,
                    num_writer_threads=None,
                    writer_buffer_size=None,
                    shuffle_on_read=None,
                    shuffle_seed=None,
                    mode=None,
                    snapshot_name=None):
  """Writes to/reads from a snapshot of a dataset.

  This function attempts to determine whether a valid snapshot exists at the
  `path`, and reads from the snapshot if so. If not, it will run the
  preprocessing pipeline as usual, and write out a snapshot of the data
  processed for future use.

  Args:
    path: A directory where we want to save our snapshots and/or read from a
      previously saved snapshot.
    compression: The type of compression to apply to the Dataset. Currently
      supports "GZIP" or None. Defaults to None (no compression).
    reader_path_prefix: A prefix to add to the path when reading from snapshots.
      Defaults to None.
    writer_path_prefix: A prefix to add to the path when writing to snapshots.
      Defaults to None.
    shard_size_bytes: The size of each shard to be written by the snapshot
      dataset op. Defaults to 10 GiB.
    pending_snapshot_expiry_seconds: How long to wait (in seconds) before the
      snapshot op considers a previously unfinished snapshot to be stale.
    num_reader_threads: Number of threads to parallelize reading from snapshot.
      Especially useful if compression is turned on since the decompression
      operation tends to be intensive. Defaults to 1. If > 1, then this might
      introduce non-determinism i.e. the order in which the elements are read
      from the snapshot are different from the order they're written.
    reader_buffer_size: Maximum number of elements we can prefetch reading from
      the snapshot. Defaults to 1. Increasing this might improve performance but
      will increase memory consumption.
    num_writer_threads: Number of threads to parallelize writing from snapshot.
      We'll open up `num_writer_threads` files and write to them in parallel.
      Especially useful if compression is turned on since the compression
      operation tends to be intensive. Defaults to 1. If > 1, then this might
      introduce non-determinism i.e. the order in which the elements are read
      from the upstream iterator are different from the order they're written.
    writer_buffer_size: Maximum number of pipeline elements to fill up the
      buffer before writing them out using `num_writer_threads`.
    shuffle_on_read: If this is True, then the order in which examples are
      produced when reading from a snapshot will be random. Defaults to False.
    shuffle_seed: Optional. If shuffle_seed is set, the random number generator
      used for shuffling (when shuffle_on_read is turned on) is seeded by the
      given seed. Otherwise, it is seeded by a random seed that differs for
      every run.
    mode: The mode at which snapshot should operate. Valid options are "auto",
      "read", "write", and "passthrough". The default mode is "auto", where the
      snapshot op will automatically determine what mode to operate in.
    snapshot_name: If set, use the supplied string as a named snapshot name
      instead of introspecting the data pipeline and automatically generating a
      unique identifier for the snapshot.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    return _LegacySnapshotDataset(
        input_dataset=dataset,
        path=path,
        compression=compression,
        reader_path_prefix=reader_path_prefix,
        writer_path_prefix=writer_path_prefix,
        shard_size_bytes=shard_size_bytes,
        pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds,
        num_reader_threads=num_reader_threads,
        reader_buffer_size=reader_buffer_size,
        num_writer_threads=num_writer_threads,
        writer_buffer_size=writer_buffer_size,
        shuffle_on_read=shuffle_on_read,
        shuffle_seed=shuffle_seed,
        mode=mode,
        snapshot_name=snapshot_name)

  return _apply_fn


@deprecation.deprecated(None, "Use `tf.data.Dataset.snapshot(...)`.")
@tf_export("data.experimental.snapshot")
def snapshot(path, compression="AUTO", reader_func=None, shard_func=None):
  """API to persist the output of the input dataset.

  The snapshot API allows users to transparently persist the output of their
  preprocessing pipeline to disk, and materialize the pre-processed data on a
  different training run.

  This API enables repeated preprocessing steps to be consolidated, and allows
  re-use of already processed data, trading off disk storage and network
  bandwidth for freeing up more valuable CPU resources and accelerator compute
  time.

  https://github.com/tensorflow/community/blob/master/rfcs/20200107-tf-data-snapshot.md
  has detailed design documentation of this feature.

  Users can specify various options to control the behavior of snapshot,
  including how snapshots are read from and written to by passing in
  user-defined functions to the `reader_func` and `shard_func` parameters.

  `shard_func` is a user specified function that maps input elements to snapshot
  shards.

  Users may want to specify this function to control how snapshot files should
  be written to disk. Below is an example of how a potential shard_func could
  be written.

  ```python
  dataset = ...
  dataset = dataset.enumerate()
  dataset = dataset.apply(tf.data.experimental.snapshot("/path/to/snapshot/dir",
      shard_func=lambda x, y: x % NUM_SHARDS, ...))
  dataset = dataset.map(lambda x, y: y)
  ```

  `reader_func` is a user specified function that accepts a single argument:
  (1) a Dataset of Datasets, each representing a "split" of elements of the
  original dataset. The cardinality of the input dataset matches the
  number of the shards specified in the `shard_func` (see above). The function
  should return a Dataset of elements of the original dataset.

  Users may want specify this function to control how snapshot files should be
  read from disk, including the amount of shuffling and parallelism.

  Here is an example of a standard reader function a user can define. This
  function enables both dataset shuffling and parallel reading of datasets:

  ```python
  def user_reader_func(datasets):
    # shuffle the datasets splits
    datasets = datasets.shuffle(NUM_CORES)
    # read datasets in parallel and interleave their elements
    return datasets.interleave(lambda x: x, num_parallel_calls=AUTOTUNE)

  dataset = dataset.apply(tf.data.experimental.snapshot("/path/to/snapshot/dir",
      reader_func=user_reader_func))
  ```

  By default, snapshot parallelizes reads by the number of cores available on
  the system, but will not attempt to shuffle the data.

  Args:
    path: Required. A directory to use for storing / loading the snapshot to /
      from.
    compression: Optional. The type of compression to apply to the snapshot
      written to disk. Supported options are `GZIP`, `SNAPPY`, `AUTO` or None.
      Defaults to AUTO, which attempts to pick an appropriate compression
      algorithm for the dataset.
    reader_func: Optional. A function to control how to read data from snapshot
      shards.
    shard_func: Optional. A function to control how to shard data when writing a
      snapshot.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    """Actual dataset transformation."""
    return dataset.snapshot(
        path=path,
        compression=compression,
        reader_func=reader_func,
        shard_func=shard_func)

  return _apply_fn
