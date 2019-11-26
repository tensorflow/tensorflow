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


COMPRESSION_GZIP = "GZIP"
COMPRESSION_SNAPPY = "SNAPPY"
COMPRESSION_NONE = None


class _SnapshotDataset(dataset_ops.UnaryUnchangedStructureDataset):
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
               seed=None):

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

    self._seed, self._seed2 = random_seed.get_seed(seed)

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
        **self._flat_structure)
    super(_SnapshotDataset, self).__init__(input_dataset, variant_tensor)


def snapshot(path,
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
             seed=None):
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
    pending_snapshot_expiry_seconds: How long to wait (in seconds) before
      the snapshot op considers a previously unfinished snapshot to be stale.
    num_reader_threads: Number of threads to parallelize reading from snapshot.
      Especially useful if compression is turned on since the decompression
      operation tends to be intensive. Defaults to 1. If > 1, then this might
      introduce non-determinism i.e. the order in which the elements are
      read from the snapshot are different from the order they're written.
    reader_buffer_size: Maximum number of elements we can prefetch reading from
      the snapshot. Defaults to 1. Increasing this might improve performance
      but will increase memory consumption.
    num_writer_threads: Number of threads to parallelize writing from snapshot.
      We'll open up `num_writer_threads` files and write to them in parallel.
      Especially useful if compression is turned on since the compression
      operation tends to be intensive. Defaults to 1. If > 1, then this might
      introduce non-determinism i.e. the order in which the elements are
      read from the upstream iterator are different from the order they're
      written.
    writer_buffer_size: Maximum number of pipeline elements to fill up the
      buffer before writing them out using `num_writer_threads`.
    shuffle_on_read: If this is True, then the order in which examples are
      produced when reading from a snapshot will be random. Defaults to False.
    seed: If seed is set, the random number generator is seeded by the given
      seed. Otherwise, it is seeded by a random seed.
  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    return _SnapshotDataset(dataset, path, compression, reader_path_prefix,
                            writer_path_prefix, shard_size_bytes,
                            pending_snapshot_expiry_seconds, num_reader_threads,
                            reader_buffer_size, num_writer_threads,
                            writer_buffer_size, shuffle_on_read, seed)

  return _apply_fn
