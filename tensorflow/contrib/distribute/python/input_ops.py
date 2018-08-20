# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Input-pipeline utilities for Distribution strategies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import readers
from tensorflow.python.data.util import nest
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging

# TODO(priyag): Any other reader datasets to consider here?
_READER_DATASET_OPS = [
    "TextLineDataset",
    "TFRecordDataset",
    "FixedLengthRecordDataset"
]


# pylint: disable=protected-access
def auto_shard_dataset(dataset, num_shards, index):
  """Shard the input pipeline by sharding the underlying list of files.

  Args:
    dataset: A `tf.data.Dataset` instance, typically the result of a bunch of
      dataset transformations.
    num_shards: A `tf.int64` scalar `tf.Tensor`, representing the number of
        shards operating in parallel. Same usage as in `Dataset.shard`.
    index: A `tf.int64` scalar `tf.Tensor`, representing the worker index.
      Same usage as in `Dataset.shard`.

  Returns:
    A modified `Dataset` obtained by updating the pipeline sharded by the
    files.

  Raises:
    NotImplementedError: If we cannot automatically determine a good way to
      shard the input dataset.
  """

  # TODO(priyag): Clone datasets instead of updating in place, similar to the
  # clone method for TFRecordDataset.
  def _auto_shard_impl(dataset, found_reader_op):
    """Recursive implementation of auto sharding."""

    if not found_reader_op:
      # TODO(priyag): Make this check more robust by enforcing some common
      # property on reader datasets.
      if (isinstance(dataset, readers.TextLineDataset) or
          isinstance(dataset, readers.FixedLengthRecordDataset)):
        filenames_tensor = dataset._filenames
        num_files = array_ops.size(filenames_tensor)
        sharded_filenames_tensor = array_ops.gather(
            filenames_tensor, math_ops.range(index, num_files, num_shards))
        dataset._filenames = sharded_filenames_tensor
        return dataset
      elif isinstance(dataset, readers.TFRecordDataset):
        # `TFRecordDataset` needs to be handled separately than other readers
        # because it converts filenames to a dataset first. Also, we clone it
        # instead of updating in place because it has special logic in the
        # constructor. Eventually we will change all cases to clone datasets
        # instead of updating in-place.
        return dataset._clone(
            filenames=dataset._filenames.shard(num_shards, index))
      elif hasattr(dataset, "_map_func"):
        # TODO(priyag): Make this check more robust by enforcing some common
        # property on all map/flatmap/interleave datasets.
        map_func_def = dataset._map_func.definition
        for node in map_func_def.node_def:
          if node.op in _READER_DATASET_OPS:
            found_reader_op = True
            break
          elif node.op == "FlatMapDataset":
            # TODO(priyag): Should this check for other map datasets? Should it
            # be recursive? It is too specific to implementation of
            # TFRecordDataset right now.
            nested_func_name = node.attr["f"].func.name
            nested_func = ops.get_default_graph()._functions[nested_func_name]
            for nested_node in nested_func.definition.node_def:
              if nested_node.op in _READER_DATASET_OPS:
                found_reader_op = True
                break
            if found_reader_op:
              break
        if found_reader_op:
          dataset._input_dataset = _auto_shard_impl(
              dataset._input_dataset, found_reader_op)
          return dataset

    # TODO(priyag): Make _input_dataset(s) a common property of all datasets to
    # make this check more robust.
    if hasattr(dataset, "_input_dataset"):
      dataset._input_dataset = _auto_shard_impl(
          dataset._input_dataset, found_reader_op)
      if hasattr(dataset, "_dataset_to_concatenate"):
        # Special case for `ConcatentateDataset`. We want to shard all input
        # datasets.
        dataset._dataset_to_concatenate = _auto_shard_impl(
            dataset._dataset_to_concatenate, found_reader_op)
      return dataset

    if hasattr(dataset, "_datasets"):
      # Special case for `ZipDataset`.
      dataset._datasets = nest.pack_sequence_as(dataset._datasets, [
          _auto_shard_impl(ds, found_reader_op)
          for ds in nest.flatten(dataset._datasets)
      ])
      return dataset

    if not found_reader_op:
      tf_logging.warn(
          "Could not find a standard reader in the input pipeline"
          "(one of TextLineDataset, TFRecordDataset, FixedLengthRecordDataset)."
          "Falling back to sharding the dataset anyway. Please verify"
          "correctness of auto-sharding for your input.")

    # TODO(priyag): What do we want to do if the number of filenames is
    # uneven in the number of shards? By default, this will just return as
    # many items it can before throwing OutOfRangeError.
    # TODO(priyag): This will shard the filenames before any shuffling of the
    # filename dataset. It might be desirable to shard after shuffling
    # filenames? If so, how do we achieve that?
    return dataset.shard(num_shards, index)

  return _auto_shard_impl(dataset=dataset, found_reader_op=False)
