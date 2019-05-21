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

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_shape
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

    self._structure = input_dataset._element_structure  # pylint: disable=protected-access
    variant_tensor = ged_ops.experimental_auto_shard_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        num_workers=num_workers,
        index=index,
        **dataset_ops.flat_structure(self))
    super(_AutoShardDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def _element_structure(self):
    return self._structure


def _AutoShardDatasetV1(input_dataset, num_workers, index):
  return dataset_ops.DatasetV1Adapter(
      _AutoShardDataset(input_dataset, num_workers, index))


class _RebatchDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that divides the batch size by `num_workers`."""

  def __init__(self, input_dataset, num_workers):
    self._input_dataset = input_dataset

    def recalculate_output_shapes(output_shapes):
      """Recalculates the output_shapes after dividing it by num_workers."""
      if len(output_shapes) < 1:
        raise ValueError("Input shape should have at least one dimension.")
      if (tensor_shape.dimension_value(output_shapes[0]) and
          tensor_shape.dimension_value(output_shapes[0]) % num_workers != 0):
        raise errors.InvalidArgumentError(
            None, None,
            "First dim of input shape: %d is not divisible by num_workers: %d" %
            (output_shapes[0], num_workers))
      output_dims = [d for d in output_shapes.dims]
      output_dims[0] = output_dims[0] // num_workers
      return tensor_shape.TensorShape(output_dims)

    input_types = dataset_ops.get_legacy_output_types(self._input_dataset)
    input_shapes = dataset_ops.get_legacy_output_shapes(self._input_dataset)
    input_classes = dataset_ops.get_legacy_output_classes(self._input_dataset)
    output_shapes = nest.map_structure(recalculate_output_shapes, input_shapes)

    self._structure = structure.convert_legacy_structure(
        input_types, output_shapes, input_classes)
    variant_tensor = ged_ops.experimental_rebatch_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        num_workers=num_workers,
        **dataset_ops.flat_structure(self))
    super(_RebatchDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def _element_structure(self):
    return self._structure


_AutoShardDatasetV1.__doc__ = _AutoShardDataset.__doc__
