# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Unique element dataset transformations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.data.python.ops import contrib_op_loader  # pylint: disable=unused-import
from tensorflow.contrib.data.python.ops import gen_dataset_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework.tensor_shape import TensorShape


def unordered_merge(datasets):
  """Creates a `Dataset` by merging the given datasets without garantee of data order.

  The input `datasets` must be an iterable of same types and shapes of datasets.
  This method merges input datasets without garantee of the data order.
  For example:

  ```python
  # NOTE: The following examples use `{ ... }` to represent the
  # contents of a dataset.
  a = { 1, 2, 3 }
  b = { 4, 5, 6 }
  c = { 13, 14 }
  d = { (7, 8), (9, 10), (11, 12) }

  # The `datasets` argument may contain an arbitrary number of
  # datasets having same type and shape.
  unorderd_merge([a, b]) == { 1, 4, 2, 5, 3, 6 }
  unorderd_merge([a, b, c]) == { 1, 4, 13, 2, 5, 3, 6, 14 }

  # NOTE: the output order might be different

  # The shapes and types in `datasets` argument must be the same.
  unorderd_merge([a, b, d]) ==> TypeError

  # sample usage:
  dataset = tf.data.Dataset.from_tensor_slices(tensors)
  datasets = [dataset.shard(10, i) for i in range(10)]
  datasets = [dataset.apply(group_by_window(key_func, reduce_func, window_size)
              for dataset in datasets]
  dataset = tf.contrib.data.unorderd_merge(datasets)
  ```

  Args:
    datasets: An iterable(such as list or tuple) of datasets.

  Returns:
    A merged `Dataset`.
  """
  return UnorderedMergeDataset(datasets)

def _shapes_to_list(shapes):
  if TensorShape == type(shapes):
    return shapes.as_list()
  flat_shapes = nest.flatten(shapes)
  list_shapes = [_shapes_to_list(s) for s in flat_shapes]
  return nest.pack_sequence_as(shapes, list_shapes)

class UnorderedMergeDataset(dataset_ops.Dataset):
  """A merged `Dataset` of its input datasets without garantee of data order."""

  def __init__(self, datasets):
    """See `unordered_merge()` for details."""
    super(UnorderedMergeDataset, self).__init__()
    self._datasets = list(datasets)

    for ds in self._datasets:
      if not isinstance(ds, dataset_ops.Dataset):
        message = ("The argument to `unordered_merge()` "
                   "must be an iterable of datasets.")
        raise TypeError(message)

    shapes0 = _shapes_to_list(self._datasets[0].output_shapes)
    types0 = self._datasets[0].output_types

    for ds in self._datasets:
      current_shapes = _shapes_to_list(ds.output_shapes)
      if shapes0 != current_shapes:
        message = ("The shapes of `datasets` {}, {} must be same."
                   .format(shapes0, current_shapes))
        raise TypeError(message)
      current_types = ds.output_types
      if types0 != current_types:
        message = ("The types of `datasets` {}, {} must be same."
                   .format(types0, current_types))
        raise TypeError(message)

  def _as_variant_tensor(self):
    # pylint: disable=protected-access
    return gen_dataset_ops.unordered_merge_dataset(
        [ds._as_variant_tensor() for ds in self._datasets],
        output_shapes=nest.flatten(self._datasets[0].output_shapes),
        output_types=nest.flatten(self._datasets[0].output_types))
    # pylint: enable=protected-access

  @property
  def output_classes(self):
    return self._datasets[0].output_classes

  @property
  def output_shapes(self):
    return self._datasets[0].output_shapes

  @property
  def output_types(self):
    return self._datasets[0].output_types
