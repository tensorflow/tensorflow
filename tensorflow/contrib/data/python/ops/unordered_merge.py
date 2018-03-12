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


def unordered_merge(datasets):
  """Creates a `Dataset` by zipping together the given datasets.

  This method has similar semantics to the built-in `zip()` function
  in Python, with the main difference being that the `datasets`
  argument can be an arbitrary nested structure of `Dataset` objects.
  For example:

  ```python
  # NOTE: The following examples use `{ ... }` to represent the
  # contents of a dataset.
  a = { 1, 2, 3 }
  b = { 4, 5, 6 }
  c = { (7, 8), (9, 10), (11, 12) }
  d = { 13, 14 }

  # The nested structure of the `datasets` argument determines the
  # structure of elements in the resulting dataset.
  Dataset.zip((a, b)) == { (1, 4), (2, 5), (3, 6) }
  Dataset.zip((b, a)) == { (4, 1), (5, 2), (6, 3) }

  # The `datasets` argument may contain an arbitrary number of
  # datasets.
  Dataset.zip((a, b, c)) == { (1, 4, (7, 8)),
                              (2, 5, (9, 10)),
                              (3, 6, (11, 12)) }

  # The number of elements in the resulting dataset is the same as
  # the size of the smallest dataset in `datasets`.
  Dataset.zip((a, d)) == { (1, 13), (2, 14) }
  ```

  Args:
    datasets: A nested structure of datasets.

  Returns:
    A `Dataset`.
  """
  return UnorderedMergeDataset(datasets)


class UnorderedMergeDataset(dataset_ops.Dataset):
  """A `Dataset` that merges its inputs together with unordered."""

  def __init__(self, datasets):
    """See `Dataset.unordered_merge()` for details."""
    super(UnorderedMergeDataset, self).__init__()
    for ds in nest.flatten(datasets):
      if not isinstance(ds, dataset_ops.Dataset):
        if isinstance(ds, list):
          message = ("The argument to `Dataset.unordered_merge()` must be a nested "
                     "structure of `Dataset` objects. Nested structures do not "
                     "support Python lists; please use a tuple instead.")
        else:
          message = ("The argument to `Dataset.unordered_merge()` must be a nested "
                     "structure of `Dataset` objects.")
        raise TypeError(message)
    self._datasets = datasets

  def _as_variant_tensor(self):
    # pylint: disable=protected-access
    return gen_dataset_ops.unordered_merge_dataset(
        [ds._as_variant_tensor() for ds in nest.flatten(self._datasets)],
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


